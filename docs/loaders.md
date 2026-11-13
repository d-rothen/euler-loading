# Loaders & writers

A loader is the **only** place where domain-specific I/O happens. euler-loading
resolves *which* file a sample needs and hands it over; the loader decides what
that file means.

- [The contract](#the-contract)
- [Per-file attributes](#per-file-attributes)
- [Automatic loader resolution](#automatic-loader-resolution)
- [Dry-running loaders](#dry-running-loaders)
- [Loader protocols](#loader-protocols)
- [Built-in loaders](#built-in-loaders)

## The contract

```python
loader(path: str | BinaryIO, meta: dict | None = None) -> Any
writer(path: str, value: Any, meta: dict | None = None) -> None
```

`meta` carries the ds-crawler metadata for the modality, or `None` when
unavailable. Use it for format parameters — units, encoding details — rather
than inventing modality-specific argument variants.

For zip-backed modalities `path` is an in-memory `io.BytesIO` buffer with a
`.name` attribute, not a filesystem path.

```python
from PIL import Image
import numpy as np

def load_rgb(path, meta=None):
    return Image.open(path).convert("RGB")

def load_depth(path, meta=None):
    return np.load(path)
```

## Per-file attributes

ds-crawler can persist arbitrary per-file metadata in `file_entry["attributes"]`.
It always surfaces on the sample as `sample["attributes"][modality]` and
`sample["meta"][modality]["attributes"]`.

A loader can additionally *receive* it by accepting an `attributes=` keyword:

```python
def load_depth(path, meta=None, *, attributes=None):
    scale = (attributes or {}).get("scale_to_meters_override", 1.0)
    return np.load(path) * scale
```

euler-loading probes each resolved loader once with `inspect.signature` and
caches whether it accepts `attributes` (or `**kwargs`). Loaders that do not are
called with the plain two-argument form, so **existing loaders keep working
unchanged**. The argument is keyword-only, so it can never collide positionally.

For hierarchical modalities the cache key includes the attributes payload, so
the same file with different per-file attributes is decoded independently.

Take `attributes` when the value must be applied *during* decode — a per-scan
depth scale, per-file clip ranges for normalisation, a lookup key selecting an
external resource. When it is purely descriptive — a training weight, a source
tag — read it off `sample["attributes"]` instead and leave the loader signature
alone.

All built-in loaders accept `attributes=None`. Most ignore it;
`generic_dense_depth.depth` consumes `attributes["scale_to_meters_override"]`.

## Automatic loader resolution

Leave `Modality.loader` as `None` and euler-loading resolves it from the
modality's canonical ds-crawler dataset contract. That contract lives at
`.ds_crawler/dataset-head.json`, or at
`.ds_crawler/<metadata_scope>/dataset-head.json` for a scoped modality.

`dataset-head.json` has an `addons` object whose keys name addon contracts. For
automatic loading it must contain an `euler_loading` entry with both `loader`
and `function`. For example:

```json
{
  "contract": {
    "kind": "dataset_head",
    "version": "1.0"
  },
  "dataset": {
    "id": "vkitti2_rgb",
    "name": "Virtual KITTI 2 RGB"
  },
  "modality": {
    "key": "rgb",
    "meta": {
      "range": [0, 255]
    }
  },
  "addons": {
    "euler_loading": {
      "version": "1.0",
      "loader": "vkitti2",
      "function": "rgb"
    }
  }
}
```

The `euler_loading` addon fields are:

| Field | Required | Meaning |
|---|---|---|
| `version` | yes | Version of the addon contract, currently `"1.0"`. |
| `loader` | for automatic loading | One of the built-in module names below, such as `vkitti2`. |
| `function` | for automatic loading | Callable within that module, such as `rgb`. It must be declared together with `loader`. |
| `writer_function` | no | Explicit writer callable name when the naming conventions below are not sufficient. |

ds-crawler carries the dataset head into its generated index. euler-loading
reads that contract and asks for the named `euler_loading` addon; `loader` and
`function` therefore do not belong at the top level of a newly authored index.
An explicit `Modality.loader` always takes precedence. Automatic resolution
uses the GPU variant; import and pass a CPU loader explicitly when NumPy output
is required.

`Modality.function` keeps the head's `loader` module but picks another callable
in it, and the writer follows it. `Modality(path, function="sky_mask")` reads a
`class_segmentation` index as boolean sky masks, taking the sky definition from
the head's `skyclass` (see [Sky masks](#sky-masks)). It cannot be combined with
`Modality.loader`. Each modality's effective loader is logged when the dataset
is built, e.g. `loaded with real_drive_sim.sky_mask (the index declares
'class_segmentation')`.

For compatibility, euler-loading can still read the legacy top-level
`euler_loading` mapping from older indexes. New datasets should declare the
addon in `dataset-head.json` as shown above.

Writers resolve from the same `addons.euler_loading` entry, in this order:

1. the explicit `writer_function` field,
2. for `function: "read_<suffix>"`, `write_<suffix>`,
3. `write_<function>`.

## Dry-running loaders

Resolution only pays off if the loader a dataset names can really read its
files. Point the bundled command at a path to find out, before a training run
depends on it:

```bash
euler-loading /data/vkitti2
python -m euler_loading /data/vkitti2   # same thing, no console script needed
```

It takes one path — a modality root, a `.zip` archive, or a folder holding
several of them — and for every ds-crawler artifact set below it:

1. loads the index, or a named split,
2. resolves the loader its `addons.euler_loading` entry declares,
3. decodes a sample of the indexed files with that loader,
4. reports the shape, dtype and value range of what came back.

Nothing is written and no dataset is constructed. A modality that cannot be
read is reported, not raised, so one broken archive does not hide the rest.

```text
euler-loading dry-run: /data/vkitti2
loader variant: gpu (torch tensors)

vkitti_2.0.3_rgb  [ok]
  path      /data/vkitti2/vkitti_2.0.3_rgb
  contract  vkitti2_rgb, "Virtual KITTI 2 RGB", modality key: rgb
  loader    vkitti2.rgb -> euler_loading.loaders.gpu.vkitti2.rgb
  writer    write_rgb
  index     21260 files; available splits: train, val
  decoded   1/1
    Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg  ->  Tensor (3, 375, 1242) float32 in [0, 1]  (11.4 ms)

vkitti_2.0.3_textgt  [ok]
  path      /data/vkitti2/vkitti_2.0.3_textgt
  contract  vkitti2_intrinsics, "Virtual KITTI 2 intrinsics", modality key: camera_intrinsics
  loader    vkitti2.read_intrinsics -> euler_loading.loaders.gpu.vkitti2.read_intrinsics (hierarchical)
  writer    write_intrinsics
  index     50 files
  decoded   1/1
    Scene01/clone/intrinsic.txt  ->  Tensor (3, 3) float32 in [0, 725]  (0.6 ms)

2 modalities checked: 2 ok, 0 failed; 2/2 files decoded
```

The exit status is 0 when every modality resolved a loader and decoded every
file it was asked to, and 1 otherwise, so the command works as a CI step or a
job prologue.

| Flag | Effect |
|---|---|
| `-n, --samples N` | Files to decode per modality, spread evenly across the index (default 1). `0` resolves loaders without decoding anything. |
| `--all` | Decode every indexed file. Slow, but checks the whole archive. |
| `--split NAME` | Dry-run a named ds-crawler split instead of the canonical index. |
| `--scope SCOPE` | Read `.ds_crawler/SCOPE/` only, instead of every artifact set a root carries. |
| `--cpu` | Resolve the NumPy loaders instead of the torch ones. Works on an install without PyTorch. |
| `--max-depth N` | Directory levels below the path to search for dataset roots (default 3). |
| `--json` | Print the report as JSON instead of text. |
| `-v, --verbose` | Log debug output, including a traceback for every failed load. |

The path takes the same inline selectors as `Modality`, so
`euler-loading /data/muses.zip:train#scope=rgb` checks one split of one scope.

Discovery stops at the first dataset root down each branch: a directory
carrying its own `.ds_crawler/` is checked, never descended into, so a
dataset's own scene directories are never walked. A root with several
metadata scopes contributes one entry per scope, labelled with the selector
that addresses it.

### Reading the report

| Line | What it is telling you |
|---|---|
| `error loader: ... no 'addons.euler_loading' entry` | the dataset head declares no loader, so only an explicit `Modality(..., loader=...)` can read it |
| `loader vkitti2.rgb (unresolved)` plus an `error` | the contract names a loader that is unknown, misspelled, or needs a dependency this install lacks — `--cpu` resolves the torch-free variant |
| `decoded 0/2` with per-file `FAILED` | the files are indexed and present, but the declared loader cannot decode them |
| `(hierarchical)` after the loader | the declared function reads per-scene data; pass that modality as `hierarchical_modalities`, not `modalities` |
| `error index: ...` | ds-crawler could not produce an index — a missing split, an unreadable archive, or a dataset head the contract rejects |
| `sky skyclass [29, 0, 0]: sky in 8/8 files, ...` | a segmentation head's files, decoded with its loader module's `sky_mask` (at least `SKY_SAMPLES` = 8 files) |
| a warning that `sky_mask` finds no sky, or marks a file as more than 90% sky | the head's `skyclass` probably names the wrong label; an `error sky_mask: ...` means the loader rejects it outright |
| a warning about shared file IDs | the modalities below the path have no file ID in common, so they cannot be intersected into one `MultiModalDataset` |

The same report is available as an API, which is what the command is a thin
shell around:

```python
from euler_loading.dry_run import dry_run

report = dry_run("/data/vkitti2", samples=3)
if not report.ok:
    for modality in report.modalities:
        print(modality.label, modality.errors)

report.to_dict()   # the same structure --json prints
```

## Sky masks

The dataset contract requires every `semantic_segmentation` head to declare
`modality.meta.skyclass`: the sky label as it appears in the segmentation file
viewed as RGB. That is the sky colour of a colour-coded map, and `[v, v, v]`
for a single-channel label `v`:

| Dataset | Encoding | `skyclass` |
|---|---|---|
| Virtual KITTI 2 | RGB class colours | `[90, 200, 255]` |
| Real Drive Sim | class ID in the red channel of an RGBA PNG | `[29, 0, 0]` |
| Synscapes | single-channel Cityscapes label IDs | `[23, 23, 23]` |
| MUSES `labelIds` | single-channel Cityscapes label IDs | `[23, 23, 23]` |
| UrbanSyn | Cityscapes train IDs in an RGB PNG with R = G = B | `[10, 10, 10]` |
| SHIFT | class ID in the red channel of an RGB PNG | `[13, 0, 0]` |
| Cityscapes train IDs, e.g. `generic` `.npy` maps | single-channel | `[10, 10, 10]` |

`skyclass: null`, the contract's default, declares no sky. `[0, 0, 0]` is taken
literally, as label 0 or black; contract releases before `null` used it as the
default, so heads written then can carry it without anyone having declared it.
A head that declares `[0, 0, 0]` is therefore logged as a warning.

Every `sky_mask` loader (and the matching `write_sky_mask`) resolves the sky in
this order, skipping a `null` `skyclass`, and logs the definition it uses once:

1. `skyclass` in the file's ds-crawler attributes,
2. `skyclass` in the dataset head's meta,
3. a deprecated per-loader key — `sky_class_id` (Synscapes, Real Drive Sim,
   MUSES), `sky_color` (VKITTI2), `sky_class` / `sky_mask` (MUSES) or
   `sky_mask` (generic dense depth) — which raises a `DeprecationWarning`,
4. the loader's built-in default; `generic.sky_mask` and
   `generic_dense_depth.sky_mask` have none and require `skyclass`.

A deprecated key that `skyclass` overrides is logged as a warning, saying
whether it named the same sky. So is a resolved sky that differs from the
loader's default. Values a loader cannot match raise instead of silently
finding no sky: a non-grey `skyclass` for single-channel maps (Synscapes,
MUSES label PNGs, `generic`), and anything but `[id, 0, 0]` for Real Drive Sim,
which stores class IDs in the red channel. The [dry run](#dry-running-loaders)
decodes segmentation heads with `sky_mask` and warns about suspicious sky
fractions.

## Loader protocols

`DenseDepthLoader` is a `runtime_checkable` Protocol defining the loader
contract for dense-depth datasets:

| Function | Returns |
|---|---|
| `rgb(path, meta=None)` | `(3, H, W)` float32 in `[0, 1]` |
| `depth(path, meta=None)` | `(1, H, W)` float32 in metres |
| `sky_mask(path, meta=None)` | `(1, H, W)` bool |
| `read_intrinsics(path, meta=None)` | `(3, 3)` float32 camera matrix |

```python
from euler_loading import DenseDepthLoader
from euler_loading.loaders.gpu import vkitti2

assert isinstance(vkitti2, DenseDepthLoader)
```

`DenseDepthWriter` and `DenseDepthCodec` provide the matching writer and
combined reader/writer contracts.

## Built-in loaders

Every module exists in two variants:

```python
from euler_loading.loaders.gpu import vkitti2   # torch.Tensor, CHW
from euler_loading.loaders.cpu import vkitti2   # numpy.ndarray, HWC
from euler_loading.loaders     import vkitti2   # shorthand for the GPU variant
```

All of them accept both `str` paths and `BinaryIO` buffers, so they work with
zip-backed modalities unchanged. Shapes below are written GPU / CPU where the
two differ. Functions marked *hierarchical* are intended for
`hierarchical_modalities`.

[`loaders.json`](../euler_loading/loaders/generate/loaders.json) is the
machine-readable version of everything in this section, generated from the
annotations on the loader functions themselves.

### Virtual KITTI 2 — `vkitti2`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | PNG, normalised to `[0, 1]` |
| `depth` | 1HW / HW | float32 | 16-bit PNG, centimetres → metres |
| `class_segmentation` | CHW / HWC | int64 | RGB-encoded class mask |
| `instance_segmentation` | CHW / HWC | int64 | RGB-encoded instance mask |
| `scene_flow` | CHW / HWC | float32 | Optical/scene flow, normalised to `[0, 1]` |
| `sky_mask` | 1HW / HW | bool | Head `skyclass` colour in the class mask, default `[90, 200, 255]` |
| `read_intrinsics` | 3×3 | float32 | From a `.txt` file · *hierarchical* |
| `read_extrinsics` | N×1 | float32 | From a `.txt` file · *hierarchical* |

Writers exist for every modality above.

### Synscapes — `synscapes`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | `img/rgb/<id>.png`, normalised to `[0, 1]` |
| `depth` | 1HW / HW | float32 | `img/depth/<id>.exr`, `Z` channel, planar depth in metres |
| `class_segmentation` | 1HW / HW | int64 | `img/class/<id>.png`, original Cityscapes label IDs |
| `instance_segmentation` | 1HW / HW | int64 | `img/instance/<id>.png`, decoded as `R + 256 * G + 65536 * B` |
| `sky_mask` | 1HW / HW | bool | Sky label in `img/class/<id>.png`: the grey `skyclass`, ID `23` by default |
| `read_intrinsics` | 3×3 | float32 | `camera.intrinsic` from `meta/<id>.json`: `fx`, `fy`, `u0`, `v0` |
| `read_extrinsics` | 4×4 | float32 | `camera.extrinsic` from `meta/<id>.json`: `x`, `y`, `z`, `pitch`, `roll`, `yaw` |

Writers exist for every modality above. `write_depth` needs a filesystem path
because OpenEXR cannot write to a stream; dataset writers hand it a temporary
file automatically, so zip outputs work unchanged. `write_sky_mask` emits a
class image labelling sky with the label `sky_mask` reads, and
`write_intrinsics` and `write_extrinsics` merge into one `meta/<id>.json`
rather than overwriting each other or the file's `scene` and `instance`
blocks.

That merge reads the file back, so it applies to filesystem destinations and
replaces the file atomically. A zip destination is handed a fresh stream per
modality instead, so writing both camera modalities into the **same** archive
produces two entries with the same name, of which only the last is
retrievable — give each modality its own output writer. Because the merge
keeps fields it does not model, writing a rescaled matrix without passing
`meta['resx']`/`['resy']` leaves the previous resolution in place; supply them
whenever the matrix no longer matches the recorded resolution.

Formats follow the [Synscapes dataset reference](https://synscapes.on.liu.se/features.html)
and [FoggySynscapes EXR reader](https://github.com/MartinHahner/FoggySynscapes/blob/main/source/Depth_processing/exr_to_mat.py).
Depth values are already in metres, so they are returned unchanged, including
non-finite values. Semantic labels retain the original IDs, including void
labels, without remapping to training IDs.

EXR depth loading and writing need the optional OpenEXR dependency:

```bash
pip install "euler-loading[gpu,synscapes]"  # omit gpu for NumPy-only use
```

RGB, segmentation, intrinsics and extrinsics work without OpenEXR; only depth
loading and writing need it. All readers accept paths and binary streams. Intrinsics are stored in per-image files,
so the native `meta` directory is a regular modality:

```python
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.gpu import synscapes  # or loaders.cpu for NumPy

dataset = MultiModalDataset(modalities={
    "rgb": Modality("/data/Synscapes/img/rgb", loader=synscapes.rgb),
    "depth": Modality("/data/Synscapes/img/depth", loader=synscapes.depth),
    "segmentation": Modality("/data/Synscapes/img/class", loader=synscapes.class_segmentation),
    "intrinsics": Modality("/data/Synscapes/meta", loader=synscapes.read_intrinsics),
})
```

#### Extrinsics conventions

Synscapes records the camera mount as six scalars — `x`, `y`, `z` in metres and
`pitch`, `roll`, `yaw` in radians — in the ego-vehicle frame, which the dataset
defines as x forward, y left, z up. `read_extrinsics` composes them into a
rigid 4×4 matrix, by default the camera's pose on the vehicle
(`X_ego = T @ X_camera`). Two keys, taken from per-file `attributes` first and
then from `meta`, select the other useful conventions:

| Key | Values | Meaning |
|---|---|---|
| `transform_direction` | `camera_to_ego` (default), `ego_to_camera` | Direction of the returned transform |
| `camera_axes` | `vehicle` (default), `optical` | Camera axes: vehicle-aligned, or x right / y down / z forward |

Combine `ego_to_camera` with `optical` to project ego-frame points — such as
the 3D bounding boxes in the instance metadata — through `read_intrinsics`:

```python
T = synscapes.read_extrinsics(
    "/data/Synscapes/meta/1.json",
    {"transform_direction": "ego_to_camera", "camera_axes": "optical"},
)
```

The dataset documents the six fields and the ego frame but not the order in
which the angles compose, so the loaders apply the usual automotive
`Rz(yaw) @ Ry(pitch) @ Rx(roll)` and record that in `loaders.json` under the
modality's `rotation_order`. The values are constant across the released
dataset. `write_extrinsics` reads the same two keys, so a matrix loaded with
one set of options writes back unchanged with the same options.

Automatic resolution uses `loader="synscapes"` in the dataset contract.
Intrinsics describe the metadata's image resolution (native 1440×720).
When using `img/rgb-2k` at 2048×1024, scale the camera matrix with
[`resize_intrinsics`](preprocessing.md) and align the other modalities to the
same resolution. Loaders preserve the stored resolution.

### MUSES — `muses`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Frame-camera PNG, `[0, 1]` |
| `reference_rgb` | CHW / HWC | float32 | Clear-weather reference frame, `[0, 1]` |
| `semantic_segmentation` | 1HW / HW | int64 | Cityscapes `labelIds` or `labelTrainIds` |
| `semantic_segmentation_color` | CHW / HWC | uint8 | Cityscapes `labelColor` PNG |
| `sky_mask` | 1HW / HW | bool | `skyclass` (grey for label PNGs, a colour for RGB class images); default `trainId=10` / `labelId=23` |
| `panoptic_segmentation` | 1HW / HW | int64 | COCO-style RGB panoptic PNG → segment IDs |
| `lidar_point_cloud` | (N, 6) | float64 | `.bin`: `x, y, z, intensity, ring, timestamp` |
| `sparse_depth` | (N, 6) | float64 | Same points, for sparse-depth workflows |
| `read_intrinsics` | 3×3 | float32 | From `calib.json`, `sensor="rgb"` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | From `calib.json`, `transform="lidar2rgb"` · *hierarchical* |
| `read_lidar_to_gnss` | 4×4 | float32 | `extrinsics.lidar2gnss` from `calib.json` · *hierarchical* |
| `gnss` | dict | — | Per-frame `gnss/…_gnss.txt` u-blox record, in physical units |
| `sensor_timing` | dict | — | The frame's sensor timestamps from `meta.json`, in UTC µs · *hierarchical, selected per sample* |

`point_cloud` is an alias for `lidar_point_cloud`. Writers exist for every
image, lidar and intrinsics function, and what they write reads back unchanged:

- `write_rgb` / `write_reference_rgb`: 8-bit RGB PNG;
- `write_semantic_segmentation`: single-channel 8-bit label PNG;
- `write_semantic_segmentation_color`: RGB PNG;
- `write_panoptic_segmentation`: COCO RGB-encoded IDs, which must fit 24 bits;
- `write_sky_mask`: only sky / not sky. A grey `skyclass` gives a
  single-channel PNG with the sky label, and `non_sky_class_id` (default 0)
  elsewhere; a colour gives an RGB PNG with that colour on black;
- `write_sparse_depth` (and `write_lidar_point_cloud` / `write_point_cloud`):
  `(N, 6)` float64 `.bin`;
- `write_intrinsics`: updates `intrinsics[<sensor>].K` (default `rgb`) in an
  existing `calib.json`. The change from the stored `K` must be a crop or
  rescale, and it is applied to the stored float64 values, whole-pixel crops
  as exact integer shifts. `D`, the other sensors and all extrinsics are kept:
  `D` acts on normalised coordinates, which a crop or rescale leaves alone.
  An unchanged matrix leaves the file byte-identical, e.g. after cutting rows
  off the bottom. A new file gets `{"intrinsics": {<sensor>: {"K": ...}}}`.

`calib.json` holds `K` and every extrinsic, so a derived dataset that changes
`K` should write the same updated file for `read_extrinsics` and
`read_lidar_to_gnss`, as `euler-preprocess crop` does. The released
frame-camera PNGs are already undistorted: project with plain `K`.

#### Ego-motion inputs

The MUSES lidar sweeps its field of view top to bottom in about 85 ms, while
the car moves. Its points therefore need ego-motion compensation before they
line up with the frame camera. The official SDK moves each point to the
camera's mid-exposure using the frame's GNSS record. These three loaders
supply that SDK's inputs per sample; the compensation itself is left to
consumers.

- `gnss`: the frame's u-blox GNSS/IMU record, a regular modality joined by
  sample ID. The record carries `ground_speed_m_s`, `roll_rate_deg_s`,
  `pitch_rate_deg_s` and `heading_rate_deg_s`; the heading rate is positive
  clockwise seen from above. It also has position, attitude, NED velocity and
  accuracies. Every key names its unit.
- `sensor_timing`: `meta.json`, a root-level hierarchical file decoded once.
  Each sample then receives its own `{"frame", "camera_exposure_start_us",
  "camera_exposure_end_us", "lidar_start_us", "lidar_end_us",
  "radar_start_us", "radar_end_us"}` record. The record is found through the
  file names the `meta.json` entries list under `path_to_*`, so it does not
  depend on how IDs are built, but the sample's files must keep their native
  MUSES names.
- `read_lidar_to_gnss`: `extrinsics.lidar2gnss` from `calib.json`. It maps
  lidar points into the GNSS record's frame.

The per-point lidar timestamp (column 6, seconds) together with these inputs
is all the compensation needs.

#### Preparing the dataset

[euler-known-datasets](https://github.com/d-rothen/euler-known-datasets)
indexes MUSES from its packages (`frame_camera_trainvaltest.zip`,
`lidar_trainvaltest.zip`, `gnss_trainvaltest.zip`, …) or the extracted
`muses/` tree. It writes every scope these loaders read, including the three
ego-motion inputs:

```bash
euler-known-datasets prepare muses /data/downloads/muses /data/muses
```

Samples are `/<split>/<weather>/<time_of_day>/<frame>`, e.g.
`/test/fog/day/670832`, with `train`, `val` and `test` splits. Its
[MUSES page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/muses.md)
documents the layout, the IDs and the scopes.

### Real Drive Sim — `real_drive_sim`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | PNG, `[0, 1]` |
| `depth` | 1HW / HW | float32 | `.npz`, metres |
| `class_segmentation` | 1HW / HW | int64 | Class IDs from the red channel of an RGBA PNG |
| `sky_mask` | 1HW / HW | bool | Head `skyclass` colour, default `[29, 0, 0]` (class ID 29) |
| `calibration` | dict | — | `{sensor: {"K": (3,3), "T": (4,4), "distortion": (8,)}}` · *hierarchical* |
| `all_intrinsics` | dict | — | `{sensor: (3, 3)}` · *hierarchical* |
| `read_intrinsics` | 3×3 | float32 | Defaults to `CS_FRONT` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | `HDL_64E` → `CS_FRONT` · *hierarchical* |

### Princeton DENSE / SeeingThroughFog — `princeton_dense`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Plain 8-bit PNG, `[0, 1]` |
| `rccb` | CHW / HWC | float32 | 12-bit Bayer (`GRBG`) TIFF debayered to RGB, `[0, 1]`; raw and unrectified |
| `sparse_depth` | (N, 5) | float32 | `.bin`: `x, y, z, intensity, ring` |
| `read_intrinsics` | 3×3 | float32 | Rectified pinhole `P[:, :3]` from `calib_cam_stereo_left.json`; raw `K` with `intrinsics_matrix: "K"` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | `lidar_hdl64_s3_roof` → `cam_stereo_left_optical`, from `calib_tf_tree_full.json` · *hierarchical* |

`lidar_point_cloud` and `point_cloud` are aliases for `sparse_depth`. Writers
exist for `rgb`, `read_intrinsics` and `sparse_depth` (aliases
`write_lidar_point_cloud` / `write_point_cloud`); `write_sparse_depth` stores
`(N, 5)` float32 `.bin` files that read back unchanged, e.g. after keeping only
the points inside a crop.

The 8-bit `cam_stereo_left_lut` images that `rgb` reads are rectified: the
official SeeingThroughFog conversion remaps the raw frames with
`cv2.initUndistortRectifyMap(K, D, R, P)`, and the official lidar projection
uses `P`. `read_intrinsics` therefore returns `P[:, :3]`, an ideal pinhole
matrix with no distortion; `P[:, 3]` (zero for the left camera, the baseline
term for the right one) is not part of it. Set `intrinsics_matrix` to `"K"` in
the per-file attributes or the head's meta for the raw camera matrix of the
12-bit `cam_stereo_left` TIFFs (`rccb`) instead. It describes them only
together with the distortion `D`, which is large (`k1 ≈ -0.57`).

`write_intrinsics` writes that matrix back into the camera-info JSON. An
existing file is updated rather than replaced. The change from its stored
matrix must be a crop or a rescale, and it is applied to `K` and `P` alike, so
the two stay consistent; a whole-pixel crop shifts the stored values by exact
integers. `D`, `R`, `roi` and the header are kept, and `meta["width"]` and
`meta["height"]` set the recorded image size. A new file or a stream target
gets an ideal pinhole camera: `K = P[:, :3]`, zero distortion, identity `R`.
The extrinsics need no counterpart: a crop or rescale only changes the pixel
grid, so lidar projected through the written matrix lands on the cropped
pixels unchanged.

### KITTI-CARLA — `kitti_carla`

[KITTI-CARLA](https://arxiv.org/abs/2109.00892) is rendered with CARLA 0.9.10
from a vehicle with KITTI's sensor layout: an HDL-64E-like lidar and two
1392×1024 cameras with a 72° horizontal field of view, over 5000 frames in
each of seven towns. Each `TownNN.zip` holds `TownNN/generated/` with
`images_rgb/<frame>_<camera>.png`, `images_ss/<frame>_1<camera>.png`,
`images_depth/<frame>_2<camera>.png`, `frames/frame_<frame>.ply`,
`lidar_to_cam0.txt`, `lidar_to_cam1.txt` and `full_poses_lidar.txt`.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | `images_rgb` RGBA PNG, alpha dropped, `[0, 1]` |
| `depth` | 1HW / HW | float32 | `images_depth`, CARLA `(R + 256 G + 65536 B) / (2**24 - 1) * 1000`, planar depth in metres; sky is the 1000 m far plane |
| `class_segmentation` | 1HW / HW | int64 | `images_ss` colours decoded to the 23 CARLA tags in `CLASS_NAMES` (`0` unlabeled … `13` sky … `22` terrain) |
| `sky_mask` | 1HW / HW | bool | `images_ss` pixels in the head's `skyclass` colour, default `[70, 130, 180]` |
| `lidar_point_cloud` | (N, 7) | float64 | `frames/*.ply`: `x, y, z, cos_angle_lidar_surface, timestamp, instance, semantic` |
| `read_intrinsics` | 3×3 | float32 | The paper's camera model, `fx = fy ≈ 957.96`, principal point at the centre · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | `lidar_to_cam<N>.txt`, lidar → camera in optical axes by default · *hierarchical* |
| `lidar_pose` | 4×4 | float64 | `full_poses_lidar.txt`, lidar → world at the sample's frame · *hierarchical, selected per sample* |

`point_cloud` and `sparse_depth` are aliases for `lidar_point_cloud`. Writers
exist for every function except `lidar_pose`, each in the dataset's own format:
`write_depth` encodes CARLA's 24 bits (NaN becomes 0, values clip to
`[0, 1000]` m), `write_class_segmentation` paints the CARLA palette, and
`write_lidar_point_cloud` emits the same binary PLY layout. `CLASS_NAMES` and
`PALETTE` are exported for decoding the tags.

`class_segmentation` returns the same tag IDs as the lidar's `semantic` column,
and raises on any colour outside the palette rather than labelling it silently;
none occurred in images sampled from all seven towns.

#### Camera model and frames

The dataset ships no intrinsics file. Both cameras are ideal CARLA pinholes,
so `read_intrinsics` computes `fx = fy = W / (2 tan(fov / 2))` with the
principal point at `(W / 2, H / 2)` from the paper's 1392×1024 and 72°.
Index a `lidar_to_cam<N>.txt` to place it in the hierarchy; its contents are
checked but carry no intrinsics. `width`, `height` and `horizontal_fov_deg`
in the file's attributes or the head's meta override the defaults, and a file
of nine numbers, as `write_intrinsics` writes after a crop or resize, is read
as the matrix itself.

Lidar points, `lidar_to_cam<N>.txt` and the poses use right-handed vehicle
axes: x forward, y left, z up. A calibration file maps lidar points into a
camera, `X_cam = T @ X_lidar`, with the camera axes still vehicle-aligned (its
rotation is the identity). `read_extrinsics` therefore returns the transform
into optical camera axes (x right, y down, z forward), so lidar projects
directly:

```python
import numpy as np
from euler_loading.loaders.cpu import kitti_carla

K = kitti_carla.read_intrinsics("Town01/generated/lidar_to_cam0.txt")
T = kitti_carla.read_extrinsics("Town01/generated/lidar_to_cam0.txt")
points = kitti_carla.lidar_point_cloud("Town01/generated/frames/frame_0000.ply")

camera = (T @ np.c_[points[:, :3], np.ones(len(points))].T)[:3]
uvw = K @ camera[:, camera[2] > 0]
pixels, depth = uvw[:2] / uvw[2], uvw[2]  # depth matches images_depth, planar
```

Two keys, taken from the file's attributes and then the head's meta, select
other conventions; `write_extrinsics` reads the same keys from `meta` and
always writes the native file:

| Key | Values | Meaning |
|---|---|---|
| `camera_axes` | `optical` (default), `vehicle` | Camera axes; `vehicle` returns the stored matrix |
| `transform_direction` | `lidar_to_camera` (default), `camera_to_lidar` | Direction; `camera_to_lidar` is the camera's pose on the lidar |

Projected this way, a Town01 sweep matches `images_depth` to about 1 cm
(median) in both cameras, and its `semantic` tags match `class_segmentation`
at 99% of the projected pixels.

#### Lidar sweeps and poses

A sweep `i` covers `[0.1 i, 0.1 i + 0.1)` s, and camera frame `i` is taken at
its start. Each point is in the lidar frame at the moment it was measured,
without ego-motion compensation; its `timestamp` column says when.
`lidar_pose` reads the 1 kHz `full_poses_lidar.txt` once per worker (a few
seconds for a town's 500,000 poses) and gives each sample the lidar-to-world
pose at its camera frame, matched through the frame number in its file names
(`frame_<frame>.ply`, `<frame>_<camera>.png`). The world frame is the lidar
frame at the first step. Relative motion between two samples is
`inv(pose_a) @ pose_b`; a camera's pose is `pose @ inv(T_vehicle)` with
`T_vehicle` read with `camera_axes: "vehicle"`.

Calling `lidar_pose` directly returns the whole `LidarPoseTable`. Its
`at_times` maps a sweep's `timestamp` column to per-point poses, which
compensates the sweep's motion. In a 40° turn that cut the misalignment
between two sweeps from about 18 cm to 7 cm (median), compared with one pose
per sweep:

```python
table = kitti_carla.lidar_pose("Town01/generated/full_poses_lidar.txt")
points_h = np.c_[points[:, :3], np.ones(len(points))]
world = np.einsum("nij,nj->ni", table.at_times(points[:, 4]), points_h)[:, :3]
```

#### Preparing the dataset

[euler-known-datasets](https://github.com/d-rothen/euler-known-datasets)
prepares KITTI-CARLA from its town archives (`TownNN.zip`, or extracted
towns):

```bash
euler-known-datasets prepare kitti_carla /data/KITTI-CARLA/dataset /data/kitti_carla
```

It writes the scopes `rgb`, `depth`, `semantic_segmentation`, `lidar`,
`intrinsics`, `camera_extrinsics` and `lidar_pose`. Samples are
`/town:TownNN/camera:N/frame-NNNN`. Each lidar sweep is indexed under every
camera, so images, depth, segmentation and lidar join by ID and each sample
takes its camera's calibration from the hierarchy. Its
[KITTI-CARLA page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/kitti_carla.md)
documents the regexes, the options (towns, cameras, frame step) and the
dataset facts they rest on.

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/kitti_carla"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb"),
        "depth": Modality(root, metadata_scope="depth"),
        "semantic": Modality(root, metadata_scope="semantic_segmentation"),
        "lidar": Modality(root, metadata_scope="lidar"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
        "extrinsics": Modality(root, metadata_scope="camera_extrinsics", collapse_single=True),
        "pose": Modality(root, metadata_scope="lidar_pose", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/town:Town01/camera:0/frame-0000"
```

### DrivingStereo — `driving_stereo`

[DrivingStereo](https://drivingstereo-dataset.github.io/) is a rectified
stereo driving dataset with lidar-derived depth and disparity. Its images are
released at half resolution, about 881×400; the foggy weather subset is
879×400. The test set and the four weather subsets (500 frames each) also ship
full-size maps at exactly twice that size. A weather archive such as
`left-image-half-size.zip` holds
`<weather>/left-image-half-size/<sequence>_<timestamp>.jpg`, and the
`depth-map-*`, `disparity-map-*` and `right-image-*` archives use the same
file stems. Calibration is one KITTI-style text file per sequence,
`calibration/<train|test>/<half|full>-image-calib/<sequence>.txt`. Camera
`101` is the left camera and `103` the right one.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Left or right JPEG, `[0, 1]` |
| `depth` | 1HW / HW | float32 | `depth-map-half-size`, uint16 / 256, planar metres; `0` invalid |
| `depth_full_size` | 1HW / HW | float32 | `depth-map-full-size` (uint16 / 256), resampled onto the half-size left image |
| `disparity` | 1HW / HW | float32 | `disparity-map-half-size`, uint16 / 256, half-size pixels; `0` invalid |
| `disparity_full_size` | 1HW / HW | float32 | `disparity-map-full-size` (uint16 / 128, full-size pixels), resampled and halved into half-size pixels |
| `read_intrinsics` | 3×3 | float32 | Rectified `P_rect_101[:, :3]`; `camera: "right"` reads `P_rect_103` · *hierarchical* |

Writers exist for every function, and what they write reads back unchanged.
`write_depth` and `write_disparity` write the half-size uint16 encodings.
`write_depth_full_size` and `write_disparity_full_size` write a native
full-size map with each value at pixel `(2y, 2x)`. `write_rgb` writes JPEG
(quality 95) for `.jpg` names and PNG otherwise. `write_intrinsics` updates
an existing calibration file. The change must be a crop or rescale. It is
applied to every `P_rect_*` row, so both cameras and the baseline term stay
consistent, and `meta["width"]`/`["height"]` update `S_rect_*`. All other
lines are kept, and an unchanged matrix leaves the file byte-identical.

#### Alignment with the half-size left image

Every function returns data on the pixel grid of `left-image-half-size`:

- Depth and disparity belong to the **rectified left camera**. The half-size
  maps have the left image's size and grid, so `depth` and `disparity`
  return them unchanged.
- The full-size calibration is the half-size one doubled exactly. On the
  foggy sequence `P_rect_101` has `fx = 2002.991`, `cx = 905.3329` against
  `1001.496`, `452.6664`, with no half-pixel offset. Half-size pixel `(y, x)`
  therefore sits on full-size pixel `(2y, 2x)`, which is also where the two
  released map sizes agree. `depth_full_size` and `disparity_full_size`
  resample onto that grid. `full_size_resampling`, read from the file's
  attributes and then the head's meta, selects how:

  | Value | Each half-size pixel takes |
  |---|---|
  | `nearest` (default) | the full-size sample at `(2y, 2x)`, else the nearest valid one within half a half-size pixel: the four direct neighbours, then the four diagonal ones. Within a ring the nearer surface wins (smaller depth, larger disparity). |
  | `subsample` | only the sample at `(2y, 2x)`; exact, but it keeps about a quarter of the measurements |

- The half-size maps are **not** a downsampling of the full-size ones. On
  the foggy subset about half of their valid pixels have no full-size sample
  in the matching 2×2 block. About 21% of half-size pixels are valid, against
  5% of full-size pixels and 26% after `nearest` resampling. Where both are
  valid, 97% of the resampled full-size depths are within 5% of the
  half-size ones. Prefer `depth` and `disparity` where both sizes exist.
- `depth × disparity = fx × baseline` holds per pixel with the half-size `fx`
  and the baseline `|T_103|` (0.545 m), to within 0.05% on the foggy subset.
  The depth is therefore planar, not radial.

Index the **`half-image-calib`** file for intrinsics. The `full-image-calib`
file describes the same camera at twice the resolution. In the half-size
files, `P_rect_103[0, 3]` was not halved and still holds the full-size
`-fx × baseline`, so take the baseline from `T_103`. `K_*`, `D_*` and `S_*`
describe the raw, unrectified sensor images, which are not released.

#### Preparing the dataset

[euler-known-datasets](https://github.com/d-rothen/euler-known-datasets)
prepares DrivingStereo from its downloads: the train, test and weather
subsets, and the calibration files.

```bash
euler-known-datasets prepare driving_stereo /data/downloads/DrivingStereo /data/driving_stereo
```

It writes one scope per modality (`rgb`, `rgb_right`, `depth`, `disparity`,
the `*_full_size` variants and `intrinsics`), with the heads this module
expects. Samples are `/subset:<subset>/sequence:<sequence>/frame-<timestamp>`,
and every subset (`train`, `test`, `foggy`, …) is a split. Its
[DrivingStereo page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/driving_stereo.md)
documents the layout, the regexes and head meta, and how calibration is
matched to weather sequences.

```python
from euler_loading import Modality, MultiModalDataset

root = "/data/driving_stereo"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="foggy"),
        "depth": Modality(root, metadata_scope="depth", split="foggy"),
        "disparity": Modality(root, metadata_scope="disparity", split="foggy"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
    },
)
sample = dataset[0]  # full_id "/subset:foggy/sequence:2018-10-25-07-37-26/frame-2018-10-25-07-41-58-446"
sample["rgb"].shape, sample["depth"].shape  # (3, 400, 879), (1, 400, 879)
```

### nuScenes — `nuscenes`

[nuScenes](https://www.nuscenes.org/nuscenes) keeps its calibration and
vehicle poses in relational JSON tables. Its lidar sweeps never share a
timestamp with the camera frames they belong to. These loaders therefore read
nuScenes as prepared by
[euler-known-datasets](https://github.com/d-rothen/euler-known-datasets),
which turns the downloads into ds-crawler scopes:

```bash
euler-known-datasets prepare nuscenes /data/downloads/nuscenes /data/nuscenes --cameras CAM_FRONT_LEFT
```

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Camera JPEG, `[0, 1]` |
| `lidar_point_cloud` | (N, 5) | float32 | `.pcd.bin`: `x, y, z` (metres, LIDAR_TOP frame: x right, y forward, z up), `intensity`, `ring` |
| `read_intrinsics` | 3×3 | float32 | `camera.intrinsics` from `calibration/<scene>/<camera>.json`; pinhole, no distortion · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | The frame's lidar→camera transform (optical axes), through both vehicle poses · *hierarchical, selected per sample* |

`point_cloud` and `sparse_depth` are aliases for `lidar_point_cloud`. The
prepared `lidar` scope holds, for every camera frame, the sweep synchronised
with it, under the frame's sample ID
`/scene:<scene>/camera:<camera>/<camera timestamp>`. Joining `rgb` and `lidar`
therefore pairs each image with its sweep. `read_extrinsics` selects the
frame's record from the scene's calibration file by that ID.

```python
import torch
from euler_loading import Modality, MultiModalDataset

root = "/data/nuscenes"
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(root, metadata_scope="rgb", split="test"),
        "lidar": Modality(root, metadata_scope="lidar", split="test"),
    },
    hierarchical_modalities={
        "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
        "extrinsics": Modality(root, metadata_scope="camera_extrinsics", collapse_single=True),
    },
)
sample = dataset[0]
points = sample["lidar"][:, :3]
camera = (sample["extrinsics"] @ torch.cat([points, torch.ones(len(points), 1)], 1).T)[:3]
uvw = sample["intrinsics"] @ camera[:, camera[2] > 1]   # pixels: uvw[:2] / uvw[2]
```

The transform is per frame, not per scene: every image and every sweep has
its own vehicle pose. Ignoring that is off by about 0.4 m at 10 m/s.
`transform_direction: camera_to_lidar` in the file's attributes or the head's
meta returns the inverse. Calling `read_extrinsics` directly returns the
scene's whole `CameraFrames` table, with `lidar_to_camera(id)` and
`camera_to_global(id)` per frame. A frame without a synchronised sweep raises
`KeyError`. The prepared `lidar` scope leaves those frames out, so they drop
from a joined dataset.

Writers exist for `rgb` (JPEG for `.jpg` names, PNG otherwise), the point
clouds (the native `.pcd.bin` layout) and `read_intrinsics`.
`write_intrinsics` updates `camera.intrinsics` in an existing calibration
file. The change must be a crop or rescale, and `meta["width"]`/`["height"]`
update the recorded size. The frame records stay as they are: a crop or
rescale only moves the pixel grid, so lidar projected with the written matrix
lands on the cropped pixels. The calibration file format, the pairing of
frames with sweeps, and how both were established are documented in
euler-known-datasets'
[nuScenes page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/nuscenes.md).

### ScatterNeRF — `scatternerf`

ScatterNeRF (Ramazzina et al., ICCV 2023) releases
foggy driving sequences in the NSVF layout of NeRF pipelines:
`<sequence>/{rgb,intrinsics,pose}/<camera>_<frame>_<timestamp ns>.{png,txt}`,
one file of each per image. The release has no lidar.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Rectified SeeingThroughFog stereo frame, 1704×908 PNG, `[0, 1]` |
| `read_intrinsics` | 3×3 | float32 | Upper-left block of the stored 4×4; ideal pinhole, principal point at the image centre |
| `camera_pose` | 4×4 | float64 | Camera-to-world, optical camera axes, world frame per sequence; stored units |
| `camera_pose_metric` | 4×4 | float64 | The same pose with translations in meters (needs `scale_to_meters`) |

Writers exist for `rgb`, `read_intrinsics` and `camera_pose`, in the native
formats.

The poses are scaled per sequence, as NeRF data usually is. The rig's 0.203 m
stereo baseline measures between 0.021 and 0.038 units depending on the
sequence. euler-known-datasets records each sequence's `scale_to_meters` in
the pose files' attributes, and `camera_pose_metric` applies it:
`Modality(root, metadata_scope="camera_pose", function="camera_pose_metric")`.

The images, intrinsics and poses are each a regular modality joined by sample
ID, `/sequence:<n>/camera:<left|right>/frame-<frame>`. Prepare the dataset
with `euler-known-datasets prepare scatternerf`. Its
[ScatterNeRF page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/scatternerf.md)
covers the release's packaging and how the metric scale is derived.

### CEREMA Pixel-Accurate Depth Benchmark — `cerema`

The Pixel-Accurate Depth Benchmark (Gruber et al., 3DV 2019) recorded four
static scenes in the CEREMA weather chamber. Each scene is shown in clear
weather, in fog at 17 visibilities from 20 to 100 m, and in rain at 15 and
55 mm/h, by day and by night, with 10 frames each. The release ships one
folder per sensor or method. Depth is z-depth in meters on the grid of its
camera, `0` where invalid.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Rectified stereo frame (`rgb_left_8bit`, `rgb_right_8bit`), 1920×1024, `[0, 1]` |
| `gated` | 1HW / HW | float32 | Gated slice (`gated{0,17,31}_10bit`), 1280×720: `(raw − 87) / 1023`, clipped to `[0, 1]` as the benchmark does |
| `depth` | 1HW / HW | float32 | npz depth map: dense ground truth (`intermetric_*`, one per scene), projected HDL64 lidar (`lidar_hdl64_*`, sparse), `sparse2dense` output |
| `sgm_depth` | 1HW / HW | float32 | SGM's half-resolution disparity of rows 70–893, as metric depth on the 1920×1024 grid |
| `psmnet_depth` | 1HW / HW | float32 | PSMNet's 960-wide disparity, as metric depth on the full grid |
| `monodepth_depth` | 1HW / HW | float32 | Monodepth's width-relative disparity, as metric depth on the full grid |
| `read_intrinsics` | 3×3 | float32 | Rectified `P[:, :3]` of `calibration/calib_cam_stereo_left.json` or `calib_gated_bwv.json` · *hierarchical* |

The `*_depth` functions convert disparity as the benchmark's evaluation code
does: resized bilinearly to the full grid, then `depth = 0.202993 m ×
2355.722801 px / disparity`. Unlike that code, they leave missing pixels at `0`
rather than filling them from their nearest neighbour.

Writers exist for `rgb`, `gated`, `depth` and `read_intrinsics`, in the
native formats. The method conversions have none.

The release has no lidar point clouds: the HDL64 scans come projected into the
left camera and the gated camera. Prepare the dataset with
`euler-known-datasets prepare cerema`. Its
[CEREMA page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/cerema.md)
covers the packaging, the scopes, how ground truth applies to every frame of
its scene, and the benchmark's evaluation protocol.

### ACDC — `acdc`

ACDC, the Adverse Conditions Dataset with Correspondences (Sakaridis et al.,
ICCV 2021), holds 4006 images of drives in fog, at night, in rain and in
snow. Each image has a reference image of the same place in normal conditions,
named with the same sequence and frame.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Adverse-condition image (`rgb_anon/<condition>/<split>/…_rgb_anon.png`), `[0, 1]` |
| `reference_rgb` | CHW / HWC | float32 | Its normal-condition reference (`<split>_ref/…_rgb_ref_anon.png`), `[0, 1]` |

Writers exist for both (8-bit PNG). The images are the anonymised ones ACDC
distributes by default. A reference image comes from another drive, so it is
not pixel-aligned with its adverse image. Prepare the dataset with
`euler-known-datasets prepare acdc`. Its
[ACDC page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/acdc.md)
covers the downloads, IDs and splits.

### Hypersim — `hypersim`

Hypersim (Roberts et al., ICCV 2021) holds V-Ray renders of indoor scenes,
1024×768, as HDF5 files. Reading and writing them needs `h5py`
(`pip install 'euler-loading[hypersim]'`).

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | `color.hdf5` (linear HDR) tone-mapped with Hypersim's operator, `[0, 1]`; the scale comes from `attributes["tonemap_scale"]` |
| `depth` | 1HW / HW | float32 | Planar depth in meters: the released `depth_meters.hdf5` distances divided by `‖K⁻¹[x, y, 1]‖`, with `K` from `attributes["intrinsics"]`; `0` = no geometry |
| `radial_distance` | 1HW / HW | float32 | The released Euclidean distances to the camera centre in meters (`radial_depth: true`) |
| `read_intrinsics` | 3×3 | float32 | Exact pinhole matrix of the scene from `calibration/<scene>/<camera>.json`, skew included · *hierarchical* |
| `camera_pose` | 4×4 | float32 | Camera-to-world pose of the sample's frame in meters, optical axes · *hierarchical*, selected per sample |

The tone-mapping scale makes the 90th-percentile brightness of the pixels with
geometry equal 0.8 after gamma 1/2.2, as Hypersim's
`scene_generate_images_tonemap.py` does. `tonemap()` is exported for other
scales.

Many scenes use tilt-shift lenses: their stored camera axes are up to 41.5°
from the image's, and their principal points lie far from the centre. The
intrinsics, poses and planar depth all refer to the image's own optical frame,
so `K @ X` projects a point `X` of that frame onto its pixel. Writers exist for
every function:
- `rgb`, `depth` and `radial_distance` write HDF5 files marked with an
  `euler_encoding` attribute, so they read back unchanged rather than being
  tone-mapped or converted again;
- `write_intrinsics` updates a calibration file after a crop or rescale.

Prepare the dataset with `euler-known-datasets prepare hypersim`. Its
[Hypersim page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/hypersim.md)
covers the partial download, the camera model and the splits.

### UrbanSyn — `urbansyn`

UrbanSyn (Gómez et al., Neurocomputing 2025) holds 7539 path-traced frames
of three synthetic districts of Barcelona and Berlin, 2048×1024, released as
one flat folder per modality: `rgb/rgb_<NNNN>.png`, `depth/depth_<NNNN>.exr`,
`ss/ss_<NNNN>.png`, plus `camera_metadata.json`. Depth and poses need OpenEXR
(`pip install 'euler-loading[urbansyn]'`).

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | 8-bit sRGB PNG, `[0, 1]` |
| `depth` | 1HW / HW | float32 | Planar depth in meters: OctaneRender's Z-depth pass (EXR channel `Y`, stored in units of 1e5 m) × 1e5; `0` where a ray hit nothing (stored `1.0`) |
| `semantic_segmentation` | 1HW / HW | int64 | Cityscapes train IDs `0`–`18`, `19` for everything else; stored as RGB with R = G = B |
| `sky_mask` | 1HW / HW | bool | Where the labels hold `skyclass`, train ID `10` by default |
| `read_intrinsics` | 3×3 | float32 | From the focal length and sensor size in `camera_metadata.json`: `fx = fy = 1730.21`, principal point at the image centre `(1023.5, 511.5)` · *hierarchical* |
| `camera_pose` | 4×4 | float64 | Camera-to-world pose in meters, optical axes, from the camera `pos`, `target` and `up` in a depth EXR's header (attribute `octane`); one world frame per district |

Writers exist for every function: `rgb`, `depth` and `semantic_segmentation`
in the release's formats (`write_depth` needs a path, since OpenEXR cannot
write to a stream); `write_sky_mask` as a label map of sky and `19`;
`write_intrinsics` and `write_camera_pose` as small JSON files that the
readers return unchanged, since a cropped or resized grid no longer matches
the physical camera description. `read_intrinsics` takes `image_size:
[height, width]` from the attributes or meta for another grid.

The sky is a finite backdrop a few hundred meters away in the depth maps:
mask it with `sky_mask` before using depth there. The depth convention, its
units, the intrinsics and the poses were checked together by reprojecting
depth between frames that see the same street. Prepare the dataset with
`euler-known-datasets prepare urbansyn`. Its
[UrbanSyn page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/urbansyn.md)
covers the downloads, the scenes, the incomplete depth maps of frames
6973–7029 and how the conventions were checked.

### SHIFT — `shift`

SHIFT (Sun et al., CVPR 2022) renders drives in CARLA under fixed weather and
light. euler-known-datasets prepares the front camera of its discrete set,
1280×800, in shift-dev's image-rate layout
(`discrete/images/<split>/front/{img,depth,semseg}/<seq>/<frame>_<group>_front.<ext>`)
with one calibration file per sequence (`calibration/<split>/<seq>.json`).
The ego vehicle's hood covers an arc over the bottom rows, from row 770 at
the image centre, 1.0–1.2 m from the camera.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | PNG (frames decoded from SHIFT's videos) or the released JPEG, `[0, 1]` |
| `depth` | 1HW / HW | float32 | Planar depth in meters from CARLA's 24-bit code, `(R + 256 G + 65536 B) / (2**24 - 1) × 1000`; the sky (and a few distant pixels beyond the far plane) reads 999.9997 m; no invalid pixels |
| `radial_distance` | 1HW / HW | float32 | The depth times `‖K⁻¹[x, y, 1]‖`: Euclidean distance to the camera centre. `K` from `intrinsics` in the attributes or meta, else the release's camera for 1280×800 |
| `semantic_segmentation` | 1HW / HW | int64 | Class ID from the red channel (green = blue = 0): CARLA's 23 tags as released, or the merged set of frames decoded from the videos (head `classes`) |
| `sky_mask` | 1HW / HW | bool | Where the labels hold `skyclass`, `[13, 0, 0]` by default; exact in both label sets |
| `read_intrinsics` | 3×3 | float32 | `camera.intrinsics` of the calibration file: `fx = fy = 640`, centre `(640, 400)` as released · *hierarchical* |
| `camera_pose` | 4×4 | float32 | Camera-to-world pose of the sample's frame in meters, optical axes, in CARLA's world with y negated (right-handed, z up) · *hierarchical*, selected per sample |

Writers exist for every function except `camera_pose`, whose frames share
one file: `rgb` as PNG (JPEG for a `.jpg` target), `depth` and
`radial_distance` in the 24-bit code, labels and sky masks as red-channel
PNGs, and `write_intrinsics` updates a calibration file after a crop or
rescale. The video-decoded labels merge nine pairs of classes that SHIFT's
segmentation videos cannot tell apart; the sky and four other classes stay
exact. Prepare the dataset with `euler-known-datasets prepare shift`. Its
[SHIFT page](https://github.com/d-rothen/euler-known-datasets/blob/main/docs/datasets/shift.md)
covers the downloads, decoding the videos, the frame selection and how the
depth and pose conventions were checked.

### Generic dense depth — `generic_dense_depth`

Format-agnostic; the loading strategy is inferred from the file extension.
Useful for datasets without a dedicated module.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Image (`.png`, `.jpg`, `.bmp`, `.tif`) or NumPy (`.npy`, `.npz`), `[0, 1]` |
| `depth` | 1HW / HW | float32 | Image or NumPy, returned as-is; honours `attributes["scale_to_meters_override"]` |
| `sky_mask` | 1HW / HW | bool | Compares pixels against the `skyclass` colour — required |
| `read_intrinsics` | 3×3 | float32 | Returns `meta["intrinsics"]`, ignores the path — requires `meta` · *hierarchical* |

### Generic NumPy modalities — `generic`

Dataset-agnostic loaders for `.npy` / `.npz` modalities, with no decoding
assumptions beyond shape and dtype.

| Function | Shape | dtype |
|---|---|---|
| `points_3d` | 3HW | float32 |
| `map_2d` | HW | float32 |
| `map_3d` | CHW / HWC | float32 |
| `spherical_map` | CHW / HWC | float32 |
| `atmospheric_light` | CHW / HWC | float32 |
| `scattering_coefficient` | HW | float32 |
| `semantic_segmentation` | HW | uint8 |
| `sky_mask` | HW | bool (head `skyclass` label, required) |
| `instance_segmentation` | HW | uint16 |
| `intrinsics` | 3×3 | float32 |
| `sh_coeffs` | (N, C) | float32 |

Writers exist for every function in this module except `sky_mask`.
