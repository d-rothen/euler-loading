# euler-loading

Multi-modal PyTorch `Dataset` that synchronises files across arbitrary dataset modalities indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler).

Each modality points at a directory (or `.zip` archive) that carries its own `ds-crawler.config` (or cached `output.json`).
ds-crawler indexes the directory tree, discovers files, and exposes hierarchical metadata (path properties, calibration files, …).
euler-loading then **intersects file IDs** across all modalities so that every sample contains exactly one file per modality. Additional hierarchical data (e.g. per-scene calibration files) can be loaded via `hierarchical_modalities`. For augmentation-style datasets — many augmented files per ground-truth sample — `euler_layout` metadata lets `MultiModalDataset.from_layout(...)` load shared targets as collapsed hierarchical modalities.
How a file is actually **loaded** (image, depth map, point cloud, …) is configurable per modality — either supply a `Callable` or let euler-loading resolve a built-in loader automatically from the ds-crawler config.
Writer functions can be resolved the same way, so inference outputs can be written back in dataset-native formats.

## Installation

```bash
uv pip install "euler-loading[gpu] @ git+https://github.com/d-rothen/euler-loading.git"
```

Requires Python >= 3.9. PyTorch and ds-crawler are pulled in automatically.

The `[gpu]` extra installs PyTorch. Without it the package still works but the GPU loader variants are unavailable — use the CPU (numpy) loaders instead.

## Quick start

```python
from euler_loading import Modality, MultiModalDataset

dataset = MultiModalDataset(
    modalities={
        "rgb":   Modality("/data/vkitti2/rgb",   loader=load_rgb, split="train"),
        "depth": Modality("/data/vkitti2/depth", loader=load_depth, split="train"),
        "classSegmentation": Modality("/data/vkitti2/classSegmentation", loader=load_classSegmentation, split="train"),
    },
    hierarchical_modalities={           # optional – for files at intermediate hierarchy levels
        "intrinsics": Modality("/data/vkitti2/intrinsics", loader=parse_intrinsics),
    },
    transforms=[normalize, augment],    # optional
)

sample = dataset[0]
# sample["rgb"]                 – whatever load_rgb returned
# sample["depth"]               – whatever load_depth returned
# sample["classSegmentation"]   – whatever load_classSegmentation returned
# sample["intrinsics"]          – dict {file_id: parsed_result} for hierarchical modality
# sample["id"]                  – the file ID (leaf only)
# sample["full_id"]             – full hierarchical path including file ID
# sample["meta"]                – per-modality ds-crawler file entries
```

Works with `torch.utils.data.DataLoader` out of the box.

## API

### `Modality(path, ..., loader=None, metadata=None)`

Frozen dataclass describing one data modality.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Absolute path to the modality root directory or `.zip` archive. Must contain ds-crawler metadata, either at `.ds_crawler/` or under the configured `metadata_scope`. Inline selectors are accepted as `/data/ds.zip:train`, `/data/ds.zip#scope=rgb`, or `/data/ds.zip:train#scope=rgb`. |
| `origin_path` | `str \| None` | Original path before copying/symlinking (e.g. for SLURM staging). Not used by euler-loading itself — useful for experiment logging to retain references to the original dataset location. |
| `loader` | `Callable[..., Any] \| None` | Receives the file path (or `BinaryIO` buffer for zip-backed modalities) and an optional `meta` dict. Returns loaded data. When `None`, the loader is resolved automatically from the ds-crawler index (see [Automatic loader resolution](#automatic-loader-resolution)). |
| `writer` | `Callable[..., Any] \| None` | Receives `(path, value, meta)` and writes modality data to disk. When `None`, euler-loading tries to resolve a built-in writer from ds-crawler metadata (`write_<function>` or `write_<suffix>` for `read_<suffix>`). |
| `used_as` | `str \| None` | Optional experiment role (e.g. `input`, `target`, `condition`). |
| `slot` | `str \| None` | Optional fully-qualified logging slot (e.g. `dehaze.input.rgb`). |
| `modality_type` | `str \| None` | Optional modality type override (e.g. `rgb`, `depth`). |
| `hierarchy_scope` | `str \| None` | Optional scope label for hierarchical modalities (e.g. `scene_camera`). |
| `applies_to` | `list[str] \| None` | Optional list of regular modality names a hierarchical modality applies to. |
| `split` | `str \| None` | Optional inline split name. Loads `.ds_crawler/split_<name>.json` from the modality root (directory or zip) and overlays it on the normal ds-crawler metadata. |
| `metadata_scope` | `str \| None` | Optional namespace below `.ds_crawler`, e.g. `.ds_crawler/camera_extrinsics/index.json`. Use this when multiple logical modalities share one physical directory or zip. It can also be supplied inline as `#scope=<metadata_scope>`. If omitted, euler-loading infers it only when the choice is deterministic. If configured artifacts are absent, loading falls back to the legacy root-level `.ds_crawler` layout. |
| `cache` | `bool \| None` | Opt-in/out for in-memory caching of decoded values (only meaningful for hierarchical modalities; regular modalities are never cached). `True` keeps every distinct loaded file in a process-lifetime dict. `False` re-reads on every access. `None` (default): hierarchical → `True` because small shared calibration files benefit. |
| `collapse_single` | `bool` | For hierarchical modalities only. When `True`, a sample that matches exactly one hierarchical file returns that loaded value directly instead of `{file_id: value}`. This is useful for shared GT tensors used by multiple augmentations of the same source sample. |
| `metadata` | `dict[str, Any]` | Optional arbitrary metadata. Keys under `metadata["euler_loading"]` are treated as euler-loading defaults. |

The loader is the **only** place where domain-specific I/O happens.
euler-loading never interprets file contents — it only resolves *which* file to load and passes the path (or in-memory buffer) to your function.

### `MultiModalDataset.get_writer(modality_name)`

Returns the resolved writer callable for a modality. Raises `ValueError` when no writer is configured/discoverable.

### `MultiModalDataset.write_sample(sample_index, outputs, output_root, ...)`

Writes one sample's modality outputs back to disk using resolved writers.

- `outputs` is `{modality_name: value}`.
- `output_root` is either one root path for all modalities or per-modality roots.
- Relative dataset paths are preserved under the output root(s), so generated data can be re-indexed with matching IDs.

### `MultiModalDataset.describe_for_runlog()`

Returns a structured descriptor for run metadata:

```python
{
  "modalities": {
    "hazy_rgb": {
      "path": "...",
      "origin_path": "...",
      "used_as": "input",
      "slot": "dehaze.input.rgb",
      "modality_type": "rgb",
    },
  },
  "hierarchical_modalities": {
    "camera_intrinsics": {
      "path": "...",
      "origin_path": "...",
      "used_as": "condition",
      "slot": "dehaze.condition.camera_intrinsics",
      "hierarchy_scope": "scene_camera",
      "applies_to": ["hazy_rgb"],
    },
  },
}
```

Resolution order is: explicit `Modality` fields -> `Modality.metadata["euler_loading"]` -> ds-crawler config `properties["euler_loading"]` -> heuristics.

### Inline split loading

If you created ds-crawler inline splits such as `.ds_crawler/split_train.json`, point a modality at that subset by setting `split`:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality("/data/rgb", split="train"),
        "depth": Modality("/data/depth", split="train"),
    },
)
```

Path-only callers can use the existing colon form directly:

```python
Modality("/data/rgb:train")
```

This works for both directory-backed and zip-backed modalities. The split file only replaces the `dataset` payload; top-level metadata such as dataset type and euler-loading loader hints still come from the canonical ds-crawler index.

### Scoped ds-crawler Metadata

When a single physical root or zip contains files for several logical modalities, keep one ds-crawler artifact set per modality under `.ds_crawler/<metadata_scope>/`:

```
muses.zip
  calib.json
  frames/...
  lidar/...
  .ds_crawler/rgb/dataset-head.json
  .ds_crawler/rgb/ds-crawler.json
  .ds_crawler/rgb/index.json
  .ds_crawler/camera_extrinsics/dataset-head.json
  .ds_crawler/camera_extrinsics/ds-crawler.json
  .ds_crawler/camera_extrinsics/index.json
```

Then point multiple modalities at the same `path` and select the appropriate metadata scope:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb": Modality("/data/muses.zip", metadata_scope="rgb"),
        "sparse_depth": Modality("/data/muses.zip", metadata_scope="sparse_depth"),
    },
    hierarchical_modalities={
        "camera_extrinsics": Modality(
            "/data/muses.zip",
            metadata_scope="camera_extrinsics",
            collapse_single=True,
        ),
    },
)
```

Path-only callers can select the same scope inline. The scope selector comes
after the optional split selector:

```python
Modality("/data/muses.zip#scope=rgb")
Modality("/data/muses.zip:train#scope=rgb")
```

If `metadata_scope` is omitted, euler-loading resolves it automatically only
when doing so is unambiguous:

1. Explicit `metadata_scope` or `#scope=...` wins.
2. If root-level `.ds_crawler` metadata exists, the legacy root metadata is used.
3. If no root metadata exists and exactly one scope exists, that scope is used.
4. If multiple scopes exist, euler-loading matches the modality dict key, `modality_type`, loader `_modality_meta["type"]`, or scoped `dataset-head.json` `modality.key`.
5. If the result is still ambiguous, construction raises and lists the available scopes.

This is an additive layout. Existing roots with `.ds_crawler/index.json` continue to load unchanged, and a scoped modality falls back to that legacy location when the scoped artifacts are not present.

### `MultiModalDataset.modality_paths()`

Returns a dict mapping each regular modality name to `{"path": ..., "origin_path": ...}` and includes `split` and `metadata_scope` when configured.

### `MultiModalDataset.hierarchical_modality_paths()`

Returns a dict mapping each hierarchical modality name to `{"path": ..., "origin_path": ...}` and includes `split` and `metadata_scope` when configured.

### `MultiModalDataset.get_modality_metadata(modality_name)`

Returns the ds-crawler metadata dict for the given modality.

### `MultiModalDataset(modalities, hierarchical_modalities=None, transforms=None)`

PyTorch `Dataset`. On construction it:

1. Loads each modality's ds-crawler index from `.ds_crawler/<metadata_scope>/index.json` when `metadata_scope` is configured and present; otherwise uses the legacy root-level ds-crawler lookup.
2. Computes the **sorted intersection** of file IDs across all regular modalities.
3. Logs warnings for unmatched files; raises `ValueError` when the intersection is empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modalities` | `dict[str, Modality]` | At least one entry required. Keys become the sample dict keys. These modalities participate in ID intersection. |
| `hierarchical_modalities` | `dict[str, Modality] \| None` | Optional modalities whose files live at intermediate hierarchy levels (e.g. per-scene intrinsics). These do **not** participate in ID intersection. Each sample will contain a dict `{file_id: loaded_result}` with all files at or above the sample's hierarchy level. Results are cached so shared files are parsed only once. |
| `transforms` | `list[Callable[[dict], dict]] \| None` | Applied in order after loading. Each receives and returns the full sample dict. |

For hierarchical modalities, the ds-crawler file ID is the key in
`{file_id: loaded_result}`. If files at different ancestor levels use the same
ID, the deepest matching file wins. This gives calibration files natural
inheritance semantics: a root-level `intrinsics` file can apply to every sample,
while a scene- or camera-level `intrinsics` file with the same ID overrides it
for descendants. Use ds-crawler's `indexing.id.override` when the physical file
name is incidental, such as `calib.json` or a UUID, and you want a stable
semantic key like `intrinsics`, `extrinsics`, or `calibration`. Do not use an
override when multiple distinct files at the same hierarchy level should all be
returned.

#### Sample dict

`dataset[i]` returns:

```python
{
    "<modality_name>": <loader result>,   # one entry per regular modality
    ...
    "<hierarchical_modality_name>": {     # one entry per hierarchical modality
        "<file_id>": <loader result>,     # all files at or above the sample's hierarchy level
        ...
    },
    ...
    "id":          str,                   # file ID (leaf only, shared across modalities)
    "full_id":     str,                   # full hierarchical path including file ID (e.g. "/scene/camera/frame")
    "meta":        {                      # per-modality ds-crawler file entries
        "<modality_name>": {"id": ..., "path": ..., "path_properties": ..., "basename_properties": ..., "attributes": ...},
        ...
    },
    "attributes":  {                      # per-modality top-level surface for file_entry["attributes"]
        "<modality_name>": {...},                # for regular modalities (single dict)
        "<hierarchical_modality_name>": {        # for hierarchical modalities (one dict per matched file)
            "<file_id>": {...},
            ...
        },
        ...
    },
}
```

Hierarchical modality results are cached by default. This can be flipped via `Modality(..., cache=True|False)` — see the [`cache` row in the `Modality` table](#modalitypath--loadernone-metadatanone).

### `FileRecord`

Frozen dataclass exposed for introspection. Each record ties a ds-crawler file entry to its position in the hierarchy.

| Field | Type | Description |
|-------|------|-------------|
| `file_entry` | `dict[str, Any]` | Raw ds-crawler entry (keys: `id`, `path`, `path_properties`, `basename_properties`). |
| `hierarchy_path` | `tuple[str, ...]` | Tuple of children keys from the dataset root to this file's parent node. Used for matching against hierarchical modalities. |

## Loader functions

A loader is any callable with the signature `(path: str | BinaryIO, meta: dict | None) -> Any`.
The `meta` argument receives the ds-crawler metadata for the modality (or `None` if unavailable).
For zip-backed modalities, `path` is an in-memory `io.BytesIO` buffer instead of a filesystem path.

A writer is any callable with the signature `(path: str, value: Any, meta: dict | None) -> None`.
Use `meta` for format parameters (units, encoding details) instead of modality-specific argument variants.

```python
from PIL import Image
import numpy as np

def load_rgb(path, meta=None):
    return Image.open(path).convert("RGB")

def load_depth(path, meta=None):
    return np.load(path)
```

## Transforms

Each transform receives the **full sample dict** (all modalities, calibration, metadata) and must return a dict.
This enables cross-modal operations:

```python
def mask_sky_in_depth(sample: dict) -> dict:
    seg = np.array(sample["segmentation"])
    sample["depth"][seg == SKY_CLASS] = 0.0
    return sample
```

For common mask-driven replacements, `MaskedValueOverride` can do the same
without writing a custom callable:

```python
from euler_loading import MaskedValueOverride

replace_sky_depth = MaskedValueOverride(
    target_key="depth",
    mask_key="sky_mask",
    value=300.0,
)
```

### Built-in spatial preprocessing

`euler_loading.SamplePreprocessor` applies shared spatial ops such as resize and crop
consistently across sample fields, including calibration-sensitive fields such as
camera intrinsics and ray maps.

```python
from euler_loading import FieldSpec, SamplePreprocessor

preprocessor = SamplePreprocessor.from_config(
    {
        "resize": [384, 768],
        "crop": {"size": [320, 640], "anchor": "center"},
        "fields": {
            "rgb": {"kind": "image"},
            "depth": {"kind": "depth"},
            "valid_mask": {"kind": "mask"},
            "intrinsics": {"kind": "intrinsics", "reduce": "first"},
            "ray_map": {"kind": "ray_map"},
        },
    }
)

dataset = MultiModalDataset(
    modalities={...},
    hierarchical_modalities={...},
    transforms=[preprocessor],
)
```

- `kind="image"` and `kind="depth"` default to bilinear interpolation.
- `kind="mask"` defaults to nearest-neighbour interpolation and preserves mask dtype.
- `kind="ray_map"` bilinearly resizes and renormalizes vectors to unit length.
- `kind="intrinsics"` rescales the camera matrix for resize and shifts the principal point for crop.
- `reduce="first"` is useful for hierarchical modalities such as one intrinsics file per scene/camera.

If you already set `modality_type` on your `Modality(...)` definitions, the preprocessor can bind to the dataset and reuse those hints automatically.

## Zip archive support

Modality paths can point to `.zip` files instead of directories. euler-loading detects zip paths automatically and reads files directly from the archive without extraction:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb":   Modality("/data/vkitti2/rgb.zip",   loader=load_rgb),
        "depth": Modality("/data/vkitti2/depth",      loader=load_depth),   # filesystem and zip can be mixed
    },
)
```

- Loaders receive an `io.BytesIO` buffer (with a `.name` attribute for extension detection) instead of a file path.
- Each DataLoader worker process gets its own `ZipFile` handle, so multi-worker loading is safe.
- Built-in loaders accept both `str` paths and `BinaryIO` buffers transparently.

## Automatic loader resolution

When `Modality.loader` is `None`, euler-loading resolves the loader from the ds-crawler index. The index must contain:

```json
{
  "euler_loading": {
    "loader": "vkitti2",
    "function": "rgb"
  }
}
```

`loader` is the module name (`vkitti2`, `real_drive_sim`, `muses`, `princeton_dense`, `generic`, or `generic_dense_depth`) and `function` is the function within that module. The GPU variant is used by default.

Writer resolution uses the same module and function metadata:

- preferred explicit key: `euler_loading.writer_function`
- fallback naming: `write_<function>`
- for read-style functions: also tries `write_<suffix>` for `read_<suffix>`

## ds-crawler integration

Each logical modality must have its own ds-crawler index. Usually that means one modality root with its own `.ds_crawler` artifacts. When several logical modalities share one physical directory or zip, put each artifact set under `.ds_crawler/<metadata_scope>/` and set `Modality(..., metadata_scope=...)` or append `#scope=<metadata_scope>` to the path.
Files across regular modalities are matched by IDs from those indexes, so **the indexed hierarchy and naming conventions must be consistent** across modalities up to modality-specific parts captured in the config.

Calibration files or other per-scene/per-sequence metadata can be loaded via `hierarchical_modalities`. These files are matched to samples based on their position in the hierarchy — all files at or above a sample's hierarchy level are included and cached for efficiency.
Root-level calibration files, for example `calib.json` at the top of a shared
zip, should be indexed by ds-crawler without a hierarchy block. euler-loading
treats that root file as an ancestor of every sample. If the modality uses
`collapse_single=True`, the loaded calibration value is returned directly as
long as exactly one hierarchical file matches the sample.

## Layout-aware augmentation loading

Augmentation datasets often contain many files for the same source sample, while a target modality such as GT depth should be reused for every augmentation of that source. The recommended layout is to index the source sample as a hierarchy level and the augmentation variant as the file ID:

```
augmented_rgb/
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_10m.png
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_20m.png
  scene_000000/CS_FRONT/file_id:000000000000000026/mor_10m.png
  ...
gt_depth/
  scene_000000/CS_FRONT/file_id:000000000000000025/depth.png
  scene_000000/CS_FRONT/file_id:000000000000000026/depth.png
  ...
```

Both modality indexes should include an `euler_layout` addon. The augmented RGB layout declares a `sample_axis` such as `file_id` at `location="hierarchy"` and a `variant_axis` at `location="file_id"`. The GT depth layout declares the same `sample_axis` and no variant axis.

```python
from euler_loading import Modality, MultiModalDataset

dataset = MultiModalDataset.from_layout(
    {
        "rgb_aug": Modality("/data/augmented_rgb", loader=load_rgb),
        "depth": Modality("/data/gt_depth", loader=load_depth),
    },
    primary="rgb_aug",
)

sample = dataset[0]
# sample["rgb_aug"]   – the per-augmentation RGB
# sample["depth"]     – the GT depth for this file_id, shared across variants
# sample["full_id"]   – e.g. "/scene_000000/CS_FRONT/file_id:000…025/mor_10m"
# sample["id"]        – the augmentation's leaf id, e.g. "mor_10m"
```

`from_layout(...)` plans the GT modality as `hierarchical_modalities={"depth": Modality(..., collapse_single=True)}`. The hierarchical match uses the regular sample's hierarchy path, so source samples remain disambiguated by their parent hierarchy as well as by `file_id`. If a shared modality declares the same `sample_axis` but its index does not expose that axis as hierarchy, construction raises and the dataset should be re-indexed with the source sample axis in hierarchy.

## DenseDepthLoader protocol

`euler_loading.DenseDepthLoader` is a `runtime_checkable` Protocol defining the loader contract for dense-depth datasets. A conforming module must expose:

| Function | Return type |
|----------|-------------|
| `rgb(path, meta=None)` | `(3, H, W)` float32 in `[0, 1]` |
| `depth(path, meta=None)` | `(1, H, W)` float32 in metres |
| `sky_mask(path, meta=None)` | `(1, H, W)` bool |
| `read_intrinsics(path, meta=None)` | `(3, 3)` float32 camera matrix |

`euler_loading.DenseDepthWriter` and `euler_loading.DenseDepthCodec` provide matching writer and combined reader/writer contracts.

```python
from euler_loading import DenseDepthLoader
from euler_loading.loaders.gpu import vkitti2

assert isinstance(vkitti2, DenseDepthLoader)
```

## Testing

```bash
pip install -e ".[dev]"

# unit tests (mocked, no data needed)
pytest

# integration tests against real on-disk datasets
pytest -m real
```

See `tests/test_real_dataset.py` for a full example of wiring up a real multi-modality dataset (VKITTI2).


## Use with pytorch DataLoaders
```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

for batch in loader:
    # batch["rgb"] is already (16, 3, H, W) — auto-collated by DataLoader
    ...
```

## Built-in loaders

`euler_loading.loaders` ships ready-made loader functions for supported datasets.
Each dataset has a **GPU** variant (returns `torch.Tensor` in CHW format) and a **CPU** variant (returns `np.ndarray` in HWC format).
The top-level imports (`euler_loading.loaders.vkitti2`, `euler_loading.loaders.real_drive_sim`) re-export the GPU variants for backward compatibility.

All built-in loaders accept both filesystem paths (`str`) and in-memory buffers (`BinaryIO`), so they work transparently with zip-backed modalities.

### Virtual KITTI 2 (`euler_loading.loaders.vkitti2`)

| Function | Description |
|----------|-------------|
| `rgb` | RGB image as float32, normalised to [0, 1] |
| `depth` | 16-bit PNG depth map, converted from centimetres to metres |
| `class_segmentation` | RGB-encoded class segmentation mask |
| `instance_segmentation` | RGB-encoded instance segmentation mask |
| `scene_flow` | Optical/scene flow map as float32, normalised to [0, 1] |
| `read_intrinsics` | Parses a 3×3 camera intrinsic matrix from a text file (use with `hierarchical_modalities`) |
| `read_extrinsics` | Parses a camera extrinsic matrix from a text file (use with `hierarchical_modalities`) |

### Real Drive Sim (`euler_loading.loaders.real_drive_sim`)

| Function | Description |
|----------|-------------|
| `rgb` | RGB image as float32, normalised to [0, 1] |
| `depth` | Depth from `.npz` files (metres) |
| `class_segmentation` | Single-channel class IDs extracted from the red channel of an RGBA PNG |
| `sky_mask` | Binary mask where class ID == 29 (sky) |
| `calibration` | Per-sensor calibration from JSON: returns `dict[sensor_name, {"K": (3,3), "T": (4,4), "distortion": (8,)}]` (use with `hierarchical_modalities`) |

### MUSES (`euler_loading.loaders.muses`)

| Function | Description |
|----------|-------------|
| `rgb` | Frame-camera RGB image as float32, normalised to [0, 1] |
| `reference_rgb` | Clear-weather reference RGB image as float32, normalised to [0, 1] |
| `semantic_segmentation` | Single-channel Cityscapes `labelIds` or `labelTrainIds` PNG as class IDs |
| `semantic_segmentation_color` | Cityscapes `labelColor` PNG as RGB uint8 labels |
| `sky_mask` | Binary mask from Cityscapes labels; uses sky `trainId=10` for `labelTrainIds`, `labelId=23` for `labelIds`, or RGB `meta["sky_class"]` such as `[0, 0, 23]` |
| `panoptic_segmentation` | COCO-style RGB panoptic PNG decoded to integer segment IDs |
| `lidar_point_cloud` / `point_cloud` | Lidar `.bin` file as `(N, 6)` float64: `x, y, z, intensity, ring, timestamp` |
| `sparse_depth` | Alias for MUSES lidar points for sparse-depth style workflows |
| `read_intrinsics` | RGB camera intrinsics from MUSES `calib.json` as a `(3, 3)` matrix; defaults to `sensor="rgb"` |
| `read_extrinsics` | Camera extrinsics from MUSES `calib.json` as a `(4, 4)` matrix; defaults to `transform="lidar2rgb"` for sparse-depth reprojection into RGB pixels |

### Princeton DENSE / SeeingThroughFog (`euler_loading.loaders.princeton_dense`)

| Function | Description |
|----------|-------------|
| `rgb` | Left/right stereo 12-bit Bayer TIFF as RGB float32, normalised to [0, 1] |
| `sparse_depth` | Lidar `.bin` point cloud as `(N, 5)` float32: `x, y, z, intensity, ring` |
| `read_intrinsics` | Left stereo camera intrinsics `K` from `calib_cam_stereo_left.json` as a `(3, 3)` matrix |
| `read_extrinsics` | Transform from HDL64 lidar frame `lidar_hdl64_s3_roof` to left camera optical frame `cam_stereo_left_optical` from `calib_tf_tree_full.json` as a `(4, 4)` matrix |

### Generic Dense Depth (`euler_loading.loaders.gpu.generic_dense_depth`)

A format-agnostic loader that infers the loading strategy from the file extension. Useful for datasets that don't have a dedicated loader module.

| Function | Description |
|----------|-------------|
| `rgb` | RGB from image files (`.png`, `.jpg`, `.bmp`, `.tif`) or NumPy files (`.npy`, `.npz`), normalised to [0, 1] |
| `depth` | Depth map from image or NumPy files, returned as-is (no unit conversion) |
| `sky_mask` | Binary mask by comparing pixels against `meta["sky_mask"]` (`[R, G, B]`). Requires `meta` |
| `read_intrinsics` | Returns `meta["intrinsics"]` as a `(3, 3)` tensor. Ignores path; requires `meta` |

### Generic NumPy Modalities (`euler_loading.loaders.gpu.generic`)

Generic loaders for `.npy` and `.npz` modalities without dataset-specific decoding.

| Function | Description |
|----------|-------------|
| `points_3d` | Dense 3D point map from NumPy files stored as `(3, H, W)` float32 |

CPU variants of all loaders live under `euler_loading.loaders.cpu.{vkitti2,real_drive_sim,muses,princeton_dense,generic,generic_dense_depth}`.

### Flattening hierarchical modalities

Hierarchical modalities always return `{file_id: loader_result}` because multiple files can match at different hierarchy levels. When a modality has exactly one file per hierarchy level (common for calibration), you can flatten this with a transform:

```python
dataset = MultiModalDataset(
    modalities={...},
    hierarchical_modalities={
        "calibration": Modality("/data/rds/calibration", loader=real_drive_sim.calibration),
    },
    transforms=[
        lambda sample: {
            **sample,
            "calibration": next(iter(sample["calibration"].values())),
        },
    ],
)

# Without the transform:  sample["calibration"]["<file_id>"]["CS_FRONT"]["K"]
# With the transform:     sample["calibration"]["CS_FRONT"]["K"]
```
