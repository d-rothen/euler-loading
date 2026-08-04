# Loaders & writers

A loader is the **only** place where domain-specific I/O happens. euler-loading
resolves *which* file a sample needs and hands it over; the loader decides what
that file means.

- [The contract](#the-contract)
- [Per-file attributes](#per-file-attributes)
- [Automatic loader resolution](#automatic-loader-resolution)
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

For compatibility, euler-loading can still read the legacy top-level
`euler_loading` mapping from older indexes. New datasets should declare the
addon in `dataset-head.json` as shown above.

Writers resolve from the same `addons.euler_loading` entry, in this order:

1. the explicit `writer_function` field,
2. for `function: "read_<suffix>"`, `write_<suffix>`,
3. `write_<function>`.

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
| `sky_mask` | 1HW / HW | bool | Sky colour `[90, 200, 255]` in the class mask |
| `read_intrinsics` | 3×3 | float32 | From a `.txt` file · *hierarchical* |
| `read_extrinsics` | N×1 | float32 | From a `.txt` file · *hierarchical* |

Writers exist for every modality above.

### MUSES — `muses`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Frame-camera PNG, `[0, 1]` |
| `reference_rgb` | CHW / HWC | float32 | Clear-weather reference frame, `[0, 1]` |
| `semantic_segmentation` | 1HW / HW | int64 | Cityscapes `labelIds` or `labelTrainIds` |
| `semantic_segmentation_color` | CHW / HWC | uint8 | Cityscapes `labelColor` PNG |
| `sky_mask` | 1HW / HW | bool | `trainId=10`, `labelId=23`, or RGB `meta["sky_class"]` |
| `panoptic_segmentation` | 1HW / HW | int64 | COCO-style RGB panoptic PNG → segment IDs |
| `lidar_point_cloud` | (N, 6) | float64 | `.bin`: `x, y, z, intensity, ring, timestamp` |
| `sparse_depth` | (N, 6) | float64 | Same points, for sparse-depth workflows |
| `read_intrinsics` | 3×3 | float32 | From `calib.json`, `sensor="rgb"` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | From `calib.json`, `transform="lidar2rgb"` · *hierarchical* |

`point_cloud` is an alias for `lidar_point_cloud`.

### Real Drive Sim — `real_drive_sim`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | PNG, `[0, 1]` |
| `depth` | 1HW / HW | float32 | `.npz`, metres |
| `class_segmentation` | 1HW / HW | int64 | Class IDs from the red channel of an RGBA PNG |
| `sky_mask` | 1HW / HW | bool | Class ID `29` |
| `calibration` | dict | — | `{sensor: {"K": (3,3), "T": (4,4), "distortion": (8,)}}` · *hierarchical* |
| `all_intrinsics` | dict | — | `{sensor: (3, 3)}` · *hierarchical* |
| `read_intrinsics` | 3×3 | float32 | Defaults to `CS_FRONT` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | `HDL_64E` → `CS_FRONT` · *hierarchical* |

### Princeton DENSE / SeeingThroughFog — `princeton_dense`

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Plain 8-bit PNG, `[0, 1]` |
| `rccb` | CHW / HWC | float32 | 12-bit Bayer (`GBRG`) TIFF debayered to RGB, `[0, 1]` |
| `sparse_depth` | (N, 5) | float32 | `.bin`: `x, y, z, intensity, ring` |
| `read_intrinsics` | 3×3 | float32 | `K` from `calib_cam_stereo_left.json` · *hierarchical* |
| `read_extrinsics` | 4×4 | float32 | `lidar_hdl64_s3_roof` → `cam_stereo_left_optical`, from `calib_tf_tree_full.json` · *hierarchical* |

`lidar_point_cloud` and `point_cloud` are aliases for `sparse_depth`.

### Generic dense depth — `generic_dense_depth`

Format-agnostic; the loading strategy is inferred from the file extension.
Useful for datasets without a dedicated module.

| Function | Shape | dtype | Notes |
|---|---|---|---|
| `rgb` | CHW / HWC | float32 | Image (`.png`, `.jpg`, `.bmp`, `.tif`) or NumPy (`.npy`, `.npz`), `[0, 1]` |
| `depth` | 1HW / HW | float32 | Image or NumPy, returned as-is; honours `attributes["scale_to_meters_override"]` |
| `sky_mask` | 1HW / HW | bool | Compares pixels against `meta["sky_mask"]` (`[R, G, B]`) — requires `meta` |
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
| `instance_segmentation` | HW | uint16 |
| `intrinsics` | 3×3 | float32 |
| `sh_coeffs` | (N, C) | float32 |

Writers exist for every function in this module.
