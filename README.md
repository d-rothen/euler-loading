# euler-loading

Multi-modal PyTorch `Dataset` that synchronises files across arbitrary dataset modalities indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler).

Each modality points at a directory (or `.zip` archive) that carries its own `ds-crawler.config` (or cached `output.json`).
ds-crawler indexes the directory tree, discovers files, and exposes hierarchical metadata (path properties, calibration files, …).
euler-loading then **intersects file IDs** across all modalities so that every sample contains exactly one file per modality. Additional hierarchical data (e.g. per-scene calibration files) can be loaded via `hierarchical_modalities`.
How a file is actually **loaded** (image, depth map, point cloud, …) is configurable per modality — either supply a `Callable` or let euler-loading resolve a built-in loader automatically from the ds-crawler config.

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
        "rgb":   Modality("/data/vkitti2/rgb",   loader=load_rgb),
        "depth": Modality("/data/vkitti2/depth", loader=load_depth),
        "classSegmentation": Modality("/data/vkitti2/classSegmentation", loader=load_classSegmentation),
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
| `path` | `str` | Absolute path to the modality root directory or `.zip` archive. Must contain a `ds-crawler.config` or cached `output.json`. |
| `origin_path` | `str \| None` | Original path before copying/symlinking (e.g. for SLURM staging). Not used by euler-loading itself — useful for experiment logging to retain references to the original dataset location. |
| `loader` | `Callable[..., Any] \| None` | Receives the file path (or `BinaryIO` buffer for zip-backed modalities) and an optional `meta` dict. Returns loaded data. When `None`, the loader is resolved automatically from the ds-crawler index (see [Automatic loader resolution](#automatic-loader-resolution)). |
| `used_as` | `str \| None` | Optional experiment role (e.g. `input`, `target`, `condition`). |
| `slot` | `str \| None` | Optional fully-qualified logging slot (e.g. `dehaze.input.rgb`). |
| `modality_type` | `str \| None` | Optional modality type override (e.g. `rgb`, `depth`). |
| `hierarchy_scope` | `str \| None` | Optional scope label for hierarchical modalities (e.g. `scene_camera`). |
| `applies_to` | `list[str] \| None` | Optional list of regular modality names a hierarchical modality applies to. |
| `metadata` | `dict[str, Any]` | Optional arbitrary metadata. Keys under `metadata["euler_loading"]` are treated as euler-loading defaults. |

The loader is the **only** place where domain-specific I/O happens.
euler-loading never interprets file contents — it only resolves *which* file to load and passes the path (or in-memory buffer) to your function.

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

### `MultiModalDataset.modality_paths()`

Returns a dict mapping each regular modality name to `{"path": ..., "origin_path": ...}`.

### `MultiModalDataset.hierarchical_modality_paths()`

Returns a dict mapping each hierarchical modality name to `{"path": ..., "origin_path": ...}`.

### `MultiModalDataset.get_modality_metadata(modality_name)`

Returns the ds-crawler metadata dict for the given modality.

### `MultiModalDataset(modalities, hierarchical_modalities=None, transforms=None)`

PyTorch `Dataset`. On construction it:

1. Runs `ds_crawler.index_dataset_from_path()` for every modality (regular and hierarchical).
2. Computes the **sorted intersection** of file IDs across all regular modalities.
3. Logs warnings for unmatched files; raises `ValueError` when the intersection is empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modalities` | `dict[str, Modality]` | At least one entry required. Keys become the sample dict keys. These modalities participate in ID intersection. |
| `hierarchical_modalities` | `dict[str, Modality] \| None` | Optional modalities whose files live at intermediate hierarchy levels (e.g. per-scene intrinsics). These do **not** participate in ID intersection. Each sample will contain a dict `{file_id: loaded_result}` with all files at or above the sample's hierarchy level. Results are cached so shared files are parsed only once. |
| `transforms` | `list[Callable[[dict], dict]] \| None` | Applied in order after loading. Each receives and returns the full sample dict. |

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
    "meta":        {                      # per-modality ds-crawler file entries (regular modalities only)
        "<modality_name>": {"id": ..., "path": ..., "path_properties": ..., "basename_properties": ...},
        ...
    },
}
```

Hierarchical modality results are cached so shared files are parsed only once.

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

`loader` is the module name (`vkitti2`, `real_drive_sim`, or `generic_dense_depth`) and `function` is the function within that module. The GPU variant is used by default.

## ds-crawler integration

Every modality root must be independently indexable by ds-crawler.
Place a `ds-crawler.config` in the root of each modality directory (or zip archive) — ds-crawler will then parse the directory tree and assign each file an ID derived from its path properties.
Files across modalities are matched by these IDs, so **the directory structure must be consistent** across modalities (identical hierarchy and naming conventions up to the modality-specific parts captured in the config).

Calibration files or other per-scene/per-sequence metadata can be loaded via `hierarchical_modalities`. These files are matched to samples based on their position in the hierarchy — all files at or above a sample's hierarchy level are included and cached for efficiency.

## DenseDepthLoader protocol

`euler_loading.DenseDepthLoader` is a `runtime_checkable` Protocol defining the loader contract for dense-depth datasets. A conforming module must expose:

| Function | Return type |
|----------|-------------|
| `rgb(path, meta=None)` | `(3, H, W)` float32 in `[0, 1]` |
| `depth(path, meta=None)` | `(1, H, W)` float32 in metres |
| `sky_mask(path, meta=None)` | `(1, H, W)` bool |
| `read_intrinsics(path, meta=None)` | `(3, 3)` float32 camera matrix |

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

### Generic Dense Depth (`euler_loading.loaders.gpu.generic_dense_depth`)

A format-agnostic loader that infers the loading strategy from the file extension. Useful for datasets that don't have a dedicated loader module.

| Function | Description |
|----------|-------------|
| `rgb` | RGB from image files (`.png`, `.jpg`, `.bmp`, `.tif`) or NumPy files (`.npy`, `.npz`), normalised to [0, 1] |
| `depth` | Depth map from image or NumPy files, returned as-is (no unit conversion) |
| `sky_mask` | Binary mask by comparing pixels against `meta["sky_mask"]` (`[R, G, B]`). Requires `meta` |
| `read_intrinsics` | Returns `meta["intrinsics"]` as a `(3, 3)` tensor. Ignores path; requires `meta` |

CPU variants of all loaders live under `euler_loading.loaders.cpu.{vkitti2,real_drive_sim,generic_dense_depth}`.

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
