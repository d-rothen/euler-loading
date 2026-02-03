# euler-loading

Multi-modal PyTorch `Dataset` that synchronises files across arbitrary dataset modalities indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler).

Each modality points at a directory that carries its own `ds-crawler.config` (or cached `output.json`).
ds-crawler indexes the directory tree, discovers files, and exposes hierarchical metadata (path properties, calibration files, …).
euler-loading then **intersects file IDs** across all modalities so that every sample contains exactly one file per modality. Additional hierarchical data (e.g. per-scene calibration files) can be loaded via `hierarchical_modalities`.
How a file is actually **loaded** (image, depth map, point cloud, …) is entirely up to the caller — a plain `Callable[[str], Any]` per modality.

## Installation

```bash
uv pip install git+https://github.com/d-rothen/euler-loading.git
```

Requires Python >= 3.9. PyTorch and ds-crawler are pulled in automatically.

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

### `Modality(path, loader)`

Frozen dataclass describing one data modality.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Absolute path to the modality root. Must contain a `ds-crawler.config` or cached `output.json`. |
| `loader` | `Callable[[str], Any]` | Receives the absolute file path, returns loaded data. |

The loader is the **only** place where domain-specific I/O happens.
euler-loading never interprets file contents — it only resolves *which* file to load and passes the path to your function.

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

A loader is any callable with the signature `(path: str) -> Any`. Examples:

```python
from PIL import Image
import numpy as np

def load_rgb(path: str):
    return Image.open(path).convert("RGB")

def load_depth(path: str):
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

## ds-crawler integration

Every modality root must be independently indexable by ds-crawler.
Place a `ds-crawler.config` in the root of each modality directory — ds-crawler will then parse the directory tree and assign each file an ID derived from its path properties.
Files across modalities are matched by these IDs, so **the directory structure must be consistent** across modalities (identical hierarchy and naming conventions up to the modality-specific parts captured in the config).

Calibration files or other per-scene/per-sequence metadata can be loaded via `hierarchical_modalities`. These files are matched to samples based on their position in the hierarchy — all files at or above a sample's hierarchy level are included and cached for efficiency.

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

CPU variants live under `euler_loading.loaders.cpu.{vkitti2,real_drive_sim}`.