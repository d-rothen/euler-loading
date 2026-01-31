# euler-loading

Multi-modal PyTorch `Dataset` that synchronises files across arbitrary dataset modalities indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler).

Each modality points at a directory that carries its own `ds-crawler.config` (or cached `output.json`).
ds-crawler indexes the directory tree, discovers files, and exposes hierarchical metadata (path properties, calibration files, …).
euler-loading then **intersects file IDs** across all modalities so that every sample contains exactly one file per modality, plus any calibration data that ds-crawler found in the hierarchy.
How a file is actually **loaded** (image, depth map, point cloud, …) is entirely up to the caller — a plain `Callable[[str], Any]` per modality.

## Installation

```bash
pip install git+https://github.com/d-rothen/euler-loading.git
```

Requires Python >= 3.10. PyTorch and ds-crawler are pulled in automatically.

## Quick start

```python
from euler_loading import Modality, MultiModalDataset

dataset = MultiModalDataset(
    modalities={
        "rgb":   Modality("/data/vkitti2/rgb",   loader=load_rgb),
        "depth": Modality("/data/vkitti2/depth", loader=load_depth),
    },
    read_intrinsics=parse_intrinsics,   # optional
    read_extrinsics=parse_extrinsics,   # optional
    transforms=[normalize, augment],    # optional
)

sample = dataset[0]
# sample["rgb"]         – whatever load_rgb returned
# sample["depth"]       – whatever load_depth returned
# sample["intrinsics"]  – parsed intrinsics (or None)
# sample["extrinsics"]  – parsed extrinsics (or None)
# sample["id"]          – the file ID shared across modalities
# sample["meta"]        – per-modality ds-crawler file entries
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

### `MultiModalDataset(modalities, read_intrinsics=None, read_extrinsics=None, transforms=None)`

PyTorch `Dataset`. On construction it:

1. Runs `ds_crawler.index_dataset_from_path()` for every modality.
2. Walks the resulting hierarchical index, inheriting calibration paths from ancestor nodes.
3. Computes the **sorted intersection** of file IDs across all modalities.
4. Logs warnings for unmatched files; raises `ValueError` when the intersection is empty.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modalities` | `dict[str, Modality]` | At least one entry required. Keys become the sample dict keys. |
| `read_intrinsics` | `Callable[[str], Any] \| None` | Parses an intrinsics calibration file. |
| `read_extrinsics` | `Callable[[str], Any] \| None` | Parses an extrinsics calibration file. |
| `transforms` | `list[Callable[[dict], dict]] \| None` | Applied in order after loading. Each receives and returns the full sample dict. |

#### Sample dict

`dataset[i]` returns:

```python
{
    "<modality_name>": <loader result>,   # one entry per modality
    ...
    "id":          str,                   # file ID (shared across modalities)
    "meta":        {                      # per-modality ds-crawler file entries
        "<modality_name>": {"id": ..., "path": ..., "path_properties": ..., "basename_properties": ...},
        ...
    },
    "intrinsics":  <parsed> | None,       # from read_intrinsics
    "extrinsics":  <parsed> | None,       # from read_extrinsics
}
```

Calibration is resolved with **first-modality-wins** semantics and cached so shared calibration files are parsed only once.

### `FileRecord`

Frozen dataclass exposed for introspection. Each record ties a ds-crawler file entry to its resolved (absolute) calibration paths.

| Field | Type |
|-------|------|
| `file_entry` | `dict[str, Any]` — raw ds-crawler entry |
| `intrinsics_path` | `str \| None` |
| `extrinsics_path` | `str \| None` |

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

Calibration files (`camera_intrinsics`, `camera_extrinsics`) discovered by ds-crawler in the hierarchy are automatically inherited by descendant files and made available in the sample dict.

## Testing

```bash
pip install -e ".[dev]"

# unit tests (mocked, no data needed)
pytest

# integration tests against real on-disk datasets
pytest -m real
```

See `tests/test_real_dataset.py` for a full example of wiring up a real multi-modality dataset (VKITTI2).
