# euler-loading

Multi-modal PyTorch `Dataset` that synchronises files across arbitrary dataset modalities indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler).

Each modality points at a directory (or `.zip` archive) that carries its own `ds-crawler.config` (or cached `output.json`).
ds-crawler indexes the directory tree, discovers files, and exposes hierarchical metadata (path properties, calibration files, …).
euler-loading then **intersects file IDs** across all modalities so that every sample contains exactly one file per modality. Additional hierarchical data (e.g. per-scene calibration files) can be loaded via `hierarchical_modalities`. For augmentation-style datasets — many augmented files per ground-truth sample — `keyed_modalities` joins each augmented sample to its single GT file via the value encoded in the augmented sample's deepest hierarchy key.
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
| `path` | `str` | Absolute path to the modality root directory or `.zip` archive. Must contain a `ds-crawler.config` or cached `output.json`. |
| `origin_path` | `str \| None` | Original path before copying/symlinking (e.g. for SLURM staging). Not used by euler-loading itself — useful for experiment logging to retain references to the original dataset location. |
| `loader` | `Callable[..., Any] \| None` | Receives the file path (or `BinaryIO` buffer for zip-backed modalities) and an optional `meta` dict. Returns loaded data. When `None`, the loader is resolved automatically from the ds-crawler index (see [Automatic loader resolution](#automatic-loader-resolution)). |
| `writer` | `Callable[..., Any] \| None` | Receives `(path, value, meta)` and writes modality data to disk. When `None`, euler-loading tries to resolve a built-in writer from ds-crawler metadata (`write_<function>` or `write_<suffix>` for `read_<suffix>`). |
| `used_as` | `str \| None` | Optional experiment role (e.g. `input`, `target`, `condition`). |
| `slot` | `str \| None` | Optional fully-qualified logging slot (e.g. `dehaze.input.rgb`). |
| `modality_type` | `str \| None` | Optional modality type override (e.g. `rgb`, `depth`). |
| `hierarchy_scope` | `str \| None` | Optional scope label for hierarchical modalities (e.g. `scene_camera`). |
| `applies_to` | `list[str] \| None` | Optional list of regular modality names a hierarchical modality applies to. |
| `split` | `str \| None` | Optional inline split name. Loads `.ds_crawler/split_<name>.json` from the modality root (directory or zip) and overlays it on the normal ds-crawler metadata. |
| `keyed_by` | `Mapping[str, str] \| None` | Optional join configuration used only when this modality is passed under `MultiModalDataset(keyed_modalities=...)`. Recognised keys: `key_name` (named-group prefix at the regular modality's deepest hierarchy key, e.g. `"file_id"`; auto-detected from the anchor's data when omitted) and `modality` (anchor regular-modality name; auto-inferred when there's exactly one regular). Both keys are optional — the kwarg can be left unset entirely. |
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

This works for both directory-backed and zip-backed modalities. The split file only replaces the `dataset` payload; top-level metadata such as dataset type and euler-loading loader hints still come from the canonical ds-crawler index.

### `MultiModalDataset.modality_paths()`

Returns a dict mapping each regular modality name to `{"path": ..., "origin_path": ...}` and includes `split` when configured.

### `MultiModalDataset.hierarchical_modality_paths()`

Returns a dict mapping each hierarchical modality name to `{"path": ..., "origin_path": ...}` and includes `split` when configured.

### `MultiModalDataset.keyed_modality_paths()`

Returns a dict mapping each keyed modality name to `{"path": ..., "origin_path": ...}` plus `keyed_by_modality` (the resolved anchor) and `keyed_by_key_name` (the resolved or auto-detected key name). Includes `split` when configured. Useful for verifying which prefix the auto-detection picked.

### `MultiModalDataset.get_modality_metadata(modality_name)`

Returns the ds-crawler metadata dict for the given modality.

### `MultiModalDataset(modalities, hierarchical_modalities=None, transforms=None, keyed_modalities=None, strict_keyed=False)`

PyTorch `Dataset`. On construction it:

1. Runs `ds_crawler.index_dataset_from_path()` for every modality (regular, hierarchical, and keyed).
2. Computes the **sorted intersection** of file IDs across all regular modalities.
3. Validates keyed-modality joins for every common id; drops samples whose join misses (or raises when `strict_keyed=True`).
4. Logs warnings for unmatched files; raises `ValueError` when the intersection is empty or all samples are dropped by keyed-join validation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modalities` | `dict[str, Modality]` | At least one entry required. Keys become the sample dict keys. These modalities participate in ID intersection. |
| `hierarchical_modalities` | `dict[str, Modality] \| None` | Optional modalities whose files live at intermediate hierarchy levels (e.g. per-scene intrinsics). These do **not** participate in ID intersection. Each sample will contain a dict `{file_id: loaded_result}` with all files at or above the sample's hierarchy level. Results are cached so shared files are parsed only once. |
| `keyed_modalities` | `dict[str, Modality] \| None` | Optional modalities joined to a regular sample by the value of the regular sample's deepest hierarchy key. See [Keyed modalities](#keyed-modalities) below. Contributes a single loaded value per sample (not a dict). Results are cached, so a GT shared by N augmented samples is loaded once. |
| `transforms` | `list[Callable[[dict], dict]] \| None` | Applied in order after loading. Each receives and returns the full sample dict. |
| `strict_keyed` | `bool` | When `True`, missing or mis-decoded keyed-modality joins raise immediately at construction instead of warning + dropping the affected samples. Default `False`. |

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
    "<keyed_modality_name>": <loader result>,  # one entry per keyed modality (single value)
    ...
    "id":          str,                   # file ID (leaf only, shared across modalities)
    "full_id":     str,                   # full hierarchical path including file ID (e.g. "/scene/camera/frame")
    "meta":        {                      # per-modality ds-crawler file entries
        "<modality_name>": {"id": ..., "path": ..., "path_properties": ..., "basename_properties": ..., "attributes": ...},
        ...
    },
    "attributes":  {                      # per-modality top-level surface for file_entry["attributes"]
        "<modality_name>": {...},                # for regular and keyed modalities (single dict)
        "<hierarchical_modality_name>": {        # for hierarchical modalities (one dict per matched file)
            "<file_id>": {...},
            ...
        },
        ...
    },
}
```

Hierarchical and keyed modality results are cached so shared files are parsed only once.

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

`loader` is the module name (`vkitti2`, `real_drive_sim`, or `generic_dense_depth`) and `function` is the function within that module. The GPU variant is used by default.

Writer resolution uses the same module and function metadata:

- preferred explicit key: `euler_loading.writer_function`
- fallback naming: `write_<function>`
- for read-style functions: also tries `write_<suffix>` for `read_<suffix>`

## ds-crawler integration

Every modality root must be independently indexable by ds-crawler.
Place a `ds-crawler.config` in the root of each modality directory (or zip archive) — ds-crawler will then parse the directory tree and assign each file an ID derived from its path properties.
Files across modalities are matched by these IDs, so **the directory structure must be consistent** across modalities (identical hierarchy and naming conventions up to the modality-specific parts captured in the config).

Calibration files or other per-scene/per-sequence metadata can be loaded via `hierarchical_modalities`. These files are matched to samples based on their position in the hierarchy — all files at or above a sample's hierarchy level are included and cached for efficiency.

## Keyed modalities

A **keyed modality** joins each regular sample to a single file in another dataset by reading the *value* of the regular sample's deepest hierarchy key. This is the right tool when one modality augments samples of another at a different hierarchy depth — typically: many augmented files per ground-truth sample.

### When to use it

Layout where the augmented modality nests files under an extra `file_id:<id>` level, while the GT keeps the file id as the filename stem:

```
augmented_rgb/
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_10m.png
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_20m.png
  scene_000000/CS_FRONT/file_id:000000000000000026/mor_10m.png
  ...
gt_depth/
  scene_000000/CS_FRONT/000000000000000025.png
  scene_000000/CS_FRONT/000000000000000026.png
  ...
```

ds-crawler indexes both layouts unchanged; the augmented modality's `indexing.hierarchy.separator` must be `":"` (or any single separator) so the deepest key `"file_id:000…025"` decodes into `("file_id", "000…025")`.

### Wiring

```python
from euler_loading import Modality, MultiModalDataset

dataset = MultiModalDataset(
    modalities={
        "rgb_aug": Modality("/data/augmented_rgb", loader=load_rgb),
    },
    keyed_modalities={
        "depth": Modality("/data/gt_depth", loader=load_depth),
    },
)

sample = dataset[0]
# sample["rgb_aug"]   – the per-aug RGB
# sample["depth"]     – the GT depth for this file_id (single value, shared across augs)
# sample["full_id"]   – e.g. "/scene_000000/CS_FRONT/file_id:000…025/mor_10m"
# sample["id"]        – the augmentation's leaf id (e.g. "mor_10m")
```

`keyed_by` is optional — both `key_name` (the named-group prefix at the regular modality's deepest hierarchy key) and `modality` (the anchor regular modality) are auto-detected when unambiguous. Set them explicitly when:

- there are multiple regular modalities (anchor must be picked: `keyed_by={"modality": "rgb_aug", "key_name": "file_id"}`); or
- the anchor's deepest hierarchy keys mix multiple prefixes (e.g. some `file_id:…` and some `frame:…`).

### Sample shape

A keyed modality contributes a **single loaded value** per sample, unlike hierarchical modalities which return `{file_id: loader_result}`. The join returns exactly one record by construction (one GT per augmented sample's file id at the parent hierarchy prefix), so a dict would be misleading.

### How the join works

For each common id in the regular modalities, euler-loading:

1. Looks up the anchor record's `hierarchy_path`, e.g. `(scene_000000, CS_FRONT, file_id:000…025)`.
2. Splits the deepest key on the regular modality's `indexing.hierarchy.separator` → `("file_id", "000…025")`.
3. Verifies the key name matches the configured (or auto-detected) `key_name`.
4. Looks up the keyed modality's record at hierarchy `(scene_000000, CS_FRONT)` with `id == "000…025"`.

Loaded values are cached: the GT for `file_id:000…025` is decoded once and reused across every augmentation that points at it.

### Validation and missing joins

Construction-time validation runs the decoding for every common id. Samples whose decode or lookup fails are dropped from the dataset with a per-modality warning summarising the count. Pass `strict_keyed=True` to raise instead — useful in pipelines where dropped samples should be a hard error. Common error cases (with precise messages):

- key-name mismatch (deepest key starts with a different prefix);
- regular modality with no `indexing.hierarchy.separator`;
- ambiguous auto-detection (multiple distinct deepest-key prefixes in the anchor);
- regular modality with no hierarchy at all.

### Writing keyed-modality outputs

`MultiModalDataset.write_sample(...)` accepts keyed modality names alongside regular ones. The destination path is derived from the keyed record (the GT's relative path), so prediction outputs land in the GT-shape layout — *not* under a synthetic `file_id:` subdirectory.

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
