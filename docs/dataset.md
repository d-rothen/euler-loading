# Dataset & modalities

Reference for `Modality` and `MultiModalDataset` — the two objects you need to
build a synchronised multi-modal dataset.

- [`Modality`](#modality)
- [`MultiModalDataset`](#multimodaldataset)
- [The sample dict](#the-sample-dict)
- [Hierarchical modalities](#hierarchical-modalities)
- [Inline splits](#inline-splits)
- [Scoped ds-crawler metadata](#scoped-ds-crawler-metadata)
- [Zip archives](#zip-archives)
- [Layout-aware augmentation loading](#layout-aware-augmentation-loading)
- [Introspection](#introspection)

## `Modality`

A frozen dataclass describing one data modality.

```python
Modality(path, *, loader=None, writer=None, split=None, metadata_scope=None, ...)
```

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to the modality root directory or `.zip` archive. Must carry ds-crawler metadata, either at `.ds_crawler/` or under the configured `metadata_scope`. Inline selectors are accepted: `/data/ds.zip:train`, `/data/ds.zip#scope=rgb`, `/data/ds.zip:train#scope=rgb`. |
| `origin_path` | `str \| None` | Original path before copying or symlinking (e.g. for SLURM staging). Unused by euler-loading — carried through so experiment logs can reference the original location. |
| `loader` | `Callable \| None` | Receives the file path (or a `BinaryIO` buffer for zip-backed modalities) and an optional `meta` dict. When `None`, resolved from the ds-crawler index — see [Automatic loader resolution](loaders.md#automatic-loader-resolution). |
| `writer` | `Callable \| None` | Receives `(path, value, meta)`. When `None`, euler-loading tries to resolve a built-in writer from ds-crawler metadata. See [Writing outputs](writing.md). |
| `used_as` | `str \| None` | Experiment role: `input`, `target`, `condition` or `output`. |
| `slot` | `str \| None` | Fully-qualified logging slot, e.g. `dehaze.input.rgb`. |
| `modality_type` | `str \| None` | Modality type override, e.g. `rgb`, `depth`. Also used as a hint by [`SamplePreprocessor`](preprocessing.md). |
| `hierarchy_scope` | `str \| None` | Scope label for hierarchical modalities, e.g. `scene_camera`. |
| `applies_to` | `list[str] \| None` | Regular modality names a hierarchical modality applies to. |
| `split` | `str \| None` | Inline split name. Loads `.ds_crawler/split_<name>.json` from the modality root and overlays it on the normal metadata. |
| `metadata_scope` | `str \| None` | Namespace below `.ds_crawler`, e.g. `.ds_crawler/camera_extrinsics/index.json`. See [Scoped ds-crawler metadata](#scoped-ds-crawler-metadata). |
| `cache` | `bool \| None` | In-memory caching of decoded values. Only meaningful for hierarchical modalities — regular modalities are never cached. `None` (default) means hierarchical modalities cache, because small shared calibration files benefit. |
| `collapse_single` | `bool` | Hierarchical modalities only. When `True` and exactly one hierarchical file matches, the loaded value is returned directly instead of `{file_id: value}`. |
| `metadata` | `dict[str, Any]` | Arbitrary metadata. Keys under `metadata["euler_loading"]` are treated as euler-loading defaults. |

`used_as`, `slot`, `modality_type`, `hierarchy_scope` and `applies_to` are
resolved in this order: explicit `Modality` field →
`Modality.metadata["euler_loading"]` → ds-crawler config
`properties["euler_loading"]` → heuristics.

## `MultiModalDataset`

```python
MultiModalDataset(modalities, hierarchical_modalities=None, transforms=None)
```

| Parameter | Type | Description |
|---|---|---|
| `modalities` | `dict[str, Modality]` | At least one entry. Keys become sample dict keys. These participate in ID intersection. |
| `hierarchical_modalities` | `dict[str, Modality] \| None` | Modalities whose files live at intermediate hierarchy levels. These do **not** participate in ID intersection. |
| `transforms` | `list[Callable[[dict], dict]] \| None` | Applied in order after loading. Each receives and returns the full sample dict. See [Preprocessing](preprocessing.md). |

On construction it:

1. Loads each modality's ds-crawler index — from
   `.ds_crawler/<metadata_scope>/index.json` when a scope is configured and
   present, otherwise from the root-level `.ds_crawler` layout.
2. Computes the **sorted intersection** of file IDs across all regular modalities.
3. Logs warnings for unmatched files, and raises `ValueError` if the
   intersection is empty.

Because IDs are matched across indexes, the indexed hierarchy and naming
conventions must be consistent across modalities, up to the modality-specific
parts captured in each ds-crawler config.

## The sample dict

`dataset[i]` returns:

```python
{
    "<modality>": <loader result>,          # one entry per regular modality
    "<hierarchical_modality>": {            # one entry per hierarchical modality
        "<file_id>": <loader result>,       # every file at or above this sample
    },
    "id":      str,                         # leaf file ID, shared across modalities
    "full_id": str,                         # full hierarchical path, e.g. "/scene/camera/frame"
    "meta": {                               # per-modality ds-crawler file entries
        "<modality>": {
            "id": ..., "path": ...,
            "path_properties": ..., "basename_properties": ...,
            "attributes": ...,
        },
    },
    "attributes": {                         # top-level surface for file_entry["attributes"]
        "<modality>": {...},                # regular modality: one dict
        "<hierarchical_modality>": {        # hierarchical: one dict per matched file
            "<file_id>": {...},
        },
    },
}
```

A hierarchical modality with `collapse_single=True` returns its loaded value
directly rather than a `{file_id: value}` dict, whenever exactly one file
matches.

## Hierarchical modalities

Files at intermediate levels — per-scene calibration, per-sequence intrinsics —
are matched to a sample by **position in the tree**: every file at or above the
sample's hierarchy level is included.

The ds-crawler file ID is the key in `{file_id: loaded_result}`. When files at
different ancestor levels share an ID, **the deepest match wins**. That gives
calibration natural inheritance semantics: a root-level `intrinsics` applies to
every sample, while a scene- or camera-level `intrinsics` overrides it for its
descendants.

Use ds-crawler's `indexing.id.override` when the physical filename is
incidental — `calib.json`, a UUID — and you want a stable semantic key such as
`intrinsics`, `extrinsics` or `calibration`. Do **not** override when several
distinct files at the same level should all be returned.

A root-level calibration file, such as `calib.json` at the top of a shared zip,
should be indexed by ds-crawler without a hierarchy block; euler-loading treats
it as an ancestor of every sample.

Results are cached by default so shared files are parsed once per process. Flip
this with `Modality(..., cache=True|False)`.

### Flattening a hierarchical modality

When a modality has exactly one file per hierarchy level, a transform can
collapse the dict — or set `collapse_single=True` and skip the transform:

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

## Inline splits

Given ds-crawler inline splits such as `.ds_crawler/split_train.json`:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb":   Modality("/data/rgb",   split="train"),
        "depth": Modality("/data/depth", split="train"),
    },
)
```

Path-only callers can use the colon form: `Modality("/data/rgb:train")`.

This works for directory- and zip-backed modalities alike. The split file only
replaces the `dataset` payload — top-level metadata such as dataset type and
loader hints still come from the canonical ds-crawler index.

## Scoped ds-crawler metadata

When one physical root or archive holds files for several logical modalities,
keep one ds-crawler artifact set per modality under `.ds_crawler/<scope>/`:

```
muses.zip
  calib.json
  frames/…
  lidar/…
  .ds_crawler/rgb/{dataset-head,ds-crawler,index}.json
  .ds_crawler/camera_extrinsics/{dataset-head,ds-crawler,index}.json
```

Then point several modalities at the same `path` and select the scope:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb":          Modality("/data/muses.zip", metadata_scope="rgb"),
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

Inline form — the scope selector comes *after* the split selector:

```python
Modality("/data/muses.zip#scope=rgb")
Modality("/data/muses.zip:train#scope=rgb")
```

When `metadata_scope` is omitted, it is resolved only if unambiguous:

1. An explicit `metadata_scope` or `#scope=…` wins.
2. If root-level `.ds_crawler` metadata exists, that root metadata is used.
3. If there is no root metadata and exactly one scope exists, that scope is used.
4. With multiple scopes, euler-loading matches the modality dict key,
   `modality_type`, the loader's `_modality_meta["type"]`, or the scoped
   `dataset-head.json` `modality.key`.
5. If still ambiguous, construction raises and lists the available scopes.

This layout is additive: roots with a plain `.ds_crawler/index.json` keep
loading unchanged, and a scoped modality falls back to that location when the
scoped artifacts are absent.

## Zip archives

Modality paths can point at `.zip` files instead of directories. Zip paths are
detected automatically and read without extraction:

```python
dataset = MultiModalDataset(
    modalities={
        "rgb":   Modality("/data/vkitti2/rgb.zip", loader=load_rgb),
        "depth": Modality("/data/vkitti2/depth",   loader=load_depth),  # mixing is fine
    },
)
```

- Loaders receive an `io.BytesIO` buffer with a `.name` attribute for extension
  detection, instead of a path.
- Each DataLoader worker process gets its own `ZipFile` handle, so multi-worker
  loading is safe.
- All built-in loaders accept `str` paths and `BinaryIO` buffers transparently.

## Layout-aware augmentation loading

Augmentation datasets hold many files per source sample, while a target such as
GT depth should be reused across every augmentation of that source. Index the
source sample as a hierarchy level and the augmentation variant as the file ID:

```
augmented_rgb/
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_10m.png
  scene_000000/CS_FRONT/file_id:000000000000000025/mor_20m.png
gt_depth/
  scene_000000/CS_FRONT/file_id:000000000000000025/depth.png
```

Both indexes carry an `euler_layout` addon. The augmented RGB layout declares a
`sample_axis` such as `file_id` at `location="hierarchy"` and a `variant_axis`
at `location="file_id"`; the GT depth layout declares the same `sample_axis` and
no variant axis.

```python
dataset = MultiModalDataset.from_layout(
    {
        "rgb_aug": Modality("/data/augmented_rgb", loader=load_rgb),
        "depth":   Modality("/data/gt_depth",      loader=load_depth),
    },
    primary="rgb_aug",
)

sample = dataset[0]
sample["rgb_aug"]  # the per-augmentation RGB
sample["depth"]    # GT depth for this file_id, shared across variants
sample["full_id"]  # "/scene_000000/CS_FRONT/file_id:000…025/mor_10m"
sample["id"]       # "mor_10m"
```

`from_layout(...)` plans the shared modality as
`hierarchical_modalities={"depth": Modality(..., collapse_single=True)}`. The
hierarchical match uses the regular sample's hierarchy path, so source samples
stay disambiguated by their parent hierarchy as well as by `file_id`.

If a shared modality declares the same `sample_axis` but its index does not
expose that axis as hierarchy, construction raises — re-index it with the source
sample axis in hierarchy.

## Introspection

| Method | Returns |
|---|---|
| `modality_paths()` | `{name: {"path", "origin_path"}}` for regular modalities, plus `split` and `metadata_scope` when configured. |
| `hierarchical_modality_paths()` | The same for hierarchical modalities. |
| `get_modality_metadata(name)` | The ds-crawler metadata dict for one modality. |
| `get_modality_index(name)` | The cached ds-crawler index for one modality. |
| `get_dataset_name()` | The dataset name from the first modality's index. |
| `describe_id_schema()` | The id-construction schema used by this dataset. |
| `describe_for_runlog()` | A structured descriptor for run metadata (below). |

```python
dataset.describe_for_runlog()
{
  "modalities": {
    "hazy_rgb": {
      "path": "…", "origin_path": "…",
      "used_as": "input", "slot": "dehaze.input.rgb", "modality_type": "rgb",
    },
  },
  "hierarchical_modalities": {
    "camera_intrinsics": {
      "path": "…", "origin_path": "…",
      "used_as": "condition", "slot": "dehaze.condition.camera_intrinsics",
      "hierarchy_scope": "scene_camera", "applies_to": ["hazy_rgb"],
    },
  },
}
```

### `FileRecord`

A frozen dataclass exposed for introspection, tying a ds-crawler file entry to
its position in the hierarchy.

| Field | Type | Description |
|---|---|---|
| `file_entry` | `dict[str, Any]` | Raw ds-crawler entry (`id`, `path`, `path_properties`, `basename_properties`, `attributes`). |
| `hierarchy_path` | `tuple[str, ...]` | Children keys from the dataset root to this file's parent node. Used to match against hierarchical modalities. |
