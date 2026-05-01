# Keyed Modalities Profile

## Scope

This note reviews the keyed-modality changes introduced by:

- `d9dbd25 add keyed modalities`
- `cb63d25 cache behaviour`
- `a51be96 docs`

The question was whether keyed modalities provide real value over the existing
`hierarchical_modalities` workflow.

## What the Commits Added

Keyed modalities add a third modality role alongside regular and hierarchical
modalities:

- `Modality.keyed_by` configures how a keyed modality is joined to a regular
  modality.
- `MultiModalDataset(..., keyed_modalities=..., strict_keyed=False)` indexes
  keyed roots separately from regular and hierarchical roots.
- A keyed lookup is built as `(parent_hierarchy_prefix, file_id) -> FileRecord`.
- The regular sample's deepest hierarchy key is decoded, for example
  `file_id:00025 -> ("file_id", "00025")`.
- The keyed record is loaded from the parent prefix with `id == "00025"`.
- The sample receives a single value, `sample["depth"]`, not a
  `{file_id: value}` dict.
- `write_sample()` can write keyed outputs back to the keyed modality's native
  path shape.
- Cache behavior is now configurable for hierarchical and keyed modalities.
  Hierarchical defaults to cached; keyed defaults to uncached.

## Benchmark Method

I profiled a synthetic augmentation workload with no-op loaders:

- Regular modality: `N` base file IDs, each with 4 augmented RGB files under
  `file_id:<id>`.
- Hierarchical depth shape: one depth file under each matching `file_id:<id>`
  hierarchy node.
- Keyed depth shape: one depth file at the parent prefix with `id == <id>`.
- Constructor timing includes indexing, lookup construction, and keyed join
  validation.
- Iteration timing loads every sample once.
- Results are median of 3 runs on this worktree.

The loader intentionally returns immediately, so these numbers show framework
overhead. Real image/depth IO will dominate per-sample time.

## Results

| IDs | Samples | Mode | Construct ms | Peak MiB | Iterate ms | us/sample | Depth loads |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 4,000 | hierarchical | 28.6 | 1.2 | 25.7 | 6.42 | 1,000 |
| 1,000 | 4,000 | hierarchical cache=False | 28.0 | 1.2 | 26.4 | 6.59 | 4,000 |
| 1,000 | 4,000 | keyed | 60.0 | 1.1 | 26.6 | 6.65 | 4,000 |
| 1,000 | 4,000 | keyed cache=True | 59.8 | 1.1 | 25.3 | 6.32 | 1,000 |
| 10,000 | 40,000 | hierarchical | 274.8 | 13.4 | 260.4 | 6.51 | 10,000 |
| 10,000 | 40,000 | hierarchical cache=False | 269.1 | 13.4 | 256.5 | 6.41 | 40,000 |
| 10,000 | 40,000 | keyed | 743.2 | 12.0 | 265.7 | 6.64 | 40,000 |
| 10,000 | 40,000 | keyed cache=True | 704.7 | 12.0 | 292.3 | 7.31 | 10,000 |
| 25,000 | 100,000 | hierarchical | 1,015.7 | 33.3 | 930.6 | 9.31 | 25,000 |
| 25,000 | 100,000 | hierarchical cache=False | 640.9 | 33.3 | 647.9 | 6.48 | 100,000 |
| 25,000 | 100,000 | keyed | 2,237.5 | 30.3 | 940.5 | 9.41 | 100,000 |
| 25,000 | 100,000 | keyed cache=True | 2,176.2 | 30.3 | 994.4 | 9.94 | 25,000 |

## Profile Notes

Constructor cost is where keyed modalities are meaningfully more expensive.
For 10,000 IDs x 4 augmentations, `cProfile` showed:

- hierarchical constructor: about 0.11 s inside `MultiModalDataset.__init__`
- keyed constructor: about 0.21 s inside `MultiModalDataset.__init__`
- keyed-specific validation `_filter_common_ids_by_keyed_joins()` accounted for
  about 0.036 s, and `_decode_keyed_lookup_key()` ran once per sample during
  validation.

Iteration cost is similar. For 40,000 samples:

- hierarchical iteration: about 0.458 s
- keyed iteration: about 0.529 s
- keyed `_decode_keyed_lookup_key()` was about 0.024 s total

Most iteration time in both paths is not keyed-specific. `_get_index_meta()` is
called per sample and dominates the profile because it re-derives metadata from
the index repeatedly.

## Value Assessment

Keyed modalities are not a speed feature. In this profile they make dataset
construction roughly 2x to 3x slower for the augmentation case because every
regular sample is validated against the keyed lookup. Steady-state access is
close to hierarchical access when loader cost is negligible.

The real value is semantic and operational:

- The keyed path preserves the natural ground-truth layout. GT files can stay at
  the parent prefix with filename/id `00025` while augmented files live under
  `file_id:00025/...`.
- The sample shape is cleaner for one-to-one joins: `sample["depth"]` is the
  depth value, not a one-entry dict that every transform or model wrapper must
  unwrap.
- Output writing can target the GT-shaped path instead of a synthetic
  `file_id:<id>` hierarchy.
- The diagnostic hint helps users who accidentally pass an augmentation-style GT
  dataset as a regular modality.
- The cache default is safer for large GT tensors. Hierarchical default caching
  is right for calibration-like files, but dangerous for hundreds of GB of
  depth/lidar tensors.

The value is much weaker when the current hierarchical workflow already models
the data naturally. If the GT index can cleanly place the GT file at the same
`file_id:<id>` hierarchy level as the augmented sample, hierarchical modalities
already solve the join and are simpler internally. The only ergonomic downside
is the returned one-entry dict.

My recommendation is to keep keyed modalities only if preserving native GT
layout and scalar sample shape matter to downstream users. If the project can
standardize on hierarchical GT indexes and consumers can tolerate or normalize
the one-entry dict, keyed modalities are a fairly large amount of extra join
logic for modest benefit.

## Follow-Up Ideas

- Cache per-modality metadata in `__init__` so `_get_index_meta()` is not called
  on every sample access.
- Consider making `strict_keyed=True` the recommended production setting. The
  default drop-and-warn behavior is convenient during exploration but can hide a
  bad join in long training runs.
- If keyed caching is useful but full caching is too risky, add a small bounded
  LRU cache instead of only `True` or `False`.
