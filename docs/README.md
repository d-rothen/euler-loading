# euler-loading documentation

| Guide | Covers |
|---|---|
| [Dataset & modalities](dataset.md) | `Modality` and `MultiModalDataset` reference, the sample dict, hierarchical modalities, splits, scoped metadata, zip archives, layout-aware loading |
| [Loaders & writers](loaders.md) | The loader contract, per-file attributes, automatic resolution, protocols, and the full built-in loader inventory |
| [Preprocessing & transforms](preprocessing.md) | Cross-modal transforms, `SamplePreprocessor`, field kinds, calibration-aware resize and crop |
| [Writing outputs](writing.md) | Writing predictions back in dataset-native formats and re-indexing them |

Runnable scripts live in [`examples/`](../examples/). Contributor-facing notes —
adding a loader, running tests, cutting a release — are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Concepts in one page

**ds-crawler indexes, euler-loading joins.** Each modality root carries its own
[ds-crawler](https://github.com/d-rothen/ds-crawler) index describing the files
it contains and the hierarchy they sit in. euler-loading reads those indexes and
intersects file IDs so every sample holds one file per modality.

**Regular vs hierarchical modalities.** Regular modalities participate in the ID
intersection — one file each, per sample. Hierarchical modalities do not; their
files are matched by tree position, so a per-scene calibration file is shared by
every sample beneath it.

**Loaders own all file semantics.** euler-loading never interprets file
contents. It resolves which file to read and passes a path — or an in-memory
buffer, for zip-backed modalities — to a loader function.

**Metadata drives resolution.** Loader and writer choice, modality roles and
logging slots can all come from the index, so a dataset can describe how it
should be read.
