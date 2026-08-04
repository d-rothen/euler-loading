# Writing outputs

euler-loading can write inference results back in dataset-native formats,
preserving the source hierarchy so the output can be re-indexed with matching
IDs — and then loaded as just another modality.

- [Writer resolution](#writer-resolution)
- [`write_sample`](#write_sample)
- [Dataset writers](#dataset-writers)
- [Scoped output](#scoped-output)

## Writer resolution

A writer is any callable with the signature:

```python
writer(path: str, value: Any, meta: dict | None = None) -> None
```

Set it explicitly with `Modality(..., writer=...)`, or leave it `None` and let
euler-loading resolve a built-in writer from the `euler_loading` addon in the
modality's `dataset-head.json` contract:

1. the explicit `addons.euler_loading.writer_function` field,
2. for `function: "read_<suffix>"`, `write_<suffix>`,
3. `write_<function>`.

The same addon supplies the required `loader` and `function` fields used for
automatic loading. See [Automatic loader resolution](loaders.md#automatic-loader-resolution)
for a complete `dataset-head.json` example.

```python
writer = dataset.get_writer("depth")
```

`get_writer` raises `KeyError` for an unknown modality and `ValueError` when no
writer is configured or discoverable.

## `write_sample`

```python
dataset.write_sample(
    sample_index,
    outputs,
    output_root,
    *,
    create_dirs=True,
    overwrite=True,
    attributes=None,
)
```

| Parameter | Description |
|---|---|
| `sample_index` | Index in this dataset — used to recover the source relative paths. |
| `outputs` | `{modality_name: value}` to write. |
| `output_root` | One destination for all modalities, or a per-modality mapping. A destination is a filesystem root, a `DatasetWriter`, or a `ZipDatasetWriter`. |
| `create_dirs` | Create parent directories for filesystem destinations. |
| `overwrite` | When `False`, raise if a target file exists or a duplicate writer entry would be created. |
| `attributes` | `{modality: {key: value}}` recorded on the destination's ds-crawler file entry. Only applies to dataset writers; ignored for plain paths. When omitted, `attributes` from the source file entry are inherited. |

Returns `{modality_name: written_location}` — an absolute path for filesystem
destinations, or `"<archive>.zip::<relative/path>"` for zip destinations.

Filenames are derived from the ds-crawler relative paths, so the source
hierarchy is preserved beneath `output_root` and the result re-indexes with the
same IDs.

```python
for i in range(len(dataset)):
    sample = dataset[i]
    prediction = model(sample["rgb"])
    dataset.write_sample(i, {"depth": prediction}, "/out/predicted_depth")
```

## Dataset writers

Writing through a `DatasetWriter` also produces the `.ds_crawler` artifacts, so
the output is a properly indexed dataset rather than a bare directory tree:

```python
writer = dataset.create_output_writer("depth", "/out/predicted_depth")

for i in range(len(dataset)):
    dataset.write_sample(i, {"depth": predict(dataset[i])}, writer)

writer.save_index()
```

`create_output_writer(modality_name, root, *, zip=False, metadata_scope=None)`
mirrors the source modality's dataset-head contract and index metadata onto the
output, including its `addons.euler_loading` entry, so the new dataset describes
itself the same way the input did.

Pass `zip=True` for a `ZipDatasetWriter`. **One `ZipDatasetWriter` owns and
finalises one archive** — never open two concurrently for the same path.

The standalone `create_dataset_writer_from_index(...)` does the same from a raw
index payload rather than a live dataset:

```python
from euler_loading import create_dataset_writer_from_index

writer = create_dataset_writer_from_index(
    index_output=index,
    root="/out/predicted_depth",
    zip=False,
    metadata_scope=None,
)
```

## Scoped output

To emit several logical modalities into one physical root or archive, give each
writer a scope. ds-crawler writes the scoped artifacts and maintains the shared
`.ds_crawler/scopes.json` manifest:

```python
rgb_writer   = dataset.create_output_writer("rgb", output_root, metadata_scope="rgb")
depth_writer = dataset.create_output_writer("sparse_depth", output_root, metadata_scope="sparse_depth")
```

The resulting layout is exactly what
[scoped metadata](dataset.md#scoped-ds-crawler-metadata) expects on the way back
in.
