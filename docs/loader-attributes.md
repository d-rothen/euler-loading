# Design note: passing per-file `attributes` to loaders

Status: **implemented**

This is a follow-up to the per-file `attributes` field added in
ds-crawler (`file_entry["attributes"]`) and surfaced on samples in
euler-loading (`sample["attributes"][modality]` and
`sample["meta"][modality]["attributes"]`).

That change persists arbitrary per-file metadata through the index
schema and exposes it on the loaded sample. Loader callables can now opt
in to receiving the per-file dict by accepting an `attributes=` keyword.
Legacy loaders that accept only `(path, meta)` keep receiving the old
call shape.

## When this matters

Cases where the loader genuinely needs `attributes` rather than the
consumer reading it post-hoc on the sample:

- Per-file noise level / scale that must be applied during decode
  (e.g. depth scale-to-meters that varies per scan).
- Per-file masks / clip ranges for normalization.
- Per-file lookup keys that select an external resource (e.g. a
  per-file calibration tag pointing into a calibration database).
- Anything where the value can't be reconstructed once the file is
  decoded into a tensor.

If `attributes` is purely descriptive (training weight, source tag,
log fields), keep it on `sample["attributes"][...]` and don't change
loader signatures.

## Loader signature

`euler_loading/loaders/contracts.py`:

```python
def rgb(
    self,
    path: str | BinaryIO,
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor: ...
```

Implemented behavior:

- `dataset.py` probes each resolved loader once with `inspect.signature`
  and caches whether it accepts `attributes` or `**kwargs`.
- Compatible loaders receive a copied per-file attributes dict (or
  `None` when the file entry has no attributes).
- Legacy loaders that do not accept the keyword receive the original
  `(path, meta)` call.
- Hierarchical-modality cache keys include the attributes payload for
  opted-in loaders, so the same path with different attributes can be
  decoded differently.
- Built-in CPU/GPU loaders accept `attributes=None`. Most ignore it;
  `generic_dense_depth.depth` consumes
  `attributes["scale_to_meters_override"]` when present.

### Rejected alternative: merge into `meta`

Extend `meta` from "modality-level meta" to "merged meta" — keep the
original keys plus a reserved `meta["__attributes__"]` (or
`meta["entry"]`) sub-dict. No signature change.

Pros: no signature change, no probe.
Cons: namespace collision risk; loaders parsing `meta` in surprising
ways may misinterpret the new key. Hides the source of each field.
Harder to type. Implicit contract.

### Rejected alternative: pass the full file entry

Replace `meta` with the file entry dict (carrying `attributes`,
`path_properties`, etc.). Strictly more information, but breaks every
existing loader.

Pros: maximum flexibility.
Cons: gratuitous churn, explicit migration of every loader.

## Implementation notes

### 1. Update contracts

```python
# euler_loading/loaders/contracts.py
class DenseDepthLoader(Protocol):
    def rgb(
        self,
        path: Union[str, BinaryIO],
        meta: dict[str, Any] | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> torch.Tensor: ...
    # ... and parallel updates to depth / sky_mask / read_intrinsics ...
```

### 2. Loader feature probe

The utility that identifies whether each loader accepts the
`attributes` kwarg:

```python
# euler_loading/_resolution.py
import inspect

def loader_accepts_attributes(loader: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(loader)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "attributes" in params:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
```

Cache the result on the resolved-loaders dict at construction time so
the probe runs once per modality.

### 3. Update call sites

In `MultiModalDataset.__getitem__`:

```python
loader = self._resolved_loaders[name]
file_attrs = _get_file_attributes(record.file_entry)
if self._loaders_accept_attributes[name]:
    sample[name] = loader(file_or_path, modality_meta, attributes=file_attrs)
else:
    sample[name] = loader(file_or_path, modality_meta)
```

The hierarchical-modality branch follows the same rule. For opted-in
loaders, the hierarchical cache key includes a serialized attributes
fragment so identical paths with different per-file attributes are
loaded independently.

### 4. Update built-in loaders

For each module under `euler_loading/loaders/{cpu,gpu}/`, every `read_*`
and modality function accepts `*, attributes: ... = None`. Most ignore
it. `generic_dense_depth.depth` consumes
`attributes.get("scale_to_meters_override")`.

### 5. Tests

Mirror the existing `tests/test_writing.py` and
`tests/test_dataset.py` structure. Minimum coverage:

- A loader that accepts `attributes=` receives the per-file dict.
- A loader that doesn't is called with the legacy 2-arg form (no
  `TypeError`).
- A loader with `**kwargs` is treated as accepting attributes.
- Hierarchical-modality cache: same hierarchical file with two different
  attributes loads twice.
- Round-trip: file entry attributes → loader receives them → if the
  loader passes them through, they appear on `sample["attributes"]`
  (already the case post the existing change, but pin it).

## Backwards compatibility

- Stability of the existing 2-arg signature is preserved by feature
  probing. Loaders that never opt in keep working forever.
- The `attributes` argument is keyword-only — no position collision.
- `sample["attributes"]` and `sample["meta"][name]["attributes"]`
  surfaces are unchanged by this work; they already exist.

## Open questions for the implementer

1. Should the feature probe also be done for *writers*? Today writers
   take `(path, value, meta)`. If we want symmetry, writers gain
   `attributes=` too. Same Option 1 treatment.
2. Should `loader_accepts_attributes` look at the function's
   `__wrapped__` chain to handle `functools.wraps`-decorated loaders?
   (Probably yes, but only if a real case appears.)
3. Worth adding a `FileContext` dataclass — `(meta, attributes,
   file_entry)` — to consolidate future fields rather than growing kwargs
   one at a time? Defer this until we have a second per-file thing to
   add; one new field doesn't justify a wrapper type.
