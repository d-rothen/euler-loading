# Design note: passing per-file `attributes` to loaders

Status: **proposal — not implemented**

This is a follow-up to the per-file `attributes` field added in
ds-crawler (`file_entry["attributes"]`) and surfaced on samples in
euler-loading (`sample["attributes"][modality]` and
`sample["meta"][modality]["attributes"]`).

That change persists arbitrary per-file metadata through the index
schema and exposes it on the loaded sample. **It does not yet pass
`attributes` to the loader callable.** Loaders today receive only the
modality-level meta (range, dimensions, etc.) — the per-file dict is
available only after the loader has returned. This document describes
how to extend loader callables to accept per-file `attributes`.

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

## Current loader signature

`euler_loading/loaders/contracts.py:35-38`:

```python
def rgb(self, path: str | BinaryIO, meta: dict[str, Any] | None = None) -> torch.Tensor: ...
def depth(self, path: str | BinaryIO, meta: dict[str, Any] | None = None) -> torch.Tensor: ...
```

Call site in `euler_loading/dataset.py` (currently around line 793):

```python
sample[name] = self._resolved_loaders[name](file_or_path, modality_meta)
```

`modality_meta` here is `head["modality"]["meta"]` — not per-file.

## Proposed signatures

Three options, in order of recommendation:

### Option 1 — keyword-only `attributes` (recommended)

```python
def rgb(
    self,
    path: str | BinaryIO,
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor: ...
```

Rollout strategy:

1. Update `loaders/contracts.py` Protocols. Existing loaders that ignore
   the kwarg keep working — Protocol is structural; missing kwargs only
   break callers that try to pass them.
2. In `dataset.py`, call sites that need to pass `attributes` use
   `inspect.signature` (or a one-time per-loader feature probe cached on
   the Modality) to decide whether to pass it. Loaders that don't accept
   `attributes` get the old call.
3. Migrate built-in loaders one by one to accept `attributes=None`. They
   can ignore it; the kwarg merely makes them future-compatible.

Pros: zero breakage, opt-in per-loader, type-checker-friendly.
Cons: one-time signature probe at construction time.

### Option 2 — merge into `meta`

Extend `meta` from "modality-level meta" to "merged meta" — keep the
original keys plus a reserved `meta["__attributes__"]` (or
`meta["entry"]`) sub-dict. No signature change.

Pros: no signature change, no probe.
Cons: namespace collision risk; loaders parsing `meta` in surprising
ways may misinterpret the new key. Hides the source of each field.
Harder to type. Implicit contract.

### Option 3 — pass the full file entry

Replace `meta` with the file entry dict (carrying `attributes`,
`path_properties`, etc.). Strictly more information, but breaks every
existing loader.

Pros: maximum flexibility.
Cons: gratuitous churn, explicit migration of every loader.

**Pick Option 1 unless there's a strong reason not to.**

## Implementation sketch (Option 1)

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

Add a small utility that caches whether each loader accepts the
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

In `MultiModalDataset.__getitem__` (currently `dataset.py:793`):

```python
loader = self._resolved_loaders[name]
file_attrs = record.file_entry.get("attributes")
if self._loaders_accept_attributes[name]:
    sample[name] = loader(file_or_path, modality_meta, attributes=file_attrs)
else:
    sample[name] = loader(file_or_path, modality_meta)
```

Same treatment for the hierarchical-modality branch (currently
`dataset.py:816`). Be careful: hierarchical-modality results are
**cached** by `cache_key = f"{modality.path}/{entry['path']}"` — if two
samples that share the same hierarchical file have *different*
`attributes` (unlikely, but possible), the cache would return the wrong
loaded value. Either include the attributes hash in the cache key, or
document that hierarchical-modality `attributes` must be intrinsic to
the file (not the regular sample referencing it).

### 4. Update built-in loaders

For each module under `euler_loading/loaders/{cpu,gpu}/`, extend every
`read_*` and modality function with `*, attributes: ... = None`. Most
will just ignore it. Pick a representative one (e.g.
`generic_dense_depth`) to actually consume `attributes` — for instance
use `attributes.get("scale_to_meters_override")` to override the
modality-level scale.

### 5. Tests

Mirror the existing `tests/test_writing.py` and
`tests/test_dataset.py` structure. Minimum coverage:

- A loader that accepts `attributes=` receives the per-file dict.
- A loader that doesn't is called with the legacy 2-arg form (no
  `TypeError`).
- A loader with `**kwargs` is treated as accepting attributes.
- Hierarchical-modality cache: same hierarchical file with two different
  regular samples loads once (or, if you keyed on attributes, twice when
  attributes differ).
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
2. Hierarchical-modality cache keying — include `attributes` hash, or
   document the constraint? See step 3 above.
3. Should `loader_accepts_attributes` look at the function's
   `__wrapped__` chain to handle `functools.wraps`-decorated loaders?
   (Probably yes, but only if a real case appears.)
4. Worth adding a `FileContext` dataclass — `(meta, attributes,
   file_entry)` — to consolidate future fields rather than growing kwargs
   one at a time? Defer this until we have a second per-file thing to
   add; one new field doesn't justify a wrapper type.
