# Preprocessing & transforms

- [Transforms](#transforms)
- [`MaskedValueOverride`](#maskedvalueoverride)
- [`SamplePreprocessor`](#samplepreprocessor)
- [Field kinds](#field-kinds)
- [Intrinsics helpers](#intrinsics-helpers)

## Transforms

Each transform receives the **full sample dict** — every modality, plus
calibration and metadata — and returns a dict. That is what makes cross-modal
operations straightforward:

```python
def mask_sky_in_depth(sample: dict) -> dict:
    seg = np.array(sample["segmentation"])
    sample["depth"][seg == SKY_CLASS] = 0.0
    return sample

dataset = MultiModalDataset(modalities={...}, transforms=[mask_sky_in_depth])
```

Transforms run in list order, after loading.

## `MaskedValueOverride`

The common case above — write a constant into one field wherever a boolean mask
field is true — without writing a callable:

```python
from euler_loading import MaskedValueOverride

replace_sky_depth = MaskedValueOverride(
    target_key="depth",
    mask_key="sky_mask",
    value=300.0,
)
```

| Parameter | Default | Description |
|---|---|---|
| `target_key` | — | Field to write into. |
| `mask_key` | — | Boolean mask field selecting the positions. |
| `value` | — | Value written wherever the mask is true. |
| `invert_mask` | `False` | Write where the mask is *false* instead. |
| `copy` | `True` | Copy the target before writing rather than mutating in place. |
| `ignore_missing` | `False` | Pass the sample through unchanged when a key is absent, instead of raising. |

## `SamplePreprocessor`

Applies shared spatial operations — resize, crop — consistently across sample
fields, **including calibration-sensitive ones**. Resizing an image without
rescaling its camera matrix silently corrupts every downstream projection;
`SamplePreprocessor` keeps them in step.

```python
from euler_loading import SamplePreprocessor

preprocessor = SamplePreprocessor.from_config(
    {
        "resize": [384, 768],
        "crop": {"size": [320, 640], "anchor": "center"},
        "fields": {
            "rgb":        {"kind": "image"},
            "depth":      {"kind": "depth"},
            "valid_mask": {"kind": "mask"},
            "intrinsics": {"kind": "intrinsics", "reduce": "first"},
            "ray_map":    {"kind": "ray_map"},
        },
    }
)

dataset = MultiModalDataset(
    modalities={...},
    hierarchical_modalities={...},
    transforms=[preprocessor],
)
```

`resize` and `crop` are shorthand for the common single-step cases. For an
explicit pipeline use `operations`, which runs in the order given:

```python
{"operations": [
    {"type": "resize", "size": [384, 768]},
    {"type": "crop", "size": [320, 640], "anchor": "center"},
]}
```

`Crop` takes `anchor` (default `"center"`) or an explicit `offset` of
`[top, left]`.

If your `Modality(...)` definitions already set `modality_type`, calling
`preprocessor.bind_to_dataset(dataset)` enriches the field specs from that
metadata. With `infer_fields=True` (the default) common field names such as
`rgb`, `depth` and `sky_mask` are inferred even without an explicit spec.

## Field kinds

| Kind | Default interpolation | Behaviour |
|---|---|---|
| `image` | bilinear | Standard image resampling. Aliases: `rgb`. |
| `depth` | bilinear | Same resampling, kept distinct for clarity. |
| `mask` | nearest | Preserves mask dtype. Aliases: `segmentation`, `sky_mask`. |
| `ray_map` | bilinear | Resized, then vectors renormalised to unit length. Aliases: `rays`, `ray`, `camera_rays`, `spherical_map`. |
| `intrinsics` | — | Camera matrix rescaled on resize; principal point shifted on crop. Alias: `intrinsic`. |
| `passthrough` | — | Left untouched. |
| `generic` | inferred | Inferred from dtype: nearest for bool/int, bilinear otherwise. Alias: `auto`. |

Per-field options:

| Option | Description |
|---|---|
| `kind` | One of the kinds above. |
| `layout` | Explicit layout: `HW`, `CHW`, `HWC`, `NHW`, `NCHW`, `NHWC`. Inferred when omitted. |
| `interpolation` | Override the default: `nearest`, `bilinear`, `bicubic`. |
| `normalize_vectors` | Force vector renormalisation on or off (defaults on for `ray_map`). |
| `threshold` | Binarisation threshold for masks (default `0.5`). |
| `reduce` | For hierarchical fields that arrive as `{file_id: value}` — `"first"` picks a single entry. Useful for one intrinsics file per scene/camera. |

## Intrinsics helpers

The intrinsics maths is also exposed directly:

```python
from euler_loading import resize_intrinsics, crop_intrinsics

K = resize_intrinsics(K, source_size=(375, 1242), target_size=(384, 768))
K = crop_intrinsics(K, top=32, left=64)
```

Both accept NumPy arrays and torch tensors and return the same type.
`resize_intrinsics` assumes an `align_corners=False` resize.

`infer_field_spec(name, value=None)` exposes the name-based inference used
internally, returning a `FieldSpec` or `None`.
