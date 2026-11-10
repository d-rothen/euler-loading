# Preprocessing & transforms

The opt-in [serializable resize/crop APIs](transform-descriptors.md) freeze these
authoring choices into validated Phase 1 descriptors. The callable path below
retains its legacy inference and numerical behavior.

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

`bind_to_dataset` (which `MultiModalDataset` calls for its transforms) also
reads how a depth or disparity head marks missing measurements, so sparse
ground truth is never blended with its invalid pixels:

| Head `modality.meta` | Depth / disparity resize |
|---|---|
| `"sparse": true` (with or without `invalid.sentinel`, default `0`) | `interpolation: "sparse"` |
| `"invalid": {"non_finite": …, "sentinel": s}` only, `s` not `null` | validity-aware bilinear with `invalid_value: s` |
| neither | bilinear, as before |

`invalid` and `sparse` are euler-dataset-contract's shared meta fields;
`invalid.sentinel: null` declares no sentinel and leaves the resize as it is.
The field counts as depth or disparity through its `modality_type`, else its
loader's `@modality_meta` type, else its name. A head that does not declare
`sparse` or `invalid` takes them from the loader annotation's `meta`, which
states what the loader always returns: DrivingStereo's `driving_stereo` maps
are sparse with `0` invalid, so they resample as sparse even from a head that
omits both. A disparity field is bound from its loader type even without
either, because its values rescale (see [Field kinds](#field-kinds)). An
explicit field spec always wins.

### Resampling sparse depth

Lidar-derived depth such as DrivingStereo or KITTI is sparse: most pixels hold
an invalid sentinel (usually `0`). Plain bilinear blends that sentinel into the
neighbouring measurements. A pixel between a 10 m sample and an empty one
becomes 5 m. The two invalid-aware modes avoid that:

- `interpolation: "sparse"` moves every valid pixel to the nearest pixel of the
  resized grid, through the same pixel-centre mapping `resize_intrinsics`
  applies to the camera, and keeps the nearest surface where several land on
  one pixel: the smallest depth, or the largest disparity. Every output value is a real measurement,
  nothing is interpolated across object edges, and upsampling leaves the map
  sparse rather than duplicating samples. Use it for lidar-like ground truth.
- `invalid_value` with `bilinear` or `bicubic` resizes the valid values and
  their support separately and divides, so invalid pixels carry no weight.
  Pixels without support get `invalid_value`. It still averages across depth
  edges, so use it for dense depth with holes (sky, missing returns), not
  for sparse samples.

For evaluation, prefer not to resample ground truth at all: resize the
prediction to the ground-truth grid and score the valid pixels.

## Field kinds

| Kind | Default interpolation | Behaviour |
|---|---|---|
| `image` | bilinear | Standard image resampling. Aliases: `rgb`. |
| `depth` | bilinear | Same resampling, kept distinct for clarity. |
| `disparity` | bilinear | Stereo disparity in pixels of its own grid. A resize multiplies valid values by the width ratio (halving the width halves the disparity); a crop leaves them. `sparse` keeps the largest disparity. Not exportable to a transform descriptor. Alias: `stereo_disparity`. |
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
| `interpolation` | Override the default: `nearest`, `bilinear`, `bicubic`, or `sparse` (see above). |
| `invalid_value` | Finite value marking pixels without a measurement; non-finite values then count as invalid too. Makes `bilinear`/`bicubic` validity-aware, and is the sentinel for `sparse` (default `0`). |
| `normalize_vectors` | Force vector renormalisation on or off (defaults on for `ray_map`). |
| `threshold` | Binarisation threshold for masks (default `0.5`). |
| `reduce` | For hierarchical fields that arrive as `{file_id: value}` — `"first"` picks a single entry. Useful for one intrinsics file per scene/camera. |

`nearest` samples at pixel centres with torch (`nearest-exact`) and Pillow
alike, the convention `resize_intrinsics` assumes. Halving a row picks source
pixels 1, 3, 5, … rather than torch's legacy 0, 2, 4, ….

## Intrinsics helpers

The intrinsics maths is also exposed directly:

```python
from euler_loading import resize_intrinsics, crop_intrinsics

K = resize_intrinsics(K, source_size=(375, 1242), target_size=(384, 768))
K = crop_intrinsics(K, top=32, left=64)
```

Both accept NumPy arrays and torch tensors and return the same type.
`resize_intrinsics` assumes an `align_corners=False` resize.

`resize_raster` resizes one raster exactly as the `resize` operation does, for
code that resizes outside a `SamplePreprocessor`:

```python
from euler_loading import resize_raster

rgb = resize_raster(rgb, layout="HWC", size=(512, 960), interpolation="bilinear")
depth = resize_raster(depth, layout="HW", size=(512, 960), interpolation="sparse")
```

The layout is explicit (`HW`, `HWC`, `CHW` or a batched form) and the
interpolation is `nearest`, `bilinear`, `bicubic` or `sparse`, all in the
`align_corners=False` convention, so the result matches `resize_intrinsics`.
`invalid_value=` works as the field option of the same name. Integer values
are rounded and booleans thresholded after interpolation; dtype and container
are kept.

Transform descriptors cannot export `sparse`, and export `invalid_value` only
with `policy.invalid_depth: "validity_aware"` and a matching profile
sentinel. Export refuses both otherwise rather than recording a different
resize than the one that runs.

`infer_field_spec(name, value=None)` exposes the name-based inference used
internally, returning a `FieldSpec` or `None`.
