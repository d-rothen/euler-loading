# Serializable resize/crop plans

The 2.23.0 source changes add opt-in Phase 1 APIs paired with
`euler-dataset-contract>=0.4.0`. Install the matching source changes until they
are released. The normative wire format, canonical encoding, schemas, and shared
fixtures live in the contract package's `docs/phase1.md`.

Existing `SamplePreprocessor(sample)` behavior is preserved. Export a plan to
freeze its fields, reference image plane, calibration, and numerical backend:

```python
from euler_dataset_contract.testing import descriptor_fixture, evidence_cases
from euler_loading import SamplePreprocessor, execution_profile, resolve_transform_descriptor

# Synthetic declarations for the documented five-field example.
fixture = descriptor_fixture("five-field")["addon"]
config = next(c.payload["config"] for c in evidence_cases()
              if c.name == "documented-five-field")
transform = SamplePreprocessor.from_config(config)
descriptor = transform.export_descriptor(
    sources=fixture["sources"],
    fields=fixture["recipe"]["fields"],
    image_planes=fixture["recipe"]["image_planes"],
    execution=execution_profile("torch_cpu"),
)
plan = resolve_transform_descriptor(descriptor.to_json())
assert plan.operations[1].offset == (32, 64)
assert plan.infer_output_profiles()["rgb"].shape == (320, 640, 3)
# plan.bind_inputs(sample) validates a representative sample without transforming.
# output = plan(sample) performs the checked computation on detached values.
```

Real callers supply their own source identity and decoded profiles. A plausible
array shape or value range cannot establish units, correspondence, or camera
frame. `SourceBinding` includes dataset ID, exact modality key, metadata scope,
revision/head/index digests, decoder ID/version/profile digest, and verification
level. Source locators are optional and excluded from bound identity.
`CalibrationBinding` separately records its qualified full ID, revision,
camera model/frame/image plane, and applicability. `selection` is just the
dictionary lookup key, so ordinary calibration maps keyed by `camera_0` work.

`SerializableTransform` is the callable/export protocol. A `ResolvedTransform`
exposes immutable normalized descriptors, ordered `ResolvedOperation` objects,
`bind_inputs(sample)`, composed `image_transform`, `infer_output_profiles()`, and
`infer_output_bindings()`. The latter describes a new image plane and virtual
calibration views; it is not writer metadata or an execution receipt.

Export accepts either concrete input profiles alone, or those declarations plus
a representative sample. It freezes shorthand in resize-then-crop order and
expands defaults before hashing. Missing fields, incompatible kind/layout/dtype,
shape/plane/frame disagreement, unknown callable effects, invalid bounds, and
ambiguous calibration/reference choices fail explicitly. A single keyed
calibration can resolve `reduce="first"`; multiple entries require an explicit
selection. A sample never proves a calibration's physical applicability.

`MultiModalDataset.export_transform_plan(**bindings)` checks that the complete
sample chain is one supported preprocessor or resolved plan. It compares source
dataset IDs, keys, scopes, canonical head/index-tree digests, and built-in decoder
identity with the dataset. It refuses changing inherited calibration and opaque
decoder/sample callables. Standalone export trusts the declared input boundary;
pass any earlier transforms via `upstream_transforms` so unsupported effects are
rejected. Legacy callables remain usable through normal loading.

Built-in verification compares function objects against the supported installed
module, so wrappers cannot impersonate decoders by copying their name or decorator
metadata. Such wrappers can still run in ordinary loading.

The first executor supports concrete spatial layouts, floating depth/rays/point
maps, and declared pinhole K. `torch_cpu` accepts NumPy and CPU tensors, while
`pillow_cpu` explicitly uses Pillow for NumPy even when Torch is installed. Backend
and dependency versions, nearest convention, antialiasing, float32 spatial work,
output dtype preservation, integer rounding, threshold, and vector epsilon are
serialized. CUDA/autograd, int32/int64 through float32 interpolation, symbolic
sizes, other camera models, ray regeneration, and unsupported versions/features
are rejected. No cross-backend or encoded-byte equivalence is claimed.

Crop-only and identity-resize chains preserve int32/int64 labels exactly. The
precision check follows the resolved operation sizes and rejects those dtypes
only when a resize actually interpolates. Boolean thresholds are applied to the
floating interpolation result before casting, including with Pillow; the named
legacy truncation policy remains specific to integer arrays.

Depth defaults to the named `legacy_blend` invalid-value policy. Set the field's
`policy.invalid_depth` to `validity_aware` to interpolate finite non-sentinel
values/support separately and restore a declared sentinel where no support exists.
The operation preserves depth units and planar/radial kind. A resized validity
mask does not describe every contribution to the interpolated depth.

Ray policy is `preserve` or `resample_normalize`. The latter resamples and divides
by the L2 norm clamped to the explicit backend epsilon; zero/tiny vectors prevent
a universal unit-length guarantee. Output profiles therefore conservatively use
`normalization: "none"` after resize. Same-size resize is identity. Cropping
selects rays and leaves magnitude unchanged. Regeneration remains unsupported.

The checked pinhole path applies `K_out = C @ R @ K_in`, including skew. For
480×960 → 384×768 → center crop 320×640, crop offsets are 32,64 and K becomes
`[[640,0,319.5],[0,640,159.5],[0,0,1]]`. Tests include projected points, nonuniform
resize, odd/off-center crops, and source immutability. The older helper retains
its known skew behavior for legacy callers.

Loading registers descriptor validators on import in the parent and spawned
workers, and idempotently initializes them during binding/resolution. Core-only
and old crawler readers can preserve the new addon envelopes without claiming
execution support. Exact `1.0` versions are required by capable readers.
The registered representation validator also compares the containing head's
legacy key with the canonical identity and alias conditions.

The descriptor's only state is `planned`. Per-sample receipt capture, writer
metadata propagation, hierarchy-aware output calibration, preprocessing CLI
wiring, and evaluation replay remain later phases. The existing source-backed
writer still has the Phase 0 metadata caveats.

Run the normal unit suite with `pytest`; development dependencies include
`jsonschema` for schema/export conformance. Tests use synthetic inputs and CPU
Torch, including two spawned workers. Real datasets and CUDA parity are not part
of this verification.
