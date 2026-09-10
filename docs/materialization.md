# Captured spatial output (2.24.0, unreleased)

This opt-in path requires contract 0.5.0 and crawler 2.11.0. It supports the
planned crop/resize descriptors from `transform-descriptors.md`, and writes
materialized `euler_transforms` 2.0. Loading stored output never runs history.
Evaluator pairing and benchmark alignment are not implemented here.

```python
from euler_loading import SamplePreprocessor, execution_profile, output_encoding

# dataset is a MultiModalDataset with the actual source decoders and no opaque
# transforms. fields/image_planes are explicit Phase 1 descriptor bindings.
sources = dataset.describe_transform_sources(fields, revisions=source_revisions)
plan = SamplePreprocessor.from_config(preprocessing).export_descriptor(
    fields=fields, image_planes=image_planes, sources=sources,
    execution=execution_profile("torch_cpu"),
)
capture = dataset.enable_transform_capture({"default": plan})
ids = [dataset.sample_full_id(i, "rgb") for i in range(len(dataset))]
writer = dataset.create_output_writer(
    "rgb", "/output/rgb", derivation=plan,
    dataset_id="cropped_rgb", revision="export_v1",
    encoding=output_encoding("png_rgb8"), expected_full_ids=ids,
)
for i in range(len(dataset)):
    sample = dataset[i]  # values plus per-sample provenance, including in workers
    dataset.write_sample(i, {"rgb": sample["rgb"]}, writer,
                         provenance=sample["provenance"], sidecar=True)
writer.finalize()
```

Each selected output needs its own writer and distinct dataset ID. Depth uses
`output_encoding("png_depth16", scale=0.001)` or exact `npy`; mask uses
`png_mask8` or `npy`. NPY also preserves K and rays. Decoded depth must be in
`meter`; implicit unit conversion is refused. PNG rejects clipping and pins
NumPy/Pillow versions. Output profiles, dimensions, file types and depth scale
come from the output codec, and per-file encoding metadata is checked on reload.
Different variant sizes omit dataset-wide dimensions. The writer never uses
source metadata as encoder arguments.
PNG refuses quantization that changes invalid-pixel classification. Ordinary
source attributes are retained; current encoding and receipt metadata are rebuilt.
Validated ZIP destinations use the `.zip` suffix (also when `zip=True` is given).

To change output IDs, pass `output_full_ids={"rgb":"/scene/frame/crop_a"}` and
optionally `output_basenames={"rgb":"crop_a.png"}` to `write_sample`. Mapping is
explicit even if IDs are preserved. For multiple geometries/calibrations, freeze
`{variant_id: descriptor}` in both capture and the writer, then use
`dataset.capture_sample(i, variant_id="crop_a")`. A variant is a declared recipe,
not a filename convention. More than one calibration requires an explicit
qualified selection. This API executes one concrete source plane per variant;
it does not guess layouts, sensor pairing or per-frame crop boxes.

`CalibrationView(plan, execution_receipt, "intrinsics")` provides a typed
`one_per_output` view, its pinned source and applicability. Its `resolve(K)`
method checks decoded identity and reproduces the calibrated matrix without
mutating source K. Alternatively, create a writer for the `intrinsics` field to
materialize per-frame K as its own output modality.

```python
from euler_loading import CalibrationView, replay_from_arrays
view = CalibrationView(plan, sample["provenance"]["execution"], "intrinsics")
K_out = view.resolve(original_K)
replayed = replay_from_arrays(plan, sample["provenance"]["execution"], original_sample)
# Also check files and their pinned source identities:
capture.replay(0, sample["provenance"])
```

Array replay checks the supplied decoded arrays, not independently supplied
file hashes. Standalone `execute_with_receipt` therefore emits metadata-only
provenance. Strict materialization requires `DatasetCapture`: it checks complete
source file manifests and the actual decoder identity, decodes the same checked
bytes, and replays again before publication. Metadata-only and unverified
immutable-snapshot sources are refused. This deliberate strict scan/replay cost
is opt-in. The low-level `create_dataset_writer_from_index` accepts the same
output options plus `capture=`; it refuses an unsupported verification context.

Failed encoders commit no entry. Finalization checks the expected qualified IDs,
receipts, actual bytes and source-backed replay before publication. Directory
resume uses `resume=True` with the identical frozen plan/revision; repeating a
write/save is idempotent, while changed values/attributes/recipes conflict. ZIPs
are private until finalization and support one owner/one finalization, without
resume. Reading a dataset uses crawler's validated generation pointer and never
executes persisted operations or installs code.

The common geometry helpers now use `K_out = C @ R @ K`, including skew under
nonuniform resize. This corrects the previous legacy skew discrepancy and can
change floating-point rounding slightly compared with elementwise arithmetic.
Resize/crop ordering, half-pixel centers and explicit depth/mask/ray policies
remain unchanged. Existing plain callables and files-only destinations retain
their ordinary APIs; they do not acquire strict replay claims automatically.
