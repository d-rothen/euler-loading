# Changelog

## 2.27.0 (unreleased)

- Add a dry-run command: `euler-loading <path>` (also `python -m
  euler_loading <path>`) walks the ds-crawler artifact sets at or below one
  folder, resolves the loader each dataset head declares in
  `addons.euler_loading`, decodes a sample of the indexed files with it, and
  reports shapes, dtypes and value ranges. Directories and `.zip` archives
  are treated alike, nothing is written, and the exit status is non-zero when
  a modality fails to resolve or decode. `euler_loading.dry_run.dry_run()`
  exposes the same report to Python.
- `resolve_loader_module` and `resolve_writer_module` take a
  `variant="gpu"|"cpu"` keyword, so the CPU modules can be resolved through
  the same contract pathway as the torch ones. This is what the dry run's
  `--cpu` uses to check archives on a torch-free install.

## 2.26.0

- Add `read_extrinsics` to the Synscapes loaders, building a 4x4 rigid
  transform from the six `camera.extrinsic` scalars. `transform_direction`
  and `camera_axes` select the pose, its inverse, and vehicle or optical
  camera axes; the assumed `Rz(yaw) @ Ry(pitch) @ Rx(roll)` order is recorded
  in the modality metadata because the dataset does not document it.
- Add Synscapes writers for every modality, making the loader a
  `DenseDepthCodec`. Depth writes a float32 EXR `Z` channel and needs a
  filesystem path; the intrinsics and extrinsics writers merge into a single
  `meta/<id>.json`, replacing it in one step and keeping its permissions,
  symlink and unmodelled fields rather than truncating it in place.
- `sky_mask` now honours `meta['sky_class_id']`, or the same key in per-file
  attributes, instead of hardcoding Cityscapes ID 23, so a dataset that
  relabels sky round-trips through `write_sky_mask`.
- Accept the plural `camera.intrinsics` / `camera.extrinsics` spellings and
  report missing camera fields by name instead of raising `KeyError`.

## 2.24.0 (unreleased)

Add source-backed capture, strict per-output writers, explicit NPY/PNG encoding, typed calibration views and replay. Consolidate pinhole geometry and correct legacy skew scaling.


## 2.23.0 (source changes; not published)

- Reject wrapped decoders during dataset export by checking actual built-in
  function identity. Ordinary callable loading remains supported.
- Apply boolean thresholds before Pillow dtype conversion, and permit exact
  int32/int64 crop and identity-resize plans while refusing float32 resampling.
- Add `SamplePreprocessor.export_descriptor`, `MultiModalDataset.export_transform_plan`,
  `SerializableTransform`, and `resolve_transform_descriptor`, paired with contract 0.4.0.
- Freeze profiles, qualified source/calibration bindings, field policies, operation
  order, and installed CPU backend versions. Infer output profiles and virtual
  calibration geometry without materialization claims.
- Add explicit Torch/Pillow execution, corrected pinhole skew in the opt-in path,
  validity-aware depth policy, source immutability, and worker validator registration.
- Cover five-field round trips, projected geometry, ambiguous bindings, unsupported
  semantics, backend choices, schemas, and spawned workers in synthetic tests.

Legacy preprocessing and writer behavior remain unchanged. Execution receipts,
materialized metadata propagation, producer wiring, and GT replay are later work.
