# Changelog

## Unreleased

- Fix `vkitti2.write_depth` and the 16-bit PNG path of
  `generic_dense_depth.write_depth` on Pillow 13, which rejects
  `Image.fromarray(uint16, mode="I;16")`; a uint16 array becomes an `I;16`
  image without the argument on every supported Pillow.
- `Modality.function` picks a callable from the loader module the dataset head
  names, instead of its `function`; the writer follows it.
  `Modality(path, function="sky_mask")` reads a `class_segmentation` index as
  boolean sky masks. It is exclusive with `Modality.loader`.
- The `sky_mask` loaders and writers of `real_drive_sim`, `vkitti2`,
  `synscapes`, `muses` and `generic_dense_depth` read the sky definition from
  the contract field `skyclass` (the sky label in the file's RGB view;
  `[v, v, v]` for a single-channel label): first from the file's attributes,
  then from the dataset head's meta. Their old hard-coded values are now
  defaults, and they log the definition they use once.
- A `null` `skyclass`, the default from euler-dataset-contract's next release,
  declares no sky: the loaders fall back to a deprecated key or their default,
  and `generic` / `generic_dense_depth` raise. `[0, 0, 0]`, the contract's old
  default, is still taken literally, but a head that declares it logs a warning
  naming it the old default, in place of the warning about differing from the
  loader's default. Loaders without a default warn too.
- Deprecate the per-loader sky keys `sky_class_id`, `sky_color`, `sky_class`
  and `sky_mask` in favour of `skyclass`. They are still read when no
  `skyclass` is given and raise a `DeprecationWarning`; when `skyclass` is
  also given it wins, and the ignored key is logged as a warning.
- Warn when the resolved sky differs from a loader's built-in default.
- Reject values a loader cannot match instead of returning empty masks: a
  non-grey `skyclass` for MUSES label PNGs (as for Synscapes and `generic`),
  and anything but `[id, 0, 0]` for Real Drive Sim.
- The dry run decodes segmentation heads with their loader module's
  `sky_mask` (at least `SKY_SAMPLES` files), reports the sky fraction, and
  warns when no sampled file has sky or one is more than 90% sky.
- `real_drive_sim.sky_mask` compares the PNG's RGB value with `skyclass`
  instead of only the red channel; for its own data, where sky is
  `(29, 0, 0)`, the masks are unchanged.
- Add `generic.sky_mask` for `.npy`/`.npz` class-id maps, which requires
  `skyclass`.
- The torch `sky_mask` loaders and writers of `real_drive_sim`, `vkitti2`,
  `muses` and `generic_dense_depth` wrap the NumPy ones instead of
  duplicating them.
- `muses.sky_mask`'s declared default meta is `skyclass: [23, 23, 23]`; the
  old `sky_class: [0, 0, 23]` matched no pixel of a `labelIds` PNG.
- `MultiModalDataset` logs each modality's effective loader, noting when
  `Modality.function` overrides the index.

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
