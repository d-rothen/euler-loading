# Changelog

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
