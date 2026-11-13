# Changelog

## Unreleased

- New loader module `urbansyn` (GPU and CPU) for UrbanSyn:
  - `rgb` for the images;
  - `depth` reads the Z-depth EXRs (stored in units of 1e5 m) as planar
    meters, with rays that hit nothing as `0`;
  - `semantic_segmentation` reads the Cityscapes train IDs (`19` for
    everything else), and `sky_mask` their sky (`skyclass` `[10, 10, 10]`);
  - `read_intrinsics` computes the camera matrix from the focal length and
    sensor size in `camera_metadata.json`;
  - `camera_pose` reads the camera-to-world pose from the render settings in
    a depth EXR's header.

  Every reader has a writer in the release's format; `write_intrinsics` and
  `write_camera_pose` write small JSON files. Depth and poses need OpenEXR,
  in the new extra `urbansyn`.
- New loader module `hypersim` (GPU and CPU) for Hypersim, as prepared by
  euler-known-datasets:
  - `rgb` tone-maps the HDR `color.hdf5` with Hypersim's operator, using each
    file's `tonemap_scale` attribute;
  - `depth` converts the released Euclidean distances to planar depth with
    each file's `intrinsics` attribute, and `radial_distance` returns them as
    released;
  - `read_intrinsics` and `camera_pose` read the per-trajectory calibration
    file: an exact pinhole matrix (tilt-shift scenes included) and the
    per-frame camera-to-world pose in meters.

  Writers mark their HDF5 files so that they read back unchanged, and
  `write_intrinsics` updates a calibration file after a crop or rescale. The
  module needs `h5py`, in the new extra `hypersim` (also added to `dev`).
- New loader module `acdc` (GPU and CPU) for ACDC: `rgb` for the
  adverse-condition images and `reference_rgb` for their normal-condition
  references, with writers for both.
- New loader module `cerema` (GPU and CPU) for the CEREMA Pixel-Accurate Depth
  Benchmark:
  - `rgb` for the stereo images;
  - `gated` for the gated slices, with the dark level removed;
  - `depth` for the npz depth maps (ground truth, projected lidar,
    `sparse2dense`);
  - `sgm_depth`, `psmnet_depth` and `monodepth_depth`, which convert those
    methods' stored disparities to metric depth on the left camera's grid, as
    the benchmark does;
  - `read_intrinsics`, which reuses the SeeingThroughFog camera-info reader.

  Writers exist for `rgb`, `gated`, `depth` and `read_intrinsics`.
- New loader module `scatternerf` (GPU and CPU) for ScatterNeRF's NSVF-layout
  sequences:
  - `rgb`, the rectified stereo frames;
  - `read_intrinsics`, the 3×3 from the stored 4×4;
  - `camera_pose`, the 4×4 camera-to-world pose in the per-sequence normalized
    units it is stored in;
  - `camera_pose_metric`, the same pose in meters, using the
    `scale_to_meters` attribute.

  Writers exist for `rgb`, `read_intrinsics` and `camera_pose`.
- Require `euler-dataset-contract>=0.14.0`, which owns the `euler_loading`
  addon rules; the local copy kept for older contracts is gone.
- New loader module `nuscenes` (GPU and CPU) for nuScenes as prepared by
  euler-known-datasets:
  - `rgb`;
  - `lidar_point_cloud` (`(N, 5)` float32 `.pcd.bin`), with aliases
    `point_cloud` and `sparse_depth`;
  - `read_intrinsics`;
  - `read_extrinsics`, the frame's lidar→camera transform through both vehicle
    poses, selected per sample by ID from the scene's calibration file
    (format `euler_known_datasets.nuscenes.camera_frames` v1).

  Writers exist for rgb, the point clouds and intrinsics. `write_intrinsics`
  accepts crops and rescales only.
- **Fixed:** with an explicit CPU loader, the modality's writer resolves from
  the CPU module too. It used to import the GPU module, which fails without
  torch. If the GPU module cannot be imported, writer resolution also falls
  back to the CPU module, whose writers accept arrays and tensors alike.
- The `euler_loading` addon rules come from euler-dataset-contract (0.14+),
  which now owns them, and ds-crawler applies them too. With an older contract
  euler-loading registers its own copy as before.
- Maps whose holes are NaN resize without the holes growing:
  `invalid_value=NaN` (`"nan"` in a config) makes `bilinear`/`bicubic`
  validity-aware with non-finite pixels as the invalid ones, and `sparse`
  leaves NaN where nothing lands. Plain bilinear spreads each NaN into every
  output pixel it touches. `bind_to_dataset` selects it for depth and
  disparity heads declaring euler-dataset-contract's
  `invalid: {"non_finite": true, "sentinel": null}`. An infinite
  `invalid_value` is still rejected, as is NaN for an integer array.
- **Fixed:** output writers (`create_output_writer`, used by euler-inference
  for outputs that mirror an input modality) save their index under
  ds-crawler's `index.json`, the file the dataset's `ds-crawler.json` names.
  They saved `output.json`, which ds-crawler 2.x does not read, so such an
  output could not be loaded back (by euler-eval, for example). The legacy
  top-level fields (`name`, `type`, `meta`, `euler_*`, `dataset`) stay in the
  index.
- Disparity resizes as a measurement in `SamplePreprocessor`, field kind
  `"disparity"` (alias `stereo_disparity`):
  - a resize multiplies valid values by the width ratio, since disparity is
    measured in pixels of the grid being resized; a crop leaves them. Plain
    bilinear used to leave DrivingStereo disparity unscaled and blend in its
    invalid zeros (a resized minimum of 0.004 px against 6.8 px);
  - `bind_to_dataset` applies the depth rules to disparity: `sparse: true`
    selects `interpolation="sparse"`, which now keeps the nearest surface for
    disparity as the largest value, and `invalid.sentinel` alone a
    validity-aware bilinear resize. A disparity field is recognised from its
    loader's type or its name even without either;
  - a head that does not declare `sparse` or `invalid` takes them from the
    loader annotation's `meta`, which states what the loader always returns.
    DrivingStereo's depth and disparity therefore resample as sparse even
    from a head that omits both; no other loader declares them;
  - descriptor export refuses disparity fields, whose rescale it cannot
    record;
  - a field kind without a default interpolation, such as an explicit
    `modality_type="semantic_segmentation"`, resizes like a generic array by
    dtype instead of raising `KeyError`.
- **Behaviour change:** `execution_profile("torch_cpu")` records
  `nearest: "half_pixel"`, so new transform descriptors resize masks by pixel
  centre (torch `nearest-exact`) like `SamplePreprocessor` does since 2.36.0.
  Before, a descriptor exported from a preprocessor replayed nearest fields
  with torch's floor-based `nearest`, half a source pixel off the live result
  and the resized intrinsics, so materialized masks did not match on-the-fly
  ones. The executor now honours a recorded `nearest`: descriptors that
  recorded `"floor"` still replay with it. `pillow_cpu` accepts
  `"half_pixel"` only.
- The GPU `driving_stereo` annotations are spelled out as literals, as in the
  other GPU modules, so euler-dataset-contract's loader snapshot can read them
  without importing the module (it failed with "Nonliteral declaration" on
  2.36.0). The annotations themselves are unchanged.

- Add the `driving_stereo` loader module (GPU and CPU) for DrivingStereo,
  resolvable as `loader: "driving_stereo"`. Everything is returned on the
  pixel grid of the half-size left image:
  - `rgb` reads the left or right JPEGs;
  - `depth` and `disparity` decode the half-size uint16 PNGs (`/ 256`, metres
    and half-size pixels, `0` invalid), which share the left image's grid;
  - `depth_full_size` and `disparity_full_size` decode the full-size maps
    (`/ 256` and `/ 128`) and resample them onto the half-size grid. Half-size
    pixel `(y, x)` sits on full-size pixel `(2y, 2x)`, where the calibrations
    (`P_full = 2 × P_half`) and the two map sizes agree. The default
    `full_size_resampling: "nearest"` falls back to the nearest valid sample
    within half a pixel, nearer surface first; `"subsample"` keeps only the
    co-located one. Disparity is halved into half-size pixels;
  - `read_intrinsics` returns the rectified `P_rect_101[:, :3]` from a
    `half-image-calib` file (`camera: "right"` for `P_rect_103`);
  - writers for every function: the uint16 encodings, full-size maps with
    each value at `(2y, 2x)`, and `write_intrinsics`, which applies a crop or
    rescale to every `P_rect_*` row of an existing calibration file.
- `_writer_utils.replace_text`, the text counterpart of `replace_json`, which
  now delegates to it.

- Resizing that respects invalid pixels, for sparse ground truth such as
  lidar depth:
  - `interpolation="sparse"` (in `FieldSpec` and `resize_raster`) moves every
    valid pixel to the nearest pixel of the resized grid, through the same
    pixel-centre mapping `resize_intrinsics` applies, and keeps the smallest
    value where several land on one pixel. It never interpolates or
    duplicates samples;
  - `invalid_value=` makes `bilinear`/`bicubic` validity-aware: valid values
    and their support are resized separately, and unsupported pixels get the
    sentinel;
  - `SamplePreprocessor.bind_to_dataset` picks these up for depth fields from
    the dataset head: `"sparse": true` selects `sparse`, and
    `"invalid": {"sentinel": s}` alone validity-aware bilinear. Explicit field
    specs still win; heads declaring neither resize as before;
  - descriptor export refuses `sparse`, and `invalid_value` without a
    matching `validity_aware` policy, rather than recording a different
    resize.
- **Behaviour change:** `nearest` resizing in `SamplePreprocessor` and
  `resize_raster` samples pixel centres on the torch path too
  (`nearest-exact`). It used torch's legacy floor-based `nearest`, half a
  source pixel off the `align_corners=False` convention `resize_intrinsics`
  assumes, and one source pixel off the Pillow fallback when halving. Masks
  and segmentations resized with `nearest` now line up with the resized
  images and intrinsics, and come out the same with or without torch.
  Transform descriptors keep their recorded per-backend convention.

- MUSES writers for every image and calibration modality, so derived datasets
  such as crops keep the MUSES formats: `write_rgb` / `write_reference_rgb`
  (8-bit PNG), `write_semantic_segmentation` (single-channel label PNG),
  `write_semantic_segmentation_color`, `write_panoptic_segmentation` (COCO RGB
  IDs), `write_sky_mask` (a sky / not-sky label PNG using the loader's sky
  label) and `write_intrinsics`. `write_intrinsics` updates
  `intrinsics[<sensor>].K` in an existing `calib.json`. The change must be a
  crop or rescale of the stored matrix and is applied to its float64 values,
  whole-pixel crops as exact integer shifts. `D`, the other sensors and all
  extrinsics are kept, and an unchanged matrix leaves the file untouched.

- Require ds-crawler 2.14.0, whose writers implement `path_prefix`. On an older
  ds-crawler, `path_prefix` now raises instead of being taken as a dataset
  property, which wrote the files to the root.

- Public accessors for producers of derived datasets:
  `MultiModalDataset.get_loader(name)` (the resolved decoding function),
  `get_loader_meta(name)` (a copy of its `@modality_meta` annotation: shape
  layout, type, dtype, ...) and `get_hierarchical_files(name)` (a hierarchical
  modality's entries by hierarchy level). `euler_loading.get_modality_meta(loader)`
  reads the annotation of any loader function.
- `create_output_writer` and `create_dataset_writer_from_index` take
  `path_prefix` (needs ds-crawler 2.14): scoped modalities sharing one root keep
  their files in separate directories.
- `euler_loading.resize_raster(value, layout=..., size=..., interpolation=...)`
  resizes one raster exactly as the `resize` preprocessing operation does.
- `princeton_dense.write_sparse_depth` (aliases `write_lidar_point_cloud`,
  `write_point_cloud`) writes SeeingThroughFog `(N, 5)` float32 lidar `.bin`
  files.

- Add the `kitti_carla` loader module (GPU and CPU) for KITTI-CARLA, the
  KITTI-like CARLA 0.9.10 dataset, resolvable as `loader: "kitti_carla"`:
  - `rgb` drops the constant alpha of the RGBA PNGs;
  - `depth` decodes CARLA's 24-bit `(R + 256 G + 65536 B) / (2**24 - 1) * 1000`
    planar depth in metres, with sky at the 1000 m far plane;
  - `class_segmentation` decodes the `images_ss` colours to CARLA's 23
    semantic tags (the IDs of the lidar's `semantic` column) and raises on a
    colour outside the palette; `sky_mask` uses `skyclass`, `(70, 130, 180)`
    by default;
  - `lidar_point_cloud` (aliases `point_cloud`, `sparse_depth`) reads the
    binary PLY sweeps as `(N, 7)` float64: `x, y, z, cos_angle_lidar_surface,
    timestamp, instance, semantic`;
  - `read_intrinsics` returns the paper's camera model (1392x1024, 72 degree
    horizontal FOV, `fx = fy ≈ 957.96`), as the dataset ships no intrinsics
    file; `read_extrinsics` reads `lidar_to_cam<N>.txt` into optical camera
    axes by default, so lidar projects through `read_intrinsics` directly;
  - `lidar_pose` reads `full_poses_lidar.txt` once and gives each sample its
    `(4, 4)` lidar-to-world pose at the start of its sweep, matched through
    the frame number in its file names;
  - writers for every image, lidar and calibration modality.

- **Behaviour change:** `princeton_dense.read_intrinsics` returns the
  rectified pinhole matrix `P[:, :3]` from `calib_cam_stereo_left.json`
  instead of the raw `K`. The `cam_stereo_left_lut` PNGs that
  `princeton_dense.rgb` loads are rectified to `P` -- the official
  SeeingThroughFog conversion remaps them with
  `cv2.initUndistortRectifyMap(K, D, R, P)`, and the official lidar projection
  uses `P` -- so `K` put projected lidar about 22 px off at the principal
  point and up to 130 px off at the image edges. Recompute anything derived
  from the old intrinsics, such as lidar projected into those images or depth
  unprojected with them. `intrinsics_matrix: "K"` in the per-file attributes
  or the head's meta selects the raw matrix, which belongs to the unrectified
  12-bit TIFFs together with the distortion `D`.
- Add `princeton_dense.write_intrinsics`. It writes the matrix
  `read_intrinsics` returns into a ROS camera-info JSON. An existing file is
  updated rather than replaced: a crop or rescale of its stored matrix is
  applied to `K` and `P` alike (whole-pixel crops as exact integer shifts),
  `D`, `R`, `roi` and the header are kept, and `meta["width"]` /
  `meta["height"]` set the recorded size. Any other change is refused. New
  files and stream targets get an ideal pinhole camera.
- **Behaviour change:** `princeton_dense.rccb` demosaics the raw 12-bit
  TIFFs as `GRBG` instead of `GBRG`, which swapped red and blue (red tail
  lights came out blue). The official conversion uses OpenCV's
  `COLOR_BAYER_GB2BGR`, and OpenCV names Bayer patterns after the sensor's
  second row. `bayer_pattern` in the attributes or meta still overrides it.
- Hierarchical loaders can narrow their shared value to each sample.
  `selects_per_sample(select)` attaches `select(value, context)` to a loader
  function. `MultiModalDataset` calls it after loading or retrieving the
  cached value. The `SampleContext` carries the sample's IDs, hierarchy and the
  index entries of its regular modalities. Use it for per-sample records in
  one shared file, such as a metadata table.
- MUSES ego-motion inputs, the ones the official SDK's lidar motion
  compensation needs:
  - `muses.gnss` loads the per-frame u-blox record of `gnss/` in physical units
    (speed, angular rates, attitude, position, NED velocity, accuracies);
  - `muses.sensor_timing` loads `meta.json` once as a root-level modality and
    gives each sample its camera-exposure, lidar and radar timestamps,
    matched through the file names the entries list;
  - `muses.read_lidar_to_gnss` reads `extrinsics.lidar2gnss` from
    `calib.json`;
  - `muses.write_sparse_depth` (aliases `write_lidar_point_cloud`,
    `write_point_cloud`) writes `(N, 6)` float64 `.bin` lidar files.

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
