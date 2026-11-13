"""Pre-defined loader functions for common datasets.

Each dataset has its own submodule with plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.

Loaders come in two variants:

- **gpu** – return ``torch.Tensor`` in CHW layout, ready for GPU training.
- **cpu** – return ``numpy.ndarray`` in HWC layout, for CPU-based processing.

Usage::

    # GPU loaders (torch tensors)
    from euler_loading.loaders.gpu import vkitti2

    # CPU loaders (numpy arrays)
    from euler_loading.loaders.cpu import vkitti2

    # Default (GPU) – backward-compatible shorthand
    from euler_loading.loaders import vkitti2

Every submodule exists under both ``gpu`` and ``cpu``:

- ``vkitti2`` — Virtual KITTI 2
- ``synscapes`` — Synscapes
- ``real_drive_sim`` — Real Drive Sim
- ``muses`` — MUSES
- ``princeton_dense`` — Princeton DENSE / SeeingThroughFog
- ``kitti_carla`` — KITTI-CARLA
- ``driving_stereo`` — DrivingStereo
- ``nuscenes`` — nuScenes, as prepared by euler-known-datasets
- ``scatternerf`` — ScatterNeRF (rectified SeeingThroughFog stereo, NSVF layout)
- ``cerema`` — CEREMA Pixel-Accurate Depth Benchmark (stereo, gated, lidar, dense ground truth)
- ``acdc`` — ACDC, adverse-condition images with normal-condition references
- ``hypersim`` — Hypersim, tone-mapped HDR renders with planar depth and exact tilt-shift cameras
- ``urbansyn`` — UrbanSyn, path-traced city frames with planar depth and Cityscapes labels
- ``generic`` — format-agnostic NumPy modalities
- ``generic_dense_depth`` — format-agnostic dense-depth modalities

``euler_loading/loaders/generate/loaders.json`` is the machine-readable
inventory of every annotated loader function, its output shape per variant,
dtype, unit and accepted file formats. Regenerate it with
``./gen_loaders.sh`` after adding or changing a loader.
"""
