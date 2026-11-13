"""GPU-oriented loader functions that return torch tensors.

Each dataset submodule provides plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.
All loaders return **torch tensors** suitable for direct GPU-based training.

Available submodules:

- :mod:`euler_loading.loaders.gpu.vkitti2` — Virtual KITTI 2
- :mod:`euler_loading.loaders.gpu.synscapes` — Synscapes
- :mod:`euler_loading.loaders.gpu.real_drive_sim` — Real Drive Sim
- :mod:`euler_loading.loaders.gpu.muses` — MUSES
- :mod:`euler_loading.loaders.gpu.princeton_dense` — Princeton DENSE /
  SeeingThroughFog
- :mod:`euler_loading.loaders.gpu.kitti_carla` — KITTI-CARLA
- :mod:`euler_loading.loaders.gpu.driving_stereo` — DrivingStereo
- :mod:`euler_loading.loaders.gpu.nuscenes` — nuScenes
- :mod:`euler_loading.loaders.gpu.scatternerf` — ScatterNeRF
- :mod:`euler_loading.loaders.gpu.cerema` — CEREMA Pixel-Accurate Depth Benchmark
- :mod:`euler_loading.loaders.gpu.acdc` — ACDC
- :mod:`euler_loading.loaders.gpu.hypersim` — Hypersim
- :mod:`euler_loading.loaders.gpu.urbansyn` — UrbanSyn
- :mod:`euler_loading.loaders.gpu.generic` — format-agnostic NumPy modalities
- :mod:`euler_loading.loaders.gpu.generic_dense_depth` — format-agnostic
  dense-depth modalities
"""
