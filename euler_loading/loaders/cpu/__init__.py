"""CPU-oriented loader functions that return numpy arrays.

Each dataset submodule provides plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.
All loaders return **numpy ndarrays** for CPU-based processing without
requiring torch.

Available submodules:

- :mod:`euler_loading.loaders.cpu.vkitti2` — Virtual KITTI 2
- :mod:`euler_loading.loaders.cpu.synscapes` — Synscapes
- :mod:`euler_loading.loaders.cpu.real_drive_sim` — Real Drive Sim
- :mod:`euler_loading.loaders.cpu.muses` — MUSES
- :mod:`euler_loading.loaders.cpu.princeton_dense` — Princeton DENSE /
  SeeingThroughFog
- :mod:`euler_loading.loaders.cpu.kitti_carla` — KITTI-CARLA
- :mod:`euler_loading.loaders.cpu.driving_stereo` — DrivingStereo
- :mod:`euler_loading.loaders.cpu.nuscenes` — nuScenes
- :mod:`euler_loading.loaders.cpu.scatternerf` — ScatterNeRF
- :mod:`euler_loading.loaders.cpu.cerema` — CEREMA Pixel-Accurate Depth Benchmark
- :mod:`euler_loading.loaders.cpu.acdc` — ACDC
- :mod:`euler_loading.loaders.cpu.generic` — format-agnostic NumPy modalities
- :mod:`euler_loading.loaders.cpu.generic_dense_depth` — format-agnostic
  dense-depth modalities
"""
