"""CPU-oriented loader functions that return numpy arrays.

Each dataset submodule provides plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.
All loaders return **numpy ndarrays** for CPU-based processing without
requiring torch.

Available submodules:

- :mod:`euler_loading.loaders.cpu.vkitti2` — Virtual KITTI 2
- :mod:`euler_loading.loaders.cpu.real_drive_sim` — Real Drive Sim
- :mod:`euler_loading.loaders.cpu.muses` — MUSES
"""
