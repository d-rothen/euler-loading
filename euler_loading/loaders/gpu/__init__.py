"""GPU-oriented loader functions that return torch tensors.

Each dataset submodule provides plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.
All loaders return **torch tensors** suitable for direct GPU-based training.

Available submodules:

- :mod:`euler_loading.loaders.gpu.vkitti2` — Virtual KITTI 2
"""
