"""Loader functions for the Virtual KITTI 2 dataset.

This module re-exports the GPU-oriented loaders from
:mod:`euler_loading.loaders.gpu.vkitti2` for backward compatibility.
For explicit control, import from :mod:`~euler_loading.loaders.gpu` or
:mod:`~euler_loading.loaders.cpu` directly::

    from euler_loading.loaders.gpu import vkitti2   # torch tensors (CHW)
    from euler_loading.loaders.cpu import vkitti2   # numpy arrays  (HWC)
"""

from euler_loading.loaders.gpu.vkitti2 import *  # noqa: F401,F403
