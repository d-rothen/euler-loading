"""Loader functions for the Real Drive Sim dataset.

This module re-exports the GPU-oriented loaders from
:mod:`euler_loading.loaders.gpu.real_drive_sim` for convenience.
For explicit control, import from :mod:`~euler_loading.loaders.gpu` or
:mod:`~euler_loading.loaders.cpu` directly::

    from euler_loading.loaders.gpu import real_drive_sim   # torch tensors (CHW)
    from euler_loading.loaders.cpu import real_drive_sim   # numpy arrays  (HWC)
"""

from euler_loading.loaders.gpu.real_drive_sim import *  # noqa: F401,F403
