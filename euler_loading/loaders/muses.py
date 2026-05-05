"""Loader functions for the MUSES dataset.

This module re-exports the GPU-oriented loaders from
:mod:`euler_loading.loaders.gpu.muses` for convenience.  For explicit
control, import from :mod:`~euler_loading.loaders.gpu` or
:mod:`~euler_loading.loaders.cpu` directly::

    from euler_loading.loaders.gpu import muses   # torch tensors (CHW)
    from euler_loading.loaders.cpu import muses   # numpy arrays  (HWC)
"""

from euler_loading.loaders.gpu.muses import *  # noqa: F401,F403
