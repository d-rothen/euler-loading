"""Generic loader functions for arbitrary modalities.

This module re-exports the GPU-oriented loaders from
:mod:`euler_loading.loaders.gpu.generic` for backward compatibility.
For explicit control, import from :mod:`~euler_loading.loaders.gpu` or
:mod:`~euler_loading.loaders.cpu` directly::

    from euler_loading.loaders.gpu import generic   # torch tensors (CHW)
    from euler_loading.loaders.cpu import generic   # numpy arrays  (HWC)
"""

from euler_loading.loaders.gpu.generic import *  # noqa: F401,F403
