"""Loader functions for the Princeton DENSE / SeeingThroughFog dataset.

This module re-exports the GPU-oriented loaders from
:mod:`euler_loading.loaders.gpu.princeton_dense` for convenience.  For
explicit control, import from :mod:`~euler_loading.loaders.gpu` or
:mod:`~euler_loading.loaders.cpu` directly::

    from euler_loading.loaders.gpu import princeton_dense   # torch tensors
    from euler_loading.loaders.cpu import princeton_dense   # numpy arrays
"""

from euler_loading.loaders.gpu.princeton_dense import *  # noqa: F401,F403
