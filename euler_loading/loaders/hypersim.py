"""Hypersim loaders, defaulting to the GPU-oriented tensor variant.

Import from :mod:`euler_loading.loaders.cpu.hypersim` for NumPy arrays or
:mod:`euler_loading.loaders.gpu.hypersim` for torch tensors explicitly.
"""

from euler_loading.loaders.gpu.hypersim import *  # noqa: F401,F403
