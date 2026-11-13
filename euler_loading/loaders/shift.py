"""SHIFT loaders, defaulting to the GPU-oriented tensor variant.

Import from :mod:`euler_loading.loaders.cpu.shift` for NumPy arrays or
:mod:`euler_loading.loaders.gpu.shift` for torch tensors explicitly.
"""

from euler_loading.loaders.gpu.shift import *  # noqa: F401,F403
