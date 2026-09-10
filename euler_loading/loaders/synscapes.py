"""Synscapes loaders, defaulting to the GPU-oriented tensor variant.

Import from :mod:`euler_loading.loaders.cpu.synscapes` for NumPy arrays or
:mod:`euler_loading.loaders.gpu.synscapes` for torch tensors explicitly.
"""

from euler_loading.loaders.gpu.synscapes import *  # noqa: F401,F403
