"""CEREMA Pixel-Accurate Depth Benchmark loaders, defaulting to the GPU-oriented tensor variant.

Import from :mod:`euler_loading.loaders.cpu.cerema` for NumPy arrays or
:mod:`euler_loading.loaders.gpu.cerema` for torch tensors explicitly.
"""

from euler_loading.loaders.gpu.cerema import *  # noqa: F401,F403
