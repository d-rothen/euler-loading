"""nuScenes loaders, defaulting to the GPU-oriented tensor variant.

Import from :mod:`euler_loading.loaders.cpu.nuscenes` for NumPy arrays or
:mod:`euler_loading.loaders.gpu.nuscenes` for torch tensors explicitly.
"""

from euler_loading.loaders.gpu.nuscenes import *  # noqa: F401,F403
