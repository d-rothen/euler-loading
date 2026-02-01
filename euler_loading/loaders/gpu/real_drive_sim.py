"""GPU-oriented loader functions for the Real Drive Sim dataset.

Each function follows the ``Callable[[str], Any]`` signature expected by
:class:`~euler_loading.Modality`.  All loaders return **torch tensors** that
can be directly used for GPU-based training.

Return types
------------
- **rgb** – ``torch.FloatTensor`` of shape ``(3, H, W)`` in ``[0, 1]``.
- **depth** – ``torch.FloatTensor`` of shape ``(1, H, W)`` in metres.
- **class_segmentation** – ``torch.LongTensor`` of shape ``(1, H, W)``
  (single-channel class IDs).

Usage::

    from euler_loading.loaders.gpu import real_drive_sim
    from euler_loading import Modality

    Modality("/data/real_drive_sim/rgb",   loader=real_drive_sim.rgb)
    Modality("/data/real_drive_sim/depth", loader=real_drive_sim.depth)
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


def rgb(path: str) -> torch.Tensor:
    """Load an RGB image as a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def depth(path: str) -> torch.Tensor:
    """Load a Real Drive Sim depth map as a ``(1, H, W)`` float32 tensor in **metres**.

    Real Drive Sim stores depth as float32 values in ``.npz`` files under
    the ``'data'`` key.  Values are already in metres.
    """
    arr = np.load(path)["data"].astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def class_segmentation(path: str) -> torch.Tensor:
    """Load a class-segmentation mask as a ``(1, H, W)`` long tensor.

    Real Drive Sim encodes class IDs in the first (red) channel of an
    RGBA PNG.  Only the red channel is returned.
    """
    arr = np.array(Image.open(path), dtype=np.int64)[:, :, 0]
    return torch.from_numpy(arr).unsqueeze(0).contiguous()
