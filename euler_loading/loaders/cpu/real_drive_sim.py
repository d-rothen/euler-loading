"""CPU-oriented loader functions for the Real Drive Sim dataset.

Each function follows the ``Callable[[str], Any]`` signature expected by
:class:`~euler_loading.Modality`.  All loaders return **numpy ndarrays**
directly from the loaded files, suitable for CPU-based processing.

Return types
------------
- **rgb** – ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** – ``np.ndarray`` of shape ``(H, W)`` float32 in metres.
- **class_segmentation** – ``np.ndarray`` of shape ``(H, W)`` int64
  (single-channel class IDs).
- **sky_mask** – ``np.ndarray`` of shape ``(H, W)`` bool
  (``True`` where class ID == 29).

Usage::

    from euler_loading.loaders.cpu import real_drive_sim
    from euler_loading import Modality

    Modality("/data/real_drive_sim/rgb",   loader=real_drive_sim.rgb)
    Modality("/data/real_drive_sim/depth", loader=real_drive_sim.depth)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta

# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB image as an ``(H, W, 3)`` float32 array in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="dense_depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={"radial_depth": False},
)
def depth(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a Real Drive Sim depth map as an ``(H, W)`` float32 array in **metres**.

    Real Drive Sim stores depth as float32 values in ``.npz`` files under
    the ``'data'`` key.  Values are already in metres.
    """
    return np.load(path)["data"].astype(np.float32)


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={"encoding": "single_channel", "sky_class_id": 29},
)
def class_segmentation(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a class-segmentation mask as an ``(H, W)`` int64 array.

    Real Drive Sim encodes class IDs in the first (red) channel of an
    RGBA PNG.  Only the red channel is returned.
    """
    return np.array(Image.open(path), dtype=np.int64)[:, :, 0]


_SKY_CLASS_ID = 29


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"sky_class_id": 29},
)
def sky_mask(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a sky mask as an ``(H, W)`` bool array.

    Reads the red channel of the segmentation PNG and returns ``True``
    where the class ID equals ``29`` (sky).
    """
    return np.array(Image.open(path), dtype=np.uint8)[:, :, 0] == _SKY_CLASS_ID
