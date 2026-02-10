"""GPU-oriented loader functions for the Virtual KITTI 2 dataset.

Each function follows the ``Callable[[str], Any]`` signature expected by
:class:`~euler_loading.Modality`.  All loaders return **torch tensors** that
can be directly used for GPU-based training.

Return types
------------
- **rgb** – ``torch.FloatTensor`` of shape ``(3, H, W)`` in ``[0, 1]``.
- **depth** – ``torch.FloatTensor`` of shape ``(1, H, W)`` in metres.
- **class_segmentation** – ``torch.LongTensor`` of shape ``(3, H, W)``
  (RGB-encoded class labels).
- **instance_segmentation** – ``torch.LongTensor`` of shape ``(3, H, W)``
  (RGB-encoded instance labels).
- **sky_mask** – ``torch.BoolTensor`` of shape ``(1, H, W)``
  (``True`` where the pixel is sky, i.e. RGB ``(90, 200, 255)``).
- **scene_flow** – ``torch.FloatTensor`` of shape ``(3, H, W)`` in ``[0, 1]``.
- **read_intrinsics** – ``torch.FloatTensor`` of shape ``(3, 3)``
  (the camera intrinsic matrix *K*).
- **read_extrinsics** – ``torch.FloatTensor`` parsed from the text file.

Usage::

    from euler_loading.loaders.gpu import vkitti2
    from euler_loading import Modality

    Modality("/data/vkitti2/rgb",   loader=vkitti2.rgb)
    Modality("/data/vkitti2/depth", loader=vkitti2.depth)
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta

# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB image as a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".png"],
    output_unit="meters",
    meta={"raw_range": [0, 65535], "radial_depth": False, "scale_to_meters": 0.01},
)
def depth(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a VKITTI2 depth map as a ``(1, H, W)`` float32 tensor in **metres**.
    VKITTI2 stores depth as 16-bit PNG where each pixel value represents
    depth in centimetres.  This loader converts to metres
    (``value / 100``).
    """
    arr = np.array(Image.open(path), dtype=np.float32) / 100.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="CHW",
    file_formats=[".png"],
    meta={"encoding": "rgb"},
)
def class_segmentation(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB-encoded class-segmentation mask as a ``(3, H, W)`` long tensor."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.int64)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="instance_segmentation",
    dtype="int64",
    shape="CHW",
    file_formats=[".png"],
    meta={"encoding": "rgb"},
)
def instance_segmentation(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB-encoded instance-segmentation mask as a ``(3, H, W)`` long tensor."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.int64)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


_SKY_COLOR = (90, 200, 255)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta={"sky_color": [90, 200, 255]},
)
def sky_mask(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a sky mask as a ``(1, H, W)`` bool tensor.

    Reads the RGB class-segmentation PNG and returns ``True`` where the
    pixel colour equals ``(90, 200, 255)`` (sky class in VKITTI2).
    """
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    mask = (
        (arr[:, :, 0] == _SKY_COLOR[0])
        & (arr[:, :, 1] == _SKY_COLOR[1])
        & (arr[:, :, 2] == _SKY_COLOR[2])
    )
    return torch.from_numpy(mask).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="scene_flow",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def scene_flow(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an optical / scene-flow map as a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Text ground-truth readers (calibration)
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".txt"],
)
def read_intrinsics(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Parse a VKITTI2 intrinsics text file into a ``(3, 3)`` float32 *K* matrix.

    The file has the header ``frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]``.
    Intrinsics are constant across frames in VKITTI2, so only the first row
    is used.  Returns the camera matrix::

        K = [[fx,  s, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fx = float(data[0][2])
    fy = float(data[0][3])
    cx = float(data[0][4])
    cy = float(data[0][5])
    return torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


@modality_meta(
    modality_type="extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="Nx1",
    file_formats=[".txt"],
)
def read_extrinsics(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Parse a VKITTI2 extrinsics text file into a float32 tensor."""
    return torch.from_numpy(np.loadtxt(path).astype(np.float32))
