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
- **sky_mask** – ``torch.BoolTensor`` of shape ``(1, H, W)``
  (``True`` where class ID == 29).
- **calibration** – ``dict[str, dict[str, torch.Tensor]]`` keyed by sensor
  name.  Each sensor dict contains ``"K"`` (3×3), ``"T"`` (4×4), and
  ``"distortion"`` (8,).

Usage::

    from euler_loading.loaders.gpu import real_drive_sim
    from euler_loading import Modality

    Modality("/data/real_drive_sim/rgb",   loader=real_drive_sim.rgb)
    Modality("/data/real_drive_sim/depth", loader=real_drive_sim.depth)
"""

from __future__ import annotations

import json
from typing import Any

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


def rgb(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB image as a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def depth(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a Real Drive Sim depth map as a ``(1, H, W)`` float32 tensor in **metres**.

    Real Drive Sim stores depth as float32 values in ``.npz`` files under
    the ``'data'`` key.  Values are already in metres.
    """
    arr = np.load(path)["data"].astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def class_segmentation(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a class-segmentation mask as a ``(1, H, W)`` long tensor.

    Real Drive Sim encodes class IDs in the first (red) channel of an
    RGBA PNG.  Only the red channel is returned.
    """
    arr = np.array(Image.open(path), dtype=np.int64)[:, :, 0]
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


_SKY_CLASS_ID = 29


def sky_mask(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a sky mask as a ``(1, H, W)`` bool tensor.

    Reads the red channel of the segmentation PNG and returns ``True``
    where the class ID equals ``29`` (sky).
    """
    arr = np.array(Image.open(path), dtype=np.uint8)[:, :, 0] == _SKY_CLASS_ID
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> torch.Tensor:
    """Convert a quaternion ``(qw, qx, qy, qz)`` to a ``(3, 3)`` rotation matrix."""
    return torch.tensor(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=torch.float32,
    )


def calibration(path: str, meta: dict[str, Any] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Load a Real Drive Sim calibration JSON.

    The file contains parallel arrays ``names``, ``intrinsics``, and
    ``extrinsics`` — one entry per sensor.  Returns a dict keyed by sensor
    name, where each value contains:

    - ``"K"`` – ``(3, 3)`` float32 camera-intrinsics matrix.
    - ``"T"`` – ``(4, 4)`` float32 extrinsics matrix (rotation + translation).
    - ``"distortion"`` – ``(8,)`` float32 distortion coefficients
      ``[k1, k2, p1, p2, k3, k4, k5, k6]``.
    """
    with open(path) as f:
        data = json.load(f)

    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, intr, extr in zip(data["names"], data["intrinsics"], data["extrinsics"]):
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        s = intr["skew"]
        K = torch.tensor(
            [[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        rot = extr["rotation"]
        R = _quat_to_rotation_matrix(rot["qw"], rot["qx"], rot["qy"], rot["qz"])
        t = torch.tensor(
            [extr["translation"]["x"], extr["translation"]["y"], extr["translation"]["z"]],
            dtype=torch.float32,
        )
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        distortion = torch.tensor(
            [intr["k1"], intr["k2"], intr["p1"], intr["p2"],
             intr["k3"], intr["k4"], intr["k5"], intr["k6"]],
            dtype=torch.float32,
        )

        result[name] = {"K": K, "T": T, "distortion": distortion}
    return result

def all_intrinsics(path: str, meta: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Load only the intrinsics from a Real Drive Sim calibration JSON."""
    with open(path) as f:
        data = json.load(f)
    result: dict[str, torch.Tensor] = {}
    
    for name, intr in zip(data["names"], data["intrinsics"]):
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        s = intr["skew"]
        K = torch.tensor(
            [[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        result[name] = K

    return result

def read_intrinsics(path: str, meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load the intrinsics for a specific sensor from a Real Drive Sim calibration JSON."""
    all_intrinsics_data = all_intrinsics(path)
    return all_intrinsics_data["CS_FRONT"]