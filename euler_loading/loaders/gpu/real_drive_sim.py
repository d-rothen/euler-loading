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
from typing import Any, BinaryIO, Union

import torch
import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    to_bool_mask,
    to_hwc_rgb,
    to_hw,
    to_numpy,
    to_uint8,
)

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
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB image as a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={"radial_depth": False},
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a Real Drive Sim depth map as a ``(1, H, W)`` float32 tensor in **metres**.

    Real Drive Sim stores depth as float32 values in ``.npz`` files under
    the ``'data'`` key.  Values are already in metres.
    """
    arr = np.load(path)["data"].astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta={"encoding": "single_channel", "sky_class_id": 29},
)
def class_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a class-segmentation mask as a ``(1, H, W)`` long tensor.

    Real Drive Sim encodes class IDs in the first (red) channel of an
    RGBA PNG.  Only the red channel is returned.
    """
    arr = np.array(Image.open(path), dtype=np.int64)[:, :, 0]
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


_SKY_CLASS_ID = 29


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta={"sky_class_id": 29},
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a sky mask as a ``(1, H, W)`` bool tensor.

    Reads the red channel of the segmentation PNG and returns ``True``
    where the class ID equals ``29`` (sky).
    """
    arr = np.array(Image.open(path), dtype=np.uint8)[:, :, 0] == _SKY_CLASS_ID
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------


def _load_json(path: Union[str, BinaryIO]) -> Any:
    """Load JSON from a file path or an in-memory buffer."""
    if isinstance(path, str):
        with open(path) as f:
            return json.load(f)
    return json.load(path)


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


@modality_meta(
    modality_type="calibration",
    dtype="dict",
    hierarchical=True,
    shape="dict",
    file_formats=[".json"],
    meta={"sensors": ["CS_FRONT", "HDL_32E", "HDL_64E"], "keys": ["K", "T", "distortion"]},
)
def calibration(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Load a Real Drive Sim calibration JSON.

    The file contains parallel arrays ``names``, ``intrinsics``, and
    ``extrinsics`` — one entry per sensor.  Returns a dict keyed by sensor
    name, where each value contains:

    - ``"K"`` – ``(3, 3)`` float32 camera-intrinsics matrix.
    - ``"T"`` – ``(4, 4)`` float32 extrinsics matrix (rotation + translation).
    - ``"distortion"`` – ``(8,)`` float32 distortion coefficients
      ``[k1, k2, p1, p2, k3, k4, k5, k6]``.
    """
    data = _load_json(path)

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

@modality_meta(
    modality_type="all_intrinsics",
    dtype="dict",
    hierarchical=True,
    shape="dict",
    file_formats=[".json"],
)
def all_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Load only the intrinsics from a Real Drive Sim calibration JSON."""
    data = _load_json(path)
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

@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"sensor": "CS_FRONT"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load the intrinsics for a specific sensor from a Real Drive Sim calibration JSON."""
    all_intrinsics_data = all_intrinsics(path)
    return all_intrinsics_data["CS_FRONT"]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_rgb(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB tensor/array to PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    Image.fromarray(arr, mode="RGB").save(path)


def write_depth(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a depth map to a Real Drive Sim ``.npz`` file under key ``data``."""
    ensure_parent(path)
    depth = to_hw(value, name="depth").astype(np.float32)
    np.savez_compressed(path, data=depth)


def write_class_segmentation(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write class IDs as an RGBA PNG with IDs stored in the red channel."""
    ensure_parent(path)
    class_ids = np.clip(to_hw(value, name="class_segmentation"), 0, 255).astype(np.uint8)
    rgba = np.zeros(class_ids.shape + (4,), dtype=np.uint8)
    rgba[:, :, 0] = class_ids
    rgba[:, :, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def write_sky_mask(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a class-ID PNG compatible with :func:`sky_mask`."""
    sky_class_id = int((meta or {}).get("sky_class_id", _SKY_CLASS_ID))
    mask = to_bool_mask(value)
    class_ids = np.zeros(mask.shape, dtype=np.uint8)
    class_ids[mask] = np.uint8(sky_class_id)
    write_class_segmentation(path, class_ids, meta=meta)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to quaternion ``(qw, qx, qy, qz)``."""
    trace = float(np.trace(R))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    quat /= norm
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _intrinsics_from_matrix(K: np.ndarray, distortion: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "skew": float(K[0, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k1": float(distortion[0]),
        "k2": float(distortion[1]),
        "p1": float(distortion[2]),
        "p2": float(distortion[3]),
        "k3": float(distortion[4]),
        "k4": float(distortion[5]),
        "k5": float(distortion[6]),
        "k6": float(distortion[7]),
    }


def _extrinsics_from_matrix(T: np.ndarray) -> dict[str, dict[str, float]]:
    qw, qx, qy, qz = _rotation_matrix_to_quaternion(T[:3, :3])
    return {
        "translation": {
            "x": float(T[0, 3]),
            "y": float(T[1, 3]),
            "z": float(T[2, 3]),
        },
        "rotation": {
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
        },
    }


def write_calibration(
    path: str,
    value: Any,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write Real Drive Sim calibration JSON from parsed calibration data."""
    ensure_parent(path)
    calibration_data = value
    if not isinstance(calibration_data, dict):
        raise ValueError("calibration value must be dict[sensor_name, sensor_data]")

    names: list[str] = []
    intrinsics: list[dict[str, float]] = []
    extrinsics: list[dict[str, dict[str, float]]] = []

    for sensor_name, sensor in calibration_data.items():
        if not isinstance(sensor, dict):
            raise ValueError(
                f"calibration[{sensor_name!r}] must be a dict with keys K, T, distortion"
            )
        K = to_numpy(sensor["K"]).astype(np.float32)
        T = to_numpy(sensor["T"]).astype(np.float32)
        distortion = to_numpy(sensor["distortion"]).astype(np.float32).reshape(-1)

        if K.shape != (3, 3):
            raise ValueError(f"calibration[{sensor_name!r}]['K'] must be shape (3, 3)")
        if T.shape != (4, 4):
            raise ValueError(f"calibration[{sensor_name!r}]['T'] must be shape (4, 4)")
        if distortion.size != 8:
            raise ValueError(
                f"calibration[{sensor_name!r}]['distortion'] must contain 8 coefficients"
            )

        names.append(sensor_name)
        intrinsics.append(_intrinsics_from_matrix(K, distortion))
        extrinsics.append(_extrinsics_from_matrix(T))

    payload = {
        "names": names,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_all_intrinsics(
    path: str,
    value: Any,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write intrinsics-only mapping to Real Drive Sim calibration JSON."""
    ensure_parent(path)
    intrinsics_map = value
    if not isinstance(intrinsics_map, dict):
        raise ValueError("all_intrinsics value must be dict[sensor_name, K]")

    names: list[str] = []
    intrinsics: list[dict[str, float]] = []
    extrinsics: list[dict[str, dict[str, float]]] = []
    zero_distortion = np.zeros(8, dtype=np.float32)

    for sensor_name, K_value in intrinsics_map.items():
        K = to_numpy(K_value).astype(np.float32)
        if K.shape != (3, 3):
            raise ValueError(
                f"all_intrinsics[{sensor_name!r}] must have shape (3, 3)"
            )
        names.append(sensor_name)
        intrinsics.append(_intrinsics_from_matrix(K, zero_distortion))
        extrinsics.append(
            {
                "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
            }
        )

    payload = {
        "names": names,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_intrinsics(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a single-sensor intrinsics matrix as calibration JSON."""
    sensor_name = str((meta or {}).get("sensor", "CS_FRONT"))
    write_all_intrinsics(path, {sensor_name: value}, meta=meta)
