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
- **calibration** – ``dict[str, dict[str, np.ndarray]]`` keyed by sensor
  name. Each sensor dict contains ``"K"`` (3x3), ``"T"`` (4x4), and
  ``"distortion"`` (8,).

Usage::

    from euler_loading.loaders.cpu import real_drive_sim
    from euler_loading import Modality

    Modality("/data/real_drive_sim/rgb",   loader=real_drive_sim.rgb)
    Modality("/data/real_drive_sim/depth", loader=real_drive_sim.depth)
"""

from __future__ import annotations

import json
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    mark_stream_supported,
    save_image,
    to_bool_mask,
    to_hwc_rgb,
    to_hw,
    to_uint8,
)

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
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB image as an ``(H, W, 3)`` float32 array in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={"radial_depth": False},
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
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
def class_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
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
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a sky mask as an ``(H, W)`` bool array.

    Reads the red channel of the segmentation PNG and returns ``True``
    where the class ID equals ``29`` (sky).
    """
    return np.array(Image.open(path), dtype=np.uint8)[:, :, 0] == _SKY_CLASS_ID


# ---------------------------------------------------------------------------
# Calibration loaders
# ---------------------------------------------------------------------------


def _load_json(path: Union[str, BinaryIO]) -> Any:
    """Load JSON from a file path or an in-memory buffer."""
    if isinstance(path, str):
        with open(path) as f:
            return json.load(f)
    return json.load(path)


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a quaternion ``(qw, qx, qy, qz)`` to a ``(3, 3)`` rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


@modality_meta(
    modality_type="calibration",
    dtype="dict",
    hierarchical=True,
    shape="dict",
    file_formats=[".json"],
    meta={"sensors": ["CS_FRONT", "HDL_32E", "HDL_64E"], "keys": ["K", "T", "distortion"]},
)
def calibration(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load a Real Drive Sim calibration JSON.

    The file contains parallel arrays ``names``, ``intrinsics``, and
    ``extrinsics`` - one entry per sensor. Returns a dict keyed by sensor
    name, where each value contains:

    - ``"K"`` - ``(3, 3)`` float32 camera-intrinsics matrix.
    - ``"T"`` - ``(4, 4)`` float32 extrinsics matrix (rotation + translation).
    - ``"distortion"`` - ``(8,)`` float32 distortion coefficients
      ``[k1, k2, p1, p2, k3, k4, k5, k6]``.
    """
    data = _load_json(path)

    result: dict[str, dict[str, np.ndarray]] = {}
    for name, intr, extr in zip(data["names"], data["intrinsics"], data["extrinsics"]):
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        s = intr["skew"]
        K = np.array(
            [[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        rot = extr["rotation"]
        R = _quat_to_rotation_matrix(rot["qw"], rot["qx"], rot["qy"], rot["qz"])
        t = np.array(
            [extr["translation"]["x"], extr["translation"]["y"], extr["translation"]["z"]],
            dtype=np.float32,
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        distortion = np.array(
            [
                intr["k1"],
                intr["k2"],
                intr["p1"],
                intr["p2"],
                intr["k3"],
                intr["k4"],
                intr["k5"],
                intr["k6"],
            ],
            dtype=np.float32,
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
def all_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Load only the intrinsics from a Real Drive Sim calibration JSON."""
    data = _load_json(path)
    result: dict[str, np.ndarray] = {}

    for name, intr in zip(data["names"], data["intrinsics"]):
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        s = intr["skew"]
        K = np.array(
            [[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
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
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load the intrinsics for a specific sensor from a Real Drive Sim calibration JSON."""
    sensor = (attributes or {}).get("sensor") or (meta or {}).get("sensor") or "CS_FRONT"
    intrinsics = all_intrinsics(path)
    if sensor not in intrinsics:
        raise KeyError(
            f"Real Drive Sim calibration has no intrinsics for sensor {sensor!r}. "
            f"Available sensors: {sorted(intrinsics)}"
        )
    return intrinsics[sensor]


DEFAULT_CAMERA_SENSOR = "CS_FRONT"
DEFAULT_LIDAR_SENSOR = "HDL_64E"

_SENSOR_ALIASES = {
    "camera": DEFAULT_CAMERA_SENSOR,
    "cam": DEFAULT_CAMERA_SENSOR,
    "rgb": DEFAULT_CAMERA_SENSOR,
    "front": DEFAULT_CAMERA_SENSOR,
    "lidar": DEFAULT_LIDAR_SENSOR,
    "hdl64": DEFAULT_LIDAR_SENSOR,
    "hdl_64": DEFAULT_LIDAR_SENSOR,
    "hdl32": "HDL_32E",
    "hdl_32": "HDL_32E",
}


def _select_sensor(
    meta: dict[str, Any] | None,
    attributes: dict[str, Any] | None,
    keys: tuple[str, ...],
    default: str,
) -> str:
    """Resolve a sensor name from ``attributes`` (highest priority) or ``meta``."""
    for source in (attributes, meta):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value:
                return _SENSOR_ALIASES.get(value.lower(), value)
    return default


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "dataset": "RealDriveSim",
        "default_source_frame": DEFAULT_LIDAR_SENSOR,
        "default_target_frame": DEFAULT_CAMERA_SENSOR,
        "transform_direction": "source_frame_to_target_frame",
    },
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a directed ``source -> target`` sensor transform.

    Real Drive Sim stores one pose per sensor, each mapping that sensor's
    coordinates into the shared vehicle frame. There is therefore no stored
    sensor-to-sensor matrix; it is composed here::

        X_target = inv(T_target) @ T_source @ X_source

    The default is ``HDL_64E -> CS_FRONT``, suitable for projecting Real Drive
    Sim lidar/sparse-depth points into the front camera. Override it with
    ``meta`` or per-file ``attributes`` using ``source_frame`` /
    ``source_sensor`` and ``target_frame`` / ``target_sensor``.

    Returns a ``(4, 4)`` float32 homogeneous matrix, row-major.
    """
    sensors = calibration(path)

    source = _select_sensor(
        meta,
        attributes,
        ("source_frame", "source_sensor", "from_frame", "from_sensor", "source"),
        DEFAULT_LIDAR_SENSOR,
    )
    target = _select_sensor(
        meta,
        attributes,
        ("target_frame", "target_sensor", "to_frame", "to_sensor", "target", "camera"),
        DEFAULT_CAMERA_SENSOR,
    )

    for role, sensor in (("source", source), ("target", target)):
        if sensor not in sensors:
            raise KeyError(
                f"Real Drive Sim calibration has no {role} sensor {sensor!r}. "
                f"Available sensors: {sorted(sensors)}"
            )

    T_source = np.asarray(sensors[source]["T"], dtype=np.float64)
    T_target = np.asarray(sensors[target]["T"], dtype=np.float64)
    return (np.linalg.inv(T_target) @ T_source).astype(np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB array/tensor to PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a depth map to a Real Drive Sim ``.npz`` file under key ``data``."""
    ensure_parent(path)
    depth = to_hw(value, name="depth").astype(np.float32)
    np.savez_compressed(path, data=depth)


@mark_stream_supported
def write_class_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write class IDs as an RGBA PNG with IDs stored in the red channel."""
    ensure_parent(path)
    class_ids = np.clip(to_hw(value, name="class_segmentation"), 0, 255).astype(np.uint8)
    rgba = np.zeros(class_ids.shape + (4,), dtype=np.uint8)
    rgba[:, :, 0] = class_ids
    rgba[:, :, 3] = 255
    save_image(path, Image.fromarray(rgba, mode="RGBA"), format="PNG")


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a class-ID PNG compatible with :func:`sky_mask`."""
    sky_class_id = int((meta or {}).get("sky_class_id", _SKY_CLASS_ID))
    mask = to_bool_mask(value)
    class_ids = np.zeros(mask.shape, dtype=np.uint8)
    class_ids[mask] = np.uint8(sky_class_id)
    write_class_segmentation(path, class_ids, meta=meta)
