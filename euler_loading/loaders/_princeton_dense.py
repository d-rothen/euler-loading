"""Shared helpers for Princeton DENSE / SeeingThroughFog loaders."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

_DEFAULT_BAYER_PATTERN = "GBRG"
_DEFAULT_RAW_MAX_VALUE = 4095.0
DEFAULT_CAMERA_FRAME = "cam_stereo_left_optical"
DEFAULT_LIDAR_FRAME = "lidar_hdl64_s3_roof"
LIDAR_COLUMNS = ["x", "y", "z", "intensity", "ring"]


def _value_from_context(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> Any:
    for source in (attributes, meta):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            if key in source:
                return source[key]
    return None


def _rgb_scale(
    raw: np.ndarray,
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> float:
    value = _value_from_context(
        meta,
        attributes,
        ("rgb_max_value", "raw_max_value", "white_level"),
    )
    if value is not None:
        return float(value)

    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8:
            return 255.0
        if raw.size and float(raw.max()) <= _DEFAULT_RAW_MAX_VALUE:
            return _DEFAULT_RAW_MAX_VALUE
        return float(np.iinfo(raw.dtype).max)

    return 1.0


def _bayer_pattern(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    value = _value_from_context(meta, attributes, ("bayer_pattern", "cfa_pattern"))
    if value is None:
        return _DEFAULT_BAYER_PATTERN
    pattern = str(value).upper()
    if len(pattern) != 4 or any(channel not in "RGB" for channel in pattern):
        raise ValueError("bayer_pattern must be a 4-character pattern using R, G, and B")
    return pattern


def _interpolate_channel(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values_pad = np.pad(values, 1, mode="edge")
    mask_pad = np.pad(mask.astype(np.float32), 1, mode="edge")

    sums = np.zeros_like(values, dtype=np.float32)
    counts = np.zeros_like(values, dtype=np.float32)
    for y_offset in range(3):
        for x_offset in range(3):
            y = slice(y_offset, y_offset + values.shape[0])
            x = slice(x_offset, x_offset + values.shape[1])
            sums += values_pad[y, x]
            counts += mask_pad[y, x]

    interpolated = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float32),
        where=counts > 0,
    )
    return np.where(mask, values, interpolated)


def _demosaic_bayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    raw_f32 = raw.astype(np.float32, copy=False)
    channel_values = {
        "R": np.zeros(raw.shape, dtype=np.float32),
        "G": np.zeros(raw.shape, dtype=np.float32),
        "B": np.zeros(raw.shape, dtype=np.float32),
    }
    channel_masks = {
        "R": np.zeros(raw.shape, dtype=bool),
        "G": np.zeros(raw.shape, dtype=bool),
        "B": np.zeros(raw.shape, dtype=bool),
    }

    positions = (
        (slice(0, None, 2), slice(0, None, 2), pattern[0]),
        (slice(0, None, 2), slice(1, None, 2), pattern[1]),
        (slice(1, None, 2), slice(0, None, 2), pattern[2]),
        (slice(1, None, 2), slice(1, None, 2), pattern[3]),
    )
    for row_slice, col_slice, channel in positions:
        channel_values[channel][row_slice, col_slice] = raw_f32[row_slice, col_slice]
        channel_masks[channel][row_slice, col_slice] = True

    channels = [
        _interpolate_channel(channel_values[channel], channel_masks[channel])
        for channel in ("R", "G", "B")
    ]
    return np.stack(channels, axis=-1)


def load_rgb_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog RGB TIFF as an ``(H, W, 3)`` float32 array."""
    with Image.open(path) as image:
        raw = np.array(image)

    if raw.ndim == 2:
        rgb = _demosaic_bayer(raw, _bayer_pattern(meta, attributes))
        scale = _rgb_scale(raw, meta, attributes)
    elif raw.ndim == 3 and raw.shape[-1] >= 3:
        rgb = raw[..., :3].astype(np.float32, copy=False)
        scale = _rgb_scale(raw, meta, attributes)
    else:
        name = getattr(path, "name", path)
        raise ValueError(f"Unsupported Princeton RGB image shape for {name!r}: {raw.shape}")

    if scale <= 0:
        raise ValueError(f"RGB scale must be positive, got {scale!r}")
    return np.clip(rgb / scale, 0.0, 1.0).astype(np.float32, copy=False)


def load_sparse_depth_array(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a SeeingThroughFog lidar ``.bin`` file as ``(N, 5)`` float32."""
    if isinstance(path, (str, os.PathLike)):
        values = np.fromfile(path, dtype=np.float32)
    else:
        values = np.frombuffer(path.read(), dtype=np.float32).copy()

    if values.size % 5 != 0:
        name = getattr(path, "name", path)
        raise ValueError(
            f"Princeton sparse-depth file {name!r} contains {values.size} "
            "float32 values, which is not divisible by 5."
        )
    return values.reshape((-1, 5))


def _load_json(path: Union[str, BinaryIO]) -> Any:
    if isinstance(path, (str, os.PathLike)):
        with open(path) as f:
            return json.load(f)
    return json.load(path)


def load_intrinsics_array(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load the left stereo camera intrinsics matrix from ROS camera-info JSON."""
    data = _load_json(path)
    if not isinstance(data, Mapping) or "K" not in data:
        name = getattr(path, "name", path)
        raise ValueError(f"Princeton camera calibration {name!r} must contain a K array.")

    K = np.asarray(data["K"], dtype=np.float32)
    if K.shape == (9,):
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError(f"Princeton camera K must have shape (3, 3), got {K.shape}.")
    return K


def _select_frame(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    keys: tuple[str, ...],
    default: str,
) -> str:
    value = _value_from_context(meta, attributes, keys)
    if value is None:
        return default
    frame = str(value)
    aliases = {
        "camera": DEFAULT_CAMERA_FRAME,
        "cam_stereo_left": DEFAULT_CAMERA_FRAME,
        "left": DEFAULT_CAMERA_FRAME,
        "rgb": DEFAULT_CAMERA_FRAME,
        "lidar": DEFAULT_LIDAR_FRAME,
        "hdl64": DEFAULT_LIDAR_FRAME,
        "hdl64_s3": DEFAULT_LIDAR_FRAME,
        "lidar_hdl64": DEFAULT_LIDAR_FRAME,
    }
    return aliases.get(frame, frame)


def _matrix_from_transform(transform: Mapping[str, Any]) -> np.ndarray:
    translation = transform.get("translation")
    rotation = transform.get("rotation")
    if not isinstance(translation, Mapping) or not isinstance(rotation, Mapping):
        raise ValueError("TF transform entries must contain translation and rotation objects.")

    x = float(rotation.get("x", 0.0))
    y = float(rotation.get("y", 0.0))
    z = float(rotation.get("z", 0.0))
    w = float(rotation.get("w", 1.0))
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm <= 0.0:
        raise ValueError("TF quaternion must have non-zero norm.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = [
        float(translation.get("x", 0.0)),
        float(translation.get("y", 0.0)),
        float(translation.get("z", 0.0)),
    ]
    return matrix


def _tf_edges(data: Any) -> dict[str, list[tuple[str, np.ndarray]]]:
    if not isinstance(data, list):
        raise ValueError("Princeton TF calibration must be a JSON list of transforms.")

    edges: dict[str, list[tuple[str, np.ndarray]]] = {}
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        header = entry.get("header")
        child = entry.get("child_frame_id")
        transform = entry.get("transform")
        if not isinstance(header, Mapping) or not isinstance(child, str):
            continue
        parent = header.get("frame_id")
        if not isinstance(parent, str) or not isinstance(transform, Mapping):
            continue

        # ROS TransformStamped stores the child pose in the parent frame.
        # This matrix maps coordinates from child frame to parent frame.
        parent_from_child = _matrix_from_transform(transform)
        child_from_parent = np.linalg.inv(parent_from_child)
        edges.setdefault(child, []).append((parent, parent_from_child))
        edges.setdefault(parent, []).append((child, child_from_parent))
    return edges


def _compose_tf(
    edges: Mapping[str, list[tuple[str, np.ndarray]]],
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    queue: deque[tuple[str, np.ndarray]] = deque(
        [(source_frame, np.eye(4, dtype=np.float64))]
    )
    seen = {source_frame}

    while queue:
        frame, frame_from_source = queue.popleft()
        if frame == target_frame:
            return frame_from_source.astype(np.float32)
        for next_frame, next_from_frame in edges.get(frame, []):
            if next_frame in seen:
                continue
            seen.add(next_frame)
            queue.append((next_frame, next_from_frame @ frame_from_source))

    available = ", ".join(sorted(edges.keys()))
    raise KeyError(
        f"No TF path from {source_frame!r} to {target_frame!r}. "
        f"Available frames: {available}"
    )


def load_extrinsics_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Load a 4x4 transform mapping HDL64 lidar points into the camera frame."""
    source_frame = _select_frame(
        meta,
        attributes,
        ("source_frame", "source_sensor", "from_frame", "from_sensor", "source"),
        DEFAULT_LIDAR_FRAME,
    )
    target_frame = _select_frame(
        meta,
        attributes,
        ("target_frame", "target_sensor", "to_frame", "to_sensor", "target", "camera"),
        DEFAULT_CAMERA_FRAME,
    )
    return _compose_tf(_tf_edges(_load_json(path)), source_frame, target_frame)
