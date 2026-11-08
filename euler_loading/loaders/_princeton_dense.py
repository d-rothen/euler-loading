"""Shared helpers for Princeton DENSE / SeeingThroughFog loaders."""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
from collections import deque
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._writer_utils import (
    mark_stream_supported,
    pixel_grid_transform,
    replace_json,
    to_numpy,
)

logger = logging.getLogger(__name__)

# The 2x2 colour layout from the sensor's top-left pixel. The official
# SeeingThroughFog conversion demosaics with OpenCV's COLOR_BAYER_GB2BGR, and
# OpenCV names patterns after the *second* row, so that is a GRBG sensor:
# G R / B G. Reading it as GBRG swaps red and blue.
_DEFAULT_BAYER_PATTERN = "GRBG"
_DEFAULT_RAW_MAX_VALUE = 4095.0
DEFAULT_CAMERA_FRAME = "cam_stereo_left_optical"
DEFAULT_LIDAR_FRAME = "lidar_hdl64_s3_roof"
LIDAR_COLUMNS = ["x", "y", "z", "intensity", "ring"]
DEFAULT_INTRINSICS_MATRIX = "P"
_INTRINSICS_MATRIX_ALIASES = {"p": "P", "rectified": "P", "k": "K", "raw": "K"}
_CAMERA_INFO_SHAPES = {"K": (3, 3), "P": (3, 4)}


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


@mark_stream_supported
def write_sparse_depth(
    path: Union[str, os.PathLike, BinaryIO],
    value: Any,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Write SeeingThroughFog lidar points as an ``(N, 5)`` float32 ``.bin`` file.

    Columns are ``x, y, z, intensity, ring``, as :func:`load_sparse_depth_array`
    reads them, so float32 points read back unchanged (e.g. after keeping only
    the points inside a crop).
    """
    points = np.ascontiguousarray(to_numpy(value), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != len(LIDAR_COLUMNS):
        raise ValueError(
            "SeeingThroughFog lidar points must have shape "
            f"(N, {len(LIDAR_COLUMNS)}), got {points.shape}"
        )
    if isinstance(path, (str, os.PathLike)):
        parent = os.path.dirname(os.fspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(points.tobytes())
        return
    path.write(points.tobytes())


def _load_json(path: Union[str, BinaryIO]) -> Any:
    if isinstance(path, (str, os.PathLike)):
        with open(path) as f:
            return json.load(f)
    return json.load(path)


def intrinsics_matrix(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    """Return which camera-info matrix the intrinsics come from: ``"P"`` or ``"K"``.

    ``intrinsics_matrix`` is read from the per-file attributes, then from the
    dataset head's meta. ``"P"`` (alias ``"rectified"``, the default) selects
    the rectified projection, ``"K"`` (alias ``"raw"``) the raw camera matrix.
    """
    value = _value_from_context(meta, attributes, ("intrinsics_matrix",))
    if value is None:
        return DEFAULT_INTRINSICS_MATRIX
    selected = _INTRINSICS_MATRIX_ALIASES.get(str(value).strip().lower())
    if selected is None:
        raise ValueError(
            "intrinsics_matrix must be 'P' (rectified) or 'K' (raw), "
            f"got {value!r}"
        )
    return selected


def _camera_info_matrix(
    data: Mapping[str, Any], key: str, name: Any
) -> np.ndarray | None:
    """Return camera-info ``K`` (3x3) or ``P`` (3x4) as float64, or ``None``."""
    raw = data.get(key)
    if raw is None:
        return None
    shape = _CAMERA_INFO_SHAPES[key]
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape == (shape[0] * shape[1],):
        matrix = matrix.reshape(shape)
    if matrix.shape != shape:
        raise ValueError(
            f"Princeton camera {key} in {name!r} must have shape {shape}, "
            f"got {matrix.shape}."
        )
    return matrix


def _check_pinhole(matrix: np.ndarray, what: str) -> None:
    """Raise unless *matrix* is a finite ``[[fx, s, cx], [0, fy, cy], [0, 0, 1]]``."""
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} must be finite")
    if (
        not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9)
        or abs(matrix[1, 0]) > 1e-9
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise ValueError(
            f"{what} must be a pinhole camera matrix [[fx, s, cx], [0, fy, cy], "
            f"[0, 0, 1]] with positive focal lengths, got {matrix.tolist()}"
        )


def load_intrinsics_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Load the left stereo camera's ``(3, 3)`` pinhole matrix from ROS camera-info JSON.

    The default is the rectified projection's ``P[:, :3]``. The 8-bit
    ``cam_stereo_left_lut`` PNGs are rectified to ``P`` -- the official
    conversion remaps them with ``cv2.initUndistortRectifyMap(K, D, R, P)`` --
    so for those images it is an ideal pinhole camera without distortion, and
    it is the matrix the official lidar projection uses. ``P[:, 3]``, zero for
    the left camera and the stereo baseline term for the right one, is not part
    of it.

    ``intrinsics_matrix: "K"`` in the per-file attributes or *meta* returns
    the raw camera matrix ``K`` instead. It belongs to the unrectified 12-bit
    TIFFs and describes them only together with the distortion ``D``.
    """
    data = _load_json(path)
    name = getattr(path, "name", path)
    selected = intrinsics_matrix(meta, attributes)
    matrix = (
        _camera_info_matrix(data, selected, name) if isinstance(data, Mapping) else None
    )
    if matrix is None:
        hint = (
            " Set intrinsics_matrix to 'K' to read the raw camera matrix instead."
            if selected == "P"
            else ""
        )
        raise ValueError(
            f"Princeton camera calibration {name!r} must contain a {selected} array.{hint}"
        )
    pinhole = matrix[:, :3]
    _check_pinhole(pinhole, f"Princeton camera {selected} in {name!r}")
    return pinhole.astype(np.float32)


def _image_size(meta: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return ``(width, height)`` from ``meta``, or ``None`` when neither is given."""
    width, height = meta.get("width"), meta.get("height")
    if width is None and height is None:
        return None
    size = []
    for key, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or value <= 0
            or int(value) != value
        ):
            raise ValueError(
                f"meta[{key!r}] must be a positive integer pixel count, got {value!r}"
            )
        size.append(int(value))
    return size[0], size[1]


def _pixel_transform(transform: np.ndarray, name: str) -> np.ndarray:
    """Return the crop or rescale *transform* represents, or raise.

    *transform* maps the stored matrix's pixel grid onto the new one. Only an
    axis-aligned scale and shift can be applied to ``K`` and ``P`` alike. A
    whole-pixel crop (unit scale, integer shift) is snapped to exact values, so
    the stored float64 entries are shifted without float32 rounding noise.
    """
    exact = pixel_grid_transform(transform)
    if exact is None:
        raise ValueError(
            f"Refusing to update {name!r}: the new intrinsics are not a crop or "
            "rescale of its stored matrix, so its K and P cannot be kept "
            "consistent. Write them to a new file instead."
        )
    return exact


def _update_camera_info(
    document: dict[str, Any],
    matrix: np.ndarray,
    selected: str,
    size: tuple[int, int] | None,
    name: str,
) -> None:
    """Move *document*'s ``K`` and ``P`` onto the pixel grid of *matrix*."""
    stored = {key: _camera_info_matrix(document, key, name) for key in _CAMERA_INFO_SHAPES}
    reference = stored[selected]
    if reference is None:
        raise ValueError(
            f"Refusing to update {name!r}: it has no {selected} matrix to update."
        )
    _check_pinhole(reference[:, :3], f"Princeton camera {selected} in {name!r}")
    transform = _pixel_transform(matrix @ np.linalg.inv(reference[:, :3]), name)
    for key, value in stored.items():
        if value is not None:
            # ``+ 0.0`` turns a -0.0 from the product into 0.0.
            document[key] = ((transform @ value) + 0.0).reshape(-1).tolist()
    if size is not None:
        document["width"], document["height"] = size
    elif not np.array_equal(transform, np.eye(3)) and (
        "width" in document or "height" in document
    ):
        logger.warning(
            "Updated the camera matrices in %r without meta['width'] and "
            "meta['height']; its recorded %sx%s image size may no longer match.",
            name,
            document.get("width"),
            document.get("height"),
        )


def _pinhole_camera_info(
    matrix: np.ndarray, meta: Mapping[str, Any], size: tuple[int, int] | None
) -> dict[str, Any]:
    """Return a camera-info document for an ideal pinhole camera *matrix*."""
    frame = meta.get("camera_frame")
    document: dict[str, Any] = {
        "header": {
            "frame_id": frame if isinstance(frame, str) and frame else DEFAULT_CAMERA_FRAME
        },
        "distortion_model": "plumb_bob",
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K": matrix.reshape(-1).tolist(),
        "R": np.eye(3).reshape(-1).tolist(),
        "P": np.hstack([matrix, np.zeros((3, 1))]).reshape(-1).tolist(),
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }
    if size is not None:
        document["width"], document["height"] = size
    return document


@mark_stream_supported
def write_intrinsics(
    path: Union[str, os.PathLike, BinaryIO],
    value: Any,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Write a pinhole matrix into a ROS camera-info JSON (``calib_cam_stereo_left.json``).

    *value* is what :func:`load_intrinsics_array` returns for the same *meta* --
    the rectified ``P[:, :3]`` by default, the raw ``K`` with
    ``intrinsics_matrix: "K"`` -- so a written file reads back as *value*.

    An existing file is updated rather than replaced. The change from its
    stored matrix to *value* must be a crop or a rescale: an axis-aligned scale
    and shift of the pixel grid. It is applied to both ``K`` and ``P``, so the
    raw and rectified matrices stay consistent, as ROS ``image_geometry`` would
    derive them for a region of interest. ``D``, ``R``, ``roi``, ``header`` and
    all other fields are kept. A whole-pixel crop shifts the stored float64
    values by exact integers. ``meta["width"]`` and ``meta["height"]`` set the
    recorded image size; pass them whenever the matrix no longer matches it.
    The update replaces the file in one step, as the Synscapes camera writers
    do.

    A new file, or a stream target (which cannot be read back), receives an
    ideal pinhole camera: ``K = P[:, :3] = value``, zero distortion and an
    identity ``R``.
    """
    matrix = to_numpy(value).astype(np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {matrix.shape}")
    _check_pinhole(matrix, "intrinsics")
    options = meta if isinstance(meta, Mapping) else {}
    selected = intrinsics_matrix(options, None)
    size = _image_size(options)

    if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
        name = os.fspath(path)
        try:
            document = _load_json(path)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to overwrite {name!r}: it exists but is not readable "
                f"JSON, so its contents cannot be preserved ({exc})."
            ) from exc
        if not isinstance(document, dict):
            raise ValueError(
                f"Refusing to overwrite {name!r}: expected a JSON object to merge "
                f"into, found {type(document).__name__}."
            )
        _update_camera_info(document, matrix, selected, size, name)
    else:
        document = _pinhole_camera_info(matrix, options, size)
    replace_json(path, document, indent=4)


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
