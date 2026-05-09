"""CPU-oriented loader functions for the MUSES dataset.

MUSES stores RGB camera frames, reference frames, semantic labels, panoptic
labels, and lidar point clouds in separate modality folders.  These loaders
return NumPy arrays for CPU-based processing.

Return types
------------
- **rgb** / **reference_rgb** -- ``np.ndarray`` of shape ``(H, W, 3)``
  float32 in ``[0, 1]``.
- **semantic_segmentation** -- ``np.ndarray`` of shape ``(H, W)`` int64
  containing raw Cityscapes ``labelIds`` or ``labelTrainIds``.
- **semantic_segmentation_color** -- ``np.ndarray`` of shape ``(H, W, 3)``
  uint8 containing Cityscapes RGB colours.
- **sky_mask** -- ``np.ndarray`` of shape ``(H, W)`` bool.
- **panoptic_segmentation** -- ``np.ndarray`` of shape ``(H, W)`` int64
  containing COCO-style panoptic segment IDs decoded from RGB.
- **lidar_point_cloud** / **sparse_depth** -- ``np.ndarray`` of shape
  ``(N, 6)`` float64 with columns ``x, y, z, intensity, ring, timestamp``.
- **read_intrinsics** -- ``np.ndarray`` of shape ``(3, 3)`` float32 parsed
  from ``calib.json``.
- **read_extrinsics** -- ``np.ndarray`` of shape ``(4, 4)`` float32 parsed
  from ``calib.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta

_LIDAR_COLUMNS = ["x", "y", "z", "intensity", "ring", "timestamp"]
_SKY_LABEL_ID = 23
_SKY_TRAIN_ID = 10


def _load_image_rgb(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load an image file as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_single_channel_labels(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a single-channel semantic label image as int64."""
    with Image.open(path) as image:
        arr = np.array(image, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(
            "MUSES semantic_segmentation expects a single-channel labelIds "
            "or labelTrainIds PNG. Use semantic_segmentation_color for "
            "labelColor PNG files."
        )
    return arr


def _select_sky_class_id(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> int:
    for source in (attributes, meta):
        if not isinstance(source, Mapping):
            continue
        value = source.get("sky_class_id")
        if isinstance(value, (int, np.integer)):
            return int(value)

    name = str(getattr(path, "name", path)).lower()
    if "labelids" in name:
        return _SKY_LABEL_ID
    return _SKY_TRAIN_ID


def _select_sky_class_color(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> np.ndarray | None:
    for source in (attributes, meta):
        if not isinstance(source, Mapping):
            continue
        value = source.get("sky_class")
        if value is None:
            value = source.get("sky_mask")
        if value is None:
            continue

        color = np.asarray(value, dtype=np.uint8)
        if color.shape != (3,):
            raise ValueError("MUSES sky_class must be a 3-element RGB value")
        return color
    return None


def _load_sky_mask_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> np.ndarray:
    color = _select_sky_class_color(meta, attributes)
    if color is not None:
        with Image.open(path) as image:
            arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return np.all(arr == color, axis=-1)

    labels = _load_single_channel_labels(path)
    return labels == _select_sky_class_id(path, meta, attributes)


def _load_rgb_uint8(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load an RGB image without normalisation."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _decode_panoptic_rgb(rgb: np.ndarray) -> np.ndarray:
    """Decode COCO-style RGB panoptic IDs as ``R + 256*G + 256^2*B``."""
    values = rgb.astype(np.int64)
    return values[:, :, 0] + 256 * values[:, :, 1] + 65536 * values[:, :, 2]


def _load_lidar_array(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a MUSES lidar ``.bin`` file as an ``(N, 6)`` float64 array."""
    if isinstance(path, (str, os.PathLike)):
        arr = np.fromfile(path, dtype=np.float64)
    else:
        arr = np.frombuffer(path.read(), dtype=np.float64).copy()

    if arr.size % 6 != 0:
        name = getattr(path, "name", path)
        raise ValueError(
            f"MUSES lidar file {name!r} contains {arr.size} float64 values, "
            "which is not divisible by 6."
        )
    return arr.reshape((-1, 6))


def _load_json(path: Union[str, BinaryIO]) -> dict[str, Any]:
    """Load JSON from a file path or an in-memory buffer."""
    if isinstance(path, (str, os.PathLike)):
        with open(path) as f:
            data = json.load(f)
    else:
        data = json.load(path)
    if not isinstance(data, dict):
        name = getattr(path, "name", path)
        raise ValueError(f"MUSES calibration file {name!r} must contain a JSON object.")
    return data


def _select_string(
    *,
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> str | None:
    for source in (attributes, meta):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _available_keys(section: Mapping[str, Any]) -> str:
    return ", ".join(sorted(str(key) for key in section.keys()))


def _read_intrinsics_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> np.ndarray:
    data = _load_json(path)
    intrinsics = data.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("MUSES calibration JSON must contain an 'intrinsics' object.")

    sensor = _select_string(
        meta=meta,
        attributes=attributes,
        keys=("sensor", "camera", "intrinsics_sensor", "target_sensor"),
    ) or "rgb"
    entry = intrinsics.get(sensor)
    if not isinstance(entry, Mapping) or "K" not in entry:
        raise KeyError(
            f"MUSES calibration has no intrinsics for sensor {sensor!r}. "
            f"Available sensors: {_available_keys(intrinsics)}"
        )

    K = np.asarray(entry["K"], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(
            f"MUSES intrinsics for sensor {sensor!r} must have shape (3, 3), "
            f"got {K.shape}."
        )
    return K


def _select_extrinsics_key(
    *,
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    explicit = _select_string(
        meta=meta,
        attributes=attributes,
        keys=("transform", "extrinsics_key", "extrinsic", "key"),
    )
    if explicit is not None:
        return explicit

    source_sensor = _select_string(
        meta=meta,
        attributes=attributes,
        keys=("source_sensor", "from_sensor", "source"),
    )
    target_sensor = _select_string(
        meta=meta,
        attributes=attributes,
        keys=("target_sensor", "to_sensor", "target", "camera"),
    )
    if source_sensor is not None and target_sensor is not None:
        return f"{source_sensor}2{target_sensor}"

    return "lidar2rgb"


def _read_extrinsics_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> np.ndarray:
    data = _load_json(path)
    extrinsics = data.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        raise ValueError("MUSES calibration JSON must contain an 'extrinsics' object.")

    transform = _select_extrinsics_key(meta=meta, attributes=attributes)
    raw = extrinsics.get(transform)
    if raw is None:
        raise KeyError(
            f"MUSES calibration has no extrinsics transform {transform!r}. "
            f"Available transforms: {_available_keys(extrinsics)}"
        )

    T = np.asarray(raw, dtype=np.float32)
    if T.shape == (3, 4):
        T = np.vstack([T, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
    if T.shape != (4, 4):
        raise ValueError(
            f"MUSES extrinsics transform {transform!r} must have shape "
            f"(4, 4) or (3, 4), got {T.shape}."
        )
    return T


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES frame-camera RGB image as float32 in ``[0, 1]``."""
    return _load_image_rgb(path)


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta={"source": "reference_frames"},
)
def reference_rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES clear-weather reference RGB image as float32 in ``[0, 1]``."""
    return _load_image_rgb(path)


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={
        "encoding": "single_channel",
        "supported_suffixes": ["labelIds", "labelTrainIds"],
    },
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load MUSES ``labelIds`` or ``labelTrainIds`` semantic labels."""
    return _load_single_channel_labels(path)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={
        "encoding": "single_channel",
        "sky_class": [0, 0, _SKY_LABEL_ID],
        "sky_label_id": _SKY_LABEL_ID,
        "sky_train_id": _SKY_TRAIN_ID,
    },
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES sky mask as an ``(H, W)`` bool array.

    Reads a single-channel Cityscapes ``labelIds`` / ``labelTrainIds`` PNG, or
    an RGB-encoded class image when ``sky_class`` is provided. Override the
    default ID with ``sky_class_id``, or set ``sky_class`` to an RGB value such
    as ``[0, 0, 23]``.
    """
    return _load_sky_mask_array(path, meta, attributes)


@modality_meta(
    modality_type="semantic_segmentation_color",
    dtype="uint8",
    shape="HWC",
    file_formats=[".png"],
    meta={"encoding": "rgb", "supported_suffixes": ["labelColor"]},
)
def semantic_segmentation_color(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load MUSES ``labelColor`` semantic labels as RGB uint8 values."""
    return _load_rgb_uint8(path)


@modality_meta(
    modality_type="panoptic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={"encoding": "coco_rgb_id"},
)
def panoptic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES panoptic PNG as an integer segment-ID map."""
    return _decode_panoptic_rgb(_load_rgb_uint8(path))


@modality_meta(
    modality_type="point_cloud",
    dtype="float64",
    shape="NC",
    file_formats=[".bin"],
    meta={
        "columns": _LIDAR_COLUMNS,
        "coordinate_unit": "meters",
        "timestamp_unit": "seconds",
    },
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES lidar point cloud as ``(N, 6)`` float64."""
    return _load_lidar_array(path)


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Alias for :func:`lidar_point_cloud`."""
    return _load_lidar_array(path)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float64",
    shape="NC",
    file_formats=[".bin"],
    meta={
        "representation": "point_cloud",
        "columns": _LIDAR_COLUMNS,
        "coordinate_unit": "meters",
        "timestamp_unit": "seconds",
    },
)
def sparse_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load MUSES lidar points for sparse-depth style workflows."""
    return _load_lidar_array(path)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"default_sensor": "rgb"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES camera intrinsics matrix from ``calib.json``.

    The default sensor is ``"rgb"``. Override it with ``meta`` or per-file
    ``attributes`` using ``sensor``, ``camera``, ``intrinsics_sensor``, or
    ``target_sensor``.
    """
    return _read_intrinsics_array(path, meta, attributes)


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "default_transform": "lidar2rgb",
        "transform_direction": "source_sensor_to_target_sensor",
    },
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES camera extrinsics matrix from ``calib.json``.

    The default transform is ``"lidar2rgb"``, suitable for projecting MUSES
    lidar/sparse-depth points into the RGB camera. Override it with ``meta`` or
    per-file ``attributes`` using ``transform`` / ``extrinsics_key`` or
    ``source_sensor`` + ``target_sensor``.
    """
    return _read_extrinsics_array(path, meta, attributes)
