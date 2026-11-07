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
- **gnss** -- ``dict`` with the frame's GNSS/IMU record in physical units.
- **sensor_timing** -- ``dict`` with the frame's sensor timestamps from
  ``meta.json`` (hierarchical; selected per sample by file name).
- **read_lidar_to_gnss** -- ``np.ndarray`` of shape ``(4, 4)``, the
  ``lidar2gnss`` transform from ``calib.json``.

Writers: ``write_sparse_depth`` (aliases ``write_lidar_point_cloud``,
``write_point_cloud``) writes ``(N, 6)`` float64 ``.bin`` files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders._sky import (
    as_rgb,
    grey,
    match_rgb,
    resolve_skyclass,
    skyclass_label,
)
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    mark_stream_supported,
    to_numpy,
)

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


def _default_sky_class_id(path: Union[str, BinaryIO]) -> int:
    name = str(getattr(path, "name", path)).lower()
    if "labelids" in name:
        return _SKY_LABEL_ID
    return _SKY_TRAIN_ID


def _legacy_class_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"MUSES 'sky_class_id' must be an integer, got {value!r}")
    return int(value)


def _legacy_color(key: str) -> Any:
    return lambda value: as_rgb(value, what=f"MUSES {key!r}")


def _load_sky_mask_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> np.ndarray:
    loader = "muses sky_mask"
    color = resolve_skyclass(
        meta,
        attributes,
        loader=loader,
        default=(_default_sky_class_id(path),) * 3,
        legacy={
            "sky_class": _legacy_color("sky_class"),
            "sky_mask": _legacy_color("sky_mask"),
            "sky_class_id": grey(_legacy_class_id),
        },
    )
    with Image.open(path) as image:
        labels = np.array(image)
        if labels.ndim == 2:
            return labels.astype(np.int64) == skyclass_label(color, loader=loader)
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return match_rgb(rgb, color)


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
        "skyclass": [_SKY_LABEL_ID] * 3,
        "sky_label_id": _SKY_LABEL_ID,
        "sky_train_id": _SKY_TRAIN_ID,
    },
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a MUSES sky mask as an ``(H, W)`` bool array.

    The sky is ``skyclass`` from the file's ``attributes`` or the dataset
    head's ``meta``: a grey ``[v, v, v]`` selects label ``v`` in a
    single-channel ``labelIds`` / ``labelTrainIds`` PNG (other values raise),
    and a colour selects it in an RGB-encoded class image. Without
    ``skyclass`` the label defaults to 23 for ``labelIds`` files and 10
    otherwise; the deprecated ``sky_class_id``, ``sky_class`` and ``sky_mask``
    keys are still read when no ``skyclass`` is given.
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


# ---------------------------------------------------------------------------
# Ego-motion inputs: GNSS/IMU records, sensor timing, lidar-to-GNSS mount
# ---------------------------------------------------------------------------

# Field order of gnss/<split>/<condition>/<time>/<frame>_gnss.txt, as listed in
# the dataset's gnss/dataformat.txt (one line of integers per frame).
_GNSS_FIELDS = (
    "iTOW", "timestamp", "tAcc", "lon", "lat", "height", "hMSL", "fixType",
    "numSV", "hAcc", "vAcc", "roll", "pitch", "heading", "accRoll", "accPitch",
    "accHeading", "angular_rate_roll", "angular_rate_pitch",
    "angular_rate_heading", "velN", "velE", "velD", "gSpeed", "sAcc", "pDOP",
    "magDec", "magAcc",
)

# meta.json timestamp fields (integer microseconds stored as strings).
_TIMING_FIELDS = {
    "camera_exposure_start_us": "frame_camera_exposure_start_timestamp_us",
    "camera_exposure_end_us": "frame_camera_exposure_end_timestamp_us",
    "lidar_start_us": "lidar_start_timestamp_us",
    "lidar_end_us": "lidar_end_timestamp_us",
    "radar_start_us": "radar_start_timestamp_us",
    "radar_end_us": "radar_end_timestamp_us",
}


def _read_text(path: Union[str, BinaryIO]) -> str:
    if isinstance(path, (str, os.PathLike)):
        with open(path, encoding="utf-8") as f:
            return f.read()
    data = path.read()
    return data.decode("utf-8") if isinstance(data, bytes) else data


def _load_gnss_record(path: Union[str, BinaryIO]) -> dict[str, int]:
    lines = [line for line in _read_text(path).splitlines() if line.strip()]
    values = lines[0].split() if lines else []
    if len(values) != len(_GNSS_FIELDS):
        name = getattr(path, "name", path)
        raise ValueError(
            f"MUSES GNSS file {name!r} must hold one line of {len(_GNSS_FIELDS)} "
            f"integers (see gnss/dataformat.txt), found {len(values)} values."
        )
    return dict(zip(_GNSS_FIELDS, (int(v) for v in values)))


@modality_meta(
    modality_type="gnss",
    dtype="dict",
    shape="dict",
    file_formats=[".txt"],
    meta={
        "source": "u-blox navigation record, gnss/dataformat.txt",
        "heading_rate_axis": "negative yaw axis (clockwise from above is positive)",
    },
)
def gnss(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a MUSES GNSS/IMU record, one per frame, in physical units.

    The file holds the integer fields of ``gnss/dataformat.txt`` in u-blox
    units; they are returned scaled, with the unit in each key:

    - ``timestamp_us`` (UTC), ``gps_time_of_week_ms``, ``time_accuracy_ns``,
      ``fix_type``, ``num_satellites``;
    - ``latitude_deg``, ``longitude_deg``, ``height_m`` (ellipsoid),
      ``height_msl_m``, ``horizontal_accuracy_m``, ``vertical_accuracy_m``;
    - ``roll_deg``, ``pitch_deg``, ``heading_deg`` and their
      ``*_accuracy_deg``;
    - ``roll_rate_deg_s``, ``pitch_rate_deg_s``, ``heading_rate_deg_s``. The
      heading rate turns about the negative yaw axis, so turning clockwise
      seen from above is positive;
    - ``velocity_north_m_s``, ``velocity_east_m_s``, ``velocity_down_m_s``,
      ``ground_speed_m_s`` (2-D, negative when reversing),
      ``speed_accuracy_m_s``;
    - ``position_dop``, ``magnetic_declination_deg``,
      ``magnetic_declination_accuracy_deg``.

    The record is sampled at the frame camera's exposure. Index ``gnss/``
    as a regular modality with the same hierarchy and ID rules as the
    ``frame_camera`` and ``lidar`` folders, so samples join it by ID.
    """
    r = _load_gnss_record(path)
    return {
        "timestamp_us": r["timestamp"],
        "gps_time_of_week_ms": r["iTOW"],
        "time_accuracy_ns": r["tAcc"],
        "fix_type": r["fixType"],
        "num_satellites": r["numSV"],
        "latitude_deg": r["lat"] * 1e-7,
        "longitude_deg": r["lon"] * 1e-7,
        "height_m": r["height"] / 1e3,
        "height_msl_m": r["hMSL"] / 1e3,
        "horizontal_accuracy_m": r["hAcc"] / 1e3,
        "vertical_accuracy_m": r["vAcc"] / 1e3,
        "roll_deg": r["roll"] * 1e-5,
        "pitch_deg": r["pitch"] * 1e-5,
        "heading_deg": r["heading"] * 1e-5,
        "roll_accuracy_deg": r["accRoll"] * 1e-5,
        "pitch_accuracy_deg": r["accPitch"] * 1e-5,
        "heading_accuracy_deg": r["accHeading"] * 1e-5,
        "roll_rate_deg_s": r["angular_rate_roll"] * 1e-5,
        "pitch_rate_deg_s": r["angular_rate_pitch"] * 1e-5,
        "heading_rate_deg_s": r["angular_rate_heading"] * 1e-5,
        "velocity_north_m_s": r["velN"] / 1e3,
        "velocity_east_m_s": r["velE"] / 1e3,
        "velocity_down_m_s": r["velD"] / 1e3,
        "ground_speed_m_s": r["gSpeed"] / 1e3,
        "speed_accuracy_m_s": r["sAcc"] / 1e3,
        "position_dop": r["pDOP"] * 1e-2,
        "magnetic_declination_deg": r["magDec"] * 1e-2,
        "magnetic_declination_accuracy_deg": r["magAcc"] * 1e-2,
    }


def _timestamp_us(value: Any) -> int | None:
    if value is None or value == "None" or value == "":
        return None
    return int(value)


class SensorTimingTable(Mapping):
    """``meta.json`` timing records keyed by MUSES frame (``REC…_frame_…``).

    Besides the records it indexes every file name the entries list under
    ``path_to_*``, so a sample's own files identify its record.
    """

    def __init__(self, records: dict[str, dict[str, Any]], frame_of_file: dict[str, str]) -> None:
        self._records = records
        self._frame_of_file = frame_of_file

    def __getitem__(self, frame: str) -> dict[str, Any]:
        return self._records[frame]

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def frame_for_file(self, path: str) -> str | None:
        """Return the frame whose ``meta.json`` entry lists *path*'s file name."""
        return self._frame_of_file.get(os.path.basename(str(path)))


def _select_sensor_timing(table: SensorTimingTable, context: SampleContext) -> dict[str, Any]:
    frames: dict[str, list[str]] = {}
    for modality, entry in context.files.items():
        frame = table.frame_for_file(entry.get("path", ""))
        if frame is not None:
            frames.setdefault(frame, []).append(modality)
    if len(frames) == 1:
        return dict(table[next(iter(frames))])
    listed = ", ".join(f"{name}={entry.get('path')!r}" for name, entry in context.files.items())
    if not frames:
        raise KeyError(
            f"No MUSES meta.json entry lists a file of sample {context.full_id!r} "
            f"({listed}). The timing is matched by the native MUSES file names, "
            "so the sample's files must keep them."
        )
    raise ValueError(
        f"Sample {context.full_id!r} combines files of several MUSES frames: "
        + "; ".join(f"{frame} via {', '.join(names)}" for frame, names in frames.items())
    )


@selects_per_sample(_select_sensor_timing)
@modality_meta(
    modality_type="sensor_timing",
    dtype="dict",
    hierarchical=True,
    shape="dict",
    file_formats=[".json"],
    meta={"source": "meta.json", "unit": "microseconds", "matched_by": "path_to_* file names"},
)
def sensor_timing(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> SensorTimingTable:
    """Load the sensor timing of every frame from MUSES ``meta.json``.

    ``meta.json`` lists one entry per frame. Index it as a root-level
    hierarchical modality, like ``calib.json``. The file is decoded once into a
    :class:`SensorTimingTable`. Each sample then receives its own record,
    matched through the file names the entries list under ``path_to_*``, so
    any per-sample modality with native MUSES file names identifies it:

    ``{"frame", "camera_exposure_start_us", "camera_exposure_end_us",
    "lidar_start_us", "lidar_end_us", "radar_start_us", "radar_end_us"}``

    Times are integer UTC microseconds; "camera" is the frame camera (the
    ``rgb`` modality). Calling the loader directly returns the whole table.
    """
    data = _load_json(path)
    records: dict[str, dict[str, Any]] = {}
    frame_of_file: dict[str, str] = {}
    for frame, entry in data.items():
        if not isinstance(entry, Mapping):
            continue
        record: dict[str, Any] = {"frame": frame}
        for key, source in _TIMING_FIELDS.items():
            record[key] = _timestamp_us(entry.get(source))
        records[frame] = record
        for field, value in entry.items():
            if not field.startswith("path_to_") or not isinstance(value, str) or value in ("", "None"):
                continue
            name = os.path.basename(value)
            other = frame_of_file.setdefault(name, frame)
            if other != frame:
                raise ValueError(
                    f"MUSES meta.json lists the file {name!r} under both {other!r} and "
                    f"{frame!r}, so it cannot identify a frame."
                )
    return SensorTimingTable(records, frame_of_file)


@modality_meta(
    modality_type="lidar_to_gnss",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={"transform": "lidar2gnss", "transform_direction": "lidar_to_gnss"},
)
def read_lidar_to_gnss(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load the ``(4, 4)`` lidar → GNSS/IMU transform from ``calib.json``.

    This is ``extrinsics.lidar2gnss``. It maps lidar points into the frame of
    the :func:`gnss` record, the frame ego-motion compensation works in.
    """
    return _read_extrinsics_array(path, {"transform": "lidar2gnss"}, None)


@mark_stream_supported
def write_sparse_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write MUSES lidar points as an ``(N, 6)`` float64 ``.bin`` file.

    Columns are ``x, y, z, intensity, ring, timestamp``, as :func:`sparse_depth`
    reads them, e.g. after ego-motion compensation.
    """
    points = np.ascontiguousarray(to_numpy(value), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(_LIDAR_COLUMNS):
        raise ValueError(
            f"MUSES lidar points must have shape (N, {len(_LIDAR_COLUMNS)}), got {points.shape}"
        )
    ensure_parent(path)
    if isinstance(path, (str, os.PathLike)):
        with open(path, "wb") as f:
            f.write(points.tobytes())
        return
    path.write(points.tobytes())


write_lidar_point_cloud = write_sparse_depth
write_point_cloud = write_sparse_depth
