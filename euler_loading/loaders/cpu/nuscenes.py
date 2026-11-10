"""CPU-oriented loader functions for nuScenes, as prepared by euler-known-datasets.

nuScenes (Caesar et al., CVPR 2020, https://www.nuscenes.org/nuscenes) keeps
its calibration and vehicle poses in relational JSON tables. The
``euler-known-datasets`` recipe ``nuscenes`` turns a download into a dataset
these loaders read directly:

- ``rgb``: the camera JPEGs, unchanged;
- ``lidar``: the ``LIDAR_TOP`` ``.pcd.bin`` sweep synchronised to each camera
  frame, indexed under that frame's sample ID;
- ``intrinsics`` and ``camera_extrinsics``: one calibration file per scene and
  camera, ``calibration/<scene>/<camera>.json``, read as the camera matrix and
  as the frame's lidar-to-camera transform.

Calibration file
----------------
Format ``euler_known_datasets.nuscenes.camera_frames``, version 1, documented
in euler-known-datasets' ``docs/datasets/nuscenes.md``. It holds the camera's
``intrinsics`` and a record per camera frame, keyed by the sample ID (the
camera timestamp in microseconds). Each record carries the frame's
``camera_to_global`` pose and, when the frame has a synchronised sweep, that
sweep's ``lidar_to_camera`` transform. The transform is chained through the
vehicle poses at both timestamps, as the nuScenes devkit projects lidar. All
matrices are row-major. Cameras use optical axes (x right, y down, z
forward), and ``LIDAR_TOP`` points are x right, y forward, z up.

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **lidar_point_cloud** -- ``(N, 5)`` float32: ``x, y, z`` in metres in the
  ``LIDAR_TOP`` frame, ``intensity``, ``ring``.
- **read_intrinsics** -- ``(3, 3)`` float32 (hierarchical).
- **read_extrinsics** -- ``(4, 4)`` float32 lidar-to-camera transform of the
  sample's sweep (hierarchical, selected per sample).

Project a sweep into its image with ``K @ (T @ [p, 1])[:3]``.

All readers accept paths and binary streams, so zip-backed modalities work
unchanged.
"""

from __future__ import annotations

import io
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
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    pixel_grid_transform,
    replace_json,
    save_image,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_json,
)

__all__ = [
    "FORMAT_NAME",
    "CameraFrames",
    "rgb",
    "lidar_point_cloud",
    "point_cloud",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
    "write_rgb",
    "write_lidar_point_cloud",
    "write_point_cloud",
    "write_sparse_depth",
    "write_intrinsics",
]

#: The calibration file format these loaders read.
FORMAT_NAME = "euler_known_datasets.nuscenes.camera_frames"
SUPPORTED_FORMAT_VERSIONS = (1,)

_LIDAR_COLUMNS = ("x", "y", "z", "intensity", "ring")
_LIDAR_META = {
    "columns": list(_LIDAR_COLUMNS),
    "coordinate_frame": "LIDAR_TOP",
    "axes": "x right, y forward, z up",
    "coordinate_unit": "meters",
}
_CAMERA_AXES = "optical: x right, y down, z forward"
_JPEG_EXTENSIONS = (".jpg", ".jpeg")
_JPEG_QUALITY = 95
_DIRECTIONS = ("lidar_to_camera", "camera_to_lidar")


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _option(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None, key: str) -> Any:
    """Return *key* from the file's attributes, else the head's meta."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source[key]
    return None


def _matrix(value: Any, shape: tuple[int, int], what: str, source: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"nuScenes calibration {source!r}: {what} must be a finite {shape} matrix")
    return array


def _read_document(path: Union[str, BinaryIO]) -> dict[str, Any]:
    source = _source_name(path)
    try:
        document = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"nuScenes calibration {source!r} is not JSON: {exc}") from exc
    file_format = document.get("format") if isinstance(document, dict) else None
    if not isinstance(file_format, Mapping) or file_format.get("name") != FORMAT_NAME:
        raise ValueError(
            f"{source!r} is not a nuScenes calibration file ({FORMAT_NAME}); prepare "
            "the dataset with euler-known-datasets"
        )
    if file_format.get("version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"nuScenes calibration {source!r} has format version {file_format.get('version')!r}; "
            f"this euler-loading reads {', '.join(map(str, SUPPORTED_FORMAT_VERSIONS))}"
        )
    return document


class CameraFrames:
    """A parsed calibration file: one camera in one scene, frame by frame.

    Attributes:
        scene: The scene name, e.g. ``"scene-0077"``.
        camera: The camera channel, e.g. ``"CAM_FRONT_LEFT"``.
        intrinsics: ``(3, 3)`` float64 camera matrix.
        frames: The file's frame records, keyed by sample ID.
    """

    def __init__(self, document: Mapping[str, Any], *, source: str = "<calibration>"):
        self.source = source
        self.scene = str(document.get("scene", {}).get("name", ""))
        camera = document.get("camera") or {}
        self.camera = str(camera.get("channel", ""))
        self.intrinsics = _matrix(camera.get("intrinsics"), (3, 3), "camera.intrinsics", source)
        self.frames: Mapping[str, Mapping[str, Any]] = document.get("frames") or {}

    def lidar_to_camera(self, sample_id: str) -> np.ndarray:
        """The ``(4, 4)`` float64 transform from the frame's sweep into the camera."""
        record = self.frames.get(sample_id)
        if record is None:
            raise KeyError(
                f"nuScenes calibration {self.source!r} ({self.scene} {self.camera}) has no "
                f"frame {sample_id!r}; the sample ID must be the camera timestamp"
            )
        lidar = record.get("lidar")
        if lidar is None:
            raise KeyError(
                f"nuScenes frame {sample_id!r} of {self.scene} {self.camera} has no "
                "synchronised lidar sweep; join it with the prepared 'lidar' scope, "
                "which leaves such frames out"
            )
        return _matrix(lidar["lidar_to_camera"], (4, 4), f"frames[{sample_id}].lidar_to_camera", self.source)

    def camera_to_global(self, sample_id: str) -> np.ndarray:
        """The ``(4, 4)`` float64 camera pose in the scene's map frame."""
        record = self.frames.get(sample_id)
        if record is None:
            raise KeyError(f"nuScenes calibration {self.source!r} has no frame {sample_id!r}")
        return _matrix(record["camera_to_global"], (4, 4), f"frames[{sample_id}].camera_to_global", self.source)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".jpg"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a camera JPEG as ``(H, W, 3)`` float32 in ``[0, 1]`` (1600x900 for nuScenes)."""
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_points(path: Union[str, BinaryIO]) -> np.ndarray:
    data = _read_bytes(path)
    width = len(_LIDAR_COLUMNS) * 4
    if len(data) % width:
        raise ValueError(
            f"nuScenes lidar sweep {_source_name(path)!r} has {len(data)} bytes, not a "
            f"multiple of {width} (five float32 columns per point)"
        )
    return np.frombuffer(data, dtype="<f4").reshape(-1, len(_LIDAR_COLUMNS)).astype(np.float32)


@modality_meta(
    modality_type="point_cloud",
    dtype="float32",
    shape="NC",
    file_formats=[".bin"],
    meta=_LIDAR_META,
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``LIDAR_TOP`` ``.pcd.bin`` sweep as ``(N, 5)`` float32.

    Columns are ``x, y, z`` in metres in the ``LIDAR_TOP`` frame (x right,
    y forward, z up), ``intensity`` and the laser ``ring`` index. Points are
    given at the sweep's timestamp, the end of its rotation, as the nuScenes
    devkit reads them. :func:`read_extrinsics` maps them into the camera.
    """
    return _load_points(path)


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Alias for :func:`lidar_point_cloud`."""
    return _load_points(path)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float32",
    shape="NC",
    file_formats=[".bin"],
    meta={"representation": "point_cloud", **_LIDAR_META},
)
def sparse_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load lidar points for sparse-depth workflows; see :func:`lidar_point_cloud`."""
    return _load_points(path)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"model": "pinhole", "distortion": None, "source": "calibration/<scene>/<camera>.json"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the camera's ``(3, 3)`` float32 matrix from a calibration file.

    nuScenes calibrates cameras as ideal pinholes; there are no distortion
    coefficients. The matrix is constant within a scene, so index the file at
    the scene/camera level.
    """
    document = _read_document(path)
    return CameraFrames(document, source=_source_name(path)).intrinsics.astype(np.float32)


def _direction(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None) -> str:
    direction = _option(meta, attributes, "transform_direction") or "lidar_to_camera"
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"transform_direction must be one of {', '.join(_DIRECTIONS)}, got {direction!r}"
        )
    return direction


class _ExtrinsicsTable(CameraFrames):
    """A calibration file plus the direction its transforms are returned in."""

    def __init__(self, document: Mapping[str, Any], *, source: str, direction: str):
        super().__init__(document, source=source)
        self.direction = direction


def _invert_rigid(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _select_extrinsics(table: _ExtrinsicsTable, context: SampleContext) -> np.ndarray:
    transform = table.lidar_to_camera(context.id)
    if table.direction == "camera_to_lidar":
        transform = _invert_rigid(transform)
    return transform.astype(np.float32)


@selects_per_sample(_select_extrinsics)
@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "transform": "lidar_to_camera",
        "source_frame": "LIDAR_TOP",
        "target_frame": "camera",
        "camera_axes": _CAMERA_AXES,
        "translation_unit": "meters",
        "selected_at": "the sample's camera frame",
        "matched_by": "sample ID (the camera timestamp)",
        "options": {"transform_direction": list(_DIRECTIONS)},
    },
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> CameraFrames:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32
    lidar-to-camera transform.

    The transform belongs to the sweep synchronised with the sample's camera
    frame, chained through the vehicle poses at both timestamps:
    ``inv(camera_to_ego) · inv(ego_pose(t_cam)) · ego_pose(t_lidar) ·
    lidar_to_ego``. It is therefore per frame, not per scene. Samples are
    matched by their ID, the camera timestamp, so join this modality with the
    prepared ``rgb`` and ``lidar`` scopes. A frame without a synchronised
    sweep raises ``KeyError``.

    ``transform_direction``, read from the file's attributes and then the
    head's meta, is ``lidar_to_camera`` (default) or ``camera_to_lidar``.
    Calling the loader directly returns the whole :class:`CameraFrames` table.
    """
    source = _source_name(path)
    return _ExtrinsicsTable(_read_document(path), source=source, direction=_direction(meta, attributes))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as 8-bit JPEG (quality 95) or, for other names, PNG."""
    ensure_parent(path)
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    if os.path.splitext(get_target_name(path))[1].lower() in _JPEG_EXTENSIONS:
        image.save(path, format="JPEG", quality=_JPEG_QUALITY)
        return
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_lidar_point_cloud(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write ``(N, 5)`` points as a nuScenes ``.pcd.bin`` (little-endian float32).

    What :func:`lidar_point_cloud` reads back unchanged, e.g. after keeping
    only the points inside a crop.
    """
    points = to_numpy(value)
    if points.ndim != 2 or points.shape[1] != len(_LIDAR_COLUMNS):
        raise ValueError(
            f"nuScenes lidar points must have shape (N, {len(_LIDAR_COLUMNS)}) "
            f"({', '.join(_LIDAR_COLUMNS)}), got {points.shape}"
        )
    data = np.ascontiguousarray(points, dtype="<f4").tobytes()
    ensure_parent(path)
    if isinstance(path, (str, os.PathLike)):
        with open(path, "wb") as stream:
            stream.write(data)
        return
    path.write(data)


write_point_cloud = write_lidar_point_cloud
write_sparse_depth = write_lidar_point_cloud


def _image_size(meta: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(meta, Mapping) or meta.get("width") is None or meta.get("height") is None:
        return None
    return int(meta["width"]), int(meta["height"])


def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a camera matrix into a nuScenes calibration file.

    *value* is what :func:`read_intrinsics` returns. An existing file is
    updated, not replaced. The change from its stored matrix must be a crop or
    a rescale; ``meta["width"]`` and ``meta["height"]`` update the recorded
    image size. The frame records stay as they are: a crop or rescale only
    moves the pixel grid, so lidar projected with the written matrix and the
    unchanged ``lidar_to_camera`` lands on the cropped pixels. An unchanged
    matrix leaves the file byte-identical.

    A new file, or a stream target, receives a calibration file with only the
    camera matrix (and size), which :func:`read_intrinsics` reads.
    """
    matrix = to_numpy(value).astype(np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"intrinsics must be a finite (3, 3) matrix, got shape {matrix.shape}")
    size = _image_size(meta)

    if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
        name = os.fspath(path)
        document = _read_document(path)
        stored = CameraFrames(document, source=name).intrinsics
        transform = pixel_grid_transform(matrix @ np.linalg.inv(stored))
        if transform is None:
            raise ValueError(
                f"Refusing to update {name!r}: the new intrinsics are not a crop or rescale "
                f"of the stored matrix ({stored.tolist()} -> {matrix.tolist()})."
            )
        camera = document["camera"]
        updated = dict(camera, intrinsics=(transform @ stored).tolist())
        if size is not None:
            updated["width"], updated["height"] = size
        if updated == camera:
            return
        document["camera"] = updated
        replace_json(path, document)
        return

    camera: dict[str, Any] = {"intrinsics": matrix.tolist()}
    if size is not None:
        camera["width"], camera["height"] = size
    write_json(path, {"format": {"name": FORMAT_NAME, "version": 1}, "camera": camera, "frames": {}})
