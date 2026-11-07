"""CPU-oriented loader functions for the KITTI-CARLA dataset.

KITTI-CARLA (Deschaud, 2021, https://arxiv.org/abs/2109.00892) is rendered
with CARLA 0.9.10 from a vehicle carrying KITTI's sensor layout: a Velodyne
HDL-64E-like lidar on the roof and two 1392x1024 colour cameras with a 72
degree horizontal field of view. Each town is laid out as::

    TownNN/generated/
        images_rgb/<frame>_<camera>.png     # rgb
        images_ss/<frame>_1<camera>.png     # CARLA semantic colours
        images_depth/<frame>_2<camera>.png  # CARLA 24-bit depth
        frames/frame_<frame>.ply            # lidar sweep
        lidar_to_cam0.txt, lidar_to_cam1.txt
        full_poses_lidar.txt                # lidar poses at 1000 Hz
        full_ts_camera.txt

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``; the constant alpha channel
  of the RGBA PNGs is dropped.
- **depth** -- ``(H, W)`` float32 planar (z-buffer) depth in metres, decoded
  from CARLA's ``(R + 256 G + 65536 B) / (2**24 - 1) * 1000``. Sky and other
  pixels without a hit sit at the 1000 m far plane.
- **class_segmentation** -- ``(H, W)`` int64 CARLA 0.9.10 semantic tags
  (``0`` unlabeled ... ``22`` terrain, see :data:`CLASS_NAMES`), decoded from
  the colour-coded ``images_ss`` PNGs.
- **sky_mask** -- ``(H, W)`` bool, true where ``images_ss`` has the
  ``skyclass`` colour, ``(70, 130, 180)`` by default.
- **lidar_point_cloud** / **point_cloud** / **sparse_depth** -- ``(N, 7)``
  float64 with columns ``x, y, z, cos_angle_lidar_surface, timestamp,
  instance, semantic``.
- **read_intrinsics** -- ``(3, 3)`` float32 pinhole matrix (hierarchical).
- **read_extrinsics** -- ``(4, 4)`` float32 lidar-to-camera transform
  (hierarchical).
- **lidar_pose** -- ``(4, 4)`` float64 lidar-to-world pose at the sample's
  camera frame, selected per sample from ``full_poses_lidar.txt``
  (hierarchical).

Coordinate frames
-----------------
Lidar points, ``lidar_to_cam*.txt`` and ``full_poses_lidar.txt`` all use
right-handed vehicle axes: x forward, y left, z up. The calibration files map
lidar coordinates into each camera, ``X_cam = T @ X_lidar``, with the camera
axes still vehicle-aligned (their rotation is the identity).
:func:`read_extrinsics` therefore returns the transform into optical camera
axes (x right, y down, z forward) by default, so that ``read_intrinsics``
projects its output directly; ``camera_axes="vehicle"`` returns the stored
matrix. A lidar sweep ``i`` spans ``[0.1 i, 0.1 i + 0.1)`` seconds, and camera
frame ``i`` is taken at its start. Points are given in the lidar frame at the
moment each was measured, without motion compensation.

All readers accept paths and binary streams, so zip-backed modalities work
unchanged. Every writer accepts both as well.
"""

from __future__ import annotations

import io
import math
import os
import re
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders._sky import RGB, match_rgb, resolve_skyclass
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    mark_stream_supported,
    save_image,
    to_bool_mask,
    to_hw,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_text,
)

__all__ = [
    "CLASS_NAMES",
    "PALETTE",
    "LidarPoseTable",
    "rgb",
    "depth",
    "class_segmentation",
    "sky_mask",
    "lidar_point_cloud",
    "point_cloud",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
    "lidar_pose",
    "write_rgb",
    "write_depth",
    "write_class_segmentation",
    "write_sky_mask",
    "write_lidar_point_cloud",
    "write_point_cloud",
    "write_sparse_depth",
    "write_intrinsics",
    "write_extrinsics",
]

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

#: CARLA 0.9.10 semantic tags, indexed by tag ID, as listed in the dataset's
#: readme. ``images_ss`` stores them with the colours in :data:`PALETTE`; the
#: ``semantic`` lidar column stores the IDs directly.
CLASS_NAMES: tuple[str, ...] = (
    "unlabeled", "building", "fence", "other", "pedestrian", "pole",
    "road_line", "road", "sidewalk", "vegetation", "vehicles", "wall",
    "traffic_sign", "sky", "ground", "bridge", "rail_track", "guard_rail",
    "traffic_light", "static", "dynamic", "water", "terrain",
)

#: ``(23, 3)`` uint8 colour of each tag in :data:`CLASS_NAMES` (CARLA's
#: ``CityScapesPalette``).
PALETTE: np.ndarray = np.array(
    [
        (0, 0, 0), (70, 70, 70), (100, 40, 40), (55, 90, 80), (220, 20, 60),
        (153, 153, 153), (157, 234, 50), (128, 64, 128), (244, 35, 232),
        (107, 142, 35), (0, 0, 142), (102, 102, 156), (220, 220, 0),
        (70, 130, 180), (81, 0, 81), (150, 100, 100), (230, 150, 140),
        (180, 165, 180), (250, 170, 30), (110, 190, 160), (170, 120, 50),
        (45, 60, 150), (145, 170, 100),
    ],
    dtype=np.uint8,
)
PALETTE.setflags(write=False)

_SKY_COLOR: RGB = (70, 130, 180)

# Camera model from the KITTI-CARLA paper; CARLA cameras are ideal pinholes
# with the principal point at the image centre.
_IMAGE_WIDTH = 1392
_IMAGE_HEIGHT = 1024
_HORIZONTAL_FOV_DEG = 72.0

# CARLA depth: a 24-bit code in R (low), G, B (high), normalised to the
# 1000 m far plane.
_DEPTH_FAR_PLANE_M = 1000.0
_DEPTH_MAX_CODE = 2**24 - 1
_DEPTH_SCALE_TO_METERS = _DEPTH_FAR_PLANE_M / _DEPTH_MAX_CODE

_LIDAR_COLUMNS = [
    "x", "y", "z", "cos_angle_lidar_surface", "timestamp", "instance", "semantic",
]
_LIDAR_PLY_TYPES = [
    ("x", "float32"), ("y", "float32"), ("z", "float32"),
    ("cos_angle_lidar_surface", "float32"), ("timestamp", "float32"),
    ("instance", "uint32"), ("semantic", "uint32"),
]

_EXTRINSICS_HEADER = (
    "#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) "
    "R(2,0) R(2,1) R(2,2) t(2)"
)
_INTRINSICS_HEADER = (
    "#K(0,0) K(0,1) K(0,2) K(1,0) K(1,1) K(1,2) K(2,0) K(2,1) K(2,2)"
)

_VEHICLE_AXES = "x_forward_y_left_z_up"
_OPTICAL_AXES = "x_right_y_down_z_forward"
# Maps vehicle-aligned camera coordinates to optical ones:
# x_opt = -y, y_opt = -z, z_opt = x.
_OPTICAL_FROM_VEHICLE = np.array(
    [[0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_CAMERA_AXES = ("optical", "vehicle")
_TRANSFORM_DIRECTIONS = ("lidar_to_camera", "camera_to_lidar")

_CAMERA_RATE_HZ = 10.0
# Poses are sampled at 1000 Hz; half a step tells a matching pose from none.
_POSE_TOLERANCE_S = 0.0005
# frame_0000.ply, 0000_0.png, 0000_10.png, 0000_20.png
_FRAME_FROM_NAME = re.compile(r"^(?:frame_)?(\d+)(?:_\d+)?\.[^./\\]+$")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _load_rgb_uint8(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a PNG as ``(H, W, 3)`` uint8, dropping the alpha channel."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _option(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    key: str,
) -> Any:
    """Return *key* from the file's attributes, else the head's meta."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source[key]
    return None


def _positive_number(value: Any, key: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"KITTI-CARLA {key!r} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"KITTI-CARLA {key!r} must be positive and finite, got {value!r}")
    return value


def _choice(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    key: str,
    allowed: tuple[str, ...],
) -> str:
    value = _option(meta, attributes, key)
    if value is None:
        return allowed[0]
    if not isinstance(value, str) or value.lower() not in allowed:
        raise ValueError(
            f"KITTI-CARLA {key!r} must be one of "
            f"{', '.join(repr(option) for option in allowed)}, got {value!r}"
        )
    return value.lower()


def _read_calibration_values(path: Union[str, BinaryIO]) -> np.ndarray:
    """Return the numbers of a ``#``-commented calibration text file."""
    text = _read_bytes(path).decode("utf-8")
    values: list[float] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            values.extend(float(token) for token in line.split())
    return np.asarray(values, dtype=np.float64)


def _format_values(values: np.ndarray) -> str:
    # repr() is the shortest text that reads back to the same float64.
    return " ".join(repr(float(value)) for value in values.reshape(-1))


# ---------------------------------------------------------------------------
# Camera images
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an ``images_rgb`` PNG as ``(H, W, 3)`` float32 in ``[0, 1]``.

    The files are RGBA with a constant alpha of 255, which is dropped.
    """
    return _load_rgb_uint8(path).astype(np.float32) / 255.0


def _decode_depth(rgb_uint8: np.ndarray) -> np.ndarray:
    code = (
        rgb_uint8[..., 0].astype(np.uint32)
        | (rgb_uint8[..., 1].astype(np.uint32) << 8)
        | (rgb_uint8[..., 2].astype(np.uint32) << 16)
    )
    # float64 keeps the far plane at exactly 1000 m after the cast.
    return (code * _DEPTH_SCALE_TO_METERS).astype(np.float32)


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, _DEPTH_FAR_PLANE_M],
    output_unit="meters",
    meta={
        "encoding": "carla_rgb24",
        "decode": "(R + 256 * G + 65536 * B) / (2**24 - 1) * 1000",
        "raw_range": [0, _DEPTH_MAX_CODE],
        "scale_to_meters": _DEPTH_SCALE_TO_METERS,
        "radial_depth": False,
        "far_plane_m": _DEPTH_FAR_PLANE_M,
    },
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an ``images_depth`` PNG as ``(H, W)`` float32 planar depth in metres.

    CARLA packs depth into a 24-bit code, R holding the low byte and B the
    high byte, normalised to the 1000 m far plane:
    ``(R + 256 G + 65536 B) / (2**24 - 1) * 1000``. The resolution is about
    0.06 mm. Values are distances along the optical axis (z-buffer depth),
    not along the ray. Sky pixels read as exactly 1000 m; mask them with
    :func:`sky_mask` or a ``< 1000`` threshold.
    """
    return _decode_depth(_load_rgb_uint8(path))


def _pack_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """Pack ``(..., 3)`` uint8 colours into ``0xRRGGBB`` int64 keys."""
    rgb = rgb_uint8.astype(np.int64)
    return rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]


# Palette colours as packed keys, sorted for a binary-search decode.
_PACKED_PALETTE = _pack_rgb(PALETTE)
_PALETTE_ORDER = np.argsort(_PACKED_PALETTE)
_SORTED_PACKED_PALETTE = _PACKED_PALETTE[_PALETTE_ORDER]


def _decode_classes(rgb_uint8: np.ndarray, *, source: str) -> np.ndarray:
    packed = _pack_rgb(rgb_uint8)
    position = np.searchsorted(_SORTED_PACKED_PALETTE, packed)
    position = np.minimum(position, len(_SORTED_PACKED_PALETTE) - 1)
    known = _SORTED_PACKED_PALETTE[position] == packed
    if not known.all():
        unknown = np.unique(packed[~known])
        shown = ", ".join(
            str([int(value >> 16), int(value >> 8 & 255), int(value & 255)])
            for value in unknown[:5]
        )
        more = f" and {len(unknown) - 5} more" if len(unknown) > 5 else ""
        raise ValueError(
            f"KITTI-CARLA segmentation {source!r} has {int((~known).sum())} pixels "
            f"in colours outside the CARLA palette: {shown}{more}."
        )
    return _PALETTE_ORDER[position].astype(np.int64)


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    output_range=[0, len(CLASS_NAMES) - 1],
    meta={
        "encoding": "carla_palette",
        "label_ids": "carla_0.9.10",
        "label_names": list(CLASS_NAMES),
        "skyclass": list(_SKY_COLOR),
    },
)
def class_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an ``images_ss`` PNG as ``(H, W)`` int64 CARLA semantic tags.

    The colour of every pixel is looked up in :data:`PALETTE`, so the result
    holds the tag IDs of :data:`CLASS_NAMES` -- the same IDs as the lidar's
    ``semantic`` column. A colour outside the palette raises ``ValueError``
    rather than being given a label silently.
    """
    return _decode_classes(_load_rgb_uint8(path), source=_source_name(path))


def _sky_color(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None = None,
) -> RGB:
    return resolve_skyclass(
        meta, attributes, loader="kitti_carla sky_mask", default=_SKY_COLOR,
    )


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"skyclass": list(_SKY_COLOR)},
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a sky mask from an ``images_ss`` PNG as ``(H, W)`` bool.

    True where the pixel has the ``skyclass`` colour from the file's
    ``attributes`` or the dataset head's ``meta``, CARLA's sky colour
    ``(70, 130, 180)`` by default.
    """
    return match_rgb(_load_rgb_uint8(path), _sky_color(meta, attributes))


# ---------------------------------------------------------------------------
# Lidar
# ---------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_PLY_ENDIANNESS = {"binary_little_endian": "<", "binary_big_endian": ">"}


def _read_ply_vertices(data: bytes, *, source: str) -> np.ndarray:
    """Parse a binary PLY holding one ``vertex`` element into a structured array."""
    stream = io.BytesIO(data)
    if stream.readline().strip() != b"ply":
        raise ValueError(f"{source!r} is not a PLY file")
    endianness = None
    count = None
    fields: list[tuple[str, str]] = []
    in_vertex = False
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError(f"PLY file {source!r} has no end_header line")
        tokens = raw.decode("ascii").split()
        if not tokens or tokens[0] in ("comment", "obj_info"):
            continue
        keyword = tokens[0]
        if keyword == "end_header":
            break
        if keyword == "format":
            if len(tokens) < 2 or tokens[1] not in _PLY_ENDIANNESS:
                raise ValueError(
                    f"PLY file {source!r} has format {' '.join(tokens[1:])!r}; "
                    "only binary_little_endian and binary_big_endian are supported"
                )
            endianness = _PLY_ENDIANNESS[tokens[1]]
        elif keyword == "element":
            if count is not None or len(tokens) != 3 or tokens[1] != "vertex":
                raise ValueError(
                    f"PLY file {source!r} must hold a single 'vertex' element, "
                    f"found {' '.join(tokens[1:])!r}"
                )
            count = int(tokens[2])
            in_vertex = True
        elif keyword == "property":
            if not in_vertex:
                raise ValueError(f"PLY file {source!r} declares a property before its element")
            if len(tokens) != 3 or tokens[1] not in _PLY_TYPES:
                raise ValueError(
                    f"PLY file {source!r} has an unsupported property {' '.join(tokens[1:])!r}"
                )
            fields.append((tokens[2], tokens[1]))
    if endianness is None or count is None:
        raise ValueError(f"PLY file {source!r} declares no format or vertex count")
    dtype = np.dtype([(name, endianness + _PLY_TYPES[kind]) for name, kind in fields])
    offset = stream.tell()
    if len(data) - offset < count * dtype.itemsize:
        raise ValueError(
            f"PLY file {source!r} declares {count} vertices but holds only "
            f"{(len(data) - offset) // dtype.itemsize}"
        )
    return np.frombuffer(data, dtype=dtype, count=count, offset=offset)


def _load_lidar_array(path: Union[str, BinaryIO]) -> np.ndarray:
    source = _source_name(path)
    vertices = _read_ply_vertices(_read_bytes(path), source=source)
    missing = [name for name in _LIDAR_COLUMNS if name not in (vertices.dtype.names or ())]
    if missing:
        raise ValueError(
            f"KITTI-CARLA lidar frame {source!r} lacks the field(s) "
            f"{', '.join(missing)}; found {', '.join(vertices.dtype.names or ())}"
        )
    points = np.empty((len(vertices), len(_LIDAR_COLUMNS)), dtype=np.float64)
    for column, name in enumerate(_LIDAR_COLUMNS):
        points[:, column] = vertices[name]
    return points


_LIDAR_META = {
    "columns": _LIDAR_COLUMNS,
    "coordinate_frame": "lidar",
    "axes": _VEHICLE_AXES,
    "coordinate_unit": "meters",
    "timestamp_unit": "seconds",
    "semantic_ids": "carla_0.9.10",
    "motion_compensated": False,
}


@modality_meta(
    modality_type="point_cloud",
    dtype="float64",
    shape="NC",
    file_formats=[".ply"],
    meta=_LIDAR_META,
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``frames/frame_<frame>.ply`` lidar sweep as ``(N, 7)`` float64.

    Columns are ``x, y, z`` in metres, ``cos_angle_lidar_surface`` (cosine
    between the ray and the surface it hit), ``timestamp`` in seconds,
    ``instance`` and ``semantic`` (the CARLA tag IDs of
    :data:`CLASS_NAMES`). float64 holds the stored float32 and uint32 columns
    exactly. Each point is in the lidar frame at its own timestamp, without
    ego-motion compensation; use :func:`lidar_pose` for the trajectory.
    """
    return _load_lidar_array(path)


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Alias for :func:`lidar_point_cloud`."""
    return _load_lidar_array(path)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float64",
    shape="NC",
    file_formats=[".ply"],
    meta={"representation": "point_cloud", **_LIDAR_META},
)
def sparse_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load lidar points for sparse-depth workflows; see :func:`lidar_point_cloud`."""
    return _load_lidar_array(path)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _intrinsics_from_fov(width: float, height: float, fov_deg: float) -> np.ndarray:
    if fov_deg >= 180.0:
        raise ValueError(f"KITTI-CARLA 'horizontal_fov_deg' must be below 180, got {fov_deg}")
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".txt"],
    meta={
        "width": _IMAGE_WIDTH,
        "height": _IMAGE_HEIGHT,
        "horizontal_fov_deg": _HORIZONTAL_FOV_DEG,
        "source": "camera model from the KITTI-CARLA paper",
    },
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(3, 3)`` float32 camera matrix for a KITTI-CARLA camera.

    KITTI-CARLA ships no intrinsics file: both cameras are ideal CARLA
    pinholes of 1392x1024 pixels with a 72 degree horizontal field of view,
    so ``fx = fy = W / (2 tan(fov / 2))`` (about 957.96) and the principal
    point is the image centre. Index a ``lidar_to_cam<N>.txt`` file to place
    the camera in the hierarchy; its contents are validated but carry no
    intrinsics. ``width``, ``height`` and ``horizontal_fov_deg`` in the
    file's attributes or the head's meta override the defaults.

    A file of nine numbers, as :func:`write_intrinsics` produces, is read as
    the matrix itself, e.g. after a crop or resize.
    """
    values = _read_calibration_values(path)
    if values.size == 9:
        return values.reshape(3, 3).astype(np.float32)
    if values.size != 12:
        raise ValueError(
            f"KITTI-CARLA intrinsics source {_source_name(path)!r} must hold a 3x4 "
            f"lidar_to_cam transform (12 numbers) or a 3x3 matrix (9), found {values.size}"
        )
    width = _positive_number(_option(meta, attributes, "width"), "width", _IMAGE_WIDTH)
    height = _positive_number(_option(meta, attributes, "height"), "height", _IMAGE_HEIGHT)
    fov = _positive_number(
        _option(meta, attributes, "horizontal_fov_deg"), "horizontal_fov_deg", _HORIZONTAL_FOV_DEG,
    )
    return _intrinsics_from_fov(width, height, fov)


def _check_rigid(T: np.ndarray) -> None:
    """Reject what a lidar-to-camera calibration cannot be."""
    if not np.all(np.isfinite(T)):
        raise ValueError("extrinsics must be finite")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"extrinsics must have the bottom row [0, 0, 0, 1], got {T[3].tolist()}")
    # float32 inputs carry about 1e-7 rounding per entry.
    rotation = T[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("extrinsics rotation must be orthonormal (no scale or shear)")
    if np.linalg.det(rotation) < 0:
        raise ValueError("extrinsics rotation must be right-handed (no mirroring)")


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation_t = T[:3, :3].T
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -rotation_t @ T[:3, 3]
    return inverse


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".txt"],
    meta={
        "source": "lidar_to_cam<N>.txt",
        "translation_unit": "meters",
        "stored_transform": "lidar_to_camera",
        "stored_axes": _VEHICLE_AXES,
        "default_transform_direction": "lidar_to_camera",
        "default_camera_axes": "optical",
        "optical_axes": _OPTICAL_AXES,
    },
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load ``lidar_to_cam<N>.txt`` as a ``(4, 4)`` float32 rigid transform.

    The file holds a row-major ``[R | t]`` that maps lidar points into the
    camera, ``X_cam = T @ X_lidar``, both in vehicle axes (x forward, y left,
    z up). By default the result maps into optical camera axes (x right,
    y down, z forward) instead, so lidar projects through
    :func:`read_intrinsics` directly::

        uvw = K @ (T @ points_h.T)[:3]

    Two keys, read from the file's attributes and then the head's meta,
    select other conventions:

    - ``camera_axes``: ``"optical"`` (default) or ``"vehicle"`` for the
      stored axes;
    - ``transform_direction``: ``"lidar_to_camera"`` (default) or
      ``"camera_to_lidar"``, the camera's pose in the lidar frame.
    """
    values = _read_calibration_values(path)
    if values.size != 12:
        raise ValueError(
            f"KITTI-CARLA extrinsics {_source_name(path)!r} must hold 12 numbers "
            f"(a row-major 3x4 [R | t]), found {values.size}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = values.reshape(3, 4)
    if _choice(meta, attributes, "camera_axes", _CAMERA_AXES) == "optical":
        transform = _OPTICAL_FROM_VEHICLE @ transform
    if _choice(meta, attributes, "transform_direction", _TRANSFORM_DIRECTIONS) == "camera_to_lidar":
        transform = _invert_rigid(transform)
    return transform.astype(np.float32)


# ---------------------------------------------------------------------------
# Lidar trajectory
# ---------------------------------------------------------------------------


class LidarPoseTable:
    """Lidar poses from ``full_poses_lidar.txt``, one per 1 ms simulation step.

    ``poses`` is ``(M, 4, 4)`` float64 lidar-to-world, ``timestamps`` the
    matching ``(M,)`` seconds. The world frame is the lidar frame at the first
    step, with the vehicle axes x forward, y left, z up.
    """

    def __init__(self, poses: np.ndarray, timestamps: np.ndarray) -> None:
        self.poses = poses
        self.timestamps = timestamps

    def __len__(self) -> int:
        return len(self.timestamps)

    def __repr__(self) -> str:
        if len(self.timestamps) == 0:
            return "LidarPoseTable(0 poses)"
        return (
            f"LidarPoseTable({len(self.timestamps)} poses, "
            f"{self.timestamps[0]:.3f}-{self.timestamps[-1]:.3f} s)"
        )

    def at_times(self, times: Any) -> np.ndarray:
        """Return the ``(..., 4, 4)`` poses at *times*, each within half a 1 ms step.

        Pass a sweep's ``timestamp`` column to move every point into the
        world frame at the moment it was measured, which compensates the
        sweep's ego-motion::

            world = np.einsum("nij,nj->ni", table.at_times(points[:, 4]), points_h)
        """
        times = np.asarray(times, dtype=np.float64)
        if len(self.timestamps) == 0:
            raise KeyError("KITTI-CARLA lidar pose table is empty")
        last = len(self.timestamps) - 1
        upper = np.clip(np.searchsorted(self.timestamps, times), 0, last)
        lower = np.clip(upper - 1, 0, last)
        closer_upper = np.abs(self.timestamps[upper] - times) < np.abs(self.timestamps[lower] - times)
        best = np.where(closer_upper, upper, lower)
        error = np.abs(self.timestamps[best] - times)
        if np.any(error > _POSE_TOLERANCE_S):
            worst = float(np.ravel(times)[int(np.argmax(error))])
            raise KeyError(
                f"No KITTI-CARLA lidar pose within {_POSE_TOLERANCE_S * 1e3:g} ms of "
                f"{worst} s; the table covers {self.timestamps[0]} to {self.timestamps[-1]} s"
            )
        # take() copies even for a scalar time, so callers never hold a view
        # into the table that hierarchical caching shares between samples.
        return np.take(self.poses, best, axis=0)

    def at(self, time: float) -> np.ndarray:
        """Return the ``(4, 4)`` pose at *time*, within half a 1 ms step."""
        return self.at_times(time)

    def for_frame(self, frame: int) -> np.ndarray:
        """Return the pose at camera frame *frame*, the start of lidar sweep *frame*."""
        return self.at(frame / _CAMERA_RATE_HZ)


def _frame_of(path: str) -> int | None:
    match = _FRAME_FROM_NAME.match(os.path.basename(str(path)))
    return int(match.group(1)) if match else None


def _select_lidar_pose(table: LidarPoseTable, context: SampleContext) -> np.ndarray:
    frames: dict[int, list[str]] = {}
    for modality, entry in context.files.items():
        frame = _frame_of(entry.get("path", ""))
        if frame is not None:
            frames.setdefault(frame, []).append(modality)
    if len(frames) == 1:
        return table.for_frame(next(iter(frames)))
    if not frames:
        listed = ", ".join(f"{name}={entry.get('path')!r}" for name, entry in context.files.items())
        raise KeyError(
            f"No KITTI-CARLA frame number in the file names of sample "
            f"{context.full_id!r} ({listed}). The pose is matched by the native "
            "names (frame_<frame>.ply, <frame>_<camera>.png), so the sample's "
            "files must keep them."
        )
    raise ValueError(
        f"Sample {context.full_id!r} combines files of several KITTI-CARLA frames: "
        + "; ".join(f"{frame} via {', '.join(names)}" for frame, names in sorted(frames.items()))
    )


@selects_per_sample(_select_lidar_pose)
@modality_meta(
    modality_type="lidar_pose",
    dtype="float64",
    hierarchical=True,
    shape="4x4",
    file_formats=[".txt"],
    meta={
        "source": "full_poses_lidar.txt",
        "transform": "lidar_to_world",
        "axes": _VEHICLE_AXES,
        "translation_unit": "meters",
        "rate_hz": 1000,
        "selected_at": "camera frame time, the start of the lidar sweep",
        "matched_by": "frame number in the sample's file names",
    },
)
def lidar_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> LidarPoseTable:
    """Load the lidar trajectory from ``full_poses_lidar.txt``.

    The file holds a row-major ``[R | t]`` and a timestamp per 1 ms
    simulation step. Index it as a town-level hierarchical modality: it is
    parsed once into a :class:`LidarPoseTable` (a few seconds for the 500,000
    poses of a town), and each sample then receives the ``(4, 4)`` float64
    lidar-to-world pose at its camera frame -- the start of its lidar sweep
    -- matched through the frame number in its file names. Relative motion
    between two samples is ``inv(pose_a) @ pose_b``. Calling the loader
    directly returns the whole table.
    """
    data = np.loadtxt(io.BytesIO(_read_bytes(path)), comments="#", dtype=np.float64, ndmin=2)
    if data.shape[1] != 13:
        raise ValueError(
            f"KITTI-CARLA lidar poses {_source_name(path)!r} must have 13 columns "
            f"(a row-major 3x4 [R | t] and a timestamp), found {data.shape[1]}"
        )
    poses = np.zeros((len(data), 4, 4), dtype=np.float64)
    poses[:, :3, :] = data[:, :12].reshape(-1, 3, 4)
    poses[:, 3, 3] = 1.0
    timestamps = np.ascontiguousarray(data[:, 12])
    if np.any(np.diff(timestamps) < 0):
        order = np.argsort(timestamps, kind="stable")
        poses, timestamps = poses[order], timestamps[order]
    return LidarPoseTable(poses, timestamps)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB array or tensor as an 8-bit RGB PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write planar depth in metres in CARLA's 24-bit RGB encoding.

    Depth is clipped to ``[0, 1000]`` m and quantised to the encoding's
    0.06 mm steps; NaN becomes 0 and ``+inf`` the 1000 m far plane.
    :func:`depth` reads the file back.
    """
    ensure_parent(path)
    metres = np.nan_to_num(
        to_hw(value, name="depth").astype(np.float64),
        nan=0.0, posinf=_DEPTH_FAR_PLANE_M, neginf=0.0,
    )
    code = np.rint(np.clip(metres, 0.0, _DEPTH_FAR_PLANE_M) / _DEPTH_SCALE_TO_METERS)
    code = code.astype(np.uint32)
    encoded = np.stack(
        [(code & 255), (code >> 8) & 255, (code >> 16) & 255], axis=-1,
    ).astype(np.uint8)
    save_image(path, Image.fromarray(encoded, mode="RGB"), format="PNG")


@mark_stream_supported
def write_class_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write CARLA semantic tags as an ``images_ss``-style colour PNG."""
    ensure_parent(path)
    labels = to_hw(value, name="class_segmentation")
    if labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"class_segmentation must hold integer tag IDs, got {labels.dtype}")
    if np.issubdtype(labels.dtype, np.floating) and not np.array_equal(labels, np.round(labels)):
        raise ValueError("class_segmentation must hold whole-number tag IDs")
    invalid = (labels < 0) | (labels >= len(CLASS_NAMES))
    if invalid.any():
        found = np.unique(labels[invalid])[:5].tolist()
        raise ValueError(
            f"class_segmentation tag IDs must lie in [0, {len(CLASS_NAMES) - 1}], found {found}"
        )
    arr = PALETTE[labels.astype(np.int64)]
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a class image: sky in the ``skyclass`` colour, the
    rest unlabeled (black), which :func:`sky_mask` reads back."""
    ensure_parent(path)
    mask = to_bool_mask(value)
    arr = np.zeros(mask.shape + (3,), dtype=np.uint8)
    arr[mask] = _sky_color(meta)
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


@mark_stream_supported
def write_lidar_point_cloud(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write ``(N, 7)`` lidar points as a binary PLY in the dataset's layout.

    Columns are those of :func:`lidar_point_cloud`; ``instance`` and
    ``semantic`` must be whole numbers in the uint32 range. The file reads
    back unchanged up to the float32 precision of the stored coordinates.
    """
    points = np.asarray(to_numpy(value), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(_LIDAR_COLUMNS):
        raise ValueError(
            f"KITTI-CARLA lidar points must have shape (N, {len(_LIDAR_COLUMNS)}), got {points.shape}"
        )
    dtype = np.dtype([(name, "<" + _PLY_TYPES[kind]) for name, kind in _LIDAR_PLY_TYPES])
    vertices = np.empty(len(points), dtype=dtype)
    for column, (name, kind) in enumerate(_LIDAR_PLY_TYPES):
        values = points[:, column]
        if kind == "uint32":
            if not np.array_equal(values, np.round(values)) or (
                values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint32).max)
            ):
                raise ValueError(f"KITTI-CARLA lidar {name!r} must hold whole numbers in the uint32 range")
        vertices[name] = values
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
        + [f"property {kind} {name}" for name, kind in _LIDAR_PLY_TYPES]
        + ["end_header", ""]
    ).encode("ascii")
    ensure_parent(path)
    payload = header + vertices.tobytes()
    if isinstance(path, (str, os.PathLike)):
        with open(path, "wb") as stream:
            stream.write(payload)
        return
    path.write(payload)


write_point_cloud = write_lidar_point_cloud
write_sparse_depth = write_lidar_point_cloud


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(3, 3)`` camera matrix as nine numbers with a ``#K`` header.

    KITTI-CARLA has no native intrinsics file; :func:`read_intrinsics` reads
    this one back as the matrix itself.
    """
    K = np.asarray(to_numpy(value), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("intrinsics must be finite")
    ensure_parent(path)
    write_text(path, f"{_INTRINSICS_HEADER}\n{_format_values(K)}\n")


@mark_stream_supported
def write_extrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a transform in the ``lidar_to_cam<N>.txt`` format.

    *value* is read with the ``camera_axes`` and ``transform_direction`` from
    *meta* that :func:`read_extrinsics` uses, and converted back to the
    stored lidar-to-camera transform in vehicle axes, so a loaded matrix
    writes back unchanged.
    """
    transform = np.asarray(to_numpy(value), dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0.0, 0.0, 0.0, 1.0]])
    if transform.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape (4, 4) or (3, 4), got {transform.shape}")
    _check_rigid(transform)
    if _choice(meta, None, "transform_direction", _TRANSFORM_DIRECTIONS) == "camera_to_lidar":
        transform = _invert_rigid(transform)
    if _choice(meta, None, "camera_axes", _CAMERA_AXES) == "optical":
        transform = _OPTICAL_FROM_VEHICLE.T @ transform
    ensure_parent(path)
    write_text(path, f"{_EXTRINSICS_HEADER}\n{_format_values(transform[:3, :])}\n")
