"""CPU-oriented loader functions for SHIFT, as prepared by euler-known-datasets.

SHIFT (Sun et al., CVPR 2022, https://www.vis.xyz/shift) is a synthetic driving
dataset rendered with CARLA. The ``euler-known-datasets`` recipe ``shift``
indexes the front camera of its discrete set in shift-dev's image-rate layout
and writes one calibration file per sequence::

    discrete/images/<split>/front/img/<seq>/<frame>_img_front.{jpg,png}   rgb
    discrete/images/<split>/front/depth/<seq>/<frame>_depth_front.png     depth
    discrete/images/<split>/front/semseg/<seq>/<frame>_semseg_front.png   labels
    calibration/<split>/<seq>.json                                         K, poses

Images are 1280x800. The ego vehicle's hood covers an arc over the bottom
rows (from row 770 at the centre), 1.0-1.2 m from the camera.

**Depth** is planar (z-buffer) depth in CARLA's 24-bit code:
``(R + 256 G + 65536 B) / (2**24 - 1) * 1000`` m. The sky is stored as code
16777210 (999.9997 m), as are a few distant pixels beyond the far plane;
there are no invalid pixels.

**Labels** carry the class ID in the red channel (green = blue = 0). The
released PNGs hold CARLA's 23 tags (sky ``13``). Frames decoded from SHIFT's
segmentation videos hold a merged set, in which the sky (13) and four other
tags are exact and nine pairs of tags share an ID (23-31); the head's
``classes`` names them. :func:`sky_mask` works for both.

**Calibration file** (format ``euler_known_datasets.shift.camera_frames``,
version 1): ``camera.intrinsics`` (3x3) and ``frames[<frame>].camera_to_world``
(4x4, optical axes x right, y down, z forward), in CARLA's world with y
negated (right-handed, z up, meters).

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** -- ``(H, W)`` float32 planar depth in meters.
- **radial_distance** -- ``(H, W)`` float32 Euclidean distance to the camera
  centre in meters: depth times ``||K^-1 [x, y, 1]||``.
- **semantic_segmentation** -- ``(H, W)`` int64 class IDs.
- **sky_mask** -- ``(H, W)`` bool.
- **read_intrinsics** -- ``(3, 3)`` float32 (hierarchical).
- **camera_pose** -- ``(4, 4)`` float32 camera-to-world pose of the sample's
  frame (hierarchical, selected per sample).

Readers accept paths and binary streams.
"""

from __future__ import annotations

import functools
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
from euler_loading.loaders._sky import RGB, resolve_skyclass
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    pixel_grid_transform,
    replace_json,
    save_image,
    to_bool_mask,
    to_hw,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_json,
)

__all__ = [
    "FORMAT_NAME",
    "CameraFrames",
    "rgb",
    "depth",
    "radial_distance",
    "semantic_segmentation",
    "sky_mask",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_radial_distance",
    "write_semantic_segmentation",
    "write_sky_mask",
    "write_intrinsics",
]

#: The calibration file format these loaders read.
FORMAT_NAME = "euler_known_datasets.shift.camera_frames"
SUPPORTED_FORMAT_VERSIONS = (1,)
IMAGE_SIZE = (800, 1280)
#: The front camera: a 90 degree pinhole on the 1280x800 grid (det_2d.json, every frame).
INTRINSICS = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 400.0], [0.0, 0.0, 1.0]])
DEPTH_FAR_PLANE_M = 1000.0
DEPTH_MAX_CODE = 2**24 - 1
DEPTH_SCALE_TO_METERS = DEPTH_FAR_PLANE_M / DEPTH_MAX_CODE
SKY_ID = 13
_SKY_LOADER = "shift sky_mask"
_CAMERA_AXES = "optical: x right, y down, z forward"


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _load_rgb_uint8(path: Union[str, BinaryIO]) -> np.ndarray:
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _option(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None, key: str) -> Any:
    """*key* from the file's attributes, else the head's meta."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source[key]
    return None


def _camera_matrix(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None,
                   shape: tuple[int, ...], source: str) -> np.ndarray:
    """``K`` from ``intrinsics`` in the attributes or meta, else the release's for its grid."""
    value = _option(meta, attributes, "intrinsics")
    if value is not None:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"SHIFT {source!r}: 'intrinsics' must be a finite 3x3 matrix")
        return matrix
    if tuple(shape[:2]) != IMAGE_SIZE:
        raise ValueError(
            f"SHIFT {source!r} is {shape[1]}x{shape[0]}, not the release's 1280x800; pass its camera "
            "matrix as 'intrinsics' in the file's attributes or the head's meta"
        )
    return INTRINSICS


@functools.lru_cache(maxsize=16)
def _ray_lengths(matrix: tuple[float, ...], height: int, width: int) -> np.ndarray:
    """``||K^-1 [x, y, 1]||`` for every pixel centre: distance / planar depth."""
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    rays = inverse[:, 0, None, None] * xs + inverse[:, 1, None, None] * ys + inverse[:, 2, None, None]
    rays /= rays[2]
    lengths = np.sqrt((rays**2).sum(axis=0)).astype(np.float32)
    lengths.setflags(write=False)
    return lengths


def _ray_lengths_for(matrix: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return _ray_lengths(tuple(float(v) for v in matrix.ravel()), int(shape[0]), int(shape[1]))


def _decode_depth(rgb_uint8: np.ndarray) -> np.ndarray:
    code = (
        rgb_uint8[..., 0].astype(np.uint32)
        | (rgb_uint8[..., 1].astype(np.uint32) << 8)
        | (rgb_uint8[..., 2].astype(np.uint32) << 16)
    )
    return (code * DEPTH_SCALE_TO_METERS).astype(np.float32)


def _encode_depth(path: Union[str, BinaryIO], meters: np.ndarray) -> None:
    meters = np.nan_to_num(np.asarray(meters, dtype=np.float64), nan=0.0, posinf=DEPTH_FAR_PLANE_M, neginf=0.0)
    code = np.rint(np.clip(meters, 0.0, DEPTH_FAR_PLANE_M) / DEPTH_SCALE_TO_METERS).astype(np.uint32)
    encoded = np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], axis=-1).astype(np.uint8)
    save_image(path, Image.fromarray(encoded, mode="RGB"), format="PNG")


def _load_labels(path: Union[str, BinaryIO]) -> np.ndarray:
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{_source_name(path)!r} has shape {values.shape}, expected a label image")
    return values.astype(np.int64)


def _sky_id(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None) -> int:
    color: RGB = resolve_skyclass(meta, attributes, loader=_SKY_LOADER, default=(SKY_ID, 0, 0))
    if tuple(color[1:]) != (0, 0):
        raise ValueError(f"SHIFT stores class IDs in the red channel, so 'skyclass' must be [id, 0, 0]; "
                         f"got {list(color)}.")
    return int(color[0])


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------


def _read_document(path: Union[str, BinaryIO]) -> dict[str, Any]:
    source = _source_name(path)
    try:
        document = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"SHIFT calibration {source!r} is not JSON: {exc}") from exc
    file_format = document.get("format") if isinstance(document, dict) else None
    if not isinstance(file_format, Mapping) or file_format.get("name") != FORMAT_NAME:
        raise ValueError(f"{source!r} is not a SHIFT calibration file ({FORMAT_NAME}); prepare the "
                         "dataset with euler-known-datasets")
    if file_format.get("version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"SHIFT calibration {source!r} has format version {file_format.get('version')!r}; "
                         f"this euler-loading reads {', '.join(map(str, SUPPORTED_FORMAT_VERSIONS))}")
    return document


def _matrix(value: Any, shape: tuple[int, int], what: str, source: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"SHIFT {source!r}: {what} must be a finite {shape} matrix")
    return array


class CameraFrames:
    """A parsed calibration file: the front camera of one sequence.

    Attributes:
        sequence: The sequence name, e.g. ``"0016-1b62"``.
        intrinsics: ``(3, 3)`` float64 camera matrix.
        frames: The file's frame records, keyed by frame ID (``"00000040"``).
    """

    def __init__(self, document: Mapping[str, Any], *, source: str = "<calibration>"):
        self.source = source
        self.sequence = str(document.get("sequence", ""))
        camera = document.get("camera") or {}
        self.intrinsics = _matrix(camera.get("intrinsics"), (3, 3), "camera.intrinsics", source)
        self.frames: Mapping[str, Mapping[str, Any]] = document.get("frames") or {}

    def camera_to_world(self, frame_id: str) -> np.ndarray:
        """The ``(4, 4)`` float64 pose of the frame's optical frame, in meters."""
        record = self.frames.get(frame_id)
        if record is None:
            raise KeyError(f"SHIFT calibration {self.source!r} ({self.sequence}) has no frame {frame_id!r}; "
                           "the sample ID must be the frame number")
        return _matrix(record["camera_to_world"], (4, 4), f"frames[{frame_id}].camera_to_world", self.source)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png", ".jpg"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a frame (PNG or the release's JPEG) as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    return _load_rgb_uint8(path).astype(np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, DEPTH_FAR_PLANE_M],
    output_unit="meters",
    meta={
        "encoding": "carla_rgb24",
        "decode": "(R + 256 * G + 65536 * B) / (2**24 - 1) * 1000",
        "raw_range": [0, DEPTH_MAX_CODE],
        "scale_to_meters": DEPTH_SCALE_TO_METERS,
        "radial_depth": False,
        "far_plane_m": DEPTH_FAR_PLANE_M,
    },
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a depth PNG as ``(H, W)`` float32 planar depth in meters.

    ``(R + 256 G + 65536 B) / (2**24 - 1) * 1000``, about 0.06 mm per step.
    The sky reads 999.9997 m (code 16777210), as do a few distant non-sky
    pixels beyond the far plane; mask the sky with :func:`sky_mask`.
    """
    return _decode_depth(_load_rgb_uint8(path))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_unit="meters",
    meta={"radial_depth": True, "scale_to_meters": 1.0, "stored": "planar depth, carla_rgb24"},
)
def radial_distance(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a depth PNG as ``(H, W)`` float32 Euclidean distance to the camera centre in meters.

    The planar depth is multiplied by ``||K^-1 [x, y, 1]||`` (pixel centres at
    integer coordinates). ``K`` is ``intrinsics`` from the file's attributes or
    the head's meta, else the release's camera for a 1280x800 map.
    """
    planar = depth(path)
    matrix = _camera_matrix(meta, attributes, planar.shape, _source_name(path))
    return planar * _ray_lengths_for(matrix, planar.shape)


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={"encoding": "id_in_red", "skyclass": [SKY_ID, 0, 0]},
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a label PNG as ``(H, W)`` int64 class IDs, read from the red channel.

    The release's labels are CARLA's 23 tags; frames decoded from the
    segmentation videos use a merged set (see the module docstring and the
    head's ``classes``). A single-channel PNG is read as well.
    """
    return _load_labels(path)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"skyclass": [SKY_ID, 0, 0]},
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return ``(H, W)`` bool, true where a label PNG holds the sky.

    The sky is ``skyclass`` (``[id, 0, 0]``, the ID in the red channel) from
    the file's attributes or the dataset head's meta, ``[13, 0, 0]`` by
    default; it is exact in both label sets.
    """
    return _load_labels(path) == _sky_id(meta, attributes)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"model": "pinhole", "distortion": None, "source": "calibration/<split>/<sequence>.json"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the sequence's ``(3, 3)`` float32 camera matrix (fx = fy = 640, centre (640, 400) as released)."""
    return CameraFrames(_read_document(path), source=_source_name(path)).intrinsics.astype(np.float32)


def _select_pose(table: CameraFrames, context: SampleContext) -> np.ndarray:
    return table.camera_to_world(context.id).astype(np.float32)


@selects_per_sample(_select_pose)
@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "transform": "camera_to_world",
        "source_frame": "camera_optical",
        "target_frame": "world",
        "camera_axes": _CAMERA_AXES,
        "world": "CARLA's world with y negated (right-handed, z up)",
        "translation_unit": "meters",
        "matched_by": "sample ID (the frame number)",
    },
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> CameraFrames:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32
    camera-to-world pose in meters.

    ``inv(pose_a) @ pose_b`` is the motion between two frames of a sequence.
    Calling the loader directly returns the whole :class:`CameraFrames`.
    """
    return CameraFrames(_read_document(path), source=_source_name(path))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image in ``[0, 1]`` as 8-bit PNG, or JPEG (quality 95) for a ``.jpg`` target."""
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    ext = os.path.splitext(get_target_name(path))[1].lower()
    if ext in (".jpg", ".jpeg"):
        ensure_parent(path)
        image.save(path, format="JPEG", quality=95)
        return
    if ext not in (".png", ""):
        raise ValueError(f"SHIFT images are PNG or JPEG, not {get_target_name(path)!r}")
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write planar depth in meters in CARLA's 24-bit RGB PNG code, which :func:`depth` reads.

    Depth is clipped to ``[0, 1000]`` m and quantised to 0.06 mm steps; NaN
    becomes 0 and ``+inf`` the 1000 m far plane.
    """
    _encode_depth(path, to_hw(value, name="depth"))


@mark_stream_supported
def write_radial_distance(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write Euclidean distances in meters as the planar depth PNG :func:`radial_distance` reads back.

    ``K`` is ``meta["intrinsics"]``, else the release's camera for a 1280x800 map.
    """
    distance = np.asarray(to_hw(value, name="radial_distance"), dtype=np.float64)
    matrix = _camera_matrix(meta, None, distance.shape, get_target_name(path))
    _encode_depth(path, distance / _ray_lengths_for(matrix, distance.shape))


def _integer_labels(value: Any, *, name: str) -> np.ndarray:
    labels = to_numpy(to_hw(value, name=name))
    if labels.dtype == np.bool_:
        labels = labels.astype(np.int64)
    elif not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.array_equal(labels, np.rint(labels)):
            raise ValueError(f"SHIFT {name} must hold integer labels")
    labels = labels.astype(np.int64)
    if labels.size and (labels.min() < 0 or labels.max() > 255):
        raise ValueError(f"SHIFT {name} labels must lie in 0..255, got {labels.min()}..{labels.max()}")
    return labels


def _save_labels(path: Union[str, BinaryIO], labels: np.ndarray) -> None:
    """Store labels as the release does: an RGB PNG with the ID in red, green = blue = 0."""
    rgb_labels = np.zeros(labels.shape + (3,), dtype=np.uint8)
    rgb_labels[..., 0] = labels.astype(np.uint8)
    save_image(path, Image.fromarray(rgb_labels, mode="RGB"), format="PNG")


@mark_stream_supported
def write_semantic_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write ``(H, W)`` or ``(1, H, W)`` class IDs (0..255) as a label PNG, the ID in red."""
    _save_labels(path, _integer_labels(value, name="semantic_segmentation"))


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a label PNG that :func:`sky_mask` reads back with the same meta.

    Sky pixels get the sky ID (``skyclass``, 13 by default), the others 0
    (``unlabeled``, or ``meta["non_sky_class_id"]``).
    """
    mask = to_bool_mask(value, name="sky_mask")
    sky = _sky_id(meta, None)
    other = int((meta or {}).get("non_sky_class_id", 0 if sky != 0 else 1))
    if other == sky or not 0 <= other <= 255:
        raise ValueError(f"SHIFT 'non_sky_class_id' {other} must differ from the sky ID {sky} and fit 0..255")
    _save_labels(path, np.where(mask, sky, other))


def _image_size(meta: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(meta, Mapping) or meta.get("width") is None or meta.get("height") is None:
        return None
    return int(meta["width"]), int(meta["height"])


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a camera matrix into a SHIFT calibration file.

    An existing file is updated, not replaced: the change from its stored
    matrix must be a crop or a rescale, ``meta["width"]`` and
    ``meta["height"]`` update the recorded image size, and the poses stay as
    they are. An unchanged matrix leaves the file byte-identical. A new file,
    or a stream target, receives a calibration file with only the camera.
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
            raise ValueError(f"Refusing to update {name!r}: the new intrinsics are not a crop or rescale "
                             f"of the stored matrix ({stored.tolist()} -> {matrix.tolist()}).")
        camera = document["camera"]
        updated = dict(camera, intrinsics=(transform @ stored).tolist())
        if size is not None:
            updated["width"], updated["height"] = size
        if updated == camera:
            return
        document["camera"] = updated
        replace_json(path, document)
        return
    camera: dict[str, Any] = {"name": "front", "intrinsics": matrix.tolist()}
    if size is not None:
        camera["width"], camera["height"] = size
    write_json(path, {"format": {"name": FORMAT_NAME, "version": 1}, "camera": camera, "frames": {}})
