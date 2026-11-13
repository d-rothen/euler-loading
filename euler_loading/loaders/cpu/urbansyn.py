"""CPU-oriented loader functions for UrbanSyn.

UrbanSyn (Gómez et al., Neurocomputing 2025; https://www.urbansyn.org) is a
synthetic urban driving dataset of 7539 path-traced frames, 2048x1024, with
depth and Cityscapes semantic labels. It is released as one zip per modality,
each holding a flat folder::

    rgb/rgb_<NNNN>.png        8-bit sRGB
    depth/depth_<NNNN>.exr    float32 EXR, one channel ``Y``: OctaneRender's Z-depth pass
    ss/ss_<NNNN>.png          Cityscapes train IDs, stored as an RGB PNG with R = G = B
    camera_metadata.json      focal length and sensor size of the one camera

**Depth** is planar (z) depth: OctaneRender's Z-depth pass measures the
distance "parallel to the view vector". It is stored in units of 1e5 meters
(UrbanSyn's site: "The measurement units are 1e5 meters"), so 10 m is stored as
``1e-4``. The pass saturates at ``1.0`` (100 km) where a ray hits nothing; the
loader returns meters with those pixels as ``0`` (invalid). The sky is a finite
backdrop a few hundred meters away; mask it with :func:`sky_mask`.

**Labels** are the 19 Cityscapes training classes (train IDs ``0``-``18``,
sky ``10``) and ``19`` for everything else, the role Cityscapes gives ``255``.

**Intrinsics** follow from the camera metadata: ``fx = focalLength_mm /
sensorWidth_mm * W`` and ``fy = focalLength_mm / sensorHeight_mm * H`` (both
1730.21 px at 2048x1024), with the principal point at the image centre,
``((W - 1) / 2, (H - 1) / 2)`` with pixel centres at integer coordinates: the
renderer's camera has no lens shift.

**Camera poses** are not documented, but every depth EXR carries the render
settings in its header (attribute ``octane``), among them the camera's
position, target and up vector in the scene's world frame (y up, meters).
:func:`camera_pose` turns them into a camera-to-world matrix in optical axes
(x right, y down, z forward). Each of the three districts has its own world.

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** -- ``(H, W)`` float32 planar depth in meters, ``0`` invalid.
- **semantic_segmentation** -- ``(H, W)`` int64 train IDs.
- **sky_mask** -- ``(H, W)`` bool.
- **read_intrinsics** -- ``(3, 3)`` float32 camera matrix.
- **camera_pose** -- ``(4, 4)`` float64 camera-to-world pose.

Readers accept paths and binary streams. Depth and poses need the optional
``OpenEXR`` dependency (``pip install 'euler-loading[urbansyn]'``).
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._sky import resolve_skyclass, skyclass_label
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    save_image,
    to_bool_mask,
    to_hw,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_json,
)

__all__ = [
    "rgb",
    "depth",
    "semantic_segmentation",
    "sky_mask",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_semantic_segmentation",
    "write_sky_mask",
    "write_intrinsics",
    "write_camera_pose",
]

#: Meters per stored depth unit.
DEPTH_SCALE_TO_METERS = 100000.0
#: The stored value of a ray that hits nothing (the Z-depth pass's maximum).
DEPTH_NO_HIT = 1.0
#: The release's image grid, (height, width).
IMAGE_SIZE = (1024, 2048)
SKY_TRAIN_ID = 10
#: The label of every pixel outside the 19 Cityscapes training classes.
OTHER_ID = 19
CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

_DEPTH_META = {
    "radial_depth": False,
    "scale_to_meters": DEPTH_SCALE_TO_METERS,
    "channel": "Y",
    "no_hit_value": DEPTH_NO_HIT,
    "invalid": {"non_finite": False, "sentinel": 0},
}
_LABEL_META = {
    "encoding": "grey_rgb",
    "label_ids": "cityscapes_train_ids",
    "classes": CLASSES,
    "other_id": OTHER_ID,
    "skyclass": [SKY_TRAIN_ID] * 3,
}
_SKY_LOADER = "urbansyn sky_mask"
#: The EXR header attribute holding OctaneRender's render settings as JSON.
OCTANE_ATTRIBUTE = "octane"
_POSE_META = {
    "source_frame": "camera_optical",
    "target_frame": "world",
    "direction": "target_from_source",
    "camera_axes": "x right, y down, z forward",
    "world_axes": "y up, per district",
    "translation_unit": "meters",
    "stored": "camera pos, target and up in the depth EXR header (attribute octane)",
}


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _openexr():
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "UrbanSyn depth and poses (EXR) need OpenEXR. Install it with pip install 'euler-loading[urbansyn]'."
        ) from exc
    return Imath, OpenEXR


def _load_labels(path: Union[str, BinaryIO]) -> np.ndarray:
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        if values.shape[2] not in (3, 4) or not (
            np.array_equal(values[..., 0], values[..., 1]) and np.array_equal(values[..., 0], values[..., 2])
        ):
            raise ValueError(
                f"{_source_name(path)!r} is not an UrbanSyn label map (R = G = B = train ID); "
                "the colour-coded ss_color files are not label maps"
            )
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{_source_name(path)!r} has shape {values.shape}, expected a label image")
    return values.astype(np.int64)


def _sky_label(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None) -> int:
    color = resolve_skyclass(meta, attributes, loader=_SKY_LOADER, default=(SKY_TRAIN_ID,) * 3)
    return skyclass_label(color, loader=_SKY_LOADER)


def _image_size(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None) -> tuple[int, int]:
    """``(height, width)`` from ``image_size`` in the attributes or meta, else the release's."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get("image_size") is not None:
            value = source["image_size"]
            if isinstance(value, Mapping):
                value = (value.get("height"), value.get("width"))
            height, width = (int(v) for v in value)
            return height, width
    return IMAGE_SIZE


def _camera_block(document: Any) -> Mapping[str, Any]:
    try:
        camera = document["parameters"][0]["Camera"]
    except (KeyError, IndexError, TypeError):
        camera = None
    if not isinstance(camera, Mapping):
        raise ValueError("UrbanSyn camera metadata needs parameters[0].Camera or a 'K' matrix")
    return camera


def _vector(block: Any, name: str) -> np.ndarray:
    try:
        values = [float(block[axis]) for axis in "xyz"] if isinstance(block, Mapping) else [float(v) for v in block]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"camera {name!r} must be {{x, y, z}} or three numbers, got {block!r}") from exc
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"camera {name!r} must be three finite numbers, got {block!r}")
    return np.array(values, dtype=np.float64)


def _camera_from_file(path: Union[str, BinaryIO]) -> Mapping[str, Any]:
    """The camera block (``pos``, ``target``, ``up``) of a depth EXR header or a pose JSON."""
    name = _source_name(path)
    if name.lower().endswith(".json"):
        document = json.loads(_read_bytes(path).decode("utf-8"))
        camera = document.get("camera") if isinstance(document, Mapping) else None
    else:
        _imath, OpenEXR = _openexr()
        source = os.fspath(path) if isinstance(path, (str, os.PathLike)) else io.BytesIO(_read_bytes(path))
        exr = OpenEXR.InputFile(source)
        try:
            raw = exr.header().get(OCTANE_ATTRIBUTE)
        finally:
            exr.close()
        if raw is None:
            raise ValueError(f"{name!r} has no {OCTANE_ATTRIBUTE!r} header attribute with the render camera")
        document = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        camera = (document.get("renderTarget") or {}).get("camera") if isinstance(document, Mapping) else None
    if not isinstance(camera, Mapping) or not all(key in camera for key in ("pos", "target", "up")):
        raise ValueError(f"{name!r} holds no camera with pos, target and up")
    return camera


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a frame (2048x1024 8-bit sRGB PNG) as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".exr"],
    output_unit="meters",
    meta=_DEPTH_META,
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a Z-depth EXR as ``(H, W)`` float32 planar depth in meters, ``0`` invalid.

    The stored values (channel ``Y``, or the only channel) are multiplied by
    1e5. Pixels that hit nothing (stored ``1.0``), non-finite and non-positive
    values are returned as ``0``.
    """
    Imath, OpenEXR = _openexr()
    source = os.fspath(path) if isinstance(path, (str, os.PathLike)) else io.BytesIO(_read_bytes(path))
    exr = OpenEXR.InputFile(source)
    try:
        header = exr.header()
        channels = list(header["channels"])
        channel = "Y" if "Y" in channels else "Z" if "Z" in channels else channels[0] if len(channels) == 1 else None
        if channel is None:
            raise ValueError(f"{_source_name(path)!r} has channels {channels}, expected Y")
        window = header["dataWindow"]
        width, height = window.max.x - window.min.x + 1, window.max.y - window.min.y + 1
        raw = np.frombuffer(exr.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
    finally:
        exr.close()
    raw = raw.reshape(height, width)
    valid = np.isfinite(raw) & (raw > 0) & (raw < DEPTH_NO_HIT)
    return np.where(valid, raw * np.float32(DEPTH_SCALE_TO_METERS), np.float32(0.0)).astype(np.float32)


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta=_LABEL_META,
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an ``ss`` label map as ``(H, W)`` int64 Cityscapes train IDs.

    ``0``-``18`` are the Cityscapes training classes and ``19`` everything
    else. The release stores them as RGB PNGs with equal channels; a
    single-channel PNG is read as well. Colour-coded maps raise.
    """
    return _load_labels(path)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"skyclass": [SKY_TRAIN_ID] * 3},
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return ``(H, W)`` bool, true where an ``ss`` label map holds the sky.

    The label is ``skyclass`` (a grey ``[v, v, v]``) from the file's
    attributes or the dataset head's meta, train ID ``10`` by default.
    """
    return _load_labels(path) == _sky_label(meta, attributes)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"camera_model": "pinhole", "principal_point": "image centre, (W - 1) / 2 and (H - 1) / 2"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(3, 3)`` float32 camera matrix from ``camera_metadata.json``.

    ``fx = focalLength_mm / sensorWidth_mm * W``, ``fy = focalLength_mm /
    sensorHeight_mm * H``, principal point ``((W - 1) / 2, (H - 1) / 2)``, for
    the release's 2048x1024 grid or ``image_size`` (``[height, width]``) from
    the attributes or meta. A file :func:`write_intrinsics` wrote holds the
    matrix itself (``K``), which is returned as stored.
    """
    document = json.loads(_read_bytes(path).decode("utf-8"))
    if isinstance(document, Mapping) and document.get("K") is not None:
        matrix = np.asarray(document["K"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"'K' in {_source_name(path)!r} is not a finite 3x3 matrix")
        return matrix.astype(np.float32)
    camera = _camera_block(document)
    height, width = _image_size(meta, attributes)
    focal = float(camera["focalLength_mm"])
    fx = focal / float(camera["sensorWidth_mm"]) * width
    fy = focal / float(camera["sensorHeight_mm"]) * height
    return np.array(
        [[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".exr", ".json"],
    meta=_POSE_META,
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(4, 4)`` float64 camera-to-world pose of a frame.

    Reads the camera's ``pos``, ``target`` and ``up`` from the ``octane``
    attribute of a depth EXR's header (only the header is read from a path),
    or from a JSON file :func:`write_camera_pose` wrote. The columns of the
    rotation are the camera's x (right), y (down) and z (forward, towards the
    target) axes in the world frame of the frame's district (y up); the
    translation is the camera position in meters. ``inv(pose_a) @ pose_b``
    is the motion between two frames of one district.
    """
    camera = _camera_from_file(path)
    position, target, up = (_vector(camera[key], key) for key in ("pos", "target", "up"))
    forward = target - position
    if not np.linalg.norm(forward) > 0:
        raise ValueError(f"{_source_name(path)!r}: the camera target equals its position")
    forward /= np.linalg.norm(forward)
    up = up - (up @ forward) * forward
    if not np.linalg.norm(up) > 1e-9:
        raise ValueError(f"{_source_name(path)!r}: the camera's up vector is parallel to its view")
    down = -up / np.linalg.norm(up)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([np.cross(down, forward), down, forward], axis=1)
    pose[:3, 3] = position
    return pose


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as an 8-bit PNG, the format :func:`rgb` reads.

    Accepts ``(H, W, 3)`` or ``(3, H, W)`` values in ``[0, 1]``.
    """
    if os.path.splitext(get_target_name(path))[1].lower() not in (".png", ""):
        raise ValueError(f"UrbanSyn images are PNG, not {get_target_name(path)!r}")
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    save_image(path, image, format="PNG")


def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write planar depth in meters as the release does: a float32 EXR channel ``Y`` in 1e5 m.

    Invalid pixels (``0``, negative, non-finite) are stored as ``1.0``, the
    value of a ray that hits nothing, so :func:`depth` returns them as ``0``
    again. The EXR is ZIP-compressed (the release's are not), which does not
    change the values. OpenEXR writes through a filename, so the target must
    be a path; dataset writers hand it a temporary file automatically.
    """
    Imath, OpenEXR = _openexr()
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("UrbanSyn write_depth needs a filesystem path: OpenEXR cannot write to a stream")
    meters = np.asarray(to_hw(value, name="depth"), dtype=np.float64)
    if not meters.size:
        raise ValueError(f"cannot write an empty depth map, got shape {meters.shape}")
    valid = np.isfinite(meters) & (meters > 0) & (meters < DEPTH_SCALE_TO_METERS)
    stored = np.where(valid, meters / DEPTH_SCALE_TO_METERS, DEPTH_NO_HIT).astype(np.float32)
    ensure_parent(path)
    height, width = stored.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    output = OpenEXR.OutputFile(os.fspath(path), header)
    try:
        output.writePixels({"Y": np.ascontiguousarray(stored).tobytes()})
    finally:
        output.close()


def _integer_labels(value: Any, *, name: str) -> np.ndarray:
    labels = to_numpy(to_hw(value, name=name))
    if labels.dtype == np.bool_:
        labels = labels.astype(np.int64)
    elif not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.array_equal(labels, np.rint(labels)):
            raise ValueError(f"UrbanSyn {name} must hold integer labels")
    labels = labels.astype(np.int64)
    if labels.size and (labels.min() < 0 or labels.max() > 255):
        raise ValueError(f"UrbanSyn {name} labels must lie in 0..255, got {labels.min()}..{labels.max()}")
    return labels


def _save_labels(path: Union[str, BinaryIO], labels: np.ndarray) -> None:
    """Store labels as the release does: an 8-bit RGB PNG with R = G = B."""
    grey = labels.astype(np.uint8)
    save_image(path, Image.fromarray(np.repeat(grey[..., None], 3, axis=2), mode="RGB"), format="PNG")


@mark_stream_supported
def write_semantic_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write ``(H, W)`` or ``(1, H, W)`` train IDs (0..255) as the release's RGB label PNG."""
    _save_labels(path, _integer_labels(value, name="semantic_segmentation"))


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a label PNG that :func:`sky_mask` reads back with the same meta.

    Sky pixels get the sky label (``skyclass``, train ID 10 by default), all
    others ``19`` (or ``meta["non_sky_class_id"]``). The file holds only sky
    and not sky, not the other classes.
    """
    mask = to_bool_mask(value, name="sky_mask")
    sky = _sky_label(meta, None)
    other = int((meta or {}).get("non_sky_class_id", OTHER_ID if sky != OTHER_ID else 0))
    if other == sky or not 0 <= other <= 255:
        raise ValueError(f"UrbanSyn 'non_sky_class_id' {other} must differ from the sky label {sky} and fit 0..255")
    _save_labels(path, np.where(mask, sky, other))


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(3, 3)`` camera matrix as JSON (``{"K": ...}``) that :func:`read_intrinsics` returns as is.

    The release describes its camera physically (focal length and sensor
    size), which a cropped or resized grid no longer matches, so the matrix
    is stored instead. ``meta["image_size"]`` (``[height, width]``) is
    recorded alongside when given.
    """
    matrix = to_numpy(value).astype(np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"intrinsics must be a finite (3, 3) matrix, got shape {matrix.shape}")
    document: dict[str, Any] = {"K": [[float(v) for v in row] for row in matrix]}
    if isinstance(meta, Mapping) and meta.get("image_size") is not None:
        document["image_size"] = [int(v) for v in _image_size(meta, None)]
    write_json(path, document, indent=2)


@mark_stream_supported
def write_camera_pose(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(4, 4)`` camera-to-world pose as JSON that :func:`camera_pose` reads back.

    The release keeps poses in the depth EXR headers; a written pose is a
    small JSON file instead, ``{"camera": {"pos", "target", "up"}}`` in the
    renderer's terms: the target lies 1 m along the camera's z axis and up is
    its -y axis. The rotation must be proper and orthonormal.
    """
    pose = to_numpy(value).astype(np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)) or not np.allclose(pose[3], (0, 0, 0, 1)):
        raise ValueError(f"camera_pose must be a finite (4, 4) rigid transform, got shape {pose.shape}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ValueError("camera_pose rotation must be orthonormal and right-handed")
    position = pose[:3, 3]

    def xyz(vector: np.ndarray) -> dict[str, float]:
        return {axis: float(v) for axis, v in zip("xyz", vector)}

    write_json(path, {"camera": {"pos": xyz(position), "target": xyz(position + rotation[:, 2]),
                                 "up": xyz(-rotation[:, 1])}}, indent=2)
