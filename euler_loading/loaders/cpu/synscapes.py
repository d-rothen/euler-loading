"""CPU-oriented loader functions for the Synscapes dataset.

Native files live under ``img/rgb``, ``img/depth``, ``img/class``,
``img/instance`` and ``meta``. RGB returns an ``(H, W, 3)`` float32 array in
``[0, 1]``; depth returns ``(H, W)`` float32 planar depth in metres. Class
and instance labels return ``(H, W)`` int64 arrays, and the sky mask is bool.
Intrinsics return a ``(3, 3)`` float32 camera matrix at the resolution recorded
in the per-image JSON (1440x720 for the native dataset), and extrinsics a
``(4, 4)`` float32 rigid transform built from the recorded camera pose.

Only EXR depth loading and writing require the optional ``OpenEXR``
dependency, installed with ``pip install 'euler-loading[synscapes]'``. All
readers accept paths or binary streams, including buffers supplied by
zip-backed modalities; writers accept paths, and all but ``write_depth``
also accept binary streams.

Format references:
https://synscapes.on.liu.se/features.html
https://github.com/MartinHahner/FoggySynscapes/blob/main/source/Depth_processing/exr_to_mat.py
"""

from __future__ import annotations

import json
import os
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
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
    "class_segmentation",
    "instance_segmentation",
    "sky_mask",
    "read_intrinsics",
    "read_extrinsics",
    "write_rgb",
    "write_depth",
    "write_class_segmentation",
    "write_instance_segmentation",
    "write_sky_mask",
    "write_intrinsics",
    "write_extrinsics",
]

_SKY_CLASS_ID = 23
_MAX_INSTANCE_ID = 256**3 - 1

# The Synscapes paper defines the ego frame through its 3D bounding boxes:
# "the x vector facing forward, the y vector to the left, and the z vector
# facing up".  The dataset records the camera pose in that frame but does not
# document the order in which pitch/roll/yaw compose, so this module applies
# the usual automotive yaw-pitch-roll order and states that in the modality
# metadata rather than leaving it implicit.
_EGO_FRAME = "x_forward_y_left_z_up"
_ROTATION_ORDER = "Rz(yaw) @ Ry(pitch) @ Rx(roll)"

# A pinhole camera used with ``read_intrinsics`` expects x right, y down and
# z along the optical axis, which is a fixed permutation of the vehicle axes.
_OPTICAL_FROM_VEHICLE = np.array(
    [[0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float64,
)

_TRANSFORM_DIRECTIONS = ("camera_to_ego", "ego_to_camera")
_CAMERA_AXES = ("vehicle", "optical")


def _load_json(path: Union[str, BinaryIO]) -> Any:
    """Load JSON from a file path or an in-memory buffer."""
    if isinstance(path, (str, os.PathLike)):
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    return json.load(path)


def _camera_section(data: Any, *names: str) -> dict[str, Any]:
    """Return the first present ``camera.<name>`` block.

    Synscapes writes ``camera.intrinsic`` and ``camera.extrinsic``; the
    plural spellings are accepted so metadata copied from other tools still
    loads.
    """
    camera = data.get("camera") if isinstance(data, dict) else None
    if not isinstance(camera, dict):
        raise ValueError("Synscapes metadata must contain a 'camera' object")
    for name in names:
        section = camera.get(name)
        if isinstance(section, dict):
            return section
    raise ValueError(
        f"Synscapes camera metadata must contain {' or '.join(repr(n) for n in names)}; "
        f"found: {', '.join(sorted(camera)) or 'nothing'}"
    )


def _required_floats(section: dict[str, Any], keys: tuple[str, ...], label: str) -> list[float]:
    """Return *keys* from *section* as floats, naming any that are missing."""
    def is_number(value: Any) -> bool:
        # bool is an int subclass, and a flag is not a camera parameter.
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    missing = [key for key in keys if not is_number(section.get(key))]
    if missing:
        raise ValueError(
            f"Synscapes camera.{label} is missing numeric "
            f"{', '.join(repr(key) for key in missing)}"
        )
    return [float(section[key]) for key in keys]


def _rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Compose ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` as a ``(3, 3)`` matrix."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Recover ``(yaw, pitch, roll)`` from a ``Rz @ Ry @ Rx`` rotation."""
    # Clip guards against a rotation that is a hair outside [-1, 1] after
    # round-tripping through float32 storage.
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    if np.isclose(abs(R[2, 0]), 1.0):
        # Gimbal lock: yaw and roll are no longer separable, so attribute the
        # whole remaining rotation to roll and leave yaw at zero. The sign of
        # the first term follows sin(pitch), which is -R[2, 0].
        sign = 1.0 if R[2, 0] < 0 else -1.0
        return 0.0, pitch, float(np.arctan2(sign * R[0, 1], R[1, 1]))
    return (
        float(np.arctan2(R[1, 0], R[0, 0])),
        pitch,
        float(np.arctan2(R[2, 1], R[2, 2])),
    )


def _select_option(
    meta: dict[str, Any] | None,
    attributes: dict[str, Any] | None,
    key: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    """Return a validated string option from *attributes*, *meta*, or *default*."""
    for source in (attributes, meta):
        if not isinstance(source, dict):
            continue
        value = source.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or value.lower() not in allowed:
            raise ValueError(
                f"Synscapes {key!r} must be one of "
                f"{', '.join(repr(option) for option in allowed)}, got {value!r}"
            )
        return value.lower()
    return default


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
def rgb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a Synscapes RGB PNG as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".exr"],
    output_unit="meters",
    meta={"channel": "Z", "radial_depth": False, "scale_to_meters": 1.0},
)
def depth(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load the EXR ``Z`` channel as ``(H, W)`` float32 planar depth in metres.

    Values are already metric and are preserved, including non-finite values.
    The EXR data window determines the shape; no resizing or sky masking is
    applied. Requires ``pip install 'euler-loading[synscapes]'``.
    """
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "Synscapes EXR depth loading requires OpenEXR. "
            "Install it with pip install 'euler-loading[synscapes]'."
        ) from exc

    source = os.fspath(path) if isinstance(path, (str, os.PathLike)) else path
    # InputFile supports binary streams; OpenEXR.File only accepts filenames.
    exr = OpenEXR.InputFile(source)
    try:
        header = exr.header()
        if "Z" not in header["channels"]:
            raise ValueError("Synscapes depth EXR must contain a 'Z' channel")
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        pixels = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(pixels, dtype=np.float32).reshape(height, width).copy()
    finally:
        exr.close()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={
        "encoding": "single_channel",
        "label_ids": "cityscapes",
        "sky_class_id": _SKY_CLASS_ID,
    },
)
def class_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load ``img/class`` as ``(H, W)`` int64 Cityscapes label IDs.

    Original label IDs (including void labels) are preserved without mapping
    to training IDs. Palette indices are preserved for indexed PNGs.
    """
    with Image.open(path) as image:
        labels = np.array(image, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("Synscapes class segmentation must be a single-channel PNG")
    return labels


@modality_meta(
    modality_type="instance_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={"encoding": "rgb", "id_formula": "R + 256 * G + 65536 * B"},
)
def instance_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Decode ``img/instance`` RGB PNGs to ``(H, W)`` int64 instance IDs.

    Synscapes packs each ID as ``R + 256 * G + 65536 * B``.
    """
    with Image.open(path) as image:
        channels = np.array(image.convert("RGB"), dtype=np.int64)
    return channels[:, :, 0] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 2]


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"sky_class_id": _SKY_CLASS_ID},
)
def sky_mask(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return ``(H, W)`` bool, true where ``img/class`` has sky label ID 23."""
    return class_segmentation(path, meta, attributes=attributes) == _SKY_CLASS_ID


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    shape="3x3",
    file_formats=[".json"],
    meta={"source": "camera.intrinsic"},
)
def read_intrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a ``(3, 3)`` float32 camera matrix from ``meta/<image_id>.json``.

    Uses ``camera.intrinsic`` fields ``fx``, ``fy``, ``u0`` and ``v0`` at the
    metadata's resolution. For ``img/rgb-2k``, scale the matrix separately with
    :func:`~euler_loading.resize_intrinsics`. Metadata files are per image and
    can be indexed as a regular modality alongside RGB and depth.
    """
    intrinsic = _camera_section(_load_json(path), "intrinsic", "intrinsics")
    fx, fy, u0, v0 = _required_floats(intrinsic, ("fx", "fy", "u0", "v0"), "intrinsic")
    return np.array(
        [[fx, 0.0, u0],
         [0.0, fy, v0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    shape="4x4",
    file_formats=[".json"],
    meta={
        "source": "camera.extrinsic",
        "dataset": "Synscapes",
        "angle_unit": "radians",
        "translation_unit": "meters",
        "ego_frame": _EGO_FRAME,
        "rotation_order": _ROTATION_ORDER,
        "default_transform_direction": "camera_to_ego",
        "default_camera_axes": "vehicle",
    },
)
def read_extrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build a ``(4, 4)`` float32 rigid transform from ``camera.extrinsic``.

    Synscapes stores the camera mount as six scalars — ``x``, ``y`` and ``z``
    in metres and ``pitch``, ``roll`` and ``yaw`` in radians — in the
    ego-vehicle frame, which the dataset defines as x forward, y left and z up.
    They are composed here into a homogeneous matrix, by default the camera's
    pose on the vehicle::

        X_ego = T @ X_camera

    Two keyword conventions, read from per-file ``attributes`` first and then
    from ``meta``, cover the other useful directions:

    ``transform_direction``
        ``"camera_to_ego"`` (the default, as above) or ``"ego_to_camera"`` for
        the inverse, which maps ego-frame points — such as the 3D bounding
        boxes in the instance metadata — into the camera.
    ``camera_axes``
        ``"vehicle"`` (the default) keeps the camera's axes aligned with the
        vehicle convention. ``"optical"`` returns the pinhole convention of x
        right, y down and z along the optical axis, so that combining it with
        ``ego_to_camera`` and :func:`read_intrinsics` projects ego-frame points
        to pixels.

    The dataset documents the six fields and the ego frame but not the order in
    which the angles compose; this reader applies the usual automotive
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` and records that in the modality
    metadata. The values are constant across the released dataset.
    """
    extrinsic = _camera_section(_load_json(path), "extrinsic", "extrinsics")
    x, y, z, pitch, roll, yaw = _required_floats(
        extrinsic, ("x", "y", "z", "pitch", "roll", "yaw"), "extrinsic"
    )

    direction = _select_option(
        meta, attributes, "transform_direction", "camera_to_ego", _TRANSFORM_DIRECTIONS
    )
    axes = _select_option(meta, attributes, "camera_axes", "vehicle", _CAMERA_AXES)

    rotation = _rotation_from_euler(yaw, pitch, roll)
    translation = np.array([x, y, z], dtype=np.float64)

    if axes == "optical":
        # The stored pose rotates vehicle-aligned camera axes into the ego
        # frame, so undo the optical permutation on the camera side first.
        rotation = rotation @ _OPTICAL_FROM_VEHICLE.T

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    if direction == "ego_to_camera":
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ translation
        transform = inverse

    return transform.astype(np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_camera_json(
    path: Union[str, BinaryIO],
    key: str,
    payload: dict[str, float],
) -> None:
    """Write ``camera.<key>`` into a Synscapes metadata file.

    One Synscapes JSON file holds both the intrinsics and the extrinsics, so
    writing to an existing file merges into it rather than dropping the other
    block. Stream targets cannot be read back, so they receive only *key*.
    """
    document: dict[str, Any] = {}
    if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
        try:
            existing = _load_json(path)
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict):
            document = existing

    camera = document.get("camera")
    if not isinstance(camera, dict):
        camera = {}
    # Replace the block being written, and drop the plural spelling so a
    # re-read cannot pick up a stale duplicate.
    camera[key] = payload
    camera.pop(f"{key}s", None)
    document["camera"] = camera
    write_json(path, document)


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB array/tensor as an 8-bit PNG, matching ``img/rgb``."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write planar metric depth as a float32 EXR ``Z`` channel.

    Values are stored unchanged, including non-finite depths, so the result
    round-trips through :func:`depth`. Requires
    ``pip install 'euler-loading[synscapes]'``. OpenEXR writes through a
    filename, so unlike the other writers this one does not accept a stream;
    dataset writers hand it a temporary file automatically.
    """
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "Synscapes EXR depth writing requires OpenEXR. "
            "Install it with pip install 'euler-loading[synscapes]'."
        ) from exc

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            "Synscapes write_depth needs a filesystem path because OpenEXR "
            "cannot write to a binary stream."
        )

    ensure_parent(path)
    arr = np.ascontiguousarray(to_hw(value, name="depth"), dtype=np.float32)
    height, width = arr.shape
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    output = OpenEXR.OutputFile(os.fspath(path), header)
    try:
        output.writePixels({"Z": arr.tobytes()})
    finally:
        output.close()


def _save_label_png(path: Union[str, BinaryIO], labels: np.ndarray, name: str) -> None:
    """Save an integer label map as a single-channel PNG."""
    if labels.size and (labels.min() < 0 or labels.max() > 65535):
        raise ValueError(
            f"Synscapes {name} must fit a single-channel PNG (0..65535), "
            f"got range {int(labels.min())}..{int(labels.max())}"
        )
    ensure_parent(path)
    # Pillow infers "L" for uint8 and "I;16" for uint16, both of which the
    # single-channel readers accept. The explicit ``mode=`` spelling is
    # deprecated and removed in Pillow 13, so the dtype carries the choice.
    wide = bool(labels.size and labels.max() > 255)
    image = Image.fromarray(labels.astype(np.uint16 if wide else np.uint8))
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_class_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write Cityscapes label IDs as a single-channel PNG, as ``img/class``."""
    labels = np.rint(to_hw(value, name="class_segmentation")).astype(np.int64)
    _save_label_png(path, labels, "class_segmentation")


@mark_stream_supported
def write_instance_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write instance IDs as an RGB PNG packed ``R + 256 * G + 65536 * B``."""
    ids = np.rint(to_hw(value, name="instance_segmentation")).astype(np.int64)
    if ids.size and (ids.min() < 0 or ids.max() > _MAX_INSTANCE_ID):
        raise ValueError(
            f"Synscapes instance IDs must fit 24 bits (0..{_MAX_INSTANCE_ID}), "
            f"got range {int(ids.min())}..{int(ids.max())}"
        )
    ensure_parent(path)
    channels = np.stack(
        [ids & 0xFF, (ids >> 8) & 0xFF, (ids >> 16) & 0xFF], axis=-1
    ).astype(np.uint8)
    save_image(path, Image.fromarray(channels, mode="RGB"), format="PNG")


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask as a class PNG, labelling sky pixels with its class ID.

    The result is an ``img/class``-shaped file that round-trips through
    :func:`sky_mask`; non-sky pixels get the unlabelled ID ``0``. Override the
    IDs with ``meta['sky_class_id']`` and ``meta['non_sky_class_id']``.
    """
    mask = to_bool_mask(value)
    options = meta or {}
    sky_id = int(options.get("sky_class_id", _SKY_CLASS_ID))
    non_sky_id = int(options.get("non_sky_class_id", 0))
    labels = np.where(mask, sky_id, non_sky_id).astype(np.int64)
    _save_label_png(path, labels, "sky_mask")


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(3, 3)`` camera matrix to ``camera.intrinsic`` JSON.

    ``meta['resx']`` and ``meta['resy']`` are recorded alongside the matrix
    when given, matching the resolution fields Synscapes stores.
    """
    K = to_numpy(value).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {K.shape}")

    payload: dict[str, float] = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "u0": float(K[0, 2]),
        "v0": float(K[1, 2]),
    }
    options = meta or {}
    for key in ("resx", "resy"):
        if isinstance(options.get(key), (int, float)):
            payload[key] = int(options[key])
    _write_camera_json(path, "intrinsic", payload)


@mark_stream_supported
def write_extrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(4, 4)`` rigid transform to ``camera.extrinsic`` JSON.

    The matrix is interpreted with the same ``transform_direction`` and
    ``camera_axes`` conventions :func:`read_extrinsics` accepts, so a value
    read with one set of options writes back unchanged with the same options.
    """
    T = to_numpy(value).astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape (4, 4), got {T.shape}")

    direction = _select_option(
        meta, None, "transform_direction", "camera_to_ego", _TRANSFORM_DIRECTIONS
    )
    axes = _select_option(meta, None, "camera_axes", "vehicle", _CAMERA_AXES)

    rotation = T[:3, :3]
    translation = T[:3, 3]
    if direction == "ego_to_camera":
        rotation, translation = rotation.T, -rotation.T @ translation
    if axes == "optical":
        rotation = rotation @ _OPTICAL_FROM_VEHICLE

    yaw, pitch, roll = _euler_from_rotation(rotation)
    _write_camera_json(
        path,
        "extrinsic",
        {
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
        },
    )
