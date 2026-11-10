"""CPU-oriented loader functions for the DrivingStereo dataset.

DrivingStereo (Yang et al., CVPR 2019, https://drivingstereo-dataset.github.io/)
is a rectified stereo driving dataset with lidar-derived depth and disparity.
Its released images are half resolution (about 881x400; the foggy weather
subset is 879x400). The test set and the weather subsets also ship
full-resolution maps at exactly twice that size. The weather archives are laid
out as::

    <weather>/left-image-half-size/<sequence>_<timestamp>.jpg      # rgb
    <weather>/right-image-half-size/<sequence>_<timestamp>.jpg
    <weather>/depth-map-half-size/<sequence>_<timestamp>.png       # metres * 256
    <weather>/disparity-map-half-size/<sequence>_<timestamp>.png   # pixels * 256
    <weather>/depth-map-full-size/<sequence>_<timestamp>.png       # metres * 256, 2x
    <weather>/disparity-map-full-size/<sequence>_<timestamp>.png   # pixels * 128, 2x

with one KITTI-style calibration text file per sequence,
``calibration/<train|test>/<half|full>-image-calib/<sequence>.txt``. Camera
``101`` is the left camera and ``103`` the right one.

Alignment
---------
Everything is returned on the pixel grid of the **half-size left image**,
the reference here. Depth and disparity are given in the rectified left
camera, and the half-size maps share that image's size and grid, so
:func:`depth` and :func:`disparity` return them unchanged. The full-size
maps have twice the rows and columns, and their calibration is the half-size
one doubled exactly (``P_full = diag(2, 2, 1) @ P_half``, no half-pixel
offset). Half-size pixel ``(y, x)`` therefore sits on full-size pixel
``(2y, 2x)``. :func:`depth_full_size` and :func:`disparity_full_size` resample
onto that grid: each pixel takes the full-size sample at ``(2y, 2x)``, or
failing that the nearest valid full-size sample within half a half-size pixel.
Ties go to the nearer surface. See :func:`depth_full_size` for the
``full_size_resampling`` option.

The half-size maps are not a downsampling of the full-size ones. About half
of their valid pixels have no full-size sample in the matching 2x2 block,
and they are denser on their grid (about 21% of pixels valid against 5% at
full size), so prefer :func:`depth` where both exist.

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** / **depth_full_size** -- ``(H, W)`` float32 planar depth in metres
  on the half-size left-image grid; ``0`` marks pixels without a measurement.
- **disparity** / **disparity_full_size** -- ``(H, W)`` float32 disparity in
  half-size pixels on the same grid; ``0`` marks invalid pixels. Where valid,
  ``depth = fx * baseline / disparity`` with ``fx`` from the half-size
  calibration.
- **read_intrinsics** -- ``(3, 3)`` float32 rectified camera matrix
  ``P_rect_101[:, :3]`` (hierarchical).

All readers accept paths and binary streams, so zip-backed modalities work
unchanged. Every writer accepts both as well.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    pixel_grid_transform,
    replace_text,
    save_image,
    to_hw,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_text,
)

__all__ = [
    "rgb",
    "depth",
    "depth_full_size",
    "disparity",
    "disparity_full_size",
    "read_intrinsics",
    "write_rgb",
    "write_depth",
    "write_depth_full_size",
    "write_disparity",
    "write_disparity_full_size",
    "write_intrinsics",
]

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# "uint16 PNG ... dividing it by 256", for depth at both resolutions and for
# half-size disparity; full-size disparity is "dividing it by 128".
_DEPTH_SCALE = 256.0
_DISPARITY_SCALE_HALF_SIZE = 256.0
_DISPARITY_SCALE_FULL_SIZE = 128.0
_UINT16_MAX = 65535

#: Full-size maps have exactly twice the rows and columns of the half-size
#: left images.
FULL_SIZE_FACTOR = 2

_CAMERAS = {"left": "101", "right": "103", "101": "101", "103": "103"}
_RESAMPLING = ("nearest", "subsample")

# Full-size neighbours of the sample at (2y, 2x): the four at half a half-size
# pixel, then the four diagonal ones at about 0.7 of one.
_EDGE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

_CALIBRATION_LINE = re.compile(r"^\s*([^:#\s]+)\s*:(.*)$")
_JPEG_EXTENSIONS = (".jpg", ".jpeg")
_JPEG_QUALITY = 95

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


def _resampling(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    value = _option(meta, attributes, "full_size_resampling")
    if value is None:
        return _RESAMPLING[0]
    if not isinstance(value, str) or value.lower() not in _RESAMPLING:
        raise ValueError(
            "DrivingStereo 'full_size_resampling' must be one of "
            f"{', '.join(repr(option) for option in _RESAMPLING)}, got {value!r}"
        )
    return value.lower()


def _camera(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    value = _option(meta, attributes, "camera")
    if value is None:
        return _CAMERAS["left"]
    camera = _CAMERAS.get(str(value).lower())
    if camera is None:
        raise ValueError(
            "DrivingStereo 'camera' must be 'left' ('101') or 'right' ('103'), "
            f"got {value!r}"
        )
    return camera


def _load_uint16(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a single-channel 16-bit PNG as ``(H, W)`` uint16."""
    with Image.open(path) as image:
        mode = image.mode
        arr = np.array(image)
    if mode not in ("I;16", "I;16B", "I;16L", "I") or arr.ndim != 2:
        raise ValueError(
            f"DrivingStereo map {_source_name(path)!r} must be a single-channel "
            f"16-bit PNG, got mode {mode!r}"
        )
    if arr.dtype != np.uint16:
        if arr.size and (arr.min() < 0 or arr.max() > _UINT16_MAX):
            raise ValueError(
                f"DrivingStereo map {_source_name(path)!r} has values outside uint16"
            )
        arr = arr.astype(np.uint16)
    return arr


def _ring_nearest(
    padded: np.ndarray,
    shape: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    *,
    nearest_is_max: bool,
) -> np.ndarray:
    """Return, per half-size pixel, the nearest-surface valid value of a ring."""
    height, width = shape
    rows = [
        padded[1 + dy:1 + dy + 2 * height:2, 1 + dx:1 + dx + 2 * width:2]
        for dy, dx in offsets
    ]
    stacked = np.stack(rows, axis=-1)
    if nearest_is_max:
        return stacked.max(axis=-1)
    best = np.where(stacked > 0, stacked, np.inf).min(axis=-1)
    return np.where(np.isfinite(best), best, 0.0)


def _to_left_half_grid(
    full: np.ndarray,
    *,
    resampling: str,
    nearest_is_max: bool,
) -> np.ndarray:
    """Resample a full-size map (``0`` = invalid) onto the half-size grid.

    Half-size pixel ``(y, x)`` sits on full-size pixel ``(2y, 2x)``.
    ``subsample`` keeps exactly that sample. ``nearest`` falls back to the
    four full-size neighbours half a half-size pixel away, then to the four
    diagonal ones. Within a ring the nearer surface wins: the smaller depth,
    or the larger disparity when *nearest_is_max*.
    """
    height = (full.shape[0] + 1) // FULL_SIZE_FACTOR
    width = (full.shape[1] + 1) // FULL_SIZE_FACTOR
    centre = full[0::FULL_SIZE_FACTOR, 0::FULL_SIZE_FACTOR]
    if resampling == "subsample":
        return np.ascontiguousarray(centre)
    padded = np.pad(
        full,
        ((1, 2 * height - full.shape[0]), (1, 2 * width - full.shape[1])),
    )
    edge = _ring_nearest(padded, (height, width), _EDGE_OFFSETS, nearest_is_max=nearest_is_max)
    diagonal = _ring_nearest(padded, (height, width), _DIAGONAL_OFFSETS, nearest_is_max=nearest_is_max)
    return np.where(centre > 0, centre, np.where(edge > 0, edge, diagonal))


def _encode_uint16(values: Any, scale: float, *, name: str) -> np.ndarray:
    """Encode a map as uint16 ``round(value * scale)``; invalid becomes ``0``."""
    arr = to_hw(values, name=name).astype(np.float64)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    return np.clip(np.rint(arr * scale), 0, _UINT16_MAX).astype(np.uint16)


def _to_full_size(raw: np.ndarray) -> np.ndarray:
    """Place half-grid samples at ``(2y, 2x)`` of an empty full-size map."""
    height, width = raw.shape
    full = np.zeros(
        (height * FULL_SIZE_FACTOR, width * FULL_SIZE_FACTOR), dtype=raw.dtype
    )
    full[0::FULL_SIZE_FACTOR, 0::FULL_SIZE_FACTOR] = raw
    return full


def _save_uint16_png(path: Union[str, BinaryIO], raw: np.ndarray) -> None:
    ensure_parent(path)
    # uint16 becomes an I;16 image; Pillow 13 rejects an explicit mode="I;16".
    save_image(path, Image.fromarray(raw), format="PNG")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".jpg", ".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a left (or right) image as ``(H, W, 3)`` float32 in ``[0, 1]``.

    The released images are rectified 8-bit JPEGs.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Depth and disparity
# ---------------------------------------------------------------------------

_DEPTH_META = {
    "encoding": "uint16_png",
    "decode": "value / 256",
    "raw_range": [0, _UINT16_MAX],
    "scale_to_meters": 1.0 / _DEPTH_SCALE,
    "radial_depth": False,
    "invalid": {"non_finite": False, "sentinel": 0},
    "sparse": True,
    "camera": "rectified_left",
}


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, _UINT16_MAX / _DEPTH_SCALE],
    output_unit="meters",
    meta={**_DEPTH_META, "resolution": "half"},
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``depth-map-half-size`` PNG as ``(H, W)`` float32 metres.

    The uint16 values divided by 256 give planar depth in the rectified left
    camera. ``0`` marks pixels without a measurement. The map is pixel-aligned
    with the half-size left image.
    """
    return _load_uint16(path).astype(np.float32) / np.float32(_DEPTH_SCALE)


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, _UINT16_MAX / _DEPTH_SCALE],
    output_unit="meters",
    meta={
        **_DEPTH_META,
        "resolution": "full",
        "aligned_to": "left_image_half_size",
        "full_size_resampling": list(_RESAMPLING),
    },
)
def depth_full_size(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``depth-map-full-size`` PNG on the half-size left-image grid.

    The full-size map is decoded like :func:`depth` (uint16 / 256, metres)
    and resampled to half its rows and columns. Half-size pixel ``(y, x)``
    sits on full-size pixel ``(2y, 2x)``. ``full_size_resampling`` in the
    file's attributes or the head's meta selects how:

    - ``"nearest"`` (default): the sample at ``(2y, 2x)``, else the nearest
      valid sample within half a half-size pixel (the four direct
      neighbours, then the four diagonal ones), the smaller depth winning
      within a ring;
    - ``"subsample"``: only the sample at ``(2y, 2x)``. It is exact but keeps
      about a quarter of the measurements.

    The full-size maps are sparser on their grid than the half-size ones
    (about 5% of pixels valid against 21%), so prefer :func:`depth` where both
    exist.
    """
    full = _load_uint16(path).astype(np.float32) / np.float32(_DEPTH_SCALE)
    return _to_left_half_grid(
        full, resampling=_resampling(meta, attributes), nearest_is_max=False
    )


_DISPARITY_META = {
    "encoding": "uint16_png",
    "raw_range": [0, _UINT16_MAX],
    "invalid": {"non_finite": False, "sentinel": 0},
    "sparse": True,
    "camera": "rectified_left",
    "disparity_unit": "half_size_pixels",
}


@modality_meta(
    modality_type="disparity",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, _UINT16_MAX / _DISPARITY_SCALE_HALF_SIZE],
    output_unit="pixels",
    meta={**_DISPARITY_META, "decode": "value / 256", "resolution": "half"},
)
def disparity(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``disparity-map-half-size`` PNG as ``(H, W)`` float32 pixels.

    The uint16 values divided by 256 give the left-image disparity in
    half-size pixels; ``0`` marks invalid pixels.
    """
    return _load_uint16(path).astype(np.float32) / np.float32(_DISPARITY_SCALE_HALF_SIZE)


@modality_meta(
    modality_type="disparity",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, _UINT16_MAX / _DISPARITY_SCALE_HALF_SIZE],
    output_unit="pixels",
    meta={
        **_DISPARITY_META,
        "decode": "value / 128 / 2",
        "resolution": "full",
        "aligned_to": "left_image_half_size",
        "full_size_resampling": list(_RESAMPLING),
    },
)
def disparity_full_size(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``disparity-map-full-size`` PNG on the half-size left-image grid.

    Full-size disparity is uint16 / 128 in full-size pixels. It is resampled
    like :func:`depth_full_size`, the larger disparity winning within a ring,
    and halved into half-size pixels, the unit of :func:`disparity`.
    """
    full = _load_uint16(path).astype(np.float32) / np.float32(_DISPARITY_SCALE_FULL_SIZE)
    half_grid = _to_left_half_grid(
        full, resampling=_resampling(meta, attributes), nearest_is_max=True
    )
    return half_grid / np.float32(FULL_SIZE_FACTOR)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _parse_calibration(text: str, *, source: str) -> dict[str, np.ndarray]:
    """Return the numeric ``key: values`` entries of a calibration file."""
    entries: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        match = _CALIBRATION_LINE.match(line)
        if match is None:
            continue
        key, values = match.group(1), match.group(2).split()
        try:
            entries[key] = np.array([float(value) for value in values], dtype=np.float64)
        except ValueError:
            continue
    if not entries:
        raise ValueError(f"DrivingStereo calibration {source!r} has no 'key: values' entries")
    return entries


def _projection(entries: Mapping[str, np.ndarray], camera: str, *, source: str) -> np.ndarray:
    key = f"P_rect_{camera}"
    values = entries.get(key)
    if values is None:
        raise ValueError(f"DrivingStereo calibration {source!r} has no {key!r}")
    if values.size != 12:
        raise ValueError(
            f"DrivingStereo calibration {source!r}: {key!r} must hold 12 values, "
            f"got {values.size}"
        )
    return values.reshape(3, 4)


def _check_pinhole(K: np.ndarray, what: str) -> None:
    if (
        not np.all(np.isfinite(K))
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[1:, 0], 0.0)
        or not np.allclose(K[2], (0.0, 0.0, 1.0))
    ):
        raise ValueError(f"{what} is not a pinhole camera matrix: {K.tolist()}")


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".txt"],
    meta={"matrix": "P_rect[:, :3]", "camera": ["left", "right"]},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the rectified ``(3, 3)`` float32 camera matrix ``P_rect_101[:, :3]``.

    Reads a ``<half|full>-image-calib/<sequence>.txt`` file. Index the
    ``half-image-calib`` file for the half-size images and the maps this
    module returns, which are all on that grid. The full-size file describes
    the same camera at twice the resolution. ``camera`` in the file's
    attributes or the head's meta selects ``"left"`` (``101``, the default)
    or ``"right"`` (``103``). Both cameras share the same rectified matrix.

    The stereo baseline is ``|T_103|`` (about 0.545 m). Do not take it from
    ``P_rect_103[0, 3]`` of a half-size file: that entry was not halved and
    still holds the full-size ``-fx * baseline``.
    """
    source = _source_name(path)
    entries = _parse_calibration(_read_bytes(path).decode("utf-8"), source=source)
    K = _projection(entries, _camera(meta, attributes), source=source)[:, :3]
    _check_pinhole(K, f"DrivingStereo calibration {source!r}")
    return np.ascontiguousarray(K, dtype=np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as 8-bit JPEG (quality 95) or, for other names, PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    image = Image.fromarray(arr, mode="RGB")
    extension = os.path.splitext(get_target_name(path))[1].lower()
    if extension in _JPEG_EXTENSIONS:
        image.save(path, format="JPEG", quality=_JPEG_QUALITY)
        return
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write depth in metres as a uint16 PNG of ``round(depth * 256)``.

    Non-finite and non-positive values become ``0`` (invalid); depths beyond
    about 256 m saturate at 65535.
    """
    _save_uint16_png(path, _encode_uint16(value, _DEPTH_SCALE, name="depth"))


@mark_stream_supported
def write_depth_full_size(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a half-grid depth map as a ``depth-map-full-size`` PNG.

    Each half-size pixel ``(y, x)`` goes to full-size pixel ``(2y, 2x)``; the
    other pixels are ``0``. :func:`depth_full_size` reads the file back
    unchanged with either resampling mode.
    """
    raw = _encode_uint16(value, _DEPTH_SCALE, name="depth")
    _save_uint16_png(path, _to_full_size(raw))


@mark_stream_supported
def write_disparity(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write half-size disparity as a uint16 PNG of ``round(disparity * 256)``."""
    _save_uint16_png(
        path, _encode_uint16(value, _DISPARITY_SCALE_HALF_SIZE, name="disparity")
    )


@mark_stream_supported
def write_disparity_full_size(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write half-grid disparity (half-size pixels) as a ``disparity-map-full-size`` PNG.

    Values are doubled into full-size pixels and stored as ``round(d * 128)``
    at ``(2y, 2x)``, so :func:`disparity_full_size` reads them back unchanged.
    """
    raw = _encode_uint16(
        value, _DISPARITY_SCALE_FULL_SIZE * FULL_SIZE_FACTOR, name="disparity"
    )
    _save_uint16_png(path, _to_full_size(raw))


def _format_values(values: np.ndarray) -> str:
    # repr() is the shortest text that reads back to the same float64.
    return " ".join(repr(float(value)) for value in np.asarray(values).reshape(-1))


def _image_size(meta: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(meta, Mapping):
        return None
    width, height = meta.get("width"), meta.get("height")
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("DrivingStereo write_intrinsics needs both meta 'width' and 'height'")
    for key, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) <= 0
            or float(value) != int(value)
        ):
            raise ValueError(f"DrivingStereo meta {key!r} must be a positive integer, got {value!r}")
    return int(width), int(height)


def _rewrite_calibration(
    text: str,
    transform: np.ndarray,
    size: tuple[int, int] | None,
    *,
    source: str,
) -> str:
    """Apply a pixel-grid *transform* to every ``P_rect_*`` line of *text*.

    Only lines whose values change are rewritten, so an identity transform
    and an unchanged *size* return *text* unchanged.
    """
    identity = np.array_equal(transform, np.eye(3))
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        match = _CALIBRATION_LINE.match(line)
        key = match.group(1) if match is not None else ""
        ending = line[len(line.rstrip("\r\n")):]
        if key.startswith("P_rect_") and not identity:
            P = _projection(
                _parse_calibration(line, source=source), key[len("P_rect_"):], source=source
            )
            out.append(f"{key}: {_format_values(transform @ P)}{ending}")
            continue
        if key.startswith("S_rect_") and size is not None:
            stored = _parse_calibration(line, source=source)[key]
            if stored.size != 2 or tuple(stored) != (float(size[0]), float(size[1])):
                out.append(f"{key}: {size[0]} {size[1]}{ending}")
                continue
        out.append(line)
    return "".join(out)


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a rectified camera matrix into a DrivingStereo calibration file.

    *value* is what :func:`read_intrinsics` returns, so a written file reads
    back as *value*. An existing file is updated, not replaced. The change
    from its stored ``P_rect_<camera>[:, :3]`` must be a crop or a rescale.
    It is applied to every ``P_rect_*`` row, so both cameras and the baseline
    term stay consistent, and a whole-pixel crop shifts the stored values by
    exact integers. ``meta["width"]`` and ``meta["height"]`` update the
    ``S_rect_*`` image sizes. Every other line is kept, including the
    unrectified ``K_*``/``D_*``, which describe the raw sensor images. An
    unchanged matrix leaves the file byte-identical.

    A new file, or a stream target, receives ``P_rect_101 = [value | 0]`` and,
    with ``width``/``height`` in *meta*, ``S_rect_101``.
    """
    matrix = to_numpy(value).astype(np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {matrix.shape}")
    _check_pinhole(matrix, "intrinsics")
    size = _image_size(meta)

    if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
        name = os.fspath(path)
        text = _read_bytes(path).decode("utf-8")
        entries = _parse_calibration(text, source=name)
        stored = _projection(entries, _camera(meta, None), source=name)[:, :3]
        transform = pixel_grid_transform(matrix @ np.linalg.inv(stored))
        if transform is None:
            raise ValueError(
                f"Refusing to update {name!r}: the new intrinsics are not a crop or "
                f"rescale of the stored P_rect matrix ({stored.tolist()} -> {matrix.tolist()})."
            )
        updated = _rewrite_calibration(text, transform, size, source=name)
        if updated != text:
            replace_text(path, updated, newline="")
        return

    ensure_parent(path)
    P = np.zeros((3, 4))
    P[:, :3] = matrix
    lines = []
    if size is not None:
        lines.append(f"S_rect_101: {size[0]} {size[1]}")
    lines.append(f"P_rect_101: {_format_values(P)}")
    write_text(path, "\n".join(lines) + "\n")
