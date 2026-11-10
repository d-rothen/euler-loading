"""CPU-oriented loader functions for the CEREMA Pixel-Accurate Depth Benchmark.

The benchmark (Gruber et al., "Pixel-Accurate Depth Evaluation in Realistic
Driving Scenarios", 3DV 2019) recorded four static scenes in the CEREMA
weather chamber under clear weather, 17 fog visibilities and two rain rates, by
day and by night. Its release holds one folder per sensor or method::

    rgb_left_8bit/, rgb_right_8bit/        rectified stereo, 1920x1024 8-bit PNG
    gated{0,17,31}_10bit/                  gated slices, 1280x720, 10 bits in 16-bit PNG
    lidar_hdl64_{rgb_left,gated}/          Velodyne HDL64-S3 depth, projected (npz)
    intermetric_{rgb_left,gated}/          dense ground-truth depth, one npz per scene
    sgm/, psmnet/, monodepth/, sparse2dense/   outputs of the benchmarked methods
    calibration/                           ROS camera-info JSON and the tf tree

Every depth map, ground truth and lidar alike, is z-depth in meters on the
pixel grid of its camera, with ``0`` where there is no value. The method
outputs are stored as each method produced them; the ``*_depth`` functions
convert them to metric depth on the ``rgb_left`` grid the way the benchmark's
evaluation code does, with ``baseline x fx = 0.202993 m x 2355.722801 px``:

- ``sgm``: half-resolution disparity (412x960) covering rows 70 to 893;
- ``psmnet``: disparity in pixels of a 960-wide image (512x960);
- ``monodepth``: disparity relative to the image width (256x512);
- ``sparse2dense``: metric depth already, read with :func:`depth`.

The evaluation code then fills every method's missing pixels with their
nearest neighbour; these loaders do not, and mark them ``0``.

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **gated** -- ``(H, W)`` float32 in ``[0, 1]``: ``(raw - 87) / 1023``, clipped.
- **depth**, **sgm_depth**, **psmnet_depth**, **monodepth_depth** --
  ``(H, W)`` float32 meters, ``0`` invalid.
- **read_intrinsics** -- ``(3, 3)`` float32 rectified pinhole matrix.

All readers accept paths and binary streams.
"""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders import _princeton_dense
from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    save_image,
    to_hw,
    to_hwc_rgb,
    to_uint8,
)

__all__ = [
    "rgb",
    "gated",
    "depth",
    "sgm_depth",
    "psmnet_depth",
    "monodepth_depth",
    "read_intrinsics",
    "write_rgb",
    "write_gated",
    "write_depth",
    "write_intrinsics",
]

#: Stereo baseline and rectified focal length the benchmark converts disparity with.
BASELINE_M = 0.202993
FOCAL_PX = 2355.722801
BASELINE_FOCAL = BASELINE_M * FOCAL_PX
#: The rgb_left grid every method output is evaluated on.
RGB_SHAPE = (1024, 1920)
#: Rows of the rgb_left grid the half-resolution SGM output covers.
SGM_ROWS = (70, 70 + 824)
GATED_MAX = 1023.0
#: Dark level the benchmark's code subtracts from the raw gated values.
GATED_DARK_LEVEL = 87

_DEPTH_META = {
    "unit": "meters",
    "depth": "z (planar)",
    "invalid": {"non_finite": False, "sentinel": 0},
}
_METHOD_META = {
    **_DEPTH_META,
    "grid": "rgb_left 1920x1024",
    "baseline_m": BASELINE_M,
    "focal_px": FOCAL_PX,
    "missing_pixels": "0, not nearest-neighbour filled as the benchmark's evaluation does",
}


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _option(meta: dict[str, Any] | None, attributes: dict[str, Any] | None, key: str, default: Any) -> Any:
    """*key* from the file's attributes, else the head's meta, else *default*."""
    for source in (attributes, meta):
        if isinstance(source, dict) and source.get(key) is not None:
            return source[key]
    return default


def _load_npz(path: Union[str, BinaryIO]) -> np.ndarray:
    with np.load(io.BytesIO(_read_bytes(path)), allow_pickle=False) as data:
        if len(data.files) != 1:
            raise ValueError(f"{_source_name(path)!r} should hold one array, found {data.files}")
        return data[data.files[0]]


def _expect_shape(array: np.ndarray, shape: tuple[int, int], path: Union[str, BinaryIO], what: str) -> None:
    if array.shape != shape:
        raise ValueError(
            f"{what} {_source_name(path)!r} has shape {array.shape}, expected {shape} as released"
        )


def _upsample(values: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear resize with pixel-centre alignment (OpenCV's ``INTER_LINEAR``).

    Returns the resized values and a mask of the output pixels whose every
    contributing input pixel is valid (finite and positive); an invalid input
    taints its neighbourhood as a NaN does in OpenCV.
    """
    valid = np.isfinite(values) & (values > 0)
    clean = np.where(valid, values, 0.0).astype(np.float64)

    def axis(size_in: int, size_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.clip((np.arange(size_out) + 0.5) * (size_in / size_out) - 0.5, 0, size_in - 1)
        low = np.floor(pos).astype(np.intp)
        return low, np.minimum(low + 1, size_in - 1), pos - low

    y0, y1, wy = axis(values.shape[0], shape[0])
    x0, x1, wx = axis(values.shape[1], shape[1])
    wy, wx = wy[:, None], wx[None, :]

    def resize(a: np.ndarray) -> np.ndarray:
        top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
        bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
        return top * (1 - wy) + bottom * wy

    tainted = resize((~valid).astype(np.float64)) > 1e-9
    return resize(clean), ~tainted


def _disparity_to_depth(disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    depth = np.zeros(disparity.shape, dtype=np.float32)
    usable = valid & (disparity > 0)
    depth[usable] = BASELINE_FOCAL / disparity[usable]
    return depth


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
    """Load a rectified stereo image (1920x1024 8-bit PNG) as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="gated",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta={"bit_depth": 10, "dark_level": GATED_DARK_LEVEL},
)
def gated(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a gated slice (1280x720 16-bit PNG) as ``(H, W)`` float32 in ``[0, 1]``.

    The raw values sit on a dark level and reach somewhat past 10 bits (up to
    about 1180). As the benchmark's code does, the dark level (87, or
    ``dark_level`` from the attributes or meta) is subtracted and the result
    clipped to ``[0, 1023]`` and divided by 1023.
    """
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype.kind not in "ui":
        raise ValueError(f"gated image {_source_name(path)!r} is not a single-channel integer PNG")
    dark = float(_option(meta, attributes, "dark_level", GATED_DARK_LEVEL))
    return np.clip(raw.astype(np.float32) - dark, 0.0, GATED_MAX) / GATED_MAX


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta=_DEPTH_META,
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a depth map (``arr_0`` of an npz) as ``(H, W)`` float32 meters, ``0`` invalid.

    Reads the dense ground truth (``intermetric_*``), the projected lidar
    (``lidar_hdl64_*``, sparse) and the ``sparse2dense`` output, on the grid of
    the camera they belong to. Non-finite values are returned as ``0``.
    """
    values = np.asarray(_load_npz(path), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth map {_source_name(path)!r} has shape {values.shape}, expected (H, W)")
    return np.where(np.isfinite(values) & (values > 0), values, np.float32(0.0))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={**_METHOD_META, "method": "SGM", "stored": "half-resolution disparity of rows 70-893"},
)
def sgm_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Convert an ``sgm`` output to ``(1024, 1920)`` float32 metric depth.

    The stored ``(412, 960)`` disparity is resized bilinearly to ``(824, 1920)``
    and doubled, as the benchmark does, and placed on rows 70 to 893. Rows
    outside them, NaN disparities and their neighbourhood are ``0``.
    """
    disparity = np.asarray(_load_npz(path), dtype=np.float64)
    _expect_shape(disparity, (412, 960), path, "SGM output")
    rows = SGM_ROWS[1] - SGM_ROWS[0]
    resized, valid = _upsample(disparity, (rows, RGB_SHAPE[1]))
    out = np.zeros(RGB_SHAPE, dtype=np.float32)
    out[SGM_ROWS[0]:SGM_ROWS[1]] = _disparity_to_depth(2.0 * resized, valid)
    return out


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={**_METHOD_META, "method": "PSMNet", "stored": "disparity in pixels of a 960-wide image"},
)
def psmnet_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Convert a ``psmnet`` output to ``(1024, 1920)`` float32 metric depth.

    The stored ``(512, 960)`` disparity is resized bilinearly to the full grid
    and doubled, as the benchmark does.
    """
    disparity = np.asarray(_load_npz(path), dtype=np.float64)
    _expect_shape(disparity, (512, 960), path, "PSMNet output")
    resized, valid = _upsample(disparity, RGB_SHAPE)
    return _disparity_to_depth(2.0 * resized, valid)


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta={**_METHOD_META, "method": "Monodepth", "stored": "disparity relative to the image width"},
)
def monodepth_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Convert a ``monodepth`` output to ``(1024, 1920)`` float32 metric depth.

    The stored ``(256, 512)`` width-relative disparity is resized to the full
    grid and multiplied by 1920, as the benchmark does. The benchmark resizes
    with OpenCV's ``INTER_AREA``; this uses bilinear interpolation, which
    differs only within the 4x upsampling's blocks.
    """
    disparity = np.asarray(_load_npz(path), dtype=np.float64)
    _expect_shape(disparity, (256, 512), path, "Monodepth output")
    resized, valid = _upsample(disparity, RGB_SHAPE)
    return _disparity_to_depth(RGB_SHAPE[1] * resized, valid)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"camera_model": "pinhole", "intrinsics_matrix": "P"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the rectified ``P[:, :3]`` of a ROS camera-info JSON as ``(3, 3)`` float32.

    ``calibration/calib_cam_stereo_left.json`` describes the rectified stereo
    images (the right image shares the matrix), ``calib_gated_bwv.json`` the
    gated camera. ``intrinsics_matrix: "K"`` selects the raw camera matrix,
    which describes the unrectified sensor images only, together with the
    distortion. See :func:`euler_loading.loaders._princeton_dense.load_intrinsics_array`.
    """
    return _princeton_dense.load_intrinsics_array(path, meta, attributes)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as 8-bit PNG."""
    if os.path.splitext(get_target_name(path))[1].lower() not in (".png", ""):
        raise ValueError(f"CEREMA images are PNG, not {get_target_name(path)!r}")
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_gated(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a gated slice as a 16-bit PNG of ``round(value * 1023) + dark level``.

    The inverse of :func:`gated` for values in ``[0, 1]``; the dark level is
    ``meta["dark_level"]``, 87 by default.
    """
    dark = float(_option(meta, None, "dark_level", GATED_DARK_LEVEL))
    scaled = np.clip(np.nan_to_num(to_hw(value, name="gated"), nan=0.0), 0.0, 1.0) * GATED_MAX
    save_image(path, Image.fromarray((np.rint(scaled) + dark).astype(np.uint16)), format="PNG")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write depth in meters as an npz holding ``arr_0`` (float32), as released.

    Non-finite and non-positive values become ``0`` (invalid).
    """
    values = np.asarray(to_hw(value, name="depth"), dtype=np.float32)
    values = np.where(np.isfinite(values) & (values > 0), values, np.float32(0.0))
    if isinstance(path, (str, os.PathLike)):
        ensure_parent(path)
        with open(path, "wb") as stream:  # a path would get ".npz" appended
            np.savez_compressed(stream, values)
    else:
        np.savez_compressed(path, values)


#: Updates an existing camera-info JSON for a crop or rescale; see
#: :func:`euler_loading.loaders._princeton_dense.write_intrinsics`.
write_intrinsics = _princeton_dense.write_intrinsics
