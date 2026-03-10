"""CPU-oriented loader functions for the Virtual KITTI 2 dataset.

Each function follows the ``Callable[[str], Any]`` signature expected by
:class:`~euler_loading.Modality`.  All loaders return **numpy ndarrays**
directly from the loaded files, suitable for CPU-based processing.

Return types
------------
- **rgb** – ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** – ``np.ndarray`` of shape ``(H, W)`` float32 in metres.
- **class_segmentation** – ``np.ndarray`` of shape ``(H, W, 3)`` int64
  (RGB-encoded class labels).
- **instance_segmentation** – ``np.ndarray`` of shape ``(H, W, 3)`` int64
  (RGB-encoded instance labels).
- **sky_mask** – ``np.ndarray`` of shape ``(H, W)`` bool
  (``True`` where the pixel is sky, i.e. RGB ``(90, 200, 255)``).
- **scene_flow** – ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **read_intrinsics** – ``np.ndarray`` of shape ``(3, 3)`` float32
  (the camera intrinsic matrix *K*).
- **read_extrinsics** – ``np.ndarray`` float32 parsed from the text file.

Usage::

    from euler_loading.loaders.cpu import vkitti2
    from euler_loading import Modality

    Modality("/data/vkitti2/rgb",   loader=vkitti2.rgb)
    Modality("/data/vkitti2/depth", loader=vkitti2.depth)
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    to_bool_mask,
    to_hwc_rgb,
    to_hw,
    to_numpy,
    to_uint8,
)

# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB image as an ``(H, W, 3)`` float32 array in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png"],
    output_unit="meters",
    meta={"raw_range": [0, 65535], "radial_depth": False, "scale_to_meters": 0.01},
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a VKITTI2 depth map as an ``(H, W)`` float32 array in **metres**.

    VKITTI2 stores depth as 16-bit PNG where each pixel value represents
    depth in centimetres.  This loader converts to metres
    (``value / 100``).
    """
    return np.array(Image.open(path), dtype=np.float32) / 100.0


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HWC",
    file_formats=[".png"],
    meta={"encoding": "rgb"},
)
def class_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB-encoded class-segmentation mask as an ``(H, W, 3)`` int64 array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.int64)


@modality_meta(
    modality_type="instance_segmentation",
    dtype="int64",
    shape="HWC",
    file_formats=[".png"],
    meta={"encoding": "rgb"},
)
def instance_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB-encoded instance-segmentation mask as an ``(H, W, 3)`` int64 array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.int64)


_SKY_COLOR = (90, 200, 255)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"sky_color": [90, 200, 255]},
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a sky mask as an ``(H, W)`` bool array.

    Reads the RGB class-segmentation PNG and returns ``True`` where the
    pixel colour equals ``(90, 200, 255)`` (sky class in VKITTI2).
    """
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return (
        (arr[:, :, 0] == _SKY_COLOR[0])
        & (arr[:, :, 1] == _SKY_COLOR[1])
        & (arr[:, :, 2] == _SKY_COLOR[2])
    )


@modality_meta(
    modality_type="scene_flow",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def scene_flow(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an optical / scene-flow map as an ``(H, W, 3)`` float32 array in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Text ground-truth readers (calibration)
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".txt"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Parse a VKITTI2 intrinsics text file into a ``(3, 3)`` float32 *K* matrix.

    The file has the header ``frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]``.
    Intrinsics are constant across frames in VKITTI2, so only the first row
    is used.  Returns the camera matrix::

        K = [[fx,  s, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fx = float(data[0][2])
    fy = float(data[0][3])
    cx = float(data[0][4])
    cy = float(data[0][5])
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@modality_meta(
    modality_type="extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="Nx1",
    file_formats=[".txt"],
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Parse a VKITTI2 extrinsics text file into a float32 array."""
    return np.loadtxt(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_rgb(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB array/tensor to PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    Image.fromarray(arr, mode="RGB").save(path)


def write_depth(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a depth map in VKITTI2 encoding (uint16 centimetres PNG)."""
    ensure_parent(path)
    depth_m = to_hw(value, name="depth").astype(np.float32)
    depth_cm = np.clip(np.rint(depth_m * 100.0), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_cm, mode="I;16").save(path)


def write_class_segmentation(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB-encoded class segmentation map."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="class_segmentation"), scale_unit_range=False)
    Image.fromarray(arr, mode="RGB").save(path)


def write_instance_segmentation(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB-encoded instance segmentation map."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="instance_segmentation"), scale_unit_range=False)
    Image.fromarray(arr, mode="RGB").save(path)


def write_sky_mask(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky-mask image in VKITTI2-compatible RGB encoding."""
    ensure_parent(path)
    mask = to_bool_mask(value)
    sky_color_raw = (meta or {}).get("sky_color", _SKY_COLOR)
    sky_color = np.asarray(sky_color_raw, dtype=np.uint8)
    if sky_color.shape != (3,):
        raise ValueError("meta['sky_color'] must be a 3-element RGB color")
    arr = np.zeros(mask.shape + (3,), dtype=np.uint8)
    arr[mask] = sky_color
    Image.fromarray(arr, mode="RGB").save(path)


def write_scene_flow(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a scene-flow map to PNG as RGB in [0, 255]."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="scene_flow"), scale_unit_range=True)
    Image.fromarray(arr, mode="RGB").save(path)


def write_intrinsics(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a 3x3 camera intrinsic matrix in VKITTI2 text format."""
    ensure_parent(path)
    K = to_numpy(value).astype(np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {K.shape}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n")
        f.write(
            f"0 0 {float(K[0, 0]):.6f} {float(K[1, 1]):.6f} "
            f"{float(K[0, 2]):.6f} {float(K[1, 2]):.6f}\n"
        )


def write_extrinsics(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an extrinsics matrix to whitespace-delimited text."""
    ensure_parent(path)
    arr = to_numpy(value).astype(np.float32)
    np.savetxt(path, arr, fmt="%.9g")
