"""Generic CPU-oriented loader for dense-depth datasets.

Unlike the dataset-specific loaders (vkitti2, real_drive_sim), this module
infers the loading strategy from the **file extension** and relies on the
``meta`` dict for dataset-specific details (sky colour, intrinsics).

Supported file extensions:

- **Images** (``.png``, ``.jpg``, ``.jpeg``, ``.bmp``, ``.tif``, ``.tiff``)
  – loaded via PIL.
- **NumPy** (``.npy``) – loaded via :func:`numpy.load`.
- **NumPy archive** (``.npz``) – first array in the archive is used.

Return types
------------
- **rgb** – ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** – ``np.ndarray`` of shape ``(H, W)`` float32.
- **sky_mask** – ``np.ndarray`` of shape ``(H, W)`` bool.
  Requires ``meta["sky_mask"]`` → ``[R, G, B]`` identifying the sky class.
- **read_intrinsics** – ``np.ndarray`` of shape ``(3, 3)`` float32.
  Ignores *path*; returns ``meta["intrinsics"]`` directly.

Usage::

    from euler_loading.loaders.cpu import generic_dense_depth
    from euler_loading import Modality

    Modality("/data/my_dataset/rgb",   loader=generic_dense_depth.rgb)
    Modality("/data/my_dataset/depth", loader=generic_dense_depth.depth)
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
    get_target_name,
    mark_stream_supported,
    save_image,
    to_bool_mask,
    to_hwc_rgb,
    to_hw,
    to_numpy,
    to_uint8,
    write_json,
    write_text,
)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_NPY_EXTENSION = ".npy"
_NPZ_EXTENSION = ".npz"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_name(path: Union[str, BinaryIO]) -> str:
    """Return a filename suitable for extension detection."""
    return get_target_name(path)


def _load_image_rgb(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load an image file as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_numpy(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a ``.npy`` or ``.npz`` file and return the array as float32."""
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext == _NPZ_EXTENSION:
        npz = np.load(path)
        arr = next(iter(npz.values()))
    else:
        arr = np.load(path)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB sample as an ``(H, W, 3)`` float32 array in ``[0, 1]``.

    - **Image files** are loaded via PIL and normalised to ``[0, 1]``.
    - **NumPy files** are loaded directly and assumed to already be in the
      correct range / layout ``(H, W, 3)``.
    """
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return _load_image_rgb(path)
    return _load_numpy(path)


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a depth map as an ``(H, W)`` float32 array.

    - **Image files** are loaded as single-channel greyscale.
    - **NumPy files** are loaded directly.

    No unit conversion is applied; values are returned as-is.
    """
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return np.array(Image.open(path), dtype=np.float32)
    return _load_numpy(path)


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
    requires_meta=["sky_mask"],
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a sky mask as an ``(H, W)`` bool array.

    Reads the file as an RGB image and compares each pixel against the sky
    colour provided in ``meta["sky_mask"]`` (an ``[R, G, B]`` list/array).
    """
    if meta is None or "sky_mask" not in meta:
        raise ValueError(
            "sky_mask requires meta['sky_mask'] to be set to an [R, G, B] array"
        )
    sky_color = meta["sky_mask"]
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return (
        (arr[:, :, 0] == sky_color[0])
        & (arr[:, :, 1] == sky_color[1])
        & (arr[:, :, 2] == sky_color[2])
    )


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    requires_meta=["intrinsics"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(3, 3)`` camera intrinsics matrix from *meta*.

    The *path* argument is ignored.  The intrinsics are expected under
    ``meta["intrinsics"]`` as anything convertible to an array (list, tensor,
    or array).
    """
    if meta is None or "intrinsics" not in meta:
        raise ValueError(
            "read_intrinsics requires meta['intrinsics'] to be set"
        )
    return np.asarray(meta["intrinsics"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write RGB data to image or NumPy formats based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    rgb = to_hwc_rgb(value, name="rgb")

    if ext in _IMAGE_EXTENSIONS:
        arr = to_uint8(rgb, scale_unit_range=True)
        save_image(path, Image.fromarray(arr, mode="RGB"))
        return

    rgb_f32 = rgb.astype(np.float32)
    if ext == _NPY_EXTENSION:
        np.save(path, rgb_f32)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=rgb_f32)
        return

    raise ValueError(f"Unsupported RGB output extension: {ext}")


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write depth data to image or NumPy formats based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    depth = to_hw(value, name="depth")

    if ext in _IMAGE_EXTENSIONS:
        depth_f32 = depth.astype(np.float32)
        if (
            np.issubdtype(depth.dtype, np.integer)
            and depth_f32.size > 0
            and float(depth_f32.min()) >= 0.0
            and float(depth_f32.max()) <= 255.0
        ):
            save_image(path, Image.fromarray(depth_f32.astype(np.uint8), mode="L"))
            return
        depth_u16 = np.clip(np.rint(depth_f32), 0, 65535).astype(np.uint16)
        save_image(path, Image.fromarray(depth_u16, mode="I;16"), format="PNG")
        return

    depth_f32 = depth.astype(np.float32)
    if ext == _NPY_EXTENSION:
        np.save(path, depth_f32)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=depth_f32)
        return

    raise ValueError(f"Unsupported depth output extension: {ext}")


@mark_stream_supported
def write_sky_mask(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask using either RGB encoding or NumPy formats."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    mask = to_bool_mask(value)

    if ext in _IMAGE_EXTENSIONS:
        sky_color_raw = (meta or {}).get("sky_mask", [255, 255, 255])
        sky_color = np.asarray(sky_color_raw, dtype=np.uint8)
        if sky_color.shape != (3,):
            raise ValueError("meta['sky_mask'] must be a 3-element RGB color")
        rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
        rgb[mask] = sky_color
        save_image(path, Image.fromarray(rgb, mode="RGB"))
        return

    if ext == _NPY_EXTENSION:
        np.save(path, mask.astype(np.bool_))
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=mask.astype(np.bool_))
        return

    raise ValueError(f"Unsupported sky-mask output extension: {ext}")


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a 3x3 intrinsics matrix to text, JSON, or NumPy format."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    K = to_numpy(value).astype(np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {K.shape}")

    if ext == ".json":
        write_json(path, {"intrinsics": K.tolist()})
        return
    if ext == _NPY_EXTENSION:
        np.save(path, K)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=K)
        return

    text = "\n".join(" ".join(f"{entry:.9g}" for entry in row) for row in K) + "\n"
    write_text(path, text)
