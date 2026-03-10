"""Generic GPU-oriented loader for dense-depth datasets.

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
- **rgb** – ``torch.FloatTensor`` of shape ``(3, H, W)`` in ``[0, 1]``.
- **depth** – ``torch.FloatTensor`` of shape ``(1, H, W)``.
- **sky_mask** – ``torch.BoolTensor`` of shape ``(1, H, W)``.
  Requires ``meta["sky_mask"]`` → ``[R, G, B]`` identifying the sky class.
- **read_intrinsics** – ``torch.FloatTensor`` of shape ``(3, 3)``.
  Ignores *path*; returns ``meta["intrinsics"]`` directly.

Usage::

    from euler_loading.loaders.gpu import generic_dense_depth
    from euler_loading import Modality

    Modality("/data/my_dataset/rgb",   loader=generic_dense_depth.rgb)
    Modality("/data/my_dataset/depth", loader=generic_dense_depth.depth)
"""

from __future__ import annotations

import json
import os
from typing import Any, BinaryIO, Union

import numpy as np
import torch
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

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_NPY_EXTENSION = ".npy"
_NPZ_EXTENSION = ".npz"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_name(path: Union[str, BinaryIO]) -> str:
    """Return a filename suitable for extension detection."""
    name = getattr(path, "name", path)
    return str(name)


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
    shape="CHW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an RGB sample as a ``(3, H, W)`` float32 tensor in ``[0, 1]``.

    - **Image files** are loaded via PIL and normalised to ``[0, 1]``.
    - **NumPy files** are loaded directly and assumed to already be in the
      correct range / layout ``(H, W, 3)``.
    """
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        arr = _load_image_rgb(path)
    else:
        arr = _load_numpy(path)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a depth map as a ``(1, H, W)`` float32 tensor.

    - **Image files** are loaded as single-channel greyscale.
    - **NumPy files** are loaded directly.

    No unit conversion is applied; values are returned as-is.
    """
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        arr = np.array(Image.open(path), dtype=np.float32)
    else:
        arr = _load_numpy(path)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
    requires_meta=["sky_mask"],
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a sky mask as a ``(1, H, W)`` bool tensor.

    Reads the file as an RGB image and compares each pixel against the sky
    colour provided in ``meta["sky_mask"]`` (an ``[R, G, B]`` list/array).
    """
    if meta is None or "sky_mask" not in meta:
        raise ValueError(
            "sky_mask requires meta['sky_mask'] to be set to an [R, G, B] array"
        )
    sky_color = meta["sky_mask"]
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    mask = (
        (arr[:, :, 0] == sky_color[0])
        & (arr[:, :, 1] == sky_color[1])
        & (arr[:, :, 2] == sky_color[2])
    )
    return torch.from_numpy(mask).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    requires_meta=["intrinsics"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the ``(3, 3)`` camera intrinsics matrix from *meta*.

    The *path* argument is ignored.  The intrinsics are expected under
    ``meta["intrinsics"]`` as anything convertible to a tensor (list, array,
    or tensor).
    """
    if meta is None or "intrinsics" not in meta:
        raise ValueError(
            "read_intrinsics requires meta['intrinsics'] to be set"
        )
    K = meta["intrinsics"]
    if isinstance(K, torch.Tensor):
        return K.to(dtype=torch.float32)
    return torch.tensor(K, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_rgb(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write RGB data to image or NumPy formats based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(path)[1].lower()
    rgb = to_hwc_rgb(value, name="rgb")

    if ext in _IMAGE_EXTENSIONS:
        arr = to_uint8(rgb, scale_unit_range=True)
        Image.fromarray(arr, mode="RGB").save(path)
        return

    rgb_f32 = rgb.astype(np.float32)
    if ext == _NPY_EXTENSION:
        np.save(path, rgb_f32)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=rgb_f32)
        return

    raise ValueError(f"Unsupported RGB output extension: {ext}")


def write_depth(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write depth data to image or NumPy formats based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(path)[1].lower()
    depth = to_hw(value, name="depth")

    if ext in _IMAGE_EXTENSIONS:
        depth_f32 = depth.astype(np.float32)
        if (
            np.issubdtype(depth.dtype, np.integer)
            and depth_f32.size > 0
            and float(depth_f32.min()) >= 0.0
            and float(depth_f32.max()) <= 255.0
        ):
            Image.fromarray(depth_f32.astype(np.uint8), mode="L").save(path)
            return
        depth_u16 = np.clip(np.rint(depth_f32), 0, 65535).astype(np.uint16)
        Image.fromarray(depth_u16, mode="I;16").save(path)
        return

    depth_f32 = depth.astype(np.float32)
    if ext == _NPY_EXTENSION:
        np.save(path, depth_f32)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=depth_f32)
        return

    raise ValueError(f"Unsupported depth output extension: {ext}")


def write_sky_mask(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a sky mask using either RGB encoding or NumPy formats."""
    ensure_parent(path)
    ext = os.path.splitext(path)[1].lower()
    mask = to_bool_mask(value)

    if ext in _IMAGE_EXTENSIONS:
        sky_color_raw = (meta or {}).get("sky_mask", [255, 255, 255])
        sky_color = np.asarray(sky_color_raw, dtype=np.uint8)
        if sky_color.shape != (3,):
            raise ValueError("meta['sky_mask'] must be a 3-element RGB color")
        rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
        rgb[mask] = sky_color
        Image.fromarray(rgb, mode="RGB").save(path)
        return

    if ext == _NPY_EXTENSION:
        np.save(path, mask.astype(np.bool_))
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=mask.astype(np.bool_))
        return

    raise ValueError(f"Unsupported sky-mask output extension: {ext}")


def write_intrinsics(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a 3x3 intrinsics matrix to text, JSON, or NumPy format."""
    ensure_parent(path)
    ext = os.path.splitext(path)[1].lower()
    K = to_numpy(value).astype(np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {K.shape}")

    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"intrinsics": K.tolist()}, f)
        return
    if ext == _NPY_EXTENSION:
        np.save(path, K)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=K)
        return

    np.savetxt(path, K, fmt="%.9g")
