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

import os
from typing import Any

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_NPY_EXTENSION = ".npy"
_NPZ_EXTENSION = ".npz"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image_rgb(path: str) -> np.ndarray:
    """Load an image file as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_numpy(path: str) -> np.ndarray:
    """Load a ``.npy`` or ``.npz`` file and return the array as float32."""
    ext = os.path.splitext(path)[1].lower()
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
def rgb(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load an RGB sample as an ``(H, W, 3)`` float32 array in ``[0, 1]``.

    - **Image files** are loaded via PIL and normalised to ``[0, 1]``.
    - **NumPy files** are loaded directly and assumed to already be in the
      correct range / layout ``(H, W, 3)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return _load_image_rgb(path)
    return _load_numpy(path)


@modality_meta(
    modality_type="dense_depth",
    dtype="float32",
    shape="HW",
    file_formats=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"],
)
def depth(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
    """Load a depth map as an ``(H, W)`` float32 array.

    - **Image files** are loaded as single-channel greyscale.
    - **NumPy files** are loaded directly.

    No unit conversion is applied; values are returned as-is.
    """
    ext = os.path.splitext(path)[1].lower()
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
def sky_mask(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
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
def read_intrinsics(path: str, meta: dict[str, Any] | None = None) -> np.ndarray:
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
