"""Generic GPU-oriented loader for arbitrary modalities.

Unlike dataset-specific loaders, this module infers the loading strategy from
the **file extension** and makes no assumptions about the source dataset.

Supported file extensions:

- **NumPy** (``.npy``) -- loaded via :func:`numpy.load`.
- **NumPy archive** (``.npz``) -- first array in the archive is used.

Return types
------------
- **map_2d** / **scattering_coefficient** -- ``torch.FloatTensor`` of shape ``(H, W)``.
- **semantic_segmentation** / **instance_segmentation** -- integer
  ``torch.Tensor`` of shape ``(H, W)``.
- **map_3d** / **atmospheric_light** -- ``torch.FloatTensor`` of shape ``(C, H, W)``.
- **points_3d** -- ``torch.FloatTensor`` of shape ``(3, H, W)``.
- **spherical_map** -- ``torch.FloatTensor`` of shape ``(C, H, W)``.
- **intrinsics** -- ``torch.FloatTensor`` of shape ``(3, 3)``.
- **sh_coeffs** -- ``torch.FloatTensor`` of shape ``(N, 3)``.

Usage::

    from euler_loading.loaders.gpu import generic
    from euler_loading import Modality

    Modality("/data/my_dataset/scattering_coefficient", loader=generic.scattering_coefficient)
    Modality("/data/my_dataset/atmospheric_light",      loader=generic.atmospheric_light)
    Modality("/data/my_dataset/points_3d",              loader=generic.points_3d)
    Modality("/data/my_dataset/spherical_map",          loader=generic.spherical_map)
    Modality("/data/my_dataset/intrinsics",             loader=generic.intrinsics)
    Modality("/data/my_dataset/sh_coeffs",              loader=generic.sh_coeffs)
"""

from __future__ import annotations

import os
from typing import Any, BinaryIO, Union

import numpy as np
import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    get_target_name,
    mark_stream_supported,
    to_hw,
    to_numpy,
)

_NPY_EXTENSION = ".npy"
_NPZ_EXTENSION = ".npz"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_name(path: Union[str, BinaryIO]) -> str:
    """Return a filename suitable for extension detection."""
    return get_target_name(path)


def _load_numpy(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a ``.npy`` or ``.npz`` file and return the array as float32."""
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext == _NPZ_EXTENSION:
        npz = np.load(path)
        arr = next(iter(npz.values()))
    else:
        arr = np.load(path)
    return arr.astype(np.float32)


def _load_label_map(path: Union[str, BinaryIO], dtype: np.dtype) -> np.ndarray:
    """Load a standardized HW integer label map without converting to float."""
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext == _NPZ_EXTENSION:
        with np.load(path) as npz:
            arr = next(iter(npz.values()))
    else:
        arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"segmentation map must have shape (H, W), got {arr.shape}")
    return arr.astype(dtype, copy=False)


def _write_numpy(path: Union[str, BinaryIO], value: Any) -> None:
    """Write an array to ``.npy`` or ``.npz`` based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    arr = to_numpy(value).astype(np.float32)

    if ext == _NPY_EXTENSION:
        np.save(path, arr)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=arr)
        return

    raise ValueError(f"Unsupported output extension: {ext}")


def _as_3d_points(value: Any) -> np.ndarray:
    """Return a float32 ``(3, H, W)`` array for dense 3D point maps."""
    arr = to_numpy(value).astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"points_3d must have shape (3, H, W), got {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="map_2d",
    dtype="float32",
    shape="HW",
    file_formats=[".npy", ".npz"],
)
def map_2d(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an arbitrary 2D map as an ``(H, W)`` float32 tensor.

    Suitable for any single-channel dense quantity (e.g. scattering
    coefficient, attenuation, opacity) where no dataset-specific decoding
    is required.
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="uint8",
    shape="HW",
    file_formats=[".npy", ".npz"],
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an HW uint8 class-id map; 255 is the standardized void id."""
    return torch.from_numpy(_load_label_map(path, np.dtype(np.uint8))).contiguous()


@modality_meta(
    modality_type="instance_segmentation",
    dtype="uint16",
    shape="HW",
    file_formats=[".npy", ".npz"],
)
def instance_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an HW uint16 instance-id map; 0 denotes stuff or void."""
    return torch.from_numpy(_load_label_map(path, np.dtype(np.uint16))).contiguous()


@modality_meta(
    modality_type="map_3d",
    dtype="float32",
    shape="CHW",
    file_formats=[".npy", ".npz"],
)
def map_3d(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an arbitrary 3D map as a ``(C, H, W)`` float32 tensor.

    The file is expected to already be stored in ``(C, H, W)`` layout,
    matching the torch convention.  Suitable for any per-pixel vector
    quantity (e.g. atmospheric light, surface normals, flow).
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


@modality_meta(
    modality_type="points_3d",
    dtype="float32",
    shape="3HW",
    file_formats=[".npy", ".npz"],
)
def points_3d(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load dense 3D points as a ``(3, H, W)`` float32 tensor.

    Files are expected to store the per-pixel 3D point coordinates directly in
    ``(3, H, W)`` layout.
    """
    return torch.from_numpy(_as_3d_points(_load_numpy(path))).contiguous()


@modality_meta(
    modality_type="scattering_coefficient",
    dtype="float32",
    shape="HW",
    file_formats=[".npy", ".npz"],
)
def scattering_coefficient(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a scattering-coefficient map as an ``(H, W)`` float32 tensor."""
    return map_2d(path, meta)


@modality_meta(
    modality_type="atmospheric_light",
    dtype="float32",
    shape="CHW",
    file_formats=[".npy", ".npz"],
)
def atmospheric_light(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an atmospheric-light map as a ``(C, H, W)`` float32 tensor."""
    return map_3d(path, meta)


@modality_meta(
    modality_type="spherical_map",
    dtype="float32",
    shape="CHW",
    file_formats=[".npy", ".npz"],
)
def spherical_map(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a spherical map as a ``(C, H, W)`` float32 tensor.

    The file is expected to already be stored in ``(C, H, W)`` layout.
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    shape="3x3",
    file_formats=[".npy", ".npz"],
)
def intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a camera intrinsics matrix as a ``(3, 3)`` float32 tensor.

    The file is expected to contain a ``(3, 3)`` array::

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


@modality_meta(
    modality_type="sh_coeffs",
    dtype="float32",
    shape="NC",
    file_formats=[".npy", ".npz"],
)
def sh_coeffs(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load spherical-harmonic coefficients as an ``(N, 3)`` float32 tensor.

    *N* is the number of SH basis functions (e.g. 15 for degree-3 SH with
    the constant term removed).  Each row is a 3-vector (one per spatial
    dimension).
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_map_2d(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a 2D map to NumPy formats based on extension.

    Accepts ``(H, W)``, ``(1, H, W)``, or ``(H, W, 1)`` input and stores
    the array in ``(H, W)`` layout.
    """
    arr = to_hw(value, name="map_2d")
    _write_numpy(path, arr)


def _write_segmentation(path: Union[str, BinaryIO], value: Any, dtype: np.dtype, name: str) -> None:
    arr = to_hw(value, name=name).astype(dtype, copy=False)
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    if ext == _NPY_EXTENSION:
        np.save(path, arr)
    elif ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=arr)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


@mark_stream_supported
def write_semantic_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write the standardized HW uint8 semantic map (255 means void)."""
    _write_segmentation(path, value, np.dtype(np.uint8), "semantic_segmentation")


@mark_stream_supported
def write_instance_segmentation(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write the standardized HW uint16 instance map (0 means stuff/void)."""
    _write_segmentation(path, value, np.dtype(np.uint16), "instance_segmentation")


@mark_stream_supported
def write_map_3d(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a 3D map to NumPy formats based on extension.

    Input is expected in ``(C, H, W)`` layout (torch convention) and is
    stored as-is.
    """
    _write_numpy(path, value)


@mark_stream_supported
def write_points_3d(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write dense 3D points to NumPy formats in ``(3, H, W)`` layout."""
    _write_numpy(path, _as_3d_points(value))


@mark_stream_supported
def write_scattering_coefficient(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a scattering-coefficient map (delegates to :func:`write_map_2d`)."""
    write_map_2d(path, value, meta)


@mark_stream_supported
def write_atmospheric_light(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an atmospheric-light map (delegates to :func:`write_map_3d`)."""
    write_map_3d(path, value, meta)


@mark_stream_supported
def write_spherical_map(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write spherical-map data to NumPy formats based on extension."""
    ensure_parent(path)
    ext = os.path.splitext(_get_name(path))[1].lower()
    arr = to_numpy(value).astype(np.float32)

    if ext == _NPY_EXTENSION:
        np.save(path, arr)
        return
    if ext == _NPZ_EXTENSION:
        np.savez_compressed(path, data=arr)
        return

    raise ValueError(f"Unsupported spherical-map output extension: {ext}")


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(3, 3)`` intrinsics matrix to a NumPy file."""
    _write_numpy(path, value)


@mark_stream_supported
def write_sh_coeffs(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write ``(N, 3)`` spherical-harmonic coefficients to a NumPy file."""
    _write_numpy(path, value)
