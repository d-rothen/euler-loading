"""Generic GPU-oriented loader for arbitrary modalities.

Unlike dataset-specific loaders, this module infers the loading strategy from
the **file extension** and makes no assumptions about the source dataset.

Supported file extensions:

- **NumPy** (``.npy``) -- loaded via :func:`numpy.load`.
- **NumPy archive** (``.npz``) -- first array in the archive is used.

Return types
------------
- **spherical_map** -- ``torch.FloatTensor`` of shape ``(C, H, W)``.

Usage::

    from euler_loading.loaders.gpu import generic
    from euler_loading import Modality

    Modality("/data/my_dataset/spherical_map", loader=generic.spherical_map)
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


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="spherical_map",
    dtype="float32",
    shape="CHW",
    file_formats=[".npy", ".npz"],
)
def spherical_map(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a spherical map as a ``(C, H, W)`` float32 tensor.

    The file is expected to already be stored in ``(C, H, W)`` layout.
    """
    arr = _load_numpy(path)
    return torch.from_numpy(arr).contiguous()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


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
