"""CPU-oriented loaders for the Princeton DENSE / SeeingThroughFog dataset.

The SeeingThroughFog release stores stereo-camera frames as 12-bit Bayer TIFF
images and Velodyne lidar point clouds as binary ``float32`` records.

Return types
------------
- **rgb** -- ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **sparse_depth** -- ``np.ndarray`` of shape ``(N, 5)`` float32 with columns
  ``x, y, z, intensity, ring``.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import numpy as np

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._princeton_dense import LIDAR_COLUMNS
from euler_loading.loaders._princeton_dense import load_rgb_array
from euler_loading.loaders._princeton_dense import load_sparse_depth_array


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".tif", ".tiff"],
    output_range=[0.0, 1.0],
    meta={
        "dataset": "SeeingThroughFog",
        "raw_bit_depth": 12,
        "bayer_pattern": "GBRG",
        "raw_max_value": 4095.0,
    },
)
def rgb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog 12-bit Bayer TIFF as RGB float32 in ``[0, 1]``."""
    return load_rgb_array(path, meta, attributes)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float32",
    shape="NC",
    file_formats=[".bin"],
    meta={
        "dataset": "SeeingThroughFog",
        "representation": "point_cloud",
        "columns": LIDAR_COLUMNS,
        "coordinate_unit": "meters",
    },
)
def sparse_depth(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog lidar point cloud as ``(N, 5)`` float32."""
    return load_sparse_depth_array(path)


def lidar_point_cloud(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Alias for :func:`sparse_depth`."""
    return load_sparse_depth_array(path)


def point_cloud(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Alias for :func:`sparse_depth`."""
    return load_sparse_depth_array(path)
