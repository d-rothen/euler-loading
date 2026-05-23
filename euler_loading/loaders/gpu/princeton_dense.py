"""GPU-oriented loaders for the Princeton DENSE / SeeingThroughFog dataset.

The SeeingThroughFog release stores stereo-camera frames as 12-bit Bayer TIFF
images and Velodyne lidar point clouds as binary ``float32`` records.

Return types
------------
- **rgb** -- ``torch.FloatTensor`` of shape ``(3, H, W)`` in ``[0, 1]``.
- **sparse_depth** -- ``torch.FloatTensor`` of shape ``(N, 5)`` with columns
  ``x, y, z, intensity, ring``.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._princeton_dense import LIDAR_COLUMNS
from euler_loading.loaders._princeton_dense import load_rgb_array
from euler_loading.loaders._princeton_dense import load_sparse_depth_array


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
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
) -> torch.Tensor:
    """Load a SeeingThroughFog 12-bit Bayer TIFF as an RGB tensor."""
    arr = load_rgb_array(path, meta, attributes)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


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
) -> torch.Tensor:
    """Load a SeeingThroughFog lidar point cloud as ``(N, 5)`` float32."""
    return torch.from_numpy(load_sparse_depth_array(path)).contiguous()


def lidar_point_cloud(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Alias for :func:`sparse_depth`."""
    return torch.from_numpy(load_sparse_depth_array(path)).contiguous()


def point_cloud(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Alias for :func:`sparse_depth`."""
    return torch.from_numpy(load_sparse_depth_array(path)).contiguous()
