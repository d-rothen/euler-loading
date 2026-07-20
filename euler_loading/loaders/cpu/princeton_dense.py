"""CPU-oriented loaders for the Princeton DENSE / SeeingThroughFog dataset.

The SeeingThroughFog release stores stereo-camera frames as 12-bit Bayer TIFF
images and Velodyne lidar point clouds as binary ``float32`` records.

Return types
------------
- **rccb** -- ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **rgb** -- ``np.ndarray`` of shape ``(H, W, 3)`` float32 in ``[0, 1]``.
- **sparse_depth** -- ``np.ndarray`` of shape ``(N, 5)`` float32 with columns
  ``x, y, z, intensity, ring``.
- **read_intrinsics** -- ``np.ndarray`` of shape ``(3, 3)`` float32 from
  ``calib_cam_stereo_left.json``.
- **read_extrinsics** -- ``np.ndarray`` of shape ``(4, 4)`` float32 from
  ``calib_tf_tree_full.json``, mapping HDL64 lidar points into the left camera
  optical frame by default.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    mark_stream_supported,
    save_image,
    to_hwc_rgb,
    to_uint8,
)
from euler_loading.loaders._princeton_dense import DEFAULT_CAMERA_FRAME
from euler_loading.loaders._princeton_dense import DEFAULT_LIDAR_FRAME
from euler_loading.loaders._princeton_dense import LIDAR_COLUMNS
from euler_loading.loaders._princeton_dense import load_extrinsics_array
from euler_loading.loaders._princeton_dense import load_intrinsics_array
from euler_loading.loaders._princeton_dense import load_rgb_array
from euler_loading.loaders._princeton_dense import load_sparse_depth_array


@modality_meta(
    modality_type="rccb",
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
def rccb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog 12-bit Bayer TIFF as RCCB float32 in ``[0, 1]``."""
    return load_rgb_array(path, meta, attributes)


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog plain 8-bit PNG as RGB float32 in ``[0, 1]``."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


@mark_stream_supported
def write_rgb(
    path: Union[str, BinaryIO],
    value: Any,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write an RGB array/tensor as an 8-bit PNG."""
    ensure_parent(path)
    arr = to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True)
    save_image(path, Image.fromarray(arr, mode="RGB"), format="PNG")


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


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={
        "dataset": "SeeingThroughFog",
        "camera_frame": DEFAULT_CAMERA_FRAME,
        "source": "calib_cam_stereo_left",
    },
)
def read_intrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load the left stereo camera intrinsics matrix ``K``."""
    return load_intrinsics_array(path)


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "dataset": "SeeingThroughFog",
        "default_source_frame": DEFAULT_LIDAR_FRAME,
        "default_target_frame": DEFAULT_CAMERA_FRAME,
        "transform_direction": "source_frame_to_target_frame",
    },
)
def read_extrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load the default ``lidar_hdl64_s3_roof`` to left-camera transform."""
    return load_extrinsics_array(path, meta, attributes)
