"""GPU-oriented loader functions for the KITTI-CARLA dataset.

Loaders return contiguous torch tensors: RGB is ``(3, H, W)`` float32 in
``[0, 1]``, depth ``(1, H, W)`` float32 planar depth in metres, semantic tags
``(1, H, W)`` int64, sky masks ``(1, H, W)`` bool, lidar points ``(N, 7)``
float64, intrinsics ``(3, 3)`` and extrinsics ``(4, 4)`` float32, and the
per-sample lidar pose ``(4, 4)`` float64. Tensors are created on the CPU for
transfer to the training device, as with the other GPU-oriented loaders.

The writers are shared with the CPU module, which accepts tensors directly.
See :mod:`euler_loading.loaders.cpu.kitti_carla` for the native file
formats, the depth encoding and the coordinate conventions.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders.cpu import kitti_carla as _cpu
from euler_loading.loaders.cpu.kitti_carla import CLASS_NAMES, PALETTE, LidarPoseTable

__all__ = [
    "CLASS_NAMES",
    "PALETTE",
    "LidarPoseTable",
    "rgb",
    "depth",
    "class_segmentation",
    "sky_mask",
    "lidar_point_cloud",
    "point_cloud",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
    "lidar_pose",
    "write_rgb",
    "write_depth",
    "write_class_segmentation",
    "write_sky_mask",
    "write_lidar_point_cloud",
    "write_point_cloud",
    "write_sparse_depth",
    "write_intrinsics",
    "write_extrinsics",
]

# The writers convert tensors through ``to_numpy``, so the CPU implementations
# are reused rather than duplicated.
write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_class_segmentation = _cpu.write_class_segmentation
write_sky_mask = _cpu.write_sky_mask
write_lidar_point_cloud = _cpu.write_lidar_point_cloud
write_point_cloud = _cpu.write_point_cloud
write_sparse_depth = _cpu.write_sparse_depth
write_intrinsics = _cpu.write_intrinsics
write_extrinsics = _cpu.write_extrinsics


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an ``images_rgb`` PNG as ``(3, H, W)`` float32 in ``[0, 1]``."""
    arr = _cpu.rgb(path, meta, attributes=attributes)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".png"],
    output_range=[0.0, _cpu._DEPTH_FAR_PLANE_M],
    output_unit="meters",
    meta=_cpu.depth._modality_meta["meta"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an ``images_depth`` PNG as ``(1, H, W)`` float32 planar depth in metres.

    Decodes CARLA's 24-bit encoding; sky pixels read as the 1000 m far plane.
    See :func:`euler_loading.loaders.cpu.kitti_carla.depth`.
    """
    arr = _cpu.depth(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    output_range=[0, len(CLASS_NAMES) - 1],
    meta=_cpu.class_segmentation._modality_meta["meta"],
)
def class_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an ``images_ss`` PNG as ``(1, H, W)`` int64 CARLA semantic tags."""
    arr = _cpu.class_segmentation(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta=_cpu.sky_mask._modality_meta["meta"],
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return ``(1, H, W)`` bool, true where ``images_ss`` has the ``skyclass``
    colour, ``(70, 130, 180)`` by default."""
    arr = _cpu.sky_mask(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="point_cloud",
    dtype="float64",
    shape="NC",
    file_formats=[".ply"],
    meta=_cpu.lidar_point_cloud._modality_meta["meta"],
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a lidar sweep as ``(N, 7)`` float64: ``x, y, z,
    cos_angle_lidar_surface, timestamp, instance, semantic``."""
    return torch.from_numpy(_cpu.lidar_point_cloud(path, meta, attributes=attributes))


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Alias for :func:`lidar_point_cloud`."""
    return lidar_point_cloud(path, meta, attributes=attributes)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float64",
    shape="NC",
    file_formats=[".ply"],
    meta=_cpu.sparse_depth._modality_meta["meta"],
)
def sparse_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load lidar points for sparse-depth workflows; see :func:`lidar_point_cloud`."""
    return torch.from_numpy(_cpu.sparse_depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".txt"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the ``(3, 3)`` float32 camera matrix (1392x1024, 72 degree
    horizontal FOV by default); see
    :func:`euler_loading.loaders.cpu.kitti_carla.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".txt"],
    meta=_cpu.read_extrinsics._modality_meta["meta"],
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load ``lidar_to_cam<N>.txt`` as a ``(4, 4)`` float32 lidar-to-camera
    transform into optical camera axes by default; see
    :func:`euler_loading.loaders.cpu.kitti_carla.read_extrinsics`."""
    return torch.from_numpy(_cpu.read_extrinsics(path, meta, attributes=attributes))


def _select_lidar_pose(table: LidarPoseTable, context: SampleContext) -> torch.Tensor:
    return torch.from_numpy(_cpu._select_lidar_pose(table, context))


@selects_per_sample(_select_lidar_pose)
@modality_meta(
    modality_type="lidar_pose",
    dtype="float64",
    hierarchical=True,
    shape="4x4",
    file_formats=[".txt"],
    meta=_cpu.lidar_pose._modality_meta["meta"],
)
def lidar_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> LidarPoseTable:
    """Load ``full_poses_lidar.txt``; each sample receives its ``(4, 4)``
    float64 lidar-to-world pose tensor. See
    :func:`euler_loading.loaders.cpu.kitti_carla.lidar_pose`."""
    return _cpu.lidar_pose(path, meta, attributes=attributes)
