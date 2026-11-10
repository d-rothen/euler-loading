"""GPU-oriented loader functions for nuScenes, as prepared by euler-known-datasets.

Loaders return torch tensors created on the CPU for transfer to the training
device: RGB is ``(3, H, W)`` float32 in ``[0, 1]``, lidar points ``(N, 5)``
float32, intrinsics ``(3, 3)`` and the per-sample lidar-to-camera transform
``(4, 4)`` float32. The writers are shared with the CPU module, which accepts
tensors directly. See :mod:`euler_loading.loaders.cpu.nuscenes` for the
calibration file and the coordinate conventions.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders.cpu import nuscenes as _cpu
from euler_loading.loaders.cpu.nuscenes import FORMAT_NAME, CameraFrames

__all__ = [
    "FORMAT_NAME",
    "CameraFrames",
    "rgb",
    "lidar_point_cloud",
    "point_cloud",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
    "write_rgb",
    "write_lidar_point_cloud",
    "write_point_cloud",
    "write_sparse_depth",
    "write_intrinsics",
]

write_rgb = _cpu.write_rgb
write_lidar_point_cloud = _cpu.write_lidar_point_cloud
write_point_cloud = _cpu.write_point_cloud
write_sparse_depth = _cpu.write_sparse_depth
write_intrinsics = _cpu.write_intrinsics


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".jpg"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a camera JPEG as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="point_cloud",
    dtype="float32",
    shape="NC",
    file_formats=[".bin"],
    meta=_cpu.lidar_point_cloud._modality_meta["meta"],
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a ``LIDAR_TOP`` sweep as ``(N, 5)`` float32 ``x, y, z, intensity, ring``;
    see :func:`euler_loading.loaders.cpu.nuscenes.lidar_point_cloud`."""
    return torch.from_numpy(_cpu.lidar_point_cloud(path, meta, attributes=attributes))


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Alias for :func:`lidar_point_cloud`."""
    return lidar_point_cloud(path, meta, attributes=attributes)


@modality_meta(
    modality_type="sparse_depth",
    dtype="float32",
    shape="NC",
    file_formats=[".bin"],
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
    file_formats=[".json"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the camera's ``(3, 3)`` float32 matrix; see
    :func:`euler_loading.loaders.cpu.nuscenes.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


def _select_extrinsics(table: CameraFrames, context: SampleContext) -> torch.Tensor:
    return torch.from_numpy(_cpu._select_extrinsics(table, context))


@selects_per_sample(_select_extrinsics)
@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta=_cpu.read_extrinsics._modality_meta["meta"],
)
def read_extrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> CameraFrames:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32
    lidar-to-camera tensor. See
    :func:`euler_loading.loaders.cpu.nuscenes.read_extrinsics`."""
    return _cpu.read_extrinsics(path, meta, attributes=attributes)
