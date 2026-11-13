"""GPU-oriented loader functions for Hypersim.

Loaders return torch tensors created on the CPU: RGB ``(3, H, W)`` float32 in
``[0, 1]``, depth and radial distance ``(1, H, W)`` float32 in meters, intrinsics
``(3, 3)`` float32 and the camera-to-world pose ``(4, 4)`` float32. The
writers are shared with the CPU module, which accepts tensors directly. See
:mod:`euler_loading.loaders.cpu.hypersim` for the tone mapping, the camera
model and the calibration file.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders.cpu import hypersim as _cpu
from euler_loading.loaders.cpu.hypersim import FORMAT_NAME, CameraTrajectory, tonemap

__all__ = [
    "FORMAT_NAME",
    "CameraTrajectory",
    "tonemap",
    "rgb",
    "depth",
    "radial_distance",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_radial_distance",
    "write_intrinsics",
]

write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_radial_distance = _cpu.write_radial_distance
write_intrinsics = _cpu.write_intrinsics


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".hdf5"],
    output_range=[0.0, 1.0],
    meta=_cpu.rgb._modality_meta["meta"],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a tone-mapped ``color.hdf5`` image as a ``(3, H, W)`` float32 tensor
    in ``[0, 1]``. See :func:`euler_loading.loaders.cpu.hypersim.rgb`."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".hdf5"],
    output_unit="meters",
    meta=_cpu.depth._modality_meta["meta"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load planar depth as a ``(1, H, W)`` float32 tensor in meters. See
    :func:`euler_loading.loaders.cpu.hypersim.depth`."""
    return torch.from_numpy(_cpu.depth(path, meta, attributes=attributes)).unsqueeze(0)


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".hdf5"],
    output_unit="meters",
    meta=_cpu.radial_distance._modality_meta["meta"],
)
def radial_distance(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load Euclidean distances as a ``(1, H, W)`` float32 tensor in meters. See
    :func:`euler_loading.loaders.cpu.hypersim.radial_distance`."""
    return torch.from_numpy(_cpu.radial_distance(path, meta, attributes=attributes)).unsqueeze(0)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the trajectory's ``(3, 3)`` float32 camera matrix. See
    :func:`euler_loading.loaders.cpu.hypersim.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


def _select_pose(table: CameraTrajectory, context: SampleContext) -> torch.Tensor:
    return torch.from_numpy(_cpu._select_pose(table, context))


@selects_per_sample(_select_pose)
@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta=_cpu.camera_pose._modality_meta["meta"],
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> CameraTrajectory:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32
    camera-to-world tensor. See :func:`euler_loading.loaders.cpu.hypersim.camera_pose`."""
    return _cpu.camera_pose(path, meta, attributes=attributes)
