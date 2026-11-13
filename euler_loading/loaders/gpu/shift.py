"""GPU-oriented loader functions for SHIFT.

Loaders return torch tensors created on the CPU: RGB ``(3, H, W)`` float32 in
``[0, 1]``, depth and radial distance ``(1, H, W)`` float32 meters, labels
``(1, H, W)`` int64, sky masks ``(1, H, W)`` bool, intrinsics ``(3, 3)``
float32 and camera poses ``(4, 4)`` float32. The writers are shared with the
CPU module, which accepts tensors directly. See
:mod:`euler_loading.loaders.cpu.shift` for the layout, the depth code, the
label sets and the calibration files.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders.cpu import shift as _cpu

__all__ = [
    "rgb",
    "depth",
    "radial_distance",
    "semantic_segmentation",
    "sky_mask",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_radial_distance",
    "write_semantic_segmentation",
    "write_sky_mask",
    "write_intrinsics",
]

write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_radial_distance = _cpu.write_radial_distance
write_semantic_segmentation = _cpu.write_semantic_segmentation
write_sky_mask = _cpu.write_sky_mask
write_intrinsics = _cpu.write_intrinsics


def _map(values) -> torch.Tensor:
    return torch.from_numpy(values).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png", ".jpg"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a frame as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".png"],
    output_range=[0.0, _cpu.DEPTH_FAR_PLANE_M],
    output_unit="meters",
    meta=_cpu.depth._modality_meta["meta"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a depth PNG as ``(1, H, W)`` float32 planar meters; see
    :func:`euler_loading.loaders.cpu.shift.depth`."""
    return _map(_cpu.depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".png"],
    output_unit="meters",
    meta=_cpu.radial_distance._modality_meta["meta"],
)
def radial_distance(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a depth PNG as ``(1, H, W)`` float32 Euclidean distance in meters; see
    :func:`euler_loading.loaders.cpu.shift.radial_distance`."""
    return _map(_cpu.radial_distance(path, meta, attributes=attributes))


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta=_cpu.semantic_segmentation._modality_meta["meta"],
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a label PNG as ``(1, H, W)`` int64 class IDs; see
    :func:`euler_loading.loaders.cpu.shift.semantic_segmentation`."""
    return _map(_cpu.semantic_segmentation(path, meta, attributes=attributes))


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta=_cpu.sky_mask._modality_meta["meta"],
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return ``(1, H, W)`` bool, true where a label PNG holds the sky (``[13, 0, 0]`` by default); see
    :func:`euler_loading.loaders.cpu.shift.sky_mask`."""
    return _map(_cpu.sky_mask(path, meta, attributes=attributes))


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the sequence's ``(3, 3)`` float32 camera matrix; see
    :func:`euler_loading.loaders.cpu.shift.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


def _select_pose(table: _cpu.CameraFrames, context: SampleContext) -> torch.Tensor:
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
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> _cpu.CameraFrames:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32 camera-to-world
    pose as a tensor; see :func:`euler_loading.loaders.cpu.shift.camera_pose`."""
    return _cpu.camera_pose(path, meta, attributes=attributes)
