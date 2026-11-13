"""GPU-oriented loader functions for UrbanSyn.

Loaders return torch tensors created on the CPU: RGB ``(3, H, W)`` float32 in
``[0, 1]``, depth ``(1, H, W)`` float32 planar meters (``0`` invalid), labels
``(1, H, W)`` int64 Cityscapes train IDs, sky masks ``(1, H, W)`` bool,
intrinsics ``(3, 3)`` float32 and camera poses ``(4, 4)`` float64. The
writers are shared with the CPU module, which accepts tensors directly. See
:mod:`euler_loading.loaders.cpu.urbansyn` for the release layout, the depth
units, the label set and the poses.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import urbansyn as _cpu

__all__ = [
    "rgb",
    "depth",
    "semantic_segmentation",
    "sky_mask",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_semantic_segmentation",
    "write_sky_mask",
    "write_intrinsics",
    "write_camera_pose",
]

write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_semantic_segmentation = _cpu.write_semantic_segmentation
write_sky_mask = _cpu.write_sky_mask
write_intrinsics = _cpu.write_intrinsics
write_camera_pose = _cpu.write_camera_pose


def _map(values) -> torch.Tensor:
    return torch.from_numpy(values).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a frame as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".exr"],
    output_unit="meters",
    meta=_cpu.depth._modality_meta["meta"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a Z-depth EXR as ``(1, H, W)`` float32 planar meters, ``0`` invalid; see
    :func:`euler_loading.loaders.cpu.urbansyn.depth`."""
    return _map(_cpu.depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta=_cpu.semantic_segmentation._modality_meta["meta"],
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an ``ss`` label map as ``(1, H, W)`` int64 Cityscapes train IDs (19: other); see
    :func:`euler_loading.loaders.cpu.urbansyn.semantic_segmentation`."""
    return _map(_cpu.semantic_segmentation(path, meta, attributes=attributes))


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta=_cpu.sky_mask._modality_meta["meta"],
)
def sky_mask(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return ``(1, H, W)`` bool, true where an ``ss`` label map holds the sky (train ID 10 by
    default); see :func:`euler_loading.loaders.cpu.urbansyn.sky_mask`."""
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
    """Return the ``(3, 3)`` float32 camera matrix; see
    :func:`euler_loading.loaders.cpu.urbansyn.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".exr", ".json"],
    meta=_cpu.camera_pose._modality_meta["meta"],
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the ``(4, 4)`` float64 camera-to-world pose from a depth EXR header; see
    :func:`euler_loading.loaders.cpu.urbansyn.camera_pose`."""
    return torch.from_numpy(_cpu.camera_pose(path, meta, attributes=attributes))
