"""GPU-oriented loader functions for the ScatterNeRF dataset.

Loaders return torch tensors created on the CPU: RGB ``(3, H, W)`` float32 in
``[0, 1]``, intrinsics ``(3, 3)`` float32 and the camera-to-world pose
``(4, 4)`` float64. The writers are shared with the CPU module, which accepts
tensors directly. See :mod:`euler_loading.loaders.cpu.scatternerf` for the
file layout, the pose conventions and the per-sequence scale.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import scatternerf as _cpu

__all__ = [
    "rgb",
    "read_intrinsics",
    "camera_pose",
    "camera_pose_metric",
    "write_rgb",
    "write_intrinsics",
    "write_camera_pose",
]

write_rgb = _cpu.write_rgb
write_intrinsics = _cpu.write_intrinsics
write_camera_pose = _cpu.write_camera_pose


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a rectified stereo frame as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    shape="3x3",
    file_formats=[".txt"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the image's ``(3, 3)`` float32 camera matrix; see
    :func:`euler_loading.loaders.cpu.scatternerf.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".txt"],
    meta=_cpu.camera_pose._modality_meta["meta"],
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the image's ``(4, 4)`` float64 camera-to-world pose; see
    :func:`euler_loading.loaders.cpu.scatternerf.camera_pose`."""
    return torch.from_numpy(_cpu.camera_pose(path, meta, attributes=attributes))


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".txt"],
    meta=_cpu.camera_pose_metric._modality_meta["meta"],
)
def camera_pose_metric(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the image's ``(4, 4)`` float64 camera-to-world pose in meters; see
    :func:`euler_loading.loaders.cpu.scatternerf.camera_pose_metric`."""
    return torch.from_numpy(_cpu.camera_pose_metric(path, meta, attributes=attributes))
