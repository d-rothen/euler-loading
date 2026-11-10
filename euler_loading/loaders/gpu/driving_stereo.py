"""GPU-oriented loader functions for the DrivingStereo dataset.

Loaders return contiguous torch tensors on the pixel grid of the half-size
left image: RGB is ``(3, H, W)`` float32 in ``[0, 1]``, depth ``(1, H, W)``
float32 planar depth in metres, disparity ``(1, H, W)`` float32 in half-size
pixels (``0`` marks invalid pixels in both), and intrinsics ``(3, 3)`` float32.
Tensors are created on the CPU for transfer to the training device, as with
the other GPU-oriented loaders.

The writers are shared with the CPU module, which accepts tensors directly.
See :mod:`euler_loading.loaders.cpu.driving_stereo` for the file encodings
and how the full-size maps are aligned to the half-size left image.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import driving_stereo as _cpu
from euler_loading.loaders.cpu.driving_stereo import FULL_SIZE_FACTOR

__all__ = [
    "rgb",
    "depth",
    "depth_full_size",
    "disparity",
    "disparity_full_size",
    "read_intrinsics",
    "write_rgb",
    "write_depth",
    "write_depth_full_size",
    "write_disparity",
    "write_disparity_full_size",
    "write_intrinsics",
]

# The writers convert tensors through ``to_numpy``, so the CPU implementations
# are reused rather than duplicated.
write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_depth_full_size = _cpu.write_depth_full_size
write_disparity = _cpu.write_disparity
write_disparity_full_size = _cpu.write_disparity_full_size
write_intrinsics = _cpu.write_intrinsics


def _annotation(name: str, shape: str) -> dict[str, Any]:
    """The CPU annotation of *name* with this variant's *shape*."""
    annotation = dict(getattr(_cpu, name)._modality_meta)
    return {
        "modality_type": annotation["type"],
        "dtype": annotation["dtype"],
        "hierarchical": annotation["hierarchical"],
        "shape": shape,
        "file_formats": annotation["file_formats"],
        "output_range": annotation["output_range"],
        "output_unit": annotation["output_unit"],
        "meta": annotation["meta"],
    }


def _map(arr: Any) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(**_annotation("rgb", "CHW"))
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a left (or right) JPEG as ``(3, H, W)`` float32 in ``[0, 1]``."""
    arr = _cpu.rgb(path, meta, attributes=attributes)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(**_annotation("depth", "1HW"))
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a ``depth-map-half-size`` PNG as ``(1, H, W)`` float32 metres
    (uint16 / 256, ``0`` invalid), aligned with the half-size left image."""
    return _map(_cpu.depth(path, meta, attributes=attributes))


@modality_meta(**_annotation("depth_full_size", "1HW"))
def depth_full_size(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a ``depth-map-full-size`` PNG as ``(1, H, W)`` float32 metres on the
    half-size left-image grid; see
    :func:`euler_loading.loaders.cpu.driving_stereo.depth_full_size`."""
    return _map(_cpu.depth_full_size(path, meta, attributes=attributes))


@modality_meta(**_annotation("disparity", "1HW"))
def disparity(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a ``disparity-map-half-size`` PNG as ``(1, H, W)`` float32 half-size
    pixels (uint16 / 256, ``0`` invalid)."""
    return _map(_cpu.disparity(path, meta, attributes=attributes))


@modality_meta(**_annotation("disparity_full_size", "1HW"))
def disparity_full_size(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a ``disparity-map-full-size`` PNG as ``(1, H, W)`` float32 half-size
    pixels on the half-size left-image grid; see
    :func:`euler_loading.loaders.cpu.driving_stereo.disparity_full_size`."""
    return _map(_cpu.disparity_full_size(path, meta, attributes=attributes))


@modality_meta(**_annotation("read_intrinsics", "3x3"))
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the rectified ``(3, 3)`` float32 matrix ``P_rect_101[:, :3]``; see
    :func:`euler_loading.loaders.cpu.driving_stereo.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))
