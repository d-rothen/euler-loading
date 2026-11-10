"""GPU-oriented loader functions for the CEREMA Pixel-Accurate Depth Benchmark.

Loaders return torch tensors created on the CPU: RGB ``(3, H, W)`` and gated
slices ``(1, H, W)`` float32 in ``[0, 1]``, depth ``(1, H, W)`` float32 meters
(``0`` invalid) and intrinsics ``(3, 3)`` float32. The writers are shared with
the CPU module, which accepts tensors directly. See
:mod:`euler_loading.loaders.cpu.cerema` for the release layout and how the
method outputs are converted to depth.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import cerema as _cpu

__all__ = [
    "rgb",
    "gated",
    "depth",
    "sgm_depth",
    "psmnet_depth",
    "monodepth_depth",
    "read_intrinsics",
    "write_rgb",
    "write_gated",
    "write_depth",
    "write_intrinsics",
]

write_rgb = _cpu.write_rgb
write_gated = _cpu.write_gated
write_depth = _cpu.write_depth
write_intrinsics = _cpu.write_intrinsics


def _map(values) -> torch.Tensor:
    return torch.from_numpy(values).unsqueeze(0)


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a rectified stereo image as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="gated",
    dtype="float32",
    shape="1HW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta=_cpu.gated._modality_meta["meta"],
)
def gated(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a gated slice as ``(1, H, W)`` float32 in ``[0, 1]``, dark level removed; see
    :func:`euler_loading.loaders.cpu.cerema.gated`."""
    return _map(_cpu.gated(path, meta, attributes=attributes))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta=_cpu.depth._modality_meta["meta"],
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a depth map as ``(1, H, W)`` float32 meters, ``0`` invalid; see
    :func:`euler_loading.loaders.cpu.cerema.depth`."""
    return _map(_cpu.depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta=_cpu.sgm_depth._modality_meta["meta"],
)
def sgm_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert an ``sgm`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.sgm_depth`."""
    return _map(_cpu.sgm_depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta=_cpu.psmnet_depth._modality_meta["meta"],
)
def psmnet_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert a ``psmnet`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.psmnet_depth`."""
    return _map(_cpu.psmnet_depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".npz"],
    output_unit="meters",
    meta=_cpu.monodepth_depth._modality_meta["meta"],
)
def monodepth_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert a ``monodepth`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.monodepth_depth`."""
    return _map(_cpu.monodepth_depth(path, meta, attributes=attributes))


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta=_cpu.read_intrinsics._modality_meta["meta"],
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the rectified ``(3, 3)`` float32 camera matrix; see
    :func:`euler_loading.loaders.cpu.cerema.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))
