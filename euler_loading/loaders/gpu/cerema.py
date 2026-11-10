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


def _like(cpu_function, **overrides):
    """The CPU function's annotation, with the tensor layout."""
    stored = cpu_function._modality_meta
    annotation = {
        "modality_type": stored["type"],
        "dtype": stored["dtype"],
        "hierarchical": stored["hierarchical"],
        "shape": stored["shape"],
        "file_formats": stored["file_formats"],
        "output_range": stored["output_range"],
        "output_unit": stored["output_unit"],
        "requires_meta": stored["requires_meta"],
        "meta": stored["meta"],
    }
    annotation.update(overrides)
    return modality_meta(**annotation)


@_like(_cpu.rgb, shape="CHW")
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a rectified stereo image as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return torch.from_numpy(_cpu.rgb(path, meta, attributes=attributes)).permute(2, 0, 1).contiguous()


@_like(_cpu.gated, shape="1HW")
def gated(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a gated slice as ``(1, H, W)`` float32 in ``[0, 1]``, dark level removed; see
    :func:`euler_loading.loaders.cpu.cerema.gated`."""
    return _map(_cpu.gated(path, meta, attributes=attributes))


@_like(_cpu.depth, shape="1HW")
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a depth map as ``(1, H, W)`` float32 meters, ``0`` invalid; see
    :func:`euler_loading.loaders.cpu.cerema.depth`."""
    return _map(_cpu.depth(path, meta, attributes=attributes))


@_like(_cpu.sgm_depth, shape="1HW")
def sgm_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert an ``sgm`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.sgm_depth`."""
    return _map(_cpu.sgm_depth(path, meta, attributes=attributes))


@_like(_cpu.psmnet_depth, shape="1HW")
def psmnet_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert a ``psmnet`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.psmnet_depth`."""
    return _map(_cpu.psmnet_depth(path, meta, attributes=attributes))


@_like(_cpu.monodepth_depth, shape="1HW")
def monodepth_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Convert a ``monodepth`` output to ``(1, 1024, 1920)`` metric depth; see
    :func:`euler_loading.loaders.cpu.cerema.monodepth_depth`."""
    return _map(_cpu.monodepth_depth(path, meta, attributes=attributes))


@_like(_cpu.read_intrinsics)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Return the rectified ``(3, 3)`` float32 camera matrix; see
    :func:`euler_loading.loaders.cpu.cerema.read_intrinsics`."""
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes))
