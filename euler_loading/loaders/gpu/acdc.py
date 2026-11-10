"""GPU-oriented loader functions for ACDC.

Loaders return torch tensors created on the CPU: ``(3, H, W)`` float32 in
``[0, 1]``. The writers are shared with the CPU module, which accepts tensors
directly. See :mod:`euler_loading.loaders.cpu.acdc` for the layout.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import acdc as _cpu

__all__ = ["rgb", "reference_rgb", "write_rgb", "write_reference_rgb"]

write_rgb = _cpu.write_rgb
write_reference_rgb = _cpu.write_reference_rgb


def _chw(values) -> torch.Tensor:
    return torch.from_numpy(values).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load an adverse-condition image as ``(3, H, W)`` float32 in ``[0, 1]``."""
    return _chw(_cpu.rgb(path, meta, attributes=attributes))


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta=_cpu.reference_rgb._modality_meta["meta"],
)
def reference_rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a normal-condition reference image as ``(3, H, W)`` float32 in ``[0, 1]``;
    see :func:`euler_loading.loaders.cpu.acdc.reference_rgb`."""
    return _chw(_cpu.reference_rgb(path, meta, attributes=attributes))
