"""GPU-oriented loader functions for the Synscapes dataset.

Loaders return contiguous torch tensors: RGB is ``(3, H, W)`` float32 in
``[0, 1]``, depth is ``(1, H, W)`` float32 planar depth in metres, class and
instance IDs are ``(1, H, W)`` int64, sky masks are ``(1, H, W)`` bool, and
intrinsics are ``(3, 3)`` float32. Tensors are created on the CPU for transfer
to the training device, as with the other GPU-oriented loaders.

See :mod:`euler_loading.loaders.cpu.synscapes` for native file formats and
the optional OpenEXR dependency required by depth loading.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Union

import torch

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders.cpu import synscapes as _cpu


__all__ = [
    "rgb",
    "depth",
    "class_segmentation",
    "instance_segmentation",
    "sky_mask",
    "read_intrinsics",
]


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Load an RGB PNG as ``(3, H, W)`` float32 in ``[0, 1]``."""
    arr = _cpu.rgb(path, meta, attributes=attributes)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="1HW",
    file_formats=[".exr"],
    output_unit="meters",
    meta={"channel": "Z", "radial_depth": False, "scale_to_meters": 1.0},
)
def depth(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Load EXR ``Z`` values as ``(1, H, W)`` float32 planar depth in metres.

    Requires ``pip install 'euler-loading[synscapes]'``. Values, including
    non-finite depths, are preserved without resizing or sky masking.
    """
    arr = _cpu.depth(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta={"encoding": "single_channel", "label_ids": "cityscapes", "sky_class_id": 23},
)
def class_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Load ``img/class`` as ``(1, H, W)`` int64 original Cityscapes label IDs."""
    arr = _cpu.class_segmentation(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="instance_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta={"encoding": "rgb", "id_formula": "R + 256 * G + 65536 * B"},
)
def instance_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Decode RGB-packed instance IDs to ``(1, H, W)`` int64 labels."""
    arr = _cpu.instance_segmentation(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="1HW",
    file_formats=[".png"],
    meta={"sky_class_id": 23},
)
def sky_mask(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Return ``(1, H, W)`` bool, true where ``img/class`` has sky label ID 23."""
    arr = _cpu.sky_mask(path, meta, attributes=attributes)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    shape="3x3",
    file_formats=[".json"],
    meta={"source": "camera.intrinsic"},
)
def read_intrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Load ``camera.intrinsic`` from per-image JSON as ``(3, 3)`` float32.

    The matrix uses the metadata's resolution; scale it separately for
    ``img/rgb-2k`` with :func:`~euler_loading.resize_intrinsics`.
    """
    return torch.from_numpy(_cpu.read_intrinsics(path, meta, attributes=attributes)).contiguous()
