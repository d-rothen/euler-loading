"""GPU-oriented loader functions for the Synscapes dataset.

Loaders return contiguous torch tensors: RGB is ``(3, H, W)`` float32 in
``[0, 1]``, depth is ``(1, H, W)`` float32 planar depth in metres, class and
instance IDs are ``(1, H, W)`` int64, sky masks are ``(1, H, W)`` bool,
intrinsics are ``(3, 3)`` float32 and extrinsics ``(4, 4)`` float32. Tensors
are created on the CPU for transfer to the training device, as with the other
GPU-oriented loaders.

The writers are shared with the CPU module, which accepts tensors directly,
so writing is identical from either entry point.

See :mod:`euler_loading.loaders.cpu.synscapes` for native file formats, the
extrinsics conventions, and the optional OpenEXR dependency required by depth
loading and writing.
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
    "read_extrinsics",
    "write_rgb",
    "write_depth",
    "write_class_segmentation",
    "write_instance_segmentation",
    "write_sky_mask",
    "write_intrinsics",
    "write_extrinsics",
]

# The writers convert tensors through ``to_numpy``, so the CPU implementations
# are reused verbatim rather than duplicated here.
write_rgb = _cpu.write_rgb
write_depth = _cpu.write_depth
write_class_segmentation = _cpu.write_class_segmentation
write_instance_segmentation = _cpu.write_instance_segmentation
write_sky_mask = _cpu.write_sky_mask
write_intrinsics = _cpu.write_intrinsics
write_extrinsics = _cpu.write_extrinsics


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
    """Return ``(1, H, W)`` bool, true where ``img/class`` holds the sky label.

    Cityscapes ID ``23`` is the default; ``meta['sky_class_id']`` or the same
    key in per-file ``attributes`` selects another.
    """
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


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    shape="4x4",
    file_formats=[".json"],
    meta={
        "source": "camera.extrinsic",
        "dataset": "Synscapes",
        "angle_unit": "radians",
        "translation_unit": "meters",
        "ego_frame": _cpu._EGO_FRAME,
        "rotation_order": _cpu._ROTATION_ORDER,
        "default_transform_direction": "camera_to_ego",
        "default_camera_axes": "vehicle",
    },
)
def read_extrinsics(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Build a ``(4, 4)`` float32 rigid transform from ``camera.extrinsic``.

    Returns the camera's pose on the ego vehicle by default. Select another
    convention with the ``transform_direction`` and ``camera_axes`` keys
    documented on :func:`euler_loading.loaders.cpu.synscapes.read_extrinsics`.
    """
    return torch.from_numpy(_cpu.read_extrinsics(path, meta, attributes=attributes)).contiguous()
