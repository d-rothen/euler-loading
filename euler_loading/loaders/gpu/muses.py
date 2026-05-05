"""GPU-oriented loader functions for the MUSES dataset.

MUSES stores RGB camera frames, reference frames, semantic labels, panoptic
labels, and lidar point clouds in separate modality folders.  These loaders
return PyTorch tensors for GPU-oriented training pipelines.

Return types
------------
- **rgb** / **reference_rgb** -- ``torch.FloatTensor`` of shape
  ``(3, H, W)`` in ``[0, 1]``.
- **semantic_segmentation** -- ``torch.LongTensor`` of shape ``(1, H, W)``
  containing raw Cityscapes ``labelIds`` or ``labelTrainIds``.
- **semantic_segmentation_color** -- ``torch.ByteTensor`` of shape
  ``(3, H, W)`` containing Cityscapes RGB colours.
- **panoptic_segmentation** -- ``torch.LongTensor`` of shape ``(1, H, W)``
  containing COCO-style panoptic segment IDs decoded from RGB.
- **lidar_point_cloud** / **sparse_depth** -- ``torch.DoubleTensor`` of shape
  ``(N, 6)`` with columns ``x, y, z, intensity, ring, timestamp``.
"""

from __future__ import annotations

import os
from typing import Any, BinaryIO, Union

import numpy as np
import torch
from PIL import Image

from euler_loading.loaders._annotations import modality_meta

_LIDAR_COLUMNS = ["x", "y", "z", "intensity", "ring", "timestamp"]


def _load_image_rgb(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load an image file as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_single_channel_labels(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a single-channel semantic label image as int64."""
    with Image.open(path) as image:
        arr = np.array(image, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(
            "MUSES semantic_segmentation expects a single-channel labelIds "
            "or labelTrainIds PNG. Use semantic_segmentation_color for "
            "labelColor PNG files."
        )
    return arr


def _load_rgb_uint8(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load an RGB image without normalisation."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _decode_panoptic_rgb(rgb: np.ndarray) -> np.ndarray:
    """Decode COCO-style RGB panoptic IDs as ``R + 256*G + 256^2*B``."""
    values = rgb.astype(np.int64)
    return values[:, :, 0] + 256 * values[:, :, 1] + 65536 * values[:, :, 2]


def _load_lidar_array(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a MUSES lidar ``.bin`` file as an ``(N, 6)`` float64 array."""
    if isinstance(path, (str, os.PathLike)):
        arr = np.fromfile(path, dtype=np.float64)
    else:
        arr = np.frombuffer(path.read(), dtype=np.float64).copy()

    if arr.size % 6 != 0:
        name = getattr(path, "name", path)
        raise ValueError(
            f"MUSES lidar file {name!r} contains {arr.size} float64 values, "
            "which is not divisible by 6."
        )
    return arr.reshape((-1, 6))


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a MUSES frame-camera RGB image as a float32 tensor in ``[0, 1]``."""
    arr = _load_image_rgb(path)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="CHW",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta={"source": "reference_frames"},
)
def reference_rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a MUSES clear-weather reference RGB image as float32 in ``[0, 1]``."""
    arr = _load_image_rgb(path)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta={
        "encoding": "single_channel",
        "supported_suffixes": ["labelIds", "labelTrainIds"],
    },
)
def semantic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load MUSES ``labelIds`` or ``labelTrainIds`` semantic labels."""
    arr = _load_single_channel_labels(path)
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="semantic_segmentation_color",
    dtype="uint8",
    shape="CHW",
    file_formats=[".png"],
    meta={"encoding": "rgb", "supported_suffixes": ["labelColor"]},
)
def semantic_segmentation_color(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load MUSES ``labelColor`` semantic labels as RGB uint8 values."""
    arr = _load_rgb_uint8(path)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@modality_meta(
    modality_type="panoptic_segmentation",
    dtype="int64",
    shape="1HW",
    file_formats=[".png"],
    meta={"encoding": "coco_rgb_id"},
)
def panoptic_segmentation(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a MUSES panoptic PNG as an integer segment-ID map."""
    arr = _decode_panoptic_rgb(_load_rgb_uint8(path))
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


@modality_meta(
    modality_type="point_cloud",
    dtype="float64",
    shape="NC",
    file_formats=[".bin"],
    meta={
        "columns": _LIDAR_COLUMNS,
        "coordinate_unit": "meters",
        "timestamp_unit": "seconds",
    },
)
def lidar_point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load a MUSES lidar point cloud as ``(N, 6)`` float64."""
    return torch.from_numpy(_load_lidar_array(path)).contiguous()


def point_cloud(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Alias for :func:`lidar_point_cloud`."""
    return torch.from_numpy(_load_lidar_array(path)).contiguous()


@modality_meta(
    modality_type="sparse_depth",
    dtype="float64",
    shape="NC",
    file_formats=[".bin"],
    meta={
        "representation": "point_cloud",
        "columns": _LIDAR_COLUMNS,
        "coordinate_unit": "meters",
        "timestamp_unit": "seconds",
    },
)
def sparse_depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> torch.Tensor:
    """Load MUSES lidar points for sparse-depth style workflows."""
    return torch.from_numpy(_load_lidar_array(path)).contiguous()
