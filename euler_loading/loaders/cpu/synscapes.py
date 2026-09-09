"""CPU-oriented loader functions for the Synscapes dataset.

Native files live under ``img/rgb``, ``img/depth``, ``img/class``,
``img/instance`` and ``meta``. RGB returns an ``(H, W, 3)`` float32 array in
``[0, 1]``; depth returns ``(H, W)`` float32 planar depth in metres. Class
and instance labels return ``(H, W)`` int64 arrays, and the sky mask is bool.
Intrinsics return a ``(3, 3)`` float32 camera matrix at the resolution recorded
in the per-image JSON (1440x720 for the native dataset).

Only EXR depth loading requires the optional ``OpenEXR`` dependency, installed
with ``pip install 'euler-loading[synscapes]'``. All functions accept paths or
binary streams, including buffers supplied by zip-backed modalities.

Format references:
https://synscapes.on.liu.se/features.html
https://github.com/MartinHahner/FoggySynscapes/blob/main/source/Depth_processing/exr_to_mat.py
"""

from __future__ import annotations

import json
import os
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta


__all__ = [
    "rgb",
    "depth",
    "class_segmentation",
    "instance_segmentation",
    "sky_mask",
    "read_intrinsics",
]

_SKY_CLASS_ID = 23


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a Synscapes RGB PNG as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".exr"],
    output_unit="meters",
    meta={"channel": "Z", "radial_depth": False, "scale_to_meters": 1.0},
)
def depth(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load the EXR ``Z`` channel as ``(H, W)`` float32 planar depth in metres.

    Values are already metric and are preserved, including non-finite values.
    The EXR data window determines the shape; no resizing or sky masking is
    applied. Requires ``pip install 'euler-loading[synscapes]'``.
    """
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "Synscapes EXR depth loading requires OpenEXR. "
            "Install it with pip install 'euler-loading[synscapes]'."
        ) from exc

    source = os.fspath(path) if isinstance(path, (str, os.PathLike)) else path
    # InputFile supports binary streams; OpenEXR.File only accepts filenames.
    exr = OpenEXR.InputFile(source)
    try:
        header = exr.header()
        if "Z" not in header["channels"]:
            raise ValueError("Synscapes depth EXR must contain a 'Z' channel")
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        pixels = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(pixels, dtype=np.float32).reshape(height, width).copy()
    finally:
        exr.close()


@modality_meta(
    modality_type="semantic_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={
        "encoding": "single_channel",
        "label_ids": "cityscapes",
        "sky_class_id": _SKY_CLASS_ID,
    },
)
def class_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load ``img/class`` as ``(H, W)`` int64 Cityscapes label IDs.

    Original label IDs (including void labels) are preserved without mapping
    to training IDs. Palette indices are preserved for indexed PNGs.
    """
    with Image.open(path) as image:
        labels = np.array(image, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("Synscapes class segmentation must be a single-channel PNG")
    return labels


@modality_meta(
    modality_type="instance_segmentation",
    dtype="int64",
    shape="HW",
    file_formats=[".png"],
    meta={"encoding": "rgb", "id_formula": "R + 256 * G + 65536 * B"},
)
def instance_segmentation(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Decode ``img/instance`` RGB PNGs to ``(H, W)`` int64 instance IDs.

    Synscapes packs each ID as ``R + 256 * G + 65536 * B``.
    """
    with Image.open(path) as image:
        channels = np.array(image.convert("RGB"), dtype=np.int64)
    return channels[:, :, 0] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 2]


@modality_meta(
    modality_type="sky_mask",
    dtype="bool",
    shape="HW",
    file_formats=[".png"],
    meta={"sky_class_id": _SKY_CLASS_ID},
)
def sky_mask(
    path: Union[str, BinaryIO],
    meta: dict[str, Any] | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return ``(H, W)`` bool, true where ``img/class`` has sky label ID 23."""
    return class_segmentation(path, meta, attributes=attributes) == _SKY_CLASS_ID


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
) -> np.ndarray:
    """Load a ``(3, 3)`` float32 camera matrix from ``meta/<image_id>.json``.

    Uses ``camera.intrinsic`` fields ``fx``, ``fy``, ``u0`` and ``v0`` at the
    metadata's resolution. For ``img/rgb-2k``, scale the matrix separately with
    :func:`~euler_loading.resize_intrinsics`. Metadata files are per image and
    can be indexed as a regular modality alongside RGB and depth.
    """
    if isinstance(path, (str, os.PathLike)):
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.load(path)
    intrinsic = data["camera"]["intrinsic"]
    return np.array(
        [[intrinsic["fx"], 0.0, intrinsic["u0"]],
         [0.0, intrinsic["fy"], intrinsic["v0"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
