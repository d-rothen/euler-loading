"""CPU-oriented loader functions for ACDC.

ACDC, the Adverse Conditions Dataset with Correspondences (Sakaridis et al.,
ICCV 2021), holds 8-bit PNG images of drives in fog, at night, in rain and in
snow, each with a corresponding image of the same place taken in normal
conditions::

    rgb_anon/<condition>/<split>/<sequence>/<sequence>_frame_<frame>_rgb_anon.png
    rgb_anon/<condition>/<split>_ref/<sequence>/<sequence>_frame_<frame>_rgb_ref_anon.png

``rgb_anon`` holds the anonymised images, which ACDC distributes by default.
A reference image carries the sequence and frame of its adverse-condition
image, so the two join by name. It was recorded on another drive, though: it
shows the same place but is not pixel-aligned with it.

Return types
------------
- **rgb**, **reference_rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.

Readers accept paths and binary streams.
"""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    get_target_name,
    mark_stream_supported,
    save_image,
    to_hwc_rgb,
    to_uint8,
)

__all__ = ["rgb", "reference_rgb", "write_rgb", "write_reference_rgb"]


def _load(path: Union[str, BinaryIO]) -> np.ndarray:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            data = stream.read()
    else:
        data = path.read()
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load an adverse-condition image (8-bit PNG) as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    return _load(path)


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
    meta={"source": "reference_frames", "condition": "normal"},
)
def reference_rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a normal-condition reference image as ``(H, W, 3)`` float32 in ``[0, 1]``.

    It shows the place of its adverse-condition image, recorded on another
    drive, so it is not pixel-aligned with it.
    """
    return _load(path)


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as an 8-bit PNG, the format :func:`rgb` reads.

    Accepts ``(H, W, 3)`` or ``(3, H, W)`` values in ``[0, 1]``.
    """
    if os.path.splitext(get_target_name(path))[1].lower() not in (".png", ""):
        raise ValueError(f"ACDC images are PNG, not {get_target_name(path)!r}")
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    save_image(path, image, format="PNG")


write_reference_rgb = write_rgb
