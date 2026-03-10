"""Shared helpers for modality writer functions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image

_IMAGE_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def to_numpy(value: Any) -> np.ndarray:
    """Convert *value* to a NumPy array.

    Torch tensors are supported without importing torch directly.
    """
    maybe_tensor = value
    detach = getattr(maybe_tensor, "detach", None)
    if callable(detach):
        maybe_tensor = detach()

    to_cpu = getattr(maybe_tensor, "cpu", None)
    if callable(to_cpu):
        maybe_tensor = to_cpu()

    to_numpy_fn = getattr(maybe_tensor, "numpy", None)
    if callable(to_numpy_fn):
        try:
            return np.asarray(to_numpy_fn())
        except Exception:
            pass

    return np.asarray(maybe_tensor)


def to_hw(value: Any, *, name: str = "value") -> np.ndarray:
    """Return a 2D ``(H, W)`` array from ``(H, W)``, ``(1, H, W)``, or ``(H, W, 1)``."""
    arr = to_numpy(value)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    raise ValueError(
        f"{name} must have shape (H, W), (1, H, W), or (H, W, 1), got {arr.shape}"
    )


def to_hwc_rgb(value: Any, *, name: str = "value") -> np.ndarray:
    """Return a 3-channel ``(H, W, 3)`` array from ``(H, W, 3)`` or ``(3, H, W)``."""
    arr = to_numpy(value)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(
        f"{name} must have shape (H, W, 3) or (3, H, W), got {arr.shape}"
    )


def to_bool_mask(value: Any, *, name: str = "mask") -> np.ndarray:
    """Return a boolean ``(H, W)`` mask."""
    return to_hw(value, name=name).astype(bool)


def to_uint8(value: Any, *, scale_unit_range: bool) -> np.ndarray:
    """Convert an array to ``uint8``.

    When ``scale_unit_range`` is true and the input appears to be in ``[0, 1]``,
    values are scaled to ``[0, 255]`` before conversion.
    """
    arr = np.nan_to_num(
        to_numpy(value), nan=0.0, posinf=255.0, neginf=0.0
    )

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    if np.issubdtype(arr.dtype, np.floating) and arr.size > 0:
        min_value = float(arr.min())
        max_value = float(arr.max())
        if scale_unit_range and min_value >= 0.0 and max_value <= 1.0 + 1e-6:
            arr = arr * 255.0

    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def mark_stream_supported(func: Any) -> Any:
    """Annotate a writer as supporting binary stream targets."""
    setattr(func, "__euler_loading_supports_stream__", True)
    return func


def supports_stream_target(writer: Any) -> bool:
    """Return whether *writer* accepts binary stream targets."""
    return bool(getattr(writer, "__euler_loading_supports_stream__", False))


def get_target_name(target: str | os.PathLike[str] | BinaryIO) -> str:
    """Return a filename suitable for extension detection."""
    name = getattr(target, "name", target)
    return str(name)


def ensure_parent(target: str | os.PathLike[str] | BinaryIO) -> None:
    """Create parent directories for filesystem targets, no-op for streams."""
    if isinstance(target, (str, os.PathLike)):
        Path(target).parent.mkdir(parents=True, exist_ok=True)


def save_image(
    target: str | os.PathLike[str] | BinaryIO,
    image: Image.Image,
    *,
    format: str | None = None,
) -> None:
    """Save a PIL image to a path or binary stream."""
    ensure_parent(target)
    if isinstance(target, (str, os.PathLike)):
        image.save(target)
        return

    image_format = format
    if image_format is None:
        ext = os.path.splitext(get_target_name(target))[1].lower()
        image_format = _IMAGE_FORMATS.get(ext)
    if image_format is None:
        raise ValueError(
            "Binary image targets must expose a filename extension via "
            "target.name or provide an explicit format."
        )
    image.save(target, format=image_format)


def write_text(
    target: str | os.PathLike[str] | BinaryIO,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text to a filesystem path or binary stream."""
    ensure_parent(target)
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding=encoding) as f:
            f.write(text)
        return
    target.write(text.encode(encoding))


def write_json(
    target: str | os.PathLike[str] | BinaryIO,
    payload: Any,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write JSON to a filesystem path or binary stream."""
    text = json.dumps(payload)
    write_text(target, text, encoding=encoding)
