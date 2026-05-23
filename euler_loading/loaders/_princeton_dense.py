"""Shared helpers for Princeton DENSE / SeeingThroughFog loaders."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

_DEFAULT_BAYER_PATTERN = "GBRG"
_DEFAULT_RAW_MAX_VALUE = 4095.0
LIDAR_COLUMNS = ["x", "y", "z", "intensity", "ring"]


def _value_from_context(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> Any:
    for source in (attributes, meta):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            if key in source:
                return source[key]
    return None


def _rgb_scale(
    raw: np.ndarray,
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> float:
    value = _value_from_context(
        meta,
        attributes,
        ("rgb_max_value", "raw_max_value", "white_level"),
    )
    if value is not None:
        return float(value)

    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8:
            return 255.0
        if raw.size and float(raw.max()) <= _DEFAULT_RAW_MAX_VALUE:
            return _DEFAULT_RAW_MAX_VALUE
        return float(np.iinfo(raw.dtype).max)

    return 1.0


def _bayer_pattern(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
) -> str:
    value = _value_from_context(meta, attributes, ("bayer_pattern", "cfa_pattern"))
    if value is None:
        return _DEFAULT_BAYER_PATTERN
    pattern = str(value).upper()
    if len(pattern) != 4 or any(channel not in "RGB" for channel in pattern):
        raise ValueError("bayer_pattern must be a 4-character pattern using R, G, and B")
    return pattern


def _interpolate_channel(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values_pad = np.pad(values, 1, mode="edge")
    mask_pad = np.pad(mask.astype(np.float32), 1, mode="edge")

    sums = np.zeros_like(values, dtype=np.float32)
    counts = np.zeros_like(values, dtype=np.float32)
    for y_offset in range(3):
        for x_offset in range(3):
            y = slice(y_offset, y_offset + values.shape[0])
            x = slice(x_offset, x_offset + values.shape[1])
            sums += values_pad[y, x]
            counts += mask_pad[y, x]

    interpolated = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float32),
        where=counts > 0,
    )
    return np.where(mask, values, interpolated)


def _demosaic_bayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    raw_f32 = raw.astype(np.float32, copy=False)
    channel_values = {
        "R": np.zeros(raw.shape, dtype=np.float32),
        "G": np.zeros(raw.shape, dtype=np.float32),
        "B": np.zeros(raw.shape, dtype=np.float32),
    }
    channel_masks = {
        "R": np.zeros(raw.shape, dtype=bool),
        "G": np.zeros(raw.shape, dtype=bool),
        "B": np.zeros(raw.shape, dtype=bool),
    }

    positions = (
        (slice(0, None, 2), slice(0, None, 2), pattern[0]),
        (slice(0, None, 2), slice(1, None, 2), pattern[1]),
        (slice(1, None, 2), slice(0, None, 2), pattern[2]),
        (slice(1, None, 2), slice(1, None, 2), pattern[3]),
    )
    for row_slice, col_slice, channel in positions:
        channel_values[channel][row_slice, col_slice] = raw_f32[row_slice, col_slice]
        channel_masks[channel][row_slice, col_slice] = True

    channels = [
        _interpolate_channel(channel_values[channel], channel_masks[channel])
        for channel in ("R", "G", "B")
    ]
    return np.stack(channels, axis=-1)


def load_rgb_array(
    path: Union[str, BinaryIO],
    meta: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Load a SeeingThroughFog RGB TIFF as an ``(H, W, 3)`` float32 array."""
    with Image.open(path) as image:
        raw = np.array(image)

    if raw.ndim == 2:
        rgb = _demosaic_bayer(raw, _bayer_pattern(meta, attributes))
        scale = _rgb_scale(raw, meta, attributes)
    elif raw.ndim == 3 and raw.shape[-1] >= 3:
        rgb = raw[..., :3].astype(np.float32, copy=False)
        scale = _rgb_scale(raw, meta, attributes)
    else:
        name = getattr(path, "name", path)
        raise ValueError(f"Unsupported Princeton RGB image shape for {name!r}: {raw.shape}")

    if scale <= 0:
        raise ValueError(f"RGB scale must be positive, got {scale!r}")
    return np.clip(rgb / scale, 0.0, 1.0).astype(np.float32, copy=False)


def load_sparse_depth_array(path: Union[str, BinaryIO]) -> np.ndarray:
    """Load a SeeingThroughFog lidar ``.bin`` file as ``(N, 5)`` float32."""
    if isinstance(path, (str, os.PathLike)):
        values = np.fromfile(path, dtype=np.float32)
    else:
        values = np.frombuffer(path.read(), dtype=np.float32).copy()

    if values.size % 5 != 0:
        name = getattr(path, "name", path)
        raise ValueError(
            f"Princeton sparse-depth file {name!r} contains {values.size} "
            "float32 values, which is not divisible by 5."
        )
    return values.reshape((-1, 5))
