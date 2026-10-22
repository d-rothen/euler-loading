"""The dataset contract's sky definition, shared by the ``sky_mask`` loaders.

A ``semantic_segmentation`` dataset head must declare ``modality.meta.skyclass``:
the sky label as it appears in the segmentation file viewed as RGB. That is the
sky colour for colour-coded maps, and ``[v, v, v]`` for a single-channel label
``v``. ``MultiModalDataset`` passes the head's meta to every loader call, and a
file's ds-crawler attributes may carry its own ``skyclass``.

Resolution order: the file's ``skyclass``, the head's ``skyclass``, a
deprecated per-loader key (``sky_class_id``, ``sky_color``, ...), then the
loader's built-in default. Deprecated keys raise a :class:`DeprecationWarning`;
a deprecated key that ``skyclass`` overrides, and a resolved sky that differs
from the loader's default, are logged as warnings.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SKYCLASS_KEY = "skyclass"

RGB = tuple[int, int, int]

_logged: set[str] = set()


def _log_once(level: int, message: str) -> None:
    if message not in _logged:
        _logged.add(message)
        logger.log(level, message)


def as_rgb(value: Any, *, what: str) -> RGB:
    """Validate *value* as three integers in ``[0, 255]``."""
    items = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    if len(items) != 3 or not all(
        isinstance(item, (int, np.integer))
        and not isinstance(item, bool)
        and 0 <= item <= 255
        for item in items
    ):
        raise ValueError(f"{what} must be three integers in [0, 255], got {value!r}")
    return tuple(int(item) for item in items)


def resolve_skyclass(
    meta: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None = None,
    *,
    loader: str,
    default: RGB | None,
    legacy: Mapping[str, Callable[[Any], RGB]] | None = None,
) -> RGB:
    """Return the sky as an RGB triple; see the module docstring for the order.

    *legacy* maps each deprecated key the loader used to read to a function
    that converts its value to an RGB triple.
    """
    sources = [
        ("the file's attributes", attributes),
        ("the dataset head", meta),
    ]
    sources = [(label, source) for label, source in sources if isinstance(source, Mapping)]

    color: RGB | None = None
    origin = ""
    for label, source in sources:
        if source.get(SKYCLASS_KEY) is not None:
            color = as_rgb(
                source[SKYCLASS_KEY], what=f"{loader}: {SKYCLASS_KEY!r} in {label}"
            )
            origin = f"{SKYCLASS_KEY!r} in {label}"
            break

    for label, source in sources:
        for key, convert in (legacy or {}).items():
            if source.get(key) is None:
                continue
            value = convert(source[key])
            message = (
                f"{loader}: {key!r} in {label} is deprecated; declare "
                f"{SKYCLASS_KEY!r}: {list(value)} instead."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            if color is None:
                color, origin = value, f"the deprecated {key!r} in {label}"
                _log_once(logging.WARNING, message)
            else:
                verdict = "the same sky" if value == color else "a different sky"
                _log_once(
                    logging.WARNING,
                    f"{loader}: ignoring the deprecated {key!r} in {label} "
                    f"({list(value)}, {verdict}); {origin} ({list(color)}) "
                    "takes precedence.",
                )

    if color is None:
        if default is None:
            raise ValueError(
                f"{loader} needs {SKYCLASS_KEY!r} in the dataset head's meta: the "
                "sky label of the segmentation map as an RGB triple ([v, v, v] "
                "for label v)."
            )
        _log_once(
            logging.INFO,
            f"{loader}: no {SKYCLASS_KEY!r} supplied; using the built-in default "
            f"{list(default)}.",
        )
        return tuple(default)

    if default is not None and color != tuple(default):
        _log_once(
            logging.WARNING,
            f"{loader}: {origin} gives sky {list(color)}, which differs from "
            f"the loader's default {list(default)}; using {list(color)}.",
        )
    else:
        _log_once(logging.INFO, f"{loader}: sky is {list(color)}, from {origin}.")
    return color


def skyclass_label(color: RGB, *, loader: str) -> int:
    """Return the single-channel label ``v`` of a grey sky colour ``[v, v, v]``."""
    if len(set(color)) != 1:
        raise ValueError(
            f"{loader} reads single-channel class maps, so {SKYCLASS_KEY!r} "
            f"must be a grey [v, v, v]; got {list(color)}."
        )
    return color[0]


def grey(convert_label: Callable[[Any], int]) -> Callable[[Any], RGB]:
    """Adapt a deprecated single-label key's parser to an RGB triple ``[v, v, v]``."""
    return lambda value: (convert_label(value),) * 3


def match_rgb(rgb: np.ndarray, color: RGB) -> np.ndarray:
    """Return ``(H, W)`` bool, true where the ``(H, W, 3)`` image equals *color*."""
    return np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1)
