"""Decorator for annotating loader functions with modality metadata.

The :func:`modality_meta` decorator attaches a ``_modality_meta`` dict to the
decorated function.  This metadata is read by the generate script
(:mod:`euler_loading.loaders.generate`) to produce ``loaders.json``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def modality_meta(
    *,
    modality_type: str,
    dtype: str,
    hierarchical: bool = False,
    shape: str,
    file_formats: list[str] | None = None,
    output_range: list[float | int] | None = None,
    output_unit: str | None = None,
    requires_meta: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """Annotate a loader function with modality metadata.

    Parameters
    ----------
    modality_type:
        Canonical modality name (e.g. ``"rgb"``, ``"dense_depth"``).
    dtype:
        Output dtype after loading (``"float32"``, ``"int64"``, ``"bool"``).
    hierarchical:
        ``True`` for modalities that apply at a higher hierarchy level
        (e.g. calibration per scene/camera rather than per frame).
    shape:
        Layout string for this variant (``"HWC"``, ``"CHW"``, ``"3x3"``, …).
    file_formats:
        Accepted file extensions (e.g. ``[".png"]``, ``[".npz"]``).
    output_range:
        Value range of the loaded tensor (e.g. ``[0.0, 1.0]`` for normalised
        RGB).  ``None`` when the range is scene-dependent or not applicable.
    output_unit:
        Physical unit of the output values (e.g. ``"meters"``).  ``None``
        when not applicable.
    requires_meta:
        List of ``meta`` dict keys that **must** be provided at load time
        (e.g. ``["sky_mask"]``).  ``None`` or ``[]`` when the loader is
        self-contained.
    meta:
        Domain-specific metadata dict (encoding, scale factors, …).
    """

    def decorator(func: _F) -> _F:
        func._modality_meta = {  # type: ignore[attr-defined]
            "type": modality_type,
            "dtype": dtype,
            "hierarchical": hierarchical,
            "shape": shape,
            "file_formats": file_formats or [],
            "output_range": output_range,
            "output_unit": output_unit,
            "requires_meta": requires_meta or [],
            "meta": meta or {},
        }
        return func

    return decorator
