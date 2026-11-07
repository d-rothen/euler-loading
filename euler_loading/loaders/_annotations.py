"""Annotations on loader functions.

The :func:`modality_meta` decorator attaches a ``_modality_meta`` dict to the
decorated function.  This metadata is read by the generate script
(:mod:`euler_loading.loaders.generate`) to produce ``loaders.json``.

The :func:`selects_per_sample` decorator lets a hierarchical loader narrow
its shared value to the sample being assembled; see :class:`SampleContext`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
_SELECTOR_ATTRIBUTE = "__euler_loading_select__"


@dataclass(frozen=True)
class SampleContext:
    """What a per-sample selector knows about the sample being assembled.

    Attributes:
        modality: Name of the hierarchical modality being selected.
        entry: ds-crawler file entry of the shared file that was loaded.
        id: The sample's file ID.
        full_id: The sample's qualified ID, ``/<hierarchy>/<id>``.
        hierarchy_path: The sample's hierarchy keys.
        files: ds-crawler file entries of the sample's regular modalities,
            keyed by modality name. Each holds at least ``path`` (relative to
            that modality's root) and ``id``.
    """

    modality: str
    entry: Mapping[str, Any]
    id: str
    full_id: str
    hierarchy_path: tuple[str, ...]
    files: Mapping[str, Mapping[str, Any]]


def selects_per_sample(
    select: Callable[[Any, SampleContext], Any],
) -> Callable[[_F], _F]:
    """Declare how a hierarchical loader's value narrows to one sample.

    Some datasets keep per-sample records in one shared file, such as a
    metadata table for a whole recording or split. Indexed as a hierarchical
    modality, that file applies to every sample beneath it, and its loader
    decodes the whole table once (hierarchical values are cached).
    ``select(value, context)`` is then called for each sample with a
    :class:`SampleContext` and returns that sample's record. It should raise
    ``KeyError`` when the table has no record for the sample.

    The selector is attached to the loader function, so it applies wherever
    that function is resolved: automatically from a dataset head, through
    ``Modality.function``, or as an explicit ``Modality.loader``.
    """

    def decorator(func: _F) -> _F:
        setattr(func, _SELECTOR_ATTRIBUTE, select)
        return func

    return decorator


def get_sample_selector(loader: Any) -> Callable[[Any, SampleContext], Any] | None:
    """Return the per-sample selector declared on *loader*, if any."""
    return getattr(loader, _SELECTOR_ATTRIBUTE, None)


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
