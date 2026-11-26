"""Annotations that make a function an euler-loading loader.

This is the public API a loader module needs, built-in or third-party
(see the README section "Extending: your own loaders")::

    from euler_loading import modality_meta

    @modality_meta(modality_type="depth", dtype="float32", shape="HW",
                   file_formats=[".npy"], output_unit="meters")
    def depth(path, meta=None, *, attributes=None):
        ...

* :func:`modality_meta` declares what a function loads. It is what makes a
  function nameable from a dataset head (``addons.euler_loading.function``),
  what ``loader_report()`` describes, and what receipts accept as a decoder.
* :func:`selects_per_sample` lets a hierarchical loader narrow its shared
  value to the sample being assembled; see :class:`SampleContext`.
* :func:`get_modality_meta` reads a function's annotation back.

Writers live in the same module, named ``write_<function>`` (or
``write_<x>`` for a reader named ``read_<x>``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

#: ``modality_type``, ``dtype`` or ``shape`` of a decoder whose values are
#: declared per file rather than fixed by the function (materialized outputs).
DECLARED = "declared"
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


def get_modality_meta(loader: Any) -> dict[str, Any] | None:
    """Return a copy of the :func:`modality_meta` annotation on *loader*, or ``None``.

    Keys: ``type``, ``dtype``, ``hierarchical``, ``shape`` (the value layout,
    e.g. ``"HWC"``, ``"CHW"``, ``"NC"``, ``"3x3"``), ``file_formats``,
    ``output_range``, ``output_unit``, ``requires_meta`` and ``meta``.
    """
    annotation = getattr(loader, "_modality_meta", None)
    if not isinstance(annotation, dict):
        return None
    return copy.deepcopy(annotation)


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
        The persisted modality key the function loads (e.g. ``"rgb"``,
        ``"depth"``). euler-dataset-contract resolves it to a modality id;
        ``loader_report()`` marks keys the installed contract does not know
        as ``"unknown"``.
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

    for label, value in (("modality_type", modality_type), ("dtype", dtype), ("shape", shape)):
        if not isinstance(value, str) or not value:
            raise TypeError(f"modality_meta({label}=...) must be a non-empty string, got {value!r}")
    for label, values in (("file_formats", file_formats), ("requires_meta", requires_meta)):
        if values is not None and (
            isinstance(values, str) or not all(isinstance(v, str) and v for v in values)
        ):
            raise TypeError(f"modality_meta({label}=...) must be a list of strings, got {values!r}")
    for extension in file_formats or []:
        if not extension.startswith("."):
            raise ValueError(
                f"modality_meta(file_formats=...) lists extensions with their dot, "
                f"such as '.png'; got {extension!r}"
            )
    if output_range is not None and len(output_range) != 2:
        raise ValueError(f"modality_meta(output_range=...) must be [low, high], got {output_range!r}")
    if output_unit is not None and not isinstance(output_unit, str):
        raise TypeError(f"modality_meta(output_unit=...) must be a string, got {output_unit!r}")
    if meta is not None and not isinstance(meta, Mapping):
        raise TypeError(f"modality_meta(meta=...) must be a mapping, got {type(meta).__name__}")

    def decorator(func: _F) -> _F:
        func._modality_meta = {  # type: ignore[attr-defined]
            "type": modality_type,
            "dtype": dtype,
            "hierarchical": hierarchical,
            "shape": shape,
            "file_formats": list(file_formats or []),
            "output_range": output_range,
            "output_unit": output_unit,
            "requires_meta": list(requires_meta or []),
            "meta": dict(meta or {}),
        }
        return func

    return decorator


__all__ = [
    "DECLARED",
    "SampleContext",
    "get_modality_meta",
    "get_sample_selector",
    "modality_meta",
    "selects_per_sample",
]
