"""Dry-run the loaders a dataset declares, without building a dataset.

:class:`~euler_loading.MultiModalDataset` resolves a loader from each
modality's ds-crawler ``dataset-head.json`` contract — the ``loader`` and
``function`` fields of its ``addons.euler_loading`` entry — and hands it files
out of the indexed root, which may be a directory or a ``.zip`` archive.

This module walks that same pathway for one path and stops short of building a
dataset: it discovers the ds-crawler artifact sets below the path, resolves the
loader each one declares, decodes a few files with it and reports what came
back.  That is enough to tell whether an archive is loadable before a training
run depends on it.

The command line wrapper is :mod:`euler_loading.cli`::

    euler-loading /data/vkitti2
    python -m euler_loading /data/vkitti2
"""

from __future__ import annotations

import logging
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from ds_crawler import (
    get_dataset_contract,
    index_dataset_from_path,
    list_dataset_splits,
)
from ds_crawler.zip_utils import get_zip_root_prefix, is_zip_path

from ._ds_crawler_utils import (
    _has_root_metadata,
    _has_scoped_metadata,
    _list_metadata_scopes_safe,
    as_non_empty_str,
    load_index_output,
    read_zip_member,
)
from ._resolution import (
    LOADER_VARIANTS,
    _get_euler_loading_meta,
    _resolve_loader,
    _resolve_writer,
    loader_accepts_attributes,
    resolve_loader_module,
)
from .dataset import (
    Modality,
    _get_file_attributes,
    _get_index_meta,
    _get_index_tree,
    _load_with_optional_attributes,
    _loader_modality_type,
    _resolve_metadata_scope,
)
from .indexing import FileRecord, collect_files

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_SAMPLES",
    "SKY_SAMPLES",
    "DryRunReport",
    "ModalityCheck",
    "SampleCheck",
    "discover_modalities",
    "dry_run",
]

#: How far below the given path dataset roots are looked for.  Only directories
#: that carry no ds-crawler metadata of their own are descended into.
DEFAULT_MAX_DEPTH = 3

#: Files decoded per modality unless asked otherwise.
DEFAULT_SAMPLES = 1

#: Files a segmentation head's ``sky_mask`` decodes at least: enough to tell a
#: wrong ``skyclass`` (no sky anywhere) from a frame that shows no sky.
SKY_SAMPLES = 8

#: Head modality keys whose contract requires ``skyclass``.
_SKY_HEAD_KEYS = ("semantic_segmentation", "segmentation")

#: A sky fraction above this marks a file as suspiciously sky-covered.
_MOSTLY_SKY = 0.9


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleCheck:
    """One file that the modality's loader was pointed at."""

    path: str
    full_id: str
    ok: bool
    value: Optional[str] = None
    error: Optional[str] = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "full_id": self.full_id,
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "seconds": round(self.seconds, 6),
        }


@dataclass
class ModalityCheck:
    """The result of dry-running one ds-crawler artifact set."""

    label: str
    path: str
    metadata_scope: Optional[str] = None
    split: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    modality_key: Optional[str] = None
    declared_loader: Optional[str] = None
    loader: Optional[str] = None
    writer: Optional[str] = None
    hierarchical: Optional[bool] = None
    file_count: Optional[int] = None
    splits: list[str] = field(default_factory=list)
    samples: list[SampleCheck] = field(default_factory=list)
    sky: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(sample.ok for sample in self.samples)

    @property
    def decoded(self) -> int:
        return sum(1 for sample in self.samples if sample.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "metadata_scope": self.metadata_scope,
            "split": self.split,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "modality_key": self.modality_key,
            "declared_loader": self.declared_loader,
            "loader": self.loader,
            "writer": self.writer,
            "hierarchical": self.hierarchical,
            "file_count": self.file_count,
            "splits": list(self.splits),
            "ok": self.ok,
            "samples": [sample.to_dict() for sample in self.samples],
            "sky": self.sky,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class DryRunReport:
    """Everything one ``euler-loading`` dry run found."""

    root: str
    variant: str
    modalities: list[ModalityCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    common_ids: Optional[int] = None

    @property
    def ok(self) -> bool:
        return not self.errors and all(check.ok for check in self.modalities)

    def to_dict(self) -> dict[str, Any]:
        samples = [sample for check in self.modalities for sample in check.samples]
        return {
            "root": self.root,
            "variant": self.variant,
            "ok": self.ok,
            "common_ids": self.common_ids,
            "modalities": [check.to_dict() for check in self.modalities],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": {
                "modalities": len(self.modalities),
                "modalities_ok": sum(1 for check in self.modalities if check.ok),
                "files_decoded": sum(1 for sample in samples if sample.ok),
                "files_attempted": len(samples),
            },
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _artifact_sets(dataset_path: Path) -> list[Optional[str]]:
    """Return the metadata scopes holding a usable artifact set at a root.

    ``None`` stands for the unscoped ``.ds_crawler/`` layout.  A root can
    carry both, and each is independently loadable, so both are returned.

    Metadata that exists but cannot be read — malformed JSON, a corrupt
    archive — still marks a dataset root, reported as the unscoped one, so
    the per-modality check says what is wrong with it instead of the whole
    run dying on one bad archive.
    """
    try:
        scopes: list[Optional[str]] = []
        if _has_root_metadata(dataset_path):
            scopes.append(None)
        scopes.extend(
            scope
            for scope in _list_metadata_scopes_safe(dataset_path)
            if _has_scoped_metadata(dataset_path, scope)
        )
        return scopes
    except Exception:
        logger.debug("Unreadable ds-crawler metadata at %s.", dataset_path, exc_info=True)
        return [None]


def _dataset_roots(path: Path, max_depth: int) -> list[tuple[Path, list[Optional[str]]]]:
    """Return ``(root, scopes)`` pairs at or below *path*, nearest first.

    A directory carrying ds-crawler metadata is a root and is never descended
    into; a directory without any is searched for roots until *max_depth* is
    exhausted.
    """
    if is_zip_path(path):
        scopes = _artifact_sets(path)
        return [(path, scopes)] if scopes else []
    if not path.is_dir():
        return []
    scopes = _artifact_sets(path)
    if scopes:
        return [(path, scopes)]
    if max_depth <= 0:
        return []

    try:
        children = sorted(path.iterdir(), key=lambda item: item.name)
    except OSError:
        logger.debug("Cannot list %s.", path, exc_info=True)
        return []

    roots: list[tuple[Path, list[Optional[str]]]] = []
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir() or is_zip_path(child):
            roots.extend(_dataset_roots(child, max_depth - 1))
    return roots


def discover_modalities(
    path: Union[str, Path],
    *,
    metadata_scope: Optional[str] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[tuple[str, Optional[str]]]:
    """Find the ds-crawler artifact sets at or below *path*.

    Args:
        path: A modality root (directory or ``.zip``), or a folder holding
            several of them.
        metadata_scope: Check only this scope on every root found, instead of
            every artifact set each root carries.
        max_depth: How many directory levels below *path* to search.  Roots
            are not descended into, so a dataset's own scene directories are
            never walked.

    Returns:
        ``(root, metadata_scope)`` pairs, where *metadata_scope* is ``None``
        for the unscoped layout.
    """
    found: list[tuple[str, Optional[str]]] = []
    for root, scopes in _dataset_roots(Path(path), max_depth):
        if metadata_scope is not None:
            scopes = [metadata_scope]
        found.extend((str(root), scope) for scope in scopes)
    return found


def _label_for(root: Path, scope: Optional[str], base: Path) -> str:
    """Name a modality the way a ``Modality`` path would select it."""
    try:
        relative = root.relative_to(base)
        label = str(relative) if str(relative) != "." else root.name
    except ValueError:
        label = root.name
    label = label or str(root)
    return f"{label}#scope={scope}" if scope else label


# ---------------------------------------------------------------------------
# Value description
# ---------------------------------------------------------------------------

def _format_number(value: float, integral: bool) -> str:
    return f"{int(value)}" if integral else f"{value:.4g}"


def _numeric_summary(array: np.ndarray) -> Optional[str]:
    """Describe an array's value range, flagging non-finite entries."""
    if array.size == 0:
        return "empty"
    kind = array.dtype.kind
    if kind not in {"b", "i", "u", "f"}:
        return None

    finite = array
    non_finite = 0
    if kind == "f":
        mask = np.isfinite(array)
        non_finite = int(array.size - np.count_nonzero(mask))
        finite = array[mask]
        if finite.size == 0:
            return f"all {non_finite} values non-finite"

    integral = kind in {"b", "i", "u"}
    low = _format_number(float(finite.min()), integral)
    high = _format_number(float(finite.max()), integral)
    summary = f"in [{low}, {high}]"
    if non_finite:
        summary = f"{summary}, {non_finite} non-finite"
    return summary


def describe_value(value: Any, _depth: int = 0) -> str:
    """Render a loaded value as a one-line shape/dtype/range summary."""
    if value is None:
        return "None"

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        text = f"{type(value).__name__} {tuple(shape)} {str(dtype).replace('torch.', '')}"
        try:
            array = np.asarray(value)
        except Exception:  # pragma: no cover - exotic array-likes
            return text
        summary = _numeric_summary(array)
        return f"{text} {summary}" if summary else text

    mode = getattr(value, "mode", None)
    size = getattr(value, "size", None)
    if isinstance(mode, str) and isinstance(size, tuple):
        return f"{type(value).__name__} {mode} {size}"

    if isinstance(value, Mapping):
        if _depth >= 2:
            return f"dict({len(value)} keys)"
        shown = list(value.items())[:4]
        inner = ", ".join(f"{key}: {describe_value(item, _depth + 1)}" for key, item in shown)
        if len(value) > len(shown):
            inner = f"{inner}, ... (+{len(value) - len(shown)})"
        return "{" + inner + "}"

    if isinstance(value, (list, tuple)):
        kind = type(value).__name__
        if not value:
            return f"empty {kind}"
        if _depth >= 2:
            return f"{kind}({len(value)})"
        return f"{kind}({len(value)}) of {describe_value(value[0], _depth + 1)}"

    text = repr(value)
    if len(text) > 60:
        text = f"{text[:57]}..."
    return f"{type(value).__name__} {text}"


def _describe_exception(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return f"{type(exc).__name__}: {text}"


# ---------------------------------------------------------------------------
# Checking one modality
# ---------------------------------------------------------------------------

def _full_id(record: FileRecord) -> str:
    file_id = record.file_entry.get("id") or ""
    return "/" + "/".join((*record.hierarchy_path, file_id))


def _select_records(records: list[FileRecord], count: Optional[int]) -> list[FileRecord]:
    """Pick *count* files spread evenly across the index, in a stable order.

    Spreading beats taking the first *count* entries: a dataset's first scene
    is a poor witness for the rest of it.
    """
    ordered = sorted(
        records,
        key=lambda record: (record.hierarchy_path, record.file_entry.get("id") or ""),
    )
    if count is None or count >= len(ordered):
        return ordered
    if count <= 0:
        return []
    if count == 1:
        return ordered[:1]
    step = (len(ordered) - 1) / (count - 1)
    picked = sorted({int(round(index * step)) for index in range(count)})
    return [ordered[index] for index in picked]


def _loader_is_hierarchical(loader: Callable[..., Any]) -> Optional[bool]:
    meta = getattr(loader, "_modality_meta", None)
    if not isinstance(meta, Mapping):
        return None
    return bool(meta.get("hierarchical"))


def _open_archive(modality_path: str) -> tuple[Optional[zipfile.ZipFile], str]:
    """Open *modality_path* when it is a zip archive; return it and its root prefix."""
    if not is_zip_path(Path(modality_path)):
        return None, ""
    archive = zipfile.ZipFile(modality_path.rstrip("/"), "r")
    try:
        return archive, get_zip_root_prefix(Path(modality_path))
    except Exception:
        archive.close()
        raise


def _load_record(
    record: FileRecord,
    *,
    archive: Optional[zipfile.ZipFile],
    prefix: str,
    modality_path: str,
    loader: Callable[..., Any],
    meta: Optional[dict[str, Any]],
) -> tuple[str, Any, Optional[str], float]:
    """Load one record, returning ``(relative_path, value, error, seconds)``."""
    relative_path = record.file_entry.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        return "", None, "index entry has no 'path'", 0.0

    started = time.perf_counter()
    try:
        if archive is not None:
            file_or_path: Any = read_zip_member(archive, prefix, relative_path)
        else:
            file_or_path = f"{modality_path}/{relative_path}"
        value = _load_with_optional_attributes(
            loader,
            file_or_path,
            meta,
            _get_file_attributes(record.file_entry),
            accepts_attributes=loader_accepts_attributes(loader),
        )
    except Exception as exc:
        logger.debug(
            "Loading %s from %s failed.", relative_path, modality_path,
            exc_info=True,
        )
        return relative_path, None, _describe_exception(exc), time.perf_counter() - started
    return relative_path, value, None, time.perf_counter() - started


def _decode_records(
    check: ModalityCheck,
    *,
    records: list[FileRecord],
    modality_path: str,
    loader: Callable[..., Any],
    meta: Optional[dict[str, Any]],
) -> None:
    """Load every record, recording one :class:`SampleCheck` per file."""
    try:
        archive, prefix = _open_archive(modality_path)
    except Exception as exc:
        check.errors.append(f"archive: {_describe_exception(exc)}")
        return

    try:
        for record in records:
            relative_path, value, error, seconds = _load_record(
                record,
                archive=archive,
                prefix=prefix,
                modality_path=modality_path,
                loader=loader,
                meta=meta,
            )
            check.samples.append(
                SampleCheck(
                    path=relative_path,
                    full_id=_full_id(record),
                    ok=error is None,
                    value=describe_value(value) if error is None else None,
                    error=error,
                    seconds=seconds,
                )
            )
    finally:
        if archive is not None:
            archive.close()


def _sky_loader(
    check: ModalityCheck,
    *,
    index: Mapping[str, Any],
    loader: Callable[..., Any],
    variant: str,
) -> Optional[Callable[..., Any]]:
    """Return the loader that turns this artifact set into sky masks, if any."""
    if _loader_modality_type(loader) == "sky_mask":
        return loader
    if check.modality_key not in _SKY_HEAD_KEYS:
        return None
    module_name = as_non_empty_str((_get_euler_loading_meta(index) or {}).get("loader"))
    try:
        module = resolve_loader_module(module_name or "", variant=variant)
    except Exception:
        return None
    sky_mask = getattr(module, "sky_mask", None)
    if not callable(sky_mask):
        check.warnings.append(
            f"loader module {module_name!r} has no sky_mask, so the head's "
            "'skyclass' cannot be checked"
        )
        return None
    return sky_mask


def _check_sky(
    check: ModalityCheck,
    *,
    records: list[FileRecord],
    modality_path: str,
    index: Mapping[str, Any],
    loader: Callable[..., Any],
    variant: str,
) -> None:
    """Decode a segmentation head's files as sky masks and check the sky fraction.

    No sky in any sampled file, or a file that is almost all sky, is the mark
    of a wrong ``skyclass``; both are reported as warnings. A ``skyclass`` the
    ``sky_mask`` loader rejects is an error.
    """
    sky_loader = _sky_loader(check, index=index, loader=loader, variant=variant)
    if sky_loader is None or not records:
        return
    meta = _get_index_meta(index)
    try:
        archive, prefix = _open_archive(modality_path)
    except Exception:
        return  # _decode_records has reported it.

    fractions: list[tuple[str, float]] = []
    try:
        for record in records:
            relative_path, value, error, _seconds = _load_record(
                record,
                archive=archive,
                prefix=prefix,
                modality_path=modality_path,
                loader=sky_loader,
                meta=meta,
            )
            name = relative_path or _full_id(record)
            if error is not None:
                check.errors.append(f"sky_mask: {name}: {error}")
                return
            mask = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
            fractions.append((name, float(np.mean(mask))))
    finally:
        if archive is not None:
            archive.close()

    skyclass = (meta or {}).get("skyclass")
    declared = f"skyclass {list(skyclass)}" if skyclass is not None else "no skyclass"
    values = [fraction for _name, fraction in fractions]
    check.sky = (
        f"{declared}: sky in {sum(value > 0 for value in values)}/{len(values)} "
        f"files, {min(values):.1%}-{max(values):.1%} of pixels"
    )
    if not any(values):
        check.warnings.append(
            f"sky_mask finds no sky in any of the {len(values)} sampled files; "
            f"check the dataset head's {declared}"
        )
    for name, fraction in fractions:
        if fraction > _MOSTLY_SKY:
            check.warnings.append(
                f"sky_mask marks {fraction:.0%} of {name} as sky; check the "
                f"dataset head's {declared}"
            )


def _read_contract(check: ModalityCheck, index: Mapping[str, Any]) -> None:
    try:
        contract = get_dataset_contract(dict(index))
    except Exception:
        logger.debug("No readable dataset head for %s.", check.path, exc_info=True)
        return
    check.dataset_id = contract.dataset_id or None
    check.dataset_name = contract.dataset_name or None
    check.modality_key = contract.modality_key or None

    addon = contract.get_addon("euler_loading")
    if isinstance(addon, Mapping):
        declared = addon.get("loader")
        function = addon.get("function")
        if declared and function:
            check.declared_loader = f"{declared}.{function}"


def _check_modality(
    root: Union[str, Path],
    *,
    metadata_scope: Optional[str] = None,
    split: Optional[str] = None,
    label: Optional[str] = None,
    samples: Optional[int] = DEFAULT_SAMPLES,
    variant: str = "gpu",
) -> tuple[ModalityCheck, set[str]]:
    """Dry-run one ds-crawler artifact set.

    Returns the check plus the set of qualified file IDs it indexed, which
    :func:`dry_run` intersects across modalities the way a dataset would.
    """
    label = label or Path(str(root)).name
    check = ModalityCheck(
        label=label,
        path=str(root),
        metadata_scope=metadata_scope,
        split=split,
    )
    ids: set[str] = set()

    try:
        modality = _resolve_metadata_scope(
            label,
            Modality(str(root), split=split, metadata_scope=metadata_scope),
        )
    except Exception as exc:
        check.errors.append(_describe_exception(exc))
        return check, ids

    check.path = modality.path
    check.metadata_scope = modality.metadata_scope
    check.split = modality.split

    try:
        check.splits = list_dataset_splits(
            modality.path, metadata_scope=modality.metadata_scope,
        )
    except Exception:
        logger.debug("Cannot list splits for %s.", modality.path, exc_info=True)

    try:
        index = load_index_output(
            modality.path,
            split=modality.split,
            metadata_scope=modality.metadata_scope,
            index_dataset_from_path_fn=index_dataset_from_path,
        )
    except Exception as exc:
        check.errors.append(f"index: {_describe_exception(exc)}")
        return check, ids

    _read_contract(check, index)

    try:
        records = collect_files(_get_index_tree(index))
    except Exception as exc:
        check.errors.append(f"index: {_describe_exception(exc)}")
        return check, ids

    check.file_count = len(records)
    ids = {_full_id(record) for record in records}
    if not records:
        check.warnings.append("the index contains no files")

    # Materialized outputs are validated as records on every dataset
    # construction; a dry run that skipped it would miss the same failure.
    from .materialization import validate_materialized_output

    try:
        validate_materialized_output(
            modality.path,
            index,
            metadata_scope=modality.metadata_scope,
            subset=modality.split is not None,
        )
    except Exception as exc:
        check.errors.append(f"materialized records: {_describe_exception(exc)}")

    try:
        loader = _resolve_loader(
            modality_name=label, modality=modality, index=index, variant=variant,
        )
    except ImportError as exc:
        hint = (
            " Install the missing dependency, or dry-run the NumPy loaders "
            "with --cpu."
        )
        check.errors.append(f"loader: {_describe_exception(exc)}.{hint}")
        return check, ids
    except Exception as exc:
        check.errors.append(f"loader: {_describe_exception(exc)}")
        return check, ids

    check.loader = (
        f"{getattr(loader, '__module__', '?')}."
        f"{getattr(loader, '__qualname__', getattr(loader, '__name__', '?'))}"
    )
    check.hierarchical = _loader_is_hierarchical(loader)

    try:
        writer = _resolve_writer(
            modality_name=label, modality=modality, index=index, variant=variant,
        )
    except Exception:
        logger.debug("Cannot resolve a writer for %s.", modality.path, exc_info=True)
        writer = None
    if writer is not None:
        check.writer = getattr(writer, "__name__", None)

    selected = _select_records(records, samples)
    if selected:
        _decode_records(
            check,
            records=selected,
            modality_path=modality.path,
            loader=loader,
            meta=_get_index_meta(index),
        )
    if samples != 0:
        _check_sky(
            check,
            records=_select_records(
                records, None if samples is None else max(samples, SKY_SAMPLES)
            ),
            modality_path=modality.path,
            index=index,
            loader=loader,
            variant=variant,
        )
    return check, ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def dry_run(
    path: Union[str, Path],
    *,
    split: Optional[str] = None,
    metadata_scope: Optional[str] = None,
    samples: Optional[int] = DEFAULT_SAMPLES,
    variant: str = "gpu",
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> DryRunReport:
    """Resolve and exercise the loaders every modality below *path* declares.

    Args:
        path: A modality root (directory or ``.zip``) or a folder holding
            several of them.  Inline ``:split`` and ``#scope=`` selectors are
            accepted, exactly as on :class:`~euler_loading.Modality`.
        split: ds-crawler split to load instead of the canonical index.
        metadata_scope: Check only this scope, instead of every artifact set
            each root carries.
        samples: Files to decode per modality; ``0`` resolves loaders without
            decoding anything, ``None`` decodes every indexed file.
        variant: ``"gpu"`` (torch) or ``"cpu"`` (NumPy) built-in loaders.
        max_depth: How many directory levels below *path* to search for roots.

    Returns:
        A :class:`DryRunReport`.  Nothing is written, and no exception is
        raised for a dataset that fails to load — failures are collected on
        the report instead.
    """
    if variant not in LOADER_VARIANTS:
        available = ", ".join(LOADER_VARIANTS)
        raise ValueError(
            f"Unknown loader variant {variant!r}. Available variants: {available}"
        )

    # The path may carry the selectors Modality accepts, so parse it the same
    # way before anything touches the filesystem.
    requested = Modality(str(path), split=split, metadata_scope=metadata_scope)
    base = Path(requested.path)
    report = DryRunReport(root=requested.path, variant=variant)

    if not base.exists():
        report.errors.append(f"{requested.path} does not exist")
        return report
    if not base.is_dir() and not is_zip_path(base):
        report.errors.append(f"{requested.path} is neither a directory nor a .zip archive")
        return report

    found = discover_modalities(
        base, metadata_scope=requested.metadata_scope, max_depth=max_depth,
    )
    if not found:
        report.errors.append(
            f"No ds-crawler metadata found at or below {requested.path} "
            f"(searched {max_depth} level(s) deep). Index the dataset with "
            "ds-crawler first, or point at the modality root directly."
        )
        return report

    id_sets: dict[str, set[str]] = {}
    for root, scope in found:
        check, ids = _check_modality(
            root,
            metadata_scope=scope,
            split=requested.split,
            label=_label_for(Path(root), scope, base),
            samples=samples,
            variant=variant,
        )
        report.modalities.append(check)
        if check.hierarchical is False:
            id_sets[check.label] = ids

    if len(id_sets) > 1:
        common = set.intersection(*id_sets.values())
        report.common_ids = len(common)
        if not common:
            report.warnings.append(
                f"No file ID is shared by all {len(id_sets)} non-hierarchical "
                "modalities, so they cannot be combined into one "
                "MultiModalDataset."
            )

    return report
