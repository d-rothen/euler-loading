from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Callable

try:
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:
    class _BaseDataset:  # type: ignore[no-redef]
        """Fallback when PyTorch is not installed."""

from ds_crawler import (
    DatasetWriter,
    ZipDatasetWriter,
    get_dataset_contract,
    index_dataset_from_path,
    load_dataset_config,
)
from ds_crawler.zip_utils import get_zip_root_prefix, is_zip_path

try:
    from ds_crawler import get_layout_addon as _ds_crawler_get_layout_addon
except ImportError:  # pragma: no cover - compatibility with older ds-crawler
    _ds_crawler_get_layout_addon = None

from ._ds_crawler_utils import load_index_output, parse_path_with_split
from ._metadata import _build_runlog_entry, _get_ds_crawler_descriptor
from ._resolution import (
    _resolve_loader,
    _resolve_writer,
    loader_accepts_attributes,
    resolve_loader_module,
    resolve_writer_module,
)
from ._writing import (
    OutputDestination,
    _build_writer_full_id,
    _destination_location,
    _destination_rel_exists,
    _resolve_output_destination,
    _write_value_to_destination,
    create_dataset_writer_from_index,
)
from .indexing import (
    FileRecord,
    collect_files,
    collect_hierarchical_files,
    match_hierarchical_files,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Modality",
    "MultiModalDataset",
    "create_dataset_writer_from_index",
    "resolve_loader_module",
    "resolve_writer_module",
]


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    name = getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _describe_transform(transform: Any) -> str:
    describe = getattr(transform, "describe", None)
    if callable(describe):
        try:
            return str(describe())
        except Exception:
            logger.debug(
                "Failed to describe transform %s; falling back to callable name.",
                _callable_name(transform),
                exc_info=True,
            )
    return _callable_name(transform)


def _get_index_tree(index_output: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dataset tree from a ds-crawler output payload."""
    index_tree = index_output.get("index")
    if isinstance(index_tree, Mapping):
        dataset_tree = index_tree.get("dataset")
        if isinstance(dataset_tree, Mapping):
            return dict(dataset_tree)
        return dict(index_tree)

    dataset_tree = index_output.get("dataset")
    if isinstance(dataset_tree, Mapping):
        return dict(dataset_tree)

    raise KeyError(
        "ds-crawler output must contain either an 'index' tree or a 'dataset' tree."
    )


def _get_index_meta(index_output: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return modality meta from a ds-crawler output payload when available."""
    try:
        meta = get_dataset_contract(dict(index_output)).meta
    except Exception:
        meta = None
    if isinstance(meta, Mapping):
        return dict(meta)

    raw_meta = index_output.get("meta")
    if isinstance(raw_meta, Mapping):
        return dict(raw_meta)
    return None


def _get_index_layout(index_output: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the optional ``euler_layout`` addon from an index payload."""
    if _ds_crawler_get_layout_addon is not None:
        try:
            return _ds_crawler_get_layout_addon(dict(index_output))
        except Exception:
            logger.debug("Failed to parse euler_layout addon.", exc_info=True)

    head = index_output.get("head")
    if isinstance(head, Mapping):
        addons = head.get("addons")
        if isinstance(addons, Mapping):
            layout = addons.get("euler_layout")
            if isinstance(layout, Mapping):
                return dict(layout)

    layout = index_output.get("euler_layout")
    if isinstance(layout, Mapping):
        return dict(layout)
    return None


def _layout_axis(layout: Mapping[str, Any] | None, axis: str) -> dict[str, Any] | None:
    if not isinstance(layout, Mapping):
        return None
    value = layout.get(axis)
    return dict(value) if isinstance(value, Mapping) else None


def _layout_axis_name(layout: Mapping[str, Any] | None, axis: str) -> str | None:
    value = _layout_axis(layout, axis)
    name = value.get("name") if value else None
    return name if isinstance(name, str) and name else None


def _layout_has_variant_axis(layout: Mapping[str, Any] | None) -> bool:
    return _layout_axis(layout, "variant_axis") is not None


def _layout_family(layout: Mapping[str, Any] | None) -> str | None:
    if not isinstance(layout, Mapping):
        return None
    family = layout.get("family")
    return family if isinstance(family, str) and family else None


def _layout_families_compatible(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    left_family = _layout_family(left)
    right_family = _layout_family(right)
    return (
        left_family is None
        or right_family is None
        or left_family == right_family
    )


def _index_hierarchy_separator(index_output: Mapping[str, Any]) -> str | None:
    indexing = index_output.get("indexing")
    hierarchy = indexing.get("hierarchy") if isinstance(indexing, Mapping) else None
    if isinstance(hierarchy, Mapping):
        separator = hierarchy.get("separator")
        if isinstance(separator, str) and separator:
            return separator
    return None


def _tree_has_hierarchy_axis(
    node: Mapping[str, Any],
    *,
    axis_name: str,
    separator: str,
) -> bool:
    prefix = f"{axis_name}{separator}"
    children = node.get("children")
    if not isinstance(children, Mapping):
        return False
    for key, child in children.items():
        if isinstance(key, str) and key.startswith(prefix):
            return True
        if isinstance(child, Mapping) and _tree_has_hierarchy_axis(
            child, axis_name=axis_name, separator=separator,
        ):
            return True
    return False


def _index_has_hierarchy_axis(
    index_output: Mapping[str, Any],
    *,
    axis_name: str,
) -> bool:
    separator = _index_hierarchy_separator(index_output)
    if not separator:
        return False
    try:
        tree = _get_index_tree(index_output)
    except KeyError:
        return False
    return _tree_has_hierarchy_axis(
        tree, axis_name=axis_name, separator=separator,
    )


def _get_file_attributes(file_entry: Mapping[str, Any]) -> dict[str, Any] | None:
    attributes = file_entry.get("attributes")
    if isinstance(attributes, Mapping):
        return dict(attributes)
    return None


def _attributes_cache_key(attributes: Mapping[str, Any] | None) -> str:
    if attributes is None:
        return "null"
    return json.dumps(attributes, sort_keys=True, separators=(",", ":"), default=repr)


def _load_with_optional_attributes(
    loader: Callable[..., Any],
    file_or_path: Any,
    meta: dict[str, Any] | None,
    attributes: dict[str, Any] | None,
    *,
    accepts_attributes: bool,
) -> Any:
    if accepts_attributes:
        loader_attributes = dict(attributes) if attributes is not None else None
        return loader(file_or_path, meta, attributes=loader_attributes)
    return loader(file_or_path, meta)


def _extract_id_schema(index_output: Mapping[str, Any]) -> dict[str, Any]:
    """Pull id/hierarchy separators from a ds-crawler output payload.

    Supports both modern (``indexing.id.join_char``) and legacy
    (top-level ``id_regex_join_char``) layouts. Missing keys fall back to
    ds-crawler defaults so the returned dict is always usable.
    """
    indexing = index_output.get("indexing")
    indexing_map = indexing if isinstance(indexing, Mapping) else {}
    id_cfg = indexing_map.get("id")
    id_map = id_cfg if isinstance(id_cfg, Mapping) else {}
    hierarchy_cfg = indexing_map.get("hierarchy")
    hierarchy_map = hierarchy_cfg if isinstance(hierarchy_cfg, Mapping) else {}

    id_join_char = id_map.get("join_char") or index_output.get("id_regex_join_char") or "+"
    hierarchy_separator = hierarchy_map.get("separator") or "-"
    id_regex = id_map.get("regex") or index_output.get("id_regex")
    hierarchy_regex = hierarchy_map.get("regex") or index_output.get("hierarchy_regex")

    schema: dict[str, Any] = {
        "hierarchy_separator": hierarchy_separator,
        "id_join_char": id_join_char,
    }
    if id_regex:
        schema["id_regex"] = id_regex
    if hierarchy_regex:
        schema["hierarchy_regex"] = hierarchy_regex
    return schema


#TODO: might want to add slots=True in a 3.10+ only codebase
@dataclass(frozen=True)
class Modality:
    """Specification of a single data modality.

    Attributes:
        path: Absolute path to the dataset root directory for this modality.
              Must contain a ``ds-crawler.config`` file (or a cached
              ``output.json`` from a prior indexing run).  A colon-separated
              split suffix is also accepted (e.g. ``/data/ds.zip:train``);
              the suffix is extracted and used as the ``split`` parameter.
        origin_path: Optional original path string before any copying or symlinking for i.e. slurm staging.  
                This is not used by euler-loading itself but can be useful for experiment 
                logging to retain references to the original dataset location.
        loader: Optional callable that takes an absolute file path and returns
                the loaded data (e.g. a numpy array, a PIL Image, a tensor).
                When *None* (the default), the loader is resolved automatically
                from ``euler_loading.loader`` and ``euler_loading.function`` in
                the ds-crawler index, using the GPU variant of the matching
                predefined loader.
        writer: Optional callable that writes a loaded/predicted modality back
                to disk. Expected signature is
                ``(target: str | BinaryIO, value: Any, meta: dict[str, Any] | None) -> None``.
                When *None* (the default), the writer is resolved automatically
                from ``euler_loading.loader`` + ``euler_loading.function`` using
                writer naming conventions (e.g. ``write_depth`` for ``depth``,
                ``write_intrinsics`` for ``read_intrinsics``).
        used_as: Optional semantic role for experiment logging
                 (e.g. ``"input"``, ``"target"``, ``"condition"``).
        slot: Optional fully-qualified slot name for experiment logging
              (e.g. ``"dehaze.input.rgb"``).
        modality_type: Optional modality type override
                       (e.g. ``"rgb"``, ``"depth"``, ``"segmentation"``).
        hierarchy_scope: Optional scope label for hierarchical modalities
                         (e.g. ``"scene_camera"``).
        applies_to: Optional list of regular-modality names a hierarchical
                    modality applies to.
        split: Optional inline split name. When set, euler-loading reads the
               full ds-crawler metadata from the modality root and overlays the
               dataset payload from ``.ds_crawler/split_<name>.json``.
        keyed_by: Optional join configuration used only when this modality is
                  passed under ``MultiModalDataset(keyed_modalities=...)``.
                  Recognized keys:

                  - ``key_name`` (optional): the named-group prefix expected
                    at the regular modality's deepest hierarchy key
                    (e.g. ``"file_id"`` for keys like ``"file_id:000…025"``).
                    When omitted, auto-detected by scanning the anchor
                    modality's samples for a single unique prefix; raises
                    when zero or multiple distinct prefixes exist.
                  - ``modality`` (optional): name of the regular modality
                    whose hierarchy provides the lookup key.  When omitted,
                    the only regular modality is used; with multiple
                    regulars the field is required.

                  ``keyed_by`` itself may be omitted entirely (or set to
                  ``{}`` / ``None``) to use both auto-detection paths.
        cache: Optional opt-in/out for in-memory caching of loaded values.
               Only meaningful for hierarchical and keyed modalities;
               regular modalities load once per sample and never cache.

               - ``True``: every distinct file is decoded at most once
                 per dataset lifetime, then served from memory.
               - ``False``: every access re-reads and re-decodes the
                 file.  Use this for large per-sample tensors where the
                 cache would balloon (e.g. multi-hundred-GB GT depth
                 keyed modalities).
               - ``None`` (default): hierarchical modalities default to
                 ``True`` (small calibration files benefit), keyed
                 modalities default to ``False`` (safe for the
                 augmentation use case where the keyed modality is
                 typically the same size as a regular sample).
        collapse_single: When ``True`` for a hierarchical modality, return the
                 single matched hierarchical value directly instead of a
                 ``{file_id: value}`` dict.  This is useful for shared GT
                 tensors loaded once per source sample across augmentations.
                 A sample with more than one matched file raises.
        metadata: Optional arbitrary metadata. Keys under
                  ``metadata["euler_loading"]`` are treated as
                  euler-loading-specific defaults.
    """

    path: str
    origin_path: str | None = None
    loader: Callable[..., Any] | None = None
    writer: Callable[..., Any] | None = None
    used_as: str | None = None
    slot: str | None = None
    modality_type: str | None = None
    hierarchy_scope: str | None = None
    applies_to: list[str] | None = None
    split: str | None = None
    keyed_by: Mapping[str, str] | None = None
    cache: bool | None = None
    collapse_single: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed_path, path_split = parse_path_with_split(self.path)
        if path_split is not None:
            if self.split is not None:
                raise ValueError(
                    f"Modality path {self.path!r} contains an inline split "
                    f"({path_split!r}) but an explicit split={self.split!r} "
                    f"was also provided. Use one or the other, not both."
                )
            object.__setattr__(self, "path", parsed_path)
            object.__setattr__(self, "split", path_split)


class MultiModalDataset(_BaseDataset):
    """PyTorch Dataset that loads synchronized multi-modal samples.

    Indexes each modality using *ds-crawler*, intersects file IDs across all
    modalities, and returns dicts containing every loaded modality.

    Args:
        modalities: Mapping of modality name to :class:`Modality` spec.
                    Example: ``{"rgb": Modality("/data/rgb", load_rgb)}``.
        hierarchical_modalities: Optional mapping of modality name to
                    :class:`Modality` spec for modalities whose files live at
                    intermediate hierarchy levels (e.g. per-scene intrinsics
                    files).  These do **not** participate in ID intersection.
                    Each sample will contain a dict ``{file_id: loaded_result}``
                    with all hierarchical-modality files at or above the
                    sample's hierarchy level.  Loaded results are cached so
                    shared files are parsed only once.
        keyed_modalities: Optional mapping of modality name to
                    :class:`Modality` spec for modalities joined to a regular
                    sample via the value of the regular sample's deepest
                    hierarchy key.  Concretely: for a regular sample at
                    ``hierarchy_path = (..., "file_id:abc")``, the keyed
                    modality returns the file at the *parent* hierarchy with
                    ``id == "abc"``.  Unlike hierarchical modalities, a keyed
                    modality contributes a single loaded value per sample
                    (not a ``{file_id: ...}`` dict).  Each keyed
                    :class:`Modality` must set ``keyed_by={"key_name": "..."}``;
                    see :class:`Modality` for details.
        transforms: Optional list of callables.  Each receives the full sample
                    dict and must return a (possibly modified) sample dict.  The
                    sample dict acts as a context object giving transforms
                    access to all modalities and metadata.

    Raises:
        ValueError: If *modalities* is empty or no common file IDs exist across
                    the provided modalities.

    Example::

        dataset = MultiModalDataset(
            modalities={
                "rgb":   Modality("/data/vkitti2/rgb",   load_rgb),
                "depth": Modality("/data/vkitti2/depth", load_depth),
            },
            hierarchical_modalities={
                "intrinsics_file": Modality("/data/vkitti2/intrinsics", load_txt),
            },
            transforms=[mask_sky_in_depth],
        )

        sample = dataset[0]
        # sample["rgb"], sample["depth"], sample["intrinsics_file"]["intrinsic"], ...
    """

    def modality_paths(self) -> dict[str, dict[str, str | None]]:
        """Return modality names mapped to their current path metadata.

        Each entry always contains ``path`` and ``origin_path``. A ``split``
        key is included when the modality is configured to load an inline
        ds-crawler split.
        """
        res = {}
        for name, mod in self._modalities.items():
            path = mod.path
            origin = mod.origin_path
            entry: dict[str, str | None] = {"path": path, "origin_path": origin}
            if mod.split is not None:
                entry["split"] = mod.split
            res[name] = entry
        return res

    def hierarchical_modality_paths(self) -> dict[str, dict[str, str | None]]:
        """Return hierarchical modality names mapped to their path metadata.

        Each entry always contains ``path`` and ``origin_path``. A ``split``
        key is included when the modality is configured to load an inline
        ds-crawler split.
        """
        res = {}
        for name, mod in self._hierarchical_modalities.items():
            path = mod.path
            origin = mod.origin_path
            entry: dict[str, str | None] = {"path": path, "origin_path": origin}
            if mod.split is not None:
                entry["split"] = mod.split
            res[name] = entry
        return res

    def keyed_modality_paths(self) -> dict[str, dict[str, str | None]]:
        """Return keyed modality names mapped to their path metadata.

        Each entry always contains ``path`` and ``origin_path``. A ``split``
        key is included when the modality is configured to load an inline
        ds-crawler split.  ``keyed_by`` (the resolved anchor + key_name)
        is included so consumers can recover the join configuration.
        """
        res: dict[str, dict[str, str | None]] = {}
        for name, mod in self._keyed_modalities.items():
            entry: dict[str, str | None] = {
                "path": mod.path,
                "origin_path": mod.origin_path,
            }
            if mod.split is not None:
                entry["split"] = mod.split
            join_cfg = self._keyed_join_config.get(name, {})
            if join_cfg:
                # Cast to str via dict copy so the value type matches.
                entry["keyed_by_modality"] = join_cfg["modality"]
                entry["keyed_by_key_name"] = join_cfg["key_name"]
            res[name] = entry
        return res

    def describe_for_runlog(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return structured dataset metadata for runlog ``meta.json``.

        The returned dict is shaped as::

            {
              "modalities": {
                "<name>": {
                  "path": "...",
                  "origin_path": "...",
                  "split": "...",
                  "used_as": "...",
                  "slot": "...",
                  "modality_type": "...",
                },
              },
              "hierarchical_modalities": {
                "<name>": {
                  "path": "...",
                  "origin_path": "...",
                  "split": "...",
                  "used_as": "...",
                  "slot": "...",
                  "modality_type": "...",
                  "hierarchy_scope": "...",
                  "applies_to": [...],
                },
              },
            }

        Resolution order for each field is:
        1) explicit values on :class:`Modality`
        2) modality ``metadata["euler_loading"]``
        3) ds-crawler config properties ``properties["euler_loading"]``
        4) naming / hierarchy heuristics
        """
        regular_names = list(self._modalities.keys())
        descriptor_cache: dict[str, dict[str, Any]] = {}

        modalities: dict[str, dict[str, Any]] = {}
        for name, modality in self._modalities.items():
            descriptor = _get_ds_crawler_descriptor(
                path=modality.path,
                index_output=self._index_outputs.get(name),
                cache=descriptor_cache,
                load_dataset_config_fn=load_dataset_config,
            )
            modalities[name] = _build_runlog_entry(
                name=name,
                modality=modality,
                descriptor=descriptor,
                is_hierarchical=False,
                regular_modality_names=regular_names,
                hierarchy_levels=None,
            )

        hierarchical_modalities: dict[str, dict[str, Any]] = {}
        for name, modality in self._hierarchical_modalities.items():
            descriptor = _get_ds_crawler_descriptor(
                path=modality.path,
                index_output=self._hierarchical_index_outputs.get(name),
                cache=descriptor_cache,
                load_dataset_config_fn=load_dataset_config,
            )
            hierarchy_levels = list(self._hierarchical_lookups.get(name, {}).keys())
            hierarchical_modalities[name] = _build_runlog_entry(
                name=name,
                modality=modality,
                descriptor=descriptor,
                is_hierarchical=True,
                regular_modality_names=regular_names,
                hierarchy_levels=hierarchy_levels,
            )

        keyed_modalities: dict[str, dict[str, Any]] = {}
        for name, modality in self._keyed_modalities.items():
            descriptor = _get_ds_crawler_descriptor(
                path=modality.path,
                index_output=self._keyed_index_outputs.get(name),
                cache=descriptor_cache,
                load_dataset_config_fn=load_dataset_config,
            )
            entry = _build_runlog_entry(
                name=name,
                modality=modality,
                descriptor=descriptor,
                is_hierarchical=False,
                regular_modality_names=regular_names,
                hierarchy_levels=None,
            )
            cfg = self._keyed_join_config.get(name, {})
            if cfg:
                entry["keyed_by"] = {
                    "modality": cfg["modality"],
                    "key_name": cfg["key_name"],
                }
            keyed_modalities[name] = entry

        return {
            "modalities": modalities,
            "hierarchical_modalities": hierarchical_modalities,
            "keyed_modalities": keyed_modalities,
        }

    @classmethod
    def from_layout(
        cls,
        modalities: dict[str, Modality],
        *,
        primary: str | None = None,
        transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        strict_layout: bool = False,
    ) -> "MultiModalDataset":
        """Create a dataset by planning modality roles from ``euler_layout``.

        When the primary modality has a variant axis (for example fog
        augmentations grouped under ``file_id:<source>``) and an auxiliary
        modality has the same sample axis but no variant axis, the auxiliary is
        loaded as shared per-source data.  If the auxiliary index already
        exposes that sample axis as a hierarchy key, it is wired as a collapsed
        hierarchical modality.  Otherwise it falls back to a keyed join.

        Existing constructor semantics remain unchanged; this helper is the
        opt-in path for layout-aware loading.
        """
        if not modalities:
            raise ValueError("At least one modality must be provided.")

        indexed: dict[str, dict[str, Any]] = {}
        layouts: dict[str, dict[str, Any] | None] = {}
        for name, modality in modalities.items():
            index = load_index_output(
                modality.path,
                split=modality.split,
                index_dataset_from_path_fn=index_dataset_from_path,
            )
            indexed[name] = index
            layouts[name] = _get_index_layout(index)

        if primary is None:
            primary = next(
                (
                    name for name, layout in layouts.items()
                    if _layout_has_variant_axis(layout)
                ),
                next(iter(modalities)),
            )
        if primary not in modalities:
            raise ValueError(
                f"primary modality {primary!r} is not in modalities; "
                f"known: {sorted(modalities)}"
            )

        primary_layout = layouts[primary]
        primary_sample_axis = _layout_axis_name(primary_layout, "sample_axis")
        primary_has_variants = _layout_has_variant_axis(primary_layout)

        regular_modalities: dict[str, Modality] = {primary: modalities[primary]}
        hierarchical_modalities: dict[str, Modality] = {}
        keyed_modalities: dict[str, Modality] = {}

        for name, modality in modalities.items():
            if name == primary:
                continue

            layout = layouts[name]
            sample_axis = _layout_axis_name(layout, "sample_axis")
            shared_with_primary = (
                primary_has_variants
                and primary_sample_axis is not None
                and sample_axis == primary_sample_axis
                and not _layout_has_variant_axis(layout)
                and _layout_families_compatible(primary_layout, layout)
            )

            if not shared_with_primary:
                regular_modalities[name] = modality
                continue

            if _index_has_hierarchy_axis(indexed[name], axis_name=sample_axis):
                hierarchical_modalities[name] = replace(
                    modality, collapse_single=True,
                )
                continue

            if strict_layout:
                raise ValueError(
                    f"Modality {name!r} shares sample_axis={sample_axis!r} with "
                    f"variant primary {primary!r}, but its ds-crawler index has "
                    "no matching hierarchy axis. Re-index it with that axis as "
                    "hierarchy, or call from_layout(..., strict_layout=False) "
                    "to use a keyed fallback."
                )

            keyed_by = {
                **dict(modality.keyed_by or {}),
                "modality": primary,
                "key_name": sample_axis,
            }
            keyed_modalities[name] = replace(modality, keyed_by=keyed_by)

        return cls(
            modalities=regular_modalities,
            hierarchical_modalities=hierarchical_modalities or None,
            keyed_modalities=keyed_modalities or None,
            transforms=transforms,
        )

    def __init__(
        self,
        modalities: dict[str, Modality],
        hierarchical_modalities: dict[str, Modality] | None = None,
        transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        keyed_modalities: dict[str, Modality] | None = None,
        strict_keyed: bool = False,
    ) -> None:
        super().__init__()

        if not modalities:
            raise ValueError("At least one modality must be provided.")

        self._modalities = modalities
        self._hierarchical_modalities = hierarchical_modalities or {}
        self._keyed_modalities: dict[str, Modality] = keyed_modalities or {}
        self._strict_keyed = strict_keyed
        self._transforms = transforms or []
        self._index_outputs: dict[str, dict[str, Any]] = {}
        self._hierarchical_index_outputs: dict[str, dict[str, Any]] = {}
        self._keyed_index_outputs: dict[str, dict[str, Any]] = {}

        # -- Index each modality and build ID → FileRecord lookups -----------
        self._lookups: dict[str, dict[str, FileRecord]] = {}
        self._resolved_loaders: dict[str, Callable[..., Any]] = {}
        self._loaders_accept_attributes: dict[str, bool] = {}
        self._resolved_writers: dict[str, Callable[..., Any] | None] = {}

        for name, modality in modalities.items():
            index = load_index_output(
                modality.path,
                split=modality.split,
                index_dataset_from_path_fn=index_dataset_from_path,
            )
            self._index_outputs[name] = index
            self._resolved_loaders[name] = _resolve_loader(
                modality_name=name, modality=modality, index=index,
            )
            self._loaders_accept_attributes[name] = loader_accepts_attributes(
                self._resolved_loaders[name]
            )
            self._resolved_writers[name] = _resolve_writer(
                modality_name=name, modality=modality, index=index,
            )
            records = collect_files(_get_index_tree(index))
            lookup = {
                "/".join(rec.hierarchy_path + (rec.file_entry["id"],)): rec
                for rec in records
            }
            self._lookups[name] = lookup
            logger.info(
                "Modality '%s': indexed %d files from %s%s",
                name,
                len(lookup),
                modality.path,
                f" (split={modality.split})" if modality.split is not None else "",
            )

        # -- Index hierarchical modalities -----------------------------------
        self._hierarchical_lookups: dict[
            str, dict[tuple[str, ...], list[dict[str, Any]]]
        ] = {}

        for name, modality in self._hierarchical_modalities.items():
            index = load_index_output(
                modality.path,
                split=modality.split,
                index_dataset_from_path_fn=index_dataset_from_path,
            )
            self._hierarchical_index_outputs[name] = index
            self._resolved_loaders[name] = _resolve_loader(
                modality_name=name, modality=modality, index=index,
            )
            self._loaders_accept_attributes[name] = loader_accepts_attributes(
                self._resolved_loaders[name]
            )
            self._resolved_writers[name] = _resolve_writer(
                modality_name=name, modality=modality, index=index,
            )
            files_by_level = collect_hierarchical_files(_get_index_tree(index))
            self._hierarchical_lookups[name] = files_by_level
            total_files = sum(len(f) for f in files_by_level.values())
            logger.info(
                "Hierarchical modality '%s': indexed %d files across %d "
                "hierarchy levels from %s%s",
                name,
                total_files,
                len(files_by_level),
                modality.path,
                f" (split={modality.split})" if modality.split is not None else "",
            )

        # -- Compute common IDs (intersection across regular modalities) -----
        all_id_sets = [set(lookup.keys()) for lookup in self._lookups.values()]
        self._common_ids: list[str] = sorted(set.intersection(*all_id_sets))

        if not self._common_ids:
            modality_counts = {
                name: len(lookup) for name, lookup in self._lookups.items()
            }
            message = (
                f"No common IDs found across modalities. "
                f"Per-modality file counts: {modality_counts}"
            )
            hints = self._diagnose_keyed_join_candidates()
            if hints:
                message += "\n" + "\n".join(hints)
            raise ValueError(message)

        for name, lookup in self._lookups.items():
            total = len(lookup)
            matched = len(self._common_ids)
            if matched < total:
                logger.warning(
                    "Modality '%s': %d/%d files matched across modalities "
                    "(%d unmatched)",
                    name,
                    matched,
                    total,
                    total - matched,
                )

        # -- Index keyed modalities + validate joins -------------------------
        self._keyed_lookups: dict[
            str, dict[tuple[tuple[str, ...], str], FileRecord]
        ] = {}
        self._keyed_join_config: dict[str, dict[str, str]] = {}

        for name, modality in self._keyed_modalities.items():
            config = self._resolve_keyed_join(name, modality)
            index = load_index_output(
                modality.path,
                split=modality.split,
                index_dataset_from_path_fn=index_dataset_from_path,
            )
            self._keyed_index_outputs[name] = index
            self._resolved_loaders[name] = _resolve_loader(
                modality_name=name, modality=modality, index=index,
            )
            self._loaders_accept_attributes[name] = loader_accepts_attributes(
                self._resolved_loaders[name]
            )
            self._resolved_writers[name] = _resolve_writer(
                modality_name=name, modality=modality, index=index,
            )
            self._keyed_lookups[name] = {
                (rec.hierarchy_path, rec.file_entry["id"]): rec
                for rec in collect_files(_get_index_tree(index))
            }
            self._keyed_join_config[name] = config
            logger.info(
                "Keyed modality '%s' (key_name=%s, anchor=%s): indexed %d "
                "files from %s%s",
                name,
                config["key_name"],
                config["modality"],
                len(self._keyed_lookups[name]),
                modality.path,
                f" (split={modality.split})" if modality.split is not None else "",
            )

        # Validate that every common id has a matching keyed-record.  Drop
        # (or raise on) unmatched samples.  Done once here so __getitem__ is
        # exception-free in the happy path.
        if self._keyed_modalities:
            self._common_ids = self._filter_common_ids_by_keyed_joins()

        # -- Zip archive support ------------------------------------------------
        self._zip_modalities: set[str] = set()
        self._zip_prefixes: dict[str, str] = {}
        self._zip_handles: dict[tuple[str, int], zipfile.ZipFile] = {}

        for name, modality in (
            list(modalities.items())
            + list(self._hierarchical_modalities.items())
            + list(self._keyed_modalities.items())
        ):
            if is_zip_path(modality.path):
                self._zip_modalities.add(name)
                self._zip_prefixes[name] = get_zip_root_prefix(
                    Path(modality.path)
                )

        # -- Caches ----------------------------------------------------------
        # Per-modality caches; ``None`` disables caching for that modality.
        # Hierarchical default: enabled (calibration-style small files).
        # Keyed default: disabled (per-sample-size files, OOM risk).
        self._caches: dict[str, dict[str, Any] | None] = {}
        for name, modality in self._hierarchical_modalities.items():
            self._caches[name] = {} if (
                modality.cache if modality.cache is not None else True
            ) else None
        for name, modality in self._keyed_modalities.items():
            self._caches[name] = {} if (
                modality.cache if modality.cache is not None else False
            ) else None

        # -- Optional transform binding -------------------------------------
        for transform in self._transforms:
            bind = getattr(transform, "bind_to_dataset", None)
            if callable(bind):
                bind(self)
        if self._transforms:
            logger.warning("Configured %d sample transform(s):", len(self._transforms))
            for idx, transform in enumerate(self._transforms, start=1):
                logger.warning(
                    "Transform %d/%d: %s",
                    idx,
                    len(self._transforms),
                    _describe_transform(transform),
                )
        else:
            logger.warning("Configured 0 sample transforms.")

    # ------------------------------------------------------------------
    # Keyed-modality helpers
    # ------------------------------------------------------------------

    def _resolve_keyed_join(
        self, name: str, modality: Modality,
    ) -> dict[str, str]:
        """Validate ``modality.keyed_by`` and resolve its anchor, separator,
        and key_name.

        ``key_name`` may be omitted; in that case it is auto-detected by
        scanning the anchor modality's deepest hierarchy keys for a single
        unique named-group prefix.  Multiple distinct prefixes (or none)
        require an explicit ``key_name``.

        Returns a dict with stable keys ``modality``, ``key_name``,
        ``separator`` for use at sample-load time.
        """
        cfg = dict(modality.keyed_by or {})
        anchor = cfg.get("modality")
        if anchor is None:
            if len(self._modalities) != 1:
                raise ValueError(
                    f"keyed_modalities['{name}'].keyed_by.modality is required "
                    f"because there are {len(self._modalities)} regular "
                    f"modalities; cannot infer the anchor."
                )
            anchor = next(iter(self._modalities))
        if anchor not in self._modalities:
            raise ValueError(
                f"keyed_modalities['{name}'] references unknown regular "
                f"modality {anchor!r}; known: {sorted(self._modalities)}."
            )
        anchor_index = self._index_outputs[anchor]
        indexing_cfg = anchor_index.get("indexing", {})
        hierarchy_cfg = indexing_cfg.get("hierarchy", {}) if isinstance(
            indexing_cfg, Mapping
        ) else {}
        separator = (
            hierarchy_cfg.get("separator") if isinstance(hierarchy_cfg, Mapping) else None
        )
        if not separator:
            raise ValueError(
                f"Regular modality {anchor!r} has no "
                f"indexing.hierarchy.separator; cannot decode keyed-join "
                f"key for keyed modality {name!r}."
            )

        key_name = cfg.get("key_name")
        if not key_name:
            key_name = self._infer_keyed_key_name(
                anchor=anchor, separator=separator, keyed_modality_name=name,
            )

        return {
            "modality": anchor,
            "key_name": key_name,
            "separator": separator,
        }

    def _infer_keyed_key_name(
        self, *, anchor: str, separator: str, keyed_modality_name: str,
    ) -> str:
        """Auto-detect the deepest-key prefix used by *anchor*'s samples.

        Returns the unique named-group prefix when every sample's deepest
        hierarchy key uses the same one.  Raises with a precise message
        when the anchor has zero hierarchy, no separator-bearing keys, or
        more than one distinct prefix.
        """
        prefixes: set[str] = set()
        skipped_no_sep = 0
        skipped_no_hier = 0
        for record in self._lookups[anchor].values():
            hier = record.hierarchy_path
            if not hier:
                skipped_no_hier += 1
                continue
            deepest = hier[-1]
            if separator not in deepest:
                skipped_no_sep += 1
                continue
            prefixes.add(deepest.split(separator, 1)[0])
        if not prefixes:
            raise ValueError(
                f"Cannot auto-detect key_name for keyed_modalities"
                f"['{keyed_modality_name}']: anchor modality {anchor!r} has "
                f"no hierarchy keys with separator {separator!r} "
                f"(skipped: {skipped_no_hier} hierarchy-less, "
                f"{skipped_no_sep} without separator). Set "
                f"keyed_by={{'key_name': ...}} explicitly."
            )
        if len(prefixes) > 1:
            raise ValueError(
                f"Cannot auto-detect key_name for keyed_modalities"
                f"['{keyed_modality_name}']: anchor modality {anchor!r} has "
                f"multiple distinct deepest-key prefixes {sorted(prefixes)!r}. "
                f"Set keyed_by={{'key_name': ...}} explicitly."
            )
        return next(iter(prefixes))

    def _decode_keyed_lookup_key(
        self,
        keyed_modality_name: str,
        anchor_record: FileRecord,
    ) -> tuple[tuple[str, ...], str]:
        """Return ``(parent_prefix, key_value)`` for a regular sample."""
        config = self._keyed_join_config[keyed_modality_name]
        hier = anchor_record.hierarchy_path
        if not hier:
            raise ValueError(
                f"Sample at path {anchor_record.file_entry.get('path')!r} has "
                f"no hierarchy; keyed modality {keyed_modality_name!r} "
                f"requires a non-empty parent hierarchy."
            )
        deepest = hier[-1]
        sep = config["separator"]
        if sep not in deepest:
            raise ValueError(
                f"Hierarchy key {deepest!r} has no {sep!r} separator; "
                f"keyed modality {keyed_modality_name!r} cannot extract "
                f"key_name={config['key_name']!r}."
            )
        key_name, key_value = deepest.split(sep, 1)
        if key_name != config["key_name"]:
            raise ValueError(
                f"Keyed modality {keyed_modality_name!r}: expected the "
                f"deepest hierarchy key to start with "
                f"{config['key_name']!r}, got {key_name!r} (full key: "
                f"{deepest!r})."
            )
        return hier[:-1], key_value

    def _diagnose_keyed_join_candidates(self) -> list[str]:
        """Scan regular-modality pairs for likely keyed relationships.

        Used only on the "No common IDs found" failure path to nudge the
        user toward ``keyed_modalities=`` when one of their modalities
        looks like an augmentation of another.  Each returned line is a
        single hint; the caller appends them to the error message.

        For every ordered pair (anchor, candidate), the candidate is
        treated *as if* it were a keyed modality of the anchor: the
        candidate's records are indexed by ``(hierarchy_path, id)`` and
        the anchor's deepest-key values are decoded against that lookup.
        A non-zero match count plus a single decoded ``key_name`` prefix
        is reported.
        """
        hints: list[str] = []
        for candidate_name, candidate_lookup in self._lookups.items():
            candidate_by_parent_id: dict[
                tuple[tuple[str, ...], str], FileRecord
            ] = {
                (rec.hierarchy_path, rec.file_entry["id"]): rec
                for rec in candidate_lookup.values()
            }
            for anchor_name, anchor_lookup in self._lookups.items():
                if anchor_name == candidate_name:
                    continue
                anchor_index = self._index_outputs[anchor_name]
                indexing_cfg = anchor_index.get("indexing", {})
                hierarchy_cfg = (
                    indexing_cfg.get("hierarchy", {})
                    if isinstance(indexing_cfg, Mapping) else {}
                )
                sep = (
                    hierarchy_cfg.get("separator")
                    if isinstance(hierarchy_cfg, Mapping) else None
                )
                if not sep:
                    continue
                matches = 0
                prefixes: set[str] = set()
                for rec in anchor_lookup.values():
                    hier = rec.hierarchy_path
                    if not hier:
                        continue
                    deepest = hier[-1]
                    if sep not in deepest:
                        continue
                    key_name, key_value = deepest.split(sep, 1)
                    prefixes.add(key_name)
                    if (hier[:-1], key_value) in candidate_by_parent_id:
                        matches += 1
                if matches == 0 or len(prefixes) != 1:
                    continue
                hints.append(
                    f"Hint: '{candidate_name}' has 0 regular matches but "
                    f"{matches}/{len(anchor_lookup)} keyed-style matches if "
                    f"treated as a keyed modality of '{anchor_name}' "
                    f"(key_name={next(iter(prefixes))!r}). "
                    f"Consider passing it under keyed_modalities=."
                )
                break  # one hint per candidate is enough
        return hints

    def _filter_common_ids_by_keyed_joins(self) -> list[str]:
        """Drop samples whose keyed-join would fail; warn or raise per policy."""
        kept: list[str] = []
        per_modality_misses: dict[str, int] = {
            name: 0 for name in self._keyed_modalities
        }
        for sample_id in self._common_ids:
            ok = True
            for name, modality in self._keyed_modalities.items():
                anchor = self._keyed_join_config[name]["modality"]
                anchor_record = self._lookups[anchor][sample_id]
                try:
                    parent, key_value = self._decode_keyed_lookup_key(
                        name, anchor_record,
                    )
                except ValueError as exc:
                    if self._strict_keyed:
                        raise
                    logger.warning(
                        "Keyed modality '%s' decode failed for sample %r: %s",
                        name, sample_id, exc,
                    )
                    per_modality_misses[name] += 1
                    ok = False
                    break
                if (parent, key_value) not in self._keyed_lookups[name]:
                    if self._strict_keyed:
                        raise KeyError(
                            f"Keyed modality '{name}' has no file for "
                            f"sample {sample_id!r}: no entry with "
                            f"id={key_value!r} at hierarchy {parent!r}."
                        )
                    per_modality_misses[name] += 1
                    ok = False
                    break
            if ok:
                kept.append(sample_id)

        for name, missed in per_modality_misses.items():
            if missed:
                logger.warning(
                    "Keyed modality '%s': %d/%d samples dropped due to "
                    "missing joins.",
                    name, missed, len(self._common_ids),
                )

        if not kept:
            counts = {
                name: len(lookup) for name, lookup in self._keyed_lookups.items()
            }
            raise ValueError(
                f"No samples remain after applying keyed-modality joins. "
                f"Per-keyed-modality file counts: {counts}"
            )
        return kept

    def get_modality_metadata(self, modality_name: str) -> dict[str, Any]:
        """Return the metadata dict for a given modality name."""
        index = self._index_outputs.get(modality_name, {})
        meta = _get_index_meta(index)
        return meta or {}

    def describe_id_schema(self) -> dict[str, Any]:
        """Return the id-construction schema used by this dataset.

        Downstream consumers that want to split or match ``sample["full_id"]``
        / ``sample["id"]`` values back to ds-crawler fields need to know the
        separators used to build them. This method exposes that schema so it
        can be embedded in experiment manifests.

        Schema keys:
            - ``hierarchy_separator``: the character between the named
              capture group *name* and its *value* within a single id part
              (ds-crawler ``indexing.hierarchy.separator``, typically ``"-"``).
            - ``id_join_char``: the character between id parts (ds-crawler
              ``indexing.id.join_char`` or legacy ``id_regex_join_char``,
              default ``"+"``).
            - ``full_id_separator``: the character between hierarchy levels
              and the file id in ``sample["full_id"]`` (currently ``"/"``).
            - ``id_regex`` / ``hierarchy_regex``: the raw regexes, when
              recorded in the ds-crawler output.
            - ``modalities``: per-modality overrides populated only when
              modalities disagree on the above.

        Example::

            {
              "hierarchy_separator": "-",
              "id_join_char": "+",
              "full_id_separator": "/",
              "id_regex": "...",
            }
        """
        per_modality: dict[str, dict[str, Any]] = {}
        for name, index in self._index_outputs.items():
            per_modality[name] = _extract_id_schema(index)

        if not per_modality:
            return {}

        base = dict(next(iter(per_modality.values())))
        base["full_id_separator"] = "/"

        overrides: dict[str, dict[str, Any]] = {}
        for mod_name, mod_schema in per_modality.items():
            mod_with_separator = dict(mod_schema)
            mod_with_separator["full_id_separator"] = "/"
            if mod_with_separator != base:
                overrides[mod_name] = mod_with_separator

        if overrides:
            base["modalities"] = overrides
        return base

    def get_modality_index(self, modality_name: str) -> dict[str, Any]:
        """Return the cached ds-crawler index for a modality."""
        if modality_name in self._modalities:
            return dict(self._index_outputs[modality_name])
        if modality_name in self._hierarchical_modalities:
            return dict(self._hierarchical_index_outputs[modality_name])
        if modality_name in self._keyed_modalities:
            return dict(self._keyed_index_outputs[modality_name])
        raise KeyError(f"Unknown modality {modality_name!r}")

    def create_output_writer(
        self,
        modality_name: str,
        root: str | os.PathLike[str],
        *,
        zip: bool = False,
    ) -> DatasetWriter | ZipDatasetWriter:
        """Create a ds-crawler writer preconfigured from a modality's index."""
        index_output = self.get_modality_index(modality_name)
        return create_dataset_writer_from_index(
            index_output=index_output,
            root=root,
            zip=zip,
        )

    def get_writer(self, modality_name: str) -> Callable[..., Any]:
        """Return the resolved writer callable for *modality_name*.

        Raises:
            KeyError: If the modality is unknown.
            ValueError: If no writer is configured or discoverable.
        """
        if (
            modality_name not in self._modalities
            and modality_name not in self._hierarchical_modalities
            and modality_name not in self._keyed_modalities
        ):
            raise KeyError(f"Unknown modality {modality_name!r}")

        writer = self._resolved_writers.get(modality_name)
        if writer is None:
            raise ValueError(
                f"No writer configured for modality {modality_name!r}. "
                "Set Modality.writer explicitly or expose a compatible built-in "
                "writer in the loader module."
            )
        return writer

    def write_sample(
        self,
        sample_index: int,
        outputs: Mapping[str, Any],
        output_root: OutputDestination | Mapping[str, OutputDestination],
        *,
        create_dirs: bool = True,
        overwrite: bool = True,
        attributes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Write model outputs for one sample to disk or a dataset writer.

        Output filenames are derived from ds-crawler relative paths, preserving
        the source dataset hierarchy under *output_root*.

        Args:
            sample_index: Index in this dataset.
            outputs: Mapping of modality name to predicted/loaded value.
            output_root: Either a single destination for all modalities, or a
                per-modality mapping. Destinations can be filesystem roots,
                :class:`ds_crawler.DatasetWriter`, or
                :class:`ds_crawler.ZipDatasetWriter`.
            create_dirs: If true, create parent directories as needed for
                filesystem destinations.
            overwrite: If false, raise when a target file already exists or a
                duplicate writer entry would be created.
            attributes: Optional per-modality ``{modality: {key: value}}``
                mapping recorded on the destination's ds-crawler file
                entry under the ``"attributes"`` field.  Only applies when
                the destination is a :class:`DatasetWriter` /
                :class:`ZipDatasetWriter`; ignored for plain filesystem
                paths.  When omitted, any ``attributes`` carried on the
                source file entry is inherited.

        Returns:
            Mapping ``{modality_name: written_location}``. For filesystem
            destinations this is an absolute path; for zip destinations it is
            ``"<archive>.zip::<relative/path>"``.
        """
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(
                f"sample_index {sample_index} out of range for dataset of length {len(self)}"
            )

        sample_id = self._common_ids[sample_index]
        written_paths: dict[str, str] = {}

        for modality_name, value in outputs.items():
            if modality_name in self._lookups:
                record = self._lookups[modality_name][sample_id]
                modality_meta = _get_index_meta(
                    self._index_outputs[modality_name]
                )
            elif modality_name in self._keyed_lookups:
                config = self._keyed_join_config[modality_name]
                anchor_record = self._lookups[config["modality"]][sample_id]
                parent_prefix, key_value = self._decode_keyed_lookup_key(
                    modality_name, anchor_record,
                )
                record = self._keyed_lookups[modality_name][
                    (parent_prefix, key_value)
                ]
                modality_meta = _get_index_meta(
                    self._keyed_index_outputs[modality_name]
                )
            else:
                raise KeyError(
                    f"Modality {modality_name!r} is not a regular or keyed "
                    "modality and cannot be addressed by sample index."
                )

            writer = self.get_writer(modality_name)
            destination = _resolve_output_destination(
                output_root=output_root, modality_name=modality_name
            )
            basename = Path(record.file_entry["path"]).name
            relative_path = record.file_entry["path"]
            full_id = _build_writer_full_id(
                relative_path=relative_path,
                file_id=record.file_entry["id"],
            )

            if _destination_rel_exists(destination, relative_path) and not overwrite:
                raise FileExistsError(_destination_location(destination, relative_path))

            per_modality_attrs = (
                attributes.get(modality_name) if attributes is not None else None
            )
            written_paths[modality_name] = _write_value_to_destination(
                destination=destination,
                writer=writer,
                value=value,
                meta=modality_meta,
                full_id=full_id,
                basename=basename,
                relative_path=relative_path,
                source_entry=record.file_entry,
                entry_attributes=per_modality_attrs,
                create_dirs=create_dirs,
            )

        return written_paths

    def get_dataset_name(self) -> str | None:
        """Return the dataset name from the first modality's ds-crawler index.

        The ``"name"`` field in each modality's ``output.json`` typically
        identifies the dataset (e.g. ``"vkitti2"``).  This method returns the
        name from the first regular modality.  If other modalities declare a
        different name, a warning is logged.

        Returns:
            The dataset name string, or *None* if no modality has a ``"name"``.
        """
        first_name: str | None = None
        first_modality: str | None = None

        for modality_name, index in self._index_outputs.items():
            try:
                name = get_dataset_contract(index).name
            except Exception:
                raw_name = index.get("name")
                name = str(raw_name) if raw_name is not None else None
            if name is None:
                continue
            if first_name is None:
                first_name = name
                first_modality = modality_name
            elif name != first_name:
                logger.warning(
                    "Modality '%s' has dataset name '%s', but '%s' has '%s'. "
                    "Using '%s'.",
                    modality_name,
                    name,
                    first_modality,
                    first_name,
                    first_name,
                )

        return first_name

    # -- Zip archive helpers -------------------------------------------------

    def _get_zip_handle(self, path: str) -> zipfile.ZipFile:
        """Return a :class:`zipfile.ZipFile` for *path*, lazily opened per process.

        Each worker process in a PyTorch :class:`~torch.utils.data.DataLoader`
        gets its own file handle so that forked processes do not share seek
        positions.
        """
        path = path.rstrip("/")
        pid = os.getpid()
        key = (path, pid)
        if key not in self._zip_handles:
            self._zip_handles[key] = zipfile.ZipFile(path, "r")
        return self._zip_handles[key]

    def _open_from_zip(self, name: str, modality_path: str, relative_path: str) -> io.BytesIO:
        """Read a file from a zip-backed modality into an in-memory buffer."""
        entry_name = self._zip_prefixes[name] + relative_path
        zf = self._get_zip_handle(modality_path)
        data = zf.read(entry_name)
        buf = io.BytesIO(data)
        buf.name = relative_path
        return buf

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._common_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self._common_ids[index]
        sample: dict[str, Any] = {}

        meta: dict[str, dict[str, Any]] = {}
        attributes: dict[str, dict[str, Any]] = {}
        file_id: str = ""
        hierarchy_path: tuple[str, ...] = ()

        for name, modality in self._modalities.items():
            record = self._lookups[name][sample_id]
            modality_meta = _get_index_meta(self._index_outputs[name])

            if name in self._zip_modalities:
                file_or_path = self._open_from_zip(
                    name, modality.path, record.file_entry["path"],
                )
            else:
                file_or_path = f"{modality.path}/{record.file_entry['path']}"

            entry_attributes = _get_file_attributes(record.file_entry)
            sample[name] = _load_with_optional_attributes(
                self._resolved_loaders[name],
                file_or_path,
                modality_meta,
                entry_attributes,
                accepts_attributes=self._loaders_accept_attributes[name],
            )
            meta[name] = record.file_entry
            attributes[name] = entry_attributes or {}

            # Hierarchy path from the first modality that has one.
            if not hierarchy_path:
                hierarchy_path = record.hierarchy_path
                file_id = record.file_entry["id"]

        # -- Load hierarchical modalities ------------------------------------
        for name, modality in self._hierarchical_modalities.items():
            files_by_level = self._hierarchical_lookups[name]
            matched = match_hierarchical_files(hierarchy_path, files_by_level)
            modality_meta = _get_index_meta(self._hierarchical_index_outputs[name])
            cache = self._caches[name]
            loaded: dict[str, Any] = {}
            attrs_for_modality: dict[str, dict[str, Any]] = {}
            entries_by_id: dict[str, dict[str, Any]] = {}
            for entry in matched:
                entry_attributes = _get_file_attributes(entry)
                accepts_attributes = self._loaders_accept_attributes[name]
                cache_key = f"{modality.path}/{entry['path']}"
                if accepts_attributes:
                    cache_key = (
                        f"{cache_key}::attributes="
                        f"{_attributes_cache_key(entry_attributes)}"
                    )
                if cache is not None and cache_key in cache:
                    value = cache[cache_key]
                else:
                    if name in self._zip_modalities:
                        file_or_path = self._open_from_zip(
                            name, modality.path, entry["path"],
                        )
                    else:
                        file_or_path = f"{modality.path}/{entry['path']}"
                    value = _load_with_optional_attributes(
                        self._resolved_loaders[name],
                        file_or_path,
                        modality_meta,
                        entry_attributes,
                        accepts_attributes=accepts_attributes,
                    )
                    if cache is not None:
                        cache[cache_key] = value
                loaded[entry["id"]] = value
                entries_by_id[entry["id"]] = entry
                if entry_attributes:
                    attrs_for_modality[entry["id"]] = entry_attributes
            if modality.collapse_single:
                if len(loaded) > 1:
                    raise ValueError(
                        f"Hierarchical modality {name!r} was configured with "
                        f"collapse_single=True, but sample {sample_id!r} "
                        f"matched {len(loaded)} files."
                    )
                if loaded:
                    only_id, only_value = next(iter(loaded.items()))
                    sample[name] = only_value
                    meta[name] = entries_by_id[only_id]
                    attributes[name] = attrs_for_modality.get(only_id, {})
                else:
                    sample[name] = None
                    attributes[name] = {}
            else:
                sample[name] = loaded
                attributes[name] = attrs_for_modality

        # -- Load keyed modalities -------------------------------------------
        for name, modality in self._keyed_modalities.items():
            config = self._keyed_join_config[name]
            anchor_record = self._lookups[config["modality"]][sample_id]
            parent_prefix, key_value = self._decode_keyed_lookup_key(
                name, anchor_record,
            )
            record = self._keyed_lookups[name][(parent_prefix, key_value)]
            modality_meta = _get_index_meta(self._keyed_index_outputs[name])
            entry_attributes = _get_file_attributes(record.file_entry)
            accepts_attributes = self._loaders_accept_attributes[name]
            cache = self._caches[name]
            cache_key = f"{modality.path}/{record.file_entry['path']}"
            if accepts_attributes:
                cache_key = (
                    f"{cache_key}::attributes="
                    f"{_attributes_cache_key(entry_attributes)}"
                )
            if cache is not None and cache_key in cache:
                value = cache[cache_key]
            else:
                if name in self._zip_modalities:
                    file_or_path = self._open_from_zip(
                        name, modality.path, record.file_entry["path"],
                    )
                else:
                    file_or_path = f"{modality.path}/{record.file_entry['path']}"
                value = _load_with_optional_attributes(
                    self._resolved_loaders[name],
                    file_or_path,
                    modality_meta,
                    entry_attributes,
                    accepts_attributes=accepts_attributes,
                )
                if cache is not None:
                    cache[cache_key] = value
            sample[name] = value
            meta[name] = record.file_entry
            attributes[name] = entry_attributes or {}

        sample["id"] = file_id
        sample["full_id"] = "/" + "/".join(hierarchy_path + (file_id,))
        sample["meta"] = meta
        sample["attributes"] = attributes

        for transform in self._transforms:
            sample = transform(sample)

        return sample
