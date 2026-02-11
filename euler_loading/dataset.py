from __future__ import annotations

import importlib
import io
import logging
import os
from pathlib import Path
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

try:
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:
    class _BaseDataset:  # type: ignore[no-redef]
        """Fallback when PyTorch is not installed."""

from ds_crawler import index_dataset_from_path, load_dataset_config
from ds_crawler.zip_utils import get_zip_root_prefix, is_zip_path

from .indexing import (
    FileRecord,
    collect_files,
    collect_hierarchical_files,
    match_hierarchical_files,
)

logger = logging.getLogger(__name__)


_DS_CRAWLER_STRUCTURAL_KEYS = frozenset({
    "name",
    "type",
    "id_regex",
    "id_regex_join_char",
    "euler_train",
    "dataset",
    "hierarchy_regex",
    "named_capture_group_value_separator",
    "sampled",
})

_LOADER_MODULES: dict[str, str] = {
    "vkitti2": "euler_loading.loaders.gpu.vkitti2",
    "real_drive_sim": "euler_loading.loaders.gpu.real_drive_sim",
    "generic_dense_depth": "euler_loading.loaders.gpu.generic_dense_depth",
}


def _resolve_loader_module(name: str) -> ModuleType:
    """Import and return the GPU loader module for *name*.

    Raises:
        ValueError: If *name* does not match any known loader.
    """
    module_path = _LOADER_MODULES.get(name)
    if module_path is None:
        available = ", ".join(sorted(_LOADER_MODULES))
        raise ValueError(
            f"Unknown loader {name!r}. Available loaders: {available}"
        )
    return importlib.import_module(module_path)


def _resolve_loader(
    *,
    modality_name: str,
    modality: "Modality",
    index: dict[str, Any],
) -> Callable[..., Any]:
    """Return the effective loader for a modality.

    If ``modality.loader`` is set, it is returned as-is.  Otherwise the loader
    is looked up from ``index["euler_loading"]["loader"]`` (the module name,
    e.g. ``"vkitti2"``) and ``index["euler_loading"]["function"]`` (the
    function name, e.g. ``"rgb"``).
    """
    if modality.loader is not None:
        return modality.loader

    euler_loading_meta = index.get("euler_loading")
    if not isinstance(euler_loading_meta, Mapping):
        raise ValueError(
            f"Modality {modality_name!r}: no explicit loader provided and the "
            f"ds-crawler index at {modality.path!r} does not contain an "
            f"'euler_loading' property."
        )

    for key in ("loader", "function"):
        if key not in euler_loading_meta:
            raise ValueError(
                f"Modality {modality_name!r}: no explicit loader provided and "
                f"'euler_loading.{key}' is missing from the ds-crawler index "
                f"at {modality.path!r}."
            )

    module_name: str = euler_loading_meta["loader"]
    func_name: str = euler_loading_meta["function"]

    module = _resolve_loader_module(module_name)

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        available = [
            attr for attr in dir(module)
            if not attr.startswith("_") and callable(getattr(module, attr))
        ]
        raise ValueError(
            f"Modality {modality_name!r}: loader module {module_name!r} has no "
            f"callable {func_name!r}. Available: {', '.join(available)}"
        )

    return func


#TODO: might want to add slots=True in a 3.10+ only codebase
@dataclass(frozen=True)
class Modality:
    """Specification of a single data modality.

    Attributes:
        path: Absolute path to the dataset root directory for this modality.
              Must contain a ``ds-crawler.config`` file (or a cached
              ``output.json`` from a prior indexing run).
        origin_path: Optional original path string before any copying or symlinking for i.e. slurm staging.  
                This is not used by euler-loading itself but can be useful for experiment 
                logging to retain references to the original dataset location.
        loader: Optional callable that takes an absolute file path and returns
                the loaded data (e.g. a numpy array, a PIL Image, a tensor).
                When *None* (the default), the loader is resolved automatically
                from ``euler_loading.loader`` and ``euler_loading.function`` in
                the ds-crawler index, using the GPU variant of the matching
                predefined loader.
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
        metadata: Optional arbitrary metadata. Keys under
                  ``metadata["euler_loading"]`` are treated as
                  euler-loading-specific defaults.
    """

    path: str
    origin_path: str | None = None
    loader: Callable[..., Any] | None = None
    used_as: str | None = None
    slot: str | None = None
    modality_type: str | None = None
    hierarchy_scope: str | None = None
    applies_to: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def modality_paths(self) -> dict[str, dict[str, str]]:
        """Return a list of the names of all modalities in this dataset."""
        #return {name: mod.path for name, mod in self._modalities.items()}
        #get mod.path and mod.origin_path if it exists, otherwise fallback to mod.path
        res = {}
        for name, mod in self._modalities.items():
            path = mod.path
            origin = mod.origin_path
            res[name] = {"path": path, "origin_path": origin}
        return res

    def hierarchical_modality_paths(self) -> dict[str, dict[str, str]]:
        """Return a dict of hierarchical modality names to their root paths."""
        res = {}
        for name, mod in self._hierarchical_modalities.items():
            path = mod.path
            origin = mod.origin_path
            res[name] = {"path": path, "origin_path": origin}
        return res

    def describe_for_runlog(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return structured dataset metadata for runlog ``meta.json``.

        The returned dict is shaped as::

            {
              "modalities": {
                "<name>": {
                  "path": "...",
                  "origin_path": "...",
                  "used_as": "...",
                  "slot": "...",
                  "modality_type": "...",
                },
              },
              "hierarchical_modalities": {
                "<name>": {
                  "path": "...",
                  "origin_path": "...",
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

        return {
            "modalities": modalities,
            "hierarchical_modalities": hierarchical_modalities,
        }

    def __init__(
        self,
        modalities: dict[str, Modality],
        hierarchical_modalities: dict[str, Modality] | None = None,
        transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()

        if not modalities:
            raise ValueError("At least one modality must be provided.")

        self._modalities = modalities
        self._hierarchical_modalities = hierarchical_modalities or {}
        self._transforms = transforms or []
        self._index_outputs: dict[str, dict[str, Any]] = {}
        self._hierarchical_index_outputs: dict[str, dict[str, Any]] = {}

        # -- Index each modality and build ID → FileRecord lookups -----------
        self._lookups: dict[str, dict[str, FileRecord]] = {}
        self._resolved_loaders: dict[str, Callable[..., Any]] = {}

        for name, modality in modalities.items():
            index = index_dataset_from_path(modality.path)
            self._index_outputs[name] = index
            self._resolved_loaders[name] = _resolve_loader(
                modality_name=name, modality=modality, index=index,
            )
            records = collect_files(index["dataset"])
            lookup = {
                "/".join(rec.hierarchy_path + (rec.file_entry["id"],)): rec
                for rec in records
            }
            self._lookups[name] = lookup
            logger.info(
                "Modality '%s': indexed %d files from %s",
                name,
                len(lookup),
                modality.path,
            )

        # -- Index hierarchical modalities -----------------------------------
        self._hierarchical_lookups: dict[
            str, dict[tuple[str, ...], list[dict[str, Any]]]
        ] = {}

        for name, modality in self._hierarchical_modalities.items():
            index = index_dataset_from_path(modality.path)
            self._hierarchical_index_outputs[name] = index
            self._resolved_loaders[name] = _resolve_loader(
                modality_name=name, modality=modality, index=index,
            )
            files_by_level = collect_hierarchical_files(index["dataset"])
            self._hierarchical_lookups[name] = files_by_level
            total_files = sum(len(f) for f in files_by_level.values())
            logger.info(
                "Hierarchical modality '%s': indexed %d files across %d "
                "hierarchy levels from %s",
                name,
                total_files,
                len(files_by_level),
                modality.path,
            )

        # -- Compute common IDs (intersection across regular modalities) -----
        all_id_sets = [set(lookup.keys()) for lookup in self._lookups.values()]
        self._common_ids: list[str] = sorted(set.intersection(*all_id_sets))

        if not self._common_ids:
            modality_counts = {
                name: len(lookup) for name, lookup in self._lookups.items()
            }
            raise ValueError(
                f"No common IDs found across modalities. "
                f"Per-modality file counts: {modality_counts}"
            )

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

        # -- Zip archive support ------------------------------------------------
        self._zip_modalities: set[str] = set()
        self._zip_prefixes: dict[str, str] = {}
        self._zip_handles: dict[tuple[str, int], zipfile.ZipFile] = {}

        for name, modality in list(modalities.items()) + list(self._hierarchical_modalities.items()):
            if is_zip_path(modality.path):
                self._zip_modalities.add(name)
                self._zip_prefixes[name] = get_zip_root_prefix(
                    Path(modality.path)
                )

        # -- Caches ----------------------------------------------------------
        self._hierarchical_cache: dict[str, Any] = {}

    def get_modality_metadata(self, modality_name: str) -> dict[str, Any]:
        """Return the metadata dict for a given modality name."""
        return self._index_outputs.get(modality_name, {}).get("meta", {})

    # -- Zip archive helpers -------------------------------------------------

    def _get_zip_handle(self, path: str) -> zipfile.ZipFile:
        """Return a :class:`zipfile.ZipFile` for *path*, lazily opened per process.

        Each worker process in a PyTorch :class:`~torch.utils.data.DataLoader`
        gets its own file handle so that forked processes do not share seek
        positions.
        """
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
        file_id: str = ""
        hierarchy_path: tuple[str, ...] = ()

        for name, modality in self._modalities.items():
            record = self._lookups[name][sample_id]
            modality_meta = self._index_outputs[name].get("meta")

            if name in self._zip_modalities:
                file_or_path = self._open_from_zip(
                    name, modality.path, record.file_entry["path"],
                )
            else:
                file_or_path = f"{modality.path}/{record.file_entry['path']}"

            sample[name] = self._resolved_loaders[name](file_or_path, modality_meta)
            meta[name] = record.file_entry

            # Hierarchy path from the first modality that has one.
            if not hierarchy_path:
                hierarchy_path = record.hierarchy_path
                file_id = record.file_entry["id"]

        # -- Load hierarchical modalities ------------------------------------
        for name, modality in self._hierarchical_modalities.items():
            files_by_level = self._hierarchical_lookups[name]
            matched = match_hierarchical_files(hierarchy_path, files_by_level)
            modality_meta = self._hierarchical_index_outputs[name].get("meta")
            loaded: dict[str, Any] = {}
            for entry in matched:
                cache_key = f"{modality.path}/{entry['path']}"
                if cache_key not in self._hierarchical_cache:
                    if name in self._zip_modalities:
                        file_or_path = self._open_from_zip(
                            name, modality.path, entry["path"],
                        )
                    else:
                        file_or_path = cache_key
                    self._hierarchical_cache[cache_key] = self._resolved_loaders[name](
                        file_or_path, modality_meta
                    )
                loaded[entry["id"]] = self._hierarchical_cache[cache_key]
            sample[name] = loaded

        sample["id"] = file_id
        sample["full_id"] = "/" + "/".join(hierarchy_path + (file_id,))
        sample["meta"] = meta

        for transform in self._transforms:
            sample = transform(sample)

        return sample


def _build_runlog_entry(
    name: str,
    modality: Modality,
    descriptor: Mapping[str, Any],
    *,
    is_hierarchical: bool,
    regular_modality_names: list[str],
    hierarchy_levels: list[tuple[str, ...]] | None,
) -> dict[str, Any]:
    euler_loading_properties = _build_euler_loading_layers(
        modality.metadata,
        descriptor.get("properties"),
    )
    used_as = _first_non_empty(
        modality.used_as,
        _as_non_empty_str(
            _resolve_euler_loading_property(euler_loading_properties, "used_as")
        ),
        _infer_used_as(name=name, is_hierarchical=is_hierarchical),
    )
    modality_type = _first_non_empty(
        modality.modality_type,
        _as_non_empty_str(
            _resolve_euler_loading_property(euler_loading_properties, "modality_type")
        ),
        _as_non_empty_str(descriptor.get("modality_type")),
        _infer_modality_type(name=name, path=modality.path),
    )
    slot = _first_non_empty(
        modality.slot,
        _as_non_empty_str(_resolve_euler_loading_property(euler_loading_properties, "slot")),
        _infer_slot(
            name=name,
            used_as=used_as,
            modality_type=modality_type,
            euler_loading_properties=euler_loading_properties,
        ),
    )

    entry: dict[str, Any] = {"path": modality.path}
    if used_as is not None:
        entry["used_as"] = used_as
    if slot is not None:
        entry["slot"] = slot
    if modality_type is not None:
        entry["modality_type"] = modality_type

    if is_hierarchical:
        hierarchy_scope = _first_non_empty(
            modality.hierarchy_scope,
            _as_non_empty_str(
                _resolve_euler_loading_property(
                    euler_loading_properties, "hierarchy_scope"
                )
            ),
            _infer_hierarchy_scope_from_regex(descriptor.get("hierarchy_regex")),
            _infer_hierarchy_scope_from_levels(hierarchy_levels),
        )
        if hierarchy_scope is not None:
            entry["hierarchy_scope"] = hierarchy_scope

        applies_to = _first_non_empty_list(
            _as_string_list(modality.applies_to),
            _as_string_list(
                _resolve_euler_loading_property(euler_loading_properties, "applies_to")
            ),
            list(regular_modality_names),
        )
        entry["applies_to"] = applies_to

    return entry


def _build_euler_loading_layers(*candidates: Any) -> list[Mapping[str, Any]]:
    layers: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        namespaced = candidate.get("euler_loading")
        if isinstance(namespaced, Mapping):
            layers.append(namespaced)
    return layers


def _resolve_euler_loading_property(
    euler_loading_properties: list[Mapping[str, Any]],
    key: str,
) -> Any:
    for layer in euler_loading_properties:
        value = layer.get(key)
        if value is not None:
            return value
    return None


def _get_ds_crawler_descriptor(
    *,
    path: str,
    index_output: Mapping[str, Any] | None,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if path not in cache:
        cache[path] = _read_ds_crawler_descriptor(path=path, index_output=index_output)
    return cache[path]


def _read_ds_crawler_descriptor(
    *,
    path: str,
    index_output: Mapping[str, Any] | None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    descriptor: dict[str, Any] = {"properties": properties}

    if isinstance(index_output, Mapping):
        properties.update(_extract_ds_crawler_properties(index_output))
        index_type = _as_non_empty_str(index_output.get("type"))
        if index_type is not None:
            descriptor["modality_type"] = index_type
        hierarchy_regex = _as_non_empty_str(index_output.get("hierarchy_regex"))
        if hierarchy_regex is not None:
            descriptor["hierarchy_regex"] = hierarchy_regex

    try:
        cfg = load_dataset_config({"path": path})
    except Exception:
        return descriptor

    cfg_properties = getattr(cfg, "properties", None)
    if isinstance(cfg_properties, Mapping):
        properties.update(dict(cfg_properties))

    cfg_type = _as_non_empty_str(getattr(cfg, "type", None))
    if cfg_type is not None:
        descriptor["modality_type"] = cfg_type

    cfg_hierarchy_regex = _as_non_empty_str(getattr(cfg, "hierarchy_regex", None))
    if cfg_hierarchy_regex is not None:
        descriptor["hierarchy_regex"] = cfg_hierarchy_regex

    return descriptor


def _extract_ds_crawler_properties(index_output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in index_output.items()
        if key not in _DS_CRAWLER_STRUCTURAL_KEYS
    }


def _infer_used_as(*, name: str, is_hierarchical: bool) -> str | None:
    lowered = name.lower()
    if any(
        token in lowered
        for token in ("condition", "cond", "camera", "intrinsics", "extrinsics", "pose")
    ):
        return "condition"
    if any(token in lowered for token in ("target", "gt", "label", "clear", "clean")):
        return "target"
    if any(token in lowered for token in ("input", "source", "src", "hazy", "noisy", "raw")):
        return "input"
    if is_hierarchical:
        return "condition"
    return None


def _infer_modality_type(*, name: str, path: str) -> str | None:
    lowered = f"{name} {path}".lower()
    if any(token in lowered for token in ("rgb", "image", "img", "color", "colour")):
        return "rgb"
    if any(token in lowered for token in ("depth", "disparity")):
        return "depth"
    if any(token in lowered for token in ("segmentation", "segment", "mask", "semantic")):
        return "segmentation"
    return None


def _infer_slot(
    *,
    name: str,
    used_as: str | None,
    modality_type: str | None,
    euler_loading_properties: list[Mapping[str, Any]],
) -> str | None:
    if used_as is None:
        return None
    task = _as_non_empty_str(
        _resolve_euler_loading_property(euler_loading_properties, "task")
    )
    leaf = modality_type or name
    if task is not None:
        return f"{task}.{used_as}.{leaf}"
    return f"{used_as}.{leaf}"


def _infer_hierarchy_scope_from_regex(value: Any) -> str | None:
    regex = _as_non_empty_str(value)
    if regex is None:
        return None
    try:
        pattern = re.compile(regex)
    except re.error:
        return None

    if not pattern.groupindex:
        return None

    ordered_names = [
        name for name, _ in sorted(pattern.groupindex.items(), key=lambda item: item[1])
    ]
    if not ordered_names:
        return None
    return "_".join(ordered_names)


def _infer_hierarchy_scope_from_levels(
    levels: list[tuple[str, ...]] | None,
) -> str | None:
    if not levels:
        return None

    non_root_levels = [level for level in levels if level]
    if not non_root_levels:
        return "root"

    max_depth = max(len(level) for level in non_root_levels)
    deepest_levels = [level for level in non_root_levels if len(level) == max_depth]

    tokens: list[str] = []
    for idx in range(max_depth):
        candidates = {
            token
            for token in (_extract_hierarchy_token(level[idx]) for level in deepest_levels)
            if token is not None
        }
        if len(candidates) != 1:
            return f"level_{max_depth}"
        tokens.append(next(iter(candidates)))

    if not tokens:
        return f"level_{max_depth}"
    return "_".join(tokens)


def _extract_hierarchy_token(value: str) -> str | None:
    for separator in (":", "=", "__", "_", "-"):
        if separator not in value:
            continue
        prefix = value.split(separator, 1)[0].strip().lower()
        if prefix and any(ch.isalpha() for ch in prefix):
            return prefix
    return None


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = [_as_non_empty_str(item) for item in value]
        return [item for item in parsed if item is not None]

    single = _as_non_empty_str(value)
    if single is None:
        return []
    return [single]


def _first_non_empty(*candidates: str | None) -> str | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _first_non_empty_list(*candidates: list[str] | None) -> list[str]:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return []
