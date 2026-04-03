from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
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

from ._ds_crawler_utils import load_index_output, parse_path_with_split
from ._metadata import _build_runlog_entry, _get_ds_crawler_descriptor
from ._resolution import (
    _resolve_loader,
    _resolve_writer,
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

        # -- Optional transform binding -------------------------------------
        for transform in self._transforms:
            bind = getattr(transform, "bind_to_dataset", None)
            if callable(bind):
                bind(self)
        if self._transforms:
            logger.info("Configured %d sample transform(s):", len(self._transforms))
            for idx, transform in enumerate(self._transforms, start=1):
                logger.info(
                    "Transform %d/%d: %s",
                    idx,
                    len(self._transforms),
                    _describe_transform(transform),
                )
        else:
            logger.info("Configured 0 sample transforms.")

    def get_modality_metadata(self, modality_name: str) -> dict[str, Any]:
        """Return the metadata dict for a given modality name."""
        index = self._index_outputs.get(modality_name, {})
        meta = _get_index_meta(index)
        return meta or {}

    def get_modality_index(self, modality_name: str) -> dict[str, Any]:
        """Return the cached ds-crawler index for a modality."""
        if modality_name in self._modalities:
            return dict(self._index_outputs[modality_name])
        if modality_name in self._hierarchical_modalities:
            return dict(self._hierarchical_index_outputs[modality_name])
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
            if modality_name not in self._lookups:
                raise KeyError(
                    f"Modality {modality_name!r} is not a regular modality and cannot "
                    "be addressed by sample index."
                )

            writer = self.get_writer(modality_name)
            destination = _resolve_output_destination(
                output_root=output_root, modality_name=modality_name
            )
            record = self._lookups[modality_name][sample_id]
            modality_meta = _get_index_meta(self._index_outputs[modality_name])
            basename = Path(record.file_entry["path"]).name
            relative_path = record.file_entry["path"]
            full_id = _build_writer_full_id(
                relative_path=relative_path,
                file_id=record.file_entry["id"],
            )

            if _destination_rel_exists(destination, relative_path) and not overwrite:
                raise FileExistsError(_destination_location(destination, relative_path))

            written_paths[modality_name] = _write_value_to_destination(
                destination=destination,
                writer=writer,
                value=value,
                meta=modality_meta,
                full_id=full_id,
                basename=basename,
                relative_path=relative_path,
                source_meta=record.file_entry,
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
            modality_meta = _get_index_meta(self._hierarchical_index_outputs[name])
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
