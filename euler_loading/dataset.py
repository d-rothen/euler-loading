from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from torch.utils.data import Dataset

from ds_crawler import index_dataset_from_path

from .indexing import (
    FileRecord,
    collect_files,
    collect_hierarchical_files,
    match_hierarchical_files,
)

logger = logging.getLogger(__name__)

#TODO: might want to add slots=True in a 3.10+ only codebase
@dataclass(frozen=True)
class Modality:
    """Specification of a single data modality.

    Attributes:
        path: Absolute path to the dataset root directory for this modality.
              Must contain a ``ds-crawler.config`` file (or a cached
              ``output.json`` from a prior indexing run).
        loader: Callable that takes an absolute file path and returns the loaded
                data (e.g. a numpy array, a PIL Image, a tensor).
    """

    path: str
    loader: Callable[[str], Any]


class MultiModalDataset(Dataset):
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

        # -- Index each modality and build ID → FileRecord lookups -----------
        self._lookups: dict[str, dict[str, FileRecord]] = {}

        for name, modality in modalities.items():
            index = index_dataset_from_path(modality.path)
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

        # -- Caches ----------------------------------------------------------
        self._hierarchical_cache: dict[str, Any] = {}

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
            file_path = f"{modality.path}/{record.file_entry['path']}"
            sample[name] = modality.loader(file_path)
            meta[name] = record.file_entry

            # Hierarchy path from the first modality that has one.
            if not hierarchy_path:
                hierarchy_path = record.hierarchy_path
                file_id = record.file_entry["id"]

        # -- Load hierarchical modalities ------------------------------------
        for name, modality in self._hierarchical_modalities.items():
            files_by_level = self._hierarchical_lookups[name]
            matched = match_hierarchical_files(hierarchy_path, files_by_level)
            loaded: dict[str, Any] = {}
            for entry in matched:
                file_path = f"{modality.path}/{entry['path']}"
                if file_path not in self._hierarchical_cache:
                    self._hierarchical_cache[file_path] = modality.loader(
                        file_path
                    )
                loaded[entry["id"]] = self._hierarchical_cache[file_path]
            sample[name] = loaded

        sample["id"] = file_id
        sample["full_id"] = "/" + "/".join(hierarchy_path + (file_id,))
        sample["meta"] = meta

        for transform in self._transforms:
            sample = transform(sample)

        return sample

