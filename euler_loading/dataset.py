from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from torch.utils.data import Dataset

from ds_crawler import index_dataset_from_path

from .indexing import FileRecord, collect_files_with_calibration

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
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
    modalities, and returns dicts containing every loaded modality together with
    calibration data.

    Args:
        modalities: Mapping of modality name to :class:`Modality` spec.
                    Example: ``{"rgb": Modality("/data/rgb", load_rgb)}``.
        read_intrinsics: Optional callable that takes an intrinsics file path
                         and returns parsed calibration data.
        read_extrinsics: Optional callable that takes an extrinsics file path
                         and returns parsed calibration data.
        transforms: Optional list of callables.  Each receives the full sample
                    dict and must return a (possibly modified) sample dict.  The
                    sample dict acts as a context object giving transforms
                    access to all modalities, calibration, and metadata.

    Raises:
        ValueError: If *modalities* is empty or no common file IDs exist across
                    the provided modalities.

    Example::

        dataset = MultiModalDataset(
            modalities={
                "rgb":   Modality("/data/vkitti2/rgb",   load_rgb),
                "depth": Modality("/data/vkitti2/depth", load_depth),
            },
            read_intrinsics=parse_intrinsics,
            transforms=[mask_sky_in_depth],
        )

        sample = dataset[0]
        # sample["rgb"], sample["depth"], sample["intrinsics"], ...
    """

    def __init__(
        self,
        modalities: dict[str, Modality],
        read_intrinsics: Callable[[str], Any] | None = None,
        read_extrinsics: Callable[[str], Any] | None = None,
        transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()

        if not modalities:
            raise ValueError("At least one modality must be provided.")

        self._modalities = modalities
        self._read_intrinsics = read_intrinsics
        self._read_extrinsics = read_extrinsics
        self._transforms = transforms or []

        # -- Index each modality and build ID → FileRecord lookups -----------
        self._lookups: dict[str, dict[str, FileRecord]] = {}

        for name, modality in modalities.items():
            index = index_dataset_from_path(modality.path)
            records = collect_files_with_calibration(index["dataset"], modality.path)
            lookup = {rec.file_entry["id"]: rec for rec in records}
            self._lookups[name] = lookup
            logger.info(
                "Modality '%s': indexed %d files from %s",
                name,
                len(lookup),
                modality.path,
            )

        # -- Compute common IDs (intersection across all modalities) ---------
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

        # -- Calibration cache -----------------------------------------------
        self._calibration_cache: dict[str, Any] = {}

    # -- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._common_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self._common_ids[index]
        sample: dict[str, Any] = {}

        meta: dict[str, dict[str, Any]] = {}
        intrinsics_path: str | None = None
        extrinsics_path: str | None = None

        for name, modality in self._modalities.items():
            record = self._lookups[name][sample_id]
            file_path = f"{modality.path}/{record.file_entry['path']}"
            sample[name] = modality.loader(file_path)
            meta[name] = record.file_entry

            # Use calibration from whichever modality provides it (first wins).
            if intrinsics_path is None and record.intrinsics_path is not None:
                intrinsics_path = record.intrinsics_path
            if extrinsics_path is None and record.extrinsics_path is not None:
                extrinsics_path = record.extrinsics_path

        sample["intrinsics"] = self._resolve_calibration(
            intrinsics_path, self._read_intrinsics
        )
        sample["extrinsics"] = self._resolve_calibration(
            extrinsics_path, self._read_extrinsics
        )
        sample["id"] = sample_id
        sample["meta"] = meta

        for transform in self._transforms:
            sample = transform(sample)

        return sample

    # -- Internals -----------------------------------------------------------

    def _resolve_calibration(
        self,
        path: str | None,
        reader: Callable[[str], Any] | None,
    ) -> Any:
        """Read and cache calibration data.

        Returns ``None`` if *path* is ``None`` or *reader* is ``None``.
        """
        if path is None or reader is None:
            return None

        if path not in self._calibration_cache:
            self._calibration_cache[path] = reader(path)

        return self._calibration_cache[path]
