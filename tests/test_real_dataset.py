"""Tests for real on-disk datasets — run on demand with ``pytest -m real``.

Each entry in :data:`REAL_DATASETS` defines a complete configuration for a
:class:`~euler_loading.MultiModalDataset`.  Add or modify entries to cover
additional databases — the test suite parametrises over every *configured*
entry automatically.

Entries whose ``modalities`` dict is empty are silently skipped so that
unconfigured stubs never cause failures.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders import vkitti2


# ---------------------------------------------------------------------------
# Dataset configurations  (stubs — fill in per database)
# ---------------------------------------------------------------------------
#
# Each value is a dict with the keys accepted by ``MultiModalDataset``:
#
#   modalities                : dict[str, Modality]                       — REQUIRED
#   hierarchical_modalities   : dict[str, Modality] | None               — optional
#   transforms                : list[Callable[[dict], dict]] | None       — optional
#
# Leave ``modalities`` empty (``{}``) for databases that are not yet wired up;
# those entries are silently skipped by the test suite.

REAL_DATASETS: dict[str, dict[str, Any]] = {
    "VKITTI2": {
        "modalities": {
            "rgb":   Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_rgb",   loader=vkitti2.rgb),
            "depth": Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_depth", loader=vkitti2.depth),
            "classSegmentation": Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_classSegmentation", loader=vkitti2.class_segmentation),
        },
        "hierarchical_modalities": None,
        "transforms": None,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configured_datasets():
    """Yield a ``pytest.param`` for every dataset whose modalities are filled in."""
    for name, cfg in REAL_DATASETS.items():
        if cfg["modalities"]:
            yield pytest.param(cfg, id=name)


def _paths_exist(cfg: dict[str, Any]) -> bool:
    """Return True when every modality root directory exists on disk."""
    return all(os.path.isdir(m.path) for m in cfg["modalities"].values())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

real = pytest.mark.real

_PARAMS = list(_configured_datasets())


@pytest.fixture(params=_PARAMS)
def real_config(request):
    """Yield each configured dataset config, skipping when paths are absent."""
    cfg = request.param
    if not _paths_exist(cfg):
        pytest.skip("One or more modality paths not found on disk")
    return cfg


@pytest.fixture()
def real_dataset(real_config):
    """Construct a :class:`MultiModalDataset` from a real on-disk config."""
    return MultiModalDataset(
        modalities=real_config["modalities"],
        hierarchical_modalities=real_config.get("hierarchical_modalities"),
        transforms=real_config["transforms"],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@real
class TestRealConstruction:
    """The dataset can be constructed from the on-disk data."""

    def test_builds_without_error(self, real_dataset):
        assert real_dataset is not None

    def test_non_empty(self, real_dataset):
        assert len(real_dataset) > 0


# ---------------------------------------------------------------------------
# Sample structure
# ---------------------------------------------------------------------------


@real
class TestRealSampleStructure:
    """Every sample exposes the expected keys with non-None modality data."""

    def test_sample_has_modality_keys(self, real_dataset, real_config):
        sample = real_dataset[0]
        for modality_name in real_config["modalities"]:
            assert modality_name in sample, f"Missing modality key: {modality_name}"

    def test_sample_has_metadata_keys(self, real_dataset):
        sample = real_dataset[0]
        for key in ("id", "meta"):
            assert key in sample, f"Missing metadata key: {key}"

    def test_modality_data_is_not_none(self, real_dataset, real_config):
        sample = real_dataset[0]
        for modality_name in real_config["modalities"]:
            assert sample[modality_name] is not None, (
                f"Modality '{modality_name}' loaded as None"
            )

    def test_meta_contains_file_entries(self, real_dataset, real_config):
        sample = real_dataset[0]
        for modality_name in real_config["modalities"]:
            entry = sample["meta"][modality_name]
            assert "id" in entry
            assert "path" in entry


# ---------------------------------------------------------------------------
# Ordering and determinism
# ---------------------------------------------------------------------------


@real
class TestRealOrdering:
    """IDs are sorted and repeated access is deterministic."""

    def test_ids_sorted(self, real_dataset):
        ids = [real_dataset[i]["id"] for i in range(len(real_dataset))]
        assert ids == sorted(ids)

    def test_deterministic_access(self, real_dataset):
        sample_a = real_dataset[0]
        sample_b = real_dataset[0]
        assert sample_a["id"] == sample_b["id"]


# ---------------------------------------------------------------------------
# Boundary access
# ---------------------------------------------------------------------------


@real
class TestRealBoundaryAccess:
    """Samples at the boundaries of the index can be loaded."""

    def test_first_sample(self, real_dataset):
        sample = real_dataset[0]
        assert sample["id"] is not None

    def test_last_sample(self, real_dataset):
        sample = real_dataset[len(real_dataset) - 1]
        assert sample["id"] is not None

    def test_middle_sample(self, real_dataset):
        mid = len(real_dataset) // 2
        sample = real_dataset[mid]
        assert sample["id"] is not None

    def test_index_out_of_range(self, real_dataset):
        with pytest.raises(IndexError):
            real_dataset[len(real_dataset)]


# ---------------------------------------------------------------------------
# Full iteration
# ---------------------------------------------------------------------------


@real
class TestRealFullIteration:
    """Every sample in the dataset can be loaded without error."""

    def log_first_example(self, real_dataset):
        sample = real_dataset[0]
        print("First sample keys:", list(sample.keys()))
        #print object
        for k, v in sample.items():
            print(f"  {k}: {type(v)}")

    def test_iterate_all_samples(self, real_dataset, real_config):
        modality_names = list(real_config["modalities"])
        self.log_first_example(real_dataset)
        for i in range(len(real_dataset)):
            sample = real_dataset[i]
            # Smoke-check: every modality key present and not None.
            for name in modality_names:
                assert name in sample
                assert sample[name] is not None
            assert sample["id"] is not None

    def test_unique_ids(self, real_dataset):
        ids = [real_dataset[i]["id"] for i in range(len(real_dataset))]
        assert len(ids) == len(set(ids)), "Duplicate sample IDs found"

    def test_all_ids_are_strings(self, real_dataset):
        for i in range(len(real_dataset)):
            assert isinstance(real_dataset[i]["id"], str)
