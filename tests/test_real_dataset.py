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
from typing import Any, Callable

import pytest

from euler_loading import Modality, MultiModalDataset


# ---------------------------------------------------------------------------
# Dataset configurations  (stubs — fill in per database)
# ---------------------------------------------------------------------------
#
# Each value is a dict with the keys accepted by ``MultiModalDataset``:
#
#   modalities      : dict[str, Modality]                       — REQUIRED
#   read_intrinsics : Callable[[str], Any] | None               — optional
#   read_extrinsics : Callable[[str], Any] | None               — optional
#   transforms      : list[Callable[[dict], dict]] | None       — optional
#
# Leave ``modalities`` empty (``{}``) for databases that are not yet wired up;
# those entries are silently skipped by the test suite.

def load_rgb(path: str) -> Any:
    """Load an RGB image from disk given its file path."""
    from PIL import Image

    return Image.open(path).convert("RGB")

def load_depth(path: str) -> Any:
    """Load a depth map from disk given its file path."""
    from PIL import Image

    return Image.open(path).convert("RGB") 

def load_class_segmentation(path: str) -> Any:
    """Load a class segmentation map from disk given its file path."""
    from PIL import Image

    return Image.open(path).convert("RGB")  # Example loader; replace with actual implementation


REAL_DATASETS: dict[str, dict[str, Any]] = {
    "VKITTI2": {
        "modalities": {
            "rgb":   Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_rgb",   loader=load_rgb),
            "depth": Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_depth", loader=load_depth),
            "classSegmentation": Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_classSegmentation", loader=load_class_segmentation),
            "hazyRgb": Modality("/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_hazyRgb", loader=load_rgb),
        },
        "read_intrinsics": None,
        "read_extrinsics": None,
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
        read_intrinsics=real_config["read_intrinsics"],
        read_extrinsics=real_config["read_extrinsics"],
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
        for key in ("id", "meta", "intrinsics", "extrinsics"):
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
# Calibration
# ---------------------------------------------------------------------------


@real
class TestRealCalibration:
    """Calibration data respects whether readers are configured."""

    def test_intrinsics_present_when_reader_set(self, real_dataset, real_config):
        if real_config["read_intrinsics"] is None:
            pytest.skip("No intrinsics reader configured")
        sample = real_dataset[0]
        assert sample["intrinsics"] is not None

    def test_extrinsics_present_when_reader_set(self, real_dataset, real_config):
        if real_config["read_extrinsics"] is None:
            pytest.skip("No extrinsics reader configured")
        sample = real_dataset[0]
        assert sample["extrinsics"] is not None

    def test_intrinsics_none_when_no_reader(self, real_dataset, real_config):
        if real_config["read_intrinsics"] is not None:
            pytest.skip("Intrinsics reader is configured")
        sample = real_dataset[0]
        assert sample["intrinsics"] is None

    def test_extrinsics_none_when_no_reader(self, real_dataset, real_config):
        if real_config["read_extrinsics"] is not None:
            pytest.skip("Extrinsics reader is configured")
        sample = real_dataset[0]
        assert sample["extrinsics"] is None

    def test_calibration_cached_across_samples(self, real_dataset, real_config):
        """Accessing two samples that share calibration reuses the cache."""
        if len(real_dataset) < 2:
            pytest.skip("Need at least two samples to verify caching")
        s0 = real_dataset[0]
        s1 = real_dataset[1]
        # When both samples come from the same calibration file the objects
        # should be identical (same id), not merely equal.
        if s0["intrinsics"] is not None and s1["intrinsics"] is not None:
            assert s0["intrinsics"] is s1["intrinsics"]
        if s0["extrinsics"] is not None and s1["extrinsics"] is not None:
            assert s0["extrinsics"] is s1["extrinsics"]


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
