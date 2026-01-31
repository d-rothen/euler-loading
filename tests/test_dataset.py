"""Tests for euler_loading.dataset."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from euler_loading import Modality, MultiModalDataset

from .conftest import _make_file, dummy_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_index(modality: str, file_ids: list[str]) -> dict[str, Any]:
    """Build a minimal flat ds-crawler index."""
    return {
        "dataset": {
            "files": [
                {
                    "id": fid,
                    "path": f"{fid}.{modality}",
                    "path_properties": {},
                    "basename_properties": {},
                }
                for fid in file_ids
            ]
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicDataset:
    """Two modalities with fully overlapping IDs."""

    def _make(self, **kwargs):
        rgb_index = _flat_index("rgb", ["f001", "f002", "f003"])
        depth_index = _flat_index("depth", ["f001", "f002", "f003"])

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            return MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
                **kwargs,
            )

    def test_length(self):
        ds = self._make()
        assert len(ds) == 3

    def test_getitem_keys(self):
        ds = self._make()
        sample = ds[0]
        assert "rgb" in sample
        assert "depth" in sample
        assert "id" in sample
        assert "meta" in sample

    def test_loader_receives_correct_path(self):
        ds = self._make()
        sample = ds[0]
        # dummy_loader returns "loaded:<path>"
        assert sample["rgb"].startswith("loaded:/data/rgb/")
        assert sample["depth"].startswith("loaded:/data/depth/")

    def test_deterministic_ordering(self):
        ds = self._make()
        ids = [ds[i]["id"] for i in range(len(ds))]
        assert ids == sorted(ids)

    def test_meta_contains_file_entries(self):
        ds = self._make()
        sample = ds[0]
        for mod in ("rgb", "depth"):
            assert "id" in sample["meta"][mod]
            assert "path" in sample["meta"][mod]


class TestPartialOverlap:
    """Modalities with different file coverage."""

    def test_intersection(self):
        rgb_index = _flat_index("rgb", ["f001", "f002", "f003"])
        depth_index = _flat_index("depth", ["f002", "f003", "f004"])

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
            )
        assert len(ds) == 2
        ids = {ds[i]["id"] for i in range(len(ds))}
        assert ids == {"f002", "f003"}


class TestNoOverlapRaises:
    def test_raises_value_error(self):
        rgb_index = _flat_index("rgb", ["f001", "f002"])
        depth_index = _flat_index("depth", ["f003", "f004"])

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            with pytest.raises(ValueError, match="No common IDs"):
                MultiModalDataset(
                    modalities={
                        "rgb": Modality("/data/rgb", loader=dummy_loader),
                        "depth": Modality("/data/depth", loader=dummy_loader),
                    },
                )


class TestEmptyModalities:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one modality"):
            MultiModalDataset(modalities={})


class TestSingleModality:
    def test_works(self):
        index = _flat_index("rgb", ["f001", "f002"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
            )
        assert len(ds) == 2



class TestTransforms:
    """Transform application and ordering."""

    def _make(self, transforms):
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            return MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
                transforms=transforms,
            )

    def test_single_transform(self):
        def add_flag(sample):
            sample["flag"] = True
            return sample

        ds = self._make([add_flag])
        assert ds[0]["flag"] is True

    def test_transform_order(self):
        log: list[str] = []

        def first(sample):
            log.append("first")
            sample["order"] = ["first"]
            return sample

        def second(sample):
            log.append("second")
            sample["order"].append("second")
            return sample

        ds = self._make([first, second])
        sample = ds[0]
        assert sample["order"] == ["first", "second"]
        assert log == ["first", "second"]

    def test_cross_modal_transform(self):
        """Transform that reads one modality to modify another."""
        rgb_index = _flat_index("rgb", ["f001"])
        depth_index = _flat_index("depth", ["f001"])

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        def mask_depth(sample):
            # Simulate: if rgb says "mask", zero out depth.
            if "mask" in sample["rgb"]:
                sample["depth"] = "masked"
            return sample

        def loader_rgb(path):
            return "mask_signal"

        def loader_depth(path):
            return "raw_depth"

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=loader_rgb),
                    "depth": Modality("/data/depth", loader=loader_depth),
                },
                transforms=[mask_depth],
            )

        sample = ds[0]
        # "mask" is in "mask_signal", so depth should be masked.
        assert sample["depth"] == "masked"

    def test_transform_receives_full_context(self):
        """Ensure transform can access id and meta."""
        received_keys: set[str] = set()

        def capture_keys(sample):
            received_keys.update(sample.keys())
            return sample

        ds = self._make([capture_keys])
        _ = ds[0]

        assert "rgb" in received_keys
        assert "id" in received_keys
        assert "meta" in received_keys


# ---------------------------------------------------------------------------
# Hierarchical modality helpers
# ---------------------------------------------------------------------------

def _deep_regular_index(file_ids: list[str]) -> dict[str, Any]:
    """Regular-modality index: Scene01 → sunset → Camera_0 → files."""
    return {
        "dataset": {
            "children": {
                "Scene01": {
                    "children": {
                        "sunset": {
                            "children": {
                                "Camera_0": {
                                    "files": [
                                        _make_file(
                                            fid,
                                            f"Scene01/sunset/Camera_0/{fid}.png",
                                        )
                                        for fid in file_ids
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }


def _hierarchical_intrinsics_index() -> dict[str, Any]:
    """Hierarchical modality index: Scene01 → sunset → file(intrinsic)."""
    return {
        "dataset": {
            "children": {
                "Scene01": {
                    "children": {
                        "sunset": {
                            "files": [
                                _make_file(
                                    "intrinsic",
                                    "Scene01/sunset/intrinsic.txt",
                                )
                            ]
                        }
                    }
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Hierarchical modality tests
# ---------------------------------------------------------------------------

class TestHierarchicalModalities:
    """Hierarchical modalities matched by hierarchy path prefix."""

    def _make(self, **kwargs):
        rgb_index = _deep_regular_index(["f001", "f002"])
        hier_index = _hierarchical_intrinsics_index()

        def mock_index(path, **kw):
            if "rgb" in path:
                return rgb_index
            return hier_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            return MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "cam_intrinsics": Modality(
                        "/data/intrinsics", loader=dummy_loader
                    ),
                },
                **kwargs,
            )

    def test_sample_contains_hierarchical_key(self):
        ds = self._make()
        sample = ds[0]
        assert "cam_intrinsics" in sample

    def test_hierarchical_value_is_dict(self):
        ds = self._make()
        sample = ds[0]
        assert isinstance(sample["cam_intrinsics"], dict)

    def test_hierarchical_dict_has_correct_id(self):
        ds = self._make()
        sample = ds[0]
        assert "intrinsic" in sample["cam_intrinsics"]

    def test_hierarchical_loader_called_with_correct_path(self):
        ds = self._make()
        sample = ds[0]
        assert sample["cam_intrinsics"]["intrinsic"] == (
            "loaded:/data/intrinsics/Scene01/sunset/intrinsic.txt"
        )

    def test_hierarchical_does_not_affect_id_intersection(self):
        """Hierarchical modalities must not participate in ID intersection."""
        ds = self._make()
        assert len(ds) == 2
        ids = {ds[i]["id"] for i in range(len(ds))}
        assert ids == {"f001", "f002"}

    def test_hierarchical_shared_across_samples(self):
        """All samples under the same hierarchy get the same files."""
        ds = self._make()
        s0 = ds[0]["cam_intrinsics"]
        s1 = ds[1]["cam_intrinsics"]
        assert s0 == s1

    def test_hierarchical_cached(self):
        """Shared hierarchical files are loaded only once."""
        loader = MagicMock(return_value="parsed")
        rgb_index = _deep_regular_index(["f001", "f002", "f003"])
        hier_index = _hierarchical_intrinsics_index()

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else hier_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "cam_intrinsics": Modality("/data/intrinsics", loader=loader),
                },
            )

        # Access all three samples.
        for i in range(3):
            _ = ds[i]

        # The intrinsics file should have been loaded exactly once.
        loader.assert_called_once()

    def test_no_hierarchy_overlap_returns_empty_dict(self):
        """When hierarchical modality has no matching ancestors, result is {}."""
        rgb_index = _deep_regular_index(["f001"])
        # Hierarchical modality under a completely different scene.
        hier_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "OtherScene": {
                        "files": [
                            _make_file("intrinsic", "OtherScene/intrinsic.txt")
                        ]
                    }
                }
            }
        }

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else hier_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "cam_intrinsics": Modality(
                        "/data/intrinsics", loader=dummy_loader
                    ),
                },
            )

        assert ds[0]["cam_intrinsics"] == {}

    def test_multiple_files_at_different_levels(self):
        """Files from multiple ancestor levels are merged into one dict."""
        rgb_index = _deep_regular_index(["f001"])
        hier_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "Scene01": {
                        "files": [
                            _make_file("scene_meta", "Scene01/meta.json"),
                        ],
                        "children": {
                            "sunset": {
                                "files": [
                                    _make_file(
                                        "intrinsic",
                                        "Scene01/sunset/intrinsic.txt",
                                    )
                                ]
                            }
                        },
                    }
                }
            }
        }

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else hier_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "extras": Modality("/data/extras", loader=dummy_loader),
                },
            )

        result = ds[0]["extras"]
        assert "intrinsic" in result
        assert "scene_meta" in result

    def test_transform_sees_hierarchical_data(self):
        """Transforms receive hierarchical modality data in the sample dict."""
        received: dict[str, Any] = {}

        def capture(sample):
            received.update(sample)
            return sample

        ds = self._make(transforms=[capture])
        _ = ds[0]
        assert "cam_intrinsics" in received
        assert isinstance(received["cam_intrinsics"], dict)
