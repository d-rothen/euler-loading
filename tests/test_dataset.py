"""Tests for euler_loading.dataset."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from euler_loading import Modality, MultiModalDataset

from .conftest import dummy_loader


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


def _hierarchical_index_with_calibration(
    file_ids: list[str],
    ext: str = "png",
    intrinsics_rel: str | None = None,
    extrinsics_rel: str | None = None,
) -> dict[str, Any]:
    """Build an index where files sit under a camera node with calibration."""
    camera_node: dict[str, Any] = {
        "children": {
            f"frame:{fid}": {
                "files": [
                    {
                        "id": fid,
                        "path": f"cam/{fid}.{ext}",
                        "path_properties": {},
                        "basename_properties": {},
                    }
                ]
            }
            for fid in file_ids
        }
    }
    if intrinsics_rel:
        camera_node["camera_intrinsics"] = intrinsics_rel
    if extrinsics_rel:
        camera_node["camera_extrinsics"] = extrinsics_rel
    return {"dataset": {"children": {"camera:cam0": camera_node}}}


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
        assert "intrinsics" in sample
        assert "extrinsics" in sample

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


class TestCalibration:
    """Calibration loading, caching, and first-modality-wins semantics."""

    def _make_with_calibration(self):
        rgb_index = _hierarchical_index_with_calibration(
            ["f001", "f002"],
            ext="png",
            intrinsics_rel="cam/intr.txt",
            extrinsics_rel="cam/extr.txt",
        )
        depth_index = _hierarchical_index_with_calibration(
            ["f001", "f002"],
            ext="exr",
        )

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        reader_intr = MagicMock(return_value={"fx": 500})
        reader_extr = MagicMock(return_value={"R": "identity"})

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
                read_intrinsics=reader_intr,
                read_extrinsics=reader_extr,
            )
        return ds, reader_intr, reader_extr

    def test_calibration_present(self):
        ds, _, _ = self._make_with_calibration()
        sample = ds[0]
        assert sample["intrinsics"] == {"fx": 500}
        assert sample["extrinsics"] == {"R": "identity"}

    def test_calibration_cached(self):
        ds, reader_intr, reader_extr = self._make_with_calibration()
        _ = ds[0]
        _ = ds[1]
        # Both samples share the same calibration file → called only once each.
        reader_intr.assert_called_once()
        reader_extr.assert_called_once()

    def test_no_calibration_returns_none(self):
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
            )
        sample = ds[0]
        assert sample["intrinsics"] is None
        assert sample["extrinsics"] is None

    def test_no_reader_returns_none(self):
        """Calibration path exists but no reader was provided."""
        index = _hierarchical_index_with_calibration(
            ["f001"], intrinsics_rel="intr.txt"
        )

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
                # read_intrinsics not set
            )
        sample = ds[0]
        assert sample["intrinsics"] is None


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
        """Ensure transform can access intrinsics, extrinsics, id, meta."""
        index = _hierarchical_index_with_calibration(
            ["f001"], intrinsics_rel="intr.txt", extrinsics_rel="extr.txt"
        )

        received_keys: set[str] = set()

        def capture_keys(sample):
            received_keys.update(sample.keys())
            return sample

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
                read_intrinsics=lambda p: "K",
                read_extrinsics=lambda p: "E",
                transforms=[capture_keys],
            )
        _ = ds[0]

        assert "rgb" in received_keys
        assert "intrinsics" in received_keys
        assert "extrinsics" in received_keys
        assert "id" in received_keys
        assert "meta" in received_keys
