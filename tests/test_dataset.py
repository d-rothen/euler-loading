"""Tests for euler_loading.dataset."""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ds_crawler import DatasetWriter

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


def _write_inline_split(
    root: os.PathLike[str] | str,
    split_name: str,
    dataset_node: dict[str, Any],
) -> None:
    metadata_dir = os.path.join(root, ".ds_crawler")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, f"split_{split_name}.json"), "w") as f:
        json.dump(dataset_node, f)


def _create_zip_with_inline_split(
    tmp_path,
    *,
    name: str,
    files: dict[str, bytes],
    split_name: str,
    dataset_node: dict[str, Any],
    prefix: str = "",
) -> str:
    zip_path = os.path.join(tmp_path, name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for entry_name, content in files.items():
            zf.writestr(prefix + entry_name, content)
        zf.writestr(
            f"{prefix}.ds_crawler/split_{split_name}.json",
            json.dumps(dataset_node),
        )
    return zip_path


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


class TestInlineSplitLoading:
    def test_regular_modalities_load_inline_splits_from_filesystem(self, tmp_path):
        rgb_root = tmp_path / "rgb"
        depth_root = tmp_path / "depth"
        rgb_root.mkdir()
        depth_root.mkdir()

        rgb_index = _flat_index("rgb", ["f001", "f002", "f003"])
        depth_index = _flat_index("depth", ["f001", "f002", "f003"])
        rgb_split = _flat_index("rgb", ["f002", "f003"])["dataset"]
        depth_split = _flat_index("depth", ["f003"])["dataset"]
        _write_inline_split(rgb_root, "train", rgb_split)
        _write_inline_split(depth_root, "train", depth_split)

        def mock_index(path, **kw):
            if str(path) == str(rgb_root):
                return rgb_index
            return depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality(
                        str(rgb_root), split="train", loader=dummy_loader
                    ),
                    "depth": Modality(
                        str(depth_root), split="train", loader=dummy_loader
                    ),
                },
            )

        assert len(ds) == 1
        assert ds[0]["id"] == "f003"
        assert ds.modality_paths()["rgb"] == {
            "path": str(rgb_root),
            "origin_path": None,
            "split": "train",
        }
        assert ds.get_modality_index("rgb")["dataset"] == rgb_split

    def test_missing_inline_split_raises_file_not_found(self, tmp_path):
        root = tmp_path / "rgb"
        root.mkdir()
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ), pytest.raises(FileNotFoundError, match="split_train.json"):
            MultiModalDataset(
                modalities={
                    "rgb": Modality(str(root), split="train", loader=dummy_loader),
                },
            )

    def test_invalid_inline_split_name_raises_value_error(self, tmp_path):
        root = tmp_path / "rgb"
        root.mkdir()
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ), pytest.raises(ValueError, match="split_name"):
            MultiModalDataset(
                modalities={
                    "rgb": Modality(str(root), split="bad/name", loader=dummy_loader),
                },
            )

    def test_zip_modalities_load_inline_splits(self, tmp_path):
        split_dataset = _flat_index("png", ["f002"])["dataset"]
        zip_path = _create_zip_with_inline_split(
            tmp_path,
            name="rgb.zip",
            files={
                "f001.png": b"fake-png-001",
                "f002.png": b"fake-png-002",
            },
            split_name="train",
            dataset_node=split_dataset,
            prefix="rgb/",
        )
        full_index = _flat_index("png", ["f001", "f002"])
        loader = MagicMock(return_value="loaded")

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=full_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality(zip_path, split="train", loader=loader),
                },
            )

        assert len(ds) == 1
        sample = ds[0]
        assert sample["id"] == "f002"
        buf = loader.call_args[0][0]
        assert isinstance(buf, io.BytesIO)
        assert buf.read() == b"fake-png-002"


class TestPathWithColonSplit:
    """Colon-separated path:split syntax."""

    def test_split_extracted_from_path(self):
        mod = Modality("/data/ds:train")
        assert mod.path == "/data/ds"
        assert mod.split == "train"

    def test_split_extracted_from_zip_path(self):
        mod = Modality("/data/ds.zip:val")
        assert mod.path == "/data/ds.zip"
        assert mod.split == "val"

    def test_no_colon_leaves_split_none(self):
        mod = Modality("/data/ds")
        assert mod.path == "/data/ds"
        assert mod.split is None

    def test_explicit_split_without_colon_still_works(self):
        mod = Modality("/data/ds", split="train")
        assert mod.path == "/data/ds"
        assert mod.split == "train"

    def test_colon_and_explicit_split_raises(self):
        with pytest.raises(ValueError, match="inline split"):
            Modality("/data/ds:train", split="val")

    def test_windows_drive_letter_not_treated_as_split(self):
        mod = Modality("C:\\data\\ds")
        assert mod.path == "C:\\data\\ds"
        assert mod.split is None

    def test_invalid_split_suffix_left_in_path(self):
        # A colon followed by an invalid split name is left as-is
        mod = Modality("/data/ds:bad/name")
        assert mod.path == "/data/ds:bad/name"
        assert mod.split is None

    def test_colon_path_integration_with_dataset(self, tmp_path):
        root = tmp_path / "rgb"
        root.mkdir()
        rgb_index = _flat_index("rgb", ["f001", "f002"])
        rgb_split = _flat_index("rgb", ["f002"])["dataset"]
        _write_inline_split(root, "train", rgb_split)

        def mock_index(path, **kw):
            return rgb_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality(
                        f"{root}:train", loader=dummy_loader
                    ),
                },
            )

        assert len(ds) == 1
        assert ds[0]["id"] == "f002"
        assert ds.modality_paths()["rgb"] == {
            "path": str(root),
            "origin_path": None,
            "split": "train",
        }


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

        def loader_rgb(path, meta=None):
            return "mask_signal"

        def loader_depth(path, meta=None):
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
        """Shared hierarchical files are loaded only once (default behavior)."""
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

    def test_hierarchical_cache_false_reloads_every_access(self):
        """Opt-out via cache=False reloads even shared hierarchical files."""
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
                    "cam_intrinsics": Modality(
                        "/data/intrinsics", loader=loader, cache=False,
                    ),
                },
            )

        for i in range(3):
            _ = ds[i]

        # 3 accesses with caching disabled → 3 loads of the same file.
        assert loader.call_count == 3

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


# ---------------------------------------------------------------------------
# Augmented RGB folder + per-id GT depth via hierarchical modality
# ---------------------------------------------------------------------------
#
# Layout produced by ds-crawler for this case
# (see ds-crawler examples/augmented_rgb_example.py)::
#
#     RGB tree
#       children["file_id:abc"].files = [
#         {id: "aug-aug_1", path: "abc/aug_1.png"},
#         {id: "aug-aug_2", path: "abc/aug_2.png"},
#       ]
#       children["file_id:xyz"].files = [
#         {id: "aug-aug_1", path: "xyz/aug_1.png"},
#         {id: "aug-aug_2", path: "xyz/aug_2.png"},
#       ]
#
#     Depth tree
#       children["file_id:abc"].files = [{id: "file_id-abc", path: "abc.png"}]
#       children["file_id:xyz"].files = [{id: "file_id-xyz", path: "xyz.png"}]
#
# Wiring depth as a hierarchical modality means each per-aug RGB sample
# at hierarchy_path=("file_id:<id>",) finds the matching depth file at the
# same prefix.


def _augmented_rgb_index() -> dict[str, Any]:
    return {
        "dataset": {
            "children": {
                "file_id:abc": {
                    "files": [
                        _make_file("aug-aug_1", "abc/aug_1.png"),
                        _make_file("aug-aug_2", "abc/aug_2.png"),
                    ]
                },
                "file_id:xyz": {
                    "files": [
                        _make_file("aug-aug_1", "xyz/aug_1.png"),
                        _make_file("aug-aug_2", "xyz/aug_2.png"),
                    ]
                },
            }
        }
    }


def _per_id_depth_hierarchical_index() -> dict[str, Any]:
    return {
        "dataset": {
            "children": {
                "file_id:abc": {
                    "files": [_make_file("file_id-abc", "abc.png")]
                },
                "file_id:xyz": {
                    "files": [_make_file("file_id-xyz", "xyz.png")]
                },
            }
        }
    }


class TestAugmentedRgbWithHierarchicalDepth:
    """End-to-end check that a file-id-as-folder augmented RGB modality joins
    correctly to a per-id GT depth modality via ``hierarchical_modalities``.
    """

    def _make(self):
        rgb_index = _augmented_rgb_index()
        depth_index = _per_id_depth_hierarchical_index()

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            return MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
            )

    def test_one_sample_per_aug(self):
        ds = self._make()
        # 2 file-ids x 2 augs per file-id
        assert len(ds) == 4

    def test_each_sample_has_depth_keyed_by_file_id(self):
        ds = self._make()
        for i in range(len(ds)):
            sample = ds[i]
            assert "depth" in sample
            assert isinstance(sample["depth"], dict)
            assert len(sample["depth"]) == 1

    def test_depth_shared_within_file_id(self):
        """Both augs of the same file-id receive the same depth file."""
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}

        abc_aug_1 = by_full_id["/file_id:abc/aug-aug_1"]["depth"]
        abc_aug_2 = by_full_id["/file_id:abc/aug-aug_2"]["depth"]
        assert abc_aug_1 == abc_aug_2

        xyz_aug_1 = by_full_id["/file_id:xyz/aug-aug_1"]["depth"]
        xyz_aug_2 = by_full_id["/file_id:xyz/aug-aug_2"]["depth"]
        assert xyz_aug_1 == xyz_aug_2

    def test_depth_differs_across_file_ids(self):
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}
        abc = by_full_id["/file_id:abc/aug-aug_1"]["depth"]
        xyz = by_full_id["/file_id:xyz/aug-aug_1"]["depth"]
        assert abc != xyz

    def test_depth_loader_called_with_correct_path(self):
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}
        depth_for_abc = by_full_id["/file_id:abc/aug-aug_1"]["depth"]
        depth_for_xyz = by_full_id["/file_id:xyz/aug-aug_1"]["depth"]
        assert depth_for_abc == {"file_id-abc": "loaded:/data/depth/abc.png"}
        assert depth_for_xyz == {"file_id-xyz": "loaded:/data/depth/xyz.png"}

    def test_rgb_loader_receives_per_aug_path(self):
        ds = self._make()
        rgb_paths = {ds[i]["rgb"] for i in range(len(ds))}
        assert rgb_paths == {
            "loaded:/data/rgb/abc/aug_1.png",
            "loaded:/data/rgb/abc/aug_2.png",
            "loaded:/data/rgb/xyz/aug_1.png",
            "loaded:/data/rgb/xyz/aug_2.png",
        }

    def test_full_id_encodes_file_id_and_aug(self):
        ds = self._make()
        full_ids = {ds[i]["full_id"] for i in range(len(ds))}
        assert full_ids == {
            "/file_id:abc/aug-aug_1",
            "/file_id:abc/aug-aug_2",
            "/file_id:xyz/aug-aug_1",
            "/file_id:xyz/aug-aug_2",
        }

    def test_depth_loaded_once_per_file_id(self):
        """Hierarchical-modality cache: depth file loaded once per file-id,
        not once per augmentation.
        """
        loader = MagicMock(return_value="loaded-depth")
        rgb_index = _augmented_rgb_index()
        depth_index = _per_id_depth_hierarchical_index()

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "depth": Modality("/data/depth", loader=loader),
                },
            )

        for i in range(len(ds)):
            _ = ds[i]

        # 2 file-ids × 4 sample accesses → still only 2 depth loads.
        assert loader.call_count == 2


# ---------------------------------------------------------------------------
# Keyed modalities (parent-prefix join via deepest-key value)
# ---------------------------------------------------------------------------


def _keyed_aug_index(*, separator: str = ":") -> dict[str, Any]:
    """Augmented index with ``children[file_id:<id>]`` deepest level.

    Two file-ids each with two augmentations. The ``indexing.hierarchy``
    block carries the separator so the keyed-join code can decode the
    deepest hierarchy key.
    """
    return {
        "indexing": {
            "hierarchy": {"separator": separator},
            "id": {"join_char": "+"},
        },
        "dataset": {
            "children": {
                f"file_id{separator}abc": {
                    "files": [
                        _make_file("aug-aug_1", "abc/aug_1.png"),
                        _make_file("aug-aug_2", "abc/aug_2.png"),
                    ]
                },
                f"file_id{separator}xyz": {
                    "files": [
                        _make_file("aug-aug_1", "xyz/aug_1.png"),
                        _make_file("aug-aug_2", "xyz/aug_2.png"),
                    ]
                },
            }
        },
    }


def _keyed_gt_index() -> dict[str, Any]:
    """GT index where files live at the parent prefix with ``id`` == file_id."""
    return {
        "indexing": {
            "hierarchy": {"separator": ":"},
            "id": {"join_char": "+"},
        },
        "dataset": {
            "files": [
                _make_file("abc", "abc.png"),
                _make_file("xyz", "xyz.png"),
            ]
        },
    }


class TestKeyedModalities:
    """Per-aug regular sample joined to a single GT via deepest-key value."""

    def _make(self, **kwargs):
        rgb_index = _keyed_aug_index()
        gt_index = _keyed_gt_index()

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else gt_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            return MultiModalDataset(
                modalities={
                    "rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader),
                },
                keyed_modalities={
                    "depth": Modality(
                        "/data/gt_depth",
                        loader=dummy_loader,
                        keyed_by={"key_name": "file_id"},
                    ),
                },
                **kwargs,
            )

    def test_one_sample_per_aug(self):
        ds = self._make()
        assert len(ds) == 4  # 2 file_ids × 2 augs

    def test_each_sample_has_a_single_keyed_value(self):
        ds = self._make()
        sample = ds[0]
        assert "depth" in sample
        # Keyed modality returns a single value, not a {file_id: ...} dict.
        assert isinstance(sample["depth"], str)

    def test_depth_shared_within_file_id(self):
        """Both augs of the same file-id receive the same loaded GT."""
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}
        abc_aug_1 = by_full_id["/file_id:abc/aug-aug_1"]["depth"]
        abc_aug_2 = by_full_id["/file_id:abc/aug-aug_2"]["depth"]
        assert abc_aug_1 == abc_aug_2

    def test_different_file_ids_yield_different_gts(self):
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}
        assert (
            by_full_id["/file_id:abc/aug-aug_1"]["depth"]
            != by_full_id["/file_id:xyz/aug-aug_1"]["depth"]
        )

    def test_loader_receives_correct_gt_path(self):
        ds = self._make()
        by_full_id = {ds[i]["full_id"]: ds[i] for i in range(len(ds))}
        assert by_full_id["/file_id:abc/aug-aug_1"]["depth"] == (
            "loaded:/data/gt_depth/abc.png"
        )
        assert by_full_id["/file_id:xyz/aug-aug_1"]["depth"] == (
            "loaded:/data/gt_depth/xyz.png"
        )

    def test_meta_records_gt_file_entry(self):
        ds = self._make()
        sample = ds[0]
        assert sample["meta"]["depth"]["id"] in ("abc", "xyz")
        assert sample["meta"]["depth"]["path"].endswith(".png")

    def test_keyed_does_not_affect_id_intersection(self):
        """The 4-aug count comes from the regular modality alone."""
        ds = self._make()
        ids = {ds[i]["id"] for i in range(len(ds))}
        assert ids == {"aug-aug_1", "aug-aug_2"}


class TestKeyedModalityCaching:
    def _make_ds(self, *, cache: bool | None) -> tuple[Any, Any]:
        rgb_index = _keyed_aug_index()
        gt_index = _keyed_gt_index()
        loader = MagicMock(return_value="loaded-depth")

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else gt_index

        depth_kwargs: dict[str, Any] = {
            "loader": loader,
            "keyed_by": {"key_name": "file_id"},
        }
        if cache is not None:
            depth_kwargs["cache"] = cache

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader),
                },
                keyed_modalities={
                    "depth": Modality("/data/gt_depth", **depth_kwargs),
                },
            )
        return ds, loader

    def test_keyed_default_does_not_cache(self):
        """Default for keyed modalities is no cache (avoids OOM on large
        per-sample files like GT depth)."""
        ds, loader = self._make_ds(cache=None)
        for i in range(len(ds)):
            _ = ds[i]
        # 4 augs total → 4 loads, no reuse.
        assert loader.call_count == 4

    def test_keyed_cache_true_loads_each_gt_once(self):
        ds, loader = self._make_ds(cache=True)
        for i in range(len(ds)):
            _ = ds[i]
        # 2 distinct file-ids × 2 augs each → still only 2 loads.
        assert loader.call_count == 2

    def test_keyed_cache_false_reloads_every_access(self):
        ds, loader = self._make_ds(cache=False)
        for i in range(len(ds)):
            _ = ds[i]
        assert loader.call_count == 4


class TestKeyedModalityValidation:
    def _make_with(
        self,
        *,
        regulars: dict[str, Modality],
        keyed: dict[str, Modality],
        rgb_index: dict | None = None,
        strict_keyed: bool = False,
    ):
        rgb_index = rgb_index if rgb_index is not None else _keyed_aug_index()
        gt_index = _keyed_gt_index()

        def mock_index(path, **kw):
            return rgb_index if any(rk in path for rk in regulars) else gt_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            return MultiModalDataset(
                modalities=regulars,
                keyed_modalities=keyed,
                strict_keyed=strict_keyed,
            )

    def test_missing_key_name_auto_detects(self):
        """When all anchor samples share a deepest-key prefix, key_name is
        inferred from the data."""
        ds = self._make_with(
            regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
            keyed={"depth": Modality(
                "/data/gt_depth",
                loader=dummy_loader,
                keyed_by={},
            )},
        )
        assert len(ds) == 4

    def test_keyed_by_omitted_entirely_auto_detects(self):
        """Same auto-detection works when keyed_by is left unset."""
        ds = self._make_with(
            regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
            keyed={"depth": Modality(
                "/data/gt_depth",
                loader=dummy_loader,
            )},
        )
        assert len(ds) == 4

    def test_auto_detect_resolves_inferred_key_name_in_join_config(self):
        """The inferred key_name shows up in modality_paths()."""
        ds = self._make_with(
            regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
            keyed={"depth": Modality(
                "/data/gt_depth",
                loader=dummy_loader,
            )},
        )
        paths = ds.keyed_modality_paths()
        assert paths["depth"]["keyed_by_key_name"] == "file_id"
        assert paths["depth"]["keyed_by_modality"] == "rgb_aug"

    def test_auto_detect_raises_on_multiple_prefixes(self):
        """If the anchor has mixed prefixes (e.g. 'file_id:...' and
        'frame:...'), require an explicit key_name."""
        rgb_index = {
            "indexing": {"hierarchy": {"separator": ":"}, "id": {"join_char": "+"}},
            "dataset": {
                "children": {
                    "file_id:abc": {"files": [_make_file("aug-1", "abc/aug_1.png")]},
                    "frame:0001": {"files": [_make_file("aug-1", "0001/aug_1.png")]},
                },
            },
        }
        with pytest.raises(ValueError, match="multiple distinct deepest-key prefixes"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                )},
                rgb_index=rgb_index,
            )

    def test_auto_detect_raises_when_no_hierarchy(self):
        """Anchor with no hierarchy keys cannot supply a key_name."""
        rgb_index = {
            "indexing": {"hierarchy": {"separator": ":"}, "id": {"join_char": "+"}},
            "dataset": {
                "files": [_make_file("aug-1", "aug_1.png")],
            },
        }
        with pytest.raises(ValueError, match="no hierarchy keys with separator"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                )},
                rgb_index=rgb_index,
            )

    def test_unknown_anchor_modality_raises(self):
        with pytest.raises(ValueError, match="unknown regular modality"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                    keyed_by={"key_name": "file_id", "modality": "nope"},
                )},
            )

    def test_anchor_inferred_when_single_regular(self):
        ds = self._make_with(
            regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
            keyed={"depth": Modality(
                "/data/gt_depth",
                loader=dummy_loader,
                keyed_by={"key_name": "file_id"},
            )},
        )
        assert len(ds) == 4

    def test_anchor_required_when_multiple_regulars(self):
        rgb_index = _keyed_aug_index()
        gt_index = _keyed_gt_index()

        def mock_index(path, **kw):
            return gt_index if "gt" in path else rgb_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            with pytest.raises(ValueError, match="keyed_by.modality is required"):
                MultiModalDataset(
                    modalities={
                        "rgb_aug_1": Modality("/data/rgb_aug_1", loader=dummy_loader),
                        "rgb_aug_2": Modality("/data/rgb_aug_2", loader=dummy_loader),
                    },
                    keyed_modalities={
                        "depth": Modality(
                            "/data/gt_depth",
                            loader=dummy_loader,
                            keyed_by={"key_name": "file_id"},
                        ),
                    },
                )

    def test_anchor_separator_required(self):
        rgb_index = _keyed_aug_index()
        # Strip the separator from the regular index.
        rgb_index["indexing"]["hierarchy"].pop("separator")

        with pytest.raises(ValueError, match="hierarchy.separator"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                    keyed_by={"key_name": "file_id"},
                )},
                rgb_index=rgb_index,
            )

    def test_key_name_mismatch_drops_samples(self):
        """Wrong key_name (e.g. 'frame' when keys are 'file_id:...') drops
        all samples; we treat it as 0 valid joins."""
        with pytest.raises(ValueError, match="No samples remain"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                    keyed_by={"key_name": "frame"},
                )},
            )

    def test_key_name_mismatch_strict_raises_eagerly(self):
        with pytest.raises(ValueError, match="expected the deepest hierarchy key"):
            self._make_with(
                regulars={"rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader)},
                keyed={"depth": Modality(
                    "/data/gt_depth",
                    loader=dummy_loader,
                    keyed_by={"key_name": "frame"},
                )},
                strict_keyed=True,
            )

    def test_missing_join_drops_samples_warns(self, caplog):
        # GT index missing one of the file_ids.
        rgb_index = _keyed_aug_index()
        gt_index = {
            "indexing": {"hierarchy": {"separator": ":"}, "id": {"join_char": "+"}},
            "dataset": {
                "files": [_make_file("abc", "abc.png")],  # xyz missing
            },
        }

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else gt_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            with caplog.at_level(logging.WARNING, logger="euler_loading.dataset"):
                ds = MultiModalDataset(
                    modalities={
                        "rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader),
                    },
                    keyed_modalities={
                        "depth": Modality(
                            "/data/gt_depth",
                            loader=dummy_loader,
                            keyed_by={"key_name": "file_id"},
                        ),
                    },
                )

        assert len(ds) == 2  # only the abc file_id's two augs survive
        assert any("samples dropped" in r.message for r in caplog.records)


class TestKeyedModalityDiagnostic:
    """When a keyed-style modality is mistakenly passed under
    ``modalities=``, the no-common-ids error should hint at the fix."""

    def test_hint_fired_when_keyed_modality_passed_as_regular(self):
        rgb_index = _keyed_aug_index()
        gt_index = _keyed_gt_index()

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else gt_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            with pytest.raises(ValueError) as exc:
                MultiModalDataset(
                    modalities={
                        "rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader),
                        "depth": Modality("/data/gt_depth", loader=dummy_loader),
                    },
                )

        msg = str(exc.value)
        assert "No common IDs found" in msg
        assert "keyed_modalities" in msg
        assert "key_name='file_id'" in msg
        # Either direction may be reported; this dataset's geometry implies
        # depth is the keyed candidate, anchored on rgb_aug.
        assert "'depth'" in msg
        assert "'rgb_aug'" in msg

    def test_no_hint_when_modalities_are_unrelated(self):
        """Two unrelated regular modalities with no overlap and no parent
        relationship should NOT produce a misleading keyed hint."""
        idx_a = _flat_index("a", ["alpha"])
        idx_b = _flat_index("b", ["beta"])

        def mock_index(path, **kw):
            return idx_a if "/a" in path else idx_b

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            with pytest.raises(ValueError) as exc:
                MultiModalDataset(
                    modalities={
                        "a": Modality("/data/a", loader=dummy_loader),
                        "b": Modality("/data/b", loader=dummy_loader),
                    },
                )

        msg = str(exc.value)
        assert "No common IDs found" in msg
        assert "keyed_modalities" not in msg


class TestKeyedModalityWriteSample:
    def test_write_sample_writes_to_gt_shape(self, tmp_path):
        rgb_index = _keyed_aug_index()
        gt_index = _keyed_gt_index()
        from ds_crawler import DatasetWriter

        def file_writer(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))

        def mock_index(path, **kw):
            return rgb_index if "rgb" in path else gt_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb_aug": Modality("/data/rgb_aug", loader=dummy_loader),
                },
                keyed_modalities={
                    "depth": Modality(
                        "/data/gt_depth",
                        loader=dummy_loader,
                        writer=file_writer,
                        keyed_by={"key_name": "file_id"},
                    ),
                },
            )

        out = tmp_path / "depth_pred"
        out.mkdir()
        writer = DatasetWriter(
            out,
            head={
                "contract": {"kind": "dataset_head", "version": "1.0"},
                "dataset": {"id": "pred", "name": "Pred"},
                "modality": {
                    "key": "depth",
                    "meta": {
                        "radial_depth": False,
                        "scale_to_meters": 1.0,
                        "range": [0, 65535],
                    },
                },
                "addons": {},
            },
        )

        ds.write_sample(0, {"depth": "prediction"}, writer)
        writer.save_index()

        # Output should be flat, mirroring the GT shape (no file_id: child).
        produced = sorted(p.relative_to(out) for p in out.rglob("*.png"))
        # There's exactly one written GT file, and it must NOT live under a
        # synthetic file_id: subdir.
        assert len(produced) == 1
        assert "file_id" not in str(produced[0])


# ---------------------------------------------------------------------------
# Per-file ``attributes`` field exposed on samples
# ---------------------------------------------------------------------------


class TestSampleAttributes:
    """Sample dict surfaces per-file ``attributes`` two ways:

    1. Top-level convenience: ``sample["attributes"][modality_name]``.
    2. Nested via the existing meta dict:
       ``sample["meta"][modality_name].get("attributes")``.
    """

    def _index_with_attrs(self) -> dict[str, Any]:
        return {
            "dataset": {
                "files": [
                    {
                        "id": "f001",
                        "path": "f001.rgb",
                        "path_properties": {},
                        "basename_properties": {},
                        "attributes": {"weight": 0.42, "src": "blender"},
                    },
                    {
                        "id": "f002",
                        "path": "f002.rgb",
                        "path_properties": {},
                        "basename_properties": {},
                        # No attributes on this entry.
                    },
                ]
            }
        }

    def _make(self):
        rgb_index = self._index_with_attrs()
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=rgb_index,
        ):
            return MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
            )

    def test_sample_has_top_level_attributes_key(self):
        ds = self._make()
        sample = ds[0]
        assert "attributes" in sample
        assert isinstance(sample["attributes"], dict)

    def test_top_level_attributes_keyed_by_modality(self):
        ds = self._make()
        sample = ds[0]
        assert sample["attributes"]["rgb"] == {"weight": 0.42, "src": "blender"}

    def test_attributes_also_nested_under_meta(self):
        ds = self._make()
        sample = ds[0]
        assert sample["meta"]["rgb"]["attributes"] == {
            "weight": 0.42, "src": "blender",
        }

    def test_missing_attributes_yields_empty_dict_at_top_level(self):
        ds = self._make()
        # f002 has no attributes.
        by_id = {ds[i]["id"]: ds[i] for i in range(len(ds))}
        assert by_id["f002"]["attributes"]["rgb"] == {}

    def test_top_level_attributes_does_not_alias_nested_dict(self):
        """Mutating sample['attributes'][name] must not corrupt the index."""
        ds = self._make()
        sample = ds[0]
        sample["attributes"]["rgb"]["weight"] = 999.0
        # Re-fetch — the index should still hold the original.
        fresh = ds[0]
        assert fresh["attributes"]["rgb"]["weight"] == 0.42

    def test_attributes_round_trip_from_ds_crawler_writer(self, tmp_path):
        """Pin ds-crawler writer output to euler-loading sample exposure."""
        root = tmp_path / "txt"
        writer = DatasetWriter(
            root,
            name="RoundTrip",
            type="txt",
            euler_train={"used_as": "input", "modality_type": "txt"},
        )
        path = writer.get_path(
            "/scene:Scene01/f001",
            "f001.txt",
            attributes={"weight": 0.42, "src": "ds-crawler"},
        )
        path.write_text("payload", encoding="utf-8")
        writer.save_index()

        def load_text(path: str, meta: dict[str, Any] | None = None) -> str:
            del meta
            with open(path, encoding="utf-8") as f:
                return f.read()

        ds = MultiModalDataset(
            modalities={"txt": Modality(str(root), loader=load_text)},
        )
        sample = ds[0]

        assert sample["txt"] == "payload"
        assert sample["attributes"]["txt"] == {"weight": 0.42, "src": "ds-crawler"}
        assert sample["meta"]["txt"]["attributes"] == {
            "weight": 0.42,
            "src": "ds-crawler",
        }


class TestLoaderAttributes:
    """Per-file attributes are passed only to loaders that opt in."""

    def _index_with_attrs(self, attributes: dict[str, Any] | None) -> dict[str, Any]:
        entry = {
            "id": "f001",
            "path": "f001.rgb",
            "path_properties": {},
            "basename_properties": {},
        }
        if attributes is not None:
            entry["attributes"] = attributes
        return {"dataset": {"files": [entry]}}

    def test_loader_with_attributes_kwarg_receives_file_attributes(self):
        received: list[dict[str, Any] | None] = []

        def loader(
            path: str,
            meta: dict[str, Any] | None = None,
            *,
            attributes: dict[str, Any] | None = None,
        ) -> str:
            del path, meta
            received.append(attributes)
            if attributes is not None:
                attributes["weight"] = 999.0
            return "loaded"

        index = self._index_with_attrs({"weight": 0.42, "src": "blender"})
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=loader)},
            )

        sample = ds[0]

        assert received == [{"weight": 999.0, "src": "blender"}]
        assert sample["attributes"]["rgb"] == {"weight": 0.42, "src": "blender"}
        assert ds[0]["attributes"]["rgb"] == {"weight": 0.42, "src": "blender"}

    def test_legacy_loader_does_not_receive_attributes_kwarg(self):
        calls: list[tuple[str, dict[str, Any] | None]] = []

        def loader(path: str, meta: dict[str, Any] | None = None) -> str:
            calls.append((path, meta))
            return "loaded"

        index = self._index_with_attrs({"weight": 0.42})
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=loader)},
            )

        assert ds[0]["rgb"] == "loaded"
        assert calls == [("/data/rgb/f001.rgb", None)]

    def test_loader_with_kwargs_receives_attributes(self):
        received: list[dict[str, Any] | None] = []

        def loader(
            path: str,
            meta: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> str:
            del path, meta
            received.append(kwargs.get("attributes"))
            return "loaded"

        index = self._index_with_attrs({"weight": 0.42})
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=loader)},
            )

        assert ds[0]["rgb"] == "loaded"
        assert received == [{"weight": 0.42}]

    def test_loader_with_attributes_kwarg_receives_none_when_missing(self):
        received: list[dict[str, Any] | None] = []

        def loader(
            path: str,
            meta: dict[str, Any] | None = None,
            *,
            attributes: dict[str, Any] | None = None,
        ) -> str:
            del path, meta
            received.append(attributes)
            return "loaded"

        index = self._index_with_attrs(None)
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=loader)},
            )

        assert ds[0]["rgb"] == "loaded"
        assert received == [None]


class TestHierarchicalModalityAttributes:
    """Hierarchical-modality attributes appear as ``{file_id: {...}}`` per modality."""

    def _make(self):
        rgb_index = _deep_regular_index(["f001"])
        hier_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "Scene01": {
                        "children": {
                            "sunset": {
                                "files": [
                                    {
                                        "id": "intrinsic",
                                        "path": "Scene01/sunset/intrinsic.txt",
                                        "path_properties": {},
                                        "basename_properties": {},
                                        "attributes": {"sensor": "FLIR"},
                                    }
                                ]
                            }
                        }
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
            return MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "cam_intrinsics": Modality(
                        "/data/intrinsics", loader=dummy_loader
                    ),
                },
            )

    def test_hierarchical_attributes_keyed_by_file_id(self):
        ds = self._make()
        sample = ds[0]
        assert sample["attributes"]["cam_intrinsics"] == {
            "intrinsic": {"sensor": "FLIR"},
        }

    def test_hierarchical_cache_key_includes_attributes_for_opted_in_loader(self):
        rgb_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "Scene01": {
                        "children": {
                            "day": {
                                "files": [
                                    _make_file("f001", "Scene01/day/f001.rgb"),
                                ],
                            },
                            "night": {
                                "files": [
                                    _make_file("f002", "Scene01/night/f002.rgb"),
                                ],
                            },
                        }
                    }
                }
            }
        }
        hier_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "Scene01": {
                        "children": {
                            "day": {
                                "files": [
                                    {
                                        "id": "intrinsic",
                                        "path": "Scene01/intrinsic.txt",
                                        "path_properties": {},
                                        "basename_properties": {},
                                        "attributes": {"variant": "day"},
                                    }
                                ],
                            },
                            "night": {
                                "files": [
                                    {
                                        "id": "intrinsic",
                                        "path": "Scene01/intrinsic.txt",
                                        "path_properties": {},
                                        "basename_properties": {},
                                        "attributes": {"variant": "night"},
                                    }
                                ],
                            },
                        }
                    }
                }
            }
        }
        calls: list[dict[str, Any] | None] = []

        def hier_loader(
            path: str,
            meta: dict[str, Any] | None = None,
            *,
            attributes: dict[str, Any] | None = None,
        ) -> str:
            del path, meta
            calls.append(attributes)
            assert attributes is not None
            return attributes["variant"]

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
                        "/data/intrinsics", loader=hier_loader
                    ),
                },
            )

        samples = [ds[i] for i in range(len(ds))]

        assert {sample["cam_intrinsics"]["intrinsic"] for sample in samples} == {
            "day",
            "night",
        }
        assert calls == [{"variant": "day"}, {"variant": "night"}]


# ---------------------------------------------------------------------------
# Multi-scene with duplicate bare IDs
# ---------------------------------------------------------------------------

def _multi_scene_index(modality: str) -> dict[str, Any]:
    """Two scenes whose files share the same bare IDs (e.g. ``f001``)."""
    return {
        "dataset": {
            "children": {
                "SceneA": {
                    "files": [
                        _make_file("f001", f"SceneA/f001.{modality}"),
                        _make_file("f002", f"SceneA/f002.{modality}"),
                    ]
                },
                "SceneB": {
                    "files": [
                        _make_file("f001", f"SceneB/f001.{modality}"),
                        _make_file("f002", f"SceneB/f002.{modality}"),
                    ]
                },
            }
        }
    }


class TestMultiSceneDuplicateIDs:
    """Scenes with overlapping bare file IDs must not collide."""

    def _make(self):
        rgb_index = _multi_scene_index("rgb")
        depth_index = _multi_scene_index("depth")

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
            )

    def test_all_files_preserved(self):
        """Four files (2 scenes × 2 frames) must not collapse to two."""
        ds = self._make()
        assert len(ds) == 4

    def test_bare_ids_appear_twice(self):
        """Each bare ID appears once per scene."""
        ds = self._make()
        bare_ids = [ds[i]["id"] for i in range(len(ds))]
        assert bare_ids.count("f001") == 2
        assert bare_ids.count("f002") == 2

    def test_full_ids_are_unique(self):
        ds = self._make()
        full_ids = {ds[i]["full_id"] for i in range(len(ds))}
        assert len(full_ids) == 4

    def test_full_id_encodes_scene(self):
        ds = self._make()
        full_ids = {ds[i]["full_id"] for i in range(len(ds))}
        assert "/SceneA/f001" in full_ids
        assert "/SceneA/f002" in full_ids
        assert "/SceneB/f001" in full_ids
        assert "/SceneB/f002" in full_ids

    def test_loader_receives_correct_scene_path(self):
        """Each sample must load from its own scene directory."""
        ds = self._make()
        paths = set()
        for i in range(len(ds)):
            sample = ds[i]
            paths.add(sample["rgb"])
        assert paths == {
            "loaded:/data/rgb/SceneA/f001.rgb",
            "loaded:/data/rgb/SceneA/f002.rgb",
            "loaded:/data/rgb/SceneB/f001.rgb",
            "loaded:/data/rgb/SceneB/f002.rgb",
        }


class TestRunlogDescription:
    def test_describe_for_runlog_prefers_explicit_modality_metadata(self):
        rgb_index = _flat_index("rgb", ["f001"])
        depth_index = _flat_index("depth", ["f001"])
        intrinsics_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "scene:Scene01": {
                        "children": {
                            "camera:Camera_0": {
                                "files": [
                                    _make_file("intrinsic", "Scene01/Camera_0/intrinsic.txt"),
                                ]
                            }
                        }
                    }
                }
            }
        }

        def mock_index(path, **kw):
            if "intrinsics" in path:
                return intrinsics_index
            if "depth" in path:
                return depth_index
            return rgb_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ), patch(
            "euler_loading.dataset.load_dataset_config",
            side_effect=FileNotFoundError,
        ):
            ds = MultiModalDataset(
                modalities={
                    "hazy_rgb": Modality(
                        "/data/rgb",
                        loader=dummy_loader,
                        used_as="input",
                        slot="dehaze.input.rgb",
                        modality_type="rgb",
                    ),
                    "depth": Modality(
                        "/data/depth",
                        loader=dummy_loader,
                        used_as="target",
                        slot="dehaze.target.depth",
                        modality_type="depth",
                    ),
                },
                hierarchical_modalities={
                    "camera_intrinsics": Modality(
                        "/data/intrinsics",
                        loader=dummy_loader,
                        used_as="condition",
                        slot="dehaze.condition.camera_intrinsics",
                        hierarchy_scope="scene_camera",
                        applies_to=["hazy_rgb", "depth"],
                    ),
                },
            )
            description = ds.describe_for_runlog()

        assert description == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/data/rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                },
                "depth": {
                    "path": "/data/depth",
                    "used_as": "target",
                    "slot": "dehaze.target.depth",
                    "modality_type": "depth",
                },
            },
            "hierarchical_modalities": {
                "camera_intrinsics": {
                    "path": "/data/intrinsics",
                    "used_as": "condition",
                    "slot": "dehaze.condition.camera_intrinsics",
                    "hierarchy_scope": "scene_camera",
                    "applies_to": ["hazy_rgb", "depth"],
                },
            },
            "keyed_modalities": {},
        }

    def test_describe_for_runlog_resolves_ds_crawler_properties(self):
        rgb_index = _flat_index("rgb", ["f001"])
        depth_index = _flat_index("depth", ["f001"])
        intrinsics_index: dict[str, Any] = {
            "dataset": {
                "children": {
                    "scene:Scene01": {
                        "children": {
                            "camera:Camera_0": {
                                "files": [
                                    _make_file("intrinsic", "Scene01/Camera_0/intrinsic.txt"),
                                ]
                            }
                        }
                    }
                }
            }
        }

        def mock_index(path, **kw):
            if "intrinsics" in path:
                return intrinsics_index
            if "depth" in path:
                return depth_index
            return rgb_index

        def mock_load_dataset_config(data):
            path = data["path"]
            by_path: dict[str, Any] = {
                "/data/rgb": SimpleNamespace(
                    type="rgb",
                    hierarchy_regex=None,
                    properties={
                        "euler_loading": {
                            "used_as": "input",
                            "slot": "dehaze.input.rgb",
                        }
                    },
                ),
                "/data/depth": SimpleNamespace(
                    type="depth",
                    hierarchy_regex=None,
                    properties={
                        "euler_loading": {
                            "used_as": "target",
                            "slot": "dehaze.target.depth",
                        }
                    },
                ),
                "/data/intrinsics": SimpleNamespace(
                    type="metadata",
                    hierarchy_regex=r"(?P<scene>[^/]+)/(?P<camera>[^/]+)",
                    properties={
                        "euler_loading": {
                            "used_as": "condition",
                            "slot": "dehaze.condition.camera_intrinsics",
                            "applies_to": ["hazy_rgb", "depth"],
                        }
                    },
                ),
            }
            return by_path[path]

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ), patch(
            "euler_loading.dataset.load_dataset_config",
            side_effect=mock_load_dataset_config,
        ):
            ds = MultiModalDataset(
                modalities={
                    "hazy_rgb": Modality("/data/rgb", loader=dummy_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "camera_intrinsics": Modality(
                        "/data/intrinsics", loader=dummy_loader
                    ),
                },
            )
            description = ds.describe_for_runlog()

        assert description == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/data/rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                },
                "depth": {
                    "path": "/data/depth",
                    "used_as": "target",
                    "slot": "dehaze.target.depth",
                    "modality_type": "depth",
                },
            },
            "hierarchical_modalities": {
                "camera_intrinsics": {
                    "path": "/data/intrinsics",
                    "used_as": "condition",
                    "slot": "dehaze.condition.camera_intrinsics",
                    "modality_type": "metadata",
                    "hierarchy_scope": "scene_camera",
                    "applies_to": ["hazy_rgb", "depth"],
                },
            },
            "keyed_modalities": {},
        }

    def test_describe_for_runlog_uses_euler_loading_namespace_only(self):
        rgb_index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=rgb_index,
        ), patch(
            "euler_loading.dataset.load_dataset_config",
            side_effect=FileNotFoundError,
        ):
            ds = MultiModalDataset(
                modalities={
                    "hazy_rgb": Modality(
                        "/data/rgb",
                        loader=dummy_loader,
                        metadata={
                            "runlog": {
                                "used_as": "target",
                                "slot": "legacy.target.rgb",
                            },
                            "euler_train": {
                                "used_as": "target",
                                "slot": "legacy2.target.rgb",
                            },
                            "euler_loading": {
                                "used_as": "input",
                                "slot": "dehaze.input.rgb",
                                "modality_type": "rgb",
                            },
                        },
                    )
                }
            )
            description = ds.describe_for_runlog()

        assert description == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/data/rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                }
            },
            "hierarchical_modalities": {},
            "keyed_modalities": {},
        }


# ---------------------------------------------------------------------------
# Zip modality tests
# ---------------------------------------------------------------------------

def _create_test_zip(tmp_path, name="modality.zip", files=None, prefix=""):
    """Create a zip file with dummy content files.

    Args:
        tmp_path: Directory where the zip is created.
        name: Filename of the zip archive.
        files: Dict of {entry_name: content_bytes}. Defaults to two PNGs.
        prefix: Optional root prefix inside the zip (simulates folder-wrapped zips).

    Returns:
        Path to the created zip file.
    """
    if files is None:
        files = {
            "f001.png": b"fake-png-001",
            "f002.png": b"fake-png-002",
        }
    zip_path = os.path.join(tmp_path, name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for entry_name, content in files.items():
            zf.writestr(prefix + entry_name, content)
    return zip_path


class TestZipModality:
    """Zip-backed modalities stream files via BytesIO."""

    def _make(self, tmp_path, *, zip_prefix="", loader=None, **kwargs):
        zip_path = _create_test_zip(tmp_path, prefix=zip_prefix)
        index = _flat_index("png", ["f001", "f002"])

        capture_loader = loader or MagicMock(return_value="loaded")

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ), patch(
            "euler_loading.dataset.is_zip_path",
            side_effect=lambda p: str(p).endswith(".zip"),
        ), patch(
            "euler_loading.dataset.get_zip_root_prefix",
            return_value=zip_prefix,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality(zip_path, loader=capture_loader),
                },
                **kwargs,
            )
        return ds, capture_loader

    def test_loader_receives_bytesio(self, tmp_path):
        ds, loader = self._make(tmp_path)
        _ = ds[0]
        args = loader.call_args[0]
        assert isinstance(args[0], io.BytesIO)

    def test_bytesio_has_name(self, tmp_path):
        ds, loader = self._make(tmp_path)
        _ = ds[0]
        buf = loader.call_args[0][0]
        assert hasattr(buf, "name")
        assert buf.name.endswith(".png")

    def test_bytesio_contains_correct_data(self, tmp_path):
        ds, loader = self._make(tmp_path)
        _ = ds[0]
        buf = loader.call_args[0][0]
        assert buf.read() == b"fake-png-001"

    def test_zip_prefix_stripped(self, tmp_path):
        """When the zip has a root prefix, entries are found correctly."""
        zip_path = _create_test_zip(
            tmp_path, files={"f001.png": b"data-001"}, prefix="wrapper/",
        )
        index = _flat_index("png", ["f001"])

        loader = MagicMock(return_value="loaded")
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ), patch(
            "euler_loading.dataset.is_zip_path",
            side_effect=lambda p: str(p).endswith(".zip"),
        ), patch(
            "euler_loading.dataset.get_zip_root_prefix",
            return_value="wrapper/",
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality(zip_path, loader=loader)},
            )

        _ = ds[0]
        buf = loader.call_args[0][0]
        assert buf.read() == b"data-001"

    def test_non_zip_still_gets_string_path(self, tmp_path):
        """Filesystem modalities are unaffected by zip support."""
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ), patch(
            "euler_loading.dataset.is_zip_path",
            return_value=False,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
            )
        sample = ds[0]
        assert isinstance(sample["rgb"], str)
        assert sample["rgb"].startswith("loaded:/data/rgb/")


class TestZipMixedModalities:
    """One zip modality + one filesystem modality in the same dataset."""

    def test_mixed(self, tmp_path):
        zip_path = _create_test_zip(tmp_path)
        rgb_index = _flat_index("png", ["f001", "f002"])
        depth_index = _flat_index("depth", ["f001", "f002"])

        zip_loader = MagicMock(return_value="zip-loaded")

        def mock_index(path, **kw):
            return rgb_index if str(path) == zip_path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ), patch(
            "euler_loading.dataset.is_zip_path",
            side_effect=lambda p: str(p).endswith(".zip"),
        ), patch(
            "euler_loading.dataset.get_zip_root_prefix",
            return_value="",
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality(zip_path, loader=zip_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                },
            )

        sample = ds[0]
        # zip modality got BytesIO
        buf = zip_loader.call_args[0][0]
        assert isinstance(buf, io.BytesIO)
        # filesystem modality got string
        assert isinstance(sample["depth"], str)
        assert sample["depth"].startswith("loaded:/data/depth/")


class TestZipHierarchicalModality:
    """Hierarchical modalities from zip archives."""

    def test_hierarchical_zip(self, tmp_path):
        zip_path = _create_test_zip(
            tmp_path,
            name="intrinsics.zip",
            files={"Scene01/sunset/intrinsic.txt": b"intrinsic-data"},
        )
        rgb_index = _deep_regular_index(["f001"])
        hier_index = _hierarchical_intrinsics_index()

        hier_loader = MagicMock(return_value="parsed-intrinsic")

        def mock_index(path, **kw):
            if "rgb" in str(path):
                return rgb_index
            return hier_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ), patch(
            "euler_loading.dataset.is_zip_path",
            side_effect=lambda p: str(p).endswith(".zip"),
        ), patch(
            "euler_loading.dataset.get_zip_root_prefix",
            return_value="",
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                },
                hierarchical_modalities={
                    "intrinsics": Modality(zip_path, loader=hier_loader),
                },
            )

        sample = ds[0]
        assert "intrinsics" in sample
        buf = hier_loader.call_args[0][0]
        assert isinstance(buf, io.BytesIO)
        assert buf.read() == b"intrinsic-data"


# ---------------------------------------------------------------------------
# get_dataset_name tests
# ---------------------------------------------------------------------------

class TestGetDatasetName:
    """Tests for MultiModalDataset.get_dataset_name()."""

    def test_returns_name_from_index(self):
        index = {**_flat_index("rgb", ["f001"]), "name": "vkitti2"}

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
            )

        assert ds.get_dataset_name() == "vkitti2"

    def test_returns_none_when_no_name(self):
        index = _flat_index("rgb", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"rgb": Modality("/data/rgb", loader=dummy_loader)},
            )

        assert ds.get_dataset_name() is None

    def test_returns_first_when_all_agree(self):
        rgb_index = {**_flat_index("rgb", ["f001"]), "name": "vkitti2"}
        depth_index = {**_flat_index("depth", ["f001"]), "name": "vkitti2"}

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

        assert ds.get_dataset_name() == "vkitti2"

    def test_warns_on_differing_names(self, caplog):
        rgb_index = {**_flat_index("rgb", ["f001"]), "name": "vkitti2"}
        depth_index = {**_flat_index("depth", ["f001"]), "name": "kitti"}

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

        with caplog.at_level(logging.WARNING, logger="euler_loading.dataset"):
            name = ds.get_dataset_name()

        assert name == "vkitti2"
        assert any("kitti" in record.message for record in caplog.records)

    def test_skips_modalities_without_name(self):
        rgb_index = _flat_index("rgb", ["f001"])  # no "name"
        depth_index = {**_flat_index("depth", ["f001"]), "name": "vkitti2"}

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

        assert ds.get_dataset_name() == "vkitti2"
