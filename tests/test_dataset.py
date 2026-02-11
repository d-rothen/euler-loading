"""Tests for euler_loading.dataset."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from types import SimpleNamespace
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
