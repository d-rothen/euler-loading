"""Tests for euler_loading.indexing."""

from __future__ import annotations

from euler_loading.indexing import (
    FileRecord,
    collect_files,
    collect_hierarchical_files,
    match_hierarchical_files,
)


class TestCollectFilesFlat:
    """Flat nodes (no hierarchy)."""

    def test_returns_all_files(self, flat_index):
        records = collect_files(flat_index["dataset"])
        assert len(records) == 2
        ids = {r.file_entry["id"] for r in records}
        assert ids == {"frame-001", "frame-002"}


class TestEmptyTree:
    def test_empty_node_returns_empty(self):
        records = collect_files({})
        assert records == []

    def test_children_only_no_files(self):
        node = {"children": {"level:A": {"children": {"level:B": {}}}}}
        records = collect_files(node)
        assert records == []


class TestHierarchyPath:
    """Hierarchy path tracking in collect_files."""

    def test_flat_files_have_empty_path(self, flat_index):
        records = collect_files(flat_index["dataset"])
        for rec in records:
            assert rec.hierarchy_path == ()

    def test_nested_files_have_correct_path(self, hierarchical_index):
        records = collect_files(hierarchical_index["dataset"])
        for rec in records:
            assert rec.hierarchy_path == (
                "scene:Scene01",
                "camera:Camera_0",
                "frame:" + rec.file_entry["id"].split("+")[-1].split("-", 1)[1],
            )

    def test_multi_camera_paths_differ(self, multi_camera_index):
        records = collect_files(multi_camera_index["dataset"])
        by_id = {r.file_entry["id"]: r for r in records}

        cam0 = by_id["scene-Scene01+camera-Camera_0+frame-00001"]
        assert cam0.hierarchy_path[:2] == ("scene:Scene01", "camera:Camera_0")

        cam1 = by_id["scene-Scene01+camera-Camera_1+frame-00001"]
        assert cam1.hierarchy_path[:2] == ("scene:Scene01", "camera:Camera_1")


class TestCollectHierarchicalFiles:
    """Tests for collect_hierarchical_files."""

    def test_empty_tree(self):
        assert collect_hierarchical_files({}) == {}

    def test_files_at_root(self, flat_index):
        result = collect_hierarchical_files(flat_index["dataset"])
        assert () in result
        assert len(result[()]) == 2

    def test_files_at_intermediate_level(self, hierarchical_modality_index):
        result = collect_hierarchical_files(
            hierarchical_modality_index["dataset"]
        )
        assert ("Scene01", "sunset") in result
        assert len(result[("Scene01", "sunset")]) == 1
        assert result[("Scene01", "sunset")][0]["id"] == "intrinsic"

    def test_files_at_multiple_levels(self, multi_level_hierarchical_index):
        result = collect_hierarchical_files(
            multi_level_hierarchical_index["dataset"]
        )
        assert ("Scene01",) in result
        assert ("Scene01", "sunset") in result
        assert result[("Scene01",)][0]["id"] == "scene_meta"
        assert result[("Scene01", "sunset")][0]["id"] == "intrinsic"

    def test_no_files_returns_empty(self):
        node = {"children": {"A": {"children": {"B": {}}}}}
        assert collect_hierarchical_files(node) == {}


class TestMatchHierarchicalFiles:
    """Tests for match_hierarchical_files."""

    def test_exact_match(self):
        files_by_level = {
            ("A", "B"): [{"id": "f1", "path": "A/B/f1.txt"}],
        }
        result = match_hierarchical_files(("A", "B"), files_by_level)
        assert len(result) == 1
        assert result[0]["id"] == "f1"

    def test_prefix_match(self):
        files_by_level = {
            ("A",): [{"id": "f1", "path": "A/f1.txt"}],
        }
        result = match_hierarchical_files(("A", "B", "C"), files_by_level)
        assert len(result) == 1
        assert result[0]["id"] == "f1"

    def test_root_match(self):
        files_by_level = {
            (): [{"id": "f1", "path": "f1.txt"}],
        }
        result = match_hierarchical_files(("A", "B"), files_by_level)
        assert len(result) == 1
        assert result[0]["id"] == "f1"

    def test_no_match(self):
        files_by_level = {
            ("X", "Y"): [{"id": "f1", "path": "X/Y/f1.txt"}],
        }
        result = match_hierarchical_files(("A", "B"), files_by_level)
        assert result == []

    def test_deepest_wins_for_same_id(self):
        files_by_level = {
            ("A",): [{"id": "f1", "path": "A/f1_shallow.txt"}],
            ("A", "B"): [{"id": "f1", "path": "A/B/f1_deep.txt"}],
        }
        result = match_hierarchical_files(("A", "B", "C"), files_by_level)
        assert len(result) == 1
        assert result[0]["path"] == "A/B/f1_deep.txt"

    def test_multiple_levels_different_ids(self):
        files_by_level = {
            ("A",): [{"id": "scene_meta", "path": "A/meta.json"}],
            ("A", "B"): [{"id": "intrinsic", "path": "A/B/intr.txt"}],
        }
        result = match_hierarchical_files(("A", "B", "C"), files_by_level)
        ids = {e["id"] for e in result}
        assert ids == {"scene_meta", "intrinsic"}

    def test_empty_hierarchy_path_matches_root_only(self):
        files_by_level = {
            (): [{"id": "root", "path": "root.txt"}],
            ("A",): [{"id": "deep", "path": "A/deep.txt"}],
        }
        result = match_hierarchical_files((), files_by_level)
        assert len(result) == 1
        assert result[0]["id"] == "root"


class TestFileRecordImmutability:
    def test_frozen(self, flat_index):
        records = collect_files(flat_index["dataset"])
        rec = records[0]
        try:
            rec.file_entry = {}  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass
