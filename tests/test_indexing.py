"""Tests for euler_loading.indexing."""

from __future__ import annotations

from euler_loading.indexing import FileRecord, collect_files_with_calibration


ROOT = "/data/test"


class TestCollectFilesFlat:
    """Flat nodes (no hierarchy)."""

    def test_returns_all_files(self, flat_index):
        records = collect_files_with_calibration(flat_index["dataset"], ROOT)
        assert len(records) == 2
        ids = {r.file_entry["id"] for r in records}
        assert ids == {"frame-001", "frame-002"}

    def test_no_calibration_when_absent(self, flat_index):
        records = collect_files_with_calibration(flat_index["dataset"], ROOT)
        for rec in records:
            assert rec.intrinsics_path is None
            assert rec.extrinsics_path is None


class TestCalibrationInheritance:
    """Calibration at an intermediate level inherited by descendant files."""

    def test_files_inherit_camera_calibration(self, hierarchical_index):
        records = collect_files_with_calibration(
            hierarchical_index["dataset"], ROOT
        )
        assert len(records) == 2
        for rec in records:
            assert rec.intrinsics_path == f"{ROOT}/Scene01/Camera_0_intr.txt"
            assert rec.extrinsics_path == f"{ROOT}/Scene01/Camera_0_extr.txt"

    def test_only_camera_with_calibration_gets_it(self, multi_camera_index):
        records = collect_files_with_calibration(
            multi_camera_index["dataset"], ROOT
        )
        by_id = {r.file_entry["id"]: r for r in records}

        cam0 = by_id["scene-Scene01+camera-Camera_0+frame-00001"]
        assert cam0.intrinsics_path == f"{ROOT}/Scene01/Camera_0_intr.txt"

        cam1 = by_id["scene-Scene01+camera-Camera_1+frame-00001"]
        assert cam1.intrinsics_path is None


class TestCalibrationOverride:
    """Child node overrides parent calibration."""

    def test_child_override(self, override_calibration_index):
        records = collect_files_with_calibration(
            override_calibration_index["dataset"], ROOT
        )
        by_id = {r.file_entry["id"]: r for r in records}

        # Camera_0 overrides the scene-level calibration.
        cam0 = by_id["s01-c0-f001"]
        assert cam0.intrinsics_path == f"{ROOT}/Scene01/Camera_0_intr.txt"

        # Camera_1 has no override → inherits scene-level calibration.
        cam1 = by_id["s01-c1-f001"]
        assert cam1.intrinsics_path == f"{ROOT}/Scene01/scene_intr.txt"


class TestEmptyTree:
    def test_empty_node_returns_empty(self):
        records = collect_files_with_calibration({}, ROOT)
        assert records == []

    def test_children_only_no_files(self):
        node = {"children": {"level:A": {"children": {"level:B": {}}}}}
        records = collect_files_with_calibration(node, ROOT)
        assert records == []


class TestFileRecordImmutability:
    def test_frozen(self, flat_index):
        records = collect_files_with_calibration(flat_index["dataset"], ROOT)
        rec = records[0]
        try:
            rec.intrinsics_path = "should fail"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass
