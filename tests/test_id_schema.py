"""Tests for MultiModalDataset.describe_id_schema."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from euler_loading import Modality, MultiModalDataset

from .conftest import dummy_loader


def _flat_index(
    file_ids: list[str],
    *,
    indexing: dict[str, Any] | None = None,
    legacy_join_char: str | None = None,
) -> dict[str, Any]:
    index: dict[str, Any] = {
        "name": "test-ds",
        "type": "depth",
        "euler_train": {"used_as": "input", "modality_type": "depth"},
        "dataset": {
            "files": [
                {
                    "id": fid,
                    "path": f"scene/{fid}.png",
                    "path_properties": {},
                    "basename_properties": {},
                }
                for fid in file_ids
            ]
        },
    }
    if indexing is not None:
        index["indexing"] = indexing
    if legacy_join_char is not None:
        index["id_regex_join_char"] = legacy_join_char
    return index


class TestDescribeIdSchema:
    def test_modern_indexing_block(self):
        index = _flat_index(
            ["f001"],
            indexing={
                "id": {"regex": "^(.+)$", "join_char": "+"},
                "hierarchy": {"separator": "-"},
            },
        )
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"depth": Modality("/data/depth", loader=dummy_loader)}
            )
        schema = ds.describe_id_schema()
        assert schema["hierarchy_separator"] == "-"
        assert schema["id_join_char"] == "+"
        assert schema["full_id_separator"] == "/"
        assert schema["id_regex"] == "^(.+)$"

    def test_legacy_top_level_join_char(self):
        index = _flat_index(["f001"], legacy_join_char="|")
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"depth": Modality("/data/depth", loader=dummy_loader)}
            )
        schema = ds.describe_id_schema()
        assert schema["id_join_char"] == "|"
        assert schema["hierarchy_separator"] == "-"

    def test_missing_indexing_falls_back_to_defaults(self):
        index = _flat_index(["f001"])
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"depth": Modality("/data/depth", loader=dummy_loader)}
            )
        schema = ds.describe_id_schema()
        assert schema["hierarchy_separator"] == "-"
        assert schema["id_join_char"] == "+"
        assert schema["full_id_separator"] == "/"

    def test_divergent_modalities_record_overrides(self):
        rgb_index = _flat_index(
            ["f001"],
            indexing={"id": {"join_char": "+"}, "hierarchy": {"separator": "-"}},
        )
        depth_index = _flat_index(
            ["f001"],
            indexing={"id": {"join_char": "|"}, "hierarchy": {"separator": "-"}},
        )

        def mock_index(path: str, **_: Any) -> dict[str, Any]:
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader),
                    "depth": Modality("/data/depth", loader=dummy_loader),
                }
            )
        schema = ds.describe_id_schema()
        assert "modalities" in schema
        assert schema["modalities"]["depth"]["id_join_char"] == "|"
