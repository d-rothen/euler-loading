"""Tests for writer resolution and dataset write-back helpers."""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch
import zipfile

import numpy as np
import pytest
from PIL import Image

from euler_loading import Modality, MultiModalDataset, resolve_writer_module
from euler_loading.loaders.gpu import vkitti2 as gpu_vkitti2

from .conftest import dummy_loader


def _flat_index(
    extension: str,
    file_ids: list[str],
    *,
    name: str | None = None,
    type_name: str | None = None,
    euler_train: dict[str, Any] | None = None,
    euler_loading: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_type = type_name or extension
    index: dict[str, Any] = {
        "name": name or resolved_type,
        "type": resolved_type,
        "euler_train": euler_train or {
            "used_as": "input",
            "modality_type": resolved_type,
        },
        "dataset": {
            "files": [
                {
                    "id": fid,
                    "path": f"scene/{fid}.{extension}",
                    "path_properties": {},
                    "basename_properties": {},
                }
                for fid in file_ids
            ]
        }
    }
    if euler_loading is not None:
        index["euler_loading"] = euler_loading
    if meta is not None:
        index["meta"] = meta
    return index


class TestWriterResolution:
    def test_resolve_writer_module(self):
        module = resolve_writer_module("vkitti2")
        assert module.__name__ == "euler_loading.loaders.gpu.vkitti2"

    def test_auto_resolves_writer_from_loader_metadata(self):
        index = _flat_index(
            "png",
            ["f001"],
            euler_loading={"loader": "vkitti2", "function": "depth"},
        )
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"depth": Modality("/data/depth")}
            )

        assert ds.get_writer("depth") is gpu_vkitti2.write_depth

    def test_auto_resolves_read_function_writer_alias(self):
        index = _flat_index(
            "txt",
            ["f001"],
            euler_loading={"loader": "vkitti2", "function": "read_intrinsics"},
        )
        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"intrinsics": Modality("/data/intrinsics")}
            )

        assert ds.get_writer("intrinsics") is gpu_vkitti2.write_intrinsics


class TestWriteSample:
    def test_write_sample_preserves_relative_path(self, tmp_path):
        calls: list[tuple[str, Any, dict[str, Any] | None]] = []

        def writer(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
            calls.append((path, value, meta))
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))

        index = _flat_index("depth", ["f001"], meta={"unit": "meters"})

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "depth": Modality(
                        "/data/depth", loader=dummy_loader, writer=writer
                    )
                }
            )

        written = ds.write_sample(0, {"depth": "prediction"}, str(tmp_path))
        expected_path = tmp_path / "scene" / "f001.depth"

        assert written["depth"] == str(expected_path)
        assert expected_path.read_text() == "prediction"
        assert calls == [(str(expected_path), "prediction", {"unit": "meters"})]

    def test_write_sample_with_modality_specific_roots(self, tmp_path):
        def writer(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))

        rgb_index = _flat_index("rgb", ["f001"])
        depth_index = _flat_index("depth", ["f001"])

        def mock_index(path: str, **_: Any) -> dict[str, Any]:
            return rgb_index if "rgb" in path else depth_index

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader, writer=writer),
                    "depth": Modality("/data/depth", loader=dummy_loader, writer=writer),
                }
            )

        roots = {
            "rgb": str(tmp_path / "rgb_out"),
            "depth": str(tmp_path / "depth_out"),
        }
        written = ds.write_sample(0, {"rgb": "r", "depth": "d"}, roots)

        assert written["rgb"] == str(tmp_path / "rgb_out" / "scene" / "f001.rgb")
        assert written["depth"] == str(tmp_path / "depth_out" / "scene" / "f001.depth")

    def test_write_sample_raises_when_writer_missing(self):
        index = _flat_index("depth", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={"depth": Modality("/data/depth", loader=dummy_loader)}
            )

        with pytest.raises(ValueError, match="No writer configured"):
            ds.write_sample(0, {"depth": "prediction"}, "/tmp/out")

    def test_write_sample_with_dataset_writer_destination(self, tmp_path):
        calls: list[tuple[str, Any, dict[str, Any] | None]] = []

        def writer(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
            calls.append((path, value, meta))
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))

        index = _flat_index(
            "txt",
            ["f001"],
            euler_loading={"loader": "generic_dense_depth", "function": "rgb"},
            meta={"unit": "meters"},
        )

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "depth": Modality(
                        "/data/depth", loader=dummy_loader, writer=writer
                    )
                }
            )

        output_writer = ds.create_output_writer("depth", tmp_path / "out")
        written = ds.write_sample(0, {"depth": "prediction"}, output_writer)
        output_writer.save_index()

        expected_path = tmp_path / "out" / "scene" / "f001.txt"
        assert written["depth"] == str(expected_path)
        assert expected_path.read_text() == "prediction"
        assert calls == [(str(expected_path), "prediction", {"unit": "meters"})]

        with open(tmp_path / "out" / ".ds_crawler" / "output.json") as f:
            output_index = json.load(f)
        assert output_index["euler_loading"]["function"] == "rgb"

    def test_write_sample_with_zip_dataset_writer_destination(self, tmp_path):
        def writer(path: str, value: Any, meta: dict[str, Any] | None = None) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))

        index = _flat_index("txt", ["f001"])

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "depth": Modality(
                        "/data/depth", loader=dummy_loader, writer=writer
                    )
                }
            )

        output_writer = ds.create_output_writer("depth", tmp_path / "out.zip", zip=True)
        written = ds.write_sample(0, {"depth": "prediction"}, output_writer)
        output_writer.save_index()

        assert written["depth"] == f"{tmp_path / 'out.zip'}::scene/f001.txt"

        with zipfile.ZipFile(tmp_path / "out.zip") as zf:
            with zf.open("scene/f001.txt") as entry:
                assert entry.read().decode("utf-8") == "prediction"
            with zf.open(".ds_crawler/output.json") as entry:
                output_index = json.load(io.TextIOWrapper(entry, encoding="utf-8"))
        assert output_index["type"] == "txt"

    def test_write_sample_with_stream_writer_to_zip_destination(self, tmp_path):
        def writer(target: Any, value: Any, meta: dict[str, Any] | None = None) -> None:
            assert not isinstance(target, str)
            image = Image.fromarray(value, mode="RGB")
            image.save(target, format="PNG")

        writer.__euler_loading_supports_stream__ = True

        index = _flat_index(
            "png",
            ["f001"],
            type_name="rgb",
            meta={"range": [0, 255]},
        )

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=index,
        ):
            ds = MultiModalDataset(
                modalities={
                    "rgb": Modality("/data/rgb", loader=dummy_loader, writer=writer)
                }
            )

        output_writer = ds.create_output_writer("rgb", tmp_path / "rgb.zip", zip=True)
        image = np.full((4, 5, 3), 64, dtype=np.uint8)
        ds.write_sample(0, {"rgb": image}, output_writer)
        output_writer.save_index()

        with zipfile.ZipFile(tmp_path / "rgb.zip") as zf:
            with zf.open("scene/f001.png") as entry:
                loaded = np.array(Image.open(io.BytesIO(entry.read())))
        np.testing.assert_array_equal(loaded, image)
