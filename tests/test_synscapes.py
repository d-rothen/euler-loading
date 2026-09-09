"""Synscapes format decoding and dataset integration tests."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import Imath
import numpy as np
import OpenEXR
from PIL import Image
import pytest
import torch

from euler_loading import DenseDepthLoader, Modality, MultiModalDataset, resolve_loader_module
from euler_loading.loaders.cpu import synscapes as cpu_synscapes
from euler_loading.loaders.generate.__main__ import generate
from euler_loading.loaders.gpu import synscapes as gpu_synscapes


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.synscapes")


@pytest.fixture(params=["str", "path", "stream"])
def source(request):
    def make_source(path):
        if request.param == "str":
            return str(path)
        if request.param == "stream":
            return io.BytesIO(path.read_bytes())
        return path

    return make_source


def _assert_loaded(result, expected, *, matrix=False):
    if isinstance(result, torch.Tensor):
        tensor = torch.from_numpy(expected)
        if not matrix:
            tensor = tensor.permute(2, 0, 1) if expected.ndim == 3 else tensor.unsqueeze(0)
        torch.testing.assert_close(result, tensor, equal_nan=True)
        assert result.is_contiguous()
    else:
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous
        assert result.flags.writeable


def _write_exr(path, values, *, offset=(0, 0), channel="Z"):
    height, width = values.shape
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(
        Imath.PixelType.HALF if values.dtype == np.float16 else Imath.PixelType.FLOAT
    )
    header["channels"] = {
        channel: Imath.Channel(pixel_type),
        "R": Imath.Channel(pixel_type),
    }
    x, y = offset
    header["dataWindow"] = Imath.Box2i(
        Imath.V2i(x, y), Imath.V2i(x + width - 1, y + height - 1)
    )
    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels({
            channel: values.tobytes(),
            "R": np.zeros_like(values).tobytes(),
        })
    finally:
        output.close()


@pytest.fixture()
def rgb_file(tmp_path):
    pixels = np.array(
        [[[0, 128, 255], [255, 64, 0], [32, 16, 8]],
         [[255, 255, 255], [0, 0, 0], [10, 20, 30]]],
        dtype=np.uint8,
    )
    path = tmp_path / "1.png"
    alpha = np.zeros((2, 3, 1), dtype=np.uint8)
    Image.fromarray(np.concatenate([pixels, alpha], axis=2)).save(path)
    return path, pixels.astype(np.float32) / 255.0


@pytest.fixture()
def depth_file(tmp_path):
    values = np.array(
        [[0.0, 1.25, 5000.5], [1e10, np.inf, np.nan]], dtype=np.float32
    )
    path = tmp_path / "1.exr"
    _write_exr(path, values, offset=(7, -3))
    return path, values


@pytest.fixture()
def intrinsics_file(tmp_path):
    data = {
        "camera": {
            "intrinsic": {
                "fx": 1590.83437,
                "fy": 1592.79032,
                "u0": 771.31406,
                "v0": 360.79945,
                "resx": 1440,
                "resy": 720,
            },
            "extrinsic": {"pitch": 0.038, "x": 1.7},
        },
        "scene": {"ego_speed": 2.0},
    }
    path = tmp_path / "1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    expected = np.array(
        [[1590.83437, 0, 771.31406], [0, 1592.79032, 360.79945], [0, 0, 1]],
        dtype=np.float32,
    )
    return path, expected


def test_rgb(loaders, source, rgb_file):
    path, expected = rgb_file
    result = loaders.rgb(source(path), meta={}, attributes={"frame": 1})
    _assert_loaded(result, expected)


def test_depth_preserves_metric_z_channel(loaders, source, depth_file):
    path, expected = depth_file
    result = loaders.depth(source(path), meta={}, attributes={"frame": 1})
    _assert_loaded(result, expected)


def test_depth_converts_half_to_float32(loaders, tmp_path):
    expected = np.array([[0.25, 2.5, 3.75]], dtype=np.float16)
    path = tmp_path / "half.exr"
    _write_exr(path, expected)
    _assert_loaded(loaders.depth(path), expected.astype(np.float32))


def test_depth_rejects_missing_z_channel(loaders, tmp_path):
    path = tmp_path / "color.exr"
    _write_exr(path, np.ones((2, 3), dtype=np.float32), channel="G")
    with pytest.raises(ValueError, match="'Z' channel"):
        loaders.depth(path)


def test_depth_reports_optional_dependency(loaders, monkeypatch):
    monkeypatch.setitem(sys.modules, "OpenEXR", None)
    with pytest.raises(ImportError, match=r"euler-loading\[synscapes\]"):
        loaders.depth("unused.exr")


def test_depth_does_not_close_caller_stream(loaders, depth_file):
    path, expected = depth_file
    stream = io.BytesIO(path.read_bytes())
    _assert_loaded(loaders.depth(stream), expected)
    assert not stream.closed
    stream.seek(0)
    _assert_loaded(loaders.depth(stream), expected)


@pytest.mark.parametrize("palette", [False, True])
def test_class_ids_and_sky_mask(loaders, source, tmp_path, palette):
    labels = np.array([[0, 7, 23], [10, 33, 255]], dtype=np.uint8)
    image = Image.fromarray(labels)
    if palette:
        image.putpalette([component for value in range(256) for component in (value, 0, 255)])
    path = tmp_path / "class.png"
    image.save(path)
    _assert_loaded(
        loaders.class_segmentation(source(path), meta={}, attributes={"frame": 1}),
        labels.astype(np.int64),
    )
    _assert_loaded(
        loaders.sky_mask(source(path), meta={}, attributes={"frame": 1}),
        np.array([[False, False, True], [False, False, False]]),
    )


def test_class_segmentation_rejects_color_visualization(loaders, rgb_file):
    path, _ = rgb_file
    with pytest.raises(ValueError, match="single-channel PNG"):
        loaders.class_segmentation(path)


def test_instance_ids_decode_all_rgb_bytes(loaders, source, tmp_path):
    pixels = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
         [[0, 0, 1], [1, 2, 3], [255, 255, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "instance.png"
    Image.fromarray(pixels).save(path)
    expected = np.array([[0, 1, 256], [65536, 197121, 16777215]], dtype=np.int64)
    _assert_loaded(
        loaders.instance_segmentation(source(path), meta={}, attributes={"frame": 1}),
        expected,
    )


def test_intrinsics_from_per_image_metadata(loaders, source, intrinsics_file):
    path, expected = intrinsics_file
    result = loaders.read_intrinsics(source(path), meta={}, attributes={"frame": 1})
    _assert_loaded(result, expected, matrix=True)


def test_intrinsics_read_file_values_instead_of_dataset_constants(loaders, intrinsics_file):
    path, _ = intrinsics_file
    data = json.loads(path.read_text())
    data["camera"]["intrinsic"].update(fx=800.0, fy=810.0, u0=700.0, v0=350.0)
    path.write_text(json.dumps(data))
    expected = np.array([[800, 0, 700], [0, 810, 350], [0, 0, 1]], dtype=np.float32)
    _assert_loaded(loaders.read_intrinsics(path), expected, matrix=True)


def test_public_imports_and_dense_depth_contract():
    from euler_loading.loaders import synscapes

    assert resolve_loader_module("synscapes") is gpu_synscapes
    assert isinstance(synscapes, DenseDepthLoader)
    for name in cpu_synscapes.__all__:
        assert getattr(synscapes, name) is getattr(gpu_synscapes, name)


def test_cpu_import_and_non_exr_loading_without_optional_dependencies(rgb_file, intrinsics_file):
    rgb_path, _ = rgb_file
    intrinsics_path, _ = intrinsics_file
    script = """
import sys
sys.modules['torch'] = None
sys.modules['OpenEXR'] = None
sys.modules['Imath'] = None
from euler_loading.loaders.cpu import synscapes
assert synscapes.rgb(sys.argv[1]).shape == (2, 3, 3)
assert synscapes.read_intrinsics(sys.argv[2]).shape == (3, 3)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(rgb_path), str(intrinsics_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_inventory_matches_annotations():
    generated = generate()
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    entries = {item["name"]: item for item in generated["supportedLoaders"]}
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert entries["synscapes"] == saved["synscapes"]
    modalities = {item["function"]: item for item in entries["synscapes"]["modalities"]}
    assert set(modalities) == set(cpu_synscapes.__all__)
    assert modalities["depth"]["meta"]["radial_depth"] is False
    assert modalities["depth"]["output_unit"] == "meters"
    assert modalities["depth"]["file_formats"] == [".exr"]
    assert modalities["read_intrinsics"]["hierarchical"] is False
    for name, cpu_shape, gpu_shape in [
        ("rgb", "HWC", "CHW"),
        ("depth", "HW", "1HW"),
        ("class_segmentation", "HW", "1HW"),
        ("instance_segmentation", "HW", "1HW"),
        ("sky_mask", "HW", "1HW"),
        ("read_intrinsics", "3x3", "3x3"),
    ]:
        assert modalities[name]["cpu"]["shape"] == cpu_shape
        assert modalities[name]["gpu"]["shape"] == gpu_shape
        cpu_meta = dict(getattr(cpu_synscapes, name)._modality_meta)
        gpu_meta = dict(getattr(gpu_synscapes, name)._modality_meta)
        cpu_meta.pop("shape")
        gpu_meta.pop("shape")
        assert cpu_meta == gpu_meta


@pytest.mark.parametrize("zipped", [False, True])
def test_dataset_automatically_loads_aligned_modalities(
    tmp_path, monkeypatch, rgb_file, depth_file, intrinsics_file, zipped
):
    files = {"rgb": rgb_file, "depth": depth_file, "intrinsics": intrinsics_file}
    modalities = {}
    indexes = {}
    for name, (path, _) in files.items():
        root = tmp_path / name
        root.mkdir()
        payload = path.read_bytes()
        (root / path.name).write_bytes(payload)
        if zipped:
            root = root.with_suffix(".zip")
            with zipfile.ZipFile(root, "w") as archive:
                archive.writestr(path.name, payload)
        modalities[name] = Modality(str(root))
        indexes[str(root)] = {
            "euler_loading": {
                "loader": "synscapes",
                "function": "read_intrinsics" if name == "intrinsics" else name,
            },
            "dataset": {"files": [{"id": "1", "path": path.name}]},
        }
    monkeypatch.setattr(
        "euler_loading.dataset.index_dataset_from_path",
        lambda path, **kwargs: indexes[str(path)],
    )
    dataset = MultiModalDataset(modalities=modalities)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["id"] == "1"
    for name, (_, expected) in files.items():
        assert isinstance(sample[name], torch.Tensor)
        _assert_loaded(sample[name], expected, matrix=name == "intrinsics")
