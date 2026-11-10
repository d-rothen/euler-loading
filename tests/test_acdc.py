"""ACDC loaders: adverse-condition and reference images."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import resolve_loader_module
from euler_loading.loaders.cpu import acdc as cpu_a
from euler_loading.loaders.generate.__main__ import generate


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.acdc")


@pytest.fixture(params=["path", "stream"])
def source(request):
    def make(path: Path):
        if request.param == "path":
            return str(path)
        stream = io.BytesIO(path.read_bytes())
        stream.name = str(path)
        return stream

    return make


def _np(value):
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


@pytest.mark.parametrize("function", ["rgb", "reference_rgb"])
def test_images(loaders, source, tmp_path, function):
    path = tmp_path / "GOPR0475_frame_000041_rgb_anon.png"
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    Image.fromarray(pixels).save(path)
    image = _np(getattr(loaders, function)(source(path)))
    expected = pixels.astype(np.float32) / 255
    if loaders is not cpu_a:
        expected = expected.transpose(2, 0, 1)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image, expected)


def test_writers_round_trip(loaders, tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (4, 5, 3)).astype(np.float32) / 255
    value = image if loaders is cpu_a else torch.from_numpy(image).permute(2, 0, 1)
    loaders.write_rgb(str(tmp_path / "a.png"), value)
    loaders.write_reference_rgb(str(tmp_path / "b.png"), value)
    np.testing.assert_allclose(cpu_a.rgb(str(tmp_path / "a.png")), image, atol=1e-6)
    np.testing.assert_allclose(cpu_a.reference_rgb(str(tmp_path / "b.png")), image, atol=1e-6)
    with pytest.raises(ValueError, match="PNG"):
        cpu_a.write_rgb(str(tmp_path / "a.jpg"), image)


def test_resolution_and_inventory():
    assert resolve_loader_module("acdc").__name__ == "euler_loading.loaders.gpu.acdc"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["acdc"] == saved["acdc"]
    functions = {m["function"]: m for m in generated["acdc"]["modalities"]}
    assert set(functions) == {"rgb", "reference_rgb"}
    assert functions["reference_rgb"]["meta"]["source"] == "reference_frames"
