"""CEREMA Pixel-Accurate Depth Benchmark loaders: depth maps, gated slices, method outputs."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import Modality, MultiModalDataset, resolve_loader_module
from euler_loading.loaders.cpu import cerema as cpu_c
from euler_loading.loaders.generate.__main__ import generate

BF = 0.202993 * 2355.722801
# calibration/calib_cam_stereo_left.json as released (abridged: fields the loader reads and keeps).
CALIB = {
    "roi": {"width": 0, "do_rectify": False, "y_offset": 0, "x_offset": 0, "height": 0},
    "D": [-0.567575, 0.0, 0.0, 0.0, 0.0],
    "P": [2355.722801, 0.0, 988.138054, 0.0, 0.0, 2355.722801, 508.051838, 0.0, 0.0, 0.0, 1.0, 0.0],
    "width": 1920,
    "K": [2612.86, 0.0, 966.525, 0.0, 2612.86, 508.297, 0.0, 0.0, 1.0],
    "height": 1024,
    "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "distortion_model": "plumb_bob",
}
P = np.array(CALIB["P"]).reshape(3, 4)[:, :3]


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.cerema")


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


def _hw(value):
    array = _np(value)
    return array[0] if array.ndim == 3 and array.shape[0] == 1 else array


def _npz(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        np.savez_compressed(stream, array)  # stored as arr_0, as released
    return path


def test_rgb(loaders, source, tmp_path):
    path = tmp_path / "scene1_day_clear_0.png"
    Image.fromarray(np.full((6, 8, 3), 51, np.uint8)).save(path)
    image = _np(loaders.rgb(source(path)))
    assert image.shape == ((6, 8, 3) if loaders is cpu_c else (3, 6, 8))
    assert image.dtype == np.float32 and np.allclose(image, 0.2)


def test_gated_removes_the_dark_level(loaders, source, tmp_path):
    raw = np.array([[0, 87, 87 + 1023], [600, 1177, 87 + 512]], dtype=np.uint16)
    path = tmp_path / "g.png"
    Image.fromarray(raw).save(path)
    values = _hw(loaders.gated(source(path)))
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, [[0, 0, 1], [513 / 1023, 1, 512 / 1023]], atol=1e-6)
    shifted = _hw(loaders.gated(source(path), attributes={"dark_level": 0}))
    assert shifted[0, 2] == pytest.approx(1.0) and shifted[1, 0] == pytest.approx(600 / 1023)


def test_depth(loaders, source, tmp_path):
    values = np.array([[0.0, 2.5, np.nan], [-1.0, 30.25, np.inf]], dtype=np.float32)
    depth = _hw(loaders.depth(source(_npz(tmp_path / "d.npz", values))))
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth, [[0, 2.5, 0], [0, 30.25, 0]])


def test_sgm_depth_is_placed_on_its_rows(loaders, source, tmp_path):
    disparity = np.full((412, 960), 20.0, np.float32)
    disparity[200, 480] = np.nan
    depth = _hw(loaders.sgm_depth(source(_npz(tmp_path / "sgm.npz", disparity))))
    assert depth.shape == (1024, 1920)
    assert (depth[:70] == 0).all() and (depth[894:] == 0).all()
    assert depth[70, 0] == pytest.approx(BF / 40) and depth[893, 1919] == pytest.approx(BF / 40)
    # A NaN invalidates the output pixels it contributes to, and no others.
    hole = depth[70 + 398:70 + 404, 958:964]
    assert (hole[1:5, 1:5] == 0).all()
    assert depth[70 + 396, 956] == pytest.approx(BF / 40)


def test_psmnet_and_monodepth(loaders, source, tmp_path):
    psm = _hw(loaders.psmnet_depth(source(_npz(tmp_path / "p.npz", np.full((512, 960), 25.0, np.float32)))))
    assert psm.shape == (1024, 1920) and np.allclose(psm, BF / 50)
    mono = _hw(loaders.monodepth_depth(source(_npz(tmp_path / "m.npz", np.full((256, 512), 0.02)))))
    assert mono.shape == (1024, 1920) and np.allclose(mono, BF / (0.02 * 1920))


def test_upsampling_is_pixel_centre_bilinear():
    ramp = np.arange(1, 5, dtype=np.float64)[None, :].repeat(2, axis=0)
    values, valid = cpu_c._upsample(ramp, (4, 8))
    assert valid.all()
    # Interior output pixels sample the ramp at (x + 0.5) / 2 - 0.5; the borders clamp.
    np.testing.assert_allclose(values[0], [1, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4])


@pytest.mark.parametrize("function, shape", [("sgm_depth", (412, 960)), ("psmnet_depth", (512, 960)), ("monodepth_depth", (256, 512))])
def test_method_outputs_must_have_their_released_shape(tmp_path, function, shape):
    path = _npz(tmp_path / "x.npz", np.ones((shape[0] + 1, shape[1]), np.float32))
    with pytest.raises(ValueError, match=r"expected \(" + str(shape[0])):
        getattr(cpu_c, function)(str(path))


def test_read_intrinsics(loaders, source, tmp_path):
    path = tmp_path / "calib_cam_stereo_left.json"
    path.write_text(json.dumps(CALIB))
    matrix = _np(loaders.read_intrinsics(source(path)))
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, P, rtol=1e-6)


def test_writers_round_trip(loaders, tmp_path):
    depth = np.array([[0.0, 3.5], [np.nan, 12.0]], dtype=np.float32)
    loaders.write_depth(str(tmp_path / "out" / "d.npz"), depth)
    assert (tmp_path / "out" / "d.npz").is_file() and not (tmp_path / "out" / "d.npz.npz").exists()
    np.testing.assert_array_equal(cpu_c.depth(str(tmp_path / "out" / "d.npz")), [[0, 3.5], [0, 12]])
    with np.load(tmp_path / "out" / "d.npz") as data:
        assert data.files == ["arr_0"]

    gated = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    loaders.write_gated(str(tmp_path / "g.png"), gated if loaders is cpu_c else torch.from_numpy(gated)[None])
    np.testing.assert_allclose(cpu_c.gated(str(tmp_path / "g.png")), gated, atol=1 / 1023)

    image = np.full((4, 5, 3), 0.5, dtype=np.float32)
    loaders.write_rgb(str(tmp_path / "a.png"), image if loaders is cpu_c else torch.from_numpy(image).permute(2, 0, 1))
    np.testing.assert_allclose(cpu_c.rgb(str(tmp_path / "a.png")), 0.5, atol=0.01)

    loaders.write_intrinsics(str(tmp_path / "k.json"), P.astype(np.float32))
    np.testing.assert_allclose(cpu_c.read_intrinsics(str(tmp_path / "k.json")), P, rtol=1e-6)


def test_resolution_and_inventory():
    assert resolve_loader_module("cerema").__name__ == "euler_loading.loaders.gpu.cerema"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["cerema"] == saved["cerema"]
    functions = {m["function"]: m for m in generated["cerema"]["modalities"]}
    assert set(functions) == {"rgb", "gated", "depth", "sgm_depth", "psmnet_depth", "monodepth_depth", "read_intrinsics"}
    assert functions["read_intrinsics"]["hierarchical"] is True
    assert functions["sgm_depth"]["meta"]["baseline_m"] == pytest.approx(0.202993)


def _scope(root: Path, scope: str, key: str, meta: dict, function: str, folder: str, ext: str):
    directory = root / ".ds_crawler" / scope
    directory.mkdir(parents=True)
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "cerema", "name": "cerema"},
        "modality": {"key": key, "meta": meta},
        "addons": {"euler_loading": {"version": "1.0", "loader": "cerema", "function": function}},
    }
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": {"regex": rf"^{folder}/scene\d_[a-z]+_[a-z0-9]+_(?P<frame>\d+)\.{ext}$", "join_char": "+"},
            "hierarchy": {
                "regex": rf"^{folder}/scene(?P<scene>\d)_(?P<time_of_day>day|night)_(?P<weather>[a-z0-9]+)_\d+\.{ext}$",
                "separator": ":",
            },
        },
        "metadata_scope": scope,
    }
    (directory / "dataset-head.json").write_text(json.dumps(head))
    (directory / "ds-crawler.json").write_text(json.dumps(config))


def test_dataset_loads_end_to_end(tmp_path):
    root = tmp_path / "cerema"
    for frame in range(2):
        stem = f"scene2_night_fog40_{frame}"
        (root / "rgb_left_8bit").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((6, 8, 3), np.uint8)).save(root / "rgb_left_8bit" / f"{stem}.png")
        _npz(root / "lidar_hdl64_rgb_left" / f"{stem}.npz", np.eye(6, 8, dtype=np.float32) * (5 + frame))
        _npz(root / "psmnet" / f"{stem}.npz", np.full((512, 960), 10.0 + frame, np.float32))
    depth_meta = {"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 100]}
    _scope(root, "rgb", "rgb", {"range": [0, 255]}, "rgb", "rgb_left_8bit", "png")
    _scope(root, "lidar", "depth", {**depth_meta, "sparse": True}, "depth", "lidar_hdl64_rgb_left", "npz")
    _scope(root, "psmnet", "depth", depth_meta, "psmnet_depth", "psmnet", "npz")
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), metadata_scope="rgb"),
            "lidar": Modality(str(root), metadata_scope="lidar"),
            "psmnet": Modality(str(root), metadata_scope="psmnet"),
        }
    )
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["full_id"].startswith("/scene:2/time_of_day:night/weather:fog40/")
    assert sample["lidar"].shape == (1, 6, 8) and float(sample["lidar"].max()) == 6.0
    assert sample["psmnet"].shape == (1, 1024, 1920)
    assert float(sample["psmnet"][0, 0, 0]) == pytest.approx(BF / 22)
