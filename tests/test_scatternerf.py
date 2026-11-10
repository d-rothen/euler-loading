"""ScatterNeRF loaders: NSVF intrinsics and poses, rectified stereo images."""

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
from euler_loading.loaders.cpu import scatternerf as cpu_sn
from euler_loading.loaders.generate.__main__ import generate

# 0/intrinsics/left_06880_1538982135676701005.txt and 0/pose/… as released.
INTRINSICS_TXT = "2355.7228 0.0 852.0 0.0 0.0 2355.7228 454.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0\n"
POSE_TXT = (
    "0.9939147347222974 0.0088638658384036 -0.10979495427524746 1.9384649707059365 "
    "-0.010116535550632024 0.9998898805938611 -0.010857365907410264 0.0015180874975010694 "
    "0.10968662548532385 0.011902040513852147 0.9938949570358273 -5.888391020062318 "
    "0.0 0.0 0.0 1.0\n"
)
K = np.array([[2355.7228, 0.0, 852.0], [0.0, 2355.7228, 454.0], [0.0, 0.0, 1.0]])
POSE = np.array(POSE_TXT.split(), dtype=float).reshape(4, 4)


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.scatternerf")


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


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_rgb(loaders, source, tmp_path):
    path = tmp_path / "left_00001_1.png"
    Image.fromarray(np.full((9, 16, 3), 128, np.uint8)).save(path)
    image = _np(loaders.rgb(source(path)))
    assert image.shape == ((9, 16, 3) if loaders is cpu_sn else (3, 9, 16))
    assert image.dtype == np.float32 and np.allclose(image, 128 / 255)


def test_read_intrinsics(loaders, source, tmp_path):
    path = _write(tmp_path / "k.txt", INTRINSICS_TXT)
    matrix = _np(loaders.read_intrinsics(source(path)))
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, K, rtol=1e-6)


@pytest.mark.parametrize(
    "text, message",
    [("1 2 3\n", "16 finite numbers"), ("2355 1 852 0 0 2355 454 0 0 0 1 0 0 0 0 1\n", "pinhole")],
)
def test_intrinsics_are_validated(tmp_path, text, message):
    path = _write(tmp_path / "k.txt", text)
    with pytest.raises(ValueError, match=message):
        cpu_sn.read_intrinsics(str(path))


def test_camera_pose_in_stored_and_metric_units(loaders, source, tmp_path):
    path = _write(tmp_path / "p.txt", POSE_TXT)
    stored = _np(loaders.camera_pose(source(path)))
    assert stored.dtype == np.float64
    np.testing.assert_allclose(stored, POSE)
    metric = _np(loaders.camera_pose_metric(source(path), attributes={"scale_to_meters": 9.5}))
    np.testing.assert_allclose(metric[:3, :3], POSE[:3, :3])
    np.testing.assert_allclose(metric[:3, 3], POSE[:3, 3] * 9.5)


def test_camera_pose_is_validated(tmp_path):
    path = _write(tmp_path / "p.txt", POSE_TXT)
    with pytest.raises(ValueError, match="scale_to_meters"):
        cpu_sn.camera_pose_metric(str(path))
    sheared = POSE.copy()
    sheared[0, 1] += 0.2
    bad = _write(tmp_path / "bad.txt", " ".join(map(str, sheared.ravel())))
    with pytest.raises(ValueError, match="rigid"):
        cpu_sn.camera_pose(str(bad))


def test_writers_round_trip(loaders, tmp_path):
    loaders.write_intrinsics(str(tmp_path / "k.txt"), K.astype(np.float32))
    np.testing.assert_allclose(cpu_sn.read_intrinsics(str(tmp_path / "k.txt")), K, rtol=1e-6)
    loaders.write_camera_pose(str(tmp_path / "p.txt"), POSE)
    np.testing.assert_array_equal(cpu_sn.camera_pose(str(tmp_path / "p.txt")), POSE)
    image = np.full((4, 5, 3), 0.5, dtype=np.float32)
    loaders.write_rgb(str(tmp_path / "a.png"), image if loaders is cpu_sn else torch.from_numpy(image).permute(2, 0, 1))
    np.testing.assert_allclose(cpu_sn.rgb(str(tmp_path / "a.png")), 0.5, atol=0.01)
    with pytest.raises(ValueError, match="pinhole"):
        cpu_sn.write_intrinsics(str(tmp_path / "x.txt"), np.array([[1.0, 2.0, 3.0], [0, 1, 1], [0, 0, 1]]))


def test_resolution_and_inventory():
    assert resolve_loader_module("scatternerf").__name__ == "euler_loading.loaders.gpu.scatternerf"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["scatternerf"] == saved["scatternerf"]
    functions = {m["function"]: m for m in generated["scatternerf"]["modalities"]}
    assert set(functions) == {"rgb", "read_intrinsics", "camera_pose", "camera_pose_metric"}
    assert functions["camera_pose_metric"]["meta"]["translation_unit"] == "meters"
    assert functions["camera_pose"]["type"] == "camera_extrinsics"


def _scope(root: Path, scope: str, key: str, meta: dict, function: str, folder: str, ext: str):
    directory = root / ".ds_crawler" / scope
    directory.mkdir(parents=True)
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "scatternerf", "name": "scatternerf"},
        "modality": {"key": key, "meta": meta},
        "addons": {"euler_loading": {"version": "1.0", "loader": "scatternerf", "function": function}},
    }
    name = rf"(?P<camera>left|right)_\d+_\d+\.{ext}"
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": {"regex": rf"^\d+/{folder}/(?:left|right)_(?P<frame>\d+)_\d+\.{ext}$", "join_char": "+"},
            "hierarchy": {"regex": rf"^(?P<sequence>\d+)/{folder}/{name}$", "separator": ":"},
        },
        "metadata_scope": scope,
    }
    (directory / "dataset-head.json").write_text(json.dumps(head))
    (directory / "ds-crawler.json").write_text(json.dumps(config))


def test_dataset_loads_end_to_end(tmp_path):
    root = tmp_path / "scatternerf"
    for camera, x in (("left", 0.0), ("right", 0.0338)):
        stem = f"{camera}_00015_1676358516936844849"
        Image.fromarray(np.zeros((9, 16, 3), np.uint8)).save(_write(root / "6/rgb" / f"{stem}.png", "").with_suffix(".png"))
        _write(root / "6/intrinsics" / f"{stem}.txt", INTRINSICS_TXT)
        pose = np.eye(4)
        pose[0, 3] = x
        _write(root / "6/pose" / f"{stem}.txt", " ".join(map(str, pose.ravel())))
    _scope(root, "rgb", "rgb", {"range": [0, 255]}, "rgb", "rgb", "png")
    _scope(root, "intrinsics", "intrinsics", {"camera_model": "pinhole"}, "read_intrinsics", "intrinsics", "txt")
    _scope(root, "camera_pose", "camera_extrinsics",
           {"source_frame": "camera", "target_frame": "world", "direction": "target_from_source"},
           "camera_pose", "pose", "txt")
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), metadata_scope="rgb"),
            "intrinsics": Modality(str(root), metadata_scope="intrinsics"),
            "pose": Modality(str(root), metadata_scope="camera_pose"),
        }
    )
    assert len(dataset) == 2
    left, right = dataset[0], dataset[1]
    assert left["full_id"] == "/sequence:6/camera:left/frame-00015"
    assert left["rgb"].shape == (3, 9, 16)
    np.testing.assert_allclose(left["intrinsics"].numpy(), K, rtol=1e-6)
    baseline = torch.linalg.inv(left["pose"]) @ right["pose"]
    assert float(baseline[0, 3]) == pytest.approx(0.0338)
