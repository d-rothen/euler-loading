"""SHIFT loaders: images, CARLA 24-bit planar depth, red-channel labels, calibration files with poses."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ds_crawler import build_dataset_head, build_index_output, save_dataset_index
from PIL import Image

from euler_loading import Modality, MultiModalDataset, resolve_loader_module
from euler_loading.loaders.cpu import shift as cpu_s
from euler_loading.loaders.generate.__main__ import generate

K = [[640.0, 0.0, 640.0], [0.0, 640.0, 400.0], [0.0, 0.0, 1.0]]


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.shift")


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


def _depth_png(path: Path, code: np.ndarray) -> None:
    code = code.astype(np.uint32)
    Image.fromarray(np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], -1).astype(np.uint8)).save(path)


def _labels_png(path: Path, labels: np.ndarray) -> None:
    rgb = np.zeros(labels.shape + (3,), np.uint8)
    rgb[..., 0] = labels
    Image.fromarray(rgb).save(path)


def _calibration(path: Path, frames: dict, intrinsics=K) -> None:
    path.write_text(json.dumps({
        "format": {"name": "euler_known_datasets.shift.camera_frames", "version": 1},
        "sequence": "0016-1b62", "split": "train",
        "camera": {"name": "front", "width": 1280, "height": 800, "intrinsics": intrinsics},
        "frames": {f: {"camera_to_world": pose.tolist()} for f, pose in frames.items()},
    }))


def test_rgb_png_and_jpeg(loaders, source, tmp_path):
    image = np.zeros((4, 6, 3), np.uint8)
    image[1, 2] = (255, 128, 0)
    Image.fromarray(image).save(tmp_path / "a.png")
    out = _np(loaders.rgb(source(tmp_path / "a.png")))
    hwc = out.transpose(1, 2, 0) if out.shape[0] == 3 else out
    assert hwc.dtype == np.float32 and hwc[1, 2].tolist() == [1.0, pytest.approx(128 / 255), 0.0]
    Image.fromarray(image).save(tmp_path / "a.jpg", quality=95)
    assert _np(loaders.rgb(source(tmp_path / "a.jpg"))).shape in ((4, 6, 3), (3, 4, 6))


def test_depth_decodes_carlas_code(loaders, source, tmp_path):
    code = np.array([[0, 1, 16777215], [16777210, 2**16 + 2**8 + 1, 167772]])
    _depth_png(tmp_path / "d.png", code)
    depth = _hw(loaders.depth(source(tmp_path / "d.png")))
    assert depth.dtype == np.float32
    assert depth[0, 2] == 1000.0 and depth[1, 0] == pytest.approx(999.9997, abs=1e-4)
    assert depth[1, 1] == pytest.approx(65793 * 1000 / 16777215, rel=1e-6)
    assert depth[1, 2] == pytest.approx(10.0, abs=1e-3)


def test_radial_distance(loaders, tmp_path):
    code = np.full((800, 1280), 167772)  # 10 m everywhere
    _depth_png(tmp_path / "d.png", code)
    distance = _hw(loaders.radial_distance(str(tmp_path / "d.png")))
    planar = _hw(loaders.depth(str(tmp_path / "d.png")))
    assert distance[400, 640] == pytest.approx(planar[400, 640])
    assert distance[0, 0] == pytest.approx(planar[0, 0] * np.sqrt(1 + 1 + (400 / 640) ** 2), rel=1e-5)
    # Another grid needs its camera matrix.
    _depth_png(tmp_path / "small.png", np.full((4, 6), 167772))
    with pytest.raises(ValueError, match="intrinsics"):
        loaders.radial_distance(str(tmp_path / "small.png"))
    small = _hw(loaders.radial_distance(str(tmp_path / "small.png"),
                                        attributes={"intrinsics": [[3, 0, 3], [0, 3, 2], [0, 0, 1]]}))
    assert small[2, 3] == pytest.approx(10.0, abs=1e-3) and small[0, 0] > small[2, 3]


def test_labels_and_sky(loaders, source, tmp_path):
    labels = np.array([[13, 13, 26], [10, 0, 31]])
    _labels_png(tmp_path / "s.png", labels)
    out = _hw(loaders.semantic_segmentation(source(tmp_path / "s.png")))
    assert out.dtype == np.int64 and out.tolist() == labels.tolist()
    sky = _hw(loaders.sky_mask(source(tmp_path / "s.png")))
    assert sky.tolist() == [[True, True, False], [False, False, False]]
    other = _hw(loaders.sky_mask(source(tmp_path / "s.png"), {"skyclass": [26, 0, 0]}))
    assert other.tolist() == [[False, False, True], [False, False, False]]
    with pytest.raises(ValueError, match=r"\[id, 0, 0\]"):
        loaders.sky_mask(str(tmp_path / "s.png"), {"skyclass": [13, 13, 13]})


def test_calibration_file(loaders, source, tmp_path):
    pose = np.eye(4)
    pose[:3, 3] = (1, -2, 1.6)
    _calibration(tmp_path / "c.json", {"00000040": pose})
    assert _np(loaders.read_intrinsics(source(tmp_path / "c.json"))).tolist() == K
    table = loaders.camera_pose(source(tmp_path / "c.json"))
    assert table.camera_to_world("00000040").tolist() == pose.tolist()
    with pytest.raises(KeyError, match="00000050"):
        table.camera_to_world("00000050")
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError, match="not a SHIFT calibration file"):
        loaders.read_intrinsics(str(tmp_path / "bad.json"))


def test_writers_round_trip(loaders, tmp_path):
    rgb = np.random.default_rng(0).random((4, 6, 3)).astype(np.float32)
    loaders.write_rgb(str(tmp_path / "r.png"), rgb)
    assert np.abs(_np(cpu_s.rgb(str(tmp_path / "r.png"))) - rgb).max() <= 0.5 / 255 + 1e-6
    loaders.write_rgb(str(tmp_path / "r.jpg"), rgb)
    assert Image.open(tmp_path / "r.jpg").format == "JPEG"

    depth = np.array([[0.5, 10.0, 999.0], [1000.0, np.inf, np.nan]], np.float32)
    loaders.write_depth(str(tmp_path / "d.png"), depth)
    back = cpu_s.depth(str(tmp_path / "d.png"))
    assert np.allclose(back[0], depth[0], atol=1e-4) and back[1].tolist() == [1000.0, 1000.0, 0.0]

    distance = np.full((800, 1280), 20.0, np.float32)
    loaders.write_radial_distance(str(tmp_path / "rd.png"), distance)
    assert np.allclose(cpu_s.radial_distance(str(tmp_path / "rd.png")), 20.0, atol=1e-3)

    labels = np.array([[13, 0], [26, 255]])
    loaders.write_semantic_segmentation(str(tmp_path / "s.png"), labels)
    assert cpu_s.semantic_segmentation(str(tmp_path / "s.png")).tolist() == labels.tolist()
    assert np.asarray(Image.open(tmp_path / "s.png"))[..., 1:].max() == 0

    mask = np.array([[True, False], [False, True]])
    loaders.write_sky_mask(str(tmp_path / "m.png"), mask)
    assert cpu_s.sky_mask(str(tmp_path / "m.png")).tolist() == mask.tolist()

    stream = io.BytesIO()
    stream.name = "x.png"
    loaders.write_depth(stream, depth)
    stream.seek(0)
    assert np.allclose(cpu_s.depth(stream)[0], depth[0], atol=1e-4)


def test_write_intrinsics_updates_crops_only(tmp_path):
    pose = np.eye(4)
    _calibration(tmp_path / "c.json", {"00000000": pose})
    before = (tmp_path / "c.json").read_bytes()
    cpu_s.write_intrinsics(str(tmp_path / "c.json"), np.array(K))
    assert (tmp_path / "c.json").read_bytes() == before
    halved = np.array(K) * [[0.5], [0.5], [1]]
    cpu_s.write_intrinsics(str(tmp_path / "c.json"), halved, {"width": 640, "height": 400})
    document = json.loads((tmp_path / "c.json").read_text())
    assert document["camera"]["intrinsics"][0] == [320.0, 0.0, 320.0] and document["camera"]["width"] == 640
    assert document["frames"]["00000000"]["camera_to_world"] == pose.tolist()
    with pytest.raises(ValueError, match="crop or rescale"):
        cpu_s.write_intrinsics(str(tmp_path / "c.json"), np.array([[320.0, 40, 320], [0, 320, 200], [0, 0, 1]]))
    cpu_s.write_intrinsics(str(tmp_path / "new.json"), np.array(K))
    assert cpu_s.read_intrinsics(str(tmp_path / "new.json")).tolist() == K


def test_resolution_and_inventory():
    assert resolve_loader_module("shift").__name__ == "euler_loading.loaders.gpu.shift"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["shift"] == saved["shift"]
    functions = {m["function"]: m for m in generated["shift"]["modalities"]}
    assert set(functions) == {"rgb", "depth", "radial_distance", "semantic_segmentation", "sky_mask",
                              "read_intrinsics", "camera_pose"}
    assert functions["depth"]["meta"]["radial_depth"] is False
    assert functions["radial_distance"]["meta"]["radial_depth"] is True
    assert functions["sky_mask"]["meta"]["skyclass"] == [13, 0, 0]
    assert functions["camera_pose"]["hierarchical"] is True


def _scope(root: Path, scope: str, key: str, meta: dict, function: str, files: list[tuple[str, str]]):
    """A prebuilt scope as euler-known-datasets writes it: frames below the sequence's hierarchy."""
    head = build_dataset_head(
        dataset={"id": "shift", "name": "shift"},
        modality={"key": key, "meta": meta},
        addons={"euler_loading": {"version": "1.0", "loader": "shift", "function": function}},
    )
    entries = [{"path": path, "id": file_id, "path_properties": {}, "basename_properties": {}}
               for file_id, path in files]
    node: dict = {"files": entries}
    for key_ in reversed(("split:train", "weather:clear", "time_of_day:daytime", "sequence:0016-1b62")):
        node = {"children": {key_: node}}
    save_dataset_index(root, build_index_output(head=head, index=node), metadata_scope=scope)


def test_dataset_loads_end_to_end(tmp_path):
    root = tmp_path / "shift"
    frames = ["00000000", "00000040"]
    folder = root / "img"
    folder.mkdir(parents=True)
    poses = {}
    for i, frame in enumerate(frames):
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(folder / f"{frame}_img.png")
        _depth_png(folder / f"{frame}_depth.png", np.full((4, 6), 167772 * (i + 1)))
        _labels_png(folder / f"{frame}_semseg.png", np.full((4, 6), 13))
        pose = np.eye(4)
        pose[0, 3] = 4.0 * i
        poses[frame] = pose
    _calibration(root / "calibration.json", poses)
    _scope(root, "rgb", "rgb", {"range": [0, 255]}, "rgb", [(f, f"img/{f}_img.png") for f in frames])
    _scope(root, "depth", "depth", {"radial_depth": False, "scale_to_meters": 1000 / (2**24 - 1),
                                    "range": [0, 1000]}, "depth", [(f, f"img/{f}_depth.png") for f in frames])
    _scope(root, "semantic_segmentation", "semantic_segmentation", {"skyclass": [13, 0, 0]},
           "semantic_segmentation", [(f, f"img/{f}_semseg.png") for f in frames])
    head = build_dataset_head(dataset={"id": "shift", "name": "shift"},
                              modality={"key": "camera_extrinsics", "meta": {}},
                              addons={"euler_loading": {"version": "1.0", "loader": "shift", "function": "camera_pose"}})
    index = {"children": {"split:train": {"children": {"weather:clear": {"children": {"time_of_day:daytime": {
        "children": {"sequence:0016-1b62": {"files": [{"path": "calibration.json", "id": "camera_pose",
                                                        "path_properties": {}, "basename_properties": {}}]}}}}}}}}}
    save_dataset_index(root, build_index_output(head=head, index=index), metadata_scope="camera_pose")
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), metadata_scope="rgb"),
            "depth": Modality(str(root), metadata_scope="depth"),
            "sky": Modality(str(root), metadata_scope="semantic_segmentation", function="sky_mask"),
        },
        hierarchical_modalities={"pose": Modality(str(root), metadata_scope="camera_pose")},
    )
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["full_id"] == "/split:train/weather:clear/time_of_day:daytime/sequence:0016-1b62/00000040"
    assert sample["rgb"].shape == (3, 4, 6)
    assert float(sample["depth"][0, 0, 0]) == pytest.approx(20.0, abs=1e-3) and bool(sample["sky"].all())
    pose = next(iter(sample["pose"].values()))
    assert isinstance(pose, torch.Tensor) and float(pose[0, 3]) == 4.0
