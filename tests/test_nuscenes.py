"""nuScenes loaders for datasets prepared by euler-known-datasets."""

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
from euler_loading.loaders._annotations import SampleContext, get_sample_selector
from euler_loading.loaders.cpu import nuscenes as cpu_nus
from euler_loading.loaders.generate.__main__ import generate

K = [[1257.86, 0.0, 827.24], [0.0, 1257.86, 450.92], [0.0, 0.0, 1.0]]
# A real CAM_FRONT_LEFT frame of v1.0-test (scene-0830), lidar -> camera through
# both ego poses, rounded.
T = [
    [0.5734, 0.819, 0.0218, 0.1714],
    [0.0039, 0.0238, -0.9997, -0.3331],
    [-0.8193, 0.5733, 0.0105, -0.4885],
    [0.0, 0.0, 0.0, 1.0],
]
FRAMES = ("1538450717104844", "1538450717154844", "1538450717204844")


def _rigid(yaw: float, translation) -> list[list[float]]:
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = translation
    return m.tolist()


def calibration_document(frames=FRAMES, *, without_lidar=(FRAMES[2],)) -> dict:
    records = {}
    for i, frame in enumerate(frames):
        records[frame] = {
            "sample_data_token": f"cam{i}",
            "sample_token": f"sample{i}",
            "is_key_frame": i == 0,
            "path": f"sweeps/CAM_FRONT_LEFT/x__CAM_FRONT_LEFT__{frame}.jpg",
            "camera_to_global": _rigid(0.1 * i, [100.0 + i, 50.0, 1.5]),
            "lidar": None
            if frame in without_lidar
            else {
                "sample_data_token": f"lidar{i}",
                "timestamp": int(frame) + 43_000,
                "offset_ms": 43.0,
                "path": f"sweeps/LIDAR_TOP/x__LIDAR_TOP__{int(frame) + 43_000}.pcd.bin",
                "lidar_to_camera": T,
            },
        }
    return {
        "format": {"name": "euler_known_datasets.nuscenes.camera_frames", "version": 1},
        "generator": {"name": "euler-known-datasets", "version": "0.1.0"},
        "dataset_version": "v1.0-test",
        "scene": {"name": "scene-0830", "token": "t", "log": "n015", "location": "x", "vehicle": "n015"},
        "conventions": {},
        "camera": {
            "channel": "CAM_FRONT_LEFT",
            "calibrated_sensor_token": "c",
            "width": 1600,
            "height": 900,
            "intrinsics": K,
            "camera_to_ego": _rigid(0.9, [1.5, 0.5, 1.5]),
        },
        "lidar": {
            "channel": "LIDAR_TOP",
            "calibrated_sensor_token": "l",
            "lidar_to_ego": _rigid(-1.57, [0.98, 0.0, 1.84]),
            "nominal_offset_ms": 43.0,
            "sync_tolerance_ms": 20.0,
        },
        "frames": records,
    }


def _write_calibration(path: Path, document=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document or calibration_document(), separators=(",", ":")))
    return path


def _context(sample_id: str) -> SampleContext:
    return SampleContext(
        modality="extrinsics",
        entry={"id": "extrinsics", "path": "calibration.json"},
        id=sample_id,
        full_id=f"/scene:scene-0830/camera:CAM_FRONT_LEFT/{sample_id}",
        hierarchy_path=("scene:scene-0830", "camera:CAM_FRONT_LEFT"),
        files={},
    )


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.nuscenes")


@pytest.fixture(params=["path", "stream"])
def source(request):
    def make(path: Path):
        if request.param == "path":
            return str(path)
        stream = io.BytesIO(path.read_bytes())
        stream.name = str(path)
        return stream

    return make


def _as_numpy(value):
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def test_rgb(loaders, source, tmp_path):
    pixels = np.zeros((9, 16, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    path = tmp_path / "frame.jpg"
    Image.fromarray(pixels).save(path, quality=100)
    image = _as_numpy(loaders.rgb(source(path)))
    expected = (9, 16, 3) if loaders is cpu_nus else (3, 9, 16)
    assert image.shape == expected and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_lidar_round_trip(loaders, source, tmp_path):
    points = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / "sweep.pcd.bin"
    loaders.write_lidar_point_cloud(str(path), points)
    assert path.stat().st_size == 4 * 5 * 4
    for reader in (loaders.lidar_point_cloud, loaders.point_cloud, loaders.sparse_depth):
        loaded = _as_numpy(reader(source(path)))
        assert loaded.dtype == np.float32
        np.testing.assert_array_equal(loaded, points)


def test_lidar_rejects_truncated_files(tmp_path):
    path = tmp_path / "sweep.pcd.bin"
    path.write_bytes(b"\x00" * 21)
    with pytest.raises(ValueError, match="multiple of 20"):
        cpu_nus.lidar_point_cloud(str(path))
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        cpu_nus.write_lidar_point_cloud(str(path), np.zeros((3, 4)))


def test_read_intrinsics(loaders, source, tmp_path):
    path = _write_calibration(tmp_path / "CAM_FRONT_LEFT.json")
    matrix = _as_numpy(loaders.read_intrinsics(source(path)))
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, K, rtol=1e-6)


@pytest.mark.parametrize(
    "file_format, message",
    [({"name": "something_else", "version": 1}, "not a nuScenes calibration"),
     ({"name": cpu_nus.FORMAT_NAME, "version": 2}, "format version 2")],
)
def test_unknown_calibration_formats_are_refused(tmp_path, file_format, message):
    document = calibration_document()
    document["format"] = file_format
    path = _write_calibration(tmp_path / "c.json", document)
    with pytest.raises(ValueError, match=message):
        cpu_nus.read_intrinsics(str(path))


def test_read_extrinsics_is_selected_per_sample(loaders, source, tmp_path):
    path = _write_calibration(tmp_path / "CAM_FRONT_LEFT.json")
    table = loaders.read_extrinsics(source(path))
    assert isinstance(table, cpu_nus.CameraFrames)
    assert table.scene == "scene-0830" and table.camera == "CAM_FRONT_LEFT"
    select = get_sample_selector(loaders.read_extrinsics)
    transform = _as_numpy(select(table, _context(FRAMES[0])))
    assert transform.dtype == np.float32
    np.testing.assert_allclose(transform, T, rtol=1e-6)
    np.testing.assert_allclose(table.camera_to_global(FRAMES[1]), _rigid(0.1, [101.0, 50.0, 1.5]))

    with pytest.raises(KeyError, match="no synchronised lidar sweep"):
        select(table, _context(FRAMES[2]))
    with pytest.raises(KeyError, match="camera timestamp"):
        select(table, _context("frame-1538450717104844"))


def test_camera_to_lidar_direction(tmp_path):
    path = _write_calibration(tmp_path / "c.json")
    table = cpu_nus.read_extrinsics(str(path), {"transform_direction": "camera_to_lidar"})
    inverse = get_sample_selector(cpu_nus.read_extrinsics)(table, _context(FRAMES[0]))
    np.testing.assert_allclose(inverse @ np.asarray(T), np.eye(4), atol=1e-4)
    with pytest.raises(ValueError, match="transform_direction"):
        cpu_nus.read_extrinsics(str(path), attributes={"transform_direction": "sideways"})


def test_write_intrinsics_applies_a_crop_and_keeps_the_frames(tmp_path):
    path = _write_calibration(tmp_path / "c.json")
    before = json.loads(path.read_text())
    cropped = np.array(K)
    cropped[0, 2] -= 100  # crop 100 px from the left
    cropped[1, 2] -= 50
    cpu_nus.write_intrinsics(str(path), cropped.astype(np.float32), {"width": 1400, "height": 800})
    after = json.loads(path.read_text())
    np.testing.assert_allclose(after["camera"]["intrinsics"], cropped)
    # float32 rounding of the input is snapped back to exact whole-pixel shifts
    assert after["camera"]["intrinsics"][0][2] == K[0][2] - 100
    assert (after["camera"]["width"], after["camera"]["height"]) == (1400, 800)
    assert after["frames"] == before["frames"]
    assert after["camera"]["camera_to_ego"] == before["camera"]["camera_to_ego"]


def test_write_intrinsics_leaves_an_unchanged_file_alone(tmp_path):
    path = _write_calibration(tmp_path / "c.json")
    original = path.read_bytes()
    cpu_nus.write_intrinsics(str(path), np.array(K), {"width": 1600, "height": 900})
    assert path.read_bytes() == original


def test_write_intrinsics_refuses_a_rotation(tmp_path):
    path = _write_calibration(tmp_path / "c.json")
    rotated = np.array(K)
    rotated[0, 1] = 10.0
    with pytest.raises(ValueError, match="crop or rescale"):
        cpu_nus.write_intrinsics(str(path), rotated)


def test_write_intrinsics_new_file_reads_back(loaders, tmp_path):
    path = tmp_path / "new" / "c.json"
    loaders.write_intrinsics(str(path), np.array(K, dtype=np.float32), {"width": 1600, "height": 900})
    np.testing.assert_allclose(_as_numpy(loaders.read_intrinsics(str(path))), K, rtol=1e-6)


def test_write_rgb_round_trip(loaders, tmp_path):
    image = np.full((9, 16, 3), 0.5, dtype=np.float32)
    for name in ("a.jpg", "a.png"):
        loaders.write_rgb(str(tmp_path / name), image if loaders is cpu_nus else torch.from_numpy(image).permute(2, 0, 1))
        loaded = cpu_nus.rgb(str(tmp_path / name))
        np.testing.assert_allclose(loaded, 0.5, atol=0.01)


def test_resolution():
    assert resolve_loader_module("nuscenes").__name__ == "euler_loading.loaders.gpu.nuscenes"
    assert resolve_loader_module("nuscenes", variant="cpu").__name__ == "euler_loading.loaders.cpu.nuscenes"


def test_inventory_matches_annotations():
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["nuscenes"] == saved["nuscenes"]
    modalities = {item["function"]: item for item in generated["nuscenes"]["modalities"]}
    assert set(modalities) == {
        "rgb", "lidar_point_cloud", "sparse_depth", "read_intrinsics", "read_extrinsics",
    }
    assert modalities["read_extrinsics"]["hierarchical"] is True
    assert modalities["lidar_point_cloud"]["meta"]["columns"] == ["x", "y", "z", "intensity", "ring"]


# -- a prepared dataset, loaded with automatic loader resolution ---------------


def _scope(root: Path, scope: str, key: str, meta: dict, function: str, id_regex: str, hierarchy_regex: str, override=None):
    directory = root / ".ds_crawler" / scope
    directory.mkdir(parents=True)
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": f"nuscenes_{scope}", "name": f"nuScenes {scope}"},
        "modality": {"key": key, "meta": meta},
        "addons": {"euler_loading": {"version": "1.0", "loader": "nuscenes", "function": function}},
    }
    id_config = {"regex": id_regex, "join_char": "+"}
    if override:
        id_config["override"] = override
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": id_config,
            "hierarchy": {"regex": hierarchy_regex, "separator": ":"},
        },
        "metadata_scope": scope,
    }
    (directory / "dataset-head.json").write_text(json.dumps(head))
    (directory / "ds-crawler.json").write_text(json.dumps(config))


def test_prepared_dataset_loads_end_to_end(tmp_path):
    root = tmp_path / "nuscenes"
    unit = root / "scene-0830" / "CAM_FRONT_LEFT"
    unit.mkdir(parents=True)
    for frame in FRAMES:
        Image.fromarray(np.zeros((9, 16, 3), dtype=np.uint8)).save(unit / f"{frame}.jpg")
    for frame in FRAMES[:2]:  # the third frame has no synchronised sweep
        cpu_nus.write_lidar_point_cloud(str(unit / f"{frame}.pcd.bin"), np.ones((7, 5), np.float32))
    _write_calibration(root / "scene-0830" / "CAM_FRONT_LEFT.json")

    hierarchy = r"^(?P<scene>[^/]+)/(?P<camera>CAM_[A-Z_]+)"
    _scope(root, "rgb", "rgb", {"range": [0, 255]}, "rgb", r"^[^/]+/[^/]+/(\d+)\.jpg$", hierarchy + "/")
    _scope(root, "lidar", "lidar_point_cloud", {}, "lidar_point_cloud",
           r"^[^/]+/[^/]+/(\d+)\.pcd\.bin$", hierarchy + "/")
    for scope, key, function, override in (
        ("intrinsics", "intrinsics", "read_intrinsics", "intrinsics"),
        ("camera_extrinsics", "camera_extrinsics", "read_extrinsics", "extrinsics"),
    ):
        _scope(root, scope, key, {}, function, r"^[^/]+/(CAM_[A-Z_]+)\.json$",
               r"^(?P<scene>[^/]+)/(?P<camera>CAM_[A-Z_]+)\.json$", override)

    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), metadata_scope="rgb"),
            "lidar": Modality(str(root), metadata_scope="lidar"),
        },
        hierarchical_modalities={
            "intrinsics": Modality(str(root), metadata_scope="intrinsics", collapse_single=True),
            "extrinsics": Modality(str(root), metadata_scope="camera_extrinsics", collapse_single=True),
        },
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["full_id"] == f"/scene:scene-0830/camera:CAM_FRONT_LEFT/{FRAMES[0]}"
    assert sample["rgb"].shape == (3, 9, 16)
    assert sample["lidar"].shape == (7, 5)
    np.testing.assert_allclose(sample["intrinsics"].numpy(), K, rtol=1e-6)
    np.testing.assert_allclose(sample["extrinsics"].numpy(), T, rtol=1e-6)
