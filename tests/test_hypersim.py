"""Hypersim loaders: tone-mapped HDR colour, distance-to-depth conversion, calibration files."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from euler_loading import resolve_loader_module  # noqa: E402
from euler_loading.loaders._annotations import SampleContext, get_sample_selector  # noqa: E402
from euler_loading.loaders.cpu import hypersim as cpu_h  # noqa: E402
from euler_loading.loaders.generate.__main__ import generate  # noqa: E402

#: A tilt-shift-like camera: off-centre principal point and some skew.
K = np.array([[600.0, -4.0, 9.0], [0.0, 610.0, 2.5], [0.0, 0.0, 1.0]])
H, W = 6, 8


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.hypersim")


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


def _hdf5(path: Path, values) -> Path:
    """A file as Hypersim releases it: the array in ``dataset``, float16."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("dataset", data=np.asarray(values, dtype=np.float16), compression="gzip")
    return path


def _ray_lengths() -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.linalg.inv(K) @ np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)])
    return np.linalg.norm(rays / rays[2], axis=0).reshape(H, W)


def test_tonemap_follows_hypersims_operator():
    color = np.random.default_rng(0).uniform(0, 3, (H, W, 3)).astype(np.float32)
    brightness = 0.3 * color[..., 0] + 0.59 * color[..., 1] + 0.11 * color[..., 2]
    valid = np.ones((H, W), bool)
    valid[0] = False
    scale = 0.8 ** 2.2 / np.percentile(brightness[valid], 90)
    expected = np.clip((scale * color) ** (1 / 2.2), 0, 1)
    np.testing.assert_allclose(cpu_h.tonemap(color, valid=valid), expected, rtol=1e-5)
    np.testing.assert_allclose(cpu_h.tonemap(color, scale), expected, rtol=1e-5)
    # The 90th-percentile brightness lands at 0.8 after gamma.
    mapped = cpu_h.tonemap(color, valid=valid)
    assert np.percentile((0.3 * mapped[..., 0] ** 2.2 + 0.59 * mapped[..., 1] ** 2.2
                          + 0.11 * mapped[..., 2] ** 2.2)[valid], 90) == pytest.approx(0.8 ** 2.2, rel=0.05)
    assert not cpu_h.tonemap(np.zeros((H, W, 3), np.float32)).any()
    nan = color.copy()
    nan[0, 0] = np.nan
    assert np.isfinite(cpu_h.tonemap(nan)).all()


def test_rgb_uses_the_files_scale(loaders, source, tmp_path):
    color = np.full((H, W, 3), 0.5, np.float32)
    path = _hdf5(tmp_path / "frame.0000.color.hdf5", color)
    image = _np(loaders.rgb(source(path), attributes={"tonemap_scale": 2.0}))
    expected = np.full((H, W, 3), 1.0, np.float32)
    assert image.dtype == np.float32
    assert image.shape == ((H, W, 3) if loaders is cpu_h else (3, H, W))
    np.testing.assert_allclose(image, expected if loaders is cpu_h else expected.transpose(2, 0, 1))
    darker = _np(loaders.rgb(source(path), attributes={"tonemap_scale": 0.5}))
    np.testing.assert_allclose(darker, (0.25 ** (1 / 2.2)) * np.ones_like(darker), rtol=1e-3)


def test_depth_converts_distances_to_planar_depth(loaders, source, tmp_path):
    planar = np.linspace(1.0, 9.0, H * W).reshape(H, W)
    distances = planar * _ray_lengths()
    distances[0, 0] = np.nan  # no geometry
    path = _hdf5(tmp_path / "frame.0000.depth_meters.hdf5", distances)
    attributes = {"intrinsics": K.tolist()}
    depth = _np(loaders.depth(source(path), attributes=attributes))
    expected = planar.copy()
    expected[0, 0] = 0
    assert depth.dtype == np.float32
    assert depth.shape == ((H, W) if loaders is cpu_h else (1, H, W))
    np.testing.assert_allclose(depth.reshape(H, W), expected, rtol=2e-3)
    distance = _np(loaders.radial_distance(source(path))).reshape(H, W)
    np.testing.assert_allclose(distance, np.nan_to_num(distances.astype(np.float16).astype(np.float32)), rtol=1e-6)
    assert distance[0, 0] == 0
    with pytest.raises(ValueError, match="radial_distance"):
        loaders.depth(source(path))


def test_writers_round_trip(loaders, tmp_path):
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 1, (H, W, 3)).astype(np.float32)
    planar = rng.uniform(1, 9, (H, W)).astype(np.float32)
    value = image if loaders is cpu_h else torch.from_numpy(image).permute(2, 0, 1)
    loaders.write_rgb(str(tmp_path / "frame.0000.color.hdf5"), value)
    # Written images are not tone-mapped again, whatever scale the entry carries.
    np.testing.assert_array_equal(cpu_h.rgb(str(tmp_path / "frame.0000.color.hdf5"),
                                            attributes={"tonemap_scale": 3.0}), image)

    depth_value = planar if loaders is cpu_h else torch.from_numpy(planar).unsqueeze(0)
    loaders.write_depth(str(tmp_path / "frame.0000.depth_meters.hdf5"), depth_value)
    attributes = {"intrinsics": K.tolist()}
    np.testing.assert_array_equal(cpu_h.depth(str(tmp_path / "frame.0000.depth_meters.hdf5"),
                                              attributes=attributes), planar)
    np.testing.assert_allclose(cpu_h.radial_distance(str(tmp_path / "frame.0000.depth_meters.hdf5"),
                                              attributes=attributes), planar * _ray_lengths(), rtol=1e-6)

    buffer = io.BytesIO()
    loaders.write_radial_distance(buffer, planar)
    buffer.seek(0)
    np.testing.assert_array_equal(cpu_h.radial_distance(buffer), planar)


def _calibration(path: Path) -> Path:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    document = {
        "format": {"name": cpu_h.FORMAT_NAME, "version": 1},
        "scene": "ai_001_001",
        "camera": {"name": "cam_00", "width": W, "height": H, "intrinsics": K.tolist()},
        "frames": {"0000": {"camera_to_world": pose.tolist()}, "0004": {"camera_to_world": np.eye(4).tolist()}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def test_calibration_file(loaders, source, tmp_path):
    path = _calibration(tmp_path / "calibration" / "ai_001_001" / "cam_00.json")
    matrix = _np(loaders.read_intrinsics(source(path)))
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, K)

    table = loaders.camera_pose(source(path))
    select = get_sample_selector(loaders.camera_pose)
    context = SampleContext(modality="pose", entry={"path": str(path), "id": "camera_pose"}, id="0000",
                            full_id="/scene:ai_001_001/camera:cam_00/0000",
                            hierarchy_path=("scene:ai_001_001", "camera:cam_00"), files={})
    pose = _np(select(table, context))
    assert pose.shape == (4, 4) and pose.dtype == np.float32
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
    with pytest.raises(KeyError, match="0001"):
        select(table, SampleContext(**{**context.__dict__, "id": "0001"}))
    with pytest.raises(ValueError, match="not a Hypersim calibration"):
        (tmp_path / "other.json").write_text("{}")
        cpu_h.read_intrinsics(str(tmp_path / "other.json"))


def test_write_intrinsics_updates_crops_only(tmp_path):
    path = _calibration(tmp_path / "cam_00.json")
    before = path.read_bytes()
    cpu_h.write_intrinsics(str(path), K)
    assert path.read_bytes() == before
    cropped = K.copy()
    cropped[0, 2] -= 2
    cropped[1, 2] -= 1
    cpu_h.write_intrinsics(str(path), cropped, {"width": W - 2, "height": H - 1})
    document = json.loads(path.read_text())
    np.testing.assert_allclose(document["camera"]["intrinsics"], cropped)
    assert (document["camera"]["width"], document["camera"]["height"]) == (W - 2, H - 1)
    assert set(document["frames"]) == {"0000", "0004"}
    with pytest.raises(ValueError, match="not a crop or rescale"):
        cpu_h.write_intrinsics(str(path), np.eye(3))


def test_resolution_and_inventory():
    assert resolve_loader_module("hypersim").__name__ == "euler_loading.loaders.gpu.hypersim"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["hypersim"] == saved["hypersim"]
    functions = {m["function"]: m for m in generated["hypersim"]["modalities"]}
    assert set(functions) == {"rgb", "depth", "radial_distance", "read_intrinsics", "camera_pose"}
    assert functions["depth"]["meta"]["radial_depth"] is False
    assert functions["radial_distance"]["meta"]["radial_depth"] is True
