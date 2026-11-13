"""UrbanSyn loaders: images, Z-depth EXRs in 1e5 m, train-ID labels, intrinsics from the camera metadata."""

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
from euler_loading.loaders.cpu import urbansyn as cpu_u
from euler_loading.loaders.generate.__main__ import generate

pytest.importorskip("OpenEXR")
import Imath
import OpenEXR

# camera_metadata.json as released.
CAMERA = {
    "parameters": [{
        "Camera": {"focalLength_mm": 4.900001049041748, "fov_deg": 61.237205505371094,
                   "sensorWidth_mm": 5.8000001907348633, "sensorHeight_mm": 2.90000009537},
        "render": {"bitdepth": 8, "colorSpace": "sRGB", "fileType": "png", "min_samples": 128, "max_samples": 256},
    }]
}
FOCAL = 4.900001049041748 / 5.8000001907348633 * 2048


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.urbansyn")


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


# The camera of depth_0001.exr's header, as released (abridged).
OCTANE_CAMERA = {
    "pos": {"x": 92.37972259521484, "y": 1.3622499704360962, "z": 96.88133239746094},
    "target": {"x": 91.62227630615234, "y": 1.3779319524765015, "z": 97.5340576171875},
    "up": {"x": 0.063985541462898254, "y": 0.99668222665786743, "z": 0.050304118543863297},
    "focalLength": 4.900001049041748, "fov": 61.237205505371094, "sensorWidth": 5.8000001907348633,
}


def _exr(path: Path, stored: np.ndarray, channel: str = "Y", camera: dict | None = None) -> Path:
    """An EXR as the release writes it: one float32 channel, uncompressed, the render settings in its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = OpenEXR.Header(stored.shape[1], stored.shape[0])
    header["channels"] = {channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header["compression"] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    if camera is not None:
        header["octane"] = json.dumps({"renderPasses": [{"name": "Z-depth"}], "renderTarget": {"camera": camera}}).encode()
    output = OpenEXR.OutputFile(str(path), header)
    output.writePixels({channel: stored.astype(np.float32).tobytes()})
    output.close()
    return path


def _labels(path: Path, labels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.repeat(labels.astype(np.uint8)[..., None], 3, axis=2)).save(path)
    return path


def test_rgb(loaders, source, tmp_path):
    path = tmp_path / "rgb_0001.png"
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    Image.fromarray(pixels).save(path)
    image = _np(loaders.rgb(source(path)))
    expected = pixels.astype(np.float32) / 255
    if loaders is not cpu_u:
        expected = expected.transpose(2, 0, 1)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image, expected)


def test_depth_is_stored_in_1e5_meters(loaders, source, tmp_path):
    stored = np.array([[3.94e-5, 1.2e-4, 1.0], [np.nan, 0.0, 0.0068]], dtype=np.float32)
    depth = _hw(loaders.depth(source(_exr(tmp_path / "depth_0001.exr", stored))))
    assert depth.dtype == np.float32 and depth.shape == (2, 3)
    # 1.0 is a ray that hit nothing; NaN and 0 are invalid too.
    np.testing.assert_allclose(depth, [[3.94, 12.0, 0.0], [0.0, 0.0, 680.0]], rtol=1e-6)


def test_depth_reads_another_single_channel(tmp_path):
    depth = cpu_u.depth(str(_exr(tmp_path / "d.exr", np.full((2, 2), 1e-4, np.float32), channel="Z")))
    np.testing.assert_allclose(depth, 10.0, rtol=1e-6)


def test_labels_and_sky(loaders, source, tmp_path):
    labels = np.array([[10, 10, 2], [0, 19, 13]])
    path = _labels(tmp_path / "ss_0001.png", labels)
    values = _hw(loaders.semantic_segmentation(source(path)))
    assert values.dtype == np.int64
    np.testing.assert_array_equal(values, labels)
    np.testing.assert_array_equal(_hw(loaders.sky_mask(source(path))), labels == 10)
    np.testing.assert_array_equal(_hw(loaders.sky_mask(str(path), {"skyclass": [2, 2, 2]})), labels == 2)
    # A single-channel label PNG is read as well.
    Image.fromarray(labels.astype(np.uint8)).save(tmp_path / "grey.png")
    np.testing.assert_array_equal(_hw(loaders.semantic_segmentation(str(tmp_path / "grey.png"))), labels)


def test_colour_coded_labels_are_refused(tmp_path):
    path = tmp_path / "ss_color_0001.png"
    Image.fromarray(np.array([[[70, 130, 180], [128, 64, 128]]], np.uint8)).save(path)
    with pytest.raises(ValueError, match="ss_color"):
        cpu_u.semantic_segmentation(str(path))
    with pytest.raises(ValueError, match="grey"):
        cpu_u.sky_mask(str(_labels(tmp_path / "ss.png", np.zeros((2, 2)))), {"skyclass": [70, 130, 180]})


def test_read_intrinsics(loaders, source, tmp_path):
    path = tmp_path / "camera_metadata.json"
    path.write_text(json.dumps(CAMERA, indent=4))
    matrix = _np(loaders.read_intrinsics(source(path)))
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, [[FOCAL, 0, 1023.5], [0, FOCAL, 511.5], [0, 0, 1]], rtol=1e-6)
    assert FOCAL == pytest.approx(1730.207, abs=1e-3)
    # The horizontal field of view the metadata states follows from the same numbers.
    assert np.degrees(2 * np.arctan(1024 / FOCAL)) == pytest.approx(61.237205505371094, abs=1e-4)
    half = _np(loaders.read_intrinsics(str(path), {"image_size": [512, 1024]}))
    np.testing.assert_allclose(half, [[FOCAL / 2, 0, 511.5], [0, FOCAL / 2, 255.5], [0, 0, 1]], rtol=1e-6)


def test_writers_round_trip(loaders, tmp_path):
    depth = np.array([[0.0, 3.94], [np.nan, 680.25]], dtype=np.float32)
    value = depth if loaders is cpu_u else torch.from_numpy(depth)[None]
    loaders.write_depth(str(tmp_path / "out" / "depth_0001.exr"), value)
    np.testing.assert_allclose(cpu_u.depth(str(tmp_path / "out" / "depth_0001.exr")), [[0, 3.94], [0, 680.25]], rtol=1e-6)
    exr = OpenEXR.InputFile(str(tmp_path / "out" / "depth_0001.exr"))
    assert list(exr.header()["channels"]) == ["Y"]
    stored = np.frombuffer(exr.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
    exr.close()
    np.testing.assert_allclose(stored, [1.0, 3.94e-5, 1.0, 680.25e-5], rtol=1e-6)
    with pytest.raises(TypeError, match="path"):
        loaders.write_depth(io.BytesIO(), depth)

    labels = np.array([[0, 10], [19, 18]])
    loaders.write_semantic_segmentation(str(tmp_path / "ss.png"), labels if loaders is cpu_u else torch.from_numpy(labels)[None])
    with Image.open(tmp_path / "ss.png") as image:
        assert image.mode == "RGB"
    np.testing.assert_array_equal(cpu_u.semantic_segmentation(str(tmp_path / "ss.png")), labels)

    mask = labels == 10
    loaders.write_sky_mask(str(tmp_path / "sky.png"), mask)
    np.testing.assert_array_equal(cpu_u.sky_mask(str(tmp_path / "sky.png")), mask)
    np.testing.assert_array_equal(cpu_u.semantic_segmentation(str(tmp_path / "sky.png")), np.where(mask, 10, 19))

    image = np.random.default_rng(0).integers(0, 256, (4, 5, 3)).astype(np.float32) / 255
    loaders.write_rgb(str(tmp_path / "a.png"), image if loaders is cpu_u else torch.from_numpy(image).permute(2, 0, 1))
    np.testing.assert_allclose(cpu_u.rgb(str(tmp_path / "a.png")), image, atol=1e-6)
    with pytest.raises(ValueError, match="PNG"):
        cpu_u.write_rgb(str(tmp_path / "a.jpg"), image)

    camera = tmp_path / "camera_metadata.json"
    camera.write_text(json.dumps(CAMERA))
    K = cpu_u.read_intrinsics(str(camera))
    cropped = K.copy()
    cropped[0, 2] -= 100
    loaders.write_intrinsics(str(tmp_path / "k.json"), cropped)
    np.testing.assert_allclose(cpu_u.read_intrinsics(str(tmp_path / "k.json")), cropped, rtol=1e-6)


def test_camera_pose_from_the_exr_header(loaders, source, tmp_path):
    path = _exr(tmp_path / "depth_0001.exr", np.full((2, 3), 1e-4, np.float32), camera=OCTANE_CAMERA)
    pose = _np(loaders.camera_pose(source(path)))
    assert pose.dtype == np.float64 and pose.shape == (4, 4)
    R, t = pose[:3, :3], pose[:3, 3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(t, [92.37972259521484, 1.3622499704360962, 96.88133239746094])
    # z looks at the target, y points down (against up), x = y cross z.
    to_target = np.array([91.62227630615234 - t[0], 1.3779319524765015 - t[1], 97.5340576171875 - t[2]])
    np.testing.assert_allclose(R[:, 2], to_target / np.linalg.norm(to_target), atol=1e-7)
    assert R[1, 1] < -0.99
    # The depth map of the same file still reads.
    np.testing.assert_allclose(_hw(loaders.depth(str(path))), 10.0, rtol=1e-6)
    with pytest.raises(ValueError, match="octane"):
        cpu_u.camera_pose(str(_exr(tmp_path / "plain.exr", np.ones((2, 2), np.float32))))


def test_camera_pose_writer_round_trips(loaders, tmp_path):
    pose = cpu_u.camera_pose(str(_exr(tmp_path / "d.exr", np.ones((2, 2), np.float32), camera=OCTANE_CAMERA)))
    loaders.write_camera_pose(str(tmp_path / "pose.json"), pose if loaders is cpu_u else torch.from_numpy(pose))
    np.testing.assert_allclose(cpu_u.camera_pose(str(tmp_path / "pose.json")), pose, atol=1e-12)
    mirrored = pose.copy()
    mirrored[:3, 0] *= -1
    with pytest.raises(ValueError, match="right-handed"):
        cpu_u.write_camera_pose(str(tmp_path / "bad.json"), mirrored)


def test_resolution_and_inventory():
    assert resolve_loader_module("urbansyn").__name__ == "euler_loading.loaders.gpu.urbansyn"
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["urbansyn"] == saved["urbansyn"]
    functions = {m["function"]: m for m in generated["urbansyn"]["modalities"]}
    assert set(functions) == {"rgb", "depth", "semantic_segmentation", "sky_mask", "read_intrinsics", "camera_pose"}
    assert functions["camera_pose"]["meta"]["direction"] == "target_from_source"
    assert functions["depth"]["meta"]["radial_depth"] is False
    assert functions["depth"]["meta"]["scale_to_meters"] == 100000.0
    assert functions["semantic_segmentation"]["meta"]["skyclass"] == [10, 10, 10]
    assert functions["read_intrinsics"]["hierarchical"] is True


def _scope(root: Path, scope: str, key: str, meta: dict, function: str, files: list[tuple[str, str]]):
    """A prebuilt scope as euler-known-datasets writes it: frames below scene:<name>."""
    head = build_dataset_head(
        dataset={"id": "urbansyn", "name": "urbansyn"},
        modality={"key": key, "meta": meta},
        addons={"euler_loading": {"version": "1.0", "loader": "urbansyn", "function": function}},
    )
    entries = [{"path": path, "id": file_id, "path_properties": {}, "basename_properties": {}}
               for file_id, path in files]
    index = {"files": entries} if scope == "intrinsics" else {"children": {"scene:poblenou": {"files": entries}}}
    save_dataset_index(root, build_index_output(head=head, index=index), metadata_scope=scope)


def test_dataset_loads_end_to_end(tmp_path):
    root = tmp_path / "urbansyn"
    frames = ["0001", "0002"]
    for i, frame in enumerate(frames):
        (root / "rgb").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(root / "rgb" / f"rgb_{frame}.png")
        _exr(root / "depth" / f"depth_{frame}.exr", np.full((4, 6), (5 + i) / 1e5, np.float32))
        _labels(root / "ss" / f"ss_{frame}.png", np.full((4, 6), 10))
    (root / "camera_metadata.json").write_text(json.dumps(CAMERA))
    grid = {"height": 4, "width": 6}
    _scope(root, "rgb", "rgb", {"range": [0, 255]}, "rgb", [(f, f"rgb/rgb_{f}.png") for f in frames])
    _scope(root, "depth", "depth", {"radial_depth": False, "scale_to_meters": 1e5, "range": [0, 2000], "dimensions": grid},
           "depth", [(f, f"depth/depth_{f}.exr") for f in frames])
    _scope(root, "semantic_segmentation", "semantic_segmentation", {"skyclass": [10, 10, 10]},
           "semantic_segmentation", [(f, f"ss/ss_{f}.png") for f in frames])
    _scope(root, "intrinsics", "intrinsics", {"camera_model": "pinhole"}, "read_intrinsics",
           [("intrinsics", "camera_metadata.json")])
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), metadata_scope="rgb"),
            "depth": Modality(str(root), metadata_scope="depth"),
            "labels": Modality(str(root), metadata_scope="semantic_segmentation"),
            "sky": Modality(str(root), metadata_scope="semantic_segmentation", function="sky_mask"),
        },
        hierarchical_modalities={
            "intrinsics": Modality(str(root), metadata_scope="intrinsics", collapse_single=True),
        },
    )
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["full_id"] == "/scene:poblenou/0002"
    assert sample["rgb"].shape == (3, 4, 6)
    assert sample["depth"].shape == (1, 4, 6) and float(sample["depth"][0, 0, 0]) == pytest.approx(6.0)
    assert sample["labels"].dtype == torch.int64 and bool(sample["sky"].all())
    assert float(sample["intrinsics"][0, 0]) == pytest.approx(FOCAL, rel=1e-6)
