"""Synscapes format decoding and dataset integration tests."""

from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path
import stat
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
    # Writers are public but carry no modality annotation, so the inventory
    # covers exactly the reader half of the public surface.
    assert set(modalities) == {
        name for name in cpu_synscapes.__all__ if not name.startswith("write_")
    }
    assert modalities["depth"]["meta"]["radial_depth"] is False
    assert modalities["depth"]["output_unit"] == "meters"
    assert modalities["depth"]["file_formats"] == [".exr"]
    assert modalities["read_intrinsics"]["hierarchical"] is False
    assert modalities["read_extrinsics"]["hierarchical"] is False
    assert modalities["read_extrinsics"]["meta"]["source"] == "camera.extrinsic"
    for name, cpu_shape, gpu_shape in [
        ("rgb", "HWC", "CHW"),
        ("depth", "HW", "1HW"),
        ("class_segmentation", "HW", "1HW"),
        ("instance_segmentation", "HW", "1HW"),
        ("sky_mask", "HW", "1HW"),
        ("read_intrinsics", "3x3", "3x3"),
        ("read_extrinsics", "4x4", "4x4"),
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


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------


# The pose Synscapes ships, which is constant across the released dataset.
_EXTRINSIC = {"pitch": 0.038, "roll": -0.0, "x": 1.7, "y": 0.1, "yaw": -0.0195, "z": 1.22}


def _meta_file(tmp_path, *, extrinsic=None, intrinsic=None, name="meta.json"):
    camera = {}
    if intrinsic is not None:
        camera["intrinsic"] = intrinsic
    if extrinsic is not None:
        camera["extrinsic"] = extrinsic
    path = tmp_path / name
    path.write_text(json.dumps({"camera": camera}), encoding="utf-8")
    return path


def _as_array(value):
    return value.numpy() if isinstance(value, torch.Tensor) else value


def _assert_matrix_close(result, expected):
    """Compare 4x4 transforms without demanding bit-exactness.

    A matrix written back out is decomposed into Euler angles and recomposed,
    so it returns numerically equal rather than identical.
    """
    np.testing.assert_allclose(
        _as_array(result).astype(np.float64),
        _as_array(expected).astype(np.float64),
        atol=1e-6,
    )


def test_extrinsics_builds_documented_pose(loaders, source, tmp_path):
    path = _meta_file(tmp_path, extrinsic=_EXTRINSIC)
    result = loaders.read_extrinsics(source(path), meta={}, attributes={"frame": 1})

    if loaders is gpu_synscapes:
        assert isinstance(result, torch.Tensor) and result.dtype == torch.float32
        assert result.is_contiguous()
    else:
        assert isinstance(result, np.ndarray) and result.dtype == np.float32
    T = _as_array(result)

    assert T.shape == (4, 4)
    # Default direction is the camera's pose on the vehicle, so the
    # translation column is the recorded mount position verbatim.
    np.testing.assert_allclose(T[:3, 3], [1.7, 0.1, 1.22], atol=1e-6)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-6)
    # A pose is a rigid transform: R is orthonormal and right-handed.
    R = T[:3, :3].astype(np.float64)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("axes", ["vehicle", "optical"])
def test_extrinsics_directions_are_exact_inverses(loaders, tmp_path, axes):
    path = _meta_file(tmp_path, extrinsic=_EXTRINSIC)
    forward = _as_array(
        loaders.read_extrinsics(path, {"transform_direction": "camera_to_ego", "camera_axes": axes})
    ).astype(np.float64)
    backward = _as_array(
        loaders.read_extrinsics(path, {"transform_direction": "ego_to_camera", "camera_axes": axes})
    ).astype(np.float64)
    np.testing.assert_allclose(forward @ backward, np.eye(4), atol=1e-5)


def test_extrinsics_optical_axes_project_ego_points_through_intrinsics(loaders, tmp_path):
    """A point straight ahead must land on the principal point.

    This pins the two conventions that are not spelled out by the dataset: the
    ego frame being x forward / y left / z up, and the optical frame being
    x right / y down / z forward. With the mount rotation set to identity the
    expected pixel is exact, so a wrong axis permutation cannot pass.
    """
    mount = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "x": 1.7, "y": 0.1, "z": 1.22}
    path = _meta_file(tmp_path, extrinsic=mount)
    T = _as_array(
        loaders.read_extrinsics(
            path, {"transform_direction": "ego_to_camera", "camera_axes": "optical"}
        )
    ).astype(np.float64)

    def to_camera(point):
        return (T @ np.array([*point, 1.0]))[:3]

    # 20 m directly in front of the camera.
    ahead = to_camera([1.7 + 20.0, 0.1, 1.22])
    np.testing.assert_allclose(ahead, [0.0, 0.0, 20.0], atol=1e-5)

    # 5 m to the vehicle's left is to the left of the optical axis (-x),
    # and 5 m up is above it (-y), matching image coordinates growing
    # rightwards and downwards.
    assert to_camera([1.7 + 20.0, 0.1 + 5.0, 1.22])[0] == pytest.approx(-5.0, abs=1e-5)
    assert to_camera([1.7 + 20.0, 0.1, 1.22 + 5.0])[1] == pytest.approx(-5.0, abs=1e-5)


def test_extrinsics_accepts_attributes_over_meta_and_plural_key(loaders, tmp_path):
    path = tmp_path / "plural.json"
    path.write_text(json.dumps({"camera": {"extrinsics": _EXTRINSIC}}), encoding="utf-8")
    pose = _as_array(loaders.read_extrinsics(path))
    inverse = _as_array(
        loaders.read_extrinsics(
            path,
            {"transform_direction": "camera_to_ego"},
            attributes={"transform_direction": "ego_to_camera"},
        )
    ).astype(np.float64)
    np.testing.assert_allclose(pose.astype(np.float64) @ inverse, np.eye(4), atol=1e-5)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"camera": {}}, "extrinsic"),
        ({"camera": {"extrinsic": dict(_EXTRINSIC, yaw=None)}}, "'yaw'"),
        ({"nocamera": {}}, "'camera'"),
    ],
)
def test_extrinsics_rejects_incomplete_metadata(loaders, tmp_path, payload, match):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        loaders.read_extrinsics(path)


def test_extrinsics_rejects_unknown_convention(loaders, tmp_path):
    path = _meta_file(tmp_path, extrinsic=_EXTRINSIC)
    with pytest.raises(ValueError, match="transform_direction"):
        loaders.read_extrinsics(path, {"transform_direction": "camera_to_world"})
    with pytest.raises(ValueError, match="camera_axes"):
        loaders.read_extrinsics(path, {"camera_axes": "opengl"})


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _to_native(loaders, array, layout):
    """Return *array* in the layout the module under test produces.

    Writers accept both NumPy arrays and tensors, so each variant is fed the
    shape its own readers hand back.
    """
    if loaders is cpu_synscapes:
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if layout == "hwc":
        return tensor.permute(2, 0, 1).contiguous()
    if layout == "hw":
        return tensor.unsqueeze(0)
    return tensor


def test_writers_satisfy_the_dense_depth_codec_contract():
    from euler_loading.loaders import synscapes
    from euler_loading.loaders.contracts import DenseDepthCodec, DenseDepthWriter

    assert isinstance(synscapes, DenseDepthWriter)
    assert isinstance(synscapes, DenseDepthCodec)
    # Writers are shared, so the two entry points stay in step by construction.
    for name in (n for n in cpu_synscapes.__all__ if n.startswith("write_")):
        assert getattr(gpu_synscapes, name) is getattr(cpu_synscapes, name)


def test_writer_module_resolves_for_every_reader():
    from euler_loading import resolve_writer_module
    from euler_loading._resolution import _writer_name_candidates

    module = resolve_writer_module("synscapes")
    for reader in (n for n in cpu_synscapes.__all__ if not n.startswith("write_")):
        candidates = _writer_name_candidates(reader, None)
        assert any(callable(getattr(module, name, None)) for name in candidates), reader


def test_write_rgb_roundtrip(loaders, tmp_path):
    # 2x4 rather than square, so (H, W, 3) and (3, H, W) stay distinguishable.
    expected = (
        np.array(
            [[[0, 128, 255], [255, 64, 0], [32, 16, 8], [7, 9, 11]],
             [[255, 255, 255], [0, 0, 0], [10, 20, 30], [40, 50, 60]]],
            dtype=np.uint8,
        ).astype(np.float32)
        / 255.0
    )
    path = tmp_path / "rgb.png"
    loaders.write_rgb(str(path), _to_native(loaders, expected, "hwc"))
    _assert_loaded(loaders.rgb(str(path)), expected)


def test_write_depth_roundtrip_preserves_non_finite(loaders, tmp_path):
    expected = np.array([[0.0, 1.25, 5000.5], [1e10, np.inf, np.nan]], dtype=np.float32)
    path = tmp_path / "depth.exr"
    loaders.write_depth(str(path), _to_native(loaders, expected, "hw"))
    _assert_loaded(loaders.depth(str(path)), expected)


def test_write_depth_requires_a_filesystem_path(loaders):
    with pytest.raises(TypeError, match="binary stream"):
        loaders.write_depth(io.BytesIO(), np.zeros((2, 2), dtype=np.float32))


@pytest.mark.parametrize("labels", [[[0, 7, 23], [10, 33, 255]], [[0, 300], [1000, 65535]]])
def test_write_class_segmentation_roundtrip(loaders, tmp_path, labels):
    expected = np.array(labels, dtype=np.int64)
    path = tmp_path / "class.png"
    loaders.write_class_segmentation(str(path), _to_native(loaders, expected, "hw"))
    _assert_loaded(loaders.class_segmentation(str(path)), expected)


def test_write_instance_segmentation_roundtrip(loaders, tmp_path):
    expected = np.array([[0, 1, 256], [65536, 197121, 16777215]], dtype=np.int64)
    path = tmp_path / "instance.png"
    loaders.write_instance_segmentation(str(path), _to_native(loaders, expected, "hw"))
    _assert_loaded(loaders.instance_segmentation(str(path)), expected)


def test_write_instance_segmentation_rejects_ids_beyond_24_bits(loaders, tmp_path):
    ids = np.array([[0, 16777216]], dtype=np.int64)
    with pytest.raises(ValueError, match="24 bits"):
        loaders.write_instance_segmentation(str(tmp_path / "bad.png"), ids)


def test_write_sky_mask_roundtrips_as_a_class_image(loaders, tmp_path):
    expected = np.array([[False, False, True], [True, False, False]])
    path = tmp_path / "sky.png"
    loaders.write_sky_mask(str(path), _to_native(loaders, expected, "hw"))
    _assert_loaded(loaders.sky_mask(str(path)), expected)
    # The file is a real class image, so sky keeps its Cityscapes ID.
    _assert_loaded(
        loaders.class_segmentation(str(path)),
        np.where(expected, 23, 0).astype(np.int64),
    )


def test_write_intrinsics_roundtrip(loaders, tmp_path, intrinsics_file):
    _, expected = intrinsics_file
    path = tmp_path / "out" / "meta.json"
    loaders.write_intrinsics(str(path), _to_native(loaders, expected, "matrix"),
                             {"resx": 1440, "resy": 720})
    _assert_loaded(loaders.read_intrinsics(str(path)), expected, matrix=True)
    stored = json.loads(path.read_text())["camera"]["intrinsic"]
    assert stored["resx"] == 1440 and stored["resy"] == 720


@pytest.mark.parametrize("direction", ["camera_to_ego", "ego_to_camera"])
@pytest.mark.parametrize("axes", ["vehicle", "optical"])
def test_write_extrinsics_roundtrip(loaders, tmp_path, direction, axes):
    source_meta = _meta_file(tmp_path, extrinsic=_EXTRINSIC)
    options = {"transform_direction": direction, "camera_axes": axes}
    original = loaders.read_extrinsics(source_meta, options)

    path = tmp_path / "written.json"
    loaders.write_extrinsics(str(path), original, options)

    _assert_matrix_close(loaders.read_extrinsics(path, options), original)
    # The six documented scalars come back, not a flattened matrix.
    stored = json.loads(path.read_text())["camera"]["extrinsic"]
    assert set(stored) == {"pitch", "roll", "yaw", "x", "y", "z"}
    if (direction, axes) == ("camera_to_ego", "vehicle"):
        for key, value in _EXTRINSIC.items():
            assert stored[key] == pytest.approx(value, abs=1e-6)


def test_writing_both_camera_blocks_keeps_one_metadata_file(loaders, tmp_path, intrinsics_file):
    _, K = intrinsics_file
    path = tmp_path / "meta.json"
    pose = loaders.read_extrinsics(_meta_file(tmp_path, extrinsic=_EXTRINSIC, name="src.json"))

    loaders.write_intrinsics(str(path), _to_native(loaders, K, "matrix"))
    loaders.write_extrinsics(str(path), pose)

    # Writing the second block must not drop the first.
    assert set(json.loads(path.read_text())["camera"]) == {"intrinsic", "extrinsic"}
    _assert_loaded(loaders.read_intrinsics(path), K, matrix=True)
    _assert_matrix_close(loaders.read_extrinsics(path), pose)


@pytest.mark.parametrize(
    "writer, value, basename",
    [
        ("write_rgb", np.zeros((2, 2, 3), dtype=np.float32), "rgb.png"),
        ("write_class_segmentation", np.zeros((2, 2), dtype=np.int64), "class.png"),
        ("write_instance_segmentation", np.zeros((2, 2), dtype=np.int64), "instance.png"),
        ("write_sky_mask", np.zeros((2, 2), dtype=bool), "sky.png"),
        ("write_intrinsics", np.eye(3, dtype=np.float32), "meta.json"),
        ("write_extrinsics", np.eye(4, dtype=np.float32), "meta.json"),
    ],
)
def test_writers_accept_stream_targets(loaders, writer, value, basename):
    from euler_loading.loaders._writer_utils import supports_stream_target

    function = getattr(loaders, writer)
    assert supports_stream_target(function)
    stream = io.BytesIO()
    stream.name = basename
    function(stream, value)
    assert stream.getvalue()


@pytest.mark.parametrize(
    "function, writer",
    [
        ("rgb", "write_rgb"),
        ("depth", "write_depth"),
        ("class_segmentation", "write_class_segmentation"),
        ("instance_segmentation", "write_instance_segmentation"),
        ("sky_mask", "write_sky_mask"),
        ("read_intrinsics", "write_intrinsics"),
        ("read_extrinsics", "write_extrinsics"),
    ],
)
def test_dataset_auto_resolves_synscapes_writers(monkeypatch, function, writer):
    """The contract-driven path must find a writer for every reader."""
    index = {
        "name": function,
        "type": function,
        "euler_loading": {"loader": "synscapes", "function": function},
        "dataset": {"files": [{"id": "1", "path": "scene/1.bin"}]},
    }
    monkeypatch.setattr(
        "euler_loading.dataset.index_dataset_from_path",
        lambda path, **kwargs: index,
    )
    dataset = MultiModalDataset(modalities={"modality": Modality("/data/modality")})

    assert dataset._resolved_loaders["modality"] is getattr(gpu_synscapes, function)
    assert dataset.get_writer("modality") is getattr(gpu_synscapes, writer)


def test_depth_writes_into_a_zip_dataset_through_a_temp_file(tmp_path, monkeypatch):
    """EXR writing is not stream-capable, so zip output must use a temp file.

    This exercises the whole path: read the source EXR, write the result into a
    zip-backed output dataset, and decode it again from the archive.
    """
    source = tmp_path / "src"
    (source / "scene").mkdir(parents=True)
    values = np.array([[1.5, 2.5], [np.inf, np.nan]], dtype=np.float32)
    cpu_synscapes.write_depth(str(source / "scene" / "f001.exr"), values)

    index = {
        "name": "depth",
        "type": "depth",
        "euler_train": {"used_as": "input", "modality_type": "depth"},
        "euler_loading": {"loader": "synscapes", "function": "depth"},
        "head": {
            "contract": {"kind": "dataset_head", "version": "1.0"},
            "dataset": {"id": "synscapes_depth", "name": "Synscapes depth"},
            "modality": {
                "key": "depth",
                "meta": {"radial_depth": False, "range": [0, 1000], "scale_to_meters": 1.0},
            },
            "addons": {
                "euler_loading": {"version": "1.0", "loader": "synscapes", "function": "depth"}
            },
        },
        "dataset": {
            "files": [
                {
                    "id": "f001",
                    "path": "scene/f001.exr",
                    "path_properties": {},
                    "basename_properties": {},
                }
            ]
        },
    }
    monkeypatch.setattr(
        "euler_loading.dataset.index_dataset_from_path",
        lambda path, **kwargs: index,
    )
    dataset = MultiModalDataset(modalities={"depth": Modality(str(source))})

    archive = tmp_path / "out.zip"
    output_writer = dataset.create_output_writer("depth", archive, zip=True)
    dataset.write_sample(0, {"depth": dataset[0]["depth"] * 2.0}, output_writer)
    output_writer.save_index()

    with zipfile.ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".exr"))
        buffer = io.BytesIO(bundle.read(name))
        buffer.name = name
        decoded = cpu_synscapes.depth(buffer)

    np.testing.assert_allclose(decoded, values * 2.0, equal_nan=True)


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1e-7, 1e-5, 1e-4, 4.47e-3, 1e-2])
@pytest.mark.parametrize("sign", [1, -1])
def test_extrinsics_roundtrip_near_vertical_keeps_yaw(loaders, tmp_path, offset, sign):
    """A near-vertical pose must not have its yaw folded into roll.

    Guarding the degenerate branch with a relative tolerance made it fire for
    any pitch within 0.26 deg of vertical, which silently zeroed yaw and cost
    up to half a degree of rotation.
    """
    extrinsic = {
        "pitch": sign * (np.pi / 2 - offset),
        "roll": -2.0944,
        "yaw": -3.14159,
        "x": 1.7,
        "y": 0.1,
        "z": 1.22,
    }
    source_meta = _meta_file(tmp_path, extrinsic=extrinsic)
    original = loaders.read_extrinsics(source_meta)

    written = tmp_path / "written.json"
    loaders.write_extrinsics(str(written), original)
    _assert_matrix_close(loaders.read_extrinsics(written), original)

    stored = json.loads(written.read_text())["camera"]["extrinsic"]
    if offset > 1e-7:
        # Still separable, so the recorded angles survive rather than collapsing.
        assert stored["yaw"] == pytest.approx(extrinsic["yaw"], abs=1e-5)


def test_sky_mask_class_id_override_roundtrips(loaders, tmp_path):
    """The reader must honour the class ID the writer accepts."""
    expected = np.array([[True, False], [False, True]])
    options = {"sky_class_id": 30}
    path = tmp_path / "sky.png"
    loaders.write_sky_mask(str(path), _to_native(loaders, expected, "hw"), options)

    _assert_loaded(loaders.sky_mask(str(path), options), expected)
    _assert_loaded(
        loaders.sky_mask(str(path), {}, attributes=options), expected
    )
    # The default ID is a different label, so it must not match.
    assert not _as_array(loaders.sky_mask(str(path))).any()


def test_write_intrinsics_preserves_the_rest_of_the_metadata_file(loaders, tmp_path):
    """Rewriting one camera block must not drop fields it does not model."""
    path = tmp_path / "1.json"
    path.write_text(
        json.dumps(
            {
                "camera": {
                    "intrinsic": {
                        "fx": 1590.83, "fy": 1592.79, "u0": 771.31, "v0": 360.79,
                        "resx": 1440, "resy": 720,
                    },
                    "extrinsic": _EXTRINSIC,
                },
                "scene": {"ego_speed": 2.0},
                "instance": {"class": {"1": 26}},
            }
        ),
        encoding="utf-8",
    )
    loaders.write_intrinsics(str(path), loaders.read_intrinsics(path))

    document = json.loads(path.read_text())
    assert document["scene"] == {"ego_speed": 2.0}
    assert document["instance"] == {"class": {"1": 26}}
    assert document["camera"]["intrinsic"]["resx"] == 1440
    assert document["camera"]["intrinsic"]["resy"] == 720
    assert set(document["camera"]) == {"intrinsic", "extrinsic"}


@pytest.mark.parametrize("content", ["[1, 2, 3]", '{"camera": {"intri', '"text"'])
def test_camera_writers_refuse_to_clobber_unreadable_metadata(loaders, tmp_path, content):
    path = tmp_path / "existing.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        loaders.write_extrinsics(str(path), np.eye(4, dtype=np.float32))
    assert path.read_text() == content


def test_write_depth_rejects_an_empty_image(loaders, tmp_path):
    """OpenEXR aborts the process on a zero-sized window, so catch it first."""
    with pytest.raises(ValueError, match="empty image"):
        loaders.write_depth(str(tmp_path / "empty.exr"), np.zeros((0, 0), dtype=np.float32))


@pytest.mark.parametrize(
    "matrix, match",
    [
        (np.diag([2.0, 2.0, 2.0, 1.0]), "orthonormal"),
        (np.array([[1, .5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float), "orthonormal"),
        (np.diag([1.0, 1.0, -1.0, 1.0]), "right-handed"),
        (np.vstack([np.eye(4)[:3], [9.0, 9.0, 9.0, 9.0]]), "bottom row"),
        (np.where(np.eye(4) > 0, np.nan, 0.0), "finite"),
    ],
)
def test_write_extrinsics_rejects_non_rigid_transforms(loaders, tmp_path, matrix, match):
    """Six Euler scalars cannot carry a scale, shear or mirror."""
    with pytest.raises(ValueError, match=match):
        loaders.write_extrinsics(str(tmp_path / "bad.json"), matrix.astype(np.float32))


def test_camera_merge_keeps_fields_written_under_the_plural_spelling(loaders, tmp_path):
    """The merge must find the block the readers accept, not just the singular."""
    path = tmp_path / "plural.json"
    path.write_text(
        json.dumps(
            {"camera": {"intrinsics": {
                "fx": 1.0, "fy": 2.0, "u0": 3.0, "v0": 4.0, "resx": 1440, "resy": 720}}}
        ),
        encoding="utf-8",
    )
    loaders.write_intrinsics(
        str(path), np.array([[9, 0, 8], [0, 7, 6], [0, 0, 1]], dtype=np.float32)
    )

    camera = json.loads(path.read_text())["camera"]
    # Normalised onto the spelling the dataset uses, without losing the rest.
    assert set(camera) == {"intrinsic"}
    assert camera["intrinsic"]["resx"] == 1440
    assert camera["intrinsic"]["resy"] == 720
    assert camera["intrinsic"]["fx"] == 9.0


def test_class_id_options_accept_numpy_scalars_and_reject_fractions(loaders, tmp_path):
    path = tmp_path / "sky.png"
    mask = np.array([[True, False]])
    loaders.write_sky_mask(str(path), mask, {"sky_class_id": np.int64(30)})
    _assert_loaded(loaders.sky_mask(str(path), {"sky_class_id": np.int64(30)}), mask)
    # A whole-valued float is fine; a fractional one is a mistake, not a floor.
    _assert_loaded(loaders.sky_mask(str(path), {"sky_class_id": 30.0}), mask)
    for bad in (30.7, -1.9):
        with pytest.raises(ValueError, match="whole number"):
            loaders.sky_mask(str(path), {"sky_class_id": bad})
    for bad in (True, "30", [30]):
        with pytest.raises(ValueError, match="must be an integer"):
            loaders.sky_mask(str(path), {"sky_class_id": bad})


def test_write_intrinsics_rejects_non_finite_like_write_extrinsics(loaders, tmp_path):
    """Both camera writers must refuse values that cannot be valid JSON."""
    with pytest.raises(ValueError, match="finite"):
        loaders.write_intrinsics(
            str(tmp_path / "m.json"),
            np.array([[np.nan, 0, 1], [0, 2, 3], [0, 0, 1]], dtype=np.float32),
        )


def test_camera_json_write_is_atomic_and_leaves_no_residue(loaders, tmp_path):
    path = tmp_path / "meta.json"
    loaders.write_intrinsics(str(path), np.eye(3, dtype=np.float32))
    loaders.write_extrinsics(str(path), np.eye(4, dtype=np.float32))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]

    before = path.read_text()
    with pytest.raises(ValueError):
        loaders.write_extrinsics(str(path), np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32))
    # A rejected write leaves neither a damaged file nor a stray temporary.
    assert path.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json"]


def test_camera_json_write_preserves_file_permissions(loaders, tmp_path):
    """Replacing via a temporary must not inherit mkstemp's private mode."""
    umask = os.umask(0o077)
    os.umask(umask)

    fresh = tmp_path / "fresh.json"
    loaders.write_intrinsics(str(fresh), np.eye(3, dtype=np.float32))
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o666 & ~umask

    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({"camera": {}, "scene": {"a": 1}}), encoding="utf-8")
    existing.chmod(0o640)
    loaders.write_extrinsics(str(existing), np.eye(4, dtype=np.float32))
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640
    assert json.loads(existing.read_text())["scene"] == {"a": 1}


def test_camera_json_write_follows_a_symlinked_metadata_file(loaders, tmp_path):
    """The pose is constant across Synscapes, so one shared file is plausible."""
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps({"camera": {}, "scene": {"shared": True}}), encoding="utf-8"
    )
    link = tmp_path / "1.json"
    link.symlink_to(canonical)

    loaders.write_extrinsics(str(link), np.eye(4, dtype=np.float32))

    assert link.is_symlink()
    document = json.loads(canonical.read_text())
    assert "extrinsic" in document["camera"]
    assert document["scene"] == {"shared": True}


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permissions")
def test_camera_json_write_refuses_a_read_only_target(loaders, tmp_path):
    """rename only needs a writable directory, so check the file itself."""
    path = tmp_path / "ro.json"
    path.write_text('{"scene": {"keep": 1}}', encoding="utf-8")
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            loaders.write_extrinsics(str(path), np.eye(4, dtype=np.float32))
        assert path.read_text() == '{"scene": {"keep": 1}}'
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_class_id_options_report_non_finite_as_a_loader_error(loaders, tmp_path, value):
    """A bad option must not surface as a bare int() OverflowError."""
    path = tmp_path / "sky.png"
    loaders.write_sky_mask(str(path), np.array([[True, False]]))
    with pytest.raises(ValueError, match="finite"):
        loaders.sky_mask(str(path), {"sky_class_id": value})
