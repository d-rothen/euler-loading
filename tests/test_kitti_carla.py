"""KITTI-CARLA format decoding, calibration conventions and dataset integration."""

from __future__ import annotations

import importlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import (
    DenseDepthLoader,
    Modality,
    MultiModalDataset,
    SampleContext,
    resolve_loader_module,
)
from euler_loading.loaders.cpu import kitti_carla as cpu_kc
from euler_loading.loaders.generate.__main__ import generate
from euler_loading.loaders.gpu import kitti_carla as gpu_kc

# The calibration every KITTI-CARLA town ships.
LIDAR_TO_CAM = {
    0: "1.0 0.0 0.0 -0.30000001192092896 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.09999990463256836",
    1: "1.0 0.0 0.0 -0.30000001192092896 0.0 1.0 0.0 0.5 0.0 0.0 1.0 0.09999990463256836",
}
CALIB_HEADER = "#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)"
FOCAL = 1392 / (2 * math.tan(math.radians(36.0)))
SKY = (70, 130, 180)


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.kitti_carla")


@pytest.fixture(params=["str", "path", "stream"])
def source(request):
    def make_source(path):
        if request.param == "str":
            return str(path)
        if request.param == "stream":
            stream = io.BytesIO(Path(path).read_bytes())
            stream.name = str(path)
            return stream
        return Path(path)

    return make_source


def _np(value):
    return value.numpy() if isinstance(value, torch.Tensor) else value


def _image(result, loaders):
    """Return an image result in HW(C) layout, checking the variant's layout."""
    if loaders is gpu_kc:
        assert isinstance(result, torch.Tensor) and result.is_contiguous()
        arr = result.numpy()
        return arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
    assert isinstance(result, np.ndarray) and result.flags.c_contiguous
    return result


def _write_rgba(path, rgb):
    """Save *rgb* the way KITTI-CARLA does: RGBA with a constant alpha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _carla_depth_png(path, metres):
    code = np.rint(np.asarray(metres, np.float64) / 1000.0 * (2**24 - 1)).astype(np.uint32)
    rgb = np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], axis=-1).astype(np.uint8)
    _write_rgba(path, rgb)
    return (code * (1000.0 / (2**24 - 1))).astype(np.float32)


def _ply_bytes(points, *, endianness="<", line_end="\r\n"):
    fmt = {"<": "binary_little_endian", ">": "binary_big_endian"}[endianness]
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("cos_angle_lidar_surface", "f4"),
              ("timestamp", "f4"), ("instance", "u4"), ("semantic", "u4")]
    names = {"f4": "float32", "u4": "uint32"}
    vertices = np.empty(len(points), dtype=[(n, endianness + t) for n, t in fields])
    for column, (name, _) in enumerate(fields):
        vertices[name] = points[:, column]
    header = line_end.join(
        ["ply", f"format {fmt} 1.0", f"element vertex {len(points)}"]
        + [f"property {names[t]} {n}" for n, t in fields]
        + ["end_header", ""]
    )
    return header.encode("ascii") + vertices.tobytes()


POINTS = np.array(
    [
        [10.3, 0.0, -0.1, 0.5, 0.0, 0, 7],
        [-3.25, 12.5, 1.75, 1.0, 0.0994, 336, 10],
        [79.5, -79.5, -2.0, 0.003, 0.099, 4294967295, 22],
    ]
)
# What the loader returns: the stored float32 / uint32 values, exactly.
POINTS_STORED = np.column_stack(
    [POINTS[:, :5].astype(np.float32).astype(np.float64), POINTS[:, 5:]]
)


def _calib(path, camera=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{CALIB_HEADER}\n{LIDAR_TO_CAM[camera]}\n")
    return path


def _poses_file(path, *, seconds=0.3, speed=10.0, yaw_rate=1.0):
    """Synthetic 1 kHz trajectory: forward at *speed* m/s while turning."""
    times = np.arange(int(round(seconds * 1000)) + 1) * 0.001
    lines = ["# R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2) timestamp"]
    poses = []
    for t in times:
        c, s = math.cos(yaw_rate * t), math.sin(yaw_rate * t)
        pose = np.array([[c, -s, 0, speed * t], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        poses.append(pose)
        lines.append(" ".join(repr(float(v)) for v in pose[:3].reshape(-1)) + f" {float(t)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return np.array(poses), times


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_surface_and_resolution():
    from euler_loading.loaders import kitti_carla as top
    from euler_loading.loaders.contracts import DenseDepthCodec

    assert resolve_loader_module("kitti_carla") is gpu_kc
    assert resolve_loader_module("kitti_carla", variant="cpu") is cpu_kc
    assert isinstance(gpu_kc, DenseDepthLoader) and isinstance(cpu_kc, DenseDepthLoader)
    assert isinstance(top, DenseDepthCodec)
    assert top.rgb is gpu_kc.rgb
    assert gpu_kc.__all__ == cpu_kc.__all__
    for name in cpu_kc.__all__:
        assert hasattr(gpu_kc, name) and hasattr(top, name), name
    # Writers are shared, so the two variants stay in step by construction.
    for name in (n for n in cpu_kc.__all__ if n.startswith("write_")):
        assert getattr(gpu_kc, name) is getattr(cpu_kc, name)


def test_writer_resolves_for_every_reader():
    from euler_loading import resolve_writer_module
    from euler_loading._resolution import _writer_name_candidates

    module = resolve_writer_module("kitti_carla")
    for reader in ("rgb", "depth", "class_segmentation", "sky_mask", "lidar_point_cloud",
                   "point_cloud", "sparse_depth", "read_intrinsics", "read_extrinsics"):
        candidates = _writer_name_candidates(reader, None)
        assert any(callable(getattr(module, name, None)) for name in candidates), reader


def test_inventory_matches_annotations():
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["kitti_carla"] == saved["kitti_carla"]
    modalities = {item["function"]: item for item in generated["kitti_carla"]["modalities"]}
    assert set(modalities) == {
        "rgb", "depth", "class_segmentation", "sky_mask", "lidar_point_cloud",
        "sparse_depth", "read_intrinsics", "read_extrinsics", "lidar_pose",
    }
    assert modalities["depth"]["meta"]["radial_depth"] is False
    assert modalities["class_segmentation"]["meta"]["label_names"] == list(cpu_kc.CLASS_NAMES)
    for name in ("read_intrinsics", "read_extrinsics", "lidar_pose"):
        assert modalities[name]["hierarchical"] is True
    for name, cpu_shape, gpu_shape in [
        ("rgb", "HWC", "CHW"), ("depth", "HW", "1HW"), ("class_segmentation", "HW", "1HW"),
        ("sky_mask", "HW", "1HW"), ("lidar_point_cloud", "NC", "NC"),
        ("read_intrinsics", "3x3", "3x3"), ("read_extrinsics", "4x4", "4x4"),
        ("lidar_pose", "4x4", "4x4"),
    ]:
        assert modalities[name]["cpu"]["shape"] == cpu_shape
        assert modalities[name]["gpu"]["shape"] == gpu_shape
        cpu_meta = dict(getattr(cpu_kc, name)._modality_meta)
        gpu_meta = dict(getattr(gpu_kc, name)._modality_meta)
        cpu_meta.pop("shape")
        gpu_meta.pop("shape")
        assert cpu_meta == gpu_meta, name


def test_palette_matches_the_dataset_readme():
    assert len(cpu_kc.CLASS_NAMES) == len(cpu_kc.PALETTE) == 23
    assert len({tuple(c) for c in cpu_kc.PALETTE.tolist()}) == 23
    assert tuple(cpu_kc.PALETTE[cpu_kc.CLASS_NAMES.index("sky")]) == SKY
    assert tuple(cpu_kc.PALETTE[cpu_kc.CLASS_NAMES.index("road")]) == (128, 64, 128)
    assert tuple(cpu_kc.PALETTE[cpu_kc.CLASS_NAMES.index("terrain")]) == (145, 170, 100)
    with pytest.raises(ValueError):
        cpu_kc.PALETTE[0, 0] = 1


# ---------------------------------------------------------------------------
# Camera images
# ---------------------------------------------------------------------------


def test_rgb_drops_alpha(loaders, source, tmp_path):
    rgb = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3) * 10
    path = _write_rgba(tmp_path / "0000_0.png", rgb)
    result = loaders.rgb(source(path))
    arr = _image(result, loaders)
    assert arr.dtype == np.float32 and arr.shape == (2, 4, 3)
    np.testing.assert_array_equal(arr, rgb.astype(np.float32) / 255.0)


def test_depth_decodes_carla_rgb24(loaders, source, tmp_path):
    metres = np.array([[1.4884, 16.2], [123.456, 1000.0]])
    expected = _carla_depth_png(tmp_path / "0000_20.png", metres)
    arr = _image(loaders.depth(source(tmp_path / "0000_20.png")), loaders)
    assert arr.dtype == np.float32 and arr.shape == (2, 2)
    np.testing.assert_array_equal(arr, expected)
    assert arr[1, 1] == 1000.0  # the far plane, i.e. sky, is exact
    np.testing.assert_allclose(arr, metres, atol=1000.0 / 2**24)


def test_depth_byte_order_is_r_low_b_high(loaders, tmp_path):
    rgb = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.uint8)
    path = _write_rgba(tmp_path / "d.png", rgb)
    step = 1000.0 / (2**24 - 1)
    np.testing.assert_allclose(
        _image(loaders.depth(str(path)), loaders)[0], [step, 256 * step, 65536 * step], rtol=1e-6
    )


def test_class_segmentation_decodes_every_palette_colour(loaders, source, tmp_path):
    path = _write_rgba(tmp_path / "0000_10.png", cpu_kc.PALETTE[None, :, :].copy())
    arr = _image(loaders.class_segmentation(source(path)), loaders)
    assert arr.dtype == np.int64 and arr.shape == (1, 23)
    np.testing.assert_array_equal(arr[0], np.arange(23))


def test_class_segmentation_rejects_colours_outside_the_palette(loaders, tmp_path):
    rgb = np.array([[SKY, (1, 2, 3)]], dtype=np.uint8)
    path = _write_rgba(tmp_path / "bad.png", rgb)
    with pytest.raises(ValueError, match=r"outside the CARLA palette: \[1, 2, 3\]"):
        loaders.class_segmentation(str(path))


def test_sky_mask_default_and_overrides(loaders, source, tmp_path):
    rgb = np.array([[SKY, (128, 64, 128)], [(250, 170, 30), SKY]], dtype=np.uint8)
    path = _write_rgba(tmp_path / "0000_11.png", rgb)
    mask = _image(loaders.sky_mask(source(path)), loaders)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
    head = {"skyclass": [128, 64, 128]}
    np.testing.assert_array_equal(
        _image(loaders.sky_mask(str(path), head), loaders), [[False, True], [False, False]]
    )
    # A file's attributes take precedence over the head.
    np.testing.assert_array_equal(
        _image(loaders.sky_mask(str(path), head, attributes={"skyclass": [250, 170, 30]}), loaders),
        [[False, False], [True, False]],
    )


def test_write_rgb_roundtrip(loaders, tmp_path):
    # 2x4 rather than 2x3, so (H, W, 3) and (3, H, W) stay distinguishable.
    rgb = np.random.default_rng(0).integers(0, 256, (2, 4, 3)).astype(np.float32) / 255.0
    value = rgb if loaders is cpu_kc else torch.from_numpy(rgb).permute(2, 0, 1)
    loaders.write_rgb(str(tmp_path / "out" / "0000_0.png"), value)
    np.testing.assert_array_equal(_image(loaders.rgb(str(tmp_path / "out" / "0000_0.png")), loaders), rgb)


def test_write_depth_roundtrip_and_limits(loaders, tmp_path):
    metres = np.array([[0.0, 1.5, 33.3333], [999.99, 1000.0, 2500.0], [np.nan, np.inf, -4.0]], np.float32)
    value = metres if loaders is cpu_kc else torch.from_numpy(metres).unsqueeze(0)
    loaders.write_depth(str(tmp_path / "0000_20.png"), value)
    back = _image(loaders.depth(str(tmp_path / "0000_20.png")), loaders)
    expected = np.array([[0.0, 1.5, 33.3333], [999.99, 1000.0, 1000.0], [0.0, 1000.0, 0.0]])
    np.testing.assert_allclose(back, expected, atol=1000.0 / 2**24)


def test_write_class_segmentation_roundtrip(loaders, tmp_path):
    labels = np.arange(23, dtype=np.int64).reshape(1, 23)
    value = labels if loaders is cpu_kc else torch.from_numpy(labels).unsqueeze(0)
    loaders.write_class_segmentation(str(tmp_path / "0000_10.png"), value)
    with Image.open(tmp_path / "0000_10.png") as image:
        np.testing.assert_array_equal(np.array(image.convert("RGB"))[0], cpu_kc.PALETTE)
    np.testing.assert_array_equal(
        _image(loaders.class_segmentation(str(tmp_path / "0000_10.png")), loaders), labels
    )


@pytest.mark.parametrize("labels", [[[0, 23]], [[-1, 0]], [[0.5, 1.0]]])
def test_write_class_segmentation_rejects_invalid_tags(loaders, tmp_path, labels):
    with pytest.raises(ValueError, match="tag IDs"):
        loaders.write_class_segmentation(str(tmp_path / "x.png"), np.array(labels))


def test_write_sky_mask_roundtrips(loaders, tmp_path):
    mask = np.array([[True, False, True]])
    loaders.write_sky_mask(str(tmp_path / "sky.png"), mask)
    np.testing.assert_array_equal(_image(loaders.sky_mask(str(tmp_path / "sky.png")), loaders), mask)
    # The rest is "unlabeled", so the written file is a valid class image too.
    np.testing.assert_array_equal(
        _image(loaders.class_segmentation(str(tmp_path / "sky.png")), loaders), [[13, 0, 13]]
    )
    head = {"skyclass": [1, 2, 3]}
    loaders.write_sky_mask(str(tmp_path / "custom.png"), mask, head)
    np.testing.assert_array_equal(_image(loaders.sky_mask(str(tmp_path / "custom.png"), head), loaders), mask)


# ---------------------------------------------------------------------------
# Lidar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", ["lidar_point_cloud", "point_cloud", "sparse_depth"])
def test_lidar_reads_native_ply_exactly(loaders, source, tmp_path, function):
    path = tmp_path / "frame_0000.ply"
    path.write_bytes(_ply_bytes(POINTS))
    result = getattr(loaders, function)(source(path))
    arr = _np(result)
    assert arr.dtype == np.float64 and arr.shape == (3, 7)
    np.testing.assert_array_equal(arr, POINTS_STORED)
    assert arr[2, 5] == 4294967295  # the uint32 maximum survives


@pytest.mark.parametrize("endianness, line_end", [(">", "\n"), ("<", "\n")])
def test_lidar_accepts_either_byte_order_and_line_ending(tmp_path, endianness, line_end):
    path = tmp_path / "frame_0000.ply"
    path.write_bytes(_ply_bytes(POINTS, endianness=endianness, line_end=line_end))
    np.testing.assert_array_equal(cpu_kc.lidar_point_cloud(str(path)), POINTS_STORED)


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", "binary_little_endian"),
        (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n\0\0\0\0",
         "lacks the field"),
        (b"ply\nformat binary_little_endian 1.0\nelement vertex 9\nproperty float x\nend_header\n", "holds only"),
        (b"not a ply", "not a PLY"),
    ],
)
def test_lidar_rejects_unusable_files(tmp_path, payload, match):
    path = tmp_path / "frame_0000.ply"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        cpu_kc.lidar_point_cloud(str(path))


def test_write_lidar_point_cloud_roundtrip(loaders, tmp_path):
    path = tmp_path / "out" / "frame_0001.ply"
    loaders.write_lidar_point_cloud(str(path), torch.from_numpy(POINTS_STORED))
    np.testing.assert_array_equal(_np(loaders.lidar_point_cloud(str(path))), POINTS_STORED)
    assert loaders.write_point_cloud is loaders.write_sparse_depth is loaders.write_lidar_point_cloud
    with pytest.raises(ValueError, match=r"\(N, 7\)"):
        loaders.write_lidar_point_cloud(str(path), np.zeros((2, 3)))
    bad = POINTS_STORED.copy()
    bad[0, 6] = 1.5
    with pytest.raises(ValueError, match="'semantic'"):
        loaders.write_lidar_point_cloud(str(path), bad)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_intrinsics_follow_the_paper_camera_model(loaders, source, tmp_path):
    K = _np(loaders.read_intrinsics(source(_calib(tmp_path / "lidar_to_cam0.txt"))))
    assert K.dtype == np.float32 and K.shape == (3, 3)
    np.testing.assert_allclose(K, [[FOCAL, 0, 696], [0, FOCAL, 512], [0, 0, 1]], rtol=1e-6)
    assert FOCAL == pytest.approx(957.9618, abs=1e-4)


def test_intrinsics_overrides_prefer_attributes(loaders, tmp_path):
    path = str(_calib(tmp_path / "lidar_to_cam1.txt", camera=1))
    K = _np(loaders.read_intrinsics(path, {"width": 800, "height": 600, "horizontal_fov_deg": 90}))
    np.testing.assert_allclose(K, [[400, 0, 400], [0, 400, 300], [0, 0, 1]], rtol=1e-6)
    K = _np(loaders.read_intrinsics(path, {"width": 800, "height": 600}, attributes={"width": 1000}))
    assert K[0, 2] == pytest.approx(500) and K[1, 2] == pytest.approx(300)
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        loaders.read_intrinsics(path, {"horizontal_fov_deg": 180})
    with pytest.raises(ValueError, match="'width'"):
        loaders.read_intrinsics(path, {"width": "wide"})


def test_write_intrinsics_roundtrip(loaders, tmp_path):
    K = np.array([[478.98, 0, 348.25], [0, 478.98, 256.5], [0, 0, 1]], dtype=np.float32)
    path = tmp_path / "cropped" / "intrinsics.txt"
    loaders.write_intrinsics(str(path), K)
    assert path.read_text().startswith("#K(0,0)")
    np.testing.assert_array_equal(_np(loaders.read_intrinsics(str(path))), K)
    with pytest.raises(ValueError, match="finite"):
        loaders.write_intrinsics(str(path), np.full((3, 3), np.nan))


def test_intrinsics_and_extrinsics_reject_other_files(loaders, tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("# one two\n1 2 3 4\n")
    with pytest.raises(ValueError, match="12 numbers"):
        loaders.read_intrinsics(str(path))
    with pytest.raises(ValueError, match="12 numbers"):
        loaders.read_extrinsics(str(path))


def test_extrinsics_default_projects_lidar_through_intrinsics(loaders, source, tmp_path):
    """Camera 0 sits 0.3 m ahead of and 0.1 m below the lidar."""
    path = _calib(tmp_path / "lidar_to_cam0.txt")
    T = _np(loaders.read_extrinsics(source(path))).astype(np.float64)
    K = _np(loaders.read_intrinsics(str(path))).astype(np.float64)
    assert T.shape == (4, 4)
    ahead, left, up = (10.3, 0.0, -0.1), (10.3, 2.0, -0.1), (10.3, 0.0, 1.9)
    pixels = []
    for point in (ahead, left, up):
        uvw = K @ (T @ np.array([*point, 1.0]))[:3]
        pixels.append(uvw[:2] / uvw[2])
        assert uvw[2] == pytest.approx(10.0, abs=1e-6)  # planar depth along the optical axis
    np.testing.assert_allclose(pixels[0], [696, 512], atol=1e-3)
    assert pixels[1][0] < 696 and pixels[1][1] == pytest.approx(512, abs=1e-3)  # left is left
    assert pixels[2][1] < 512 and pixels[2][0] == pytest.approx(696, abs=1e-3)  # up is up


def test_extrinsics_conventions(loaders, tmp_path):
    path = str(_calib(tmp_path / "lidar_to_cam1.txt", camera=1))
    stored = np.eye(4)
    stored[:3, :] = np.array(LIDAR_TO_CAM[1].split(), float).reshape(3, 4)
    vehicle = _np(loaders.read_extrinsics(path, {"camera_axes": "vehicle"}))
    np.testing.assert_allclose(vehicle, stored, atol=1e-7)
    pose = _np(loaders.read_extrinsics(path, None, attributes={"transform_direction": "camera_to_lidar",
                                                               "camera_axes": "vehicle"}))
    # Camera 1 is 0.3 m forward, 0.5 m to the right and 0.1 m below the lidar.
    np.testing.assert_allclose(pose[:3, 3], [0.3, -0.5, -0.1], atol=1e-6)
    default = _np(loaders.read_extrinsics(path)).astype(np.float64)
    inverse = _np(loaders.read_extrinsics(path, {"transform_direction": "camera_to_lidar"})).astype(np.float64)
    np.testing.assert_allclose(default @ inverse, np.eye(4), atol=1e-6)
    with pytest.raises(ValueError, match="camera_axes"):
        loaders.read_extrinsics(path, {"camera_axes": "ned"})


@pytest.mark.parametrize("direction", ["lidar_to_camera", "camera_to_lidar"])
@pytest.mark.parametrize("axes", ["optical", "vehicle"])
def test_write_extrinsics_roundtrip(loaders, tmp_path, direction, axes):
    options = {"transform_direction": direction, "camera_axes": axes}
    native = _calib(tmp_path / "lidar_to_cam1.txt", camera=1)
    T = loaders.read_extrinsics(str(native), options)
    out = tmp_path / "out" / "lidar_to_cam1.txt"
    loaders.write_extrinsics(str(out), T, options)
    np.testing.assert_array_equal(_np(loaders.read_extrinsics(str(out), options)), _np(T))
    # The file is written in the native convention, whatever was read.
    np.testing.assert_allclose(
        np.array(out.read_text().splitlines()[1].split(), float),
        np.array(LIDAR_TO_CAM[1].split(), float), atol=1e-7,
    )
    assert out.read_text().splitlines()[0] == CALIB_HEADER


@pytest.mark.parametrize(
    "matrix, match",
    [
        (np.diag([2.0, 2.0, 2.0, 1.0]), "orthonormal"),
        (np.diag([1.0, 1.0, -1.0, 1.0]), "right-handed"),
        (np.vstack([np.eye(4)[:3], [9.0, 9.0, 9.0, 9.0]]), "bottom row"),
        (np.where(np.eye(4) > 0, np.nan, 0.0), "finite"),
    ],
)
def test_write_extrinsics_rejects_non_rigid_transforms(tmp_path, matrix, match):
    with pytest.raises(ValueError, match=match):
        cpu_kc.write_extrinsics(str(tmp_path / "bad.txt"), matrix)


@pytest.mark.parametrize(
    "writer, value, basename",
    [
        ("write_rgb", np.zeros((2, 2, 3), np.float32), "0000_0.png"),
        ("write_depth", np.ones((2, 2), np.float32), "0000_20.png"),
        ("write_class_segmentation", np.zeros((2, 2), np.int64), "0000_10.png"),
        ("write_sky_mask", np.zeros((2, 2), bool), "0000_10.png"),
        ("write_lidar_point_cloud", POINTS_STORED, "frame_0000.ply"),
        ("write_intrinsics", np.eye(3, dtype=np.float32), "intrinsics.txt"),
        ("write_extrinsics", np.eye(4, dtype=np.float32), "lidar_to_cam0.txt"),
    ],
)
def test_writers_accept_stream_targets(writer, value, basename):
    from euler_loading.loaders._writer_utils import supports_stream_target

    function = getattr(cpu_kc, writer)
    assert supports_stream_target(function)
    stream = io.BytesIO()
    stream.name = basename
    function(stream, value)
    assert stream.getvalue()


# ---------------------------------------------------------------------------
# Lidar trajectory
# ---------------------------------------------------------------------------


def _context(**paths):
    return SampleContext(
        modality="pose", entry={"path": "Town01/generated/full_poses_lidar.txt", "id": "lidar_pose"},
        id="frame", full_id="/town:Town01/frame", hierarchy_path=("town:Town01",),
        files={name: {"path": path, "id": "frame"} for name, path in paths.items()},
    )


def test_pose_table_and_per_sample_selection(loaders, source, tmp_path):
    poses, times = _poses_file(tmp_path / "full_poses_lidar.txt")
    table = loaders.lidar_pose(source(tmp_path / "full_poses_lidar.txt"))
    assert isinstance(table, cpu_kc.LidarPoseTable) and len(table) == len(times)
    np.testing.assert_allclose(table.poses, poses)
    # Camera frame i is the start of sweep i, 100 steps of 1 ms each.
    np.testing.assert_allclose(table.for_frame(2), poses[200])
    np.testing.assert_allclose(table.at(0.1004), poses[100])
    np.testing.assert_allclose(table.at_times([0.0, 0.0996, 0.25]), poses[[0, 100, 250]])
    with pytest.raises(KeyError, match="within 0.5 ms"):
        table.for_frame(5)

    select = getattr(loaders.lidar_pose, "__euler_loading_select__")
    context = _context(rgb="Town01/generated/images_rgb/0002_1.png",
                       depth="Town01/generated/images_depth/0002_21.png",
                       lidar="Town01/generated/frames/frame_0002.ply")
    pose = select(table, context)
    if loaders is gpu_kc:
        assert isinstance(pose, torch.Tensor) and pose.dtype == torch.float64
    np.testing.assert_allclose(_np(pose), poses[200])
    _np(pose)[:] = 0  # the selected pose is a copy, never a view into the cached table
    np.testing.assert_allclose(table.poses[200], poses[200])


def test_pose_selection_needs_one_native_frame_number(tmp_path):
    _poses_file(tmp_path / "full_poses_lidar.txt")
    table = cpu_kc.lidar_pose(str(tmp_path / "full_poses_lidar.txt"))
    select = cpu_kc.lidar_pose.__euler_loading_select__
    with pytest.raises(ValueError, match="several KITTI-CARLA frames"):
        select(table, _context(rgb="images_rgb/0001_0.png", lidar="frames/frame_0002.ply"))
    with pytest.raises(KeyError, match="native names"):
        select(table, _context(rgb="renamed/sample.png"))


def test_pose_file_must_have_13_columns(tmp_path):
    path = tmp_path / "full_poses_lidar.txt"
    path.write_text("# header\n1 2 3\n")
    with pytest.raises(ValueError, match="13 columns"):
        cpu_kc.lidar_pose(str(path))


# ---------------------------------------------------------------------------
# Dataset integration through ds-crawler
# ---------------------------------------------------------------------------

_TOWN = r"(?P<town>Town\d+)/generated"


def _camera_scope(folder, prefix):
    return (rf"^Town\d+/generated/{folder}/(?P<frame>\d+)_{prefix}\d\.png$",
            rf"^{_TOWN}/{folder}/\d+_{prefix}(?P<camera>\d)\.png$", None)


_SCOPES = {
    "rgb": ("rgb", {"range": [0, 255]}, "rgb", _camera_scope("images_rgb", "")),
    "depth": ("depth", {"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 1000]}, "depth",
              _camera_scope("images_depth", "2")),
    "semantic_segmentation": ("semantic_segmentation", {"skyclass": list(SKY)}, "class_segmentation",
                              _camera_scope("images_ss", "1")),
    "intrinsics": ("intrinsics", {}, "read_intrinsics",
                   (r"^Town\d+/generated/(lidar_to_cam)\d\.txt$",
                    rf"^{_TOWN}/lidar_to_cam(?P<camera>\d)\.txt$", "intrinsics")),
    "camera_extrinsics": ("camera_extrinsics", {}, "read_extrinsics",
                          (r"^Town\d+/generated/(lidar_to_cam)\d\.txt$",
                           rf"^{_TOWN}/lidar_to_cam(?P<camera>\d)\.txt$", "extrinsics")),
    "lidar_pose": ("lidar_pose", {}, "lidar_pose",
                   (r"^Town\d+/generated/(full_poses_lidar)\.txt$",
                    rf"^{_TOWN}/full_poses_lidar\.txt$", "lidar_pose")),
}


def _kitti_carla_root(root):
    """Write a two-frame, two-camera town with scoped ds-crawler metadata."""
    town = root / "Town01" / "generated"
    rng = np.random.default_rng(0)
    for frame in range(2):
        for camera in range(2):
            _write_rgba(town / "images_rgb" / f"{frame:04d}_{camera}.png",
                        rng.integers(0, 256, (2, 3, 3)).astype(np.uint8))
            _carla_depth_png(town / "images_depth" / f"{frame:04d}_2{camera}.png",
                             np.full((2, 3), 10.0 * (frame + 1) + camera))
            _write_rgba(town / "images_ss" / f"{frame:04d}_1{camera}.png",
                        np.array([[SKY, SKY, (128, 64, 128)]] * 2, np.uint8))
        (town / "frames").mkdir(parents=True, exist_ok=True)
        (town / "frames" / f"frame_{frame:04d}.ply").write_bytes(_ply_bytes(POINTS))
    for camera in range(2):
        _calib(town / f"lidar_to_cam{camera}.txt", camera)
    poses, _ = _poses_file(town / "full_poses_lidar.txt", seconds=0.2)
    for scope, (key, meta, function, (id_regex, hierarchy_regex, override)) in _SCOPES.items():
        metadata = root / ".ds_crawler" / scope
        metadata.mkdir(parents=True)
        head = {
            "contract": {"kind": "dataset_head", "version": "1.0"},
            "dataset": {"id": f"kitti_carla_{scope}", "name": f"KITTI-CARLA {scope}"},
            "modality": {"key": key, "meta": meta},
            "addons": {"euler_loading": {"version": "1.0", "loader": "kitti_carla", "function": function}},
        }
        id_config = {"regex": id_regex, "join_char": "+"}
        if override:
            id_config["override"] = override
        config = {
            "contract": {"kind": "ds_crawler_config", "version": "2.0"},
            "head_file": "dataset-head.json",
            "source": {"path": "."},
            "indexing": {"id": id_config, "hierarchy": {"regex": hierarchy_regex, "separator": ":"}},
            "metadata_scope": scope,
        }
        (metadata / "dataset-head.json").write_text(json.dumps(head))
        (metadata / "ds-crawler.json").write_text(json.dumps(config))
    return poses


@pytest.mark.parametrize("zipped", [False, True])
def test_dataset_resolves_kitti_carla_from_the_contract(tmp_path, zipped):
    root = tmp_path / "kitti_carla"
    poses = _kitti_carla_root(root)
    if zipped:
        archive = tmp_path / "Town01.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(root).as_posix())
        root = archive
    root = str(root)
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(root, metadata_scope="rgb"),
            "depth": Modality(root, metadata_scope="depth"),
            "classes": Modality(root, metadata_scope="semantic_segmentation"),
            "sky": Modality(root, metadata_scope="semantic_segmentation", function="sky_mask"),
        },
        hierarchical_modalities={
            "intrinsics": Modality(root, metadata_scope="intrinsics", collapse_single=True),
            "extrinsics": Modality(root, metadata_scope="camera_extrinsics", collapse_single=True),
            "pose": Modality(root, metadata_scope="lidar_pose", collapse_single=True),
        },
    )
    assert len(dataset) == 4
    for index in range(len(dataset)):
        sample = dataset[index]
        _, camera, frame = sample["full_id"].strip("/").split("/")
        camera, frame = int(camera.split(":")[1]), int(frame.split("-")[1])
        assert sample["rgb"].shape == (3, 2, 3)
        torch.testing.assert_close(sample["depth"], torch.full((1, 2, 3), 10.0 * (frame + 1) + camera),
                                   atol=1e-4, rtol=0)
        assert sample["classes"].tolist() == [[[13, 13, 7]] * 2]
        assert sample["sky"].tolist() == [[[True, True, False]] * 2]
        assert sample["intrinsics"][0, 0].item() == pytest.approx(FOCAL, rel=1e-6)
        # Each camera gets its own calibration file through the hierarchy.
        torch.testing.assert_close(sample["extrinsics"][0, 3], torch.tensor(-0.5 * camera))
        np.testing.assert_allclose(sample["pose"].numpy(), poses[100 * frame])
    assert dataset.get_writer("depth") is gpu_kc.write_depth
    assert dataset.get_writer("extrinsics") is gpu_kc.write_extrinsics
