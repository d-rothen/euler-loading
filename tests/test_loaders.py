"""Unit tests for the loaders package."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import resolve_loader_module
from euler_loading.loaders import vkitti2
from euler_loading.loaders import generic as generic_top
from euler_loading.loaders import muses as muses_top
from euler_loading.loaders import princeton_dense as princeton_dense_top
from euler_loading.loaders.gpu import vkitti2 as gpu_vkitti2
from euler_loading.loaders.gpu import real_drive_sim as gpu_rds
from euler_loading.loaders.gpu import generic as gpu_generic
from euler_loading.loaders.gpu import generic_dense_depth as gpu_generic_dense_depth
from euler_loading.loaders.gpu import muses as gpu_muses
from euler_loading.loaders.gpu import princeton_dense as gpu_princeton_dense
from euler_loading.loaders.cpu import vkitti2 as cpu_vkitti2
from euler_loading.loaders.cpu import real_drive_sim as cpu_rds
from euler_loading.loaders.cpu import generic as cpu_generic
from euler_loading.loaders.cpu import generic_dense_depth as cpu_generic_dense_depth
from euler_loading.loaders.cpu import muses as cpu_muses
from euler_loading.loaders.cpu import princeton_dense as cpu_princeton_dense

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LOADER_NAMES = [
    "rgb",
    "depth",
    "class_segmentation",
    "instance_segmentation",
    "scene_flow",
    "read_intrinsics",
    "read_extrinsics",
]

WRITER_NAMES = [
    "write_rgb",
    "write_depth",
    "write_class_segmentation",
    "write_instance_segmentation",
    "write_sky_mask",
    "write_scene_flow",
    "write_intrinsics",
    "write_extrinsics",
]


@pytest.fixture()
def rgb_path(tmp_path):
    """Write a tiny 2x2 RGB PNG."""
    from PIL import Image

    img = Image.new("RGB", (2, 2), color=(128, 64, 32))
    p = tmp_path / "rgb.png"
    img.save(p)
    return str(p)


@pytest.fixture()
def depth_path(tmp_path):
    """Write a tiny 2x2 16-bit PNG (values in centimetres)."""
    from PIL import Image

    arr = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    img = Image.fromarray(arr, mode="I;16")
    p = tmp_path / "depth.png"
    img.save(p)
    return str(p)


@pytest.fixture()
def text_path(tmp_path):
    """Write a small VKITTI2-style intrinsics text file with header."""
    p = tmp_path / "intrinsic.txt"
    p.write_text(
        "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
        "0 0 725.0087 725.0087 620.5 187\n"
        "0 1 725.0087 725.0087 620.5 187\n"
    )
    return str(p)


@pytest.fixture()
def extrinsic_text_path(tmp_path):
    """Write a small whitespace-delimited numeric text file."""
    p = tmp_path / "extrinsic.txt"
    p.write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")
    return str(p)


# ---------------------------------------------------------------------------
# Module contents (all three import paths)
# ---------------------------------------------------------------------------


class TestVKITTI2ModuleContents:
    """The vkitti2 module exposes the expected loader functions."""

    @pytest.mark.parametrize("name", LOADER_NAMES)
    def test_top_level_has_callable(self, name):
        assert callable(getattr(vkitti2, name))

    @pytest.mark.parametrize("name", LOADER_NAMES)
    def test_gpu_has_callable(self, name):
        assert callable(getattr(gpu_vkitti2, name))

    @pytest.mark.parametrize("name", LOADER_NAMES)
    def test_cpu_has_callable(self, name):
        assert callable(getattr(cpu_vkitti2, name))

    @pytest.mark.parametrize("name", WRITER_NAMES)
    def test_top_level_has_writer_callable(self, name):
        assert callable(getattr(vkitti2, name))

    @pytest.mark.parametrize("name", WRITER_NAMES)
    def test_gpu_has_writer_callable(self, name):
        assert callable(getattr(gpu_vkitti2, name))

    @pytest.mark.parametrize("name", WRITER_NAMES)
    def test_cpu_has_writer_callable(self, name):
        assert callable(getattr(cpu_vkitti2, name))


class TestGenericDenseDepthAttributes:
    """Representative built-in loader consumption of per-file attributes."""

    def test_gpu_depth_uses_scale_to_meters_override(self, depth_path):
        result = gpu_generic_dense_depth.depth(
            depth_path,
            attributes={"scale_to_meters_override": 0.01},
        )

        expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_cpu_depth_uses_scale_to_meters_override(self, depth_path):
        result = cpu_generic_dense_depth.depth(
            depth_path,
            attributes={"scale_to_meters_override": 0.01},
        )

        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Princeton DENSE / SeeingThroughFog loader smoke tests
# ---------------------------------------------------------------------------

PRINCETON_DENSE_LOADER_NAMES = [
    "rgb",
    "rccb",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
]


@pytest.fixture()
def princeton_dense_rgb_path(tmp_path):
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    path = tmp_path / "2018-02-05_12-09-01_00000.png"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture()
def princeton_dense_rccb_path(tmp_path):
    arr = np.full((4, 4), 2048, dtype=np.uint16)
    path = tmp_path / "2018-02-05_12-09-01_00000.tiff"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture()
def princeton_dense_sparse_depth_path(tmp_path):
    arr = np.array(
        [
            [1.0, 2.0, 3.0, 128.0, 12.0],
            [4.0, 5.0, 6.0, 255.0, 63.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "2018-02-06_15-48-12_00200.bin"
    arr.tofile(path)
    return str(path), arr


@pytest.fixture()
def princeton_dense_intrinsics_path(tmp_path):
    data = {
        "K": [2612.86, 0.0, 966.525, 0.0, 2612.86, 508.297, 0.0, 0.0, 1.0],
        "D": [-0.567575, 0.0, 0.0, 0.0, 0.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [2355.722801, 0.0, 988.138054, 0.0, 0.0, 2355.722801, 508.051838, 0.0, 0.0, 0.0, 1.0, 0.0],
        "width": 1920,
        "height": 1024,
        "header": {"frame_id": "cam_stereo_left_optical"},
    }
    path = tmp_path / "calib_cam_stereo_left.json"
    path.write_text(json.dumps(data))
    return str(path), np.asarray(data["K"], dtype=np.float32).reshape(3, 3)


@pytest.fixture()
def princeton_dense_tf_tree_path(tmp_path):
    data = [
        {
            "header": {"frame_id": "body"},
            "child_frame_id": "lidar_hdl64_s3_roof",
            "transform": {
                "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        {
            "header": {"frame_id": "body"},
            "child_frame_id": "cam_stereo_left_optical",
            "transform": {
                "translation": {"x": 0.0, "y": 2.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
    ]
    path = tmp_path / "calib_tf_tree_full.json"
    path.write_text(json.dumps(data))
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return str(path), expected


class TestPrincetonDenseModuleContents:
    """The Princeton DENSE module exposes the expected loader functions."""

    @pytest.mark.parametrize("name", PRINCETON_DENSE_LOADER_NAMES)
    def test_top_level_has_callable(self, name):
        assert callable(getattr(princeton_dense_top, name))

    @pytest.mark.parametrize("name", PRINCETON_DENSE_LOADER_NAMES)
    def test_gpu_has_callable(self, name):
        assert callable(getattr(gpu_princeton_dense, name))

    @pytest.mark.parametrize("name", PRINCETON_DENSE_LOADER_NAMES)
    def test_cpu_has_callable(self, name):
        assert callable(getattr(cpu_princeton_dense, name))

    def test_auto_resolution_uses_gpu_module(self):
        module = resolve_loader_module("princeton_dense")
        assert module.__name__ == "euler_loading.loaders.gpu.princeton_dense"

    def test_auto_resolution_supports_dataset_alias(self):
        module = resolve_loader_module("seeing_through_fog")
        assert module.__name__ == "euler_loading.loaders.gpu.princeton_dense"


class TestPrincetonDenseGPULoaders:
    """GPU Princeton DENSE loaders produce torch tensors."""

    def test_rccb_shape_dtype_and_range(self, princeton_dense_rccb_path):
        result = gpu_princeton_dense.rccb(princeton_dense_rccb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 4, 4)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_shape_dtype_and_range(self, princeton_dense_rgb_path):
        result = gpu_princeton_dense.rgb(princeton_dense_rgb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 4, 4)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_sparse_depth_preserves_float32_columns(self, princeton_dense_sparse_depth_path):
        path, expected = princeton_dense_sparse_depth_path
        result = gpu_princeton_dense.sparse_depth(path)
        assert result.dtype == torch.float32
        assert result.shape == (2, 5)
        assert torch.equal(result, torch.from_numpy(expected))

    def test_read_intrinsics_returns_camera_k(self, princeton_dense_intrinsics_path):
        path, expected = princeton_dense_intrinsics_path
        result = gpu_princeton_dense.read_intrinsics(path)
        assert result.dtype == torch.float32
        assert result.shape == (3, 3)
        assert torch.equal(result, torch.from_numpy(expected))

    def test_read_extrinsics_maps_hdl64_lidar_to_camera(self, princeton_dense_tf_tree_path):
        path, expected = princeton_dense_tf_tree_path
        result = gpu_princeton_dense.read_extrinsics(path)
        assert result.dtype == torch.float32
        assert result.shape == (4, 4)
        assert torch.equal(result, torch.from_numpy(expected))


class TestPrincetonDenseCPULoaders:
    """CPU Princeton DENSE loaders produce numpy arrays."""

    def test_rccb_shape_dtype_and_range(self, princeton_dense_rccb_path):
        result = cpu_princeton_dense.rccb(princeton_dense_rccb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (4, 4, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_shape_dtype_and_range(self, princeton_dense_rgb_path):
        result = cpu_princeton_dense.rgb(princeton_dense_rgb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (4, 4, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_sparse_depth_preserves_float32_columns(self, princeton_dense_sparse_depth_path):
        path, expected = princeton_dense_sparse_depth_path
        result = cpu_princeton_dense.sparse_depth(path)
        assert result.dtype == np.float32
        assert result.shape == (2, 5)
        assert np.array_equal(result, expected)

    def test_sparse_depth_supports_binary_streams(self, princeton_dense_sparse_depth_path):
        _, expected = princeton_dense_sparse_depth_path
        stream = io.BytesIO(expected.tobytes())
        result = cpu_princeton_dense.sparse_depth(stream)
        assert np.array_equal(result, expected)

    def test_read_intrinsics_returns_camera_k(self, princeton_dense_intrinsics_path):
        path, expected = princeton_dense_intrinsics_path
        result = cpu_princeton_dense.read_intrinsics(path)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)
        assert np.array_equal(result, expected)

    def test_read_extrinsics_maps_hdl64_lidar_to_camera(self, princeton_dense_tf_tree_path):
        path, expected = princeton_dense_tf_tree_path
        result = cpu_princeton_dense.read_extrinsics(path)
        assert result.dtype == np.float32
        assert result.shape == (4, 4)
        assert np.array_equal(result, expected)

    def test_read_extrinsics_supports_frame_overrides(self, princeton_dense_tf_tree_path):
        path, expected_lidar_to_camera = princeton_dense_tf_tree_path
        result = cpu_princeton_dense.read_extrinsics(
            path,
            attributes={
                "source_frame": "cam_stereo_left_optical",
                "target_frame": "lidar_hdl64_s3_roof",
            },
        )
        assert np.allclose(result, np.linalg.inv(expected_lidar_to_camera))


class TestPrincetonDenseBackwardCompat:
    """``from euler_loading.loaders import princeton_dense`` returns GPU loaders."""

    def test_top_level_rccb_matches_gpu(self, princeton_dense_rccb_path):
        assert torch.equal(
            princeton_dense_top.rccb(princeton_dense_rccb_path),
            gpu_princeton_dense.rccb(princeton_dense_rccb_path),
        )

    def test_top_level_rgb_matches_gpu(self, princeton_dense_rgb_path):
        assert torch.equal(
            princeton_dense_top.rgb(princeton_dense_rgb_path),
            gpu_princeton_dense.rgb(princeton_dense_rgb_path),
        )

    def test_top_level_sparse_depth_matches_gpu(self, princeton_dense_sparse_depth_path):
        path, _ = princeton_dense_sparse_depth_path
        assert torch.equal(
            princeton_dense_top.sparse_depth(path),
            gpu_princeton_dense.sparse_depth(path),
        )


# ---------------------------------------------------------------------------
# MUSES loader smoke tests
# ---------------------------------------------------------------------------

MUSES_LOADER_NAMES = [
    "rgb",
    "reference_rgb",
    "semantic_segmentation",
    "semantic_segmentation_color",
    "sky_mask",
    "panoptic_segmentation",
    "lidar_point_cloud",
    "point_cloud",
    "sparse_depth",
    "read_intrinsics",
    "read_extrinsics",
]


@pytest.fixture()
def muses_rgb_path(tmp_path):
    arr = np.array(
        [
            [[0, 64, 128], [255, 128, 0]],
            [[32, 16, 8], [4, 2, 1]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "REC0001_frame_000001_frame_camera.png"
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path), arr


@pytest.fixture()
def muses_semantic_path(tmp_path):
    arr = np.array([[0, 7], [24, 255]], dtype=np.uint8)
    path = tmp_path / "REC0001_frame_000001_gt_labelTrainIds.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path), arr


@pytest.fixture()
def muses_sky_mask_path(tmp_path):
    arr = np.array([[0, 10], [10, 255]], dtype=np.uint8)
    path = tmp_path / "REC0001_frame_000001_gt_labelTrainIds.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path), arr == 10


@pytest.fixture()
def muses_sky_label_ids_path(tmp_path):
    arr = np.array([[0, 23], [23, 255]], dtype=np.uint8)
    path = tmp_path / "REC0001_frame_000001_gt_labelIds.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path), arr == 23


@pytest.fixture()
def muses_sky_rgb_class_path(tmp_path):
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[0, 1] = [0, 0, 23]
    arr[1, 0] = [0, 0, 23]
    arr[1, 1] = [0, 0, 7]
    path = tmp_path / "REC0001_frame_000001_gt_labelIds_rgb.png"
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path), np.all(arr == np.array([0, 0, 23], dtype=np.uint8), axis=-1)


@pytest.fixture()
def muses_semantic_color_path(tmp_path):
    arr = np.array(
        [
            [[128, 64, 128], [70, 70, 70]],
            [[220, 20, 60], [0, 0, 142]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "REC0001_frame_000001_gt_labelColor.png"
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path), arr


@pytest.fixture()
def muses_panoptic_path(tmp_path):
    arr = np.array(
        [
            [[1, 0, 0], [2, 1, 0]],
            [[3, 2, 1], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "REC0001_frame_000001_gt_panoptic.png"
    Image.fromarray(arr, mode="RGB").save(path)
    expected = (
        arr[:, :, 0].astype(np.int64)
        + 256 * arr[:, :, 1].astype(np.int64)
        + 65536 * arr[:, :, 2].astype(np.int64)
    )
    return str(path), expected


@pytest.fixture()
def muses_lidar_path(tmp_path):
    arr = np.array(
        [
            [1.0, 2.0, 3.0, 19.0, 1.0, 1671182560.125],
            [4.0, 5.0, 6.0, 27.0, 5.0, 1671182560.250],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "REC0001_frame_000001_lidar.bin"
    arr.tofile(path)
    return str(path), arr


@pytest.fixture()
def muses_calib_path(tmp_path):
    data = {
        "intrinsics": {
            "rgb": {
                "K": [
                    [1055.0, 0.0, 941.0],
                    [0.0, 1056.0, 550.0],
                    [0.0, 0.0, 1.0],
                ],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "event": {
                "K": [
                    [1038.0, 0.0, 625.0],
                    [0.0, 1039.0, 344.0],
                    [0.0, 0.0, 1.0],
                ],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "extrinsics": {
            "lidar2rgb": [
                [1.0, 0.0, 0.0, 0.12],
                [0.0, 1.0, 0.0, 0.03],
                [0.0, 0.0, 1.0, -0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "radar2rgb": [
                [0.0, -1.0, 0.0, -0.4],
                [1.0, 0.0, 0.0, 0.15],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(data))
    return str(path), data


class TestMUSESModuleContents:
    """The muses module exposes the expected loader functions."""

    @pytest.mark.parametrize("name", MUSES_LOADER_NAMES)
    def test_top_level_has_callable(self, name):
        assert callable(getattr(muses_top, name))

    @pytest.mark.parametrize("name", MUSES_LOADER_NAMES)
    def test_gpu_has_callable(self, name):
        assert callable(getattr(gpu_muses, name))

    @pytest.mark.parametrize("name", MUSES_LOADER_NAMES)
    def test_cpu_has_callable(self, name):
        assert callable(getattr(cpu_muses, name))

    def test_reference_rgb_is_marked_as_rgb_modality(self):
        assert gpu_muses.reference_rgb._modality_meta["type"] == "rgb"
        assert cpu_muses.reference_rgb._modality_meta["type"] == "rgb"


class TestMUSESGPULoaders:
    """GPU MUSES loaders produce torch tensors from minimal on-disk data."""

    def test_rgb_shape_dtype_and_range(self, muses_rgb_path):
        path, _ = muses_rgb_path
        result = gpu_muses.rgb(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 2, 2)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_reference_rgb_matches_rgb_loader(self, muses_rgb_path):
        path, _ = muses_rgb_path
        assert torch.equal(gpu_muses.reference_rgb(path), gpu_muses.rgb(path))

    def test_semantic_segmentation_is_single_channel_long(self, muses_semantic_path):
        path, expected = muses_semantic_path
        result = gpu_muses.semantic_segmentation(path)
        assert result.dtype == torch.int64
        assert result.shape == (1, 2, 2)
        assert torch.equal(result, torch.from_numpy(expected.astype(np.int64)).unsqueeze(0))

    def test_sky_mask_uses_train_ids(self, muses_sky_mask_path):
        path, expected = muses_sky_mask_path
        result = gpu_muses.sky_mask(path)
        assert result.dtype == torch.bool
        assert result.shape == (1, 2, 2)
        assert torch.equal(result, torch.from_numpy(expected).unsqueeze(0))

    def test_sky_mask_uses_label_ids_from_filename(self, muses_sky_label_ids_path):
        path, expected = muses_sky_label_ids_path
        result = gpu_muses.sky_mask(path)
        assert torch.equal(result, torch.from_numpy(expected).unsqueeze(0))

    def test_sky_mask_uses_rgb_sky_class_meta(self, muses_sky_rgb_class_path):
        path, expected = muses_sky_rgb_class_path
        result = gpu_muses.sky_mask(path, meta={"sky_class": [0, 0, 23]})
        assert torch.equal(result, torch.from_numpy(expected).unsqueeze(0))

    def test_semantic_color_is_chw_uint8(self, muses_semantic_color_path):
        path, expected = muses_semantic_color_path
        result = gpu_muses.semantic_segmentation_color(path)
        assert result.dtype == torch.uint8
        assert result.shape == (3, 2, 2)
        assert torch.equal(result, torch.from_numpy(expected).permute(2, 0, 1))

    def test_panoptic_segmentation_decodes_rgb_ids(self, muses_panoptic_path):
        path, expected = muses_panoptic_path
        result = gpu_muses.panoptic_segmentation(path)
        assert result.dtype == torch.int64
        assert result.shape == (1, 2, 2)
        assert torch.equal(result, torch.from_numpy(expected).unsqueeze(0))

    def test_lidar_point_cloud_preserves_float64_columns(self, muses_lidar_path):
        path, expected = muses_lidar_path
        result = gpu_muses.lidar_point_cloud(path)
        assert result.dtype == torch.float64
        assert result.shape == (2, 6)
        assert torch.equal(result, torch.from_numpy(expected))

    def test_point_cloud_alias_matches_lidar_point_cloud(self, muses_lidar_path):
        path, _ = muses_lidar_path
        assert torch.equal(gpu_muses.point_cloud(path), gpu_muses.lidar_point_cloud(path))

    def test_sparse_depth_matches_lidar_point_cloud(self, muses_lidar_path):
        path, _ = muses_lidar_path
        assert torch.equal(gpu_muses.sparse_depth(path), gpu_muses.lidar_point_cloud(path))

    def test_read_intrinsics_defaults_to_rgb(self, muses_calib_path):
        path, data = muses_calib_path
        result = gpu_muses.read_intrinsics(path)
        assert result.dtype == torch.float32
        assert result.shape == (3, 3)
        assert torch.equal(result, torch.tensor(data["intrinsics"]["rgb"]["K"], dtype=torch.float32))

    def test_read_intrinsics_can_select_sensor_from_attributes(self, muses_calib_path):
        path, data = muses_calib_path
        result = gpu_muses.read_intrinsics(path, attributes={"sensor": "event"})
        assert torch.equal(result, torch.tensor(data["intrinsics"]["event"]["K"], dtype=torch.float32))

    def test_read_extrinsics_defaults_to_lidar2rgb(self, muses_calib_path):
        path, data = muses_calib_path
        result = gpu_muses.read_extrinsics(path)
        assert result.dtype == torch.float32
        assert result.shape == (4, 4)
        assert torch.equal(result, torch.tensor(data["extrinsics"]["lidar2rgb"], dtype=torch.float32))

    def test_read_extrinsics_can_select_transform_from_meta(self, muses_calib_path):
        path, data = muses_calib_path
        result = gpu_muses.read_extrinsics(path, meta={"transform": "radar2rgb"})
        assert torch.equal(result, torch.tensor(data["extrinsics"]["radar2rgb"], dtype=torch.float32))


class TestMUSESCPULoaders:
    """CPU MUSES loaders produce numpy arrays from minimal on-disk data."""

    def test_rgb_shape_dtype_and_range(self, muses_rgb_path):
        path, _ = muses_rgb_path
        result = cpu_muses.rgb(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_semantic_segmentation_is_hw_int64(self, muses_semantic_path):
        path, expected = muses_semantic_path
        result = cpu_muses.semantic_segmentation(path)
        assert result.dtype == np.int64
        assert result.shape == (2, 2)
        assert np.array_equal(result, expected.astype(np.int64))

    def test_sky_mask_uses_train_ids(self, muses_sky_mask_path):
        path, expected = muses_sky_mask_path
        result = cpu_muses.sky_mask(path)
        assert result.dtype == np.bool_
        assert result.shape == (2, 2)
        assert np.array_equal(result, expected)

    def test_sky_mask_uses_label_ids_from_filename(self, muses_sky_label_ids_path):
        path, expected = muses_sky_label_ids_path
        result = cpu_muses.sky_mask(path)
        assert np.array_equal(result, expected)

    def test_sky_mask_uses_rgb_sky_class_meta(self, muses_sky_rgb_class_path):
        path, expected = muses_sky_rgb_class_path
        result = cpu_muses.sky_mask(path, meta={"sky_class": [0, 0, 23]})
        assert np.array_equal(result, expected)

    def test_semantic_color_is_hwc_uint8(self, muses_semantic_color_path):
        path, expected = muses_semantic_color_path
        result = cpu_muses.semantic_segmentation_color(path)
        assert result.dtype == np.uint8
        assert result.shape == (2, 2, 3)
        assert np.array_equal(result, expected)

    def test_panoptic_segmentation_decodes_rgb_ids(self, muses_panoptic_path):
        path, expected = muses_panoptic_path
        result = cpu_muses.panoptic_segmentation(path)
        assert result.dtype == np.int64
        assert result.shape == (2, 2)
        assert np.array_equal(result, expected)

    def test_lidar_point_cloud_preserves_float64_columns(self, muses_lidar_path):
        path, expected = muses_lidar_path
        result = cpu_muses.lidar_point_cloud(path)
        assert result.dtype == np.float64
        assert result.shape == (2, 6)
        assert np.array_equal(result, expected)

    def test_point_cloud_alias_matches_lidar_point_cloud(self, muses_lidar_path):
        path, _ = muses_lidar_path
        assert np.array_equal(cpu_muses.point_cloud(path), cpu_muses.lidar_point_cloud(path))

    def test_lidar_point_cloud_supports_binary_streams(self, muses_lidar_path):
        _, expected = muses_lidar_path
        stream = io.BytesIO(expected.tobytes())
        result = cpu_muses.lidar_point_cloud(stream)
        assert np.array_equal(result, expected)

    def test_read_intrinsics_defaults_to_rgb(self, muses_calib_path):
        path, data = muses_calib_path
        result = cpu_muses.read_intrinsics(path)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)
        assert np.array_equal(result, np.asarray(data["intrinsics"]["rgb"]["K"], dtype=np.float32))

    def test_read_intrinsics_can_select_sensor_from_attributes(self, muses_calib_path):
        path, data = muses_calib_path
        result = cpu_muses.read_intrinsics(path, attributes={"sensor": "event"})
        assert np.array_equal(result, np.asarray(data["intrinsics"]["event"]["K"], dtype=np.float32))

    def test_read_extrinsics_defaults_to_lidar2rgb(self, muses_calib_path):
        path, data = muses_calib_path
        result = cpu_muses.read_extrinsics(path)
        assert result.dtype == np.float32
        assert result.shape == (4, 4)
        assert np.array_equal(result, np.asarray(data["extrinsics"]["lidar2rgb"], dtype=np.float32))

    def test_read_extrinsics_can_select_transform_from_meta(self, muses_calib_path):
        path, data = muses_calib_path
        result = cpu_muses.read_extrinsics(path, meta={"transform": "radar2rgb"})
        assert np.array_equal(result, np.asarray(data["extrinsics"]["radar2rgb"], dtype=np.float32))


class TestMUSESBackwardCompat:
    """``from euler_loading.loaders import muses`` returns GPU loaders."""

    def test_top_level_rgb_matches_gpu(self, muses_rgb_path):
        path, _ = muses_rgb_path
        assert torch.equal(muses_top.rgb(path), gpu_muses.rgb(path))

    def test_top_level_panoptic_matches_gpu(self, muses_panoptic_path):
        path, _ = muses_panoptic_path
        assert torch.equal(
            muses_top.panoptic_segmentation(path),
            gpu_muses.panoptic_segmentation(path),
        )


# ---------------------------------------------------------------------------
# GPU loader smoke tests
# ---------------------------------------------------------------------------


class TestGPULoaders:
    """GPU loaders produce torch tensors from minimal on-disk data."""

    # -- rgb ----------------------------------------------------------------

    def test_rgb_returns_float_tensor(self, rgb_path):
        result = gpu_vkitti2.rgb(rgb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_rgb_shape_is_chw(self, rgb_path):
        result = gpu_vkitti2.rgb(rgb_path)
        assert result.shape == (3, 2, 2)

    def test_rgb_values_in_unit_range(self, rgb_path):
        result = gpu_vkitti2.rgb(rgb_path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    # -- depth --------------------------------------------------------------

    def test_depth_returns_float_tensor(self, depth_path):
        result = gpu_vkitti2.depth(depth_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_depth_shape_has_channel_dim(self, depth_path):
        result = gpu_vkitti2.depth(depth_path)
        assert result.shape == (1, 2, 2)

    def test_depth_converts_to_metres(self, depth_path):
        result = gpu_vkitti2.depth(depth_path)
        assert torch.isclose(result[0, 0, 0], torch.tensor(1.0))
        assert torch.isclose(result[0, 0, 1], torch.tensor(2.0))

    # -- class_segmentation -------------------------------------------------

    def test_class_segmentation_returns_long_tensor(self, rgb_path):
        result = gpu_vkitti2.class_segmentation(rgb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.int64

    def test_class_segmentation_shape_is_chw(self, rgb_path):
        result = gpu_vkitti2.class_segmentation(rgb_path)
        assert result.shape == (3, 2, 2)

    # -- instance_segmentation ----------------------------------------------

    def test_instance_segmentation_returns_long_tensor(self, rgb_path):
        result = gpu_vkitti2.instance_segmentation(rgb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.int64

    def test_instance_segmentation_shape_is_chw(self, rgb_path):
        result = gpu_vkitti2.instance_segmentation(rgb_path)
        assert result.shape == (3, 2, 2)

    # -- scene_flow ---------------------------------------------------------

    def test_scene_flow_returns_float_tensor(self, rgb_path):
        result = gpu_vkitti2.scene_flow(rgb_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_scene_flow_shape_is_chw(self, rgb_path):
        result = gpu_vkitti2.scene_flow(rgb_path)
        assert result.shape == (3, 2, 2)

    # -- read_intrinsics ----------------------------------------------------

    def test_read_intrinsics_returns_3x3_tensor(self, text_path):
        result = gpu_vkitti2.read_intrinsics(text_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 3)

    def test_read_intrinsics_k_matrix_values(self, text_path):
        K = gpu_vkitti2.read_intrinsics(text_path)
        expected = torch.tensor(
            [[725.0087, 0.0, 620.5],
             [0.0, 725.0087, 187.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(K, expected)

    # -- read_extrinsics ----------------------------------------------------

    def test_read_extrinsics_returns_float_tensor(self, extrinsic_text_path):
        result = gpu_vkitti2.read_extrinsics(extrinsic_text_path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# CPU loader smoke tests
# ---------------------------------------------------------------------------


class TestCPULoaders:
    """CPU loaders produce numpy ndarrays from minimal on-disk data."""

    # -- rgb ----------------------------------------------------------------

    def test_rgb_returns_float_array(self, rgb_path):
        result = cpu_vkitti2.rgb(rgb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_rgb_shape_is_hwc(self, rgb_path):
        result = cpu_vkitti2.rgb(rgb_path)
        assert result.shape == (2, 2, 3)

    def test_rgb_values_in_unit_range(self, rgb_path):
        result = cpu_vkitti2.rgb(rgb_path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    # -- depth --------------------------------------------------------------

    def test_depth_returns_float_array(self, depth_path):
        result = cpu_vkitti2.depth(depth_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_depth_shape_is_hw(self, depth_path):
        result = cpu_vkitti2.depth(depth_path)
        assert result.shape == (2, 2)

    def test_depth_converts_to_metres(self, depth_path):
        result = cpu_vkitti2.depth(depth_path)
        assert np.isclose(result[0, 0], 1.0)
        assert np.isclose(result[0, 1], 2.0)

    # -- class_segmentation -------------------------------------------------

    def test_class_segmentation_returns_int_array(self, rgb_path):
        result = cpu_vkitti2.class_segmentation(rgb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    def test_class_segmentation_shape_is_hwc(self, rgb_path):
        result = cpu_vkitti2.class_segmentation(rgb_path)
        assert result.shape == (2, 2, 3)

    # -- instance_segmentation ----------------------------------------------

    def test_instance_segmentation_returns_int_array(self, rgb_path):
        result = cpu_vkitti2.instance_segmentation(rgb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64

    def test_instance_segmentation_shape_is_hwc(self, rgb_path):
        result = cpu_vkitti2.instance_segmentation(rgb_path)
        assert result.shape == (2, 2, 3)

    # -- scene_flow ---------------------------------------------------------

    def test_scene_flow_returns_float_array(self, rgb_path):
        result = cpu_vkitti2.scene_flow(rgb_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_scene_flow_shape_is_hwc(self, rgb_path):
        result = cpu_vkitti2.scene_flow(rgb_path)
        assert result.shape == (2, 2, 3)

    # -- read_intrinsics ----------------------------------------------------

    def test_read_intrinsics_returns_3x3_array(self, text_path):
        result = cpu_vkitti2.read_intrinsics(text_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)

    def test_read_intrinsics_k_matrix_values(self, text_path):
        K = cpu_vkitti2.read_intrinsics(text_path)
        expected = np.array(
            [[725.0087, 0.0, 620.5],
             [0.0, 725.0087, 187.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        assert np.allclose(K, expected)

    # -- read_extrinsics ----------------------------------------------------

    def test_read_extrinsics_returns_float_array(self, extrinsic_text_path):
        result = cpu_vkitti2.read_extrinsics(extrinsic_text_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Backward-compatible top-level import
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """``from euler_loading.loaders import vkitti2`` still returns GPU loaders."""

    def test_top_level_rgb_matches_gpu(self, rgb_path):
        top = vkitti2.rgb(rgb_path)
        gpu = gpu_vkitti2.rgb(rgb_path)
        assert torch.equal(top, gpu)

    def test_top_level_depth_matches_gpu(self, depth_path):
        top = vkitti2.depth(depth_path)
        gpu = gpu_vkitti2.depth(depth_path)
        assert torch.equal(top, gpu)


# ---------------------------------------------------------------------------
# Real Drive Sim calibration loader
# ---------------------------------------------------------------------------

_RDS_CALIB_PATH = str(Path(__file__).parent / "example_rds_calib.json")


class TestRDSCalibration:
    """GPU calibration loader parses the Real Drive Sim JSON format."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.result = gpu_rds.calibration(_RDS_CALIB_PATH)

    def test_returns_dict_keyed_by_sensor_name(self):
        assert set(self.result.keys()) == {"CS_FRONT", "HDL_32E", "HDL_64E"}

    def test_each_sensor_has_expected_keys(self):
        for sensor in self.result.values():
            assert set(sensor.keys()) == {"K", "T", "distortion"}

    # -- intrinsics ---------------------------------------------------------

    def test_intrinsics_shape(self):
        assert self.result["CS_FRONT"]["K"].shape == (3, 3)

    def test_intrinsics_dtype(self):
        assert self.result["CS_FRONT"]["K"].dtype == torch.float32

    def test_intrinsics_values(self):
        K = self.result["CS_FRONT"]["K"]
        assert torch.isclose(K[0, 0], torch.tensor(2262.52001953125))  # fx
        assert torch.isclose(K[1, 1], torch.tensor(2265.3017578125))   # fy
        assert torch.isclose(K[0, 2], torch.tensor(1096.97998046875))  # cx
        assert torch.isclose(K[1, 2], torch.tensor(513.1370239257812)) # cy
        assert K[2, 2] == 1.0

    # -- extrinsics ---------------------------------------------------------

    def test_extrinsics_shape(self):
        assert self.result["CS_FRONT"]["T"].shape == (4, 4)

    def test_extrinsics_dtype(self):
        assert self.result["CS_FRONT"]["T"].dtype == torch.float32

    def test_extrinsics_last_row(self):
        T = self.result["CS_FRONT"]["T"]
        assert torch.equal(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_extrinsics_translation(self):
        T = self.result["CS_FRONT"]["T"]
        assert torch.isclose(T[0, 3], torch.tensor(1.100000023841858))
        assert torch.isclose(T[1, 3], torch.tensor(0.20000000298023224))
        assert torch.isclose(T[2, 3], torch.tensor(1.25))

    def test_rotation_is_orthonormal(self):
        R = self.result["CS_FRONT"]["T"][:3, :3]
        eye = R @ R.T
        assert torch.allclose(eye, torch.eye(3), atol=1e-6)
        assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    # -- distortion ---------------------------------------------------------

    def test_distortion_shape(self):
        assert self.result["CS_FRONT"]["distortion"].shape == (8,)

    def test_distortion_dtype(self):
        assert self.result["CS_FRONT"]["distortion"].dtype == torch.float32


class TestRDSCPUCalibration:
    """CPU calibration loader parses the Real Drive Sim JSON format."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.result = cpu_rds.calibration(_RDS_CALIB_PATH)

    def test_returns_dict_keyed_by_sensor_name(self):
        assert set(self.result.keys()) == {"CS_FRONT", "HDL_32E", "HDL_64E"}

    def test_each_sensor_has_expected_keys(self):
        for sensor in self.result.values():
            assert set(sensor.keys()) == {"K", "T", "distortion"}

    def test_intrinsics_shape(self):
        assert self.result["CS_FRONT"]["K"].shape == (3, 3)

    def test_intrinsics_dtype(self):
        assert self.result["CS_FRONT"]["K"].dtype == np.float32

    def test_intrinsics_values(self):
        K = self.result["CS_FRONT"]["K"]
        assert np.isclose(K[0, 0], 2262.52001953125)
        assert np.isclose(K[1, 1], 2265.3017578125)
        assert np.isclose(K[0, 2], 1096.97998046875)
        assert np.isclose(K[1, 2], 513.1370239257812)
        assert K[2, 2] == 1.0

    def test_read_intrinsics_returns_front_camera_matrix(self):
        K = cpu_rds.read_intrinsics(_RDS_CALIB_PATH)
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        assert np.isclose(K[0, 0], 2262.52001953125)

    def test_read_intrinsics_can_select_sensor_from_attributes(self):
        K = cpu_rds.read_intrinsics(_RDS_CALIB_PATH, attributes={"sensor": "HDL_32E"})
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        assert K[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Writer round-trip tests
# ---------------------------------------------------------------------------


class TestVKITTI2Writers:
    def test_gpu_write_depth_roundtrip(self, tmp_path):
        depth = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.float32,
        )
        path = tmp_path / "depth.png"

        gpu_vkitti2.write_depth(str(path), depth)
        loaded = gpu_vkitti2.depth(str(path))

        assert torch.allclose(loaded, depth, atol=1e-4)

    def test_cpu_write_depth_roundtrip(self, tmp_path):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "depth.png"

        cpu_vkitti2.write_depth(str(path), depth)
        loaded = cpu_vkitti2.depth(str(path))

        assert np.allclose(loaded, depth, atol=1e-4)

    def test_gpu_write_intrinsics_roundtrip(self, tmp_path):
        K = torch.tensor(
            [[725.0, 0.0, 620.5], [0.0, 725.0, 187.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        path = tmp_path / "intrinsics.txt"

        gpu_vkitti2.write_intrinsics(str(path), K)
        loaded = gpu_vkitti2.read_intrinsics(str(path))

        assert torch.allclose(loaded, K, atol=1e-4)


class TestRDSWriters:
    def test_gpu_write_depth_roundtrip(self, tmp_path):
        depth = torch.tensor(
            [[[1.25, 2.5], [3.75, 4.0]]],
            dtype=torch.float32,
        )
        path = tmp_path / "depth.npz"

        gpu_rds.write_depth(str(path), depth)
        loaded = gpu_rds.depth(str(path))

        assert torch.allclose(loaded, depth, atol=1e-6)

    def test_cpu_write_depth_roundtrip(self, tmp_path):
        depth = np.array([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32)
        path = tmp_path / "depth.npz"

        cpu_rds.write_depth(str(path), depth)
        loaded = cpu_rds.depth(str(path))

        assert np.allclose(loaded, depth, atol=1e-6)

    def test_gpu_write_sky_mask_roundtrip(self, tmp_path):
        mask = torch.tensor(
            [[[True, False], [False, True]]],
            dtype=torch.bool,
        )
        path = tmp_path / "sky.png"

        gpu_rds.write_sky_mask(str(path), mask)
        loaded = gpu_rds.sky_mask(str(path))

        assert torch.equal(loaded, mask)


# ---------------------------------------------------------------------------
# Generic spherical_map loader tests
# ---------------------------------------------------------------------------

GENERIC_LOADER_NAMES = [
    "map_2d",
    "map_3d",
    "semantic_segmentation",
    "instance_segmentation",
    "points_3d",
    "scattering_coefficient",
    "atmospheric_light",
    "spherical_map",
]
GENERIC_WRITER_NAMES = [
    "write_map_2d",
    "write_map_3d",
    "write_semantic_segmentation",
    "write_instance_segmentation",
    "write_points_3d",
    "write_scattering_coefficient",
    "write_atmospheric_light",
    "write_spherical_map",
]


@pytest.fixture()
def spherical_npy_path(tmp_path):
    """Write a small (3, 4, 5) float32 .npy file."""
    arr = np.random.default_rng(42).random((3, 4, 5), dtype=np.float32)
    p = tmp_path / "spherical.npy"
    np.save(str(p), arr)
    return str(p), arr


@pytest.fixture()
def spherical_npz_path(tmp_path):
    """Write a small (3, 4, 5) float32 .npz file."""
    arr = np.random.default_rng(42).random((3, 4, 5), dtype=np.float32)
    p = tmp_path / "spherical.npz"
    np.savez_compressed(str(p), data=arr)
    return str(p), arr


class TestGenericModuleContents:
    """The generic module exposes the expected loader and writer functions."""

    @pytest.mark.parametrize("name", GENERIC_LOADER_NAMES)
    def test_gpu_has_callable(self, name):
        assert callable(getattr(gpu_generic, name))

    @pytest.mark.parametrize("name", GENERIC_LOADER_NAMES)
    def test_cpu_has_callable(self, name):
        assert callable(getattr(cpu_generic, name))

    @pytest.mark.parametrize("name", GENERIC_WRITER_NAMES)
    def test_gpu_has_writer_callable(self, name):
        assert callable(getattr(gpu_generic, name))

    @pytest.mark.parametrize("name", GENERIC_WRITER_NAMES)
    def test_cpu_has_writer_callable(self, name):
        assert callable(getattr(cpu_generic, name))

    @pytest.mark.parametrize("name", GENERIC_LOADER_NAMES)
    def test_top_level_has_callable(self, name):
        assert callable(getattr(generic_top, name))


class TestGPUGenericLoaders:
    """GPU generic loaders produce torch tensors."""

    def test_spherical_map_npy_returns_float_tensor(self, spherical_npy_path):
        path, _ = spherical_npy_path
        result = gpu_generic.spherical_map(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_spherical_map_npy_shape_is_chw(self, spherical_npy_path):
        path, expected = spherical_npy_path
        result = gpu_generic.spherical_map(path)
        assert result.shape == (3, 4, 5)

    def test_spherical_map_npy_values_match(self, spherical_npy_path):
        path, expected = spherical_npy_path
        result = gpu_generic.spherical_map(path)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_spherical_map_npz_returns_float_tensor(self, spherical_npz_path):
        path, _ = spherical_npz_path
        result = gpu_generic.spherical_map(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_spherical_map_npz_shape_is_chw(self, spherical_npz_path):
        path, expected = spherical_npz_path
        result = gpu_generic.spherical_map(path)
        assert result.shape == (3, 4, 5)

    def test_spherical_map_npz_values_match(self, spherical_npz_path):
        path, expected = spherical_npz_path
        result = gpu_generic.spherical_map(path)
        assert torch.allclose(result, torch.from_numpy(expected))


class TestCPUGenericLoaders:
    """CPU generic loaders produce numpy arrays in HWC layout."""

    def test_spherical_map_npy_returns_float_array(self, spherical_npy_path):
        path, _ = spherical_npy_path
        result = cpu_generic.spherical_map(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_spherical_map_npy_shape_is_hwc(self, spherical_npy_path):
        path, _ = spherical_npy_path
        result = cpu_generic.spherical_map(path)
        assert result.shape == (4, 5, 3)

    def test_spherical_map_npy_values_match(self, spherical_npy_path):
        path, expected = spherical_npy_path
        result = cpu_generic.spherical_map(path)
        assert np.allclose(result, np.transpose(expected, (1, 2, 0)))

    def test_spherical_map_npz_returns_float_array(self, spherical_npz_path):
        path, _ = spherical_npz_path
        result = cpu_generic.spherical_map(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_spherical_map_npz_shape_is_hwc(self, spherical_npz_path):
        path, _ = spherical_npz_path
        result = cpu_generic.spherical_map(path)
        assert result.shape == (4, 5, 3)


class TestGenericWriters:
    """Writer round-trip tests for generic loaders."""

    def test_gpu_write_spherical_map_npy_roundtrip(self, tmp_path):
        data = torch.rand(3, 4, 5, dtype=torch.float32)
        path = tmp_path / "spherical.npy"
        gpu_generic.write_spherical_map(str(path), data)
        loaded = gpu_generic.spherical_map(str(path))
        assert torch.allclose(loaded, data)

    def test_gpu_write_spherical_map_npz_roundtrip(self, tmp_path):
        data = torch.rand(3, 4, 5, dtype=torch.float32)
        path = tmp_path / "spherical.npz"
        gpu_generic.write_spherical_map(str(path), data)
        loaded = gpu_generic.spherical_map(str(path))
        assert torch.allclose(loaded, data)

    def test_cpu_write_spherical_map_npy_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 3)).astype(np.float32)
        path = tmp_path / "spherical.npy"
        cpu_generic.write_spherical_map(str(path), data)
        loaded = cpu_generic.spherical_map(str(path))
        assert np.allclose(loaded, data)

    def test_cpu_write_spherical_map_npz_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 3)).astype(np.float32)
        path = tmp_path / "spherical.npz"
        cpu_generic.write_spherical_map(str(path), data)
        loaded = cpu_generic.spherical_map(str(path))
        assert np.allclose(loaded, data)


class TestGenericBackwardCompat:
    """``from euler_loading.loaders import generic`` returns GPU loaders."""

    def test_top_level_spherical_map_matches_gpu(self, spherical_npy_path):
        path, _ = spherical_npy_path
        top = generic_top.spherical_map(path)
        gpu = gpu_generic.spherical_map(path)
        assert torch.equal(top, gpu)


class TestGenericSegmentation:
    """Generic segmentation codecs preserve the standardized CUPS format."""

    @pytest.mark.parametrize(
        ("function", "writer", "dtype", "torch_dtype", "values"),
        [
            ("semantic_segmentation", "write_semantic_segmentation", np.uint8, torch.uint8, [[0, 18], [255, 7]]),
            ("instance_segmentation", "write_instance_segmentation", np.uint16, torch.uint16, [[0, 1], [257, 65535]]),
        ],
    )
    def test_cpu_gpu_npy_roundtrip(self, tmp_path, function, writer, dtype, torch_dtype, values):
        expected = np.asarray(values, dtype=dtype)
        path = tmp_path / f"{function}.npy"
        getattr(cpu_generic, writer)(str(path), expected)

        cpu_value = getattr(cpu_generic, function)(str(path))
        gpu_value = getattr(gpu_generic, function)(str(path))
        assert cpu_value.dtype == dtype
        assert np.array_equal(cpu_value, expected)
        assert gpu_value.dtype == torch_dtype
        assert torch.equal(gpu_value, torch.from_numpy(expected))

    def test_writer_accepts_singleton_channel(self, tmp_path):
        value = np.array([[[0, 1], [2, 255]]], dtype=np.uint8)
        path = tmp_path / "semantic.npz"
        cpu_generic.write_semantic_segmentation(str(path), value)
        loaded = cpu_generic.semantic_segmentation(str(path))
        assert loaded.shape == (2, 2)
        assert np.array_equal(loaded, value[0])

    def test_loader_rejects_non_hw_storage(self, tmp_path):
        path = tmp_path / "semantic.npy"
        np.save(path, np.zeros((1, 2, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match=r"must have shape \(H, W\)"):
            cpu_generic.semantic_segmentation(str(path))


# ---------------------------------------------------------------------------
# Generic map_2d / map_3d loader tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def map_2d_npy_path(tmp_path):
    """Write a small (4, 5) float32 .npy file."""
    arr = np.random.default_rng(0).random((4, 5), dtype=np.float32)
    p = tmp_path / "map_2d.npy"
    np.save(str(p), arr)
    return str(p), arr


@pytest.fixture()
def map_2d_npz_path(tmp_path):
    """Write a small (4, 5) float32 .npz file."""
    arr = np.random.default_rng(1).random((4, 5), dtype=np.float32)
    p = tmp_path / "map_2d.npz"
    np.savez_compressed(str(p), data=arr)
    return str(p), arr


@pytest.fixture()
def map_3d_npy_path(tmp_path):
    """Write a small (2, 4, 5) float32 .npy file in CHW layout."""
    arr = np.random.default_rng(2).random((2, 4, 5), dtype=np.float32)
    p = tmp_path / "map_3d.npy"
    np.save(str(p), arr)
    return str(p), arr


@pytest.fixture()
def map_3d_npz_path(tmp_path):
    """Write a small (2, 4, 5) float32 .npz file in CHW layout."""
    arr = np.random.default_rng(3).random((2, 4, 5), dtype=np.float32)
    p = tmp_path / "map_3d.npz"
    np.savez_compressed(str(p), data=arr)
    return str(p), arr


@pytest.fixture()
def points_3d_npy_path(tmp_path):
    """Write a small (3, 4, 5) float32 .npy file in 3HW layout."""
    arr = np.random.default_rng(4).random((3, 4, 5), dtype=np.float32)
    p = tmp_path / "points_3d.npy"
    np.save(str(p), arr)
    return str(p), arr


@pytest.fixture()
def points_3d_npz_path(tmp_path):
    """Write a small (3, 4, 5) float32 .npz file in 3HW layout."""
    arr = np.random.default_rng(5).random((3, 4, 5), dtype=np.float32)
    p = tmp_path / "points_3d.npz"
    np.savez_compressed(str(p), data=arr)
    return str(p), arr


class TestGPUMap2DLoader:
    """GPU ``map_2d`` loads a 2D map as an ``(H, W)`` float32 tensor."""

    def test_npy_dtype_and_shape(self, map_2d_npy_path):
        path, _ = map_2d_npy_path
        result = gpu_generic.map_2d(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (4, 5)

    def test_npy_values_match(self, map_2d_npy_path):
        path, expected = map_2d_npy_path
        result = gpu_generic.map_2d(path)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_npz_dtype_and_shape(self, map_2d_npz_path):
        path, _ = map_2d_npz_path
        result = gpu_generic.map_2d(path)
        assert result.dtype == torch.float32
        assert result.shape == (4, 5)


class TestCPUMap2DLoader:
    """CPU ``map_2d`` loads a 2D map as an ``(H, W)`` float32 array."""

    def test_npy_dtype_and_shape(self, map_2d_npy_path):
        path, _ = map_2d_npy_path
        result = cpu_generic.map_2d(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (4, 5)

    def test_npy_values_match(self, map_2d_npy_path):
        path, expected = map_2d_npy_path
        result = cpu_generic.map_2d(path)
        assert np.allclose(result, expected)

    def test_npz_dtype_and_shape(self, map_2d_npz_path):
        path, _ = map_2d_npz_path
        result = cpu_generic.map_2d(path)
        assert result.dtype == np.float32
        assert result.shape == (4, 5)


class TestGPUMap3DLoader:
    """GPU ``map_3d`` loads a 3D map as a ``(C, H, W)`` float32 tensor."""

    def test_npy_dtype_and_shape(self, map_3d_npy_path):
        path, _ = map_3d_npy_path
        result = gpu_generic.map_3d(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (2, 4, 5)

    def test_npy_values_match(self, map_3d_npy_path):
        path, expected = map_3d_npy_path
        result = gpu_generic.map_3d(path)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_npz_dtype_and_shape(self, map_3d_npz_path):
        path, _ = map_3d_npz_path
        result = gpu_generic.map_3d(path)
        assert result.shape == (2, 4, 5)


class TestCPUMap3DLoader:
    """CPU ``map_3d`` transposes CHW-on-disk to ``(H, W, C)``."""

    def test_npy_dtype_and_shape(self, map_3d_npy_path):
        path, _ = map_3d_npy_path
        result = cpu_generic.map_3d(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (4, 5, 2)

    def test_npy_values_match(self, map_3d_npy_path):
        path, expected = map_3d_npy_path
        result = cpu_generic.map_3d(path)
        assert np.allclose(result, np.transpose(expected, (1, 2, 0)))


class TestGPUPoints3DLoader:
    """GPU ``points_3d`` loads dense 3D points as a ``(3, H, W)`` tensor."""

    def test_npy_dtype_and_shape(self, points_3d_npy_path):
        path, _ = points_3d_npy_path
        result = gpu_generic.points_3d(path)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 4, 5)

    def test_npy_values_match(self, points_3d_npy_path):
        path, expected = points_3d_npy_path
        result = gpu_generic.points_3d(path)
        assert torch.allclose(result, torch.from_numpy(expected))

    def test_npz_dtype_and_shape(self, points_3d_npz_path):
        path, _ = points_3d_npz_path
        result = gpu_generic.points_3d(path)
        assert result.dtype == torch.float32
        assert result.shape == (3, 4, 5)


class TestCPUPoints3DLoader:
    """CPU ``points_3d`` preserves the ``(3, H, W)`` dense point layout."""

    def test_npy_dtype_and_shape(self, points_3d_npy_path):
        path, _ = points_3d_npy_path
        result = cpu_generic.points_3d(path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 4, 5)

    def test_npy_values_match(self, points_3d_npy_path):
        path, expected = points_3d_npy_path
        result = cpu_generic.points_3d(path)
        assert np.allclose(result, expected)

    def test_rejects_hwc_input(self, tmp_path):
        arr = np.random.default_rng(6).random((4, 5, 3), dtype=np.float32)
        path = tmp_path / "points_3d.npy"
        np.save(str(path), arr)

        with pytest.raises(ValueError, match=r"points_3d must have shape"):
            cpu_generic.points_3d(str(path))


class TestMapWriters:
    """Writer round-trip tests for ``map_2d`` / ``map_3d``."""

    def test_gpu_write_map_2d_npy_roundtrip(self, tmp_path):
        data = torch.rand(4, 5, dtype=torch.float32)
        path = tmp_path / "m.npy"
        gpu_generic.write_map_2d(str(path), data)
        loaded = gpu_generic.map_2d(str(path))
        assert torch.allclose(loaded, data)

    def test_gpu_write_map_2d_accepts_1hw(self, tmp_path):
        data = torch.rand(1, 4, 5, dtype=torch.float32)
        path = tmp_path / "m.npy"
        gpu_generic.write_map_2d(str(path), data)
        loaded = gpu_generic.map_2d(str(path))
        assert loaded.shape == (4, 5)
        assert torch.allclose(loaded, data.squeeze(0))

    def test_gpu_write_map_2d_npz_roundtrip(self, tmp_path):
        data = torch.rand(4, 5, dtype=torch.float32)
        path = tmp_path / "m.npz"
        gpu_generic.write_map_2d(str(path), data)
        loaded = gpu_generic.map_2d(str(path))
        assert torch.allclose(loaded, data)

    def test_cpu_write_map_2d_npy_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5)).astype(np.float32)
        path = tmp_path / "m.npy"
        cpu_generic.write_map_2d(str(path), data)
        loaded = cpu_generic.map_2d(str(path))
        assert np.allclose(loaded, data)

    def test_gpu_write_map_3d_npy_roundtrip(self, tmp_path):
        data = torch.rand(2, 4, 5, dtype=torch.float32)
        path = tmp_path / "m.npy"
        gpu_generic.write_map_3d(str(path), data)
        loaded = gpu_generic.map_3d(str(path))
        assert torch.allclose(loaded, data)

    def test_gpu_write_points_3d_npy_roundtrip(self, tmp_path):
        data = torch.rand(3, 4, 5, dtype=torch.float32)
        path = tmp_path / "points.npy"
        gpu_generic.write_points_3d(str(path), data)
        loaded = gpu_generic.points_3d(str(path))
        assert torch.allclose(loaded, data)

    def test_cpu_write_map_3d_npy_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 2)).astype(np.float32)
        path = tmp_path / "m.npy"
        cpu_generic.write_map_3d(str(path), data)
        loaded = cpu_generic.map_3d(str(path))
        assert np.allclose(loaded, data)

    def test_cpu_write_points_3d_npz_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
        path = tmp_path / "points.npz"
        cpu_generic.write_points_3d(str(path), data)
        loaded = cpu_generic.points_3d(str(path))
        assert np.allclose(loaded, data)

    def test_cpu_write_map_3d_npz_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 2)).astype(np.float32)
        path = tmp_path / "m.npz"
        cpu_generic.write_map_3d(str(path), data)
        loaded = cpu_generic.map_3d(str(path))
        assert np.allclose(loaded, data)

    def test_gpu_to_cpu_map_3d_matches_transpose(self, tmp_path):
        """A GPU-written map_3d reads back on CPU as the HWC transpose."""
        data = torch.rand(2, 4, 5, dtype=torch.float32)
        path = tmp_path / "m.npy"
        gpu_generic.write_map_3d(str(path), data)
        loaded = cpu_generic.map_3d(str(path))
        assert np.allclose(loaded, data.permute(1, 2, 0).numpy())


class TestSpecificMapAliases:
    """``scattering_coefficient`` / ``atmospheric_light`` mirror map_2d/map_3d."""

    def test_gpu_scattering_coefficient_matches_map_2d(self, map_2d_npy_path):
        path, _ = map_2d_npy_path
        assert torch.equal(
            gpu_generic.scattering_coefficient(path),
            gpu_generic.map_2d(path),
        )

    def test_cpu_scattering_coefficient_matches_map_2d(self, map_2d_npy_path):
        path, _ = map_2d_npy_path
        assert np.array_equal(
            cpu_generic.scattering_coefficient(path),
            cpu_generic.map_2d(path),
        )

    def test_gpu_atmospheric_light_matches_map_3d(self, map_3d_npy_path):
        path, _ = map_3d_npy_path
        assert torch.equal(
            gpu_generic.atmospheric_light(path),
            gpu_generic.map_3d(path),
        )

    def test_cpu_atmospheric_light_matches_map_3d(self, map_3d_npy_path):
        path, _ = map_3d_npy_path
        assert np.array_equal(
            cpu_generic.atmospheric_light(path),
            cpu_generic.map_3d(path),
        )

    def test_modality_meta_distinct_types(self):
        """Each alias carries its own modality_type even though logic is shared."""
        assert gpu_generic.points_3d._modality_meta["type"] == "points_3d"
        assert cpu_generic.points_3d._modality_meta["type"] == "points_3d"
        assert (
            gpu_generic.scattering_coefficient._modality_meta["type"]
            == "scattering_coefficient"
        )
        assert (
            gpu_generic.atmospheric_light._modality_meta["type"]
            == "atmospheric_light"
        )
        assert (
            cpu_generic.scattering_coefficient._modality_meta["type"]
            == "scattering_coefficient"
        )
        assert (
            cpu_generic.atmospheric_light._modality_meta["type"]
            == "atmospheric_light"
        )

    def test_gpu_write_scattering_coefficient_roundtrip(self, tmp_path):
        data = torch.rand(4, 5, dtype=torch.float32)
        path = tmp_path / "sc.npy"
        gpu_generic.write_scattering_coefficient(str(path), data)
        assert torch.allclose(gpu_generic.scattering_coefficient(str(path)), data)

    def test_gpu_write_atmospheric_light_roundtrip(self, tmp_path):
        data = torch.rand(2, 4, 5, dtype=torch.float32)
        path = tmp_path / "al.npy"
        gpu_generic.write_atmospheric_light(str(path), data)
        assert torch.allclose(gpu_generic.atmospheric_light(str(path)), data)

    def test_cpu_write_atmospheric_light_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).random((4, 5, 2)).astype(np.float32)
        path = tmp_path / "al.npy"
        cpu_generic.write_atmospheric_light(str(path), data)
        assert np.allclose(cpu_generic.atmospheric_light(str(path)), data)
