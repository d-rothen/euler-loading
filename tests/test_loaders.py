"""Unit tests for the loaders package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from euler_loading.loaders import vkitti2
from euler_loading.loaders.gpu import vkitti2 as gpu_vkitti2
from euler_loading.loaders.gpu import real_drive_sim as gpu_rds
from euler_loading.loaders.cpu import vkitti2 as cpu_vkitti2

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
