"""Princeton DENSE / SeeingThroughFog camera matrices: selection and write-back."""

from __future__ import annotations

import io
import json
import logging
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import resize_intrinsics
from euler_loading.loaders._princeton_dense import intrinsics_matrix
from euler_loading.loaders.cpu import princeton_dense as cpu
from euler_loading.loaders.gpu import princeton_dense as gpu

# The values and layout of the official calib_cam_stereo_left.json shipped with
# the SeeingThroughFog tools (tools/DatasetViewer/calibs/).
CAMERA_INFO = {
    "roi": {"width": 0, "do_rectify": False, "y_offset": 0, "x_offset": 0, "height": 0},
    "D": [-0.5675749999999999, 0.0, 0.0, 0.0, 0.0],
    "P": [
        2355.722801, 0.0, 988.138054, 0.0,
        0.0, 2355.722801, 508.051838, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ],
    "width": 1920,
    "K": [2612.86, 0.0, 966.525, 0.0, 2612.86, 508.297, 0.0, 0.0, 1.0],
    "height": 1024,
    "header": {
        "stamp": {"secs": 1517930040, "nsecs": 610230780},
        "frame_id": "cam_stereo_left_optical",
        "seq": 259044,
    },
    "R": [
        0.9999929999999999, -0.001011, -0.003531,
        0.0010019999999999999, 0.999996, -0.002456,
        0.0035329999999999997, 0.002453, 0.999991,
    ],
    "binning_y": 0,
    "binning_x": 0,
    "distortion_model": "plumb_bob",
}
P = np.asarray(CAMERA_INFO["P"]).reshape(3, 4)
K = np.asarray(CAMERA_INFO["K"]).reshape(3, 3)
# The official RGB2Gatedv2 crop: rows 202-970 and columns 280-1560.
TOP, LEFT, HEIGHT, WIDTH = 202, 280, 768, 1280


@pytest.fixture()
def calib(tmp_path) -> Path:
    path = tmp_path / "calib_cam_stereo_left.json"
    path.write_text(json.dumps(CAMERA_INFO, indent=4), encoding="utf-8")
    return path


def _crop(matrix: np.ndarray, top: int = TOP, left: int = LEFT) -> np.ndarray:
    cropped = np.array(matrix, dtype=np.float32, copy=True)
    cropped[0, 2] -= left
    cropped[1, 2] -= top
    return cropped


class TestMatrixSelection:
    def test_default_is_the_rectified_pinhole_matrix(self, calib):
        result = cpu.read_intrinsics(str(calib))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, P[:, :3].astype(np.float32))

    def test_gpu_default_matches_cpu(self, calib):
        result = gpu.read_intrinsics(str(calib))
        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), cpu.read_intrinsics(str(calib)))

    @pytest.mark.parametrize("name", ["K", "k", "raw", " RAW "])
    def test_raw_k_from_meta(self, calib, name):
        result = cpu.read_intrinsics(str(calib), {"intrinsics_matrix": name})
        np.testing.assert_array_equal(result, K.astype(np.float32))

    def test_attributes_override_meta(self, calib):
        result = gpu.read_intrinsics(
            str(calib),
            {"intrinsics_matrix": "K"},
            attributes={"intrinsics_matrix": "rectified"},
        )
        np.testing.assert_array_equal(result.numpy(), P[:, :3].astype(np.float32))

    def test_unknown_selection_raises(self, calib):
        with pytest.raises(ValueError, match="intrinsics_matrix must be"):
            cpu.read_intrinsics(str(calib), {"intrinsics_matrix": "Q"})

    def test_missing_p_points_to_the_raw_matrix(self, tmp_path):
        path = tmp_path / "k_only.json"
        path.write_text(json.dumps({"K": CAMERA_INFO["K"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a P array.*'K'"):
            cpu.read_intrinsics(str(path))
        np.testing.assert_array_equal(
            cpu.read_intrinsics(str(path), {"intrinsics_matrix": "K"}),
            K.astype(np.float32),
        )

    def test_unset_projection_is_rejected(self, tmp_path):
        """An uncalibrated camera-info carries an all-zero P."""
        path = tmp_path / "zeros.json"
        path.write_text(json.dumps({"K": CAMERA_INFO["K"], "P": [0.0] * 12}))
        with pytest.raises(ValueError, match="pinhole camera matrix"):
            cpu.read_intrinsics(str(path))

    def test_nested_matrices_and_streams(self):
        document = dict(CAMERA_INFO, P=P.tolist(), K=K.tolist())
        stream = io.BytesIO(json.dumps(document).encode("utf-8"))
        np.testing.assert_array_equal(
            cpu.read_intrinsics(stream), P[:, :3].astype(np.float32)
        )

    def test_selection_helper_default(self):
        assert intrinsics_matrix(None, None) == "P"


class TestWriteIntrinsics:
    def test_cpu_and_gpu_share_the_stream_capable_writer(self):
        assert cpu.write_intrinsics is gpu.write_intrinsics
        assert getattr(cpu.write_intrinsics, "__euler_loading_supports_stream__", False)

    def test_crop_shifts_k_and_p_exactly_and_keeps_the_rest(self, calib):
        cropped = _crop(cpu.read_intrinsics(str(calib)))
        cpu.write_intrinsics(str(calib), cropped, {"width": WIDTH, "height": HEIGHT})

        document = json.loads(calib.read_text(encoding="utf-8"))
        # Integer shifts of the stored float64 values, not float32 round trips.
        assert document["K"] == [2612.86, 0.0, 686.525, 0.0, 2612.86, 306.297, 0.0, 0.0, 1.0]
        assert document["P"] == [
            2355.722801, 0.0, 708.138054, 0.0,
            0.0, 2355.722801, 306.051838, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        assert (document["width"], document["height"]) == (WIDTH, HEIGHT)
        for key in ("D", "R", "roi", "header", "binning_x", "binning_y", "distortion_model"):
            assert document[key] == CAMERA_INFO[key], key
        np.testing.assert_array_equal(cpu.read_intrinsics(str(calib)), cropped)

    def test_crop_of_raw_k_moves_p_by_the_same_offset(self, calib):
        meta = {"intrinsics_matrix": "K", "width": WIDTH, "height": HEIGHT}
        cropped = _crop(cpu.read_intrinsics(str(calib), meta))
        cpu.write_intrinsics(str(calib), cropped, meta)

        document = json.loads(calib.read_text(encoding="utf-8"))
        assert document["K"][2] == 686.525 and document["K"][5] == 306.297
        assert document["P"][2] == 708.138054 and document["P"][6] == 306.051838
        np.testing.assert_array_equal(cpu.read_intrinsics(str(calib), meta), cropped)

    def test_rescale_scales_k_and_p_consistently(self, calib):
        half = resize_intrinsics(
            cpu.read_intrinsics(str(calib)), source_size=(1024, 1920), target_size=(512, 960)
        )
        gpu.write_intrinsics(str(calib), torch.from_numpy(half), {"width": 960, "height": 512})

        np.testing.assert_allclose(cpu.read_intrinsics(str(calib)), half, rtol=1e-6)
        expected_k = resize_intrinsics(K, source_size=(1024, 1920), target_size=(512, 960))
        np.testing.assert_allclose(
            cpu.read_intrinsics(str(calib), {"intrinsics_matrix": "K"}),
            expected_k,
            rtol=1e-6,
        )

    def test_change_that_is_no_crop_or_rescale_is_refused(self, calib):
        before = calib.read_bytes()
        skewed = cpu.read_intrinsics(str(calib))
        skewed[0, 1] = 5.0
        with pytest.raises(ValueError, match="not a crop or rescale"):
            cpu.write_intrinsics(str(calib), skewed)
        assert calib.read_bytes() == before
        assert sorted(p.name for p in calib.parent.iterdir()) == [calib.name]

    def test_missing_size_is_kept_with_a_warning(self, calib, caplog):
        with caplog.at_level(logging.WARNING, logger="euler_loading.loaders._princeton_dense"):
            cpu.write_intrinsics(str(calib), _crop(cpu.read_intrinsics(str(calib))))
        document = json.loads(calib.read_text(encoding="utf-8"))
        assert (document["width"], document["height"]) == (1920, 1024)
        assert "image size may no longer match" in caplog.text

    @pytest.mark.parametrize(
        "meta",
        [{"width": 1280}, {"width": 1280, "height": 76.8}, {"width": True, "height": 768}],
    )
    def test_size_must_be_two_positive_integers(self, calib, meta):
        with pytest.raises(ValueError, match="positive integer"):
            cpu.write_intrinsics(str(calib), cpu.read_intrinsics(str(calib)), meta)

    @pytest.mark.parametrize("selection", ["P", "K"])
    def test_new_file_is_an_ideal_pinhole_camera(self, tmp_path, selection):
        path = tmp_path / "out" / "calib_cam_stereo_left.json"
        matrix = np.array([[500.0, 0.0, 320.5], [0.0, 510.0, 240.25], [0.0, 0.0, 1.0]])
        meta = {"intrinsics_matrix": selection, "width": 640, "height": 480}
        cpu.write_intrinsics(str(path), matrix, meta)

        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["D"] == [0.0] * 5
        assert document["R"] == np.eye(3).reshape(-1).tolist()
        assert document["P"][3::4] == [0.0, 0.0, 0.0]
        assert (document["width"], document["height"]) == (640, 480)
        assert document["header"]["frame_id"] == "cam_stereo_left_optical"
        for read_as in ("P", "K"):
            np.testing.assert_array_equal(
                cpu.read_intrinsics(str(path), {"intrinsics_matrix": read_as}),
                matrix.astype(np.float32),
            )

    def test_stream_target_receives_a_fresh_document(self):
        stream = io.BytesIO()
        matrix = np.array([[400.0, 0.0, 200.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]])
        gpu.write_intrinsics(stream, matrix)
        stream.seek(0)
        np.testing.assert_array_equal(cpu.read_intrinsics(stream), matrix.astype(np.float32))

    @pytest.mark.parametrize(
        "value",
        [np.eye(4), np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]]), np.full((3, 3), np.nan)],
    )
    def test_non_pinhole_values_are_rejected(self, tmp_path, value):
        with pytest.raises(ValueError, match="intrinsics must"):
            cpu.write_intrinsics(str(tmp_path / "x.json"), value)

    def test_unreadable_existing_file_is_not_overwritten(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            cpu.write_intrinsics(str(path), np.eye(3))
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_existing_permissions_are_preserved(self, calib):
        calib.chmod(0o640)
        cpu.write_intrinsics(str(calib), _crop(cpu.read_intrinsics(str(calib))), {"width": WIDTH, "height": HEIGHT})
        assert stat.S_IMODE(calib.stat().st_mode) == 0o640

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_symlinked_calibration_is_updated_in_place(self, calib, tmp_path):
        link = tmp_path / "link.json"
        link.symlink_to(calib)
        cpu.write_intrinsics(str(link), _crop(cpu.read_intrinsics(str(link))), {"width": WIDTH, "height": HEIGHT})
        assert link.is_symlink()
        assert json.loads(calib.read_text(encoding="utf-8"))["width"] == WIDTH


class TestRawBayer:
    def test_default_pattern_matches_the_official_conversion(self, tmp_path):
        """OpenCV's COLOR_BAYER_GB2BGR, used upstream, is a GRBG sensor: G R / B G."""
        mosaic = np.full((8, 8), 1000, dtype=np.uint16)  # green sites
        mosaic[0::2, 1::2] = 4000  # red
        mosaic[1::2, 0::2] = 100  # blue
        path = tmp_path / "2018-10-08_08-18-59_01050.tiff"
        Image.fromarray(mosaic).save(path)

        red, green, blue = cpu.rccb(str(path)).reshape(-1, 3).mean(axis=0)
        assert red > green > blue
        red, _, blue = gpu.rccb(str(path)).permute(1, 2, 0).reshape(-1, 3).mean(dim=0)
        assert red > blue

    def test_pattern_override_still_applies(self, tmp_path):
        mosaic = np.full((8, 8), 1000, dtype=np.uint16)
        mosaic[0::2, 1::2] = 4000
        mosaic[1::2, 0::2] = 100
        path = tmp_path / "raw.tiff"
        Image.fromarray(mosaic).save(path)
        red, _, blue = cpu.rccb(str(path), {"bayer_pattern": "GBRG"}).reshape(-1, 3).mean(axis=0)
        assert blue > red


def _project(points: np.ndarray, matrix: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    camera = (extrinsics[:3, :3] @ points.T).T + extrinsics[:3, 3]
    pixels = (matrix @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def test_crop_needs_no_extrinsics_change(calib, tmp_path):
    """Lidar projected with the cropped matrix lands on the cropped pixels.

    The extrinsics are read before and after from the same file and only the
    intrinsics are rewritten: the crop is a pixel-grid change, so the whole
    correction lives in the camera matrix.
    """
    tf_tree = tmp_path / "calib_tf_tree_full.json"
    tf_tree.write_text(
        json.dumps(
            [
                {
                    "header": {"frame_id": "body"},
                    "child_frame_id": "lidar_hdl64_s3_roof",
                    "transform": {
                        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
                {
                    "header": {"frame_id": "body"},
                    "child_frame_id": "cam_stereo_left_optical",
                    # Optical axes: x right, y down, z forward = lidar x forward.
                    "transform": {
                        "translation": {"x": 0.1, "y": -0.2, "z": 0.3},
                        "rotation": {"x": -0.5, "y": 0.5, "z": -0.5, "w": 0.5},
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    extrinsics = cpu.read_extrinsics(str(tf_tree)).astype(np.float64)
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [rng.uniform(5.0, 80.0, 500), rng.uniform(-20.0, 20.0, 500), rng.uniform(-3.0, 3.0, 500)]
    )

    before = _project(points, cpu.read_intrinsics(str(calib)).astype(np.float64), extrinsics)
    cpu.write_intrinsics(
        str(calib), _crop(cpu.read_intrinsics(str(calib))), {"width": WIDTH, "height": HEIGHT}
    )
    after = _project(points, cpu.read_intrinsics(str(calib)).astype(np.float64), extrinsics)

    np.testing.assert_array_equal(cpu.read_extrinsics(str(tf_tree)), extrinsics.astype(np.float32))
    np.testing.assert_allclose(after, before - [LEFT, TOP], atol=1e-3)
