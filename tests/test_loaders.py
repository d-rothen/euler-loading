"""Unit tests for the loaders package."""

from __future__ import annotations

import pytest

from euler_loading.loaders import vkitti2


# ---------------------------------------------------------------------------
# VKITTI2 module contents
# ---------------------------------------------------------------------------


class TestVKITTI2ModuleContents:
    """The vkitti2 module exposes the expected loader functions."""

    @pytest.mark.parametrize(
        "name",
        [
            "rgb",
            "depth",
            "class_segmentation",
            "instance_segmentation",
            "scene_flow",
            "read_intrinsics",
            "read_extrinsics",
        ],
    )
    def test_has_callable(self, name):
        assert callable(getattr(vkitti2, name))


# ---------------------------------------------------------------------------
# VKITTI2 loader smoke tests (using temp files)
# ---------------------------------------------------------------------------


class TestVKITTI2Loaders:
    """Loaders produce expected output types from minimal on-disk data."""

    @pytest.fixture()
    def rgb_path(self, tmp_path):
        """Write a tiny 2x2 RGB PNG."""
        from PIL import Image

        img = Image.new("RGB", (2, 2), color=(128, 64, 32))
        p = tmp_path / "rgb.png"
        img.save(p)
        return str(p)

    @pytest.fixture()
    def depth_path(self, tmp_path):
        """Write a tiny 2x2 16-bit PNG (values in centimetres)."""
        import numpy as np
        from PIL import Image

        arr = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        img = Image.fromarray(arr, mode="I;16")
        p = tmp_path / "depth.png"
        img.save(p)
        return str(p)

    @pytest.fixture()
    def text_path(self, tmp_path):
        """Write a small VKITTI2-style intrinsics text file with header."""
        p = tmp_path / "intrinsic.txt"
        p.write_text(
            "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
            "0 0 725.0087 725.0087 620.5 187\n"
            "0 1 725.0087 725.0087 620.5 187\n"
        )
        return str(p)

    def test_rgb_returns_pil_image(self, rgb_path):
        from PIL import Image

        result = vkitti2.rgb(rgb_path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_depth_returns_float_array(self, depth_path):
        import numpy as np

        result = vkitti2.depth(depth_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_depth_converts_to_metres(self, depth_path):
        import numpy as np

        result = vkitti2.depth(depth_path)
        # 100 cm → 1.0 m, 200 cm → 2.0 m, etc.
        assert np.isclose(result[0, 0], 1.0)
        assert np.isclose(result[0, 1], 2.0)

    def test_class_segmentation_returns_pil_image(self, rgb_path):
        from PIL import Image

        result = vkitti2.class_segmentation(rgb_path)
        assert isinstance(result, Image.Image)

    def test_instance_segmentation_returns_pil_image(self, rgb_path):
        from PIL import Image

        result = vkitti2.instance_segmentation(rgb_path)
        assert isinstance(result, Image.Image)

    def test_scene_flow_returns_pil_image(self, rgb_path):
        from PIL import Image

        result = vkitti2.scene_flow(rgb_path)
        assert isinstance(result, Image.Image)

    def test_read_intrinsics_returns_dict(self, text_path):
        result = vkitti2.read_intrinsics(text_path)
        assert isinstance(result, dict)
        assert result == {
            "fx": 725.0087,
            "fy": 725.0087,
            "cx": 620.5,
            "cy": 187.0,
            "s": 0.0,
        }

    @pytest.fixture()
    def extrinsic_text_path(self, tmp_path):
        """Write a small whitespace-delimited numeric text file."""
        p = tmp_path / "extrinsic.txt"
        p.write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")
        return str(p)

    def test_read_extrinsics_returns_array(self, extrinsic_text_path):
        import numpy as np

        result = vkitti2.read_extrinsics(extrinsic_text_path)
        assert isinstance(result, np.ndarray)
