"""Tests for sample-level preprocessing utilities."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from euler_loading import Modality, MultiModalDataset
from euler_loading.preprocessing import SamplePreprocessor


def _flat_index(modality: str, file_ids: list[str]) -> dict[str, object]:
    return {
        "dataset": {
            "files": [
                {
                    "id": file_id,
                    "path": f"{file_id}.{modality}",
                    "path_properties": {},
                    "basename_properties": {},
                }
                for file_id in file_ids
            ]
        }
    }


class TestSamplePreprocessorTorch:
    def test_resize_updates_modalities_intrinsics_and_rays(self):
        rays = torch.randn(3, 4, 6, dtype=torch.float32)
        rays = torch.nn.functional.normalize(rays, dim=0)

        sample = {
            "rgb": torch.linspace(0.0, 1.0, steps=3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6),
            "metric_depth": torch.arange(24, dtype=torch.float32).reshape(1, 4, 6),
            "valid_mask": torch.tensor(
                [[[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]]],
                dtype=torch.bool,
            ),
            "intrinsics": torch.tensor(
                [[6.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            "ray_map_gt": rays,
        }

        preprocessor = SamplePreprocessor.from_config({"resize": [2, 3]})
        processed = preprocessor(sample)

        assert processed["rgb"].shape == (3, 2, 3)
        assert processed["metric_depth"].shape == (1, 2, 3)
        assert processed["valid_mask"].shape == (1, 2, 3)
        assert processed["valid_mask"].dtype == torch.bool
        assert processed["ray_map_gt"].shape == (3, 2, 3)

        expected_intrinsics = torch.tensor(
            [[3.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(processed["intrinsics"], expected_intrinsics)

        norms = torch.linalg.norm(processed["ray_map_gt"], dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)

    def test_reduce_first_handles_hierarchical_intrinsics(self):
        sample = {
            "rgb": torch.zeros((3, 4, 6), dtype=torch.float32),
            "intrinsics": {
                "b": torch.tensor(
                    [[10.0, 0.0, 5.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                "a": torch.tensor(
                    [[6.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
            },
        }

        preprocessor = SamplePreprocessor.from_config(
            {
                "resize": [2, 3],
                "fields": {
                    "intrinsics": {"kind": "intrinsics", "reduce": "first"},
                },
            }
        )
        processed = preprocessor(sample)

        expected = torch.tensor(
            [[3.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(processed["intrinsics"], expected)


class TestSamplePreprocessorNumpy:
    def test_crop_uses_shared_box_and_updates_intrinsics(self):
        rgb = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        depth = np.arange(24, dtype=np.float32).reshape(4, 6)
        rays = np.ones((4, 6, 3), dtype=np.float32)

        sample = {
            "rgb": rgb,
            "metric_depth": depth,
            "ray_map_precomputed": rays,
            "valid_mask": np.array(
                [
                    [True, False, True, False, True, False],
                    [False, True, False, True, False, True],
                    [True, True, False, False, True, True],
                    [False, False, True, True, False, False],
                ],
                dtype=np.bool_,
            ),
            "intrinsics": np.array(
                [[6.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        }

        preprocessor = SamplePreprocessor.from_config(
            {"operations": [{"type": "crop", "size": [2, 4], "anchor": "bottom_right"}]}
        )
        processed = preprocessor(sample)

        assert processed["rgb"].shape == (2, 4, 3)
        assert processed["metric_depth"].shape == (2, 4)
        assert processed["valid_mask"].shape == (2, 4)
        assert processed["valid_mask"].dtype == np.bool_
        assert processed["ray_map_precomputed"].shape == (2, 4, 3)
        assert np.array_equal(processed["rgb"], rgb[2:4, 2:6, :])
        assert np.array_equal(processed["metric_depth"], depth[2:4, 2:6])

        expected_intrinsics = np.array(
            [[6.0, 0.0, 0.5], [0.0, 4.0, -0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        assert np.allclose(processed["intrinsics"], expected_intrinsics)


class TestDatasetTransformBinding:
    def test_dataset_binds_modality_types_into_preprocessor(self):
        rgb_index = _flat_index("rgb", ["f001"])
        depth_index = _flat_index("depth", ["f001"])

        def mock_index(path, **kwargs):
            return rgb_index if "rgb" in path else depth_index

        def rgb_loader(path, meta=None):
            return torch.ones((3, 4, 6), dtype=torch.float32)

        def depth_loader(path, meta=None):
            return torch.ones((1, 4, 6), dtype=torch.float32)

        preprocessor = SamplePreprocessor.from_config(
            {"resize": [2, 3], "infer_fields": False}
        )

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            side_effect=mock_index,
        ):
            dataset = MultiModalDataset(
                modalities={
                    "rgb": Modality(
                        "/data/rgb",
                        loader=rgb_loader,
                        modality_type="rgb",
                    ),
                    "metric_depth": Modality(
                        "/data/depth",
                        loader=depth_loader,
                        modality_type="depth",
                    ),
                },
                transforms=[preprocessor],
            )

        sample = dataset[0]
        assert sample["rgb"].shape == (3, 2, 3)
        assert sample["metric_depth"].shape == (1, 2, 3)

    def test_dataset_logs_transform_behaviour_once_on_init(self, caplog):
        rgb_index = _flat_index("rgb", ["f001"])

        def rgb_loader(path, meta=None):
            return torch.ones((3, 4, 6), dtype=torch.float32)

        def identity(sample):
            return sample

        preprocessor = SamplePreprocessor.from_config(
            {
                "resize": [2, 3],
                "infer_fields": False,
                "fields": {
                    "valid_mask": {"kind": "mask"},
                    "intrinsics": {"kind": "intrinsics", "reduce": "first"},
                },
            }
        )

        with patch(
            "euler_loading.dataset.index_dataset_from_path",
            return_value=rgb_index,
        ), caplog.at_level(logging.INFO, logger="euler_loading.dataset"):
            MultiModalDataset(
                modalities={
                    "rgb": Modality(
                        "/data/rgb",
                        loader=rgb_loader,
                        modality_type="rgb",
                    ),
                },
                transforms=[preprocessor, identity],
            )

        messages = [record.message for record in caplog.records]
        assert messages.count("Configured 2 sample transform(s):") == 1

        joined = "\n".join(messages)
        assert "SamplePreprocessor(" in joined
        assert "resize(2x3)" in joined
        assert "rgb: kind=image" in joined
        assert "valid_mask: kind=mask" in joined
        assert "interpolation=nearest" in joined
        assert "intrinsics: kind=intrinsics" in joined
        assert "reduce=first" in joined
        assert "identity" in joined
