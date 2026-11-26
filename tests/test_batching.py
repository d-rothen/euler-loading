"""Batches of MultiModalDataset samples (collate, dataloader) and generic readers."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from euler_loading import (
    BatchError,
    Modality,
    MultiModalDataset,
    collate,
    dataloader,
    generic_reader,
)


def _sample(index: int, **values: Any) -> dict[str, Any]:
    return {
        **values,
        "id": f"f{index:03d}",
        "full_id": f"/scene/f{index:03d}",
        "meta": {"rgb": {"id": f"f{index:03d}", "path": f"scene/f{index:03d}.png"}},
        "attributes": {"rgb": {"exposure": index}},
    }


def test_arrays_are_stacked_and_metadata_stays_per_sample():
    samples = [
        _sample(i, rgb=np.full((3, 4, 5), i, np.float32), depth=np.full((1, 4, 5), i, np.uint16))
        for i in range(3)
    ]
    batch = collate(samples)
    assert list(batch) == ["rgb", "depth", "id", "full_id", "meta", "attributes"]
    assert isinstance(batch["rgb"], np.ndarray) and batch["rgb"].shape == (3, 3, 4, 5)
    assert batch["depth"].dtype == np.uint16  # dtype kept
    assert batch["rgb"][2].max() == 2
    assert batch["id"] == ["f000", "f001", "f002"]
    assert batch["full_id"][1] == "/scene/f001"
    assert batch["meta"][2]["rgb"]["path"] == "scene/f002.png"
    assert [item["rgb"]["exposure"] for item in batch["attributes"]] == [0, 1, 2]


def test_torch_tensors_are_stacked_as_tensors():
    torch = pytest.importorskip("torch")
    samples = [_sample(i, rgb=torch.rand(3, 4, 5), mask=np.ones((4, 5), bool)) for i in range(2)]
    samples[1]["mask"] = torch.zeros(4, 5, dtype=torch.bool)  # a mix becomes a tensor
    batch = collate(samples)
    assert torch.is_tensor(batch["rgb"]) and batch["rgb"].shape == (2, 3, 4, 5)
    assert torch.is_tensor(batch["mask"]) and batch["mask"].dtype == torch.bool
    assert batch["mask"][0].all() and not batch["mask"][1].any()


def test_a_missing_modality_stays_visible_as_none():
    samples = [
        _sample(0, rgb=np.zeros((3, 2, 2)), intrinsics=np.eye(3)),
        _sample(1, rgb=np.zeros((3, 2, 2)), intrinsics=None),
        _sample(2, rgb=np.zeros((3, 2, 2))),  # key absent
    ]
    batch = collate(samples)
    assert batch["rgb"].shape == (3, 3, 2, 2)
    assert isinstance(batch["intrinsics"], list)
    assert batch["intrinsics"][1] is None and batch["intrinsics"][2] is None
    assert np.array_equal(batch["intrinsics"][0], np.eye(3))


def test_dicts_strings_and_numbers_are_lists():
    samples = [
        _sample(i, calibration={"cam_0": np.eye(3)}, scene="a", weight=0.5 * i) for i in range(2)
    ]
    batch = collate(samples)
    assert batch["calibration"][1]["cam_0"].shape == (3, 3)
    assert batch["scene"] == ["a", "a"]
    assert batch["weight"] == [0.0, 0.5]


def test_different_shapes_raise_with_a_fix():
    samples = [_sample(0, rgb=np.zeros((3, 4, 4))), _sample(1, rgb=np.zeros((3, 8, 8)))]
    with pytest.raises(BatchError, match=r"cannot batch 'rgb'.*\[3, 4, 4\], \[3, 8, 8\].*Resize"):
        collate(samples)


def test_empty_and_non_sample_input_raise():
    with pytest.raises(BatchError, match="empty"):
        collate([])
    with pytest.raises(BatchError, match="dicts"):
        collate([np.zeros(3)])


def _array_loader(path: str, meta: Any = None) -> np.ndarray:
    return np.full((1, 2, 3), float(path.rsplit("f", 1)[-1].split(".")[0]), np.float32)


def _dataset() -> MultiModalDataset:
    index = {
        "dataset": {
            "files": [
                {"id": f"f{i:03d}", "path": f"f{i:03d}.npy", "path_properties": {}, "basename_properties": {}}
                for i in range(5)
            ]
        }
    }
    with patch("euler_loading.dataset.index_dataset_from_path", return_value=index):
        return MultiModalDataset(
            modalities={
                "depth": Modality("/data/depth", loader=_array_loader),
                "rgb": Modality("/data/rgb", loader=_array_loader),
            }
        )


def test_collate_batches_real_dataset_samples():
    dataset = _dataset()
    batch = collate([dataset[0], dataset[3]])
    assert batch["depth"].shape == (2, 1, 2, 3)
    assert batch["depth"][1].max() == 3.0
    assert batch["id"] == ["f000", "f003"]
    assert set(batch["meta"][0]) == {"depth", "rgb"}


def test_dataloader_uses_collate():
    torch = pytest.importorskip("torch")
    loader = dataloader(_dataset(), batch_size=2, shuffle=False)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batches = list(loader)
    assert [len(batch["id"]) for batch in batches] == [2, 2, 1]
    assert batches[0]["rgb"].shape == (2, 1, 2, 3)
    with pytest.raises(TypeError, match="collate_fn"):
        dataloader(_dataset(), collate_fn=lambda items: items)


@pytest.mark.parametrize(
    ("key", "file_type", "expected"),
    [
        ("depth", "npy", {"loader": "generic_dense_depth", "function": "depth"}),
        ("depth", ".png", {"loader": "generic_dense_depth", "function": "depth"}),
        ("rgb", "png", {"loader": "generic_dense_depth", "function": "rgb"}),
        ("points_3d", "npy", {"loader": "generic", "function": "points_3d"}),
        ("semantic_segmentation", "npy", {"loader": "generic", "function": "semantic_segmentation"}),
        ("intrinsics", "npy", {"loader": "generic", "function": "intrinsics"}),
        ("sky_mask", "png", None),  # needs the head's skyclass
        ("depth", "exr", None),
        ("thermal", "npy", None),
    ],
)
def test_generic_reader(key, file_type, expected):
    assert generic_reader(key, file_type) == expected
