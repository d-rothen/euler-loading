"""Public accessors for consumers that write derived datasets (e.g. euler-preprocess crop)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from PIL import Image

from euler_loading import (
    Modality,
    MultiModalDataset,
    Resize,
    SamplePreprocessor,
    create_dataset_writer_from_index,
    get_modality_meta,
    resize_raster,
)
from euler_loading.loaders.cpu import princeton_dense as cpu
from euler_loading.loaders.gpu import princeton_dense as gpu
from euler_loading.preprocessing import FieldSpec

CAMERA_INFO = {
    "K": [2612.86, 0.0, 966.525, 0.0, 2612.86, 508.297, 0.0, 0.0, 1.0],
    "P": [2355.722801, 0.0, 988.138054, 0.0, 0.0, 2355.722801, 508.051838, 0.0, 0.0, 0.0, 1.0, 0.0],
    "width": 32,
    "height": 16,
}


def _writer(root, key, function, **meta):
    return DatasetWriter(
        root,
        name=f"stf_{key}",
        type=key,
        separator=None,
        euler_loading={"loader": "princeton_dense", "function": function},
        **({"meta": meta} if meta else {}),
    )


@pytest.fixture()
def stf(tmp_path):
    rgb = _writer(tmp_path / "rgb", "rgb", "rgb", range=[0, 255])
    lidar = _writer(tmp_path / "lidar", "sparse_depth", "sparse_depth")
    for index in range(2):
        Image.fromarray(np.full((16, 32, 3), 10 * index, np.uint8)).save(rgb.get_path(f"/f{index}", f"f{index}.png"))
        np.arange(10, dtype=np.float32).tofile(lidar.get_path(f"/f{index}", f"f{index}.bin"))
    rgb.save_index()
    lidar.save_index()
    calib = _writer(tmp_path / "calib", "intrinsics", "read_intrinsics")
    calib.get_path("/calib_cam_stereo_left", "calib_cam_stereo_left.json").write_text(json.dumps(CAMERA_INFO))
    calib.save_index()
    return MultiModalDataset(
        modalities={"rgb": Modality(str(tmp_path / "rgb")), "sparse_depth": Modality(str(tmp_path / "lidar"))},
        hierarchical_modalities={"intrinsics": Modality(str(tmp_path / "calib"))},
    )


def test_get_loader_returns_the_resolved_function(stf):
    loader = stf.get_loader("rgb")
    assert loader.__name__ == "rgb"
    assert stf.get_loader("intrinsics").__name__ == "read_intrinsics"
    with pytest.raises(KeyError, match="Unknown modality"):
        stf.get_loader("depth")


def test_get_loader_meta_returns_a_copy_of_the_annotation(stf):
    meta = stf.get_loader_meta("sparse_depth")
    assert (meta["type"], meta["shape"]) == ("sparse_depth", "NC")
    assert stf.get_loader_meta("intrinsics")["shape"] == "3x3"
    meta["shape"] = "changed"
    assert stf.get_loader_meta("sparse_depth")["shape"] == "NC"
    assert get_modality_meta(lambda path: None) is None


def test_get_hierarchical_files_keys_entries_by_level(stf):
    files = stf.get_hierarchical_files("intrinsics")
    assert list(files) == [()]
    assert [entry["id"] for entry in files[()]] == ["calib_cam_stereo_left"]
    files[()].clear()
    assert stf.get_hierarchical_files("intrinsics")[()]
    with pytest.raises(KeyError, match="not a hierarchical modality"):
        stf.get_hierarchical_files("rgb")


def test_writer_path_prefix_shares_one_scoped_root(stf, tmp_path):
    root = tmp_path / "bundle"
    for name in ("rgb", "sparse_depth"):
        writer = stf.create_output_writer(name, root, metadata_scope=name, path_prefix=name)
        sample = stf[0]
        writer.get_path(sample["full_id"], f"x{name}.dat").write_bytes(b"1")
        writer.save_index()
    assert (root / "rgb" / "xrgb.dat").is_file()
    assert (root / "sparse_depth" / "xsparse_depth.dat").is_file()
    assert sorted(json.loads((root / ".ds_crawler" / "scopes.json").read_text())["scopes"]) == ["rgb", "sparse_depth"]


def test_path_prefix_is_not_supported_for_materialized_outputs(stf, tmp_path):
    with pytest.raises(ValueError, match="path_prefix is not supported"):
        create_dataset_writer_from_index(
            index_output=stf.get_modality_index("rgb"),
            root=tmp_path / "m",
            derivation=object(),
            field="rgb",
            dataset_id="d",
            revision="r",
            path_prefix="rgb",
        )


@pytest.mark.parametrize("module", [cpu, gpu])
def test_stf_sparse_depth_writer_round_trips(module, tmp_path):
    points = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)
    path = tmp_path / "sub" / "points.bin"
    module.write_sparse_depth(str(path), points)
    assert module.write_point_cloud is module.write_sparse_depth
    np.testing.assert_array_equal(np.asarray(module.sparse_depth(str(path))), points)
    with pytest.raises(ValueError, match=r"shape \(N, 5\)"):
        module.write_sparse_depth(str(path), points[:, :4])


def test_stf_sparse_depth_writer_is_resolved_for_the_modality(stf):
    assert stf.get_writer("sparse_depth").__name__ == "write_sparse_depth"


def test_resize_raster_matches_the_resize_operation():
    torch = pytest.importorskip("torch")
    image = np.random.default_rng(1).random((3, 8, 12)).astype(np.float32)
    expected = SamplePreprocessor(
        operations=[Resize(size=(4, 6))],
        field_specs={"rgb": FieldSpec(kind="image", layout="CHW")},
        infer_fields=False,
    )({"rgb": torch.from_numpy(image)})["rgb"]
    resized = resize_raster(torch.from_numpy(image), layout="CHW", size=(4, 6), interpolation="bilinear")
    assert torch.equal(resized, expected)

    hwc = resize_raster(np.moveaxis(image, 0, -1), layout="HWC", size=(4, 6))
    np.testing.assert_allclose(hwc, np.moveaxis(expected.numpy(), 0, -1), atol=1e-6)


def test_resize_raster_nearest_keeps_values_and_dtypes():
    labels = np.arange(16, dtype=np.int64).reshape(4, 4)
    halved = resize_raster(labels, layout="HW", size=(2, 2), interpolation="nearest")
    assert halved.dtype == np.int64
    assert set(np.unique(halved)) <= set(np.unique(labels))
    mask = resize_raster(np.eye(4, dtype=bool), layout="HW", size=(8, 8), interpolation="nearest")
    assert mask.dtype == np.bool_
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        resize_raster(labels, layout="HW", size=(2, 2), interpolation="area")
    with pytest.raises(ValueError, match="Unsupported layout"):
        resize_raster(labels, layout="1HW", size=(2, 2))
