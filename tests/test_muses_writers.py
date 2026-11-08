"""MUSES writers: every image and calibration format round-trips through its loader."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from ds_crawler import DatasetWriter
from PIL import Image

from euler_loading import Modality, MultiModalDataset, crop_intrinsics, resize_intrinsics
from euler_loading.loaders.cpu import muses as cpu
from euler_loading.loaders.gpu import muses as gpu

# Synthetic values (not MUSES calibration) in the layout of MUSES calib.json:
# nested lists, integer zeros and ones, full float64 precision.
CALIB = {
    "intrinsics": {
        "rgb": {
            "K": [[1043.1234567890123, 0, 951.7654321098765], [0, 1043.9876543210987, 537.1357913579135], [0, 0, 1]],
            "D": [-0.25, 0.125, 0.001, -0.002, -0.03125],
        },
        "event": {
            "K": [[1020.5, 0, 640.25], [0, 1021.75, 360.5], [0, 0, 1]],
            "D": [-0.375, 0.1875, 0.0, 0.0, -0.0625],
        },
    },
    "extrinsics": {
        "lidar2rgb": [[0.0, -1.0, 0.0, 0.1], [0.0, 0.0, -1.0, -0.05], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 0.0, 1.0]],
        "lidar2gnss": [[1.0, 0.0, 0.0, 0.05], [0.0, -1.0, 0.0, 0.3], [0.0, 0.0, -1.0, -0.04], [0.0, 0.0, 0.0, 1.0]],
    },
}


@pytest.fixture()
def calib(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(CALIB, indent=2) + "\n")
    return path


@pytest.mark.parametrize("module", [cpu, gpu])
def test_rgb_round_trips(module, tmp_path):
    pixels = np.random.default_rng(0).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    source = tmp_path / "REC0001_frame_000001_frame_camera.png"
    Image.fromarray(pixels).save(source)
    value = module.rgb(str(source))
    target = tmp_path / "out" / source.name
    module.write_rgb(str(target), value)
    assert np.array_equal(np.asarray(Image.open(target)), pixels)
    reloaded = module.rgb(str(target))
    assert np.array_equal(np.asarray(reloaded), np.asarray(value))
    assert module.write_reference_rgb is module.write_rgb


def test_label_writers_round_trip(tmp_path):
    labels = np.array([[0, 7, 23], [33, 255, 11]], dtype=np.int64)
    path = tmp_path / "x_gt_labelIds.png"
    gpu.write_semantic_segmentation(str(path), torch.from_numpy(labels).unsqueeze(0))
    assert Image.open(path).mode == "L"
    assert np.array_equal(cpu.semantic_segmentation(str(path)), labels)
    cpu.write_semantic_segmentation(str(path), labels.astype(np.float32))
    assert np.array_equal(cpu.semantic_segmentation(str(path)), labels)
    with pytest.raises(ValueError, match=r"0\.\.255"):
        cpu.write_semantic_segmentation(str(path), labels + 1)
    with pytest.raises(ValueError, match="integer"):
        cpu.write_semantic_segmentation(str(path), labels + 0.5)

    colors = np.random.default_rng(1).integers(0, 256, size=(2, 4, 3), dtype=np.uint8)
    path = tmp_path / "x_gt_labelColor.png"
    gpu.write_semantic_segmentation_color(str(path), torch.from_numpy(colors).permute(2, 0, 1))
    assert np.array_equal(cpu.semantic_segmentation_color(str(path)), colors)


def test_panoptic_writer_round_trips_24_bit_ids(tmp_path):
    ids = np.array([[0, 1, 255], [256, 65536, 2**24 - 1]], dtype=np.int64)
    path = tmp_path / "x_gt_panoptic.png"
    gpu.write_panoptic_segmentation(str(path), torch.from_numpy(ids).unsqueeze(0))
    assert torch.equal(gpu.panoptic_segmentation(str(path)), torch.from_numpy(ids).unsqueeze(0))
    with pytest.raises(ValueError, match="0..16777215"):
        cpu.write_panoptic_segmentation(str(path), ids + 1)


@pytest.mark.parametrize(
    ("name", "meta", "stored"),
    [
        ("x_gt_labelIds.png", None, {True: 23, False: 0}),
        ("x_gt_labelTrainIds.png", None, {True: 10, False: 0}),
        ("x_gt_labelIds.png", {"skyclass": [0, 0, 0]}, {True: 0, False: 255}),
        ("x_gt_labelIds.png", {"skyclass": [23, 23, 23], "non_sky_class_id": 255}, {True: 23, False: 255}),
    ],
)
def test_sky_mask_writer_uses_the_loaders_sky_label(tmp_path, name, meta, stored):
    mask = np.array([[True, False], [False, True]])
    path = tmp_path / name
    gpu.write_sky_mask(str(path), torch.from_numpy(mask).unsqueeze(0), meta)
    data = np.asarray(Image.open(path))
    assert data.ndim == 2
    assert data[0, 0] == stored[True] and data[0, 1] == stored[False]
    assert np.array_equal(cpu.sky_mask(str(path), meta), mask)


def test_sky_mask_writer_writes_a_colour_for_rgb_skyclass(tmp_path):
    mask = np.array([[True, False]])
    path = tmp_path / "x_gt_labelColor.png"
    cpu.write_sky_mask(str(path), mask, {"skyclass": [70, 130, 180]})
    assert np.asarray(Image.open(path)).tolist() == [[[70, 130, 180], [0, 0, 0]]]
    assert np.array_equal(cpu.sky_mask(str(path), {"skyclass": [70, 130, 180]}), mask)
    with pytest.raises(ValueError, match="equals the sky label"):
        cpu.write_sky_mask(str(tmp_path / "y_gt_labelIds.png"), mask, {"non_sky_class_id": 23})


def test_intrinsics_writer_applies_a_crop_exactly_and_keeps_the_rest(calib):
    K = gpu.read_intrinsics(str(calib))
    gpu.write_intrinsics(str(calib), crop_intrinsics(K, top=34, left=16), {"width": 1904, "height": 1046})
    written = json.loads(calib.read_text())
    stored = written["intrinsics"]["rgb"]["K"]
    source = CALIB["intrinsics"]["rgb"]["K"]
    # Exact float64 shifts, not float32 round trips; untouched entries keep their JSON ints.
    assert stored[0][2] == source[0][2] - 16 and stored[1][2] == source[1][2] - 34
    assert stored[0][0] == source[0][0] and stored[1][1] == source[1][1]
    assert stored[0][1] == 0 and isinstance(stored[0][1], int) and stored[2] == [0, 0, 1]
    assert written["intrinsics"]["rgb"]["D"] == CALIB["intrinsics"]["rgb"]["D"]
    assert written["intrinsics"]["event"] == CALIB["intrinsics"]["event"]
    assert written["extrinsics"] == CALIB["extrinsics"]
    assert "width" not in written["intrinsics"]["rgb"]


def test_intrinsics_writer_leaves_an_unchanged_matrix_untouched(calib):
    before = calib.read_bytes()
    K = cpu.read_intrinsics(str(calib))
    # A bottom-only crop keeps the principal point: K is unchanged.
    cpu.write_intrinsics(str(calib), crop_intrinsics(K, top=0, left=0), {"width": 1920, "height": 920})
    assert calib.read_bytes() == before


def test_intrinsics_writer_applies_a_rescale_and_a_selected_sensor(calib):
    K = cpu.read_intrinsics(str(calib), {"sensor": "event"})
    half = resize_intrinsics(K, source_size=(720, 1280), target_size=(360, 640))
    cpu.write_intrinsics(str(calib), half, {"sensor": "event"})
    np.testing.assert_allclose(cpu.read_intrinsics(str(calib), {"sensor": "event"}), half, atol=1e-4)
    assert json.loads(calib.read_text())["intrinsics"]["rgb"] == CALIB["intrinsics"]["rgb"]


def test_intrinsics_writer_refuses_what_it_cannot_merge(calib, tmp_path):
    K = cpu.read_intrinsics(str(calib))
    sheared = K.copy()
    sheared[0, 1] = 3.0
    with pytest.raises(ValueError, match="not a crop or rescale"):
        cpu.write_intrinsics(str(calib), sheared)
    with pytest.raises(ValueError, match="no intrinsics for sensor 'thermal'"):
        cpu.write_intrinsics(str(calib), K, {"sensor": "thermal"})
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        cpu.write_intrinsics(str(broken), K)
    assert json.loads(calib.read_text()) == CALIB


def test_intrinsics_writer_creates_a_minimal_calibration(tmp_path):
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    path = tmp_path / "new" / "calib.json"
    cpu.write_intrinsics(str(path), K)
    assert json.loads(path.read_text()) == {"intrinsics": {"rgb": {"K": K.tolist()}}}
    stream = io.BytesIO()
    cpu.write_intrinsics(stream, K, {"sensor": "event"})
    assert json.loads(stream.getvalue()) == {"intrinsics": {"event": {"K": K.tolist()}}}


def test_writers_resolve_for_muses_modalities(tmp_path, calib):
    def index(root, key, function, basename, data, **meta):
        writer = DatasetWriter(
            root, name=f"muses_{key}", type=key, separator=None,
            euler_loading={"loader": "muses", "function": function},
            **({"meta": meta} if meta else {}),
        )
        writer.get_path(f"/test/fog/day/{basename.split('.')[0]}", basename).write_bytes(data)
        writer.save_index()
        return str(root)

    image = io.BytesIO()
    Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(image, format="PNG")
    labels = io.BytesIO()
    Image.fromarray(np.zeros((4, 6), np.uint8)).save(labels, format="PNG")
    calib_root = tmp_path / "calib"
    writer = DatasetWriter(
        calib_root, name="muses_calib", type="intrinsics", separator=None,
        euler_loading={"loader": "muses", "function": "read_intrinsics"},
    )
    writer.get_path("/calib", "calib.json").write_bytes(calib.read_bytes())
    writer.save_index()
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(index(tmp_path / "rgb", "rgb", "rgb", "000001.png", image.getvalue(), range=[0, 255])),
            "semantic": Modality(
                index(
                    tmp_path / "sem", "semantic_segmentation", "semantic_segmentation", "000001.png",
                    labels.getvalue(), skyclass=[23, 23, 23],
                )
            ),
        },
        hierarchical_modalities={"intrinsics": Modality(str(calib_root))},
    )
    assert dataset.get_writer("rgb") is gpu.write_rgb
    assert dataset.get_writer("semantic") is gpu.write_semantic_segmentation
    assert dataset.get_writer("intrinsics") is gpu.write_intrinsics
