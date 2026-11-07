"""MUSES ego-motion inputs: GNSS records, sensor timing, lidar-to-GNSS mount."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from ds_crawler import DatasetWriter

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu import muses as cpu
from euler_loading.loaders.gpu import muses as gpu

# One line of gnss/dataformat.txt integers (synthetic values).
GNSS_LINE = (
    "123456 1676367100761000 25 85432100 473766700 512300 465400 3 17 850 1200 "
    "-120000 250000 9000000 50000 50000 120000 35000 -2000 947000 5800 -1200 "
    "30 5833 120 145 300 150\n"
)
FRAMES = {
    "REC0176_frame_670857": ("/test/fog/day/670857", 1676367085765123, 1676367085765803, 1676367085705051, 1676367085789506),
    "REC0176_frame_670890": ("/test/fog/day/670890", 1676367089038100, 1676367089038674, 1676367089004414, 1676367089089059),
}
LIDAR_TO_GNSS = [
    [0.999997568, -0.00134750821, 0.00174611242, 0.048],
    [-0.00134607756, -0.999998758, -0.00082024856, 0.286],
    [0.00174721554, 0.000817896162, -0.999998139, -0.039],
    [0.0, 0.0, 0.0, 1.0],
]


def _meta_json() -> dict:
    meta = {}
    for frame, (full_id, cam0, cam1, lid0, lid1) in FRAMES.items():
        split, weather, tod, _ = full_id.strip("/").split("/")
        sub = f"{split}/{weather}/{tod}/{frame}"
        meta[frame] = {
            "frame_camera_exposure_start_timestamp_us": str(cam0),
            "frame_camera_exposure_end_timestamp_us": str(cam1),
            "lidar_start_timestamp_us": str(lid0),
            "lidar_end_timestamp_us": str(lid1),
            "radar_start_timestamp_us": str(lid0 - 60000),
            "radar_end_timestamp_us": str(lid1 + 1000),
            "path_to_frame_camera": f"frame_camera/{sub}_frame_camera.png",
            "path_to_lidar": f"lidar/{sub}_lidar.bin",
            "path_to_gnss": f"gnss/{sub}_gnss.txt",
            "path_to_reference_frame": "None",
            "weather": weather,
        }
    return meta


def _index(root, key: str, function: str, files: dict[str, tuple[str, bytes]]) -> str:
    writer = DatasetWriter(
        root, name=f"muses_{key}", type=key, separator=None,
        euler_loading={"loader": "muses", "function": function},
    )
    for full_id, (basename, data) in files.items():
        writer.get_path(full_id, basename).write_bytes(data)
    writer.save_index()
    return str(root)


@pytest.fixture()
def muses_inputs(tmp_path):
    rng = np.random.default_rng(0)
    lidar, gnss_files = {}, {}
    points = {}
    for frame, (full_id, *_rest) in FRAMES.items():
        cloud = np.column_stack([rng.normal(size=(5, 5)), np.linspace(0, 0.08, 5) + 1676367085.7])
        stream = io.BytesIO()
        cpu.write_sparse_depth(stream, cloud)
        lidar[full_id] = (f"{frame}_lidar.bin", stream.getvalue())
        gnss_files[full_id] = (f"{frame}_gnss.txt", GNSS_LINE.encode())
        points[full_id] = cloud
    calib = json.dumps({"intrinsics": {}, "extrinsics": {"lidar2gnss": LIDAR_TO_GNSS, "lidar2rgb": np.eye(4).tolist()}}).encode()
    return {
        "lidar": _index(tmp_path / "lidar", "sparse_depth", "sparse_depth", lidar),
        "gnss": _index(tmp_path / "gnss", "gnss", "gnss", gnss_files),
        "timing": _index(tmp_path / "timing", "sensor_timing", "sensor_timing",
                         {"/meta": ("meta.json", json.dumps(_meta_json()).encode())}),
        "calib": _index(tmp_path / "calib", "lidar_to_gnss", "read_lidar_to_gnss",
                        {"/calib": ("calib.json", calib)}),
        "points": points,
    }


def test_gnss_record_in_physical_units(tmp_path):
    path = tmp_path / "REC0176_frame_671007_gnss.txt"
    path.write_text(GNSS_LINE, encoding="utf-8")
    record = cpu.gnss(str(path))
    assert record["timestamp_us"] == 1676367100761000
    assert record["latitude_deg"] == pytest.approx(47.37667)
    assert record["longitude_deg"] == pytest.approx(8.54321)
    assert record["height_m"] == pytest.approx(512.3)
    assert record["heading_deg"] == pytest.approx(90.0)
    assert record["heading_rate_deg_s"] == pytest.approx(9.47)
    assert record["pitch_rate_deg_s"] == pytest.approx(-0.02)
    assert record["ground_speed_m_s"] == pytest.approx(5.833)
    assert record["velocity_down_m_s"] == pytest.approx(0.03)
    assert record["position_dop"] == pytest.approx(1.45)
    assert gpu.gnss(str(path)) == record
    assert cpu.gnss(io.BytesIO(GNSS_LINE.encode())) == record


def test_gnss_rejects_a_truncated_record(tmp_path):
    path = tmp_path / "bad_gnss.txt"
    path.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="28 integers"):
        cpu.gnss(str(path))


def test_sensor_timing_table_and_file_index(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(_meta_json()), encoding="utf-8")
    table = cpu.sensor_timing(str(path))
    assert set(table) == set(FRAMES)
    record = table["REC0176_frame_670890"]
    assert record["camera_exposure_start_us"] == 1676367089038100
    assert record["lidar_end_us"] == 1676367089089059
    assert table.frame_for_file("lidar/x/REC0176_frame_670857_lidar.bin") == "REC0176_frame_670857"
    assert table.frame_for_file("REC0001_frame_000001_lidar.bin") is None


def test_sensor_timing_refuses_a_file_listed_by_two_frames(tmp_path):
    meta = _meta_json()
    first, second = list(meta)
    meta[second]["path_to_lidar"] = meta[first]["path_to_lidar"]
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot identify a frame"):
        cpu.sensor_timing(str(path))


def test_samples_receive_their_own_inputs(muses_inputs):
    ds = MultiModalDataset(
        modalities={"sparse_depth": Modality(muses_inputs["lidar"]), "gnss": Modality(muses_inputs["gnss"])},
        hierarchical_modalities={
            "sensor_timing": Modality(muses_inputs["timing"], collapse_single=True),
            "lidar_to_gnss": Modality(muses_inputs["calib"], collapse_single=True),
        },
    )
    assert len(ds) == 2
    for i in range(len(ds)):
        sample = ds[i]
        frame = "REC0176_frame_" + sample["id"]
        _full_id, cam0, cam1, lid0, lid1 = FRAMES[frame]
        assert sample["sensor_timing"] == {
            "frame": frame,
            "camera_exposure_start_us": cam0,
            "camera_exposure_end_us": cam1,
            "lidar_start_us": lid0,
            "lidar_end_us": lid1,
            "radar_start_us": lid0 - 60000,
            "radar_end_us": lid1 + 1000,
        }
        assert sample["gnss"]["ground_speed_m_s"] == pytest.approx(5.833)
        assert isinstance(sample["lidar_to_gnss"], torch.Tensor)
        np.testing.assert_allclose(sample["lidar_to_gnss"].numpy(), np.asarray(LIDAR_TO_GNSS), atol=1e-6)
        np.testing.assert_array_equal(sample["sparse_depth"].numpy(), muses_inputs["points"][sample["full_id"]])
    assert ds.get_writer("sparse_depth") is gpu.write_sparse_depth


def test_timing_selection_needs_native_file_names(tmp_path):
    lidar = _index(tmp_path / "lidar", "sparse_depth", "sparse_depth",
                   {"/test/fog/day/670857": ("670857.bin", np.zeros((1, 6)).tobytes())})
    timing = _index(tmp_path / "timing", "sensor_timing", "sensor_timing",
                    {"/meta": ("meta.json", json.dumps(_meta_json()).encode())})
    ds = MultiModalDataset(
        modalities={"sparse_depth": Modality(lidar)},
        hierarchical_modalities={"sensor_timing": Modality(timing)},
    )
    with pytest.raises(KeyError, match="native MUSES file names"):
        ds[0]


def test_lidar_to_gnss_reads_its_own_transform(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"extrinsics": {"lidar2gnss": LIDAR_TO_GNSS, "lidar2rgb": np.eye(4).tolist()}}))
    result = cpu.read_lidar_to_gnss(str(path), {"transform": "lidar2rgb"})
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.asarray(LIDAR_TO_GNSS), atol=1e-6)
    assert torch.equal(gpu.read_lidar_to_gnss(str(path)), torch.from_numpy(result))


def test_sparse_depth_writer_round_trips(tmp_path):
    cloud = np.arange(18, dtype=np.float64).reshape(3, 6)
    path = tmp_path / "out" / "REC0001_frame_000001_lidar.bin"
    gpu.write_sparse_depth(str(path), torch.from_numpy(cloud))
    np.testing.assert_array_equal(cpu.sparse_depth(str(path)), cloud)
    assert cpu.write_lidar_point_cloud is cpu.write_sparse_depth
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        cpu.write_sparse_depth(str(path), np.zeros((3, 5)))
