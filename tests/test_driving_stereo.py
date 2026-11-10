"""DrivingStereo decoding, half-size left-image alignment and dataset integration."""

from __future__ import annotations

import importlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from euler_loading import Modality, MultiModalDataset, resolve_loader_module
from euler_loading.loaders.cpu import driving_stereo as cpu_ds
from euler_loading.loaders.generate.__main__ import generate
from euler_loading.loaders.gpu import driving_stereo as gpu_ds

# calibration/train/half-image-calib/2018-10-25-07-37-26.txt as released, the
# calibration of the foggy weather subset.
HALF_CALIB = """\
calib_time: 20180820
corner_dist: 1.000000e-01
S_101: 9.600000e+2 6.000000e+2
K_101: 2.061940e+03 5.499030e-01 9.719003e+02 0.000000e+00 2.060674e+03 5.797886e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_101: -9.138600e-02 1.127830e-01 1.650000e-04 9.200000e-05
R_101: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
T_101: 0.000000e+00 0.000000e+00 0.000000e+00
S_rect_101: 8.790000e+2 4.000000e+2
P_rect_101: 1.001496e+3 0.000000e+0 4.526664e+2 0.000000e+0 0.000000e+0 1.001496e+3 1.942570e+2 0.000000e+0 0.000000e+0 0.000000e+0 1.000000e+0 0.000000e+0
R_rect_101: 9.995644e-01 2.191951e-02 -1.976031e-02 -2.198217e-02 9.997540e-01 -2.959274e-03 1.969058e-02 3.392360e-03 9.998004e-01
S_103: 9.600000e+2 6.000000e+2
K_103: 2.061268e+03 -5.951140e-01 9.684596e+02 0.000000e+00 2.060783e+03 5.868436e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_103: -9.292800e-02 1.271090e-01 -6.620000e-04 -4.950000e-04
R_103: 9.994810e-01 2.160800e-02 -2.390500e-02 -2.176000e-02 9.997440e-01 -6.141000e-03 2.376600e-02 6.658000e-03 9.996950e-01
T_103: -5.449150e-01 -1.550000e-04 -2.260000e-03
S_rect_103: 8.790000e+2 4.000000e+2
P_rect_103: 1.001496e+3 0.000000e+0 4.526664e+2 -1.091469e+3 0.000000e+0 1.001496e+3 1.942570e+2 0.000000e+0 0.000000e+0 0.000000e+0 1.000000e+0 0.000000e+0
R_rect_103: 9.999914e-01 2.844456e-04 4.147400e-03 -2.976170e-04 9.999949e-01 3.175553e-03 -4.146476e-03 -3.176760e-03 9.999864e-01
"""
K_HALF = np.array([[1001.496, 0.0, 452.6664], [0.0, 1001.496, 194.257], [0.0, 0.0, 1.0]])
SEQUENCE = "2018-10-25-07-37-26"
FRAMES = ("2018-10-25-07-41-58-446", "2018-10-25-07-42-00-537")


@pytest.fixture(params=["cpu", "gpu"])
def loaders(request):
    return importlib.import_module(f"euler_loading.loaders.{request.param}.driving_stereo")


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


def _image(result, loaders):
    """Return an image result in HW(C) layout, checking the variant's layout."""
    if loaders is gpu_ds:
        assert isinstance(result, torch.Tensor) and result.is_contiguous()
        arr = result.numpy()
        return arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
    assert isinstance(result, np.ndarray) and result.flags.c_contiguous
    return result


def _png16(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(raw, dtype=np.uint16)).save(path)
    return path


def _calib(path, text=HALF_CALIB):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_surface_and_resolution():
    from euler_loading.loaders import driving_stereo as top

    assert resolve_loader_module("driving_stereo") is gpu_ds
    assert resolve_loader_module("driving_stereo", variant="cpu") is cpu_ds
    assert top.depth is gpu_ds.depth
    assert gpu_ds.__all__ == cpu_ds.__all__
    for name in cpu_ds.__all__:
        assert hasattr(gpu_ds, name) and hasattr(top, name), name
    # Writers are shared, so the two variants stay in step by construction.
    for name in (n for n in cpu_ds.__all__ if n.startswith("write_")):
        assert getattr(gpu_ds, name) is getattr(cpu_ds, name)


def test_writer_resolves_for_every_reader():
    from euler_loading import resolve_writer_module
    from euler_loading._resolution import _writer_name_candidates

    module = resolve_writer_module("driving_stereo")
    for reader in (n for n in cpu_ds.__all__ if not n.startswith("write_")):
        candidates = _writer_name_candidates(reader, None)
        assert any(callable(getattr(module, name, None)) for name in candidates), reader


def test_inventory_matches_annotations():
    generated = {item["name"]: item for item in generate()["supportedLoaders"]}
    checked_in = json.loads(
        (Path(__file__).parents[1] / "euler_loading/loaders/generate/loaders.json").read_text()
    )
    saved = {item["name"]: item for item in checked_in["supportedLoaders"]}
    assert generated["driving_stereo"] == saved["driving_stereo"]
    modalities = {item["function"]: item for item in generated["driving_stereo"]["modalities"]}
    assert set(modalities) == {
        "rgb", "depth", "depth_full_size", "disparity", "disparity_full_size", "read_intrinsics",
    }
    assert modalities["depth"]["meta"]["radial_depth"] is False
    assert modalities["depth_full_size"]["meta"]["aligned_to"] == "left_image_half_size"
    assert modalities["read_intrinsics"]["hierarchical"] is True
    for name, cpu_shape, gpu_shape in [
        ("rgb", "HWC", "CHW"), ("depth", "HW", "1HW"), ("depth_full_size", "HW", "1HW"),
        ("disparity", "HW", "1HW"), ("disparity_full_size", "HW", "1HW"),
        ("read_intrinsics", "3x3", "3x3"),
    ]:
        assert modalities[name]["cpu"]["shape"] == cpu_shape
        assert modalities[name]["gpu"]["shape"] == gpu_shape
        cpu_meta = dict(getattr(cpu_ds, name)._modality_meta)
        gpu_meta = dict(getattr(gpu_ds, name)._modality_meta)
        cpu_meta.pop("shape")
        gpu_meta.pop("shape")
        assert cpu_meta == gpu_meta, name


# ---------------------------------------------------------------------------
# Images and half-size maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_rgb(tmp_path, loaders, source, suffix):
    pixels = np.full((4, 6, 3), (10, 128, 250), np.uint8)
    path = tmp_path / f"frame{suffix}"
    Image.fromarray(pixels).save(path, quality=100)
    result = _image(loaders.rgb(source(path)), loaders)
    assert result.shape == (4, 6, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result, pixels / 255.0, atol=3 / 255)


def test_half_size_maps_decode_by_256(tmp_path, loaders, source):
    raw = np.array([[0, 256, 30206], [65535, 1, 7]], np.uint16)
    depth_path = _png16(tmp_path / "depth.png", raw)
    depth = _image(loaders.depth(source(depth_path)), loaders)
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth, raw / 256.0)
    disparity = _image(loaders.disparity(source(_png16(tmp_path / "disp.png", raw))), loaders)
    np.testing.assert_array_equal(disparity, raw / 256.0)


def test_maps_must_be_16_bit(tmp_path):
    path = tmp_path / "depth.png"
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(path)
    with pytest.raises(ValueError, match="16-bit"):
        cpu_ds.depth(str(path))


# ---------------------------------------------------------------------------
# Full-size maps on the half-size left-image grid
# ---------------------------------------------------------------------------


def _full(shape, samples):
    full = np.zeros(shape, np.uint16)
    for (y, x), value in samples.items():
        full[y, x] = value
    return full


def test_full_size_depth_takes_the_co_located_sample(tmp_path, loaders):
    # Half-size pixel (1, 1) sits on full-size (2, 2).
    full = _full((6, 8), {(2, 2): 2560, (2, 3): 256, (1, 2): 512, (3, 3): 768})
    result = _image(loaders.depth_full_size(str(_png16(tmp_path / "d.png", full))), loaders)
    assert result.shape == (3, 4)
    assert result[1, 1] == 10.0


def test_full_size_depth_falls_back_to_the_nearest_ring(tmp_path):
    full = _full((8, 10), {
        # (1, 1): two direct neighbours, the nearer surface wins over a nearer
        # diagonal one.
        (1, 2): 2560, (2, 3): 1280, (1, 1): 256,
        # (3, 3): only a diagonal neighbour.
        (7, 7): 768,
    })
    result = cpu_ds.depth_full_size(str(_png16(tmp_path / "d.png", full)))
    assert result[1, 1] == 5.0
    assert result[3, 3] == 3.0
    # Pixel (0, 0) reaches (1, 1) diagonally; (2, 4) sees nothing.
    assert result[0, 0] == 1.0
    assert result[2, 4] == 0.0


def test_full_size_subsample_keeps_only_the_co_located_samples(tmp_path):
    full = _full((4, 4), {(0, 0): 256, (1, 2): 512, (2, 2): 768})
    path = str(_png16(tmp_path / "d.png", full))
    expected = np.array([[1.0, 0.0], [0.0, 3.0]], np.float32)
    np.testing.assert_array_equal(
        cpu_ds.depth_full_size(path, {"full_size_resampling": "subsample"}), expected
    )
    # Per-file attributes take precedence over the head's meta.
    np.testing.assert_array_equal(
        cpu_ds.depth_full_size(path, {"full_size_resampling": "nearest"},
                               attributes={"full_size_resampling": "SubSample"}),
        expected,
    )
    with pytest.raises(ValueError, match="full_size_resampling"):
        cpu_ds.depth_full_size(path, {"full_size_resampling": "bilinear"})


def test_full_size_maps_with_odd_sizes(tmp_path):
    full = _full((5, 7), {(4, 6): 256, (3, 5): 512})
    result = cpu_ds.depth_full_size(str(_png16(tmp_path / "d.png", full)))
    assert result.shape == (3, 4)
    assert result[2, 3] == 1.0
    assert result[1, 2] == 2.0  # (3, 5) is diagonal to (2, 4)


def test_full_size_disparity_is_halved_and_prefers_the_nearer_surface(tmp_path, loaders):
    # Full-size disparity is uint16 / 128 in full-size pixels.
    full = _full((4, 6), {(0, 0): 1280, (1, 2): 1280, (2, 3): 2560})
    result = _image(loaders.disparity_full_size(str(_png16(tmp_path / "p.png", full))), loaders)
    assert result.shape == (2, 3)
    assert result[0, 0] == 5.0           # 1280 / 128 / 2
    assert result[1, 1] == 10.0          # the larger disparity of (1, 2) and (2, 3)


def test_depth_and_disparity_agree_after_resampling(tmp_path):
    rng = np.random.default_rng(0)
    focal_baseline = 1091.469  # full-size fx * baseline, in full-size pixels * metres
    depth_m = rng.uniform(5.0, 100.0, (40, 60))
    valid = rng.random((40, 60)) < 0.1
    full_depth = np.where(valid, np.rint(depth_m * 256), 0)
    full_disp = np.where(valid, np.rint(focal_baseline / (full_depth / 256 + ~valid) * 128), 0)
    depth = cpu_ds.depth_full_size(str(_png16(tmp_path / "d.png", full_depth)))
    disparity = cpu_ds.disparity_full_size(str(_png16(tmp_path / "p.png", full_disp)))
    assert ((depth > 0) == (disparity > 0)).all() and (depth > 0).any()
    product = depth[depth > 0] * disparity[depth > 0]
    np.testing.assert_allclose(product, focal_baseline / 2, rtol=2e-3)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_read_intrinsics(tmp_path, loaders, source):
    path = _calib(tmp_path / "half-image-calib" / f"{SEQUENCE}.txt")
    K = loaders.read_intrinsics(source(path))
    if loaders is gpu_ds:
        assert isinstance(K, torch.Tensor)
        K = K.numpy()
    assert K.dtype == np.float32 and K.flags.c_contiguous
    np.testing.assert_allclose(K, K_HALF, rtol=1e-7)
    right = loaders.read_intrinsics(str(path), {"camera": "right"})
    np.testing.assert_allclose(np.asarray(right), K_HALF, rtol=1e-7)


def test_read_intrinsics_errors(tmp_path):
    path = _calib(tmp_path / "c.txt")
    with pytest.raises(ValueError, match="'camera'"):
        cpu_ds.read_intrinsics(str(path), {"camera": "middle"})
    missing = _calib(tmp_path / "m.txt", "calib_time: 1\nS_rect_101: 1 2\n")
    with pytest.raises(ValueError, match="P_rect_101"):
        cpu_ds.read_intrinsics(str(missing))
    skewed = _calib(tmp_path / "s.txt", "P_rect_101: 1 0 0 0 1 1 0 0 1 1 1 0\n")
    with pytest.raises(ValueError, match="pinhole"):
        cpu_ds.read_intrinsics(str(skewed))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_depth_and_disparity_writers_round_trip(tmp_path, loaders, source):
    raw = np.array([[0, 256, 30206], [65535, 1, 7]], np.uint16)
    for reader, writer in [("depth", cpu_ds.write_depth), ("disparity", cpu_ds.write_disparity)]:
        value = getattr(loaders, reader)(str(_png16(tmp_path / f"{reader}.png", raw)))
        out = tmp_path / "out" / f"{reader}.png"
        writer(str(out), value)
        np.testing.assert_array_equal(np.array(Image.open(out)), raw)
        np.testing.assert_array_equal(
            _image(getattr(loaders, reader)(source(out)), loaders), raw / 256.0
        )


def test_map_writers_mark_invalid_and_saturate(tmp_path):
    out = tmp_path / "depth.png"
    cpu_ds.write_depth(str(out), np.array([[np.nan, -1.0, np.inf], [1000.0, 0.5, 0.0]]))
    np.testing.assert_array_equal(np.array(Image.open(out)), [[0, 0, 0], [65535, 128, 0]])


@pytest.mark.parametrize("resampling", ["nearest", "subsample"])
def test_full_size_writers_round_trip(tmp_path, resampling):
    rng = np.random.default_rng(1)
    full = np.where(rng.random((8, 10)) < 0.3, rng.integers(1, 65535, (8, 10)), 0)
    meta = {"full_size_resampling": resampling}
    for reader, writer in [("depth_full_size", cpu_ds.write_depth_full_size),
                           ("disparity_full_size", cpu_ds.write_disparity_full_size)]:
        value = getattr(cpu_ds, reader)(str(_png16(tmp_path / f"{reader}.png", full)), meta)
        out = tmp_path / "out" / f"{reader}.png"
        writer(str(out), torch.from_numpy(value).unsqueeze(0))
        written = np.array(Image.open(out))
        assert written.shape == (8, 10)
        assert not written[1::2].any() and not written[:, 1::2].any()
        np.testing.assert_array_equal(getattr(cpu_ds, reader)(str(out), meta), value)


def test_rgb_writer(tmp_path):
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) * 3
    value = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    cpu_ds.write_rgb(str(tmp_path / "a.png"), value)
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / "a.png")), pixels)
    stream = io.BytesIO()
    stream.name = "frame.jpg"
    cpu_ds.write_rgb(stream, value)
    stream.seek(0)
    with Image.open(stream) as image:
        assert image.format == "JPEG" and image.size == (6, 4)


def test_write_intrinsics_applies_a_crop_to_every_rectified_camera(tmp_path):
    path = _calib(tmp_path / "calib.txt")
    K = cpu_ds.read_intrinsics(str(path))
    cropped = K.copy()
    cropped[0, 2] -= 100
    cropped[1, 2] -= 50
    cpu_ds.write_intrinsics(str(path), cropped, {"width": 679, "height": 300})

    text = path.read_text()
    entries = cpu_ds._parse_calibration(text, source="calib")
    for camera in ("101", "103"):
        P = entries[f"P_rect_{camera}"].reshape(3, 4)
        assert P[0, 2] == 452.6664 - 100 and P[1, 2] == 194.257 - 50
        assert P[0, 0] == 1001.496
        np.testing.assert_array_equal(entries[f"S_rect_{camera}"], [679, 300])
    assert entries["P_rect_103"][3] == -1091.469
    # Lines the crop does not touch are kept verbatim.
    kept = [line for line in HALF_CALIB.splitlines()
            if not line.startswith(("P_rect_", "S_rect_"))]
    assert [line for line in text.splitlines()
            if not line.startswith(("P_rect_", "S_rect_"))] == kept
    np.testing.assert_allclose(cpu_ds.read_intrinsics(str(path)), cropped)


def test_write_intrinsics_rescale_and_identity(tmp_path):
    path = _calib(tmp_path / "calib.txt")
    K = cpu_ds.read_intrinsics(str(path))
    cpu_ds.write_intrinsics(str(path), K)
    cpu_ds.write_intrinsics(str(path), K, {"width": 879, "height": 400})
    assert path.read_text() == HALF_CALIB

    halved = K.astype(np.float64) * np.array([[0.5], [0.5], [1.0]])
    cpu_ds.write_intrinsics(str(path), halved)
    entries = cpu_ds._parse_calibration(path.read_text(), source="calib")
    np.testing.assert_allclose(entries["P_rect_103"][3], -1091.469 / 2, rtol=1e-6)

    rotated = K.copy()
    rotated[0, 1] = 5.0
    with pytest.raises(ValueError, match="crop or rescale"):
        cpu_ds.write_intrinsics(str(path), rotated)


def test_write_intrinsics_new_file_and_stream(tmp_path):
    path = tmp_path / "new" / "calib.txt"
    cpu_ds.write_intrinsics(str(path), K_HALF, {"width": 879, "height": 400})
    np.testing.assert_allclose(cpu_ds.read_intrinsics(str(path)), K_HALF, rtol=1e-7)
    assert "S_rect_101: 879 400" in path.read_text()
    stream = io.BytesIO()
    cpu_ds.write_intrinsics(stream, torch.from_numpy(K_HALF))
    stream.seek(0)
    np.testing.assert_allclose(cpu_ds.read_intrinsics(stream), K_HALF, rtol=1e-7)


# ---------------------------------------------------------------------------
# Dataset integration through ds-crawler
# ---------------------------------------------------------------------------

_ID = r"^[^/]+/{folder}/[0-9-]+_(?P<frame>[0-9-]+)\.{ext}$"
_HIERARCHY = r"^[^/]+/{folder}/(?P<sequence>[0-9-]+)_[0-9-]+\.{ext}$"
# What the dataset contract requires of each head (the stored encodings), plus
# the invalid sentinel and sparsity the preprocessor resamples by.
_HEAD_META = {
    "rgb": {"range": [0, 255]},
    "depth": {"radial_depth": False, "scale_to_meters": 1 / 256, "range": [0, 65535],
              "invalid": {"non_finite": False, "sentinel": 0}, "sparse": True},
}


def _index(root, *, key, function, id_regex, hierarchy_regex, extension, override=None):
    meta = _HEAD_META.get(key, {})
    metadata = root / ".ds_crawler"
    metadata.mkdir(parents=True)
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": f"driving_stereo_{function}", "name": f"DrivingStereo {function}"},
        "modality": {"key": key, "meta": meta},
        "addons": {"euler_loading": {"version": "1.0", "loader": "driving_stereo", "function": function}},
    }
    id_config = {"regex": id_regex, "join_char": "+"}
    if override:
        id_config["override"] = override
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": id_config,
            "hierarchy": {"regex": hierarchy_regex, "separator": ":"},
            "files": {"extensions": [extension]},
        },
    }
    (metadata / "dataset-head.json").write_text(json.dumps(head))
    (metadata / "ds-crawler.json").write_text(json.dumps(config))


def _zip(root):
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    return archive


def _weather_roots(tmp_path, zipped):
    """Write the foggy layout: one root per archive, each with its own index."""
    roots = {}
    for name, key, function, ext in [
        ("left-image-half-size", "rgb", "rgb", "jpg"),
        ("depth-map-half-size", "depth", "depth", "png"),
        ("depth-map-full-size", "depth", "depth_full_size", "png"),
        ("disparity-map-full-size", "disparity", "disparity_full_size", "png"),
    ]:
        root = tmp_path / name
        for index, frame in enumerate(FRAMES):
            path = root / "foggy" / name / f"{SEQUENCE}_{frame}.{ext}"
            if ext == "jpg":
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((2, 3, 3), 40 * index, np.uint8)).save(path)
            elif name.endswith("half-size"):
                _png16(path, np.full((2, 3), 256 * (index + 10)))
            else:
                full = np.zeros((4, 6), np.uint16)
                full[0::2, 0::2] = 256 * (index + 10) if "depth" in name else 128 * 2 * (index + 1)
                _png16(path, full)
        _index(root, key=key, function=function, extension=f".{ext}",
               id_regex=_ID.format(folder=name, ext=ext),
               hierarchy_regex=_HIERARCHY.format(folder=name, ext=ext))
        roots[function] = root
    calibration = tmp_path / "half-image-calib"
    _calib(calibration / "half-image-calib" / f"{SEQUENCE}.txt")
    _index(calibration, key="intrinsics", function="read_intrinsics", extension=".txt",
           id_regex=r"^(half-image-calib)/[0-9-]+\.txt$",
           hierarchy_regex=r"^half-image-calib/(?P<sequence>[0-9-]+)\.txt$",
           override="intrinsics")
    roots["read_intrinsics"] = calibration
    if zipped:
        roots = {function: _zip(root) for function, root in roots.items()}
    return {function: str(root) for function, root in roots.items()}


@pytest.mark.parametrize("zipped", [False, True])
def test_dataset_resolves_driving_stereo_from_the_contract(tmp_path, zipped):
    roots = _weather_roots(tmp_path, zipped)
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(roots["rgb"]),
            "depth": Modality(roots["depth"]),
            "depth_full": Modality(roots["depth_full_size"]),
            "disparity_full": Modality(roots["disparity_full_size"]),
        },
        hierarchical_modalities={
            "intrinsics": Modality(roots["read_intrinsics"], collapse_single=True),
        },
    )
    assert len(dataset) == len(FRAMES)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["full_id"].startswith(f"/sequence:{SEQUENCE}/")
        frame = FRAMES.index(sample["id"].split("-", 1)[1])
        assert sample["rgb"].shape == (3, 2, 3)
        # Every map lands on the half-size left-image grid.
        for name in ("depth", "depth_full", "disparity_full"):
            assert sample[name].shape == (1, 2, 3), name
        torch.testing.assert_close(sample["depth"], torch.full((1, 2, 3), 10.0 + frame))
        torch.testing.assert_close(sample["depth_full"], sample["depth"])
        torch.testing.assert_close(sample["disparity_full"], torch.full((1, 2, 3), 1.0 + frame))
        np.testing.assert_allclose(sample["intrinsics"].numpy(), K_HALF, rtol=1e-7)
    assert dataset.get_writer("depth") is gpu_ds.write_depth
    assert dataset.get_writer("depth_full") is gpu_ds.write_depth_full_size
    assert dataset.get_writer("intrinsics") is gpu_ds.write_intrinsics


def test_preprocessor_resizes_sparse_depth_by_the_head(tmp_path):
    from euler_loading.preprocessing import SamplePreprocessor

    roots = _weather_roots(tmp_path, zipped=False)
    preprocessor = SamplePreprocessor.from_config({"resize": [1, 2]})
    dataset = MultiModalDataset(
        modalities={"rgb": Modality(roots["rgb"]), "depth": Modality(roots["depth"])},
        transforms=[preprocessor],
    )
    spec = preprocessor._bound_field_specs["depth"]
    assert (spec.interpolation, spec.invalid_value) == ("sparse", 0.0)
    sample = dataset[0]
    assert sample["rgb"].shape == (3, 1, 2) and sample["depth"].shape == (1, 1, 2)
    # Every value is a measurement, not a blend with the invalid zeros.
    assert set(sample["depth"].unique().tolist()) <= {10.0, 11.0}
