"""Resizing rasters with invalid pixels: sparse scatter, validity-aware
interpolation, pixel-centred nearest and their binding from dataset heads."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from euler_loading import resize_raster
from euler_loading.preprocessing import FieldSpec, SamplePreprocessor, resize_intrinsics


@pytest.fixture(params=["numpy", "torch"])
def container(request):
    if request.param == "numpy":
        return np.asarray
    return lambda value: torch.as_tensor(np.asarray(value))


def _np(value):
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


# ---------------------------------------------------------------------------
# nearest samples pixel centres
# ---------------------------------------------------------------------------


def test_nearest_samples_pixel_centres_with_torch_and_pillow(container):
    row = container(np.arange(8, dtype=np.float32)[None])
    halved = resize_raster(row, layout="HW", size=(1, 4), interpolation="nearest")
    # Target pixel i covers source [2i, 2i + 2); its centre lies at 2i + 0.5.
    np.testing.assert_array_equal(_np(halved), [[1, 3, 5, 7]])


# ---------------------------------------------------------------------------
# sparse
# ---------------------------------------------------------------------------


def test_sparse_downsampling_keeps_measurements_and_the_nearest_surface(container):
    row = container(np.array([[5.0, 3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]], np.float32))
    out = resize_raster(row, layout="HW", size=(1, 4), interpolation="sparse")
    assert _np(out).dtype == np.float32
    np.testing.assert_array_equal(_np(out), [[3.0, 0.0, 7.0, 0.0]])


def test_sparse_upsampling_does_not_duplicate_samples():
    out = resize_raster(np.array([[2.0, 0.0]], np.float32), layout="HW",
                        size=(1, 4), interpolation="sparse")
    # Source centre 0 maps to 0.5 and rounds half up.
    np.testing.assert_array_equal(out, [[0.0, 2.0, 0.0, 0.0]])


def test_sparse_lands_where_resize_intrinsics_moves_the_camera():
    rng = np.random.default_rng(0)
    K = np.array([[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]])
    points = np.c_[rng.uniform(-1, 1, (40, 2)), rng.uniform(2, 9, 40)]
    depth = np.zeros((48, 64), np.float32)
    uv = (K @ (points / points[:, 2:]).T).T
    cols, rows = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    keep = (cols >= 0) & (cols < 64) & (rows >= 0) & (rows < 48)
    depth[rows[keep], cols[keep]] = points[keep, 2]
    resized = resize_raster(depth, layout="HW", size=(24, 32), interpolation="sparse")
    K2 = resize_intrinsics(K, source_size=(48, 64), target_size=(24, 32))
    ys, xs = np.nonzero(resized)
    assert len(ys) > 10
    # Every resized sample sits within half a target pixel of a point that
    # projects there through the resized camera, at that point's depth.
    uv2 = (K2 @ (points / points[:, 2:]).T).T
    for y, x in zip(ys, xs):
        distance = np.hypot(uv2[:, 0] - x, uv2[:, 1] - y)
        candidates = points[distance <= 0.5 * np.sqrt(2) + 0.5 + 1e-9, 2]
        assert np.isclose(candidates, resized[y, x]).any()


def test_sparse_treats_non_finite_and_a_custom_sentinel_as_invalid():
    row = np.array([[np.nan, -1.0, 4.0, np.inf]], np.float32)
    out = resize_raster(row, layout="HW", size=(1, 2), interpolation="sparse", invalid_value=-1)
    np.testing.assert_array_equal(out, [[-1.0, 4.0]])


def test_sparse_keeps_layout_and_tensor_container():
    chw = torch.zeros(1, 4, 4)
    chw[0, 1, 1] = 3.0
    out = resize_raster(chw, layout="CHW", size=(2, 2), interpolation="sparse")
    assert isinstance(out, torch.Tensor) and out.shape == (1, 2, 2) and out.dtype == torch.float32
    assert out[0, 0, 0] == 3.0 and out.sum() == 3.0


# ---------------------------------------------------------------------------
# validity-aware interpolation
# ---------------------------------------------------------------------------


def test_invalid_value_stops_bilinear_from_blending_in_invalid_pixels(container):
    depth = container(np.array([[10.0, 0.0], [0.0, 0.0]], np.float32))
    blended = resize_raster(depth, layout="HW", size=(1, 1))
    aware = resize_raster(depth, layout="HW", size=(1, 1), invalid_value=0)
    assert _np(blended).item() == pytest.approx(2.5)
    assert _np(aware).item() == pytest.approx(10.0)
    empty = resize_raster(container(np.zeros((2, 2), np.float32)), layout="HW",
                          size=(1, 1), invalid_value=0)
    assert _np(empty).item() == 0.0


def test_field_spec_options():
    spec = FieldSpec.from_config({"kind": "depth", "interpolation": "Sparse", "invalid_value": 0})
    assert spec.interpolation == "sparse" and spec.invalid_value == 0.0
    assert "interpolation=sparse" in spec.describe() and "invalid_value=0" in spec.describe()
    with pytest.raises(ValueError, match="sparse"):
        FieldSpec(interpolation="area")
    with pytest.raises(ValueError, match="finite"):
        FieldSpec(invalid_value=float("nan"))


def test_preprocessor_applies_field_specs():
    depth = np.zeros((1, 4, 4), np.float32)
    depth[0, 0, 0] = 8.0
    sample = {"depth": torch.from_numpy(depth), "rgb": torch.zeros(3, 4, 4)}
    preprocessor = SamplePreprocessor.from_config(
        {"resize": [2, 2], "fields": {"depth": {"kind": "depth", "interpolation": "sparse"}}}
    )
    out = preprocessor(sample)
    assert out["depth"][0, 0, 0] == 8.0 and out["depth"].sum() == 8.0
    assert out["rgb"].shape == (3, 2, 2)


# ---------------------------------------------------------------------------
# Binding from dataset heads
# ---------------------------------------------------------------------------


def _dataset(heads, loader_types=None, modality_types=None):
    loader_types = loader_types or {}
    modality_types = modality_types or {}
    return SimpleNamespace(
        _modalities={name: SimpleNamespace(modality_type=modality_types.get(name)) for name in heads},
        _hierarchical_modalities={},
        get_modality_metadata=lambda name: heads[name],
        get_loader_meta=lambda name: {"type": loader_types[name]} if name in loader_types else None,
    )


def test_bind_selects_resampling_from_the_depth_head():
    sparse = {"invalid": {"non_finite": False, "sentinel": 0}, "sparse": True}
    holes = {"invalid": {"non_finite": False, "sentinel": -1}}
    preprocessor = SamplePreprocessor.from_config(
        {"resize": [2, 2], "fields": {"explicit_depth": {"kind": "depth", "interpolation": "nearest"}}}
    )
    preprocessor.bind_to_dataset(_dataset(
        {"depth": sparse, "gt": sparse, "holes_depth": holes, "plain_depth": {},
         "explicit_depth": sparse, "rgb": sparse, "typed": holes},
        loader_types={"gt": "depth", "rgb": "rgb"},
        modality_types={"typed": "depth"},
    ))
    specs = preprocessor._bound_field_specs
    assert (specs["depth"].interpolation, specs["depth"].invalid_value) == ("sparse", 0.0)
    assert specs["gt"].interpolation == "sparse"  # kind from the loader annotation
    assert (specs["holes_depth"].interpolation, specs["holes_depth"].invalid_value) == (None, -1.0)
    assert specs["typed"].invalid_value == -1.0
    assert "plain_depth" not in specs and "rgb" not in specs
    assert specs["explicit_depth"] == FieldSpec(kind="depth", interpolation="nearest")


def test_export_refuses_sparse_resampling():
    from euler_dataset_contract.testing import fixtures_root
    from euler_dataset_contract.testing.checks._support import evidence, make_sample
    from euler_loading import execution_profile

    fixture = json.loads((fixtures_root() / "descriptors/five-field.json").read_text())
    addon = copy.deepcopy(fixture["addon"])
    fields = addon["recipe"]["fields"]
    depth = next(name for name, b in fields.items() if b["profile"]["kind"] == "depth")
    config = copy.deepcopy(evidence("documented-five-field")["config"])
    spec = dict(config.setdefault("fields", {}).get(depth) or {"kind": "depth"})
    bindings = dict(sources=addon["sources"], fields=fields,
                    image_planes=addon["recipe"]["image_planes"],
                    execution=execution_profile("torch_cpu"))
    sample = make_sample(evidence("documented-five-field"))
    for extra, message in [({"interpolation": "sparse"}, "sparse"),
                           ({"invalid_value": 0}, "validity_aware")]:
        config["fields"][depth] = {**spec, **extra}
        with pytest.raises(ValueError, match=message):
            SamplePreprocessor.from_config(config).export_descriptor(
                **copy.deepcopy(bindings), sample=sample
            )


# ---------------------------------------------------------------------------
# disparity
# ---------------------------------------------------------------------------


def _resize(config, sample):
    return SamplePreprocessor.from_config(config)(sample)


def test_disparity_scales_with_the_width_and_not_with_a_crop(container):
    disparity = container(np.full((4, 8), 10.0, dtype=np.float32))
    spec = {"disparity": {"kind": "disparity", "layout": "HW"}}
    for size, expected in [([2, 4], 5.0), ([4, 16], 20.0), ([2, 8], 10.0)]:
        out = _resize({"resize": size, "fields": spec}, {"disparity": disparity})
        np.testing.assert_allclose(_np(out["disparity"]), expected)
    out = _resize({"crop": {"size": [2, 4]}, "fields": spec}, {"disparity": disparity})
    np.testing.assert_allclose(_np(out["disparity"]), 10.0)


def test_sparse_disparity_keeps_the_nearest_surface_and_scales_it(container):
    disparity = container(np.array([[2.0, 4.0], [0.0, 6.0]], dtype=np.float32))
    out = _resize(
        {"resize": [1, 1], "fields": {"disparity": {
            "kind": "disparity", "layout": "HW", "interpolation": "sparse"}}},
        {"disparity": disparity},
    )
    # The largest disparity is the nearest surface; halving the width halves it.
    np.testing.assert_allclose(_np(out["disparity"]), [[3.0]])


def test_disparity_sentinel_survives_the_rescale():
    disparity = np.array([[-1.0, -1.0, 8.0, 8.0]], dtype=np.float32)
    out = _resize(
        {"resize": [1, 2], "fields": {"disparity": {
            "kind": "disparity", "layout": "HW", "invalid_value": -1}}},
        {"disparity": disparity},
    )
    np.testing.assert_allclose(out["disparity"], [[-1.0, 4.0]])


def test_bind_recognises_disparity_and_falls_back_to_the_loader_meta():
    sparse_meta = {"invalid": {"non_finite": False, "sentinel": 0}, "sparse": True}
    annotations = {
        "disparity": {"type": "disparity", "meta": sparse_meta},
        "disp": {"type": "disparity"},
        "depth": {"type": "depth", "meta": sparse_meta},
        "opted_out": {"type": "depth", "meta": sparse_meta},
    }
    dataset = SimpleNamespace(
        _modalities={name: SimpleNamespace(modality_type=None) for name in annotations},
        _hierarchical_modalities={},
        get_modality_metadata=lambda name: {"sparse": False} if name == "opted_out" else {},
        get_loader_meta=lambda name: annotations[name],
    )
    preprocessor = SamplePreprocessor.from_config({"resize": [2, 2]})
    preprocessor.bind_to_dataset(dataset)
    specs = preprocessor._bound_field_specs
    # A head that declares nothing gets what the loader always returns.
    assert specs["disparity"] == FieldSpec(kind="disparity", interpolation="sparse", invalid_value=0)
    assert specs["depth"] == FieldSpec(kind="depth", interpolation="sparse", invalid_value=0)
    # Disparity is bound for its rescale even without invalid pixels.
    assert specs["disp"] == FieldSpec(kind="disparity")
    # A head's own key wins over the loader's.
    assert specs["opted_out"] == FieldSpec(kind="depth", invalid_value=0)


def test_a_modality_type_without_a_resize_default_resizes_like_a_generic_array():
    labels = np.arange(16, dtype=np.int64).reshape(4, 4)
    out = _resize(
        {"resize": [2, 2], "fields": {"labels": {"kind": "semantic_segmentation", "layout": "HW"}}},
        {"labels": labels},
    )
    assert out["labels"].dtype == np.int64
    assert set(np.unique(out["labels"])) <= set(range(16))  # nearest, never blended


def test_driving_stereo_depth_and_disparity_stay_consistent_after_a_resize():
    from euler_loading.loaders.cpu import driving_stereo

    fx, baseline = 1001.496, 0.545
    rng = np.random.default_rng(0)
    depth = np.where(rng.random((40, 88)) < 0.2, rng.uniform(5, 80, (40, 88)), 0).astype(np.float32)
    disparity = np.where(depth > 0, fx * baseline / np.maximum(depth, 1e-6), 0).astype(np.float32)
    annotations = {"depth": driving_stereo.depth._modality_meta,
                   "disparity": driving_stereo.disparity._modality_meta}
    dataset = SimpleNamespace(
        _modalities={name: SimpleNamespace(modality_type=None) for name in annotations},
        _hierarchical_modalities={},
        get_modality_metadata=lambda name: {},
        get_loader_meta=lambda name: annotations[name],
    )
    for size in ([20, 44], [80, 176], [23, 50]):
        preprocessor = SamplePreprocessor.from_config({"resize": size})
        preprocessor.bind_to_dataset(dataset)
        out = preprocessor({"depth": depth, "disparity": disparity})
        both = (out["depth"] > 0) & (out["disparity"] > 0)
        assert both.any()
        np.testing.assert_allclose(
            out["depth"][both] * out["disparity"][both],
            fx * size[1] / 88 * baseline,
            rtol=1e-5,
        )
        assert np.isin(out["depth"][out["depth"] > 0], depth).all()


def test_export_refuses_disparity():
    from euler_dataset_contract.testing import fixtures_root
    from euler_dataset_contract.testing.checks._support import evidence, make_sample
    from euler_loading import execution_profile

    fixture = json.loads((fixtures_root() / "descriptors/five-field.json").read_text())
    addon = copy.deepcopy(fixture["addon"])
    fields = addon["recipe"]["fields"]
    depth = next(name for name, b in fields.items() if b["profile"]["kind"] == "depth")
    config = copy.deepcopy(evidence("documented-five-field")["config"])
    config.setdefault("fields", {})[depth] = {"kind": "disparity"}
    with pytest.raises(ValueError, match="disparity"):
        SamplePreprocessor.from_config(config).export_descriptor(
            sources=addon["sources"], fields=fields,
            image_planes=addon["recipe"]["image_planes"],
            execution=execution_profile("torch_cpu"),
            sample=make_sample(evidence("documented-five-field")),
        )
