from __future__ import annotations

import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

import numpy as np
import pytest
import torch
from euler_loading import (
    Crop,
    Resize,
    SamplePreprocessor,
    SerializableTransform,
    execution_profile,
    resolve_transform_descriptor,
)

from euler_dataset_contract import (
    TransformsAddon,
    build_descriptor_schema,
    recipe_digest,
)
from euler_dataset_contract.canonical import canonical_digest
from euler_dataset_contract.testing import fixtures_root
from euler_dataset_contract.testing.checks._support import (
    as_numpy,
    assert_preprocessed,
    evidence,
    make_dataset,
    make_sample,
)


@pytest.fixture
def fixture():
    value = json.loads((fixtures_root() / "descriptors/five-field.json").read_text())
    # The shared wire vector pins its capture environment. Exercise export under
    # this test environment's explicitly selected profile, including on Python 3.9.
    addon = value["addon"]
    addon["recipe"]["execution"] = execution_profile("torch_cpu").to_dict()
    addon["recipe_digest"] = recipe_digest(addon["recipe"])
    return value


def _sample(*, tensors=False):
    return make_sample(evidence("documented-five-field"), tensors=tensors)


def _bindings(fixture, backend="torch_cpu"):
    addon = copy.deepcopy(fixture["addon"])
    return dict(
        sources=addon["sources"],
        fields=addon["recipe"]["fields"],
        image_planes=addon["recipe"]["image_planes"],
        execution=execution_profile(backend),
    )


def _preprocessor():
    return SamplePreprocessor.from_config(evidence("documented-five-field")["config"])


def _rehash(addon):
    for binding in addon["recipe"]["fields"].values():
        addon["sources"][binding["source"]]["decoder"]["profile_digest"] = (
            canonical_digest(binding["profile"])
        )
    addon["recipe_digest"] = recipe_digest(addon["recipe"])
    return addon


@pytest.mark.parametrize("tensors", [False, True])
def test_five_field_export_resolve_and_execute(fixture, tensors):
    sample = _sample(tensors=tensors)
    original = {name: copy.deepcopy(value) for name, value in sample.items()}
    preprocessor = _preprocessor()
    assert isinstance(preprocessor, SerializableTransform)
    descriptor = preprocessor.export_descriptor(**_bindings(fixture), sample=sample)
    assert descriptor == TransformsAddon(fixture["addon"])
    resolved = resolve_transform_descriptor(descriptor.to_json(), sample=sample)
    assert resolved.export_descriptor() == descriptor
    assert resolved.operations[1].offset == (32, 64)
    assert resolved.operations[0].input_size == (480, 960)
    assert resolved.operations[0].output_size == (384, 768)
    output = resolved(sample)
    assert_preprocessed(output, evidence("documented-five-field"))
    profiles = resolved.infer_output_profiles()
    assert profiles["rgb"].shape == (320, 640, 3)
    assert profiles["depth"]["unit"] == "meter"
    assert profiles["depth"]["depth_kind"] == "planar_z"
    assert profiles["intrinsics"]["image_plane"] != "camera_0"
    view = resolved.infer_output_bindings()["fields"]["intrinsics"]["calibration_view"]
    assert view["source"]["full_id"] == "/scene_01/camera_0/calibration"
    assert view["image_plane"] == profiles["rgb"]["image_plane"]
    k = as_numpy(output["intrinsics"])
    projection = k @ np.array([0, 0, 2])
    np.testing.assert_allclose(projection[:2] / projection[2], [319.5, 159.5])
    for name in ("rgb", "depth", "valid_mask", "ray_map"):
        np.testing.assert_array_equal(as_numpy(sample[name]), as_numpy(original[name]))
    np.testing.assert_array_equal(
        as_numpy(sample["intrinsics"]["camera_0"]),
        as_numpy(original["intrinsics"]["camera_0"]),
    )
    output["intrinsics"][0, 0] = 0
    assert as_numpy(sample["intrinsics"]["camera_0"])[0, 0] == 800


@pytest.mark.parametrize("backend", ["torch_cpu", "pillow_cpu"])
def test_backend_is_pinned_for_numpy(fixture, backend):
    sample = _sample()
    descriptor = _preprocessor().export_descriptor(
        **_bindings(fixture, backend), sample=sample
    )
    resolved = resolve_transform_descriptor(descriptor)
    result = resolved(sample)
    assert result["depth"].shape == (320, 640)
    np.testing.assert_allclose(
        result["intrinsics"], [[640, 0, 319.5], [0, 640, 159.5], [0, 0, 1]]
    )
    np.testing.assert_allclose(result["ray_map"][0, 0], [0, 0, 1])
    assert descriptor["recipe"]["execution"]["backend"] == backend
    if backend == "pillow_cpu":
        with pytest.raises(ValueError, match="container"):
            resolved(_sample(tensors=True))


@pytest.mark.parametrize("tensors", [False, True])
@pytest.mark.parametrize(
    "operations,expected",
    [
        (
            [Resize((240, 1920)), Crop((233, 1901))],
            [[1600, 4, 950.5], [0, 400, 116.5], [0, 0, 1]],
        ),
        (
            [Resize((240, 1920)), Crop((230, 1800), offset=(2, 3))],
            [[1600, 4, 956.5], [0, 400, 117.5], [0, 0, 1]],
        ),
        (
            [Crop((479, 959), anchor="bottom_right")],
            [[800, 2, 478.5], [0, 800, 238.5], [0, 0, 1]],
        ),
    ],
)
def test_nonzero_skew_nonuniform_odd_and_offcenter_geometry(
    fixture, tensors, operations, expected
):
    sample = _sample(tensors=tensors)
    sample["intrinsics"]["camera_0"][0, 1] = 2
    preprocessor = _preprocessor()
    preprocessor.operations = operations
    resolved = resolve_transform_descriptor(
        preprocessor.export_descriptor(**_bindings(fixture), sample=sample)
    )
    result = resolved(sample)
    np.testing.assert_allclose(as_numpy(result["intrinsics"]), expected, atol=1e-5)
    old = as_numpy(sample["intrinsics"]["camera_0"])
    np.testing.assert_allclose(np.asarray(resolved.image_transform) @ old, expected)
    points = np.array([[0.3, -0.2, 2], [0.5, 0.8, 4]]).T
    projected = old @ points
    projected /= projected[2]
    mapped = np.asarray(resolved.image_transform) @ projected
    actual = as_numpy(result["intrinsics"]) @ points
    actual /= actual[2]
    np.testing.assert_allclose(actual, mapped)
    assert old[0, 1] == 2  # cached source did not change


def _small_depth_descriptor(fixture, backend, policy):
    addon = copy.deepcopy(fixture["addon"])
    recipe = addon["recipe"]
    recipe["fields"] = {"depth": recipe["fields"]["depth"]}
    recipe["fields"]["depth"]["profile"]["shape"] = [2, 2]
    recipe["fields"]["depth"]["policy"]["invalid_depth"] = policy
    recipe["reference_field"] = "depth"
    recipe["image_planes"]["camera_0"]["size"] = [2, 2]
    recipe["operations"] = [
        {
            "id": "resize_1",
            "op": "euler_loading.resize",
            "version": "1.0",
            "parameters": {"size": [1, 1]},
        }
    ]
    recipe["execution"] = execution_profile(backend).to_dict()
    addon["sources"] = {"source_depth": addon["sources"]["source_depth"]}
    return _rehash(addon)


@pytest.mark.parametrize("backend", ["torch_cpu", "pillow_cpu"])
@pytest.mark.parametrize(
    "policy,expected", [("legacy_blend", 3), ("validity_aware", 4)]
)
def test_invalid_depth_policy_is_explicit(fixture, backend, policy, expected):
    resolved = resolve_transform_descriptor(
        _small_depth_descriptor(fixture, backend, policy)
    )
    source = np.array([[0, 4], [4, 4]], dtype=np.float32)
    assert resolved({"depth": source})["depth"].item() == pytest.approx(expected)
    assert resolved({"depth": np.zeros((2, 2), dtype=np.float32)})["depth"].item() == 0
    assert source[0, 0] == 0


def test_export_freezes_single_calibration_and_rejects_ambiguity(fixture):
    preprocessor = _preprocessor()
    bindings = _bindings(fixture)
    del bindings["fields"]["intrinsics"]["selection"]
    descriptor = preprocessor.export_descriptor(**bindings, sample=_sample())
    assert descriptor["recipe"]["fields"]["intrinsics"]["selection"] == "camera_0"
    sample = _sample()
    sample["intrinsics"]["camera_1"] = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="ambiguous calibration"):
        preprocessor.export_descriptor(**bindings, sample=sample)
    with pytest.raises(ValueError, match="requires a sample"):
        preprocessor.export_descriptor(**bindings)
    descriptor = preprocessor.export_descriptor(**_bindings(fixture), sample=sample)
    output = resolve_transform_descriptor(descriptor)(sample)
    assert output["intrinsics"][0, 0] == 640
    assert sample["intrinsics"]["camera_1"][0, 0] == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p, b, s: s.pop("depth"), "Missing required field"),
        (lambda p, b, s: b["fields"].pop("depth"), "Missing declared field"),
        (lambda p, b, s: s.update(rgb=np.zeros((1, 2, 3), np.float32)), "shape/dtype"),
        (
            lambda p, b, s: s.update(depth=np.zeros((480, 960), np.float64)),
            "shape/dtype",
        ),
        (
            lambda p, b, s: b["fields"]["depth"]["profile"].update(frame="world"),
            "frame disagrees",
        ),
        (
            lambda p, b, s: b["fields"]["depth"]["profile"].update(image_plane="other"),
            "image plane mismatch",
        ),
        (lambda p, b, s: setattr(p, "reference_field", None), "Ambiguous reference"),
        (lambda p, b, s: setattr(p, "operations", [Crop((999, 999))]), "bounds"),
        (
            lambda p, b, s: setattr(p, "operations", [Crop((2, 2), offset=(-1, 0))]),
            "variant",
        ),
        (
            lambda p, b, s: setattr(p, "operations", [lambda value: value]),
            "Unknown callable operation",
        ),
        (
            lambda p, b, s: b["fields"]["rgb"]["profile"].update(
                layout="CHW", shape=[3, 480, 960]
            ),
            "layout conflicts",
        ),
        (
            lambda p, b, s: b["fields"]["rgb"]["policy"].update(
                interpolation="nearest"
            ),
            "policy.interpolation conflicts",
        ),
    ],
)
def test_export_rejections(fixture, mutation, match):
    preprocessor, bindings, sample = _preprocessor(), _bindings(fixture), _sample()
    mutation(preprocessor, bindings, sample)
    with pytest.raises(ValueError, match=match):
        preprocessor.export_descriptor(**bindings, sample=sample)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda d: d["recipe"]["execution"]["versions"].update(torch="0.0"),
            "execution.versions",
        ),
        (
            lambda d: d["recipe"]["execution"].update(nearest="half_pixel"),
            "execution.nearest",
        ),
        (
            lambda d: d["recipe"]["execution"].update(antialias=True),
            "execution.antialias",
        ),
        (
            lambda d: d["recipe"]["fields"]["ray_map"]["policy"].update(
                rays="regenerate"
            ),
            "regeneration",
        ),
        (
            lambda d: d["recipe"]["fields"]["rgb"]["profile"].update(
                shape=["H", "W", 3]
            ),
            "symbolic",
        ),
        (
            lambda d: d["recipe"]["fields"]["valid_mask"]["profile"].update(
                dtype="int64"
            ),
            "int32/int64",
        ),
    ],
)
def test_structurally_valid_but_unsupported_executor_profiles(fixture, mutation, match):
    addon = fixture["addon"]
    mutation(addon)
    _rehash(addon)
    TransformsAddon(
        addon
    )  # Contract can describe semantics that this executor cannot implement.
    with pytest.raises(ValueError, match=match):
        resolve_transform_descriptor(addon)


def test_non_pinhole_model_rejected(fixture):
    addon = fixture["addon"]
    addon["recipe"]["image_planes"]["camera_0"]["camera_model"] = "fisheye"
    binding = addon["recipe"]["fields"]["intrinsics"]
    binding["profile"]["camera_model"] = "fisheye"
    binding["calibration"]["camera_model"] = "fisheye"
    with pytest.raises(ValueError, match="camera model"):
        resolve_transform_descriptor(_rehash(addon))


def test_ray_policy_and_source_immutability(fixture):
    addon = fixture["addon"]
    addon["recipe"]["fields"]["ray_map"]["policy"]["rays"] = "preserve"
    resolved = resolve_transform_descriptor(_rehash(addon))
    sample = _sample()
    output = resolved(sample)
    np.testing.assert_array_equal(output["ray_map"][0, 0], [0, 0, 3])
    output["ray_map"][0, 0] = 100
    np.testing.assert_array_equal(sample["ray_map"][0, 0], [0, 0, 3])
    assert resolved.infer_output_profiles()["ray_map"]["normalization"] == "none"


def _worker_roundtrip(descriptor):
    from euler_loading import resolve_transform_descriptor

    from euler_dataset_contract import (
        get_registered_addon_validators,
        parse_dataset_head,
    )

    validators = get_registered_addon_validators()
    assert {"euler_loading", "euler_representation", "euler_transforms"} <= set(
        validators
    )
    result = resolve_transform_descriptor(descriptor)(_sample())
    bad = json.loads(descriptor)
    bad["recipe"]["operations"][0]["version"] = "999.0"
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "d", "name": "d"},
        "modality": {"key": "map_3d"},
        "addons": {"euler_transforms": bad},
    }
    try:
        parse_dataset_head(head)
    except ValueError:
        pass
    else:
        raise AssertionError("Worker accepted an unknown operation version")
    return result["intrinsics"].tolist(), result["depth"][100, 200].item()


def _worker_execute_plan(plan):
    return plan(_sample())["intrinsics"].tolist()


def test_spawned_workers_register_and_execute(fixture):
    descriptor = _preprocessor().export_descriptor(
        **_bindings(fixture), sample=_sample()
    )
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_worker_roundtrip, [descriptor.to_json()] * 2))
        pickled_results = list(
            pool.map(
                _worker_execute_plan, [resolve_transform_descriptor(descriptor)] * 2
            )
        )
    assert results[0] == results[1]
    assert pickled_results == [results[0][0], results[0][0]]
    np.testing.assert_allclose(
        results[0][0], [[640, 0, 319.5], [0, 640, 159.5], [0, 0, 1]]
    )
    assert results[0][1] == pytest.approx(2.981375, abs=1e-5)


def test_unknown_callable_chain_remains_usable_but_not_exportable(tmp_path, fixture):
    def transform(sample):
        return dict(sample, marker=True)

    dataset, _, _ = make_dataset(tmp_path / "dataset", transforms=[transform])
    assert dataset[0]["marker"] is True
    with pytest.raises(ValueError, match="unknown callable"):
        dataset.export_transform_plan(**_bindings(fixture))
    with pytest.raises(ValueError, match="upstream callable"):
        _preprocessor().export_descriptor(
            **_bindings(fixture), upstream_transforms=[transform]
        )


def _rgb_dataset_bindings(tmp_path, fixture):
    from ds_crawler import get_dataset_contract
    from euler_loading.dataset import _get_index_tree

    dataset, _, _ = make_dataset(tmp_path / "dataset")
    sample = dataset[0]
    preprocessor = SamplePreprocessor.from_config(
        {
            "resize": [2, 4],
            "fields": {"rgb": {"kind": "image", "layout": "HWC"}},
            "infer_fields": False,
            "reference_field": "rgb",
        }
    )
    preprocessor.bind_to_dataset(dataset)
    dataset._transforms = [preprocessor]
    bindings = _bindings(fixture)
    bindings["fields"] = {"rgb": bindings["fields"]["rgb"]}
    bindings["fields"]["rgb"]["profile"]["shape"] = [4, 8, 3]
    bindings["sources"] = {"source_rgb": bindings["sources"]["source_rgb"]}
    bindings["image_planes"]["camera_0"]["size"] = [4, 8]
    source = bindings["sources"]["source_rgb"]
    head = get_dataset_contract(dataset._index_outputs["rgb"])
    source.update(
        dataset_id=head.dataset_id,
        modality_key=head.modality_key,
        metadata_scope=".",
        head_digest=canonical_digest(head.to_mapping()),
        index_digest=canonical_digest(_get_index_tree(dataset._index_outputs["rgb"])),
    )
    source["decoder"].update(
        id="euler_loading.loaders.cpu.generic_dense_depth.rgb",
        profile_digest=canonical_digest(bindings["fields"]["rgb"]["profile"]),
    )
    return dataset, bindings, sample


def test_dataset_export_includes_entire_chain(tmp_path, fixture):
    dataset, bindings, sample = _rgb_dataset_bindings(tmp_path, fixture)
    descriptor = dataset.export_transform_plan(**bindings, sample=sample)
    assert descriptor["state"] == "planned"
    assert resolve_transform_descriptor(descriptor)(sample)["rgb"].shape == (2, 4, 3)
    source = bindings["sources"]["source_rgb"]
    source["head_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="source.head_digest"):
        dataset.export_transform_plan(**bindings, sample=sample)
    dataset._transforms.append(lambda sample: sample)
    with pytest.raises(ValueError, match="unknown callable"):
        dataset.export_transform_plan(**bindings, sample=sample)


def test_dataset_export_rejects_wrapped_builtin_decoder(tmp_path, fixture):
    dataset, bindings, sample = _rgb_dataset_bindings(tmp_path, fixture)
    original = dataset._resolved_loaders["rgb"]
    dataset.export_transform_plan(**bindings, sample=sample)

    @wraps(original)
    def scaled_rgb(*args, **kwargs):
        return original(*args, **kwargs) * 2

    # The wrapper remains a valid ordinary loader, including its copied metadata.
    before = dataset[0]["rgb"]
    dataset._resolved_loaders["rgb"] = scaled_rgb
    np.testing.assert_allclose(dataset[0]["rgb"], before * 2)
    with pytest.raises(ValueError, match="decoder callable effects"):
        dataset.export_transform_plan(**bindings, sample=sample)


def _small_label_descriptor(fixture, backend, dtype, operations):
    addon = _small_depth_descriptor(fixture, backend, "legacy_blend")
    profile = addon["recipe"]["fields"]["depth"]["profile"]
    profile.update(kind="generic", dtype=dtype, unit="dimensionless")
    profile.pop("depth_kind")
    profile.pop("frame")
    profile["invalid"]["sentinel"] = None
    addon["recipe"]["operations"] = operations
    return _rehash(addon)


@pytest.mark.parametrize("backend", ["torch_cpu", "pillow_cpu"])
@pytest.mark.parametrize(
    "threshold,expected", [(0.0, True), (0.25, False), (0.5, False)]
)
def test_boolean_threshold_applies_to_interpolated_values(
    fixture, backend, threshold, expected
):
    operations = [
        {
            "id": "resize",
            "op": "euler_loading.resize",
            "version": "1.0",
            "parameters": {"size": [1, 1]},
        }
    ]
    addon = _small_label_descriptor(fixture, backend, "bool", operations)
    addon["recipe"]["fields"]["depth"]["policy"]["mask_threshold"] = threshold
    plan = resolve_transform_descriptor(_rehash(addon))
    result = plan({"depth": np.array([[True, False], [False, False]])})["depth"]
    # Both filters give 0.25 for this symmetric case; compare before conversion.
    assert result.dtype == np.bool_
    assert result.item() is expected


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize(
    "backend,tensors",
    [("torch_cpu", False), ("torch_cpu", True), ("pillow_cpu", False)],
)
@pytest.mark.parametrize(
    "chain", ["crop", "identity_resize", "crop_then_identity_resize"]
)
def test_integer_labels_remain_exact_without_interpolation(
    fixture, dtype, backend, tensors, chain
):
    crop = {
        "id": "crop",
        "op": "euler_loading.crop",
        "version": "1.0",
        "parameters": {"size": [1, 1], "offset": [1, 1]},
    }
    resize = {
        "id": "resize",
        "op": "euler_loading.resize",
        "version": "1.0",
        "parameters": {
            "size": [1, 1] if chain == "crop_then_identity_resize" else [2, 2]
        },
    }
    operations = (
        [crop, resize]
        if chain == "crop_then_identity_resize"
        else [crop if chain == "crop" else resize]
    )
    addon = _small_label_descriptor(fixture, backend, dtype, operations)
    addon["recipe"]["fields"]["depth"]["profile"].update(
        kind="mask", meaning="class IDs"
    )
    addon["recipe"]["fields"]["depth"]["policy"]["interpolation"] = "nearest"
    plan = resolve_transform_descriptor(_rehash(addon))
    source = (np.arange(4, dtype=dtype) + 2**24 + 1).reshape(2, 2)
    expected = source if chain == "identity_resize" else source[1:, 1:]
    sample = torch.from_numpy(source) if tensors else source
    result = plan({"depth": sample})["depth"]
    np.testing.assert_array_equal(as_numpy(result), expected)
    assert result.dtype == sample.dtype
    result[...] = 0
    assert source[1, 1] == 2**24 + 4  # Crops must not expose the cached input.


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("backend", ["torch_cpu", "pillow_cpu"])
def test_integer_labels_still_reject_resize_after_crop(fixture, dtype, backend):
    operations = [
        {
            "id": "crop",
            "op": "euler_loading.crop",
            "version": "1.0",
            "parameters": {"size": [1, 1]},
        },
        {
            "id": "resize",
            "op": "euler_loading.resize",
            "version": "1.0",
            "parameters": {"size": [2, 2]},
        },
    ]
    # Final and source shapes match, but the intermediate resize still interpolates.
    addon = _small_label_descriptor(fixture, backend, dtype, operations)
    addon["recipe"]["fields"]["depth"]["policy"]["interpolation"] = "nearest"
    with pytest.raises(ValueError, match="int32/int64"):
        resolve_transform_descriptor(_rehash(addon))


@pytest.mark.parametrize("backend,expected", [("torch_cpu", 2), ("pillow_cpu", 1)])
def test_integer_rounding_policy_is_preserved(fixture, backend, expected):
    operations = [
        {
            "id": "resize",
            "op": "euler_loading.resize",
            "version": "1.0",
            "parameters": {"size": [1, 1]},
        }
    ]
    plan = resolve_transform_descriptor(
        _small_label_descriptor(fixture, backend, "uint8", operations)
    )
    result = plan({"depth": np.arange(4, dtype=np.uint8).reshape(2, 2)})["depth"]
    assert result.dtype == np.uint8
    assert result.item() == expected  # The unrounded bilinear value is 1.5.


def test_schema_generated_from_contract_accepts_export(fixture):
    # jsonschema is a development dependency only, never used by the executor.
    from jsonschema import Draft202012Validator

    descriptor = _preprocessor().export_descriptor(
        **_bindings(fixture), sample=_sample()
    )
    Draft202012Validator(build_descriptor_schema("euler_transforms")).validate(
        descriptor.to_dict()
    )


@pytest.mark.parametrize(
    "config",
    [
        {"resize": [384.1, 768]},
        {"resize": [True, 768]},
        {"resize": [384, 768], "operations": [{"type": "resize", "size": [384, 768]}]},
        {"resize": {"size": [384, 768], "target_size": [300, 600]}},
        {"resize": [384, 768], "unknown_effect": "something"},
    ],
)
def test_legacy_config_coercion_does_not_leak_into_strict_export(fixture, config):
    cfg = evidence("documented-five-field")["config"]
    cfg.update(config)
    preprocessor = SamplePreprocessor.from_config(
        cfg
    )  # Legacy authoring still accepts.
    with pytest.raises(ValueError, match="authoring|Authoring"):
        preprocessor.export_descriptor(**_bindings(fixture), sample=_sample())


def test_default_shorthand_and_explicit_operations_have_same_identity(fixture):
    config = evidence("documented-five-field")["config"]
    config["operations"] = [
        {"type": "resize", "size": config.pop("resize")},
        {"type": "crop", "size": config.pop("crop")["size"]},
    ]
    explicit = SamplePreprocessor.from_config(config).export_descriptor(
        **_bindings(fixture)
    )
    assert explicit == _preprocessor().export_descriptor(**_bindings(fixture))


@pytest.mark.parametrize("layout", ["HW", "CHW", "HWC", "NHW", "NCHW", "NHWC"])
@pytest.mark.parametrize("backend", ["torch_cpu", "pillow_cpu"])
def test_explicit_layouts_preserve_nonspatial_axes(fixture, layout, backend):
    addon = _small_depth_descriptor(fixture, backend, "legacy_blend")
    dimensions = {"N": 2, "C": 1, "H": 2, "W": 2}
    shape = [dimensions[axis] for axis in layout]
    profile = addon["recipe"]["fields"]["depth"]["profile"]
    profile.update(layout=layout, shape=shape)
    plan = resolve_transform_descriptor(_rehash(addon))
    result = plan({"depth": np.full(shape, 4, np.float32)})["depth"]
    expected_shape = [1 if axis in "HW" else dimensions[axis] for axis in layout]
    assert list(result.shape) == expected_shape
    np.testing.assert_array_equal(result, np.full(expected_shape, 4, np.float32))


@pytest.mark.parametrize(
    "backend,observation", [("torch_cpu", "torch_cpu"), ("pillow_cpu", "pillow")]
)
def test_pinned_backend_matches_shared_sampling_evidence(fixture, backend, observation):
    data = evidence("backend-resize")
    addon = _small_depth_descriptor(fixture, backend, "legacy_blend")
    recipe = addon["recipe"]
    binding = recipe["fields"].pop("depth")
    profile = binding["profile"]
    profile.pop("depth_kind")
    profile.update(kind="image", shape=[1, 4], unit="dimensionless")
    recipe["fields"] = {"image": binding}
    recipe["reference_field"] = "image"
    recipe["image_planes"]["camera_0"]["size"] = [1, 4]
    recipe["operations"][0]["parameters"]["size"] = [1, 2]
    plan = resolve_transform_descriptor(_rehash(addon))
    actual = plan({"image": np.array(data["inputs"]["image"], np.float32)})["image"]
    np.testing.assert_allclose(
        actual, data["observed"][observation]["image"], atol=data["atol"], rtol=0
    )
    profile.update(kind="mask", dtype="bool", meaning="source membership")
    binding["policy"]["interpolation"] = "nearest"
    plan = resolve_transform_descriptor(_rehash(addon))
    actual = plan({"image": np.array(data["inputs"]["mask"], np.bool_)})["image"]
    np.testing.assert_array_equal(actual, data["observed"][observation]["mask"])


@pytest.mark.parametrize(
    "backend,tensors",
    [("torch_cpu", False), ("torch_cpu", True), ("pillow_cpu", False)],
)
def test_validity_aware_restores_sentinel_in_output_dtype(fixture, backend, tensors):
    addon = _small_depth_descriptor(fixture, backend, "validity_aware")
    profile = addon["recipe"]["fields"]["depth"]["profile"]
    profile["dtype"] = "float64"
    profile["invalid"]["sentinel"] = 0.1
    plan = resolve_transform_descriptor(_rehash(addon))
    sample = np.full((2, 2), 0.1, dtype=np.float64)
    if tensors:
        sample = torch.from_numpy(sample)
    result = plan({"depth": sample})["depth"]
    assert as_numpy(result).dtype == np.float64
    assert result.item() == 0.1


def test_output_plane_binds_sources_without_hashing_locators(fixture):
    addon = fixture["addon"]
    original = resolve_transform_descriptor(addon).infer_output_profiles()["rgb"][
        "image_plane"
    ]
    addon["sources"]["source_rgb"]["locator"] = "/moved/input"
    assert (
        resolve_transform_descriptor(addon).infer_output_profiles()["rgb"][
            "image_plane"
        ]
        == original
    )
    addon["sources"]["source_rgb"]["revision"] = canonical_digest({"revision": 2})
    assert (
        resolve_transform_descriptor(addon).infer_output_profiles()["rgb"][
            "image_plane"
        ]
        != original
    )
