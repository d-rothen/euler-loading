"""Real capture/encoding/persistence/replay boundaries using the shared corpus."""

import functools
import multiprocessing
from copy import deepcopy

import numpy as np
import pytest
from ds_crawler import (
    copy_dataset,
    read_record,
)
from ds_crawler.zip_utils import read_metadata_json
from euler_dataset_contract import TransformsAddon, recipe_digest, resolve_receipt
from euler_dataset_contract.testing.checks._phase2_support import make_source
from euler_dataset_contract.testing.checks._support import assert_preprocessed

from euler_loading import (
    CalibrationView,
    Modality,
    MultiModalDataset,
    output_encoding,
    replay_from_arrays,
    write_materialized,
)


def _worker_sample(dataset):
    return dataset[0]


@pytest.mark.parametrize(
    "zip_output,scope,sidecar",
    [
        (False, None, False),
        (False, "rgb", True),
        (True, None, True),
        (True, "view", False),
    ],
)
def test_five_field_write_reload_replay(tmp_path, zip_output, scope, sidecar):
    ds, descriptor, data, original = make_source(tmp_path / "source")
    before = {
        name: ds.get_modality_index(name) for name in descriptor["recipe"]["fields"]
    }
    ds.enable_transform_capture({"default": descriptor})
    sample = ds[0]
    assert_preprocessed(sample, data)
    assert sample["depth"][100, 200] == pytest.approx(2.981375, abs=1e-6)
    np.testing.assert_allclose(
        sample["intrinsics"], [[640, 0, 319.5], [0, 640, 159.5], [0, 0, 1]]
    )
    original["intrinsics"] = {
        "calibration": next(iter(original["intrinsics"].values()))
    }
    replay = replay_from_arrays(descriptor, sample["provenance"]["execution"], original)
    for field in descriptor["recipe"]["fields"]:
        writer = ds.create_output_writer(
            field,
            tmp_path / (field + (".zip" if zip_output else "")),
            derivation=descriptor,
            dataset_id="derived_" + field,
            revision="v1",
            metadata_scope=scope,
            expected_full_ids=["/changed/frame/a"],
        )
        ds.write_sample(
            0,
            {field: sample[field]},
            writer,
            provenance=sample["provenance"],
            output_full_ids={field: "/changed/frame/a"},
            sidecar=sidecar,
            attributes={field: {"note": "retained"}},
        )
        writer.finalize()
        out = MultiModalDataset(
            {field: Modality(str(writer.destination), metadata_scope=scope)}
        )
        loaded = out[0]
        assert loaded["full_id"] == "/changed/frame/a"
        np.testing.assert_array_equal(loaded[field], sample[field])
        np.testing.assert_array_equal(replay[field], sample[field])
        location = loaded["attributes"][field]["euler_transforms"]
        receipt = resolve_receipt(
            location,
            lambda p, root=writer.destination: read_record(
                root, p, metadata_scope=scope
            ),
        )
        assert receipt["output_full_id"] == loaded["full_id"]
        assert loaded["attributes"][field]["note"] == "retained"
        assert receipt["execution"]["operations"][1]["offset"] == [32, 64]
        head = out.get_modality_index(field)["head"]
        assert head["addons"]["euler_transforms"]["plan"]["field"] == field
        if field != "intrinsics":
            assert head["modality"]["meta"]["dimensions"]["height"] == 320
        np.testing.assert_array_equal(out[0][field], loaded[field])
    for name, value in before.items():
        assert ds.get_modality_index(name) == value
    camera = original["intrinsics"]["calibration"]
    view = CalibrationView(descriptor, sample["provenance"]["execution"], "intrinsics")
    np.testing.assert_array_equal(view.resolve(camera), sample["intrinsics"])
    assert view.cardinality == "one_per_output" and "rgb" in view.applies_to
    np.testing.assert_array_equal(camera, [[800, 0, 479.5], [0, 800, 239.5], [0, 0, 1]])


@pytest.mark.parametrize(
    "field,codec,tolerance",
    [
        ("rgb", "png_rgb8", 1 / 510 + 1e-7),
        ("depth", "png_depth16", 0.000501),
        ("valid_mask", "png_mask8", 0),
    ],
)
def test_explicit_png_output_metadata(tmp_path, field, codec, tolerance):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=(field != "rgb"))
    ds.enable_transform_capture({"default": plan})
    sample = ds[0]
    writer = ds.create_output_writer(
        field,
        tmp_path / "out",
        derivation=plan,
        dataset_id="derived",
        revision="v1",
        encoding=output_encoding(codec),
    )
    write_materialized(
        writer, sample[field], sample["provenance"], output_full_id="/variant/new"
    )
    writer.finalize()
    out = MultiModalDataset({field: Modality(str(tmp_path / "out"))})
    loaded = out[0]
    np.testing.assert_allclose(loaded[field], sample[field], atol=tolerance, rtol=0)
    assert loaded["attributes"][field]["note"] == "preserved"
    head = out.get_modality_index(field)["head"]
    assert head["modality"]["meta"]["file_types"] == ["png"]
    if field == "depth":
        assert head["modality"]["meta"]["scale_to_meters"] == 0.001
        assert head["modality"]["meta"]["range"] == [0, 65.535]


def test_failure_resume_conflicting_values_and_wrong_sample(tmp_path, monkeypatch):
    from euler_loading import materialization

    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    ds.enable_transform_capture({"default": plan})
    sample = ds[0]
    args = {
        "derivation": plan,
        "dataset_id": "derived",
        "revision": "v1",
        "expected_full_ids": [sample["full_id"], "/second"],
    }
    writer = ds.create_output_writer("rgb", tmp_path / "out", **args)
    with pytest.raises(ValueError, match="missing captured"):
        ds.write_sample(0, {"rgb": sample["rgb"]}, writer)
    with pytest.raises(ValueError, match="sample index"):
        ds.write_sample(
            1, {"rgb": sample["rgb"]}, writer, provenance=sample["provenance"]
        )
    camera_writer = ds.create_output_writer("intrinsics", tmp_path / "camera", **args)
    with pytest.raises(ValueError, match="sample index"):
        ds.write_sample(
            1,
            {"intrinsics": sample["intrinsics"]},
            camera_writer,
            provenance=sample["provenance"],
        )
    with pytest.raises(ValueError, match="differs from captured"):
        write_materialized(
            writer,
            sample["rgb"] + 1,
            sample["provenance"],
            output_full_id=sample["full_id"],
        )
    real_encoder = materialization.encode_output

    def fail(*args, **kwargs):
        raise OSError("encoder failed")

    monkeypatch.setattr(materialization, "encode_output", fail)
    with pytest.raises(OSError, match="encoder failed"):
        ds.write_sample(
            0, {"rgb": sample["rgb"]}, writer, provenance=sample["provenance"]
        )
    assert len(writer) == 0
    assert read_metadata_json(tmp_path / "out", "dataset-head.json") is None
    monkeypatch.setattr(materialization, "encode_output", real_encoder)
    ds.write_sample(0, {"rgb": sample["rgb"]}, writer, provenance=sample["provenance"])
    with pytest.raises(ValueError, match="incomplete output"):
        writer.finalize()
    assert read_metadata_json(tmp_path / "out", "dataset-head.json") is None
    write_materialized(
        writer, sample["rgb"], sample["provenance"], output_full_id="/second"
    )
    writer.finalize()
    writer.finalize()
    resumed = ds.create_output_writer("rgb", tmp_path / "out", resume=True, **args)
    ds.write_sample(0, {"rgb": sample["rgb"]}, resumed, provenance=sample["provenance"])
    write_materialized(
        resumed,
        np.asfortranarray(sample["rgb"]),
        sample["provenance"],
        output_full_id="/second",
    )
    assert len(resumed) == 2
    resumed.finalize()
    with pytest.raises(ValueError, match="conflicting rewrite"):
        write_materialized(
            resumed,
            sample["rgb"],
            sample["provenance"],
            output_full_id="/second",
            attributes={"changed": True},
        )
    with pytest.raises(ValueError, match="conflicts"):
        ds.create_output_writer(
            "rgb", tmp_path / "out", resume=True, **{**args, "revision": "v2"}
        )


def test_variants_variable_size_and_equal_size_different_pixels(tmp_path):
    ds, base, _, _ = make_source(tmp_path / "source", tiny=True)
    crop = base.to_dict()
    crop["recipe"]["operations"] = [
        {
            "id": "crop",
            "op": "euler_loading.crop",
            "version": "1.0",
            "parameters": {"size": [2, 2], "offset": [1, 1]},
        }
    ]
    crop["recipe_digest"] = recipe_digest(crop["recipe"])
    crop = TransformsAddon(crop)
    larger = crop.to_dict()
    larger["recipe"]["operations"][0]["parameters"]["size"] = [3, 3]
    larger["recipe_digest"] = recipe_digest(larger["recipe"])
    larger = TransformsAddon(larger)
    variants = {"resize_crop": base, "crop": crop, "larger": larger}
    ds.enable_transform_capture(variants)
    writer = ds.create_output_writer(
        "rgb",
        tmp_path / "out",
        derivation=variants,
        dataset_id="derived",
        revision="v1",
    )
    samples = {}
    for variant in variants:
        sample = ds.capture_sample(0, variant_id=variant)
        samples[variant] = sample
        write_materialized(
            writer,
            sample["rgb"],
            sample["provenance"],
            output_full_id="/sample/" + variant,
        )
    assert samples["crop"]["rgb"].shape == samples["resize_crop"]["rgb"].shape
    assert not np.array_equal(samples["crop"]["rgb"], samples["resize_crop"]["rgb"])
    writer.finalize()
    out = MultiModalDataset({"rgb": Modality(str(tmp_path / "out"))})
    assert len(out) == 3
    assert "dimensions" not in out.get_modality_index("rgb")["head"]["modality"]["meta"]
    assert sorted(tuple(out[i]["rgb"].shape) for i in range(len(out))) == [
        (2, 2, 3),
        (2, 2, 3),
        (3, 3, 3),
    ]


def test_changed_source_bytes_and_opaque_decoder_fail(tmp_path):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    original_loader = ds._resolved_loaders["rgb"]

    @functools.wraps(original_loader)
    def wrapped(*args, **kwargs):
        return original_loader(*args, **kwargs) * 2

    ds._resolved_loaders["rgb"] = wrapped
    with pytest.raises(ValueError, match="unknown decoder"):
        ds.enable_transform_capture({"default": plan})
    ds._resolved_loaders["rgb"] = original_loader
    capture = ds.enable_transform_capture({"default": plan})
    capture(0)
    ds._resolved_loaders["rgb"] = wrapped
    with pytest.raises(ValueError, match="changed after capture"):
        capture(0)
    ds._resolved_loaders["rgb"] = original_loader
    path = next((tmp_path / "source/rgb").rglob("0.npy"))
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="source bytes changed"):
        capture(0)
    with pytest.raises(ValueError, match="manifest mismatch"):
        ds.enable_transform_capture({"default": plan})


def test_missing_corrupt_receipt_and_relocation(tmp_path):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    ds.enable_transform_capture({"default": plan})
    writer = ds.create_output_writer(
        "rgb",
        tmp_path / "out",
        derivation=plan,
        dataset_id="derived",
        revision="v1",
        metadata_scope="a",
    )
    for i in range(len(ds)):
        sample = ds[i]
        write_materialized(
            writer,
            sample["rgb"],
            sample["provenance"],
            output_full_id=sample["full_id"],
            sidecar=True,
        )
    writer.finalize()
    copy_dataset(
        tmp_path / "out",
        tmp_path / "copy.zip",
        input_metadata_scope="a",
        output_metadata_scope="b",
        sample=2,
    )
    out = MultiModalDataset(
        {"rgb": Modality(str(tmp_path / "copy.zip"), metadata_scope="b")}
    )
    assert len(out) == 1
    np.testing.assert_array_equal(out[0]["rgb"], ds[0]["rgb"])
    record = next((tmp_path / "out/.ds_crawler/a/generations").rglob("records/*.json"))
    record.write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        MultiModalDataset({"rgb": Modality(str(tmp_path / "out"), metadata_scope="a")})
    record.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        copy_dataset(tmp_path / "out", tmp_path / "bad", input_metadata_scope="a")
    assert not (tmp_path / "bad").exists()


def test_spawned_capture_is_repeatable(tmp_path):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    ds.enable_transform_capture({"default": plan})
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        worker = pool.apply(_worker_sample, (ds,))
    main = ds[0]
    assert worker["provenance"] == main["provenance"] == ds[0]["provenance"]
    np.testing.assert_array_equal(worker["rgb"], main["rgb"])
    main["intrinsics"][:] = 0
    assert ds[0]["intrinsics"][0, 0] > 0


def test_multiple_calibrations_select_exact_sensor_and_vary_per_frame(tmp_path):
    from ds_crawler import DatasetWriter

    from euler_loading import SamplePreprocessor, execution_profile
    from euler_loading.output_encoding import encode_output
    from euler_loading.receipts import _digest_bytes, array_digest

    ds, original_plan, data, original = make_source(tmp_path / "source", tiny=True)
    root = tmp_path / "source" / "intrinsics"
    head = ds.get_modality_index("intrinsics")["head"]
    profile = original_plan["recipe"]["fields"]["intrinsics"]["profile"]
    camera = next(iter(original["intrinsics"].values()))
    camera2 = camera.copy()
    camera2[0, 1] = 2
    writer = DatasetWriter(root, head=head, separator=None)
    for name, matrix in [("calibration", camera), ("other_sensor", camera2)]:
        raw, decoded, profiles = encode_output(matrix, output_encoding(), profile)
        attributes = {
            "output_encoding": {
                "encoding": output_encoding().to_dict(),
                "decoded": profiles["decoded"],
                "artifact_digest": _digest_bytes(raw),
                "decoded_digest": array_digest(decoded),
            }
        }
        writer.get_path(
            "/scene_01/camera_0/" + name, name + ".npy", attributes=attributes
        ).write_bytes(raw)
    writer.save_index()
    ds = MultiModalDataset(
        {
            field: Modality(str(tmp_path / "source" / field))
            for field in ("rgb", "depth", "valid_mask", "ray_map")
        },
        hierarchical_modalities={"intrinsics": Modality(str(root))},
    )
    variants = {}
    for name in ("calibration", "other_sensor"):
        fields = original_plan["recipe"]["fields"]
        fields["intrinsics"]["selection"] = name
        fields["intrinsics"]["calibration"]["full_id"] = "/scene_01/camera_0/" + name
        sources = ds.describe_transform_sources(
            fields,
            revisions={
                alias: source["revision"]
                for alias, source in original_plan["sources"].items()
            },
        )
        variants[name] = SamplePreprocessor.from_config(
            data["config"]
        ).export_descriptor(
            fields=fields,
            sources=sources,
            image_planes=original_plan["recipe"]["image_planes"],
            execution=execution_profile("torch_cpu"),
        )
    ds.enable_transform_capture(variants)
    left = ds.capture_sample(0, variant_id="calibration")
    right = ds.capture_sample(1, variant_id="other_sensor")
    assert left["provenance"]["execution"]["source_full_ids"][
        "source_intrinsics"
    ].endswith("/calibration")
    assert right["provenance"]["execution"]["source_full_ids"][
        "source_intrinsics"
    ].endswith("/other_sensor")
    assert left["intrinsics"][0, 1] == 0
    assert right["intrinsics"][0, 1] == 1
    # A receipt explicitly selecting another sensor cannot be relabelled later.
    bad = deepcopy(right["provenance"])
    bad["execution"]["source_content"]["source_intrinsics"] = "sha256:" + "f" * 64
    out = ds.create_output_writer(
        "rgb",
        tmp_path / "out",
        derivation=variants,
        dataset_id="derived",
        revision="v1",
    )
    with pytest.raises(ValueError, match="replay disagrees"):
        write_materialized(out, right["rgb"], bad, output_full_id="/frame/variant")
    np.testing.assert_array_equal(camera, next(iter(original["intrinsics"].values())))


@pytest.mark.parametrize("missing_record", [False, True])
def test_capture_refuses_missing_or_mismatched_calibration_selection(
    tmp_path, missing_record
):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    changed = plan.to_dict()
    camera = changed["recipe"]["fields"]["intrinsics"]
    if missing_record:
        camera["calibration"]["full_id"] = "/unknown/sensor/calibration"
    else:
        camera["selection"] = "wrong_sensor"
    changed["recipe_digest"] = recipe_digest(changed["recipe"])
    ds.enable_transform_capture({"default": changed})
    with pytest.raises(ValueError, match="selected calibration|calibration selection"):
        ds[0]


def test_source_changed_after_successful_write_refuses_publication(tmp_path):
    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    ds.enable_transform_capture({"default": plan})
    sample = ds[0]
    writer = ds.create_output_writer(
        "rgb", tmp_path / "out", derivation=plan, dataset_id="derived", revision="v1"
    )
    write_materialized(
        writer, sample["rgb"], sample["provenance"], output_full_id=sample["full_id"]
    )
    source = next((tmp_path / "source/rgb").rglob("0.npy"))
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source bytes changed"):
        writer.finalize()
    assert read_metadata_json(tmp_path / "out", "dataset-head.json") is None


def test_depth_quantization_preserves_invalid_pixel_classification(tmp_path):
    from euler_loading.output_encoding import encode_output

    _, descriptor, _, _ = make_source(tmp_path / "source", tiny=True)
    profile = descriptor["recipe"]["fields"]["depth"]["profile"]
    values = np.ones(profile["shape"], dtype=np.float32)
    profile["invalid"]["sentinel"] = 0.0004
    with pytest.raises(ValueError, match="cannot preserve.*sentinel"):
        encode_output(values, output_encoding("png_depth16"), profile)
    profile["invalid"]["sentinel"] = 0
    values[0, 0] = 0.0004
    with pytest.raises(ValueError, match="invalid-pixel classification"):
        encode_output(values, output_encoding("png_depth16"), profile)
    values[0, 0] = 0
    _, decoded, _ = encode_output(values, output_encoding("png_depth16"), profile)
    np.testing.assert_array_equal(values == 0, decoded == 0)


def test_shared_geometry_preserves_legacy_integer_camera_values():
    from euler_loading import resize_intrinsics

    K = np.array([[800, 2, 479], [0, 800, 239], [0, 0, 1]], dtype=np.int64)
    before = K.copy()
    scaled = resize_intrinsics(K, source_size=(480, 960), target_size=(384, 768))
    np.testing.assert_array_equal(scaled, [[640, 1, 383], [0, 640, 191], [0, 0, 1]])
    np.testing.assert_array_equal(K, before)


def test_output_cannot_replace_source_head_or_trust_hashes_alone(tmp_path):
    from euler_loading import create_dataset_writer_from_index

    ds, plan, _, _ = make_source(tmp_path / "source", tiny=True)
    with pytest.raises(ValueError, match="source-backed DatasetCapture"):
        create_dataset_writer_from_index(
            index_output=ds.get_modality_index("rgb"),
            root=tmp_path / "out",
            derivation=plan,
            field="rgb",
            dataset_id="derived",
            revision="v1",
        )
    ds.enable_transform_capture({"default": plan})
    with pytest.raises(ValueError, match="source roots"):
        ds.create_output_writer(
            "rgb",
            tmp_path / "source/rgb",
            derivation=plan,
            dataset_id="derived",
            revision="v1",
        )


def test_chained_materialized_reload_executes_only_new_crop(tmp_path):
    from euler_loading import SamplePreprocessor, execution_profile

    ds, first, _, _ = make_source(tmp_path / "source", tiny=True)
    ds.enable_transform_capture({"default": first})
    sample = ds[0]
    writer = ds.create_output_writer(
        "rgb",
        tmp_path / "first",
        derivation=first,
        dataset_id="first_rgb",
        revision="v1",
    )
    write_materialized(
        writer, sample["rgb"], sample["provenance"], output_full_id="/frame"
    )
    writer.finalize()
    reloaded = MultiModalDataset({"rgb": Modality(str(tmp_path / "first"))})
    profile = reloaded[0]["attributes"]["rgb"]["output_encoding"]["decoded"]
    fields = {
        "rgb": {
            "source": "parent_rgb",
            "profile": profile,
            "policy": {"interpolation": "bilinear"},
        }
    }
    sources = reloaded.describe_transform_sources(
        fields, revisions={"parent_rgb": "sha256:" + "b" * 64}
    )
    second = SamplePreprocessor.from_config(
        {
            "crop": {"size": [1, 1], "offset": [1, 1]},
            "fields": {"rgb": {"kind": "image", "layout": "HWC"}},
            "reference_field": "rgb",
            "infer_fields": False,
        }
    ).export_descriptor(
        fields=fields,
        sources=sources,
        image_planes={
            profile["image_plane"]: {
                "size": [2, 2],
                "frame": "camera_0_optical",
                "camera_model": "pinhole",
            }
        },
        execution=execution_profile("torch_cpu"),
    )
    reloaded.enable_transform_capture({"second": second})
    result = reloaded[0]
    np.testing.assert_array_equal(result["rgb"], sample["rgb"][1:2, 1:2])
    assert len(result["provenance"]["execution"]["operations"]) == 1
    assert result["provenance"]["execution"]["source_full_ids"] == {
        "parent_rgb": "/frame"
    }
