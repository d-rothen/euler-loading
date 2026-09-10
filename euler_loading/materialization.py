"""Freeze one logical output, then bind successful encodings to executions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ds_crawler import ValidatedDatasetWriter, get_dataset_contract
from euler_dataset_contract import (
    ArtifactReceipt,
    OutputPlan,
    TransformsAddon,
    canonical_digest,
    receipt_location,
    validate_execution,
)

from .output_encoding import encode_output, encoding_profiles, output_encoding
from .receipts import _digest_bytes, array_digest
from .transform_descriptors import _validate_value, resolve_transform_descriptor


def create_materialized_writer(
    *,
    index_output,
    root,
    derivation,
    field,
    dataset_id,
    revision,
    encoding=None,
    metadata_scope=None,
    expected_full_ids=None,
    resume=False,
    capture=None,
):
    from .receipts import DatasetCapture

    if type(capture) is not DatasetCapture:
        raise ValueError(
            "strict materialization requires a source-backed DatasetCapture verifier"
        )
    variants = (
        {"default": derivation}
        if isinstance(derivation, TransformsAddon) or "recipe" in derivation
        else derivation
    )
    variants = {name: TransformsAddon(value) for name, value in variants.items()}
    encoding = output_encoding() if encoding is None else encoding
    profiles = {
        variant: encoding_profiles(
            resolve_transform_descriptor(plan).infer_output_profiles()[field], encoding
        )
        for variant, plan in variants.items()
    }
    for modality in {
        **capture.dataset._modalities,
        **capture.dataset._hierarchical_modalities,
    }.values():
        if Path(root).resolve() == Path(modality.path).resolve():
            raise ValueError(
                "output root must differ from source roots; source heads are immutable"
            )
    head = deepcopy(get_dataset_contract(dict(index_output)).to_mapping())
    head["dataset"]["id"] = dataset_id
    head["dataset"]["name"] = dataset_id
    head["dataset"].setdefault("attributes", {})["revision"] = revision
    head.setdefault("addons", {}).pop("euler_transforms", None)
    head["addons"].pop("euler_representation", None)
    head["addons"].pop("euler_layout", None)
    head["addons"]["euler_loading"] = {
        "version": "1.0",
        "loader": "materialized",
        "function": "data",
    }
    # Rebuild encoding-sensitive metadata instead of inheriting source dimensions,
    # scale/range or a scene-wide source calibration claim.
    meta = {}
    selected_profiles = [p["decoded"] for p in profiles.values()]
    profile = selected_profiles[0]
    kind = profile["kind"]
    if kind == "depth":
        if any(p["unit"] != "meter" for p in selected_profiles):
            raise ValueError(
                "output depth metadata requires decoded unit meter; implicit unit conversion is unsupported"
            )
        if any(p.get("depth_kind") != profile["depth_kind"] for p in selected_profiles):
            raise ValueError(
                "one depth output requires a consistent depth kind across variants"
            )
        meta.update(
            radial_depth=profile["depth_kind"] == "radial",
            scale_to_meters=encoding["scale"],
            range=encoding["range"],
        )
    if head["modality"]["key"] == "rgb":
        meta["range"] = [0, 255] if encoding["id"] == "png_rgb8" else encoding["range"]
    shapes = {tuple(p["shape"]) for p in selected_profiles}
    if len(shapes) == 1 and len({p["layout"] for p in selected_profiles}) == 1:
        axes = {"H": "height", "W": "width", "C": "channels"}
        dims = {
            axis: profile["shape"][profile["layout"].index(letter)]
            for letter, axis in axes.items()
            if letter in profile["layout"]
        }
        if dims:
            meta["dimensions"] = dims
    meta["file_types"] = ["npy" if encoding["id"] == "npy" else "png"]
    head["modality"]["meta"] = meta
    plan = OutputPlan(
        {
            "version": "2.0",
            "dataset_id": dataset_id,
            "revision": revision,
            "modality_key": head["modality"]["key"],
            "field": field,
            "metadata_scope": metadata_scope or ".",
            "encoding": dict(encoding),
            "profiles": profiles,
            "dependencies": {
                variant: sorted(
                    {field}
                    | {
                        name
                        for name, binding in desc["recipe"]["fields"].items()
                        if field in binding.get("calibration", {}).get("applies_to", [])
                    }
                )
                for variant, desc in variants.items()
            },
            "variants": {k: v.to_dict() for k, v in variants.items()},
        }
    )

    def verify_publication(output, read_record, read_bytes):
        from ds_crawler.records import iter_entries
        from euler_dataset_contract import resolve_receipt

        for _, entry in iter_entries(output["index"]):
            receipt = resolve_receipt(
                entry["attributes"]["euler_transforms"], read_record
            )
            provenance = {
                "version": "2.0",
                "variant_id": receipt["variant_id"],
                "descriptor": plan["variants"][receipt["variant_id"]],
                "execution": receipt["execution"],
            }
            replayed = capture.verify_provenance(provenance)
            raw, _, _ = encode_output(
                replayed[field],
                plan["encoding"],
                plan["profiles"][receipt["variant_id"]]["decoded"],
            )
            if _digest_bytes(raw) != receipt["artifact"]["digest"] or raw != read_bytes(
                entry["path"]
            ):
                raise ValueError(
                    "encoded artifact disagrees with source-backed replay before publication"
                )
            validate_encoding_attributes(entry["attributes"], receipt, plan)

    writer = ValidatedDatasetWriter(
        root,
        head=head,
        plan=plan,
        metadata_scope=metadata_scope,
        publication_validator=verify_publication,
        expected_full_ids=expected_full_ids,
        resume=resume,
        separator=index_output.get("indexing", {})
        .get("hierarchy", {})
        .get("separator"),
    )
    writer.execution_verifier = capture.verify_provenance
    return writer


def write_materialized(
    writer,
    value,
    provenance,
    *,
    output_full_id,
    output_basename=None,
    attributes=None,
    sidecar=False,
):
    if not isinstance(writer, ValidatedDatasetWriter) or writer.output_plan is None:
        raise ValueError(
            "materialization requires a validated writer with a frozen output plan"
        )
    if not provenance or provenance.get("version") != "2.0":
        raise ValueError("missing captured execution provenance")
    plan = writer.output_plan
    variant = provenance["variant_id"]
    descriptor = plan["variants"].get(variant)
    if descriptor is None or descriptor != provenance["descriptor"]:
        raise ValueError("producer changed the frozen recipe/variant/source revisions")
    verifier = getattr(writer, "execution_verifier", None)
    if verifier is None:
        raise ValueError(
            "strict materialization requires source-backed execution verification"
        )
    verified_sample = verifier(provenance)
    execution = validate_execution(provenance["execution"], descriptor)
    field = plan["field"]
    if execution["verification"] != "content":
        raise ValueError("materialization requires content-verified producer execution")
    if array_digest(value) != execution["output_digests"][field]:
        raise ValueError(
            "output value differs from captured execution; opaque producer effects are not replayable"
        )
    profile = plan["profiles"][variant]["decoded"]
    _validate_value(field, value, profile, descriptor["recipe"]["execution"])
    raw, decoded, profiles = encode_output(value, plan["encoding"], profile)
    artifact = {
        "digest": _digest_bytes(raw),
        "decoded_digest": array_digest(decoded),
        **profiles,
    }
    receipt = ArtifactReceipt(
        {
            "version": "2.0",
            "plan_digest": canonical_digest(plan.to_dict()),
            "variant_id": variant,
            "output_full_id": output_full_id,
            "execution": execution.to_dict(),
            "artifact": artifact,
        }
    )
    location, records = receipt_location(receipt, sidecar=sidecar)
    attrs = deepcopy(verified_sample.get("attributes", {}).get(field, {}))
    # Parent history stays in its source binding; this artifact has one current
    # authoritative receipt and codec. Other source attributes remain available.
    attrs.pop("euler_transforms", None)
    attrs.pop("output_encoding", None)
    attrs.update(deepcopy(attributes or {}))
    if "euler_transforms" in attrs and attrs["euler_transforms"] != location:
        raise ValueError("contradictory per-file receipt")
    attrs["euler_transforms"] = location
    encoded_attributes = {
        "encoding": plan["encoding"],
        "decoded": profiles["decoded"],
        "artifact_digest": artifact["digest"],
        "decoded_digest": artifact["decoded_digest"],
    }
    if "output_encoding" in attrs and attrs["output_encoding"] != encoded_attributes:
        raise ValueError("contradictory per-file output encoding metadata")
    attrs["output_encoding"] = encoded_attributes
    suffix = ".npy" if plan["encoding"]["id"] == "npy" else ".png"
    basename = output_basename or output_full_id.rsplit("/", 1)[-1] + suffix
    if Path(basename).suffix.lower() != suffix:
        raise ValueError("output filename extension disagrees with encoding")
    return writer.commit_bytes(
        output_full_id, basename, raw, attributes=attrs, records=records
    )


def validate_encoding_attributes(attributes, receipt, plan):
    expected = {
        "encoding": plan["encoding"],
        "decoded": receipt["artifact"]["decoded"],
        "artifact_digest": receipt["artifact"]["digest"],
        "decoded_digest": receipt["artifact"]["decoded_digest"],
    }
    if attributes.get("output_encoding") != expected:
        raise ValueError(
            "per-file output encoding metadata contradicts materialized receipt/plan"
        )


def validate_materialized_output(root, output, *, metadata_scope=None, subset=False):
    """Check persisted records and codec metadata without executing history."""
    from ds_crawler import validate_output_records
    from ds_crawler.records import iter_entries, read_record
    from euler_dataset_contract import resolve_receipt

    addon = output.get("head", {}).get("addons", {}).get("euler_transforms", {})
    if addon.get("version") != "2.0":
        return
    validate_output_records(root, output, metadata_scope=metadata_scope, subset=subset)
    for _, entry in iter_entries(output["index"]):
        receipt = resolve_receipt(
            entry["attributes"]["euler_transforms"],
            lambda path: read_record(root, path, metadata_scope=metadata_scope),
        )
        validate_encoding_attributes(entry["attributes"], receipt, addon["plan"])
