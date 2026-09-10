"""Execution capture and explicit replay at the actual decoded-source boundary."""

from __future__ import annotations

import hashlib
import io
from copy import deepcopy

import numpy as np
from ds_crawler import get_dataset_contract
from ds_crawler.records import read_artifact
from euler_dataset_contract import (
    ExecutionReceipt,
    FieldBinding,
    TransformsAddon,
    bound_derivation_digest,
    canonical_digest,
    validate_execution,
)

from ._resolution import _builtin_loader_id
from .indexing import (
    collect_files,
    collect_hierarchical_files,
    match_hierarchical_files,
)
from .transform_descriptors import (
    _check_dataset_bindings,
    _selected,
    resolve_transform_descriptor,
)

DECODER_VERSION = "2.24.0"


def array_digest(value):
    """Decoded identity: dtype, shape and little-endian C-order bytes (v1)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in "buif":
        raise ValueError("unsupported decoded identity dtype")
    data = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
    return canonical_digest(
        {
            "kind": "decoded_array_1",
            "dtype": data.dtype.name,
            "shape": list(data.shape),
            "bytes": "sha256:" + hashlib.sha256(data.tobytes()).hexdigest(),
        }
    )


def _digest_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _full_id(record):
    return "/" + "/".join((*record.hierarchy_path, record.file_entry["id"]))


def _field_records(dataset, name):
    from .dataset import _get_index_tree

    records = collect_files(_get_index_tree(dataset.get_modality_index(name)))
    qualified = [(_full_id(r), r) for r in records]
    if len({fid for fid, _ in qualified}) != len(qualified):
        raise ValueError(f"Field {name}: ambiguous duplicate qualified source IDs")
    return dict(qualified)


def _manifest(dataset, name):
    modality = {**dataset._modalities, **dataset._hierarchical_modalities}[name]
    records = _field_records(dataset, name)
    digests = {
        fid: _digest_bytes(read_artifact(modality.path, r.file_entry["path"]))
        for fid, r in records.items()
    }
    return canonical_digest({"kind": "source_files_1", "files": digests}), digests


def describe_sources(dataset, fields, *, revisions):
    """Hash actual indexed source bytes. Revision names remain author declarations.

    The manifest binds all qualified IDs in each source index, including files
    outside the eventual output subset. No supplied hash is treated as verified.
    """
    from .dataset import _get_index_tree

    sources = {}
    for name, raw in fields.items():
        binding = FieldBinding(raw)
        alias = binding["source"]
        index = dataset.get_modality_index(name)
        head = get_dataset_contract(index)
        modality = {**dataset._modalities, **dataset._hierarchical_modalities}[name]
        decoder = _builtin_loader_id(dataset._resolved_loaders[name])
        if decoder is None:
            raise ValueError(f"Field {name}: unknown decoder callable effects")
        content, _ = _manifest(dataset, name)
        sources[alias] = {
            "dataset_id": head.dataset_id,
            "modality_key": head.modality_key,
            "metadata_scope": modality.metadata_scope or ".",
            "revision": revisions[alias],
            "head_digest": canonical_digest(head.to_mapping()),
            "index_digest": canonical_digest(_get_index_tree(index)),
            "verification": "content",
            "content_digest": content,
            "decoder": {
                "id": decoder,
                "version": DECODER_VERSION,
                "profile_digest": canonical_digest(binding["profile"]),
            },
        }
    return sources


def _execute_with_receipt(
    plan, sample, *, source_full_ids, source_content, verification
):
    """Capture actual input/output values and ordered geometry; no artifact claim.

    ``content`` is used by DatasetCapture after checking the manifest and decoding
    the same bytes. Standalone caller assertions must use ``metadata``.
    """
    plan = resolve_transform_descriptor(plan)
    recipe = plan.descriptor["recipe"]
    inputs = {
        name: _selected(sample, name, binding)
        for name, binding in recipe["fields"].items()
    }
    input_digests = {name: array_digest(value) for name, value in inputs.items()}
    output = plan(sample)
    operations = []
    for op in plan.operations:
        identity = op.op == "euler_loading.resize" and op.input_size == op.output_size
        operations.append(
            {
                **op.to_dict(),
                "version": "1.0",
                "status": "skipped" if identity else "executed",
                "reason": "identity_size" if identity else "applied",
            }
        )
    receipt = ExecutionReceipt(
        {
            "version": "2.0",
            "recipe_digest": plan.descriptor["recipe_digest"],
            "bound_digest": bound_derivation_digest(plan.descriptor, source_full_ids),
            "source_full_ids": source_full_ids,
            "source_content": source_content,
            "input_digests": input_digests,
            "output_digests": {name: array_digest(output[name]) for name in inputs},
            "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
            "output_shapes": {name: list(output[name].shape) for name in inputs},
            "operations": operations,
            "execution": recipe["execution"],
            "verification": verification,
        }
    )
    validate_execution(receipt, plan.descriptor)
    return output, receipt


def execute_with_receipt(plan, sample, *, source_full_ids, source_content):
    """Capture decoded arrays with metadata-only provenance.

    This standalone API does not verify supplied source hashes. Strict producer
    capture requires DatasetCapture, which reads and checks actual source files.
    """
    return _execute_with_receipt(
        plan,
        sample,
        source_full_ids=source_full_ids,
        source_content=source_content,
        verification="metadata",
    )


def replay_from_arrays(descriptor, receipt, original_sample):
    """Explicit decoded-value replay. Validates original array hashes and backend.

    This checks supplied decoded values; use DatasetCapture.replay for source
    file IO/manifest verification. Loading stored bytes never calls this API.
    """
    descriptor = TransformsAddon(descriptor)
    receipt = validate_execution(receipt, descriptor)
    for name, binding in descriptor["recipe"]["fields"].items():
        if (
            array_digest(_selected(original_sample, name, binding))
            != receipt["input_digests"][name]
        ):
            raise ValueError(f"replay input decoded digest mismatch: {name}")
    output, replay = _execute_with_receipt(
        descriptor,
        original_sample,
        source_full_ids=receipt["source_full_ids"],
        source_content=receipt["source_content"],
        verification=receipt["verification"],
    )
    if replay.to_dict() != receipt.to_dict():
        raise ValueError("replayed execution differs from recorded receipt")
    return output


class DatasetCapture:
    """Frozen source metadata, per-access byte checks and worker-safe execution.

    Indexes are checked and detached once. Access costs depend on the selected
    fields and their bytes, without rescanning the whole index for every sample.
    """

    def __init__(self, dataset, variants):
        from .dataset import _get_index_meta, _get_index_tree
        from .preprocessing import SamplePreprocessor
        from .transform_descriptors import ResolvedTransform

        self.dataset = dataset
        self._original_chain = tuple(dataset._transforms)
        self._variants = {
            name: TransformsAddon(value) for name, value in variants.items()
        }
        if not self._variants:
            raise ValueError("capture requires explicit variants")
        for name, modality in {
            **dataset._modalities,
            **dataset._hierarchical_modalities,
        }.items():
            if getattr(modality, "transform", None) is not None or getattr(
                modality, "transforms", None
            ):
                raise ValueError(f"unknown callable effects for modality {name}")
        if dataset._transforms:
            if len(dataset._transforms) != 1 or type(dataset._transforms[0]) not in {
                SamplePreprocessor,
                ResolvedTransform,
            }:
                raise ValueError("capture rejects opaque sample transformations")
            for descriptor in self._variants.values():
                if isinstance(dataset._transforms[0], ResolvedTransform):
                    actual = dataset._transforms[0].descriptor
                else:
                    actual = dataset._transforms[0].export_descriptor(
                        sources=descriptor["sources"],
                        fields=descriptor["recipe"]["fields"],
                        image_planes=descriptor["recipe"]["image_planes"],
                        execution=descriptor["recipe"]["execution"],
                    )
                if actual["recipe_digest"] != descriptor["recipe_digest"]:
                    raise ValueError(
                        "capture plan differs from configured sample chain"
                    )
        self._sources = {}
        self._references = {}
        self._common_ids = tuple(dataset._common_ids)
        for variant, descriptor in self._variants.items():
            _check_dataset_bindings(dataset, descriptor, sample_bound=True)
            resolve_transform_descriptor(descriptor)
            for name, binding in descriptor["recipe"]["fields"].items():
                source = descriptor["sources"][binding["source"]]
                if source["verification"] != "content":
                    raise ValueError(
                        "strict capture supports verified file content manifests only; metadata/snapshot assertions are unsupported"
                    )
                if source["decoder"]["version"] != DECODER_VERSION:
                    raise ValueError("source decoder version mismatch")
                if name not in self._sources:
                    index = dataset.get_modality_index(name)
                    tree = _get_index_tree(index)
                    records = collect_files(tree)
                    qualified = {_full_id(r): r for r in records}
                    if len(qualified) != len(records):
                        raise ValueError(
                            f"Field {name}: ambiguous duplicate qualified source IDs"
                        )
                    modality = {
                        **dataset._modalities,
                        **dataset._hierarchical_modalities,
                    }[name]
                    files = {
                        fid: _digest_bytes(
                            read_artifact(modality.path, r.file_entry["path"])
                        )
                        for fid, r in qualified.items()
                    }
                    self._sources[name] = {
                        "path": modality.path,
                        "loader": dataset._resolved_loaders[name],
                        "accepts_attributes": dataset._loaders_accept_attributes[name],
                        "meta": deepcopy(_get_index_meta(index)),
                        "records": qualified,
                        "hierarchy": collect_hierarchical_files(tree),
                        "regular": {
                            cid: qualified[_full_id(record)]
                            for cid, record in dataset._lookups.get(name, {}).items()
                        },
                        "files": files,
                        "content_digest": canonical_digest(
                            {"kind": "source_files_1", "files": files}
                        ),
                    }
                if self._sources[name]["content_digest"] != source["content_digest"]:
                    raise ValueError(f"source content manifest mismatch: {name}")
            reference = self._sources[descriptor["recipe"]["reference_field"]][
                "regular"
            ]
            self._references[variant] = {
                _full_id(reference[cid]): i for i, cid in enumerate(self._common_ids)
            }

    @property
    def variants(self):
        """Detached variant mapping; individual descriptors are immutable."""
        return dict(self._variants)

    def __call__(self, index, variant_id=None):
        from .dataset import _load_with_optional_attributes

        variant_id = variant_id or next(iter(self._variants))
        descriptor = self._variants[variant_id]
        dataset = self.dataset
        if tuple(dataset._transforms) != self._original_chain:
            raise ValueError("configured sample chain changed after capture was frozen")
        common_id = self._common_ids[index]
        reference = descriptor["recipe"]["reference_field"]
        reference_record = self._sources[reference]["regular"][common_id]
        hierarchy = reference_record.hierarchy_path
        full_id = _full_id(reference_record)
        sample = {
            "id": reference_record.file_entry["id"],
            "full_id": full_id,
            "meta": {},
            "attributes": {},
        }
        source_ids, source_content = {}, {}
        for name, binding in descriptor["recipe"]["fields"].items():
            pinned = self._sources[name]
            modality = {**dataset._modalities, **dataset._hierarchical_modalities}[name]
            if (
                modality.path != pinned["path"]
                or dataset._resolved_loaders[name] is not pinned["loader"]
                or getattr(modality, "transform", None) is not None
                or getattr(modality, "transforms", None)
            ):
                raise ValueError(
                    f"Field {name}: source path/decoder/effects changed after capture was frozen"
                )
            if pinned["regular"]:
                record = pinned["regular"][common_id]
                fid = _full_id(record)
            else:
                fid = binding["calibration"]["full_id"]
                record = pinned["records"].get(fid)
                matched = match_hierarchical_files(hierarchy, pinned["hierarchy"])
                if record is None or not any(
                    entry is record.file_entry for entry in matched
                ):
                    raise ValueError(
                        f"selected calibration is missing or not applicable: {fid}"
                    )
                if (
                    binding.get("selection", record.file_entry["id"])
                    != record.file_entry["id"]
                ):
                    raise ValueError(
                        "calibration selection disagrees with actual qualified record"
                    )
            raw = read_artifact(pinned["path"], record.file_entry["path"])
            digest = _digest_bytes(raw)
            if pinned["files"].get(fid) != digest:
                raise ValueError(f"source bytes changed: {fid}")
            stream = io.BytesIO(raw)
            stream.name = record.file_entry["path"]
            attributes = deepcopy(record.file_entry.get("attributes", {}))
            if (
                _builtin_loader_id(pinned["loader"])
                == "euler_loading.loaders.materialized.data"
                and attributes.get("output_encoding", {}).get("decoded")
                != binding["profile"]
            ):
                raise ValueError(
                    f"Field {name}: declared source profile contradicts decoder's per-file profile"
                )
            value = _load_with_optional_attributes(
                pinned["loader"],
                stream,
                deepcopy(pinned["meta"]),
                attributes,
                accepts_attributes=pinned["accepts_attributes"],
            )
            sample[name] = (
                {binding["selection"]: value} if "selection" in binding else value
            )
            sample["meta"][name] = deepcopy(record.file_entry)
            sample["attributes"][name] = attributes
            alias = binding["source"]
            source_ids[alias], source_content[alias] = fid, digest
        output, receipt = _execute_with_receipt(
            descriptor,
            sample,
            source_full_ids=source_ids,
            source_content=source_content,
            verification="content",
        )
        output["provenance"] = {
            "version": "2.0",
            "variant_id": variant_id,
            "descriptor": descriptor.to_dict(),
            "execution": receipt.to_dict(),
        }
        return output

    def verify_provenance(self, provenance):
        """Verify supplied receipts by reading and replaying the pinned originals.

        Writers call this at the publication boundary. A user-supplied digest or
        verification label alone cannot establish a strict materialization claim.
        """
        variant = provenance.get("variant_id")
        if variant not in self._variants:
            raise ValueError("unknown capture variant")
        descriptor = self._variants[variant]
        reference = descriptor["recipe"]["reference_field"]
        alias = descriptor["recipe"]["fields"][reference]["source"]
        full_id = provenance["execution"]["source_full_ids"].get(alias)
        if full_id not in self._references[variant]:
            raise ValueError("receipt source full ID is not in the bound source index")
        return self.replay(self._references[variant][full_id], provenance)

    def replay(self, index, provenance):
        output = self(index, provenance["variant_id"])
        if output["provenance"] != provenance:
            raise ValueError("source-backed replay disagrees with persisted execution")
        return output


class CalibrationView:
    """A per-output pinhole view resolved from a pinned calibration and receipt."""

    def __init__(self, descriptor, receipt, field):
        self.descriptor = TransformsAddon(descriptor)
        self.receipt = validate_execution(receipt, self.descriptor)
        self.field = field
        binding = self.descriptor["recipe"]["fields"][field]
        if binding["profile"]["kind"] != "intrinsics":
            raise ValueError("calibration view requires an intrinsics field")
        self.source = deepcopy(binding["calibration"])
        self.applies_to = tuple(self.source["applies_to"])
        self.cardinality = "one_per_output"

    def resolve(self, source_intrinsics):
        from .geometry import transform_intrinsics

        if array_digest(source_intrinsics) != self.receipt["input_digests"][self.field]:
            raise ValueError("calibration decoded digest mismatch")
        result = source_intrinsics
        for operation in self.receipt["operations"]:
            result = transform_intrinsics(result, operation["image_transform"])
        if array_digest(result) != self.receipt["output_digests"][self.field]:
            raise ValueError("calibration view replay mismatch")
        return result
