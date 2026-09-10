"""Opt-in resize/crop descriptors. Plans describe computation, not materialization.

The legacy SamplePreprocessor callable keeps its historical backend dispatch,
inference, and geometry. This module provides explicit, checked execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import PIL
from euler_dataset_contract import (
    ExecutionProfile,
    RepresentationProfile,
    TransformsAddon,
    recipe_digest,
    register_descriptor_validators,
)
from euler_dataset_contract.canonical import canonical_digest

from . import preprocessing as legacy


@runtime_checkable
class SerializableTransform(Protocol):
    """Authoring boundary; a later receipt protocol can extend this interface."""

    def export_descriptor(self, **bindings: Any) -> TransformsAddon: ...

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]: ...


def execution_profile(backend: str) -> ExecutionProfile:
    """Pin an installed CPU backend and its Phase 1 numerical conventions.

    Torch works on NumPy or CPU tensors. Pillow accepts NumPy only. No CUDA or
    cross-backend equality, codec, or artifact-byte guarantee is made.
    """
    if backend not in {"torch_cpu", "pillow_cpu"}:
        raise ValueError(f"Unsupported execution backend {backend!r}")
    versions = {"numpy": np.__version__}
    if backend == "torch_cpu":
        if legacy.torch is None:
            raise ValueError("torch_cpu requires an installed Torch")
        versions["torch"] = str(legacy.torch.__version__)
    else:
        versions["Pillow"] = PIL.__version__
    return ExecutionProfile(
        {
            "backend": backend,
            "versions": versions,
            "nearest": "floor" if backend == "torch_cpu" else "half_pixel",
            "antialias": backend == "pillow_cpu",
            "integer_rounding": "ties_to_even"
            if backend == "torch_cpu"
            else "legacy_truncate",
            "vector_epsilon": 1e-12 if backend == "torch_cpu" else 1e-8,
            "numeric_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        }
    )


def _dict(value: Any) -> dict[str, Any]:
    return value.to_dict() if hasattr(value, "to_dict") else deepcopy(dict(value))


def _check_authoring_config(config: dict[str, Any] | None) -> None:
    if config is None:
        return
    if set(config) - {
        "operations",
        "resize",
        "crop",
        "fields",
        "field_specs",
        "reference_field",
        "infer_fields",
    }:
        raise ValueError("Unknown authoring configuration keys cannot be exported")
    if "fields" in config and "field_specs" in config:
        raise ValueError("Ambiguous authoring field aliases")
    if config.get("operations") is not None and any(
        config.get(key) is not None for key in ("resize", "crop")
    ):
        raise ValueError(
            "Ambiguous authoring operation order: use operations or shorthand"
        )
    entries = config.get("operations")
    if entries is None:
        entries = [
            config[key] for key in ("resize", "crop") if config.get(key) is not None
        ]
    for entry in entries:
        if isinstance(entry, Mapping):
            if "size" in entry and "target_size" in entry:
                raise ValueError("Ambiguous authoring size aliases")
            if set(entry) - {"type", "size", "target_size", "anchor", "offset"}:
                raise ValueError("Unknown authoring operation parameters")
            values = [entry.get("size", entry.get("target_size"))]
            if entry.get("offset") is not None:
                values.append(entry["offset"])
        else:
            values = [entry]
        for pair in values:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value != int(value)
                    for value in pair
                )
            ):
                raise ValueError(
                    "Export requires integer authoring sizes/offsets without lossy coercion"
                )


def export_preprocessor(
    preprocessor: legacy.SamplePreprocessor,
    *,
    sources: Mapping[str, Any],
    fields: Mapping[str, Any],
    image_planes: Mapping[str, Any],
    execution: ExecutionProfile | Mapping[str, Any],
    sample: Mapping[str, Any] | None = None,
    upstream_transforms: Sequence[Any] = (),
) -> TransformsAddon:
    """Freeze an existing preprocessor against explicit profiles and provenance.

    Units, frames, sensor correspondence, and source revisions are declarations;
    a sample can validate shape/dtype and select a single qualified calibration.
    It cannot discover these semantics. FieldSpec defaults are expanded here.
    """
    if type(preprocessor) is not legacy.SamplePreprocessor:
        raise ValueError(
            "Unknown callable effects: export requires an unmodified SamplePreprocessor"
        )
    if upstream_transforms:
        raise ValueError(
            "Unknown upstream callable effects; export the complete checked chain"
        )
    _check_authoring_config(preprocessor._authoring_config)
    register_descriptor_validators()
    bindings = {name: _dict(value) for name, value in fields.items()}
    configured = preprocessor._bound_field_specs or preprocessor.field_specs
    missing = set(configured) - set(bindings)
    if missing:
        raise ValueError(f"Missing declared field bindings: {sorted(missing)}")
    if sample is not None and preprocessor.infer_fields:
        for name, value in sample.items():
            inferred = preprocessor._resolve_field_spec(name, value)
            if inferred is not None and name not in bindings:
                raise ValueError(f"Missing binding for inferred field {name!r}")
    for name, binding in bindings.items():
        profile = RepresentationProfile(binding["profile"]).to_dict()
        binding["profile"] = profile
        spec = configured.get(name)
        if spec is None:
            if not preprocessor.infer_fields:
                raise ValueError(f"Field {name!r} is not configured for preprocessing")
            # A declared profile takes precedence over authoring name heuristics.
            spec = legacy.FieldSpec(kind=profile["kind"])
        if spec.kind != profile["kind"]:
            raise ValueError(f"Field {name!r}: kind conflicts with declared profile")
        if spec.layout is not None and spec.layout != profile["layout"]:
            raise ValueError(f"Field {name!r}: layout conflicts with declared profile")
        if spec.kind == "passthrough":
            raise ValueError(
                f"Field {name!r}: passthrough has no spatial output binding"
            )
        default_interpolation = "nearest" if profile["kind"] == "mask" else "bilinear"
        if profile["kind"] == "generic" and profile["dtype"] not in {
            "float32",
            "float64",
        }:
            default_interpolation = "nearest"
        policy = {
            "interpolation": spec.interpolation or default_interpolation,
            "rays": "resample_normalize"
            if spec.should_normalize_vectors()
            else "preserve",
            "invalid_depth": "legacy_blend",
            "mask_threshold": spec.threshold,
        }
        # Explicit new policies authorize opt-in semantics, e.g. validity-aware depth.
        supplied_policy = binding.get("policy", {})
        for key in ("interpolation", "rays", "mask_threshold"):
            if key in supplied_policy and supplied_policy[key] != policy[key]:
                raise ValueError(
                    f"Field {name!r}: policy.{key} conflicts with FieldSpec"
                )
        policy.update(supplied_policy)
        binding["policy"] = policy
        if spec.reduce is not None and "selection" not in binding:
            if sample is None or name not in sample:
                raise ValueError(
                    f"Field {name!r}: reduce=first requires a sample or explicit selection"
                )
            value = sample[name]
            if isinstance(value, Mapping):
                if len(value) != 1:
                    raise ValueError(
                        f"Field {name!r}: ambiguous calibration; supply an explicit selection"
                    )
                binding["selection"] = next(iter(value))
        if (
            sample is not None
            and name in sample
            and isinstance(sample[name], Mapping)
            and "selection" not in binding
        ):
            raise ValueError(
                f"Field {name!r}: keyed calibration requires a frozen selection"
            )
    spatial = [
        name
        for name, binding in bindings.items()
        if "H" in binding["profile"]["layout"]
    ]
    reference = preprocessor.reference_field
    if reference is None:
        if len(spatial) != 1:
            raise ValueError(
                "Ambiguous reference field; specify reference_field explicitly"
            )
        reference = spatial[0]
    operations = []
    for index, operation in enumerate(preprocessor.operations, 1):
        if type(operation) not in {legacy.Resize, legacy.Crop}:
            raise ValueError(
                "Unknown callable operation effects; only Resize and Crop have descriptors"
            )
        parameters = {"size": list(operation.size)}
        if type(operation) is legacy.Crop:
            if operation.offset is None:
                parameters["anchor"] = operation.anchor
            else:
                parameters["offset"] = list(operation.offset)
        name = "resize" if type(operation) is legacy.Resize else "crop"
        operations.append(
            {
                "id": f"{name}_{index}",
                "op": f"euler_loading.{name}",
                "version": "1.0",
                "parameters": parameters,
            }
        )
    recipe = {
        "version": "1.0",
        "required_features": [
            "spatial.resize_crop",
            "bindings.qualified",
            "profiles.decoded",
        ],
        "reference_field": reference,
        "fields": bindings,
        "image_planes": {key: _dict(value) for key, value in image_planes.items()},
        "execution": _dict(execution),
        "operations": operations,
    }
    descriptor = TransformsAddon(
        {
            "version": "1.0",
            "state": "planned",
            "sources": {key: _dict(value) for key, value in sources.items()},
            "recipe": recipe,
            "recipe_digest": recipe_digest(recipe),
        }
    )
    # Export promises a supported plan, even when no representative sample is supplied.
    resolve_transform_descriptor(descriptor, sample=sample)
    return descriptor


def _check_dataset_bindings(dataset: Any, descriptor: TransformsAddon, *, sample_bound: bool = False) -> None:
    """Check the declarations against this dataset's actual metadata boundary.

    Content verification remains caller-owned. A changing inherited calibration
    requires separate sample-bound descriptors; Phase 2 will capture that choice.
    """
    from ds_crawler import get_dataset_contract

    from ._resolution import _builtin_loader_id
    from .dataset import _get_index_tree

    modalities = {**dataset._modalities, **dataset._hierarchical_modalities}
    indexes = {**dataset._index_outputs, **dataset._hierarchical_index_outputs}
    fields, sources = descriptor["recipe"]["fields"], descriptor["sources"]
    if not set(fields) <= set(modalities):
        raise ValueError("Declared fields are absent from the dataset")
    for name, binding in fields.items():
        source = sources[binding["source"]]
        head = get_dataset_contract(indexes[name])
        expected = {
            "dataset_id": head.dataset_id,
            "modality_key": head.modality_key,
            "metadata_scope": modalities[name].metadata_scope or ".",
            "head_digest": canonical_digest(head.to_mapping()),
            "index_digest": canonical_digest(_get_index_tree(indexes[name])),
        }
        for key, value in expected.items():
            if source[key] != value:
                raise ValueError(
                    f"Field {name!r}: source.{key} disagrees with dataset metadata"
                )
        loader = dataset._resolved_loaders[name]
        decoder_id = _builtin_loader_id(loader)
        if decoder_id is None or source["decoder"]["id"] != decoder_id:
            raise ValueError(
                f"Field {name!r}: unknown decoder callable effects or decoder identity mismatch"
            )
        if name in dataset._hierarchical_modalities and not sample_bound:
            entries = [
                ("/" + "/".join(path + (entry["id"],)), entry["id"])
                for path, items in dataset._hierarchical_lookups[name].items()
                for entry in items
            ]
            if (
                len(entries) != 1
                or binding.get("calibration", {}).get("full_id") != entries[0][0]
            ):
                raise ValueError(
                    f"Field {name!r}: inherited calibration varies; export a sample-bound descriptor"
                )
            selection = binding.get("selection")
            if not modalities[name].collapse_single and selection != entries[0][1]:
                raise ValueError(
                    f"Field {name!r}: calibration selection disagrees with the qualified dataset entry"
                )


@dataclass(frozen=True)
class ResolvedOperation:
    id: str
    op: str
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    offset: tuple[int, int] | None
    image_transform: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "input_size": list(self.input_size),
            "output_size": list(self.output_size),
            "offset": list(self.offset) if self.offset is not None else None,
            "image_transform": [list(row) for row in self.image_transform],
        }


def _resolve_geometry(recipe: dict) -> tuple[ResolvedOperation, ...]:
    from .geometry import crop_matrix, resize_matrix

    size = tuple(next(iter(recipe["image_planes"].values()))["size"])
    result = []
    for operation in recipe["operations"]:
        parameters = operation["parameters"]
        target = tuple(parameters["size"])
        offset = None
        if operation["op"] == "euler_loading.resize":
            matrix = resize_matrix(size, target)
        else:
            crop = legacy.Crop(
                size=target,
                anchor=parameters.get("anchor", "center"),
                offset=tuple(parameters["offset"]) if "offset" in parameters else None,
            )
            top, left, _, _ = crop.resolve_box(size)
            offset = (top, left)
            matrix = crop_matrix(top, left)
        result.append(
            ResolvedOperation(
                operation["id"], operation["op"], size, target, offset, matrix
            )
        )
        size = target
    return tuple(result)


def _check_support(recipe: dict) -> None:
    execution = recipe["execution"]
    installed = execution_profile(execution["backend"]).to_dict()
    for key, value in installed.items():
        if execution[key] != value:
            raise ValueError(
                f"Unsupported execution.{key}: declared {execution[key]!r}, installed {value!r}"
            )
    interpolates = any(
        op.op == "euler_loading.resize" and op.input_size != op.output_size
        for op in _resolve_geometry(recipe)
    )
    for name, binding in recipe["fields"].items():
        profile, policy = binding["profile"], binding["policy"]
        if any(type(dim) is not int for dim in profile["shape"]):
            raise ValueError(
                f"Field {name!r}: resolve symbolic profile dimensions before export"
            )
        if profile["kind"] in {
            "depth",
            "ray_map",
            "point_map",
            "intrinsics",
        } and profile["dtype"] not in {"float32", "float64"}:
            raise ValueError(
                f"Field {name!r}: geometric values require a floating dtype"
            )
        # Interpolation through float32 cannot safely preserve arbitrary int64 labels.
        if interpolates and profile["dtype"] in {"int32", "int64"}:
            raise ValueError(
                f"Field {name!r}: this float32 executor does not support int32/int64 profiles"
            )
        if profile["kind"] == "intrinsics" and profile["camera_model"] != "pinhole":
            raise ValueError(
                f"Field {name!r}: unsupported camera model; only pinhole K is implemented"
            )
        if profile["kind"] in {"ray_map", "point_map"} and profile["components"] != [
            "x",
            "y",
            "z",
        ]:
            raise ValueError(f"Field {name!r}: unsupported vector component convention")
        if policy["rays"] == "regenerate":
            raise ValueError(f"Field {name!r}: ray regeneration is not implemented")
        if (
            policy["invalid_depth"] == "validity_aware"
            and profile["invalid"]["sentinel"] is None
        ):
            raise ValueError(
                f"Field {name!r}: validity-aware depth requires an explicit finite sentinel"
            )


def _selected(sample: Mapping[str, Any], name: str, binding: dict) -> Any:
    if name not in sample or sample[name] is None:
        raise ValueError(f"Missing required field {name!r}")
    value = sample[name]
    if "selection" in binding:
        key = binding["selection"]
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(
                f"Field {name!r}: missing qualified calibration selection {key!r}"
            )
        value = value[key]
    elif isinstance(value, Mapping):
        raise ValueError(f"Field {name!r}: unfrozen calibration selection")
    return value


def _validate_value(name: str, value: Any, profile: dict, execution: dict) -> None:
    tensor = legacy._is_torch_tensor(value)
    if not tensor and not isinstance(value, np.ndarray):
        raise ValueError(f"Field {name!r}: expected a NumPy array or CPU tensor")
    if tensor and (execution["backend"] != "torch_cpu" or value.device.type != "cpu"):
        raise ValueError(
            f"Field {name!r}: array device/container is incompatible with execution backend"
        )
    if tensor and value.requires_grad:
        raise ValueError(
            f"Field {name!r}: descriptor execution is for decoded values without autograd"
        )
    dtype = str(value.dtype).removeprefix("torch.")
    if tuple(value.shape) != tuple(profile["shape"]) or dtype != profile["dtype"]:
        raise ValueError(
            f"Field {name!r}: shape/dtype does not match declared input profile"
        )
    array = value.detach().numpy() if tensor else value
    if not profile["invalid"]["non_finite"] and not np.isfinite(array).all():
        raise ValueError(f"Field {name!r}: non-finite values forbidden by profile")
    if profile["kind"] == "intrinsics":
        if (
            not np.isfinite(array).all()
            or not np.array_equal(array[2], [0, 0, 1])
            or array[1, 0] != 0
            or array[0, 0] <= 0
            or array[1, 1] <= 0
        ):
            raise ValueError(f"Field {name!r}: unsupported pinhole matrix convention")
    if profile["kind"] == "ray_map" and profile["normalization"] in {
        "unit",
        "unit_or_zero",
    }:
        norms = np.linalg.norm(array, axis=profile["layout"].index("C"))
        valid = np.isclose(norms, 1, **execution["numeric_tolerance"])
        if profile["normalization"] == "unit_or_zero":
            valid |= norms == 0
        if not valid.all():
            raise ValueError(f"Field {name!r}: declared ray normalization is false")


def _float_resize(
    value: Any, layout: str, size: tuple[int, int], policy: dict, execution: dict
) -> Any:
    if execution["backend"] == "torch_cpu":
        tensor, _ = legacy._to_torch_nchw(value, layout)
        return legacy.F.interpolate(
            tensor.to(dtype=legacy.torch.float32),
            size=size,
            mode=policy["interpolation"],
            align_corners=False if policy["interpolation"] != "nearest" else None,
            antialias=False,
        )
    return legacy._resize_numpy_fallback(
        np.asarray(value, dtype=np.float32),
        layout=layout,
        size=size,
        interpolation=policy["interpolation"],
    )


def _resize(
    value: Any, profile: dict, policy: dict, size: tuple[int, int], execution: dict
) -> Any:
    layout = profile["layout"]
    use_torch = execution["backend"] == "torch_cpu"
    tensor_input = legacy._is_torch_tensor(value)
    supported = None
    if policy["invalid_depth"] == "validity_aware":
        sentinel = profile["invalid"]["sentinel"]
        if tensor_input:
            valid = legacy.torch.isfinite(value) & (value != sentinel)
            clean = legacy.torch.where(valid, value, 0)
        else:
            valid = np.isfinite(value) & (value != sentinel)
            clean = np.where(valid, value, 0)
        numerator = _float_resize(clean, layout, size, policy, execution)
        weight = _float_resize(valid, layout, size, policy, execution)
        supported = weight > 0
        if use_torch:
            work = numerator / legacy.torch.where(supported, weight, 1)
        else:
            work = np.zeros_like(numerator)
            np.divide(numerator, weight, out=work, where=supported)
    else:
        work = _float_resize(value, layout, size, policy, execution)
    if not use_torch and profile["dtype"] not in {"bool", "float32", "float64"}:
        # Named legacy Pillow behavior casts before the old final rounding step.
        work = work.astype(value.dtype)
    if policy["rays"] == "resample_normalize":
        if use_torch:
            work = legacy.F.normalize(work, dim=1, eps=execution["vector_epsilon"])
        else:
            norms = np.linalg.norm(work, axis=layout.index("C"), keepdims=True)
            work = work / np.maximum(norms, execution["vector_epsilon"])
    if profile["dtype"] == "bool":
        work = work > policy["mask_threshold"]
    elif profile["dtype"] not in {"float32", "float64"}:
        work = work.round() if use_torch else np.rint(work)
    if use_torch:
        result = legacy._from_torch_nchw(
            work, layout=layout, original=value, original_is_torch=tensor_input
        )
        if supported is not None:
            supported = (
                legacy._from_torch_nchw(
                    supported,
                    layout=layout,
                    original=value,
                    original_is_torch=tensor_input,
                )
                != 0
            )
    else:
        result = work.astype(value.dtype, copy=False)
    if supported is not None:
        # Restore after working-dtype conversion, including float64 sentinels.
        result[~supported] = sentinel
    return result


@dataclass(frozen=True)
class ResolvedTransform:
    """A checked, immutable plan, including inferred geometry; no execution receipt."""

    descriptor: TransformsAddon
    operations: tuple[ResolvedOperation, ...]

    def __post_init__(self) -> None:
        parsed = TransformsAddon(self.descriptor)
        _check_support(parsed["recipe"])
        if self.operations != _resolve_geometry(parsed["recipe"]):
            raise ValueError("Resolved geometry disagrees with the descriptor")
        object.__setattr__(self, "descriptor", parsed)

    def export_descriptor(self, **bindings: Any) -> TransformsAddon:
        if bindings:
            raise ValueError(
                "Resolved descriptors are frozen; export a new plan to rebind"
            )
        return self.descriptor

    def bind_inputs(self, sample: Mapping[str, Any]) -> None:
        register_descriptor_validators()
        recipe = self.descriptor["recipe"]
        _check_support(recipe)  # Also runs after unpickling in a worker.
        for name, binding in recipe["fields"].items():
            _validate_value(
                name,
                _selected(sample, name, binding),
                binding["profile"],
                recipe["execution"],
            )

    @property
    def image_transform(self) -> tuple[tuple[float, ...], ...]:
        matrix = np.eye(3, dtype=np.float64)
        for operation in self.operations:
            matrix = np.asarray(operation.image_transform) @ matrix
        return tuple(tuple(float(value) for value in row) for row in matrix)

    def infer_output_profiles(self) -> dict[str, RepresentationProfile]:
        recipe = self.descriptor["recipe"]
        sources = self.descriptor["sources"]
        for source in sources.values():
            source.pop("locator", None)
            source.pop("diagnostics", None)
        output_plane = (
            "derived_"
            + canonical_digest(
                {
                    "kind": "planned_image_plane",
                    "recipe_digest": self.descriptor["recipe_digest"],
                    "sources": sources,
                }
            ).split(":")[1]
        )
        profiles = {}
        resized = any(
            op.op == "euler_loading.resize" and op.input_size != op.output_size
            for op in self.operations
        )
        for name, binding in recipe["fields"].items():
            profile = binding["profile"]
            profile["image_plane"] = output_plane
            for axis, size in zip("HW", self.operations[-1].output_size):
                if axis in profile["layout"]:
                    profile["shape"][profile["layout"].index(axis)] = size
            if profile["kind"] == "ray_map" and resized:
                # Epsilon clamping permits sub-unit tiny vectors; zero stays zero.
                profile["normalization"] = "none"
            profiles[name] = RepresentationProfile(profile)
        return profiles

    def infer_output_bindings(self) -> dict[str, Any]:
        """Virtual per-sample calibration views; these are not persisted sources."""
        recipe = self.descriptor["recipe"]
        profiles = self.infer_output_profiles()
        plane_id = next(iter(profiles.values()))["image_plane"]
        plane = next(iter(recipe["image_planes"].values()))
        plane["size"] = list(self.operations[-1].output_size)
        return {
            "state": "planned",
            "image_planes": {plane_id: plane},
            "fields": {
                name: {
                    "profile": profile.to_dict(),
                    "input_field": name,
                    "calibration_view": {
                        "source": recipe["fields"][name]["calibration"],
                        "image_plane": plane_id,
                        "image_transform": [list(row) for row in self.image_transform],
                    }
                    if "calibration" in recipe["fields"][name]
                    else None,
                }
                for name, profile in profiles.items()
            },
        }

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        self.bind_inputs(sample)
        recipe = self.descriptor["recipe"]
        result = dict(sample)
        for name, binding in recipe["fields"].items():
            value = _selected(sample, name, binding)
            # No crop view can expose mutable cached source arrays to downstream code.
            value = (
                value.clone()
                if legacy._is_torch_tensor(value)
                else np.array(value, copy=True)
            )
            for operation in self.operations:
                if binding["profile"]["kind"] == "intrinsics":
                    from .geometry import transform_intrinsics
                    value = transform_intrinsics(value, operation.image_transform)
                elif operation.op == "euler_loading.crop":
                    value = legacy._crop_spatial_value(
                        value,
                        layout=binding["profile"]["layout"],
                        top=operation.offset[0],
                        left=operation.offset[1],
                        size=operation.output_size,
                    )
                elif operation.input_size != operation.output_size:
                    value = _resize(
                        value,
                        binding["profile"],
                        binding["policy"],
                        operation.output_size,
                        recipe["execution"],
                    )
            result[name] = value
        # Profiles are predictions checked against output values, not write receipts.
        for name, profile in self.infer_output_profiles().items():
            _validate_value(name, result[name], profile.to_dict(), recipe["execution"])
        return result


def resolve_transform_descriptor(
    descriptor: TransformsAddon | Mapping[str, Any] | str,
    *,
    sample: Mapping[str, Any] | None = None,
) -> ResolvedTransform:
    """Parse exact semantic versions, reject unsupported policies, and bind geometry."""
    register_descriptor_validators()
    parsed = (
        TransformsAddon.from_json(descriptor)
        if isinstance(descriptor, str)
        else TransformsAddon(descriptor)
    )
    recipe = parsed["recipe"]
    _check_support(recipe)
    resolved = ResolvedTransform(parsed, _resolve_geometry(recipe))
    if sample is not None:
        resolved.bind_inputs(sample)
    return resolved
