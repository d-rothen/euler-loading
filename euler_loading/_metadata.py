from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable

from ._ds_crawler_utils import (
    as_non_empty_str,
    as_string_list,
    extract_ds_crawler_properties,
    first_non_empty,
    first_non_empty_list,
)

if TYPE_CHECKING:
    from .dataset import Modality


_PROPERTY_NAMESPACE_KEYS = ("euler_loading", "euler_train")


def _build_runlog_entry(
    name: str,
    modality: Modality,
    descriptor: Mapping[str, Any],
    *,
    is_hierarchical: bool,
    regular_modality_names: list[str],
    hierarchy_levels: list[tuple[str, ...]] | None,
) -> dict[str, Any]:
    euler_loading_properties = _build_euler_loading_layers(
        modality.metadata,
        descriptor.get("properties"),
    )
    used_as = first_non_empty(
        modality.used_as,
        as_non_empty_str(
            _resolve_euler_loading_property(euler_loading_properties, "used_as")
        ),
        _infer_used_as(name=name, is_hierarchical=is_hierarchical),
    )
    modality_type = first_non_empty(
        modality.modality_type,
        as_non_empty_str(
            _resolve_euler_loading_property(euler_loading_properties, "modality_type")
        ),
        as_non_empty_str(descriptor.get("modality_type")),
        _infer_modality_type(name=name, path=modality.path),
    )
    slot = first_non_empty(
        modality.slot,
        as_non_empty_str(
            _resolve_euler_loading_property(euler_loading_properties, "slot")
        ),
        _infer_slot(
            name=name,
            used_as=used_as,
            modality_type=modality_type,
            euler_loading_properties=euler_loading_properties,
        ),
    )

    entry: dict[str, Any] = {"path": modality.path}
    if used_as is not None:
        entry["used_as"] = used_as
    if slot is not None:
        entry["slot"] = slot
    if modality_type is not None:
        entry["modality_type"] = modality_type

    if is_hierarchical:
        hierarchy_scope = first_non_empty(
            modality.hierarchy_scope,
            as_non_empty_str(
                _resolve_euler_loading_property(
                    euler_loading_properties, "hierarchy_scope"
                )
            ),
            _infer_hierarchy_scope_from_regex(descriptor.get("hierarchy_regex")),
            _infer_hierarchy_scope_from_levels(hierarchy_levels),
        )
        if hierarchy_scope is not None:
            entry["hierarchy_scope"] = hierarchy_scope

        applies_to = first_non_empty_list(
            as_string_list(modality.applies_to),
            as_string_list(
                _resolve_euler_loading_property(euler_loading_properties, "applies_to")
            ),
            list(regular_modality_names),
        )
        entry["applies_to"] = applies_to

    return entry


def _build_euler_loading_layers(*candidates: Any) -> list[Mapping[str, Any]]:
    layers: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for ns in _PROPERTY_NAMESPACE_KEYS:
            namespaced = candidate.get(ns)
            if isinstance(namespaced, Mapping):
                layers.append(namespaced)
    return layers


def _resolve_euler_loading_property(
    euler_loading_properties: list[Mapping[str, Any]],
    key: str,
) -> Any:
    for layer in euler_loading_properties:
        value = layer.get(key)
        if value is not None:
            return value
    return None


def _get_ds_crawler_descriptor(
    *,
    path: str,
    index_output: Mapping[str, Any] | None,
    cache: dict[str, dict[str, Any]],
    load_dataset_config_fn: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    if path not in cache:
        cache[path] = _read_ds_crawler_descriptor(
            path=path,
            index_output=index_output,
            load_dataset_config_fn=load_dataset_config_fn,
        )
    return cache[path]


def _read_ds_crawler_descriptor(
    *,
    path: str,
    index_output: Mapping[str, Any] | None,
    load_dataset_config_fn: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    descriptor: dict[str, Any] = {"properties": properties}

    if isinstance(index_output, Mapping):
        properties.update(extract_ds_crawler_properties(index_output))
        index_type = as_non_empty_str(index_output.get("type"))
        if index_type is not None:
            descriptor["modality_type"] = index_type
        hierarchy_regex = as_non_empty_str(index_output.get("hierarchy_regex"))
        if hierarchy_regex is not None:
            descriptor["hierarchy_regex"] = hierarchy_regex

    try:
        cfg = load_dataset_config_fn({"path": path})
    except Exception:
        return descriptor

    cfg_properties = getattr(cfg, "properties", None)
    if isinstance(cfg_properties, Mapping):
        properties.update(dict(cfg_properties))

    cfg_type = as_non_empty_str(getattr(cfg, "type", None))
    if cfg_type is not None:
        descriptor["modality_type"] = cfg_type

    cfg_hierarchy_regex = as_non_empty_str(getattr(cfg, "hierarchy_regex", None))
    if cfg_hierarchy_regex is not None:
        descriptor["hierarchy_regex"] = cfg_hierarchy_regex

    return descriptor


def _infer_used_as(*, name: str, is_hierarchical: bool) -> str | None:
    lowered = name.lower()
    if any(
        token in lowered
        for token in ("condition", "cond", "camera", "intrinsics", "extrinsics", "pose")
    ):
        return "condition"
    if any(token in lowered for token in ("target", "gt", "label", "clear", "clean")):
        return "target"
    if any(token in lowered for token in ("input", "source", "src", "hazy", "noisy", "raw")):
        return "input"
    if is_hierarchical:
        return "condition"
    return None


def _infer_modality_type(*, name: str, path: str) -> str | None:
    lowered = f"{name} {path}".lower()
    if any(token in lowered for token in ("rgb", "image", "img", "color", "colour")):
        return "rgb"
    if any(token in lowered for token in ("depth", "disparity")):
        return "depth"
    if any(token in lowered for token in ("segmentation", "segment", "mask", "semantic")):
        return "segmentation"
    return None


def _infer_slot(
    *,
    name: str,
    used_as: str | None,
    modality_type: str | None,
    euler_loading_properties: list[Mapping[str, Any]],
) -> str | None:
    if used_as is None:
        return None
    task = as_non_empty_str(
        _resolve_euler_loading_property(euler_loading_properties, "task")
    )
    leaf = modality_type or name
    if task is not None:
        return f"{task}.{used_as}.{leaf}"
    return f"{used_as}.{leaf}"


def _infer_hierarchy_scope_from_regex(value: Any) -> str | None:
    regex = as_non_empty_str(value)
    if regex is None:
        return None
    try:
        pattern = re.compile(regex)
    except re.error:
        return None

    if not pattern.groupindex:
        return None

    ordered_names = [
        name for name, _ in sorted(pattern.groupindex.items(), key=lambda item: item[1])
    ]
    if not ordered_names:
        return None
    return "_".join(ordered_names)


def _infer_hierarchy_scope_from_levels(
    levels: list[tuple[str, ...]] | None,
) -> str | None:
    if not levels:
        return None

    non_root_levels = [level for level in levels if level]
    if not non_root_levels:
        return "root"

    max_depth = max(len(level) for level in non_root_levels)
    deepest_levels = [level for level in non_root_levels if len(level) == max_depth]

    tokens: list[str] = []
    for idx in range(max_depth):
        candidates = {
            token
            for token in (_extract_hierarchy_token(level[idx]) for level in deepest_levels)
            if token is not None
        }
        if len(candidates) != 1:
            return f"level_{max_depth}"
        tokens.append(next(iter(candidates)))

    if not tokens:
        return f"level_{max_depth}"
    return "_".join(tokens)


def _extract_hierarchy_token(value: str) -> str | None:
    for separator in (":", "=", "__", "_", "-"):
        if separator not in value:
            continue
        prefix = value.split(separator, 1)[0].strip().lower()
        if prefix and any(ch.isalpha() for ch in prefix):
            return prefix
    return None
