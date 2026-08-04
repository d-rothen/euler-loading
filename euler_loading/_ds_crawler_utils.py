from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from ds_crawler import (
    get_dataset_contract,
    list_metadata_scopes,
    load_dataset_config,
    load_dataset_split as _load_dataset_split,
    validate_metadata_scope,
)
from ds_crawler.config import CONFIG_FILENAME as _DS_CRAWLER_CONFIG_FILENAME
from ds_crawler.zip_utils import (
    DATASET_HEAD_FILENAME as _DS_CRAWLER_HEAD_FILENAME,
    OUTPUT_FILENAME as _DS_CRAWLER_OUTPUT_FILENAME,
    get_split_filename,
    read_metadata_json,
    validate_split_name,
)


def as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parsed = [as_non_empty_str(item) for item in value]
        return [item for item in parsed if item is not None]

    single = as_non_empty_str(value)
    if single is None:
        return []
    return [single]


def first_non_empty(*candidates: str | None) -> str | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def first_non_empty_list(*candidates: list[str] | None) -> list[str]:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return []


def extract_ds_crawler_properties(index_output: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return get_dataset_contract(dict(index_output)).to_properties_dict()
    except Exception:
        properties = index_output.get("properties")
        if isinstance(properties, Mapping):
            return dict(properties)
        return {}


_METADATA_FILENAMES = (
    _DS_CRAWLER_OUTPUT_FILENAME,
    _DS_CRAWLER_CONFIG_FILENAME,
    _DS_CRAWLER_HEAD_FILENAME,
)


def _parse_path_with_scope(path: str) -> tuple[str, str | None]:
    """Extract an optional ``#scope=<name>`` selector from a path."""
    candidate_path, sep, fragment = path.rpartition("#")
    if not sep:
        return path, None

    params = parse_qsl(fragment, keep_blank_values=True)
    scope_values = [value for key, value in params if key == "scope"]
    if not scope_values:
        return path, None

    unsupported = [key for key, _ in params if key != "scope"]
    if unsupported:
        raise ValueError(
            "Unsupported modality path fragment parameter(s): "
            f"{', '.join(sorted(set(unsupported)))}. "
            "Only '#scope=<metadata_scope>' is supported."
        )
    if len(scope_values) > 1:
        raise ValueError("Modality path may contain at most one '#scope=' selector.")

    return candidate_path, validate_metadata_scope(scope_values[0])


def _parse_path_with_split(path: str) -> tuple[str, str | None]:
    """Extract an optional inline split suffix from a colon-separated path.

    Supports paths like ``/data/ds.zip:train`` where the part after the
    last colon is treated as the split name.

    Returns a ``(path, split)`` tuple.  When no valid split suffix is
    found, *split* is ``None`` and *path* is returned unchanged.
    """
    colon_pos = path.rfind(":")
    # No colon, or colon at position 1 is a Windows drive letter (e.g. "C:\\")
    if colon_pos <= 1:
        return path, None

    candidate_split = path[colon_pos + 1:]
    candidate_path = path[:colon_pos]

    try:
        split = validate_split_name(candidate_split)
    except ValueError:
        return path, None

    return candidate_path, split


def parse_modality_path(path: str) -> tuple[str, str | None, str | None]:
    """Extract inline ``:split`` and ``#scope=...`` selectors from a path."""
    parsed_path, metadata_scope = _parse_path_with_scope(path)
    parsed_path, split = _parse_path_with_split(parsed_path)
    return parsed_path, split, metadata_scope


def _read_metadata_json_safe(
    dataset_path: Path,
    filename: str,
    *,
    metadata_scope: str | None = None,
) -> dict[str, Any] | None:
    try:
        return read_metadata_json(
            dataset_path,
            filename,
            metadata_scope=metadata_scope,
        )
    except FileNotFoundError:
        return None


def _has_root_metadata(dataset_path: Path) -> bool:
    """Return whether unscoped ds-crawler metadata exists at a dataset root."""
    return any(
        _read_metadata_json_safe(dataset_path, filename) is not None
        for filename in _METADATA_FILENAMES
    )


def _has_scoped_metadata(dataset_path: Path, metadata_scope: str) -> bool:
    """Return whether ds-crawler metadata exists under a configured scope."""
    return any(
        _read_metadata_json_safe(
            dataset_path,
            filename,
            metadata_scope=metadata_scope,
        ) is not None
        for filename in _METADATA_FILENAMES
    )


def _list_metadata_scopes_safe(dataset_path: Path) -> list[str]:
    try:
        return list_metadata_scopes(dataset_path)
    except FileNotFoundError:
        return []


def _read_scope_modality_key(dataset_path: Path, metadata_scope: str) -> str | None:
    head = _read_metadata_json_safe(
        dataset_path,
        _DS_CRAWLER_HEAD_FILENAME,
        metadata_scope=metadata_scope,
    )
    modality = head.get("modality") if isinstance(head, Mapping) else None
    key = modality.get("key") if isinstance(modality, Mapping) else None
    return as_non_empty_str(key)


def infer_metadata_scope(
    path: str | Path,
    *,
    candidates: list[str | None],
) -> str | None:
    """Infer a ds-crawler metadata scope only when the choice is deterministic.

    Legacy root-level metadata always wins for an unscoped modality. If no
    root metadata exists, a single available scope is selected automatically.
    With multiple scopes, a scope is selected only when exactly one scope name
    or scoped ``dataset-head.json`` modality key matches the provided
    candidates.
    """
    dataset_path = Path(path)
    if _has_root_metadata(dataset_path):
        return None

    scopes = _list_metadata_scopes_safe(dataset_path)
    if not scopes:
        return None
    if len(scopes) == 1:
        return scopes[0]

    candidate_set = {
        candidate
        for candidate in (as_non_empty_str(value) for value in candidates)
        if candidate is not None
    }
    if not candidate_set:
        _raise_ambiguous_metadata_scope(path, scopes, candidate_set)

    matches = {
        scope
        for scope in scopes
        if scope in candidate_set
        or _read_scope_modality_key(dataset_path, scope) in candidate_set
    }
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        _raise_ambiguous_metadata_scope(path, sorted(matches), candidate_set)

    _raise_ambiguous_metadata_scope(path, scopes, candidate_set)


def _raise_ambiguous_metadata_scope(
    path: str | Path,
    scopes: list[str],
    candidates: set[str],
) -> None:
    candidate_text = (
        f" Candidates were: {', '.join(sorted(candidates))}."
        if candidates
        else ""
    )
    raise ValueError(
        f"Multiple ds-crawler metadata scopes are available for {path!s}: "
        f"{', '.join(scopes)}.{candidate_text} "
        "Specify Modality(..., metadata_scope=...) or append "
        "'#scope=<metadata_scope>' to the path."
    )


def load_dataset_config_for_scope(
    path: str | Path,
    *,
    metadata_scope: str | None,
    load_dataset_config_fn: Any = load_dataset_config,
    fallback_to_legacy: bool = True,
) -> Any:
    """Load a ds-crawler config, delegating scoped layout handling to ds-crawler."""
    dataset_path = Path(path)
    if metadata_scope is None:
        return load_dataset_config_fn({"path": str(path)})

    normalized_scope = validate_metadata_scope(metadata_scope)
    try:
        return load_dataset_config_fn(
            {"path": str(path)},
            metadata_scope=normalized_scope,
        )
    except FileNotFoundError:
        if fallback_to_legacy and not _has_scoped_metadata(dataset_path, normalized_scope):
            return load_dataset_config_fn({"path": str(path)})
        raise


def load_index_output(
    path: str | Path,
    *,
    split: str | None,
    metadata_scope: str | None = None,
    index_dataset_from_path_fn: Any,
    strict: bool = False,
    save_index: bool = False,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Load a ds-crawler output, optionally overlaying an inline split.

    Scoped metadata is loaded through ds-crawler's native ``metadata_scope``
    support. If a scope is configured but no scoped metadata exists at all,
    euler-loading still falls back to the legacy root-level lookup for
    backward compatibility with older datasets.
    """
    dataset_path = Path(path)
    normalized_scope = (
        validate_metadata_scope(metadata_scope)
        if metadata_scope is not None
        else None
    )

    if normalized_scope is not None:
        try:
            if split is not None:
                return _load_dataset_split(
                    path,
                    validate_split_name(split),
                    strict=strict,
                    save_index=save_index,
                    force_reindex=force_reindex,
                    metadata_scope=normalized_scope,
                )
            return index_dataset_from_path_fn(
                path,
                strict=strict,
                save_index=save_index,
                force_reindex=force_reindex,
                metadata_scope=normalized_scope,
            )
        except FileNotFoundError:
            if _has_scoped_metadata(dataset_path, normalized_scope):
                raise

    if split is None:
        return index_dataset_from_path_fn(
            path,
            strict=strict,
            save_index=save_index,
            force_reindex=force_reindex,
        )

    normalized_split = validate_split_name(split)
    try:
        return _load_dataset_split(
            path,
            normalized_split,
            strict=strict,
            save_index=save_index,
            force_reindex=force_reindex,
        )
    except FileNotFoundError:
        # Fallback for mocked tests or callers that provide synthetic index
        # loaders without a full ds-crawler config on disk.
        pass

    base_output = index_dataset_from_path_fn(
        path,
        strict=strict,
        save_index=save_index,
        force_reindex=force_reindex,
    )
    split_filename = get_split_filename(normalized_split)
    split_dataset = read_metadata_json(dataset_path, split_filename)
    if split_dataset is None:
        raise FileNotFoundError(
            f"Inline split metadata {split_filename!r} not found for {dataset_path}"
        )

    result = dict(base_output)
    result["index"] = split_dataset
    result["dataset"] = split_dataset
    return result
