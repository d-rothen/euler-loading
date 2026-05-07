from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ds_crawler import (
    get_dataset_contract,
    load_dataset_config,
    load_dataset_split as _load_dataset_split,
    validate_metadata_scope,
)
from ds_crawler.zip_utils import read_metadata_json
from ds_crawler.config import CONFIG_FILENAME as _DS_CRAWLER_CONFIG_FILENAME
from ds_crawler.zip_utils import (
    OUTPUT_FILENAME as _DS_CRAWLER_OUTPUT_FILENAME,
    get_split_filename,
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


def parse_path_with_split(path: str) -> tuple[str, str | None]:
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


def _has_scoped_metadata(dataset_path: Path, metadata_scope: str) -> bool:
    """Return whether ds-crawler metadata exists under a configured scope."""
    for filename in (_DS_CRAWLER_OUTPUT_FILENAME, _DS_CRAWLER_CONFIG_FILENAME):
        if read_metadata_json(
            dataset_path,
            filename,
            metadata_scope=metadata_scope,
        ) is not None:
            return True
    return False


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
