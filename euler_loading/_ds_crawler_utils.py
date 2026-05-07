from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ds_crawler import get_dataset_contract, load_dataset_config
from ds_crawler.zip_utils import read_metadata_json

try:
    from ds_crawler.artifacts import (
        OUTPUT_FILENAME as _DS_CRAWLER_OUTPUT_FILENAME,
        hydrate_index_artifact as _hydrate_index_artifact,
        hydrate_split_artifact as _hydrate_split_artifact,
    )
except ImportError:  # pragma: no cover - compatibility with older ds-crawler
    _DS_CRAWLER_OUTPUT_FILENAME = "index.json"
    _hydrate_index_artifact = None
    _hydrate_split_artifact = None

try:
    from ds_crawler.config import (
        CONFIG_FILENAME as _DS_CRAWLER_CONFIG_FILENAME,
        DATASET_HEAD_FILENAME as _DS_CRAWLER_HEAD_FILENAME,
    )
except ImportError:  # pragma: no cover - compatibility with older ds-crawler
    _DS_CRAWLER_CONFIG_FILENAME = "ds-crawler.json"
    _DS_CRAWLER_HEAD_FILENAME = "dataset-head.json"

try:
    from ds_crawler import load_dataset_split as _load_dataset_split
except ImportError:
    _load_dataset_split = None


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


_SPLIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_METADATA_SCOPE_PATTERN = _SPLIT_NAME_PATTERN


def validate_metadata_scope(metadata_scope: str) -> str:
    """Validate and normalize a ds-crawler metadata namespace.

    A metadata scope selects files below ``.ds_crawler/<scope>/``.  It is
    intentionally restricted to one path segment so caller-provided values
    cannot escape the metadata directory.
    """
    if not isinstance(metadata_scope, str):
        raise ValueError("metadata_scope must be a string")
    normalized = metadata_scope.strip()
    if not normalized:
        raise ValueError("metadata_scope must be a non-empty string")
    if not _METADATA_SCOPE_PATTERN.match(normalized):
        raise ValueError(
            "metadata_scope may only contain letters, digits, '.', '-', or '_'"
        )
    return normalized


def get_scoped_metadata_filename(metadata_scope: str, filename: str) -> str:
    """Return ``<scope>/<filename>`` after validating both parts."""
    normalized_scope = validate_metadata_scope(metadata_scope)
    if not isinstance(filename, str) or not filename or filename.startswith("/"):
        raise ValueError("metadata filename must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in filename.split("/")):
        raise ValueError("metadata filename must not contain empty, '.', or '..' parts")
    return f"{normalized_scope}/{filename}"


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

    if not candidate_split or not _SPLIT_NAME_PATTERN.match(candidate_split):
        return path, None

    return candidate_path, candidate_split


def validate_split_name(split_name: str) -> str:
    """Validate and normalize an inline split name."""
    if not isinstance(split_name, str):
        raise ValueError("split_name must be a string")
    normalized = split_name.strip()
    if not normalized:
        raise ValueError("split_name must be a non-empty string")
    if not _SPLIT_NAME_PATTERN.match(normalized):
        raise ValueError(
            "split_name may only contain letters, digits, '.', '-', or '_'"
        )
    return normalized


def get_split_filename(split_name: str) -> str:
    """Return the ds-crawler metadata filename for *split_name*."""
    normalized = validate_split_name(split_name)
    return f"split_{normalized}.json"


def read_metadata_json_for_scope(
    dataset_path: Path,
    filename: str,
    *,
    metadata_scope: str | None,
) -> dict[str, Any] | None:
    """Read a ds-crawler metadata file, optionally under a modality scope."""
    if metadata_scope is None:
        return read_metadata_json(dataset_path, filename)
    return read_metadata_json(
        dataset_path,
        get_scoped_metadata_filename(metadata_scope, filename),
    )


def load_dataset_config_for_scope(
    path: str | Path,
    *,
    metadata_scope: str | None,
    load_dataset_config_fn: Any = load_dataset_config,
    fallback_to_legacy: bool = True,
) -> Any:
    """Load a ds-crawler config from the legacy or scoped metadata layout.

    ``metadata_scope=None`` preserves the ds-crawler default lookup.  When a
    scope is provided, this first looks for
    ``.ds_crawler/<scope>/ds-crawler.json`` and the corresponding scoped
    dataset head; if the scoped config is absent it falls back to the legacy
    root lookup for backward compatibility.
    """
    dataset_path = Path(path)
    if metadata_scope is None:
        return load_dataset_config_fn({"path": str(path)})

    normalized_scope = validate_metadata_scope(metadata_scope)
    scoped_config = read_metadata_json_for_scope(
        dataset_path,
        _DS_CRAWLER_CONFIG_FILENAME,
        metadata_scope=normalized_scope,
    )
    if scoped_config is None and fallback_to_legacy:
        return load_dataset_config_fn({"path": str(path)})
    if scoped_config is None:
        raise FileNotFoundError(
            f"No {_DS_CRAWLER_CONFIG_FILENAME} found for "
            f"{dataset_path} in metadata_scope={normalized_scope!r}"
        )

    if not isinstance(scoped_config, dict):
        raise ValueError(
            f"Scoped ds-crawler config for metadata_scope={normalized_scope!r} "
            "must be a JSON object"
        )

    config = dict(scoped_config)
    head_file = config.get("head_file", _DS_CRAWLER_HEAD_FILENAME)
    if not isinstance(head_file, str) or not head_file:
        raise ValueError("Scoped ds-crawler config head_file must be a string")

    scoped_head = read_metadata_json_for_scope(
        dataset_path,
        head_file,
        metadata_scope=normalized_scope,
    )
    if scoped_head is not None:
        config["head"] = scoped_head

    return load_dataset_config_fn(config, workdir=str(dataset_path))


def _load_scoped_index_output(
    dataset_path: Path,
    metadata_scope: str,
) -> dict[str, Any] | None:
    index_artifact = read_metadata_json_for_scope(
        dataset_path,
        _DS_CRAWLER_OUTPUT_FILENAME,
        metadata_scope=metadata_scope,
    )
    if index_artifact is None:
        return None
    if not isinstance(index_artifact, dict):
        raise ValueError(
            f"Scoped ds-crawler index for metadata_scope={metadata_scope!r} "
            "must be a JSON object"
        )

    if "head" in index_artifact and "indexing" in index_artifact:
        return index_artifact

    if _hydrate_index_artifact is None:
        return index_artifact

    try:
        ds_config = load_dataset_config_for_scope(
            dataset_path,
            metadata_scope=metadata_scope,
            fallback_to_legacy=False,
        )
    except FileNotFoundError:
        return index_artifact

    return _hydrate_index_artifact(index_artifact, ds_config)


def _overlay_scoped_split(
    dataset_path: Path,
    base_output: dict[str, Any],
    *,
    metadata_scope: str,
    split: str,
) -> dict[str, Any]:
    split_filename = get_split_filename(split)
    split_artifact = read_metadata_json_for_scope(
        dataset_path,
        split_filename,
        metadata_scope=metadata_scope,
    )
    if split_artifact is None:
        raise FileNotFoundError(
            f"Inline split metadata {split_filename!r} not found for "
            f"{dataset_path} in metadata_scope={metadata_scope!r}"
        )
    if not isinstance(split_artifact, dict):
        raise ValueError(
            f"Scoped split metadata {split_filename!r} for "
            f"metadata_scope={metadata_scope!r} must be a JSON object"
        )

    if (
        _hydrate_split_artifact is not None
        and "split" in split_artifact
        and "index" in split_artifact
    ):
        return _hydrate_split_artifact(split_artifact, base_output)

    result = dict(base_output)
    result["index"] = split_artifact
    result["dataset"] = split_artifact
    return result


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

    When ``metadata_scope`` is set, scoped artifacts under
    ``.ds_crawler/<metadata_scope>/`` are preferred and the legacy root-level
    lookup is used only if no scoped index exists.  For legacy unscoped
    loading, modern ds-crawler ``load_dataset_split()`` is used when available;
    older versions are supported by loading the canonical output via
    ``index_dataset_from_path()`` and replacing ``output["dataset"]`` with the
    contents of ``.ds_crawler/split_<name>.json``.
    """
    dataset_path = Path(path)
    normalized_scope = (
        validate_metadata_scope(metadata_scope)
        if metadata_scope is not None
        else None
    )

    if normalized_scope is not None and not force_reindex:
        scoped_output = _load_scoped_index_output(dataset_path, normalized_scope)
        if scoped_output is not None:
            if split is None:
                return scoped_output
            return _overlay_scoped_split(
                dataset_path,
                scoped_output,
                metadata_scope=normalized_scope,
                split=validate_split_name(split),
            )

    if split is None:
        return index_dataset_from_path_fn(
            path,
            strict=strict,
            save_index=save_index,
            force_reindex=force_reindex,
        )

    normalized_split = validate_split_name(split)
    if _load_dataset_split is not None:
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
