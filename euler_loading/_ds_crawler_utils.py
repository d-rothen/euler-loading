from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ds_crawler.zip_utils import read_metadata_json

try:
    from ds_crawler import load_dataset_split as _load_dataset_split
except ImportError:
    _load_dataset_split = None


DS_CRAWLER_STRUCTURAL_KEYS = frozenset({
    "name",
    "type",
    "id_regex",
    "id_regex_join_char",
    "euler_train",
    "dataset",
    "hierarchy_regex",
    "named_capture_group_value_separator",
    "sampled",
})


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
    return {
        str(key): value
        for key, value in index_output.items()
        if key not in DS_CRAWLER_STRUCTURAL_KEYS
    }


_SPLIT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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


def load_index_output(
    path: str | Path,
    *,
    split: str | None,
    index_dataset_from_path_fn: Any,
    strict: bool = False,
    save_index: bool = False,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Load a ds-crawler output, optionally overlaying an inline split.

    When the installed ds-crawler version exposes ``load_dataset_split()``, it
    is used directly. Older versions are supported by loading the canonical
    output via ``index_dataset_from_path()`` and replacing ``output["dataset"]``
    with the contents of ``.ds_crawler/split_<name>.json``.
    """
    dataset_path = Path(path)

    if split is None:
        return index_dataset_from_path_fn(
            path,
            strict=strict,
            save_index=save_index,
            force_reindex=force_reindex,
        )

    normalized_split = validate_split_name(split)
    if _load_dataset_split is not None:
        return _load_dataset_split(
            path,
            normalized_split,
            strict=strict,
            save_index=save_index,
            force_reindex=force_reindex,
        )

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
    result["dataset"] = split_dataset
    return result
