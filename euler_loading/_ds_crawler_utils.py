from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
