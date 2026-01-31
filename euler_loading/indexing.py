from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FileRecord:
    """A file entry with its position in the dataset hierarchy.

    Attributes:
        file_entry: The raw ds-crawler file dict
                    (keys: ``"id"``, ``"path"``, ``"path_properties"``,
                    ``"basename_properties"``).
        hierarchy_path: Tuple of children keys from the dataset root to this
                        file's parent node.  Used for matching against
                        hierarchical modalities.
    """

    file_entry: dict[str, Any]
    hierarchy_path: tuple[str, ...] = ()


def collect_files(
    node: dict[str, Any],
    _hierarchy_path: tuple[str, ...] = (),
) -> list[FileRecord]:
    """Recursively collect all files from a ds-crawler hierarchy node.

    Args:
        node: A dict node from ``index["dataset"]``.  May contain
              ``"children"`` and/or ``"files"``.
        _hierarchy_path: Internal accumulator — callers should not set this.

    Returns:
        List of :class:`FileRecord` objects for every file entry in the subtree.
    """
    records: list[FileRecord] = []

    for file_entry in node.get("files", []):
        records.append(
            FileRecord(
                file_entry=file_entry,
                hierarchy_path=_hierarchy_path,
            )
        )

    for child_key, child_node in node.get("children", {}).items():
        records.extend(
            collect_files(
                child_node,
                _hierarchy_path + (child_key,),
            )
        )

    return records


def collect_hierarchical_files(
    node: dict[str, Any],
    _hierarchy_path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Recursively collect files from a ds-crawler hierarchy, keyed by level.

    Unlike :func:`collect_files_with_calibration`, this function preserves the
    hierarchy level at which each file was found.  It is used for *hierarchical
    modalities* whose files live at intermediate levels and apply to all
    descendants in the regular modality hierarchy.

    Args:
        node: A dict node from ``index["dataset"]``.
        _hierarchy_path: Internal accumulator — callers should not set this.

    Returns:
        Mapping from hierarchy path (tuple of children keys) to the list of
        file entries found at that exact level.
    """
    result: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    files = node.get("files", [])
    if files:
        result[_hierarchy_path] = list(files)

    for child_key, child_node in node.get("children", {}).items():
        child_result = collect_hierarchical_files(
            child_node, _hierarchy_path + (child_key,)
        )
        result.update(child_result)

    return result


def match_hierarchical_files(
    hierarchy_path: tuple[str, ...],
    files_by_level: dict[tuple[str, ...], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return hierarchical-modality files at ancestor levels of *hierarchy_path*.

    All prefixes of *hierarchy_path* (including itself and the root ``()``) are
    checked.  When different levels provide a file with the same ``"id"``, the
    deepest (most specific) level wins.

    Args:
        hierarchy_path: The hierarchy position of a regular-modality sample.
        files_by_level: Output of :func:`collect_hierarchical_files`.

    Returns:
        De-duplicated list of file entries, deepest level taking precedence for
        duplicate IDs.
    """
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []

    # Walk from deepest to shallowest so deeper files take precedence.
    for depth in range(len(hierarchy_path), -1, -1):
        prefix = hierarchy_path[:depth]
        for entry in files_by_level.get(prefix, []):
            if entry["id"] not in seen_ids:
                seen_ids.add(entry["id"])
                result.append(entry)

    return result
