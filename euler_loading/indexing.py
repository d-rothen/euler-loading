from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FileRecord:
    """A file entry enriched with resolved calibration paths.

    Attributes:
        file_entry: The raw ds-crawler file dict
                    (keys: ``"id"``, ``"path"``, ``"path_properties"``,
                    ``"basename_properties"``).
        intrinsics_path: Absolute path to the intrinsics file inherited from the
                         nearest ancestor node, or ``None``.
        extrinsics_path: Absolute path to the extrinsics file inherited from the
                         nearest ancestor node, or ``None``.
    """

    file_entry: dict[str, Any]
    intrinsics_path: str | None
    extrinsics_path: str | None


def collect_files_with_calibration(
    node: dict[str, Any],
    dataset_root: str,
    inherited_intrinsics: str | None = None,
    inherited_extrinsics: str | None = None,
) -> list[FileRecord]:
    """Recursively collect all files from a ds-crawler hierarchy node.

    Calibration paths are inherited from ancestor nodes following ds-crawler's
    convention: if a node has ``"camera_intrinsics"`` or
    ``"camera_extrinsics"``, that value overrides the inherited one for all
    descendants.

    Args:
        node: A dict node from ``index["dataset"]``.  May contain
              ``"children"``, ``"files"``, ``"camera_intrinsics"``, and/or
              ``"camera_extrinsics"``.
        dataset_root: Absolute path to the dataset root directory.  Calibration
                      relative paths are resolved against this.
        inherited_intrinsics: Relative intrinsics path from an ancestor, or
                              ``None``.
        inherited_extrinsics: Relative extrinsics path from an ancestor, or
                              ``None``.

    Returns:
        List of :class:`FileRecord` objects for every file entry in the subtree.
    """
    # Override inherited calibration if this node provides its own.
    intrinsics_rel = node.get("camera_intrinsics", inherited_intrinsics)
    extrinsics_rel = node.get("camera_extrinsics", inherited_extrinsics)

    # Resolve to absolute paths.
    intrinsics_abs = f"{dataset_root}/{intrinsics_rel}" if intrinsics_rel else None
    extrinsics_abs = f"{dataset_root}/{extrinsics_rel}" if extrinsics_rel else None

    records: list[FileRecord] = []

    for file_entry in node.get("files", []):
        records.append(
            FileRecord(
                file_entry=file_entry,
                intrinsics_path=intrinsics_abs,
                extrinsics_path=extrinsics_abs,
            )
        )

    for child in node.get("children", {}).values():
        records.extend(
            collect_files_with_calibration(
                child, dataset_root, intrinsics_rel, extrinsics_rel
            )
        )

    return records
