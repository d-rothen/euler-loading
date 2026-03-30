from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from ds_crawler import DatasetWriter, ZipDatasetWriter, get_dataset_contract

from ._ds_crawler_utils import (
    as_non_empty_str,
    first_non_empty,
)
from .loaders._writer_utils import supports_stream_target


logger = logging.getLogger(__name__)


OutputDestination = str | os.PathLike[str] | DatasetWriter | ZipDatasetWriter


def create_dataset_writer_from_index(
    *,
    index_output: Mapping[str, Any],
    root: str | os.PathLike[str],
    zip: bool = False,
) -> DatasetWriter | ZipDatasetWriter:
    """Create a ds-crawler writer that mirrors an existing index's metadata."""
    contract = get_dataset_contract(dict(index_output))

    indexing = index_output.get("indexing")
    hierarchy = indexing.get("hierarchy") if isinstance(indexing, Mapping) else None
    id_cfg = indexing.get("id") if isinstance(indexing, Mapping) else None
    separator = first_non_empty(
        as_non_empty_str(hierarchy.get("separator")) if isinstance(hierarchy, Mapping) else None,
        as_non_empty_str(id_cfg.get("join_char")) if isinstance(id_cfg, Mapping) else None,
    )
    writer_cls = ZipDatasetWriter if zip else DatasetWriter
    return writer_cls(
        root,
        head=contract.to_mapping(),
        separator=separator,
    )


def _resolve_output_destination(
    *,
    output_root: OutputDestination | Mapping[str, OutputDestination],
    modality_name: str,
) -> OutputDestination:
    if isinstance(output_root, (str, os.PathLike, DatasetWriter, ZipDatasetWriter)):
        return output_root

    destination = output_root.get(modality_name)
    if destination is None:
        raise KeyError(
            f"Missing output destination for modality {modality_name!r}. "
            "Provide a shared destination or a mapping containing this modality."
        )
    return destination


def _destination_location(
    destination: OutputDestination,
    relative_path: str,
) -> str:
    if isinstance(destination, ZipDatasetWriter):
        return f"{destination.root}::{relative_path}"
    if isinstance(destination, DatasetWriter):
        return str(destination.root / relative_path)
    return str(Path(destination) / relative_path)


def _build_writer_full_id(*, relative_path: str, file_id: str) -> str:
    parent_parts = Path(relative_path).parent.parts
    hierarchy_parts = tuple(part for part in parent_parts if part not in ("", "."))
    return "/" + "/".join(hierarchy_parts + (file_id,))


def _destination_rel_exists(
    destination: OutputDestination,
    relative_path: str,
) -> bool:
    seen_paths = getattr(destination, "__euler_loading_written_paths__", None)
    if isinstance(seen_paths, set) and relative_path in seen_paths:
        return True

    if isinstance(destination, ZipDatasetWriter):
        return False
    if isinstance(destination, DatasetWriter):
        return (destination.root / relative_path).exists()
    return (Path(destination) / relative_path).exists()


def _register_destination_rel_path(
    destination: OutputDestination,
    relative_path: str,
) -> None:
    if not isinstance(destination, (DatasetWriter, ZipDatasetWriter)):
        return
    seen_paths = getattr(destination, "__euler_loading_written_paths__", None)
    if not isinstance(seen_paths, set):
        seen_paths = set()
        setattr(destination, "__euler_loading_written_paths__", seen_paths)
    seen_paths.add(relative_path)


def _set_stream_name(stream: Any, basename: str) -> None:
    if getattr(stream, "name", None):
        return
    try:
        setattr(stream, "name", basename)
    except Exception:
        logger.debug("Could not assign basename %r to stream target.", basename)


def _write_value_to_destination(
    *,
    destination: OutputDestination,
    writer: Callable[..., Any],
    value: Any,
    meta: dict[str, Any] | None,
    full_id: str,
    basename: str,
    relative_path: str,
    source_meta: Mapping[str, Any] | None,
    create_dirs: bool,
) -> str:
    if isinstance(destination, ZipDatasetWriter):
        if supports_stream_target(writer):
            with destination.open(
                full_id,
                basename,
                source_meta=dict(source_meta or {}),
            ) as stream:
                _set_stream_name(stream, basename)
                writer(stream, value, meta)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / basename
                writer(str(temp_path), value, meta)
                destination.write(
                    full_id,
                    basename,
                    temp_path.read_bytes(),
                    source_meta=dict(source_meta or {}),
                )
        _register_destination_rel_path(destination, relative_path)
        return _destination_location(destination, relative_path)

    if isinstance(destination, DatasetWriter):
        target_path = destination.get_path(
            full_id,
            basename,
            source_meta=dict(source_meta or {}),
        )
        writer(str(target_path), value, meta)
        _register_destination_rel_path(destination, relative_path)
        return str(target_path)

    target_path = Path(destination) / relative_path
    if create_dirs:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    writer(str(target_path), value, meta)
    return str(target_path)
