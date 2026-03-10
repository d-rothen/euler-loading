from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from ._ds_crawler_utils import as_non_empty_str

if TYPE_CHECKING:
    from .dataset import Modality


logger = logging.getLogger(__name__)


_LOADER_MODULES: dict[str, str] = {
    "vkitti2": "euler_loading.loaders.gpu.vkitti2",
    "real_drive_sim": "euler_loading.loaders.gpu.real_drive_sim",
    "generic_dense_depth": "euler_loading.loaders.gpu.generic_dense_depth",
}


def resolve_loader_module(name: str) -> ModuleType:
    """Import and return the GPU loader module for *name*.

    Example::

        module = resolve_loader_module("vkitti2")
        sky_fn = module.sky_mask  # get a specific function

    Raises:
        ValueError: If *name* does not match any known loader.
    """
    module_path = _LOADER_MODULES.get(name)
    if module_path is None:
        available = ", ".join(sorted(_LOADER_MODULES))
        raise ValueError(
            f"Unknown loader {name!r}. Available loaders: {available}"
        )
    return importlib.import_module(module_path)


def resolve_writer_module(name: str) -> ModuleType:
    """Import and return the writer module for *name*.

    Writers live next to loader functions in the same modules.
    """
    return resolve_loader_module(name)


def _resolve_loader(
    *,
    modality_name: str,
    modality: Modality,
    index: dict[str, Any],
) -> Callable[..., Any]:
    """Return the effective loader for a modality."""
    if modality.loader is not None:
        return modality.loader

    euler_loading_meta = index.get("euler_loading")
    if not isinstance(euler_loading_meta, Mapping):
        raise ValueError(
            f"Modality {modality_name!r}: no explicit loader provided and the "
            f"ds-crawler index at {modality.path!r} does not contain an "
            f"'euler_loading' property."
        )

    for key in ("loader", "function"):
        if key not in euler_loading_meta:
            raise ValueError(
                f"Modality {modality_name!r}: no explicit loader provided and "
                f"'euler_loading.{key}' is missing from the ds-crawler index "
                f"at {modality.path!r}."
            )

    module_name: str = euler_loading_meta["loader"]
    func_name: str = euler_loading_meta["function"]

    module = resolve_loader_module(module_name)

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        available = [
            attr for attr in dir(module)
            if not attr.startswith("_") and callable(getattr(module, attr))
        ]
        raise ValueError(
            f"Modality {modality_name!r}: loader module {module_name!r} has no "
            f"callable {func_name!r}. Available: {', '.join(available)}"
        )

    return func


def _writer_name_candidates(read_function_name: str, explicit: Any) -> list[str]:
    names: list[str] = []
    explicit_name = as_non_empty_str(explicit)
    if explicit_name is not None:
        names.append(explicit_name)

    if read_function_name.startswith("read_"):
        names.append(f"write_{read_function_name[len('read_'):]}")
    names.append(f"write_{read_function_name}")

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _resolve_writer(
    *,
    modality_name: str,
    modality: Modality,
    index: dict[str, Any],
) -> Callable[..., Any] | None:
    """Return the effective writer for a modality, if available."""
    if modality.writer is not None:
        return modality.writer

    euler_loading_meta = index.get("euler_loading")
    if not isinstance(euler_loading_meta, Mapping):
        return None

    module_name = as_non_empty_str(euler_loading_meta.get("loader"))
    read_function_name = as_non_empty_str(euler_loading_meta.get("function"))
    if module_name is None or read_function_name is None:
        return None

    try:
        module = resolve_writer_module(module_name)
    except ValueError:
        logger.warning(
            "Modality '%s': cannot resolve writer module %r.",
            modality_name,
            module_name,
        )
        return None

    writer_names = _writer_name_candidates(
        read_function_name,
        euler_loading_meta.get("writer_function"),
    )
    for writer_name in writer_names:
        writer = getattr(module, writer_name, None)
        if callable(writer):
            return writer

    logger.debug(
        "Modality '%s': no writer found in %s for read function %r (tried %s).",
        modality_name,
        module_name,
        read_function_name,
        ", ".join(writer_names),
    )
    return None
