"""Resolve a modality's loader and writer from its dataset head.

A head's ``addons.euler_loading`` names a registered loader (``loader``) and a
function in it (``function``); :mod:`euler_loading.registry` maps the name to
modules. Writers live next to the loader functions, in the same module.
"""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from ds_crawler import get_dataset_contract

from ._ds_crawler_utils import as_non_empty_str
from .annotations import get_modality_meta
from .registry import LOADER_VARIANTS, LoaderUnavailableError, loader_registry

if TYPE_CHECKING:
    from .dataset import Modality


logger = logging.getLogger(__name__)

__all__ = [
    "LOADER_VARIANTS",
    "resolve_loader_module",
    "resolve_writer_module",
]


def _euler_loading_addon(index: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The index's ``head.addons.euler_loading``; ``None`` when it has neither.

    A head that is present but malformed raises: it would otherwise read as
    "no loader declared".
    """
    if not isinstance(index.get("head"), Mapping):
        return None
    addon = get_dataset_contract(dict(index)).get_addon("euler_loading")
    return addon if isinstance(addon, Mapping) else None


def resolve_loader_module(name: str, *, variant: str = "gpu") -> ModuleType:
    """Import and return the module registered as loader *name*.

    Example::

        module = resolve_loader_module("vkitti2")
        sky_fn = module.sky_mask  # get a specific function

    Args:
        name: Loader name, as declared by ``addons.euler_loading.loader``:
            a built-in such as ``"vkitti2"`` or a name an installed package
            registers under the ``euler.loaders`` entry-point group.
        variant: ``"gpu"`` for the torch loaders (the default, and what
            automatic resolution uses) or ``"cpu"`` for the NumPy ones.

    Raises:
        LoaderUnavailableError: (a ``ValueError``) if no loader is registered
            as *name*, including when an installed package declares it but
            the declaration was rejected; the message says why.
        ValueError: if *variant* is neither ``"gpu"`` nor ``"cpu"``.
        ImportError: if the module cannot be imported (e.g. torch is missing
            for a ``gpu`` module).
    """
    return loader_registry().get(name).import_module(variant)


def resolve_writer_module(name: str, *, variant: str = "gpu") -> ModuleType:
    """Import and return the writer module for *name*.

    Writers live next to loader functions in the same modules.
    """
    return resolve_loader_module(name, variant=variant)


def _registered_module(loader: Callable[..., Any] | None) -> ModuleType | None:
    """The module of *loader* when it is a registered loader module."""
    module_name = getattr(loader, "__module__", None)
    if not module_name or loader_registry().owner(module_name) is None:
        return None
    return sys.modules.get(module_name)


def _loader_functions(module: ModuleType) -> list[str]:
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and callable(value)
        and get_modality_meta(value) is not None
    )


def loader_accepts_attributes(loader: Callable[..., Any]) -> bool:
    """Return whether *loader* can accept an ``attributes=`` keyword.

    Loaders historically accepted ``(path, meta=None)``.  Per-file
    attributes are opt-in so legacy loaders keep receiving the old call
    shape, while loaders declaring ``attributes`` or ``**kwargs`` receive
    the new context.
    """
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return False

    attributes_param = signature.parameters.get("attributes")
    if (
        attributes_param is not None
        and attributes_param.kind is not inspect.Parameter.POSITIONAL_ONLY
    ):
        return True

    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _resolve_loader(
    *,
    modality_name: str,
    modality: Modality,
    index: Mapping[str, Any],
    variant: str = "gpu",
) -> Callable[..., Any]:
    """Return the effective loader for a modality.

    An explicit ``Modality.loader`` always wins; ``Modality.function`` replaces
    the dataset head's function within its loader module. *variant* selects
    the registered loader's module during automatic resolution.
    """
    if modality.loader is not None:
        return modality.loader

    addon = _euler_loading_addon(index)
    if addon is None:
        raise ValueError(
            f"Modality {modality_name!r}: no explicit loader provided and the "
            f"dataset head at {modality.path!r} has no 'addons.euler_loading' "
            "entry naming one."
        )

    required = ("loader",) if modality.function is not None else ("loader", "function")
    for key in required:
        if key not in addon:
            raise ValueError(
                f"Modality {modality_name!r}: no explicit loader provided and "
                f"'addons.euler_loading.{key}' is missing from the "
                f"dataset-head contract at {modality.path!r}."
            )

    loader_name: str = addon["loader"]
    func_name: str = modality.function or addon["function"]
    try:
        module = resolve_loader_module(loader_name, variant=variant)
    except LoaderUnavailableError as exc:
        raise LoaderUnavailableError(
            f"Modality {modality_name!r} (dataset head at {modality.path!r}): {exc}"
        ) from exc

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(
            f"Modality {modality_name!r}: loader {loader_name!r} "
            f"({module.__name__}) has no callable {func_name!r}. Loader "
            f"functions: {', '.join(_loader_functions(module)) or 'none'}"
        )

    return func


def describe_loader(
    loader: Callable[..., Any],
    *,
    modality: Modality,
    index: Mapping[str, Any],
) -> str:
    """Name the effective loader for logs, e.g. ``real_drive_sim.sky_mask``."""
    module = getattr(loader, "__module__", None) or ""
    name = getattr(loader, "__qualname__", None) or repr(loader)
    label = f"{module.rsplit('.', 1)[-1]}.{name}" if module else name
    registered = loader_registry().owner(module) if module else None
    if registered is not None and not registered.builtin:
        label = f"{label} from {registered.label}"
    if modality.loader is not None:
        return f"{label} (explicit loader)"
    declared = as_non_empty_str((_euler_loading_addon(index) or {}).get("function"))
    if modality.function is not None and declared not in (None, modality.function):
        return f"{label} (the index declares {declared!r})"
    return label


def _writer_name_candidates(read_function_name: str, explicit: Any) -> list[str]:
    names: list[str] = []
    explicit_name = as_non_empty_str(explicit)
    if explicit_name is not None:
        names.append(explicit_name)

    if read_function_name.startswith("read_"):
        names.append(f"write_{read_function_name[len('read_') :]}")
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
    index: Mapping[str, Any],
    loader: Callable[..., Any] | None = None,
    variant: str = "gpu",
) -> Callable[..., Any] | None:
    """Return the effective writer for a modality, if one exists.

    Writers live next to their readers: when the effective *loader* belongs
    to a registered loader module (always, for automatic resolution; also for
    an explicit built-in or extension function), its writer comes from that
    same module, whichever variant it is. Otherwise the head's ``loader``
    module for *variant* is used. A writer that cannot be resolved is not an
    error until one is asked for (``MultiModalDataset.get_writer``); the
    reason is logged as a warning.
    """
    if modality.writer is not None:
        return modality.writer

    addon = _euler_loading_addon(index)
    if addon is None:
        return None

    loader_name = as_non_empty_str(addon.get("loader"))
    read_function_name = as_non_empty_str(modality.function or addon.get("function"))
    if loader_name is None or read_function_name is None:
        return None

    module = _registered_module(loader)
    if module is None:
        try:
            module = resolve_writer_module(loader_name, variant=variant)
        except (LoaderUnavailableError, ImportError) as exc:
            logger.warning(
                "Modality '%s': no writer, because loader %r cannot be resolved "
                "(%s). Set Modality.writer to write this modality.",
                modality_name,
                loader_name,
                exc,
            )
            return None

    # The head's writer_function pairs with its own function, not an override.
    writer_names = _writer_name_candidates(
        read_function_name,
        None if modality.function is not None else addon.get("writer_function"),
    )
    for writer_name in writer_names:
        writer = getattr(module, writer_name, None)
        if callable(writer):
            return writer

    logger.debug(
        "Modality '%s': no writer found in %s for read function %r (tried %s).",
        modality_name,
        module.__name__,
        read_function_name,
        ", ".join(writer_names),
    )
    return None
