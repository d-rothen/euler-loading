"""Which loader modules a dataset head can name: the ``euler.loaders`` registry.

A dataset head names its decoder with ``addons.euler_loading.loader`` (a
token such as ``"vkitti2"``) and ``function`` (a callable in that module).
The token resolves through installed entry points in the ``euler.loaders``
group. euler-loading registers its own loaders there exactly as a
third-party package registers its own::

    [project.entry-points."euler.loaders"]
    # one module serves both variants
    my_dataset = "my_lab.loaders.my_dataset"
    # or a mapping naming each variant's module
    my_other = "my_lab.loaders:MY_OTHER"   # {"gpu": "...", "cpu": "...", "api_version": "1.0"}

``pip install`` of that package is the whole integration: every process
(CLI, DataLoader worker, SLURM job) reads the same installed entry points.

Rules (on top of :mod:`euler_dataset_contract.extensions`):

* **Built-ins win.** A name euler-loading registers cannot be taken over.
  Another package declaring it is rejected with a ``shadows_builtin`` problem
  and the built-in keeps working.
* Two third-party packages claiming one name is a ``conflict``: neither is
  registered.
* Nothing is hidden. :attr:`LoaderRegistry.problems` keeps every rejected or
  broken declaration, ``loader_report()`` publishes them, and a head naming a
  loader that failed to register gets an error quoting why.
* Building the registry imports only the entry-point targets: a mapping names
  its modules, and those are imported when a head asks for them. The module
  form imports the module itself, so packages with heavy imports should
  prefer the mapping form.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import Any

from euler_dataset_contract import validate_token
from euler_dataset_contract.extensions import (
    GROUPS,
    Extension,
    ExtensionError,
    ExtensionProblem,
    iter_extensions,
    load_extensions,
    strict_mode,
)
from euler_dataset_contract.suite import normalize_name

LOADERS_GROUP = "euler.loaders"
LOADER_API_VERSION = GROUPS[LOADERS_GROUP].api_version
BUILTIN_DISTRIBUTION = "euler-loading"

#: Loader variants. ``gpu`` modules return torch tensors, ``cpu`` modules
#: NumPy arrays; automatic resolution from a dataset head uses ``gpu``.
LOADER_VARIANTS = ("gpu", "cpu")

_MAPPING_KEYS = frozenset((*LOADER_VARIANTS, "api_version"))
_MODULE_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


class LoaderUnavailableError(ValueError):
    """A head names a loader this environment cannot provide."""


@dataclass(frozen=True)
class RegisteredLoader:
    """One name a dataset head can use, and the modules behind it."""

    name: str
    #: ``{variant: module path}`` for the variants this loader offers.
    modules: Mapping[str, str]
    #: The entry point's target, e.g. ``"my_lab.loaders.my_dataset"``.
    target: str
    distribution: str | None
    version: str | None
    builtin: bool

    @property
    def provider(self) -> str | dict[str, str | None]:
        """``"builtin"``, or the ``{distribution, version}`` that provides it."""
        if self.builtin:
            return "builtin"
        return {"distribution": self.distribution, "version": self.version}

    @property
    def label(self) -> str:
        """Who provides this loader, for messages."""
        if self.builtin:
            return "built-in"
        if self.distribution is None:
            return "an unknown package"
        return f"{self.distribution} {self.version or '(unknown version)'}"

    def module_path(self, variant: str = "gpu") -> str:
        _check_variant(variant)
        path = self.modules.get(variant)
        if path is None:
            offered = ", ".join(self.modules)
            raise LoaderUnavailableError(
                f"Loader {self.name!r} ({self.label}) has no {variant} variant; "
                f"it provides: {offered}."
            )
        return path

    def import_module(self, variant: str = "gpu") -> ModuleType:
        """Import this loader's module for *variant*.

        Any import failure is re-raised as an ``ImportError`` (a
        ``ModuleNotFoundError`` when a module is missing) naming the loader,
        its provider and the module, so a missing dependency of a third-party
        loader reads as such.
        """
        path = self.module_path(variant)
        try:
            return importlib.import_module(path)
        except Exception as exc:
            message = (
                f"Loader {self.name!r} ({self.label}): cannot import its {variant} "
                f"module {path!r}: {type(exc).__name__}: {exc}"
            )
            error = ModuleNotFoundError if isinstance(exc, ModuleNotFoundError) else ImportError
            raise error(
                message, name=getattr(exc, "name", None), path=getattr(exc, "path", None)
            ) from exc


@dataclass(frozen=True)
class LoaderRegistry:
    """Every loader name usable in this environment, plus what was rejected."""

    loaders: Mapping[str, RegisteredLoader]
    problems: tuple[ExtensionProblem, ...] = field(default_factory=tuple)

    def names(self) -> list[str]:
        return sorted(self.loaders)

    def problems_for(self, name: str) -> list[ExtensionProblem]:
        return [problem for problem in self.problems if problem.name == name]

    def get(self, name: str) -> RegisteredLoader:
        """The loader registered as *name*; raises with the reason it is not."""
        loader = self.loaders.get(name)
        if loader is not None:
            return loader
        problems = self.problems_for(name)
        if problems:
            reasons = " ".join(problem.message for problem in problems)
            raise LoaderUnavailableError(
                f"Loader {name!r} is declared by an installed package but is not "
                f"available: {reasons}"
            )
        if not any(loader.builtin for loader in self.loaders.values()):
            raise LoaderUnavailableError(
                f"Unknown loader {name!r}: no built-in loaders are registered. "
                "euler-loading registers its loaders as `euler.loaders` entry "
                "points, which exist only in an installed package: install it "
                "(`pip install -e .` in a checkout) instead of importing it from "
                "a source tree."
            )
        raise LoaderUnavailableError(
            f"Unknown loader {name!r}. Registered loaders: "
            f"{', '.join(self.names())}. A package adds its own under the "
            "`euler.loaders` entry-point group (README: 'Extending: your own "
            "loaders')."
        )

    def owner(self, module_name: str) -> RegisteredLoader | None:
        """The registered loader one of whose variants is *module_name*.

        A built-in module stays built-in even if a package registers it again
        under another name.
        """
        owners = [
            self.loaders[name]
            for name in self.names()
            if module_name in self.loaders[name].modules.values()
        ]
        return next((loader for loader in owners if loader.builtin), owners[0] if owners else None)


def _check_variant(variant: str) -> None:
    if variant not in LOADER_VARIANTS:
        raise ValueError(
            f"Unknown loader variant {variant!r}. Available variants: "
            f"{', '.join(LOADER_VARIANTS)}"
        )


def _is_builtin(extension: Extension) -> bool:
    return (
        extension.distribution is not None
        and normalize_name(extension.distribution) == BUILTIN_DISTRIBUTION
    )


def _registered(extension: Extension, value: Any) -> RegisteredLoader:
    """Validate one ``euler.loaders`` value (the ``validate`` hook)."""
    validate_token(extension.name, "an euler.loaders entry-point name")
    if isinstance(value, ModuleType):
        modules = {variant: value.__name__ for variant in LOADER_VARIANTS}
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - _MAPPING_KEYS)
        if unknown:
            raise ValueError(
                f"unknown key(s) {', '.join(map(repr, unknown))}; a loader mapping "
                "has 'gpu' and/or 'cpu' module paths and an optional 'api_version'"
            )
        modules = {}
        for variant in LOADER_VARIANTS:
            path = value.get(variant)
            if path is None:
                continue
            if not isinstance(path, str) or not _MODULE_PATH.fullmatch(path):
                raise ValueError(f"{variant!r} must be a dotted module path, got {path!r}")
            modules[variant] = path
        if not modules:
            raise ValueError("the mapping names neither a 'gpu' nor a 'cpu' module")
    else:
        raise TypeError(
            "the entry point must name a module, or a mapping of variant module "
            f"paths; it names a {type(value).__name__}"
        )
    return RegisteredLoader(
        name=extension.name,
        modules=MappingProxyType(modules),
        target=extension.target,
        distribution=extension.distribution,
        version=extension.version,
        builtin=_is_builtin(extension),
    )


def _shadowing_problem(extension: Extension) -> ExtensionProblem:
    provider = extension.distribution or "An unknown package"
    return ExtensionProblem(
        LOADERS_GROUP,
        extension.name,
        "shadows_builtin",
        f"{provider} declares loader {extension.name!r} ({extension.target}), a "
        "name euler-loading ships built in. The built-in is used and this "
        "declaration is ignored; rename the entry point to make the loader "
        "nameable from a dataset head.",
        (extension.distribution,),
    )


def build_loader_registry() -> LoaderRegistry:
    """Read the ``euler.loaders`` entry points (uncached; see :func:`loader_registry`)."""
    result = load_extensions(LOADERS_GROUP, validate=_registered, strict=False)
    loaders: dict[str, RegisteredLoader] = {
        item.extension.name: item.value for item in result.loaded
    }
    problems: list[ExtensionProblem] = []
    declared: list[Extension] | None = None
    for problem in result.problems:
        if problem.code == "conflict":
            declared = declared if declared is not None else iter_extensions(LOADERS_GROUP)
            candidates = [ext for ext in declared if ext.name == problem.name]
            builtins = [ext for ext in candidates if _is_builtin(ext)]
            if len(builtins) == 1:
                # A built-in name is not up for grabs: keep it, reject the rest.
                builtin = builtins[0]
                loaders[builtin.name] = _registered(builtin, builtin.load())
                problems.extend(
                    _shadowing_problem(ext) for ext in candidates if ext is not builtin
                )
                continue
        problems.append(problem)

    registry = LoaderRegistry(MappingProxyType(dict(sorted(loaders.items()))), tuple(problems))
    if problems and strict_mode():
        raise ExtensionError("; ".join(problem.message for problem in problems))
    return registry


#: Built-in loaders that read a file by its modality key and extension rather
#: than by a dataset's layout, in the order :func:`generic_reader` asks them.
GENERIC_LOADERS = ("generic_dense_depth", "generic")


def generic_reader(
    modality_key: str,
    file_type: str,
    *,
    registry: LoaderRegistry | None = None,
) -> dict[str, str] | None:
    """The built-in loader function that reads *modality_key* files of *file_type*.

    For writers of new datasets (euler-inference's outputs in a key's default
    storage format): the result, ``{"loader": ..., "function": ...}``, is the
    ``addons.euler_loading`` of the head they write, so euler-loading loads the
    files back. ``None`` when no generic loader reads that key from that file
    type without extra head meta (a ``requires_meta`` loader, such as a sky
    mask that needs its ``skyclass``, is never chosen).
    """
    extension = "." + str(file_type).lower().lstrip(".")
    registry = registry or loader_registry()
    for name in GENERIC_LOADERS:
        loader = registry.loaders.get(name)
        if loader is None:
            continue
        variant = "cpu" if "cpu" in loader.modules else "gpu"
        module = loader.import_module(variant)
        for function in sorted(vars(module)):
            value = getattr(module, function)
            annotation = getattr(value, "_modality_meta", None)
            if (
                function.startswith("_")
                or not callable(value)
                or not isinstance(annotation, dict)
            ):
                continue
            if (
                annotation.get("type") == modality_key
                and extension in (annotation.get("file_formats") or [])
                and not annotation.get("requires_meta")
                and not annotation.get("hierarchical")
            ):
                return {"loader": name, "function": function}
    return None


_REGISTRY: LoaderRegistry | None = None


def loader_registry(*, refresh: bool = False) -> LoaderRegistry:
    """The process-wide registry, built once from the installed entry points.

    Pass ``refresh=True`` after installing a package into a running process.
    """
    global _REGISTRY
    if _REGISTRY is None or refresh:
        _REGISTRY = build_loader_registry()
    return _REGISTRY


__all__ = [
    "BUILTIN_DISTRIBUTION",
    "GENERIC_LOADERS",
    "LOADERS_GROUP",
    "LOADER_API_VERSION",
    "LOADER_VARIANTS",
    "LoaderRegistry",
    "LoaderUnavailableError",
    "RegisteredLoader",
    "build_loader_registry",
    "generic_reader",
    "loader_registry",
]
