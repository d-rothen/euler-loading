"""A versioned, JSON-ready report of the loaders this environment provides.

euler-loading's ``euler.capabilities`` entry: ``euler-suite report`` (and
euler-view, per cluster environment) calls :func:`loader_report`;
``python -m euler_loading.capability_report`` prints it.

Everything comes from the loader registry (:mod:`euler_loading.registry`):
the built-ins and every loader an installed package registers under
``euler.loaders``, each introspected from its ``@modality_meta`` functions.
Nothing is read from a packaged file, so the report describes what *this*
environment can load: a variant whose module cannot be imported here (the
``gpu`` modules without torch, an extension missing a dependency) is listed
as unavailable, with the reason.

The report (``loader_capability_report`` 1.1) carries:

``loaders``
    Coverage per persisted modality key (1.0): which loaders read it, how many
    functions, which file formats, plus identity and value semantics DERIVED
    from euler-dataset-contract, never declared here: ``modality_id`` /
    ``status`` from :func:`euler_dataset_contract.resolve_modality`
    (conditional aliases report their ``candidate_id``), ``value_kind`` from
    the inventory's ``semantics`` block, with point-cloud representations
    reading as ``points``.
``supportedLoaders`` (1.1)
    Every loader's annotated functions, in the shape of the ``loaders.json``
    file euler-view vendored: ``[{name, modalities: [...]}]``.
``registered_loaders`` (1.1)
    Per loader name: its ``provider`` (``"builtin"`` or ``{distribution,
    version}``), the entry point's ``target``, and per variant the ``module``
    and whether it is ``available`` here (else an ``error``).
``problems`` (1.1)
    ``euler.loaders`` declarations that were rejected (conflicts, built-in
    names taken, invalid values, import failures), in
    :class:`euler_dataset_contract.extensions.ExtensionProblem` form.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from types import ModuleType
from typing import Any

from euler_dataset_contract import lookup_alias, lookup_modality, resolve_modality
from euler_dataset_contract.extensions import ExtensionProblem

from .annotations import DECLARED
from .registry import (
    LOADER_API_VERSION,
    LOADER_VARIANTS,
    LOADERS_GROUP,
    LoaderRegistry,
    RegisteredLoader,
    loader_registry,
)

REPORT_KIND = "loader_capability_report"
REPORT_VERSION = "1.1"
PACKAGE_NAME = "euler-loading"

__all__ = [
    "REPORT_KIND",
    "REPORT_VERSION",
    "describe_loader_modules",
    "loader_report",
    "main",
    "supported_loaders",
]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _annotated_functions(module: ModuleType) -> dict[str, dict[str, Any]]:
    """``{function name: _modality_meta}`` for a module's public loader functions."""
    result: dict[str, dict[str, Any]] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        meta = getattr(obj, "_modality_meta", None)
        if isinstance(meta, dict):
            result[name] = meta
    return result


def describe_loader_modules(name: str, modules: dict[str, ModuleType]) -> dict[str, Any]:
    """One ``supportedLoaders`` entry from a loader's imported variant modules.

    Shared fields come from the ``cpu`` annotation when there is one (else
    ``gpu``); each variant contributes its own ``shape``.
    """
    per_variant = {variant: _annotated_functions(module) for variant, module in modules.items()}
    function_names = sorted(set().union(*per_variant.values())) if per_variant else []
    modalities: list[dict[str, Any]] = []
    for function in function_names:
        metas = {variant: funcs[function] for variant, funcs in per_variant.items() if function in funcs}
        canonical = metas.get("cpu") or metas["gpu"]
        entry: dict[str, Any] = {
            "type": canonical["type"],
            "function": function,
            "hierarchical": canonical["hierarchical"],
            "dtype": canonical["dtype"],
            "access": "key",
        }
        # Optional fields only when they carry information.
        if canonical["output_range"] is not None:
            entry["output_range"] = canonical["output_range"]
        if canonical["output_unit"] is not None:
            entry["output_unit"] = canonical["output_unit"]
        if canonical["file_formats"]:
            entry["file_formats"] = canonical["file_formats"]
        if canonical["requires_meta"]:
            entry["requires_meta"] = canonical["requires_meta"]
        if canonical["meta"]:
            entry["meta"] = canonical["meta"]
        for variant in ("cpu", "gpu"):
            if variant in metas:
                entry[variant] = {"shape": metas[variant]["shape"]}
        modalities.append(entry)
    return {"name": name, "modalities": modalities}


def _import_variants(
    loader: RegisteredLoader,
) -> tuple[dict[str, ModuleType], dict[str, dict[str, Any]], list[ExtensionProblem]]:
    """Import every variant module of *loader*; report, never raise, failures."""
    modules: dict[str, ModuleType] = {}
    variants: dict[str, dict[str, Any]] = {}
    problems: list[ExtensionProblem] = []
    for variant in LOADER_VARIANTS:
        path = loader.modules.get(variant)
        if path is None:
            continue
        try:
            modules[variant] = loader.import_module(variant)
        except Exception as exc:  # noqa: BLE001 - reported per variant
            variants[variant] = {"module": path, "available": False, "error": str(exc)}
            if not loader.builtin:
                # A built-in gpu module without torch is an install choice; an
                # extension that cannot import is something to fix.
                problems.append(
                    ExtensionProblem(
                        LOADERS_GROUP,
                        loader.name,
                        "load_failed",
                        str(exc),
                        (loader.distribution,),
                    )
                )
        else:
            variants[variant] = {"module": path, "available": True}
    return modules, variants, problems


def _inventory(
    registry: LoaderRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ExtensionProblem]]:
    supported: list[dict[str, Any]] = []
    registered: dict[str, Any] = {}
    problems: list[ExtensionProblem] = list(registry.problems)
    for name in registry.names():
        loader = registry.loaders[name]
        modules, variants, import_problems = _import_variants(loader)
        problems.extend(import_problems)
        registered[name] = {
            "provider": loader.provider,
            "target": loader.target,
            "variants": variants,
        }
        description = describe_loader_modules(name, modules)
        if description["modalities"]:
            supported.append(description)
    return supported, registered, problems


def supported_loaders(registry: LoaderRegistry | None = None) -> list[dict[str, Any]]:
    """Every registered loader's annotated functions, in ``loaders.json`` shape."""
    return _inventory(registry or loader_registry())[0]


def _value_kind(alias_key: str, modality_id: str | None) -> str | None:
    """The contract's value kind for this persisted key, when published."""
    if modality_id is None:
        return None
    modality = lookup_modality(modality_id)
    semantics = (modality or {}).get("semantics")
    if not semantics:
        return None
    alias = lookup_alias(alias_key) or {}
    representation = alias.get("representation", {})
    conditions = {
        path.removeprefix("representation."): expected
        for path, expected in alias.get("conditions", {}).items()
    }
    # A fixed/conditional point-cloud representation carries points, whatever
    # the base quantity's dense form would be (e.g. sparse_depth).
    if {**representation, **conditions}.get("form") == "point_cloud":
        return "points"
    return semantics.get("value_kind")


def _coverage(supported: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate loader coverage per persisted modality key."""
    by_type: dict[str, dict[str, Any]] = {}
    for loader in supported:
        for modality in loader.get("modalities", []):
            key = modality.get("type")
            if not key or key == DECLARED:
                # A per-file declared type is not a key this loader covers.
                continue
            entry = by_type.setdefault(
                key, {"datasets": [], "functions": 0, "file_formats": []}
            )
            if loader["name"] not in entry["datasets"]:
                entry["datasets"].append(loader["name"])
            entry["functions"] += 1
            for file_format in modality.get("file_formats", []) or []:
                if file_format not in entry["file_formats"]:
                    entry["file_formats"].append(file_format)

    coverage: dict[str, Any] = {}
    for key in sorted(by_type):
        entry = by_type[key]
        resolution = resolve_modality(key)
        modality_id = resolution.modality_id or resolution.candidate_id
        coverage[key] = {
            "reads": True,
            "datasets": sorted(entry["datasets"]),
            "functions": entry["functions"],
            "file_formats": sorted(entry["file_formats"]),
            "modality_id": modality_id,
            "status": resolution.status,
            "value_kind": _value_kind(key, modality_id),
        }
    return coverage


def loader_report(registry: LoaderRegistry | None = None) -> dict[str, Any]:
    """The loader capability report for this environment (see the module docs)."""
    supported, registered, problems = _inventory(registry or loader_registry())
    return {
        "contract": {"kind": REPORT_KIND, "version": REPORT_VERSION},
        "package": {"name": PACKAGE_NAME, "version": _package_version(PACKAGE_NAME)},
        "packages": {
            "euler-dataset-contract": _package_version("euler-dataset-contract"),
        },
        "extension_api": {"group": LOADERS_GROUP, "api_version": LOADER_API_VERSION},
        "loaders": _coverage(supported),
        "supportedLoaders": supported,
        "registered_loaders": registered,
        "problems": [problem.describe() for problem in problems],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m euler_loading.capability_report",
        description="Print the loaders this environment provides, as JSON.",
    )
    parser.add_argument(
        "--supported-loaders",
        action="store_true",
        help="print only {\"supportedLoaders\": [...]}, the former loaders.json",
    )
    args = parser.parse_args(argv)
    if args.supported_loaders:
        document: dict[str, Any] = {"supportedLoaders": supported_loaders()}
    else:
        document = loader_report()
    print(json.dumps(document, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
