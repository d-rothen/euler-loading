"""Generate ``loaders.json`` by introspecting annotated loader functions.

Usage::

    python -m euler_loading.loaders.generate
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path


def _discover_modules(package_path: str, package_name: str) -> dict[str, object]:
    """Import all public modules under *package_name*."""
    pkg = importlib.import_module(package_name)
    modules: dict[str, object] = {}
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        if name.startswith("_"):
            continue
        modules[name] = importlib.import_module(f"{package_name}.{name}")
    return modules


def _collect_annotated(module: object) -> dict[str, dict]:
    """Return ``{func_name: _modality_meta}`` for annotated functions."""
    result: dict[str, dict] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        meta = getattr(obj, "_modality_meta", None)
        if meta is not None:
            result[name] = meta
    return result


def generate() -> dict:
    """Build the ``supportedLoaders`` structure from annotations."""
    cpu_modules = _discover_modules(
        "euler_loading.loaders.cpu", "euler_loading.loaders.cpu"
    )
    gpu_modules = _discover_modules(
        "euler_loading.loaders.gpu", "euler_loading.loaders.gpu"
    )

    all_loader_names = sorted(set(cpu_modules) | set(gpu_modules))
    supported_loaders: list[dict] = []

    for loader_name in all_loader_names:
        cpu_mod = cpu_modules.get(loader_name)
        gpu_mod = gpu_modules.get(loader_name)

        cpu_funcs = _collect_annotated(cpu_mod) if cpu_mod else {}
        gpu_funcs = _collect_annotated(gpu_mod) if gpu_mod else {}

        # Union of all annotated function names for this loader
        all_func_names = sorted(set(cpu_funcs) | set(gpu_funcs))
        if not all_func_names:
            continue

        modalities: list[dict] = []
        for func_name in all_func_names:
            cpu_meta = cpu_funcs.get(func_name)
            gpu_meta = gpu_funcs.get(func_name)
            # Use whichever variant is available for shared fields
            canonical = cpu_meta or gpu_meta
            assert canonical is not None

            entry: dict = {
                "type": canonical["type"],
                "function": func_name,
                "hierarchical": canonical["hierarchical"],
                "dtype": canonical["dtype"],
                "access": "key",
            }

            # Only include optional top-level fields when they carry information
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

            if cpu_meta:
                entry["cpu"] = {"shape": cpu_meta["shape"]}
            if gpu_meta:
                entry["gpu"] = {"shape": gpu_meta["shape"]}

            modalities.append(entry)

        supported_loaders.append(
            {"name": loader_name, "modalities": modalities}
        )

    return {"supportedLoaders": supported_loaders}


def main() -> None:
    result = generate()
    out_path = Path(__file__).parent / "loaders.json"
    out_path.write_text(json.dumps(result, indent=4) + "\n")
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
