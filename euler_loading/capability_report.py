"""A versioned, JSON-ready report of the loader coverage this install provides.

The loader-side counterpart of euler-dataset-contract's ``modality_report()``
(change L1 in euler-view's ``docs/modality-semantics-and-registry.md``):
one payload per environment answering *which persisted modality keys does this
install load, and what kind of value does each one carry?*  euler-view's
cluster compatibility routine runs it over SSH
(``python -m euler_loading.capability_report``) and diffs it against its
synced contract snapshot and active modality catalog.

Identity and semantics are DERIVED from euler-dataset-contract — never
declared here — so loader coverage and the contract vocabulary cannot drift
apart silently:

- ``modality_id`` / ``status`` come from :func:`euler_dataset_contract.resolve_modality`
  (conditional aliases report their ``candidate_id``; the condition is the
  alias's own evidence requirement, not the loader's concern);
- ``value_kind`` comes from the contract inventory's per-modality
  ``semantics`` block (contract change C1), refined by the alias's fixed
  representation (``form: point_cloud`` reads as ``points``). Absent until the
  installed contract ships semantics for that modality.

Writers are not reported yet: euler-loading's writing is descriptor/codec
driven without a per-type registry.  A ``writes`` facet can be added when one
exists.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from euler_dataset_contract import lookup_alias, lookup_modality, resolve_modality

REPORT_KIND = "loader_capability_report"
REPORT_VERSION = "1.0"
PACKAGE_NAME = "euler-loading"

__all__ = ["REPORT_KIND", "REPORT_VERSION", "loader_report", "main"]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _supported_loaders() -> list[dict[str, Any]]:
    """The packaged ``loaders.json`` (what this install ships)."""
    path = Path(__file__).parent / "loaders" / "generate" / "loaders.json"
    return json.loads(path.read_text())["supportedLoaders"]


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


def loader_report() -> dict[str, Any]:
    """Aggregate loader coverage per persisted modality key."""
    by_type: dict[str, dict[str, Any]] = {}
    for loader in _supported_loaders():
        for modality in loader.get("modalities", []):
            key = modality.get("type")
            if not key:
                continue
            entry = by_type.setdefault(
                key,
                {"reads": True, "datasets": [], "functions": 0, "file_formats": []},
            )
            if loader["name"] not in entry["datasets"]:
                entry["datasets"].append(loader["name"])
            entry["functions"] += 1
            for file_format in modality.get("file_formats", []) or []:
                if file_format not in entry["file_formats"]:
                    entry["file_formats"].append(file_format)

    loaders: dict[str, Any] = {}
    for key in sorted(by_type):
        entry = by_type[key]
        resolution = resolve_modality(key)
        modality_id = resolution.modality_id or resolution.candidate_id
        loaders[key] = {
            "reads": True,
            "datasets": sorted(entry["datasets"]),
            "functions": entry["functions"],
            "file_formats": sorted(entry["file_formats"]),
            "modality_id": modality_id,
            "status": resolution.status,
            "value_kind": _value_kind(key, modality_id),
        }

    return {
        "contract": {"kind": REPORT_KIND, "version": REPORT_VERSION},
        "package": {"name": PACKAGE_NAME, "version": _package_version(PACKAGE_NAME)},
        "packages": {
            "euler-dataset-contract": _package_version("euler-dataset-contract"),
        },
        "loaders": loaders,
    }


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001 - uniform entry
    print(json.dumps(loader_report(), indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
