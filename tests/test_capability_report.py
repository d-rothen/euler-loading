"""The loader capability report (euler-view change L1).

Loader coverage must stay resolvable against euler-dataset-contract: a loader
emitting a persisted key the contract does not know is release-blocking drift.
Value kinds are derived from the contract's ``semantics`` block (change C1)
and tighten automatically once the installed contract ships them.
"""

from __future__ import annotations

import euler_loading.capability_report as capability_report
from euler_loading.capability_report import loader_report


def test_report_shape() -> None:
    report = loader_report()
    assert report["contract"] == {"kind": "loader_capability_report", "version": "1.0"}
    assert report["package"]["name"] == "euler-loading"
    assert "euler-dataset-contract" in report["packages"]
    assert report["loaders"], "a loader install must cover at least one key"


def test_every_loader_type_is_known_to_the_contract() -> None:
    """The agreement gate: no loader may emit a key the contract never heard of."""
    report = loader_report()
    unknown = {
        key: entry["status"]
        for key, entry in report["loaders"].items()
        if entry["status"] == "unknown"
    }
    assert not unknown, (
        f"loader types unknown to euler-dataset-contract: {sorted(unknown)} — "
        "register the alias in the contract (or rename the loader type) before release"
    )


def test_aggregation_covers_datasets_and_formats() -> None:
    loaders = loader_report()["loaders"]
    depth = loaders["depth"]
    assert depth["reads"] is True
    assert len(depth["datasets"]) >= 5
    assert depth["datasets"] == sorted(depth["datasets"])
    assert depth["file_formats"] == sorted(set(depth["file_formats"]))

    sparse = loaders["sparse_depth"]
    assert sparse["modality_id"] == "geometry.scene.points"
    assert sparse["status"] == "conditional"


def test_value_kind_fills_once_the_contract_ships_semantics() -> None:
    """Forward-tightening: wherever the installed contract publishes semantics,
    the report must carry a value kind (C1 coupling)."""
    from euler_dataset_contract import lookup_modality

    for key, entry in loader_report()["loaders"].items():
        if not entry["modality_id"]:
            continue
        semantics = (lookup_modality(entry["modality_id"]) or {}).get("semantics")
        if semantics:
            assert entry["value_kind"], key


def test_value_kind_derivation_with_contract_semantics(monkeypatch) -> None:
    modalities = {
        "geometry.camera.depth": {"semantics": {"value_kind": "scalar_field"}},
        "geometry.scene.points": {"semantics": {"value_kind": "points"}},
        "appearance.camera.color": {},  # no semantics published yet
    }
    aliases = {
        "depth": {"id": "geometry.camera.depth"},
        "sparse_depth": {
            "id": "geometry.scene.points",
            "conditions": {"representation.form": "point_cloud"},
        },
        "lidar_depth_map": {
            "id": "geometry.camera.depth",
            "representation": {"form": "point_cloud"},
        },
        "rgb": {"id": "appearance.camera.color"},
    }
    monkeypatch.setattr(capability_report, "lookup_modality", modalities.get)
    monkeypatch.setattr(capability_report, "lookup_alias", aliases.get)

    assert capability_report._value_kind("depth", "geometry.camera.depth") == "scalar_field"
    assert capability_report._value_kind("sparse_depth", "geometry.scene.points") == "points"
    # A point-cloud REPRESENTATION of a dense quantity reads as points.
    assert capability_report._value_kind("lidar_depth_map", "geometry.camera.depth") == "points"
    # No published semantics -> no value kind (never guessed).
    assert capability_report._value_kind("rgb", "appearance.camera.color") is None
    assert capability_report._value_kind("depth", None) is None
