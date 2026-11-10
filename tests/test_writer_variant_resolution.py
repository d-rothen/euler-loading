"""Writer resolution must not need torch when the loaders do not."""

from __future__ import annotations

from euler_loading import Modality, _resolution
from euler_loading.loaders.cpu import nuscenes as cpu_nus

INDEX = {
    "head": {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "nuscenes_rgb", "name": "nuScenes rgb"},
        "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
        "addons": {
            "euler_loading": {"version": "1.0", "loader": "nuscenes", "function": "rgb"}
        },
    }
}


def _record_variants(monkeypatch, *, gpu_error=None):
    requested = []
    original = _resolution.resolve_writer_module

    def resolve(name, *, variant="gpu"):
        requested.append(variant)
        if variant == "gpu" and gpu_error is not None:
            raise gpu_error
        return original(name, variant=variant)

    monkeypatch.setattr(_resolution, "resolve_writer_module", resolve)
    return requested


def test_explicit_cpu_loader_resolves_cpu_writers(monkeypatch):
    requested = _record_variants(monkeypatch)
    writer = _resolution._resolve_writer(
        modality_name="rgb",
        modality=Modality("/unused", loader=cpu_nus.rgb),
        index=INDEX,
    )
    assert writer is cpu_nus.write_rgb
    assert requested == ["cpu"]


def test_missing_torch_falls_back_to_cpu_writers(monkeypatch):
    requested = _record_variants(
        monkeypatch, gpu_error=ModuleNotFoundError("No module named 'torch'")
    )
    writer = _resolution._resolve_writer(
        modality_name="rgb", modality=Modality("/unused"), index=INDEX
    )
    assert writer is cpu_nus.write_rgb
    assert requested == ["gpu", "cpu"]
