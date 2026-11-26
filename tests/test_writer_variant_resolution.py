"""Writers come from the module of the loader that reads the modality.

There is no variant fallback: a writer is taken from the loader's own module
when that module is registered, else from the head's loader for the requested
variant; when neither works there is no writer, and the reason is logged.
"""

from __future__ import annotations

import logging

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


def _record_variants(monkeypatch, *, error=None):
    requested = []
    original = _resolution.resolve_writer_module

    def resolve(name, *, variant="gpu"):
        requested.append(variant)
        if error is not None:
            raise error
        return original(name, variant=variant)

    monkeypatch.setattr(_resolution, "resolve_writer_module", resolve)
    return requested


def test_writer_comes_from_the_explicit_loaders_own_module(monkeypatch):
    requested = _record_variants(monkeypatch)
    writer = _resolution._resolve_writer(
        modality_name="rgb",
        modality=Modality("/unused", loader=cpu_nus.rgb),
        index=INDEX,
        loader=cpu_nus.rgb,
    )
    assert writer is cpu_nus.write_rgb
    assert requested == []  # never looked up by name, so torch is never needed


def test_cpu_variant_resolution_uses_cpu_writers():
    writer = _resolution._resolve_writer(
        modality_name="rgb", modality=Modality("/unused"), index=INDEX, variant="cpu"
    )
    assert writer is cpu_nus.write_rgb


def test_custom_loader_without_an_importable_module_has_no_writer(monkeypatch, caplog):
    requested = _record_variants(
        monkeypatch, error=ModuleNotFoundError("No module named 'torch'")
    )

    def custom(path, meta=None):
        return path

    with caplog.at_level(logging.WARNING, logger="euler_loading._resolution"):
        writer = _resolution._resolve_writer(
            modality_name="rgb",
            modality=Modality("/unused", loader=custom),
            index=INDEX,
            loader=custom,
        )
    assert writer is None
    assert requested == ["gpu"]  # no second attempt with another variant
    assert "No module named 'torch'" in caplog.text
    assert "Modality.writer" in caplog.text


def test_index_without_a_head_has_no_writer():
    writer = _resolution._resolve_writer(
        modality_name="rgb",
        modality=Modality("/unused", loader=cpu_nus.rgb),
        index={"dataset": {"files": []}},
        loader=cpu_nus.rgb,
    )
    assert writer is None
