"""Fake installed packages that register ``euler.loaders`` extensions.

Fakes are added on top of the real installed entry points, so euler-loading's
own (built-in) registrations stay in place, as they would in a real
environment. Patch :func:`euler_dataset_contract.extensions._raw_entry_points`
through :func:`install`; the loader registry is rebuilt from it.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from euler_dataset_contract import extensions
from euler_loading import registry

LAB_PACKAGE = "my_lab_loaders"

#: A third-party loader module: one module for both variants.
THERMAL_MODULE = '''
import numpy as np

from euler_loading import modality_meta

api_version = "1.0"


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".npy"],
    output_unit="meters",
)
def depth(path, meta=None):
    """Stored at half scale; decodes to meters."""
    return np.load(path).astype(np.float32) * 2.0


def write_depth(path, value, meta=None):
    np.save(path, np.asarray(value, dtype=np.float32) / 2.0)
'''

#: Mapping-form values the fake package exposes from its ``__init__``.
PACKAGE_INIT = f'''
GPU_ONLY = {{"api_version": "1.0", "gpu": "{LAB_PACKAGE}.thermal"}}
BROKEN_GPU = {{"gpu": "{LAB_PACKAGE}.missing", "cpu": "{LAB_PACKAGE}.thermal"}}
FUTURE_API = {{"api_version": "2.0", "gpu": "{LAB_PACKAGE}.thermal"}}
UNKNOWN_KEYS = {{"gpu": "{LAB_PACKAGE}.thermal", "tpu": "{LAB_PACKAGE}.thermal"}}
NO_VARIANTS = {{"api_version": "1.0"}}
NOT_A_MAPPING = 42
'''


@dataclass
class FakeDist:
    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    dist: Any = None
    group: str = registry.LOADERS_GROUP


MY_LAB = FakeDist("my-lab", "0.3.1")
OTHER_LAB = FakeDist("other-lab", "1.0.0")


def write_lab_package(site: Path) -> Path:
    """Write the fake lab's importable package under *site*."""
    package = site / LAB_PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(textwrap.dedent(PACKAGE_INIT))
    (package / "thermal.py").write_text(textwrap.dedent(THERMAL_MODULE))
    return package


def forget_lab_modules() -> None:
    for name in [m for m in sys.modules if m == LAB_PACKAGE or m.startswith(LAB_PACKAGE + ".")]:
        del sys.modules[name]


def install(monkeypatch: pytest.MonkeyPatch, *entry_points: FakeEntryPoint) -> registry.LoaderRegistry:
    """Make *entry_points* look installed and rebuild the loader registry."""
    real = extensions._raw_entry_points

    def raw(group: str) -> list[Any]:
        return list(real(group)) + [ep for ep in entry_points if ep.group == group]

    monkeypatch.setattr(extensions, "_raw_entry_points", raw)
    return registry.loader_registry(refresh=True)
