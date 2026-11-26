"""The ``euler.loaders`` registry: built-ins and installed packages, one path.

A third-party package registers a loader module with an entry point; a
dataset head names it in ``addons.euler_loading.loader`` and resolution finds
it. Built-in names cannot be taken over, and every rejected declaration is
reported with its reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import euler_loading.loaders.builtin as builtin_values
from euler_dataset_contract.extensions import ExtensionError
from euler_loading import (
    LoaderUnavailableError,
    Modality,
    MultiModalDataset,
    decoder_identity,
    loader_registry,
    resolve_loader_module,
)
from euler_loading import registry
from euler_loading._resolution import _resolve_loader, describe_loader
from euler_loading.dry_run import dry_run
from euler_loading.receipts import DECODER_VERSION, DecoderIdentity, require_decoder

from .loader_extension_fakes import (
    LAB_PACKAGE,
    MY_LAB,
    OTHER_LAB,
    FakeDist,
    FakeEntryPoint,
    forget_lab_modules,
    install,
    write_lab_package,
)

THERMAL = f"{LAB_PACKAGE}.thermal"


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Every test sees the registry of its own (possibly faked) environment."""
    registry._REGISTRY = None
    yield
    registry._REGISTRY = None
    forget_lab_modules()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """The fake lab's package, importable (but not yet registered)."""
    site = tmp_path / "site"
    write_lab_package(site)
    monkeypatch.syspath_prepend(str(site))
    forget_lab_modules()
    return site


def _npy_dataset(root: Path, *, loader: str, function: str = "depth") -> tuple[Path, dict[str, np.ndarray]]:
    """A ds-crawler dataset of half-scale depth maps whose head names *loader*."""
    root.mkdir(parents=True)
    stored = {}
    for frame in ("000", "001"):
        # ds-crawler names a file by its id regex's groups: frame-000.
        stored[f"frame-{frame}"] = np.arange(6, dtype=np.float32).reshape(2, 3) + int(frame)
        np.save(root / f"{frame}.npy", stored[f"frame-{frame}"])
    metadata = root / ".ds_crawler"
    metadata.mkdir()
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "lab_depth", "name": "Lab depth"},
        "modality": {
            "key": "depth",
            "meta": {"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 100]},
        },
        "addons": {
            "euler_loading": {"version": "1.0", "loader": loader, "function": function}
        },
    }
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": {"regex": r"^(?P<frame>\d+)\.npy$", "join_char": "+"},
            "files": {"extensions": [".npy"]},
        },
    }
    (metadata / "dataset-head.json").write_text(json.dumps(head))
    (metadata / "ds-crawler.json").write_text(json.dumps(config))
    return root, stored


def _index_naming(loader: str, function: str = "depth") -> dict:
    return {
        "head": {
            "contract": {"kind": "dataset_head", "version": "1.0"},
            "dataset": {"id": "lab", "name": "Lab"},
            "modality": {"key": "metadata", "meta": {}},
            "addons": {
                "euler_loading": {"version": "1.0", "loader": loader, "function": function}
            },
        }
    }


# -- built-ins are registered like any extension -------------------------------


def test_builtins_are_registered_through_entry_points():
    reg = loader_registry()
    assert reg.problems == ()
    declared = {
        name
        for name, value in vars(builtin_values).items()
        if isinstance(value, dict) and {"gpu", "cpu"} <= set(value)
    }
    assert set(reg.names()) == declared
    for name in reg.names():
        loader = reg.loaders[name]
        assert loader.builtin and loader.provider == "builtin"
        assert loader.distribution == "euler-loading"
        assert loader.target == f"euler_loading.loaders.builtin:{name}"
        assert dict(loader.modules) == {"gpu": builtin_values.__dict__[name]["gpu"], "cpu": builtin_values.__dict__[name]["cpu"]}


def test_every_builtin_variant_module_imports():
    for name in loader_registry().names():
        for variant in ("gpu", "cpu"):
            assert resolve_loader_module(name, variant=variant).__name__ == (
                loader_registry().loaders[name].modules[variant]
            )


def test_variants_come_from_the_declared_mapping_not_from_module_names():
    assert resolve_loader_module("vkitti2").__name__ == "euler_loading.loaders.gpu.vkitti2"
    assert resolve_loader_module("vkitti2", variant="cpu").__name__ == "euler_loading.loaders.cpu.vkitti2"
    # One module serves both variants when the mapping says so.
    assert resolve_loader_module("materialized", variant="cpu").__name__ == (
        "euler_loading.loaders.materialized"
    )


def test_unknown_loader_and_variant_messages():
    with pytest.raises(LoaderUnavailableError, match=r"Unknown loader 'nope'.*vkitti2.*euler\.loaders"):
        resolve_loader_module("nope")
    with pytest.raises(ValueError, match="Unknown loader variant 'tpu'"):
        resolve_loader_module("vkitti2", variant="tpu")


def test_without_builtin_entry_points_the_error_says_to_install(monkeypatch):
    from euler_dataset_contract import extensions

    monkeypatch.setattr(extensions, "_raw_entry_points", lambda group: [])
    with pytest.raises(LoaderUnavailableError, match="pip install -e"):
        loader_registry(refresh=True).get("vkitti2")


# -- extensions ----------------------------------------------------------------


def test_extension_module_resolves_for_both_variants(lab, monkeypatch):
    reg = install(monkeypatch, FakeEntryPoint("thermal_depth", THERMAL, MY_LAB))
    loader = reg.get("thermal_depth")
    assert not loader.builtin
    assert loader.provider == {"distribution": "my-lab", "version": "0.3.1"}
    assert resolve_loader_module("thermal_depth").__name__ == THERMAL
    assert resolve_loader_module("thermal_depth", variant="cpu").__name__ == THERMAL


def test_head_resolves_an_extension_loader_and_loads_files(lab, monkeypatch, tmp_path):
    install(monkeypatch, FakeEntryPoint("thermal_depth", THERMAL, MY_LAB))
    root, stored = _npy_dataset(tmp_path / "lab_depth", loader="thermal_depth")

    dataset = MultiModalDataset(modalities={"depth": Modality(str(root))})
    module = sys.modules[THERMAL]
    assert dataset._resolved_loaders["depth"] is module.depth
    assert dataset.get_writer("depth") is module.write_depth
    values = {dataset[i]["id"]: np.asarray(dataset[i]["depth"]) for i in range(len(dataset))}
    assert set(values) == set(stored)
    for frame, raw in stored.items():
        np.testing.assert_array_equal(values[frame], raw * 2.0)

    report = dry_run(root)
    assert report.ok, report.to_dict()
    check = report.modalities[0]
    assert check.loader == f"{THERMAL}.depth"
    assert check.writer == "write_depth"


def test_dry_run_warns_about_rejected_loader_packages(lab, monkeypatch, tmp_path):
    install(
        monkeypatch,
        FakeEntryPoint("thermal_depth", THERMAL, MY_LAB),
        FakeEntryPoint("vkitti2", THERMAL, MY_LAB),
    )
    root, _ = _npy_dataset(tmp_path / "lab_depth", loader="thermal_depth")
    report = dry_run(root)
    assert report.ok
    assert any("euler.loaders 'vkitti2'" in w and "ships built in" in w for w in report.warnings)


def test_log_label_names_the_providing_package(lab, monkeypatch):
    install(monkeypatch, FakeEntryPoint("thermal_depth", THERMAL, MY_LAB))
    module = resolve_loader_module("thermal_depth")
    label = describe_loader(module.depth, modality=Modality("/x"), index=_index_naming("thermal_depth"))
    assert label == "thermal.depth from my-lab 0.3.1"


def test_mapping_form_offers_only_the_declared_variants(lab, monkeypatch):
    install(monkeypatch, FakeEntryPoint("gpu_only", f"{LAB_PACKAGE}:GPU_ONLY", MY_LAB))
    assert resolve_loader_module("gpu_only").__name__ == THERMAL
    with pytest.raises(LoaderUnavailableError, match="'gpu_only' \\(my-lab 0.3.1\\) has no cpu variant"):
        resolve_loader_module("gpu_only", variant="cpu")


def test_mapping_modules_are_imported_only_when_asked_for(lab, monkeypatch):
    reg = install(monkeypatch, FakeEntryPoint("half_broken", f"{LAB_PACKAGE}:BROKEN_GPU", MY_LAB))
    assert "half_broken" in reg.loaders and reg.problems == ()
    assert resolve_loader_module("half_broken", variant="cpu").__name__ == THERMAL
    with pytest.raises(ModuleNotFoundError, match=r"Loader 'half_broken' \(my-lab 0.3.1\): cannot import its gpu module"):
        resolve_loader_module("half_broken")


# -- built-ins win; problems are reported ---------------------------------------


def test_an_extension_cannot_shadow_a_builtin(lab, monkeypatch):
    reg = install(
        monkeypatch,
        FakeEntryPoint("vkitti2", THERMAL, MY_LAB),
        FakeEntryPoint("vkitti2", THERMAL, OTHER_LAB),
    )
    assert reg.get("vkitti2").builtin
    assert resolve_loader_module("vkitti2").__name__ == "euler_loading.loaders.gpu.vkitti2"
    shadows = reg.problems_for("vkitti2")
    assert [(p.code, p.distributions) for p in shadows] == [
        ("shadows_builtin", ("my-lab",)),
        ("shadows_builtin", ("other-lab",)),
    ]
    assert "ships built in" in shadows[0].message and "rename the entry point" in shadows[0].message


def test_a_builtin_module_registered_again_stays_builtin(lab, monkeypatch):
    from euler_loading.loaders.cpu import vkitti2

    reg = install(
        monkeypatch,
        FakeEntryPoint("aaa_alias", "euler_loading.loaders.builtin:vkitti2", MY_LAB),
    )
    assert reg.get("aaa_alias").modules == reg.get("vkitti2").modules
    assert reg.owner("euler_loading.loaders.cpu.vkitti2").name == "vkitti2"
    assert decoder_identity(vkitti2.rgb).id == "euler_loading.loaders.cpu.vkitti2.rgb"


def test_two_packages_claiming_a_name_is_a_conflict(lab, monkeypatch):
    reg = install(
        monkeypatch,
        FakeEntryPoint("thermal_depth", THERMAL, MY_LAB),
        FakeEntryPoint("thermal_depth", THERMAL, OTHER_LAB),
    )
    assert "thermal_depth" not in reg.loaders
    assert [p.code for p in reg.problems] == ["conflict"]
    with pytest.raises(LoaderUnavailableError, match="declared by an installed package but is not available.*my-lab, other-lab"):
        resolve_loader_module("thermal_depth")


@pytest.mark.parametrize(
    "target, code, message",
    [
        (f"{LAB_PACKAGE}.does_not_exist", "load_failed", "No module named"),
        (f"{LAB_PACKAGE}:FUTURE_API", "api_version", "targets euler.loaders API 2.0"),
        (f"{LAB_PACKAGE}:UNKNOWN_KEYS", "invalid", "unknown key\\(s\\) 'tpu'"),
        (f"{LAB_PACKAGE}:NO_VARIANTS", "invalid", "neither a 'gpu' nor a 'cpu' module"),
        (f"{LAB_PACKAGE}:NOT_A_MAPPING", "invalid", "names a int"),
    ],
)
def test_broken_extensions_are_reported_and_quoted(lab, monkeypatch, target, code, message):
    reg = install(monkeypatch, FakeEntryPoint("lab_loader", target, MY_LAB))
    assert "lab_loader" not in reg.loaders
    (problem,) = reg.problems
    assert problem.code == code
    # The rest of the suite keeps working.
    assert resolve_loader_module("vkitti2", variant="cpu")
    # A head naming it gets the reason, with the modality it was resolving.
    with pytest.raises(LoaderUnavailableError, match=f"Modality 'depth'.*{message}"):
        _resolve_loader(
            modality_name="depth", modality=Modality("/data/lab"), index=_index_naming("lab_loader")
        )


def test_entry_point_names_must_be_head_tokens(lab, monkeypatch):
    reg = install(monkeypatch, FakeEntryPoint("thermal-depth", THERMAL, MY_LAB))
    (problem,) = reg.problems
    assert problem.code == "invalid" and "letters, digits, or underscores" in problem.message


def test_strict_mode_raises_on_problems(lab, monkeypatch):
    monkeypatch.setenv("EULER_EXTENSIONS_STRICT", "1")
    with pytest.raises(ExtensionError, match="No module named"):
        install(monkeypatch, FakeEntryPoint("lab_loader", f"{LAB_PACKAGE}.nope", MY_LAB))


# -- receipts ------------------------------------------------------------------


def test_receipt_identity_of_builtin_and_extension_decoders(lab, monkeypatch):
    from euler_loading.loaders.cpu import vkitti2

    assert decoder_identity(vkitti2.rgb) == DecoderIdentity(
        "euler_loading.loaders.cpu.vkitti2.rgb", DECODER_VERSION
    )
    install(monkeypatch, FakeEntryPoint("thermal_depth", THERMAL, MY_LAB))
    module = resolve_loader_module("thermal_depth")
    assert decoder_identity(module.depth) == DecoderIdentity(
        f"ext:my-lab/{THERMAL}.depth", "0.3.1"
    )
    # Not decoders: an unannotated function, and anything unregistered.
    assert decoder_identity(module.write_depth) is None
    assert decoder_identity(lambda path, meta=None: path) is None
    with pytest.raises(ValueError, match="unknown decoder callable effects"):
        require_decoder("depth", module.write_depth)


def test_extension_decoder_without_a_version_is_refused(lab, monkeypatch):
    install(monkeypatch, FakeEntryPoint("thermal_depth", THERMAL, None))
    module = resolve_loader_module("thermal_depth")
    assert decoder_identity(module.depth) is None
    with pytest.raises(ValueError, match="no installed version"):
        require_decoder("depth", module.depth)


# -- a real installed package, in a fresh process ---------------------------------


def test_installed_package_is_found_in_a_fresh_process(tmp_path):
    """Installed metadata (what `pip install` writes) is the whole integration."""
    site = tmp_path / "site"
    write_lab_package(site)
    dist_info = site / "my_lab-0.3.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: my-lab\nVersion: 0.3.1\n")
    (dist_info / "entry_points.txt").write_text(
        f"[euler.loaders]\nthermal_depth = {THERMAL}\n"
    )
    root, stored = _npy_dataset(tmp_path / "lab_depth", loader="thermal_depth")
    script = textwrap.dedent(
        """
        import json, sys
        import numpy as np
        from euler_loading import Modality, MultiModalDataset, decoder_identity
        from euler_loading.capability_report import loader_report

        dataset = MultiModalDataset(modalities={"depth": Modality(sys.argv[1])})
        sample = dataset[0]
        report = loader_report()
        print(json.dumps({
            "id": sample["id"],
            "depth": np.asarray(sample["depth"]).tolist(),
            "decoder": decoder_identity(dataset._resolved_loaders["depth"]).id,
            "provider": report["registered_loaders"]["thermal_depth"]["provider"],
            "problems": report["problems"],
        }))
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True, text=True, env=env, check=True,
    )
    output = json.loads(result.stdout.strip().splitlines()[-1])
    np.testing.assert_array_equal(np.asarray(output["depth"]), stored[output["id"]] * 2.0)
    assert output["decoder"] == f"ext:my-lab/{THERMAL}.depth"
    assert output["provider"] == {"distribution": "my-lab", "version": "0.3.1"}
    assert output["problems"] == []


# -- the capability report sees extensions -------------------------------------


def test_report_includes_extension_loaders_and_problems(lab, monkeypatch):
    from euler_loading.capability_report import loader_report

    install(
        monkeypatch,
        FakeEntryPoint("thermal_depth", THERMAL, MY_LAB),
        FakeEntryPoint("vkitti2", THERMAL, MY_LAB),
        FakeEntryPoint("lab_broken", f"{LAB_PACKAGE}.does_not_exist", MY_LAB),
    )
    report = loader_report()
    json.dumps(report)  # JSON-ready

    supported = {entry["name"]: entry for entry in report["supportedLoaders"]}
    assert supported["thermal_depth"] == {
        "name": "thermal_depth",
        "modalities": [
            {
                "type": "depth",
                "function": "depth",
                "hierarchical": False,
                "dtype": "float32",
                "access": "key",
                "output_unit": "meters",
                "file_formats": [".npy"],
                "cpu": {"shape": "HW"},
                "gpu": {"shape": "HW"},
            }
        ],
    }
    assert report["registered_loaders"]["thermal_depth"] == {
        "provider": {"distribution": "my-lab", "version": "0.3.1"},
        "target": THERMAL,
        "variants": {
            "gpu": {"module": THERMAL, "available": True},
            "cpu": {"module": THERMAL, "available": True},
        },
    }
    assert report["registered_loaders"]["vkitti2"]["provider"] == "builtin"
    assert "thermal_depth" in report["loaders"]["depth"]["datasets"]
    assert sorted((p["name"], p["code"]) for p in report["problems"]) == [
        ("lab_broken", "load_failed"),
        ("vkitti2", "shadows_builtin"),
    ]


def test_report_marks_unimportable_variants(lab, monkeypatch):
    from euler_loading.capability_report import loader_report

    install(
        monkeypatch,
        FakeEntryPoint("half_broken", f"{LAB_PACKAGE}:BROKEN_GPU", MY_LAB),
        # A built-in variant that cannot import (torch missing) is an install
        # choice, reported per variant but not as a problem.
        FakeEntryPoint("builtin_half", f"{LAB_PACKAGE}:BROKEN_GPU", FakeDist("euler-loading", "2.48.0")),
    )
    report = loader_report()
    for name in ("half_broken", "builtin_half"):
        variants = report["registered_loaders"][name]["variants"]
        assert variants["cpu"] == {"module": THERMAL, "available": True}
        assert variants["gpu"]["available"] is False
        assert "No module named" in variants["gpu"]["error"]
        (entry,) = [e for e in report["supportedLoaders"] if e["name"] == name]
        assert entry["modalities"][0]["cpu"] == {"shape": "HW"}
        assert "gpu" not in entry["modalities"][0]
    assert [(p["name"], p["code"]) for p in report["problems"]] == [("half_broken", "load_failed")]
