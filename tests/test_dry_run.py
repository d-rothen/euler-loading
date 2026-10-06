"""Tests for the loader dry run and its command line wrapper."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from euler_loading import cli, dry_run as dry_run_module
from euler_loading.dry_run import describe_value, discover_modalities, dry_run

SCENES = ("Scene01", "Scene02")
FRAMES = ("00000", "00001")


# ---------------------------------------------------------------------------
# Fixture builders — tiny but real ds-crawler datasets on disk
# ---------------------------------------------------------------------------

def _head(
    dataset_id: str,
    modality_key: str,
    *,
    loader: str | None = "vkitti2",
    function: str | None = "rgb",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    head: dict[str, Any] = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": dataset_id, "name": dataset_id.replace("_", " ")},
        "modality": {"key": modality_key, "meta": meta or {"range": [0, 255]}},
    }
    if loader is not None and function is not None:
        head["addons"] = {
            "euler_loading": {
                "version": "1.0",
                "loader": loader,
                "function": function,
            }
        }
    return head


def _config(stem: str, extension: str) -> dict[str, Any]:
    return {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": {
                "regex": (
                    r"^(?P<scene>[^/]+)/(?P<camera>[^/]+)/"
                    rf"{stem}_(?P<frame>\d+)\{extension}$"
                ),
                "join_char": "+",
            },
            "hierarchy": {
                "regex": r"^(?P<scene>[^/]+)/(?P<camera>[^/]+)/",
                "separator": ":",
            },
            "files": {"extensions": [extension]},
        },
    }


def _write_metadata(
    root: Path,
    head: dict[str, Any],
    config: dict[str, Any],
    *,
    scope: str | None = None,
) -> None:
    directory = root / ".ds_crawler"
    if scope is not None:
        directory = directory / scope
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset-head.json").write_text(json.dumps(head))
    (directory / "ds-crawler.json").write_text(json.dumps(config))


def _rgb_bytes(seed: int) -> bytes:
    array = np.full((4, 6, 3), seed % 256, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_rgb_modality(
    root: Path,
    *,
    dataset_id: str = "rgb_dataset",
    head: dict[str, Any] | None = None,
    content: bytes | None = None,
) -> Path:
    for index, scene in enumerate(SCENES):
        directory = root / scene / "Camera_0"
        directory.mkdir(parents=True, exist_ok=True)
        for offset, frame in enumerate(FRAMES):
            payload = content if content is not None else _rgb_bytes(index * 2 + offset)
            (directory / f"rgb_{frame}.png").write_bytes(payload)
    _write_metadata(
        root,
        head if head is not None else _head(dataset_id, "rgb"),
        _config("rgb", ".png"),
    )
    return root


def _make_rgb_zip(
    path: Path,
    *,
    dataset_id: str = "rgb_archive",
    prefix: str = "",
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for index, scene in enumerate(SCENES):
            for offset, frame in enumerate(FRAMES):
                archive.writestr(
                    f"{prefix}{scene}/Camera_0/rgb_{frame}.png",
                    _rgb_bytes(index * 2 + offset),
                )
        archive.writestr(
            f"{prefix}.ds_crawler/dataset-head.json",
            json.dumps(_head(dataset_id, "rgb")),
        )
        archive.writestr(
            f"{prefix}.ds_crawler/ds-crawler.json",
            json.dumps(_config("rgb", ".png")),
        )
    return path


def _make_intrinsics_modality(root: Path) -> Path:
    for scene in SCENES:
        directory = root / scene / "Camera_0"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "intrinsic_000.txt").write_text(
            "frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n"
            "0 0 725.0 725.0 620.5 187.0\n"
        )
    _write_metadata(
        root,
        _head("intrinsics_dataset", "camera_intrinsics", function="read_intrinsics"),
        _config("intrinsic", ".txt"),
    )
    return root


@pytest.fixture()
def rgb_root(tmp_path) -> Path:
    return _make_rgb_modality(tmp_path / "rgb")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_finds_the_root_itself(self, rgb_root):
        assert discover_modalities(rgb_root) == [(str(rgb_root), None)]

    def test_finds_modalities_below_a_plain_folder(self, tmp_path):
        _make_rgb_modality(tmp_path / "dataset" / "rgb", dataset_id="rgb")
        _make_rgb_modality(tmp_path / "dataset" / "other", dataset_id="other")

        found = discover_modalities(tmp_path / "dataset")

        assert [Path(path).name for path, _ in found] == ["other", "rgb"]

    def test_finds_zip_archives(self, tmp_path):
        (tmp_path / "dataset").mkdir()
        archive = _make_rgb_zip(tmp_path / "dataset" / "rgb.zip")

        assert discover_modalities(tmp_path / "dataset") == [(str(archive), None)]

    def test_does_not_descend_into_a_dataset_root(self, rgb_root):
        # A root's own scene directories must never be searched for datasets.
        _write_metadata(
            rgb_root / "Scene01",
            _head("nested", "rgb"),
            _config("rgb", ".png"),
        )

        assert discover_modalities(rgb_root) == [(str(rgb_root), None)]

    def test_respects_max_depth(self, tmp_path):
        _make_rgb_modality(tmp_path / "a" / "b" / "rgb")

        assert discover_modalities(tmp_path, max_depth=1) == []
        assert len(discover_modalities(tmp_path, max_depth=3)) == 1

    def test_lists_every_scope_of_a_scoped_root(self, tmp_path):
        root = _make_rgb_modality(tmp_path / "muses")
        (root / ".ds_crawler" / "dataset-head.json").unlink()
        (root / ".ds_crawler" / "ds-crawler.json").unlink()
        _write_metadata(root, _head("muses_rgb", "rgb"), _config("rgb", ".png"), scope="rgb")
        _write_metadata(root, _head("muses_alt", "rgb"), _config("rgb", ".png"), scope="alt")

        assert discover_modalities(root) == [
            (str(root), "alt"),
            (str(root), "rgb"),
        ]

    def test_explicit_scope_wins_over_discovery(self, tmp_path):
        root = _make_rgb_modality(tmp_path / "muses")
        _write_metadata(root, _head("muses_rgb", "rgb"), _config("rgb", ".png"), scope="rgb")

        assert discover_modalities(root, metadata_scope="rgb") == [(str(root), "rgb")]

    def test_ignores_scopes_without_an_artifact_set(self, tmp_path):
        root = _make_rgb_modality(tmp_path / "rgb")
        split_only = root / ".ds_crawler" / "leftover"
        split_only.mkdir()
        (split_only / "split_train.json").write_text("{}")

        assert discover_modalities(root) == [(str(root), None)]


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_resolves_and_decodes_the_declared_loader(self, rgb_root):
        report = dry_run(rgb_root)

        assert report.ok
        check = report.modalities[0]
        assert check.declared_loader == "vkitti2.rgb"
        assert check.loader == "euler_loading.loaders.gpu.vkitti2.rgb"
        assert check.writer == "write_rgb"
        assert check.dataset_id == "rgb_dataset"
        assert check.modality_key == "rgb"
        assert check.hierarchical is False
        assert check.file_count == 4
        assert [sample.ok for sample in check.samples] == [True]
        assert "torch.Size" not in (check.samples[0].value or "")
        assert "(3, 4, 6) float32" in (check.samples[0].value or "")

    def test_reads_files_out_of_a_zip_archive(self, tmp_path):
        archive = _make_rgb_zip(tmp_path / "rgb.zip")

        report = dry_run(archive, samples=2)

        assert report.ok
        assert len(report.modalities[0].samples) == 2
        assert all(sample.ok for sample in report.modalities[0].samples)

    def test_reads_an_archive_wrapped_in_a_root_directory(self, tmp_path):
        # Zipping a folder nests every entry under it; ds-crawler strips that
        # prefix when indexing, so reading files back must strip it too.
        archive = _make_rgb_zip(tmp_path / "wrapped.zip", prefix="wrapped/")

        report = dry_run(archive, samples=None)

        assert report.ok
        assert [sample.path for sample in report.modalities[0].samples] == [
            "Scene01/Camera_0/rgb_00000.png",
            "Scene01/Camera_0/rgb_00001.png",
            "Scene02/Camera_0/rgb_00000.png",
            "Scene02/Camera_0/rgb_00001.png",
        ]

    def test_samples_are_spread_across_the_index(self, rgb_root):
        report = dry_run(rgb_root, samples=2)

        paths = [sample.path for sample in report.modalities[0].samples]
        assert paths == [
            "Scene01/Camera_0/rgb_00000.png",
            "Scene02/Camera_0/rgb_00001.png",
        ]

    def test_samples_none_decodes_every_file(self, rgb_root):
        report = dry_run(rgb_root, samples=None)

        assert len(report.modalities[0].samples) == 4

    def test_samples_zero_resolves_without_decoding(self, rgb_root):
        report = dry_run(rgb_root, samples=0)

        assert report.ok
        assert report.modalities[0].loader is not None
        assert report.modalities[0].samples == []

    def test_cpu_variant_resolves_the_numpy_loader(self, rgb_root):
        report = dry_run(rgb_root, variant="cpu")

        assert report.modalities[0].loader == "euler_loading.loaders.cpu.vkitti2.rgb"
        assert "ndarray (4, 6, 3)" in (report.modalities[0].samples[0].value or "")

    def test_unknown_variant_is_rejected(self, rgb_root):
        with pytest.raises(ValueError, match="Unknown loader variant"):
            dry_run(rgb_root, variant="tpu")

    def test_a_failed_decode_is_reported_not_raised(self, tmp_path):
        root = _make_rgb_modality(tmp_path / "broken", content=b"not a png")

        report = dry_run(root)

        assert not report.ok
        sample = report.modalities[0].samples[0]
        assert not sample.ok
        assert "UnidentifiedImageError" in (sample.error or "")
        # Resolution still succeeded, so the contract is still reported.
        assert report.modalities[0].loader is not None

    def test_a_missing_addon_is_reported(self, tmp_path):
        root = _make_rgb_modality(
            tmp_path / "plain",
            head=_head("plain", "rgb", loader=None, function=None),
        )

        report = dry_run(root)

        assert not report.ok
        assert report.modalities[0].loader is None
        assert "addons.euler_loading" in report.modalities[0].errors[0]

    def test_an_unknown_loader_name_is_reported(self, tmp_path):
        root = _make_rgb_modality(
            tmp_path / "unknown",
            head=_head("unknown", "rgb", loader="nonexistent", function="rgb"),
        )

        report = dry_run(root)

        assert not report.ok
        assert "Unknown loader" in report.modalities[0].errors[0]

    def test_a_missing_path_is_reported(self, tmp_path):
        report = dry_run(tmp_path / "absent")

        assert not report.ok
        assert "does not exist" in report.errors[0]

    def test_an_unindexed_folder_is_reported(self, tmp_path):
        (tmp_path / "empty").mkdir()

        report = dry_run(tmp_path / "empty")

        assert not report.ok
        assert "No ds-crawler metadata" in report.errors[0]

    def test_a_split_selects_the_split_index(self, rgb_root):
        from ds_crawler import create_dataset_splits

        create_dataset_splits(
            str(rgb_root), split_names=["train", "val"], ratios=[50, 50], seed=1,
        )

        report = dry_run(rgb_root, split="train", samples=0)

        check = report.modalities[0]
        assert check.split == "train"
        assert check.file_count == 2
        assert check.splits == ["train", "val"]

    def test_an_inline_split_selector_is_honoured(self, rgb_root):
        from ds_crawler import create_dataset_splits

        create_dataset_splits(
            str(rgb_root), split_names=["train", "val"], ratios=[50, 50], seed=1,
        )

        report = dry_run(f"{rgb_root}:val", samples=0)

        assert report.root == str(rgb_root)
        assert report.modalities[0].split == "val"

    def test_a_hierarchical_loader_is_flagged(self, tmp_path):
        root = _make_intrinsics_modality(tmp_path / "textgt")

        report = dry_run(root)

        check = report.modalities[0]
        assert check.hierarchical is True
        assert check.ok
        assert "(3, 3) float32" in (check.samples[0].value or "")

    def test_shared_ids_are_counted_across_modalities(self, tmp_path):
        _make_rgb_modality(tmp_path / "set" / "rgb", dataset_id="rgb")
        _make_rgb_modality(tmp_path / "set" / "copy", dataset_id="copy")

        report = dry_run(tmp_path / "set", samples=0)

        assert report.ok
        assert report.common_ids == 4
        assert report.warnings == []

    def test_hierarchical_modalities_are_left_out_of_the_intersection(self, tmp_path):
        _make_rgb_modality(tmp_path / "set" / "rgb", dataset_id="rgb")
        _make_intrinsics_modality(tmp_path / "set" / "textgt")

        report = dry_run(tmp_path / "set", samples=0)

        # Only one non-hierarchical modality is left, so no intersection is
        # claimed and no warning is raised about one that cannot exist.
        assert report.common_ids is None
        assert report.warnings == []

    def test_disjoint_modalities_warn(self, tmp_path):
        _make_rgb_modality(tmp_path / "set" / "rgb", dataset_id="rgb")
        other = _make_rgb_modality(tmp_path / "set" / "other", dataset_id="other")
        for scene in SCENES:
            for frame in FRAMES:
                target = other / scene / "Camera_0" / f"rgb_{frame}.png"
                target.rename(target.with_name(f"rgb_9{frame}.png"))

        report = dry_run(tmp_path / "set", samples=0)

        assert report.common_ids == 0
        assert "cannot be combined" in report.warnings[0]

    def test_a_corrupt_archive_is_reported_not_raised(self, tmp_path):
        (tmp_path / "set").mkdir()
        (tmp_path / "set" / "corrupt.zip").write_bytes(b"PK\x03\x04 not an archive")
        _make_rgb_modality(tmp_path / "set" / "rgb", dataset_id="rgb")

        report = dry_run(tmp_path / "set")

        assert not report.ok
        failed = {check.label: check for check in report.modalities}
        assert "BadZipFile" in failed["corrupt.zip"].errors[0]
        # The healthy modality beside it is still checked.
        assert failed["rgb"].ok

    def test_malformed_metadata_is_reported_not_raised(self, tmp_path):
        root = tmp_path / "malformed"
        (root / ".ds_crawler").mkdir(parents=True)
        (root / ".ds_crawler" / "index.json").write_text("{ not json")

        report = dry_run(root)

        assert not report.ok
        assert "JSONDecodeError" in report.modalities[0].errors[0]

    def test_nothing_is_written_to_the_dataset(self, rgb_root):
        before = sorted(path.name for path in (rgb_root / ".ds_crawler").iterdir())

        dry_run(rgb_root, samples=None)

        after = sorted(path.name for path in (rgb_root / ".ds_crawler").iterdir())
        assert before == after == ["dataset-head.json", "ds-crawler.json"]


# ---------------------------------------------------------------------------
# Value descriptions
# ---------------------------------------------------------------------------

class TestDescribeValue:
    def test_describes_a_numpy_array(self):
        array = np.array([[0.0, 1.0]], dtype=np.float32)

        assert describe_value(array) == "ndarray (1, 2) float32 in [0, 1]"

    def test_describes_a_torch_tensor_without_the_torch_prefix(self):
        torch = pytest.importorskip("torch")

        assert describe_value(torch.zeros(2, 2)) == "Tensor (2, 2) float32 in [0, 0]"

    def test_flags_non_finite_values(self):
        array = np.array([1.0, np.nan, np.inf], dtype=np.float32)

        assert describe_value(array) == "ndarray (3,) float32 in [1, 1], 2 non-finite"

    def test_reports_an_all_nan_array(self):
        array = np.array([np.nan, np.nan], dtype=np.float32)

        assert "all 2 values non-finite" in describe_value(array)

    def test_describes_integer_ranges_without_scientific_notation(self):
        array = np.array([0, 123456], dtype=np.int64)

        assert describe_value(array) == "ndarray (2,) int64 in [0, 123456]"

    def test_describes_a_dict_of_arrays(self):
        value = {"K": np.zeros((3, 3), dtype=np.float32)}

        assert describe_value(value) == "{K: ndarray (3, 3) float32 in [0, 0]}"

    def test_describes_a_pil_image_without_its_address(self):
        image = Image.new("RGB", (3, 2))

        assert describe_value(image) == "Image RGB (3, 2)"

    def test_describes_none_and_scalars(self):
        assert describe_value(None) == "None"
        assert describe_value(7) == "int 7"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class TestCli:
    def test_reports_success(self, rgb_root, capsys):
        assert cli.main([str(rgb_root)]) == 0

        out = capsys.readouterr().out
        assert "[ok]" in out
        assert "vkitti2.rgb -> euler_loading.loaders.gpu.vkitti2.rgb" in out
        assert "1 modality checked: 1 ok, 0 failed" in out

    def test_exits_non_zero_on_failure(self, tmp_path, capsys):
        root = _make_rgb_modality(tmp_path / "broken", content=b"not a png")

        assert cli.main([str(root)]) == 1
        assert "[FAILED]" in capsys.readouterr().out

    def test_dry_run_subcommand_is_accepted(self, rgb_root, capsys):
        assert cli.main(["dry-run", str(rgb_root)]) == 0
        assert "[ok]" in capsys.readouterr().out

    def test_json_output_is_machine_readable(self, rgb_root, capsys):
        assert cli.main([str(rgb_root), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["summary"] == {
            "modalities": 1,
            "modalities_ok": 1,
            "files_decoded": 1,
            "files_attempted": 1,
        }
        assert payload["modalities"][0]["declared_loader"] == "vkitti2.rgb"

    def test_cpu_flag_selects_the_numpy_loaders(self, rgb_root, capsys):
        assert cli.main([str(rgb_root), "--cpu"]) == 0

        out = capsys.readouterr().out
        assert "loader variant: cpu (numpy arrays)" in out
        assert "euler_loading.loaders.cpu.vkitti2.rgb" in out

    def test_all_flag_decodes_every_file(self, rgb_root, capsys):
        assert cli.main([str(rgb_root), "--all"]) == 0
        assert "4/4 files decoded" in capsys.readouterr().out

    def test_samples_and_all_are_mutually_exclusive(self, rgb_root):
        with pytest.raises(SystemExit):
            cli.main([str(rgb_root), "--all", "-n", "2"])

    def test_negative_samples_are_rejected(self, rgb_root):
        with pytest.raises(SystemExit):
            cli.main([str(rgb_root), "-n", "-1"])

    def test_scope_flag_is_passed_through(self, tmp_path, capsys, monkeypatch):
        seen: dict[str, Any] = {}

        def fake_dry_run(path, **kwargs):
            seen.update(kwargs)
            seen["path"] = path
            return dry_run_module.DryRunReport(root=str(path), variant="gpu")

        monkeypatch.setattr(cli, "dry_run", fake_dry_run)
        cli.main([str(tmp_path), "--scope", "rgb", "--split", "train", "--max-depth", "1"])

        assert seen["metadata_scope"] == "rgb"
        assert seen["split"] == "train"
        assert seen["max_depth"] == 1

    def test_logging_setup_leaves_the_root_logger_alone(self, rgb_root):
        import logging

        root_handlers = list(logging.root.handlers)
        root_level = logging.root.level
        package = logging.getLogger("euler_loading")
        package_level = package.level

        cli.main([str(rgb_root)])
        cli.main([str(rgb_root)])

        assert logging.root.handlers == root_handlers
        assert logging.root.level == root_level
        # Repeated runs must not stack handlers, and a quiet run must not
        # lower the level other code logs through.
        assert len(package.handlers) == 1
        assert package.level == package_level

    def test_an_invalid_variant_is_reported_as_an_error(self, rgb_root, capsys, monkeypatch):
        def explode(*args, **kwargs):
            raise ValueError("Unknown loader variant 'tpu'")

        monkeypatch.setattr(cli, "dry_run", explode)

        assert cli.main([str(rgb_root)]) == 1
        assert "Unknown loader variant" in capsys.readouterr().err
