"""Tests for the contract ``skyclass`` sky definition and ``Modality.function``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from euler_loading.dry_run import dry_run
from euler_loading.loaders import _sky
from euler_loading.loaders.cpu import generic as cpu_generic
from euler_loading.loaders.cpu import generic_dense_depth as cpu_gdd
from euler_loading.loaders.cpu import muses as cpu_muses
from euler_loading.loaders.cpu import real_drive_sim as cpu_rds
from euler_loading.loaders.cpu import synscapes as cpu_synscapes
from euler_loading.loaders.cpu import vkitti2 as cpu_vkitti2

torch = pytest.importorskip("torch")

from euler_loading.loaders.gpu import generic as gpu_generic  # noqa: E402
from euler_loading.loaders.gpu import real_drive_sim as gpu_rds  # noqa: E402
from euler_loading.loaders.gpu import vkitti2 as gpu_vkitti2  # noqa: E402

# Column 0 is sky (29), column 1 a class sharing nothing with it, column 2 the
# class that a relabelled head declares as sky (31).
CLASS_IDS = np.array([[29, 7, 31], [29, 7, 31]], dtype=np.uint8)


def _rds_png(path: Path, class_ids: np.ndarray = CLASS_IDS) -> Path:
    rgba = np.zeros(class_ids.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = class_ids
    rgba[..., 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


def _make_rds_segmentation(root: Path, skyclass: list[int]) -> Path:
    """A real ds-crawler Real Drive Sim class-segmentation index on disk."""
    for frame in ("00000", "00001"):
        _rds_png(root / "Scene01" / "Camera_0" / f"seg_{frame}.png")
    head: dict[str, Any] = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "rds", "name": "rds"},
        "modality": {
            "key": "semantic_segmentation",
            "meta": {"skyclass": skyclass, "file_types": ["png"]},
        },
        "addons": {
            "euler_loading": {
                "version": "1.0",
                "loader": "real_drive_sim",
                "function": "class_segmentation",
            }
        },
    }
    config = {
        "contract": {"kind": "ds_crawler_config", "version": "2.0"},
        "head_file": "dataset-head.json",
        "source": {"path": "."},
        "indexing": {
            "id": {
                "regex": r"^(?P<scene>[^/]+)/(?P<camera>[^/]+)/seg_(?P<frame>\d+)\.png$",
                "join_char": "+",
            },
            "hierarchy": {
                "regex": r"^(?P<scene>[^/]+)/(?P<camera>[^/]+)/",
                "separator": ":",
            },
            "files": {"extensions": [".png"]},
        },
    }
    metadata = root / ".ds_crawler"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "dataset-head.json").write_text(json.dumps(head))
    (metadata / "ds-crawler.json").write_text(json.dumps(config))
    return root


class TestModalityFunction:
    def test_function_reads_the_index_with_another_loader(self, tmp_path, caplog):
        root = _make_rds_segmentation(tmp_path / "seg", [29, 0, 0])
        with caplog.at_level(logging.INFO, logger="euler_loading"):
            ds = MultiModalDataset(
                modalities={"sky": Modality(str(root), function="sky_mask")}
            )
        mask = ds[0]["sky"]
        assert mask.dtype == torch.bool and tuple(mask.shape) == (1, 2, 3)
        np.testing.assert_array_equal(mask[0].numpy(), CLASS_IDS == 29)
        assert any(
            "loaded with real_drive_sim.sky_mask (the index declares "
            "'class_segmentation')" in message
            for message in caplog.messages
        )
        assert ds._resolved_writers["sky"] is gpu_rds.write_sky_mask

    def test_head_skyclass_is_the_sky_definition(self, tmp_path):
        root = _make_rds_segmentation(tmp_path / "seg", [31, 0, 0])
        ds = MultiModalDataset(
            modalities={"sky": Modality(str(root), function="sky_mask")}
        )
        np.testing.assert_array_equal(ds[0]["sky"][0].numpy(), CLASS_IDS == 31)

    def test_without_function_the_declared_loader_is_used(self, tmp_path, caplog):
        root = _make_rds_segmentation(tmp_path / "seg", [29, 0, 0])
        with caplog.at_level(logging.INFO, logger="euler_loading"):
            ds = MultiModalDataset(modalities={"seg": Modality(str(root))})
        np.testing.assert_array_equal(ds[0]["seg"][0].numpy(), CLASS_IDS)
        assert any(
            message.endswith("loaded with real_drive_sim.class_segmentation")
            for message in caplog.messages
        )

    def test_loader_and_function_are_exclusive(self):
        with pytest.raises(ValueError, match="both loader and function"):
            Modality("/data/seg", loader=cpu_rds.sky_mask, function="sky_mask")


class TestResolveSkyclass:
    LEGACY = {"old_id": _sky.grey(int)}

    def resolve(self, meta, attributes=None, default=(9, 9, 9)):
        return _sky.resolve_skyclass(
            meta, attributes, loader="t", default=default, legacy=self.LEGACY
        )

    def test_head_value_wins_over_default(self):
        assert self.resolve({"skyclass": [9, 9, 9]}) == (9, 9, 9)
        assert self.resolve(None) == (9, 9, 9)

    def test_file_attributes_win_over_the_head(self):
        assert self.resolve({"skyclass": [9, 9, 9]}, {"skyclass": [4, 4, 4]}) == (4, 4, 4)

    def test_skyclass_wins_over_a_deprecated_key(self, caplog):
        with caplog.at_level(logging.WARNING, logger="euler_loading"):
            with pytest.warns(DeprecationWarning, match="'old_id' in the dataset head is deprecated"):
                assert self.resolve({"skyclass": [9, 9, 9], "old_id": 5}) == (9, 9, 9)
        assert any(
            "ignoring the deprecated 'old_id' in the dataset head ([5, 5, 5], a "
            "different sky)" in message
            for message in caplog.messages
        )

    def test_a_deprecated_key_alone_is_still_used(self):
        with pytest.warns(DeprecationWarning, match="declare 'skyclass': \\[7, 7, 7\\] instead"):
            assert self.resolve({"old_id": 7}) == (7, 7, 7)

    def test_disagreement_with_the_default_is_warned(self, caplog):
        with caplog.at_level(logging.WARNING, logger="euler_loading"):
            self.resolve({"skyclass": [3, 3, 3]})
        assert any(
            "'skyclass' in the dataset head gives sky [3, 3, 3], which differs "
            "from the loader's default [9, 9, 9]" in message
            for message in caplog.messages
        )

    def test_missing_without_default_raises(self):
        with pytest.raises(ValueError, match="needs 'skyclass'"):
            self.resolve({}, default=None)

    def test_null_skyclass_is_undeclared(self):
        # The contract's default: heads that set no sky carry null.
        assert self.resolve({"skyclass": None}) == (9, 9, 9)
        assert self.resolve({"skyclass": [4, 4, 4]}, {"skyclass": None}) == (4, 4, 4)
        with pytest.warns(DeprecationWarning):
            assert self.resolve({"skyclass": None, "old_id": 7}) == (7, 7, 7)
        with pytest.raises(ValueError, match="needs 'skyclass'"):
            self.resolve({"skyclass": None}, default=None)

    def test_black_skyclass_is_taken_literally(self, caplog):
        with caplog.at_level(logging.WARNING, logger="euler_loading"):
            assert self.resolve({"skyclass": [0, 0, 0]}) == (0, 0, 0)
        # One warning, naming the old contract default instead of the default.
        warned = [m for m in caplog.messages if "[0, 0, 0]" in m]
        assert len(warned) == 1
        assert "filled into heads that set no sky" in warned[0]
        assert "or null to use the default [9, 9, 9]" in warned[0]

    def test_black_skyclass_in_file_attributes_is_not_the_old_default(self, caplog):
        with caplog.at_level(logging.WARNING, logger="euler_loading"):
            _sky.resolve_skyclass(
                {}, {"skyclass": [0, 0, 0]}, loader="attr-black", default=None
            )
        assert not any("filled into heads" in m for m in caplog.messages)

    @pytest.mark.parametrize("value", [[1, 2], [1, 2, 256], [1.0, 2.0, 3.0], [True, 0, 0]])
    def test_invalid_values_raise(self, value):
        with pytest.raises(ValueError, match="three integers"):
            self.resolve({"skyclass": value})

    def test_logs_each_definition_once(self, caplog):
        with caplog.at_level(logging.INFO, logger="euler_loading"):
            for _ in range(3):
                _sky.resolve_skyclass(
                    {"skyclass": [4, 5, 6]}, loader="log-once", default=None
                )
        assert (
            sum("log-once: sky is [4, 5, 6], from 'skyclass' in the dataset head" in m for m in caplog.messages)
            == 1
        )


class TestRealDriveSimSky:
    def test_mask_follows_head_skyclass(self, tmp_path):
        path = _rds_png(tmp_path / "seg.png")
        np.testing.assert_array_equal(cpu_rds.sky_mask(str(path)), CLASS_IDS == 29)
        relabelled = cpu_rds.sky_mask(str(path), {"skyclass": [31, 0, 0]})
        np.testing.assert_array_equal(relabelled, CLASS_IDS == 31)
        gpu = gpu_rds.sky_mask(str(path), {"skyclass": [31, 0, 0]})
        np.testing.assert_array_equal(gpu[0].numpy(), CLASS_IDS == 31)

    def test_null_skyclass_uses_the_default(self, tmp_path):
        path = _rds_png(tmp_path / "seg.png")
        np.testing.assert_array_equal(
            cpu_rds.sky_mask(str(path), {"skyclass": None}), CLASS_IDS == 29
        )
        sky = tmp_path / "sky.png"
        cpu_rds.write_sky_mask(str(sky), CLASS_IDS == 29, {"skyclass": None})
        np.testing.assert_array_equal(cpu_rds.sky_mask(str(sky)), CLASS_IDS == 29)

    def test_skyclass_must_keep_the_id_in_the_red_channel(self, tmp_path):
        path = _rds_png(tmp_path / "seg.png")
        with pytest.raises(ValueError, match="must be \\[id, 0, 0\\]"):
            cpu_rds.sky_mask(str(path), {"skyclass": [0, 0, 29]})

    def test_writer_round_trips_the_head_skyclass(self, tmp_path):
        meta = {"skyclass": [31, 0, 0]}
        path = tmp_path / "sky.png"
        cpu_rds.write_sky_mask(str(path), CLASS_IDS == 31, meta)
        np.testing.assert_array_equal(cpu_rds.class_segmentation(str(path)), np.where(CLASS_IDS == 31, 31, 0))
        np.testing.assert_array_equal(cpu_rds.sky_mask(str(path), meta), CLASS_IDS == 31)


class TestVKITTI2Sky:
    def _png(self, path: Path) -> Path:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[:, 0] = (90, 200, 255)
        rgb[:, 1] = (90, 240, 0)  # vegetation shares the sky's red channel
        rgb[:, 2] = (1, 2, 3)
        Image.fromarray(rgb, mode="RGB").save(path)
        return path

    def test_mask_follows_head_skyclass(self, tmp_path):
        path = self._png(tmp_path / "class.png")
        expected = np.zeros((2, 3), dtype=bool)
        expected[:, 0] = True
        np.testing.assert_array_equal(cpu_vkitti2.sky_mask(str(path)), expected)
        relabelled = gpu_vkitti2.sky_mask(str(path), {"skyclass": [1, 2, 3]})
        np.testing.assert_array_equal(relabelled[0].numpy(), np.roll(expected, 2, axis=1))

    def test_null_skyclass_falls_back_to_the_deprecated_key(self, tmp_path):
        # What a migrated head looks like: its old key plus the contract's null.
        path = self._png(tmp_path / "class.png")
        expected = np.zeros((2, 3), dtype=bool)
        expected[:, 0] = True
        np.testing.assert_array_equal(
            cpu_vkitti2.sky_mask(str(path), {"skyclass": None}), expected
        )
        with pytest.warns(DeprecationWarning, match="'sky_color'"):
            legacy = cpu_vkitti2.sky_mask(
                str(path), {"skyclass": None, "sky_color": [1, 2, 3]}
            )
        np.testing.assert_array_equal(legacy, np.roll(expected, 2, axis=1))

    def test_writer_round_trips_the_head_skyclass(self, tmp_path):
        meta = {"skyclass": [1, 2, 3]}
        mask = np.array([[True, False, True]])
        path = tmp_path / "sky.png"
        gpu_vkitti2.write_sky_mask(str(path), torch.from_numpy(mask), meta)
        np.testing.assert_array_equal(cpu_vkitti2.sky_mask(str(path), meta), mask)


class TestSynscapesSky:
    def _png(self, path: Path) -> Path:
        Image.fromarray(np.array([[23, 7, 30]], dtype=np.uint8), mode="L").save(path)
        return path

    def test_grey_skyclass_names_the_label(self, tmp_path):
        path = str(self._png(tmp_path / "class.png"))
        np.testing.assert_array_equal(cpu_synscapes.sky_mask(path), [[True, False, False]])
        np.testing.assert_array_equal(
            cpu_synscapes.sky_mask(path, {"skyclass": [30, 30, 30]}), [[False, False, True]]
        )

    def test_skyclass_takes_precedence_over_sky_class_id(self, tmp_path):
        path = str(self._png(tmp_path / "class.png"))
        meta = {"skyclass": [30, 30, 30], "sky_class_id": 7}
        with pytest.warns(DeprecationWarning, match="sky_class_id"):
            mask = cpu_synscapes.sky_mask(path, meta)
        np.testing.assert_array_equal(mask, [[False, False, True]])

    def test_non_grey_skyclass_raises(self, tmp_path):
        path = str(self._png(tmp_path / "class.png"))
        with pytest.raises(ValueError, match="grey"):
            cpu_synscapes.sky_mask(path, {"skyclass": [0, 0, 23]})


class TestMusesSky:
    def _png(self, tmp_path) -> Path:
        path = tmp_path / "frame_gt_labelTrainIds.png"
        Image.fromarray(np.array([[10, 23, 0]], dtype=np.uint8), mode="L").save(path)
        return path

    def test_head_skyclass_selects_the_label(self, tmp_path):
        path = self._png(tmp_path)
        assert cpu_muses.sky_mask(str(path)).tolist() == [[True, False, False]]
        relabelled = cpu_muses.sky_mask(str(path), {"skyclass": [23, 23, 23]})
        assert relabelled.tolist() == [[False, True, False]]

    def test_non_grey_skyclass_on_a_label_png_raises(self, tmp_path):
        with pytest.raises(ValueError, match="grey"):
            cpu_muses.sky_mask(str(self._png(tmp_path)), {"skyclass": [0, 0, 23]})

    def test_colour_skyclass_matches_an_rgb_class_image(self, tmp_path):
        path = tmp_path / "frame_gt_labelIds_rgb.png"
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        rgb[0, 1] = (0, 0, 23)
        Image.fromarray(rgb, mode="RGB").save(path)
        assert cpu_muses.sky_mask(str(path), {"skyclass": [0, 0, 23]}).tolist() == [[False, True]]


class TestGenericDenseDepthSky:
    def test_skyclass_replaces_the_deprecated_sky_mask_key(self, tmp_path):
        path = tmp_path / "sky.png"
        cpu_gdd.write_sky_mask(str(path), np.array([[True, False]]), {"skyclass": [1, 2, 3]})
        assert cpu_gdd.sky_mask(str(path), {"skyclass": [1, 2, 3]}).tolist() == [[True, False]]
        with pytest.warns(DeprecationWarning, match="'sky_mask'"):
            legacy = cpu_gdd.sky_mask(str(path), {"sky_mask": [1, 2, 3]})
        assert legacy.tolist() == [[True, False]]
        with pytest.raises(ValueError, match="needs 'skyclass'"):
            cpu_gdd.sky_mask(str(path))

    def test_old_contract_default_is_warned_without_a_loader_default(self, tmp_path, caplog):
        # No default to differ from, so only the old-default check catches it.
        path = tmp_path / "white_sky.png"
        Image.fromarray(np.array([[[255] * 3, [0] * 3]], dtype=np.uint8)).save(path)
        with caplog.at_level(logging.WARNING, logger="euler_loading"):
            mask = cpu_gdd.sky_mask(str(path), {"skyclass": [0, 0, 0]})
        assert mask.tolist() == [[False, True]]
        assert any(
            "generic_dense_depth sky_mask: 'skyclass' in the dataset head is "
            "[0, 0, 0]" in message and message.endswith("declare the dataset's sky.")
            for message in caplog.messages
        )


class TestGenericSky:
    def test_mask_needs_the_head_skyclass(self, tmp_path):
        path = tmp_path / "frame_sem.npy"
        np.save(path, np.array([[10, 2, 255]], dtype=np.uint8))
        assert cpu_generic.sky_mask(str(path), {"skyclass": [10, 10, 10]}).tolist() == [
            [True, False, False]
        ]
        gpu = gpu_generic.sky_mask(str(path), {"skyclass": [10, 10, 10]})
        assert gpu.dtype == torch.bool and tuple(gpu.shape) == (1, 1, 3)
        with pytest.raises(ValueError, match="needs 'skyclass'"):
            cpu_generic.sky_mask(str(path))
        with pytest.raises(ValueError, match="needs 'skyclass'"):
            cpu_generic.sky_mask(str(path), {"skyclass": None})


class TestDryRunSkyCheck:
    def test_reports_the_sky_fraction(self, tmp_path):
        root = _make_rds_segmentation(tmp_path / "seg", [29, 0, 0])
        (check,) = dry_run(root, samples=1).modalities
        assert check.ok and not check.warnings
        assert check.sky == "skyclass [29, 0, 0]: sky in 2/2 files, 33.3%-33.3% of pixels"

    def test_warns_when_no_file_has_sky(self, tmp_path):
        root = _make_rds_segmentation(tmp_path / "seg", [30, 0, 0])
        (check,) = dry_run(root, samples=1).modalities
        assert any(
            "sky_mask finds no sky in any of the 2 sampled files; check the "
            "dataset head's skyclass [30, 0, 0]" in warning
            for warning in check.warnings
        )

    def test_a_skyclass_the_loader_rejects_is_an_error(self, tmp_path):
        root = _make_rds_segmentation(tmp_path / "seg", [0, 0, 29])
        report = dry_run(root, samples=1)
        assert not report.ok
        assert any("must be [id, 0, 0]" in error for error in report.modalities[0].errors)
