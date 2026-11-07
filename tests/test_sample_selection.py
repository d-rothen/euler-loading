"""Per-sample selection of hierarchical values (``selects_per_sample``)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from ds_crawler import DatasetWriter

from euler_loading import Modality, MultiModalDataset, SampleContext, selects_per_sample

FRAMES = ("/day/a_001", "/day/a_002", "/night/b_003")


def _frames(root) -> str:
    writer = DatasetWriter(root, name="frames", type="txt", separator=None)
    for full_id in FRAMES:
        name = full_id.rsplit("/", 1)[-1]
        writer.get_path(full_id, f"{name}_frame.txt").write_text(name, encoding="utf-8")
    writer.save_index()
    return str(root)


def _table(root, records: dict[str, Any]) -> str:
    writer = DatasetWriter(root, name="table", type="table", separator=None)
    writer.get_path("/table", "table.json").write_text(json.dumps(records), encoding="utf-8")
    writer.save_index()
    return str(root)


def _read_text(path: str, meta: dict[str, Any] | None = None) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _by_file_name(table: dict[str, Any], context: SampleContext) -> Any:
    name = context.files["frames"]["path"].rsplit("/", 1)[-1].split("_frame")[0]
    return table[name]


@pytest.fixture()
def dataset_factory(tmp_path):
    def make(loader, **modality_options):
        return MultiModalDataset(
            modalities={"frames": Modality(_frames(tmp_path / "frames"), loader=_read_text)},
            hierarchical_modalities={
                "table": Modality(
                    _table(tmp_path / "table", {"a_001": 1, "a_002": 2, "b_003": 3}),
                    loader=loader,
                    **modality_options,
                )
            },
        )

    return make


def test_each_sample_receives_its_record_from_one_decode(dataset_factory):
    calls = []

    @selects_per_sample(_by_file_name)
    def load_table(path, meta=None):
        calls.append(path)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ds = dataset_factory(load_table)
    values = {ds[i]["full_id"]: ds[i]["table"] for i in range(len(ds))}

    assert values == {
        "/day/a_001": {"table": 1},
        "/day/a_002": {"table": 2},
        "/night/b_003": {"table": 3},
    }
    assert len(calls) == 1  # the shared file is decoded once and cached


def test_selector_sees_the_sample_and_its_regular_files(dataset_factory):
    seen: list[SampleContext] = []

    def remember(table, context):
        seen.append(context)
        return _by_file_name(table, context)

    @selects_per_sample(remember)
    def load_table(path, meta=None):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ds = dataset_factory(load_table)
    ds[2]

    context = seen[-1]
    assert context.modality == "table"
    assert context.entry["path"] == "table.json"
    assert context.id == "b_003"
    assert context.full_id == "/night/b_003"
    assert context.hierarchy_path == ("night",)
    assert set(context.files) == {"frames"}
    assert context.files["frames"]["path"] == "night/b_003_frame.txt"


def test_selected_value_collapses_like_any_hierarchical_value(dataset_factory):
    @selects_per_sample(_by_file_name)
    def load_table(path, meta=None):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ds = dataset_factory(load_table, collapse_single=True)
    assert [ds[i]["table"] for i in range(len(ds))] == [1, 2, 3]
    assert ds[0]["meta"]["table"]["path"] == "table.json"


def test_missing_record_is_an_error(dataset_factory):
    @selects_per_sample(lambda table, context: table["unknown"])
    def load_table(path, meta=None):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ds = dataset_factory(load_table)
    with pytest.raises(KeyError, match="unknown"):
        ds[0]


def test_loaders_without_a_selector_keep_the_shared_value(dataset_factory):
    def load_table(path, meta=None):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ds = dataset_factory(load_table)
    assert ds[0]["table"] == {"table": {"a_001": 1, "a_002": 2, "b_003": 3}}
