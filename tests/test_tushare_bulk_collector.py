from pathlib import Path
import json

import pandas as pd
import v6.tushare_bulk_collector as collector_module

from v6.tushare_bulk_collector import (
    Collector,
    atomic_write_json,
    consolidate_shards,
    date_windows,
    fetch_offset_pages,
)


def test_date_windows_are_complete_and_non_overlapping():
    windows = date_windows("20200101", "20221231", years=1)
    assert windows == [
        ("20200101", "20201231"),
        ("20210101", "20211231"),
        ("20220101", "20221231"),
    ]


def test_offset_paging_stops_on_short_page():
    calls = []

    def api(**kwargs):
        calls.append(kwargs)
        offset = kwargs["offset"]
        return pd.DataFrame({"id": range(offset, min(offset + 2, 5))})

    out = fetch_offset_pages(api, page_size=2)
    assert out["id"].tolist() == [0, 1, 2, 3, 4]
    assert [c["offset"] for c in calls] == [0, 2, 4]


def test_consolidate_shards_deduplicates(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame({"ts_code": ["A", "B"], "date": ["1", "1"]}).to_csv(
        shards / "a.csv", index=False
    )
    pd.DataFrame({"ts_code": ["B", "C"], "date": ["1", "2"]}).to_csv(
        shards / "b.csv", index=False
    )
    target = tmp_path / "all.csv"
    rows = consolidate_shards(shards, target, ["ts_code", "date"])
    assert rows == 3
    assert pd.read_csv(target).shape[0] == 3


def test_consolidate_ignores_schema_less_empty_shard(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "empty.csv").write_bytes(b"\xef\xbb\xbf\n")
    pd.DataFrame({"id": [1]}).to_csv(shards / "data.csv", index=False)
    assert consolidate_shards(shards, tmp_path / "all.csv", ["id"]) == 1


def test_atomic_json_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"done": ["a"]})
    atomic_write_json(target, {"done": ["a", "b"]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"done": ["a", "b"]}


def test_atomic_json_retries_transient_windows_replace_lock(tmp_path: Path, monkeypatch):
    target = tmp_path / "state.json"
    real_replace = collector_module.os.replace
    calls = []

    def flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) == 1:
            raise PermissionError(5, "transient lock")
        return real_replace(source, destination)

    monkeypatch.setattr(collector_module.os, "replace", flaky_replace)
    collector_module.atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert len(calls) == 2


def test_fund_dates_strategy_runs_without_dispatch_arity_error(tmp_path: Path):
    class FakePro:
        def fund_nav(self, **kwargs):
            return pd.DataFrame({"ts_code": [kwargs["ts_code"]], "nav_date": ["20200101"]})

    collector = Collector(FakePro(), tmp_path, "20200101", "20200102", sleep=0)
    collector.collect("fund_nav", ["A.OF"], [])
    assert pd.read_csv(tmp_path / "fund_nav.csv").shape[0] == 1
