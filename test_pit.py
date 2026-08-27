# -*- coding: utf-8 -*-
"""pit 管线端到端合成测试（不联网，不依赖 Tushare/AKShare）。

运行：python -m pytest test_pit.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from pit.common import month_ends, to_ts, sha256_bytes
from pit.events import load_event_tables, purchase_status_at
from pit.materialize import BuildConfig, build_snapshots
from pit.quality import run_qa
from pit.schema import (dedup_share_classes, normalize_type, purchase_state,
                        share_class_of)
from pit.sources_csrc import parse_csrc_index
from pit.sources_master import import_vendor_master
from pit_universe import PITUniverseError, PITUniverseStore


# ---------------- fixtures ----------------
MASTER_ROWS = [
    # code, name, fund_type(当前), found, delist, market, status
    ("100001", "测试股票A", "股票型", "2012-01-01", "", "O", "L"),
    ("100002", "测试股票C", "股票型", "2012-01-01", "", "O", "L"),
    ("200001", "已清盘混合", "混合型", "2011-05-01", "2014-06-30", "O", "D"),
    ("300001", "纯债基金", "债券型", "2013-01-01", "", "O", "L"),
    ("400001", "曾暂停基", "股票型", "2013-01-01", "", "O", "L"),
    ("500001", "曾限购基", "股票型", "2013-01-01", "", "O", "L"),
    ("600001", "索引外基金", "股票型", "2013-06-01", "", "O", "L"),
]


def make_master_csv(tmp: Path) -> Path:
    df = pd.DataFrame(MASTER_ROWS, columns=["code", "name", "fund_type", "found_date",
                                            "delist_date", "market", "status"])
    p = tmp / "master.csv"
    df.to_csv(p, index=False)
    return p


def make_csrc_xlsx(tmp: Path, as_of: str) -> Path:
    rows = [
        (1, "100001", "测试股票A", "股票型", "2012-01-01", "A"),
        (2, "100002", "测试股票C", "股票型", "2012-01-01", "C"),
        (3, "200001", "已清盘混合", "混合型", "2011-05-01", "A"),
        (4, "300001", "纯债基金", "债券型", "2013-01-01", "A"),
        (5, "400001", "曾暂停基", "股票型", "2013-01-01", "A"),
        (6, "500001", "曾限购基", "股票型", "2013-01-01", "A"),
    ]
    df = pd.DataFrame(rows, columns=["序号", "基金代码", "基金简称", "基金类型",
                                     "成立日期", "份额类别"])
    p = tmp / f"index_{as_of}.xlsx"
    df.to_excel(p, index=False)
    return p


def write_state_file(tmp: Path, as_of: str, rows: list[tuple[str, str]]) -> Path:
    d = tmp / "data" / "pit_raw" / "purchase_status" / as_of
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["code", "purchase_status"])
    p = d / "purchase_status.csv"
    df.to_csv(p, index=False)
    (d / "meta.json").write_text(json.dumps({"as_of": as_of, "known_at": as_of,
                                             "source": "vendor:purchase_status"}), encoding="utf-8")
    return p


def write_events(tmp: Path) -> None:
    rows = [
        ("e1", "400001", "suspend_all", "2014-03-01", "2014-02-25", "", "", "公告", 0.9),
        ("e2", "400001", "restore_all", "2014-06-01", "2014-05-28", "", "", "公告", 0.9),
        ("e3", "500001", "suspend_limit", "2014-01-10", "2014-01-08", "5000", "", "公告", 0.9),
    ]
    df = pd.DataFrame(rows, columns=["event_id", "code", "event_type", "effective_date",
                                     "known_at", "value", "value_prev", "source", "confidence"])
    d = tmp / "data" / "pit_events"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "suspension.csv", index=False)


def setup_raw(tmp: Path) -> dict:
    tmp = Path(tmp)
    master_csv = make_master_csv(tmp)
    master_dir, _ = import_vendor_master(master_csv, tmp / "data" / "pit_raw",
                                         source="vendor:test", fetched_at="2026-08-27T090000Z")
    from pit.common import archive_immutable
    # 3月30日信号时，截至3/31、4/15才发布的索引尚未可知 → 严格模式只能用更早一期
    for as_of, known in [("2013-12-31", "2014-01-15"), ("2014-03-31", "2014-04-15")]:
        xlsx = make_csrc_xlsx(tmp, as_of)
        raw = xlsx.read_bytes()
        archive_immutable(tmp / "data" / "pit_raw" / "csrc" / as_of,
                          "index.xlsx", raw, {
                              "as_of": as_of, "known_at": known,
                              "kind": "csrc_index_raw", "filename": "index.xlsx"})
    write_state_file(tmp, "2014-02-28", [
        ("100001", "开放申购"), ("100002", "开放申购"), ("200001", "开放申购"),
        ("300001", "开放申购"), ("400001", "开放申购"), ("500001", "开放申购"),
        ("600001", "开放申购")])
    write_state_file(tmp, "2014-03-31", [
        ("100001", "开放申购"), ("100002", "开放申购"), ("200001", "开放申购"),
        ("300001", "开放申购"), ("400001", "暂停申购"), ("500001", "暂停大额申购(单日5千元)"),
        ("600001", "开放申购")])
    write_state_file(tmp, "2014-06-30", [
        ("100001", "开放申购"), ("100002", "开放申购"), ("200001", "开放申购"),
        ("300001", "开放申购"), ("400001", "开放申购"), ("500001", "开放申购"),
        ("600001", "开放申购")])
    write_events(tmp)
    return {"tmp": tmp, "master": master_dir}


def cfg_common(tmp: Path, **kw) -> BuildConfig:
    defaults = dict(raw_dir=tmp / "data" / "pit_raw",
                    universe_dir=tmp / "data" / "pit_universe",
                    event_dir=tmp / "data" / "pit_events",
                    nav_dir=tmp / "cache", start="2014-01-31", end="2014-06-30",
                    slot_capital=10000)
    defaults.update(kw)
    return BuildConfig(**defaults)


# ---------------- unit ----------------
def test_to_ts_and_month_ends():
    assert to_ts("20260630") == pd.Timestamp("2026-06-30")
    assert to_ts("2026年6月30日") == pd.Timestamp("2026-06-30")
    assert to_ts(45100) == pd.Timestamp("2023-06-23")
    assert month_ends(pd.Timestamp("2014-01-15"), pd.Timestamp("2014-03-05")) == [
        pd.Timestamp("2014-01-31"), pd.Timestamp("2014-02-28"), pd.Timestamp("2014-03-31")]


def test_type_and_share():
    assert normalize_type("股票型") == ("股票型", "coarse")
    assert normalize_type("偏股混合基金") == ("混合型-偏股", "fine")
    cls, grp = share_class_of("某某成长混合C", "")
    assert cls == "C" and grp == "某某成长混合"
    df = pd.DataFrame([{"code": "000001", "name": "某某成长混合A", "share_class": "A",
                        "share_class_group": "某某成长混合"},
                       {"code": "000002", "name": "某某成长混合C", "share_class": "C",
                        "share_class_group": "某某成长混合"}])
    out = dedup_share_classes(df)
    assert out["code"].tolist() == ["000001"]


def test_purchase_state_and_events(tmp_path):
    write_events(tmp_path)
    ev = load_event_tables(tmp_path / "data" / "pit_events")
    mode, _, pit = purchase_status_at(ev, "400001", pd.Timestamp("2014-03-31"))
    assert mode == "suspend_all" and pit == "event"
    mode, _, _ = purchase_status_at(ev, "400001", pd.Timestamp("2014-06-30"))
    assert mode == "open"
    mode, _, pit = purchase_status_at(ev, "400001", pd.Timestamp("2014-01-01"))
    assert mode == "unknown" and pit == "unknown"
    assert purchase_state("暂停申购")[0] == "suspend_all"
    assert purchase_state("暂停大额申购")[0] == "suspend_limit"
    assert purchase_state("开放申购")[0] == "open"


def test_csrc_parse(tmp_path):
    p = make_csrc_xlsx(tmp_path, "2014-03-31")
    df = parse_csrc_index(p, pd.Timestamp("2014-03-31"), pd.Timestamp("2014-04-15"))
    assert len(df) == 6
    assert set(df["code"]) == {"100001", "100002", "200001", "300001", "400001", "500001"}
    assert df.set_index("code").loc["100001", "share_class"] == "A"
    assert df["known_at"].iloc[0] == "2014-04-15"
    assert df["source_sha256"].iloc[0] == sha256_bytes(p.read_bytes())


# ---------------- end-to-end ----------------
def test_build_strict(tmp_path):
    setup_raw(tmp_path)
    cfg = cfg_common(tmp_path)
    summary = build_snapshots(cfg, enforce_qa=True)
    snaps = {s["as_of"]: s for s in summary["snapshots"]}
    assert snaps["2014-03-31"]["pit_level"] == "strict"
    df = pd.read_csv(tmp_path / "data" / "pit_universe" / "2014-03-31.csv", dtype=str)
    # 100002(C) 去重；300001 债基剔除；400001 暂停剔除；500001 限购<槽位资金剔除；
    # 200001 未清盘保留；600001 不在索引且 strict → 剔除。
    assert sorted(df["code"]) == ["100001", "200001"]
    assert df["pit_level"].eq("strict").all()
    assert int(df["history_ok"].eq("False").sum()) == len(df)  # 无净值缓存，仅标记

    # 2014-06-30: 200001 清盘剔除；400001 已恢复；500001 恢复开放
    df2 = pd.read_csv(tmp_path / "data" / "pit_universe" / "2014-06-30.csv", dtype=str)
    assert sorted(df2["code"]) == ["100001", "400001", "500001"]


def test_build_lite_policy(tmp_path):
    setup_raw(tmp_path)
    cfg = cfg_common(tmp_path, type_policy="fallback", purchase_policy="unknown-keep")
    summary = build_snapshots(cfg)
    snaps = {s["as_of"]: s for s in summary["snapshots"]}
    assert snaps["2014-03-31"]["pit_level"] == "lite"
    df = pd.read_csv(tmp_path / "data" / "pit_universe" / "2014-03-31.csv", dtype=str)
    # fallback: 600001 以“当前类型=股票型”假定进入（lite）
    assert "600001" in set(df["code"])
    assert (df["pit_level"] == "lite").any()


def test_store_strict_and_lite(tmp_path):
    setup_raw(tmp_path)
    build_snapshots(cfg_common(tmp_path))
    store = PITUniverseStore(tmp_path / "data" / "pit_universe")
    df, audit = store.universe("2014-03-31")
    assert audit["pit_level"] == "strict"
    assert sorted(df["code"]) == ["100001", "200001"]
    assert "history_ok" in df.columns  # 可计算性独立返回
    # 信号日早于任何快照
    with pytest.raises(PITUniverseError):
        store.universe("2010-01-31")

    # lite 快照目录默认拒绝（快照不可变，Lite 版另建目录）
    build_snapshots(cfg_common(tmp_path, universe_dir=tmp_path / "data" / "pit_universe_lite",
                               type_policy="fallback", purchase_policy="unknown-keep"))
    with pytest.raises(PITUniverseError, match="PIT-lite"):
        PITUniverseStore(tmp_path / "data" / "pit_universe_lite").universe("2014-03-31")
    store2 = PITUniverseStore(tmp_path / "data" / "pit_universe_lite", allow_lite=True)
    df2, audit2 = store2.universe("2014-03-31")
    assert audit2["pit_level"] == "lite"
    assert "600001" in set(df2["code"])


def test_qa_gates(tmp_path):
    setup_raw(tmp_path)
    universe = tmp_path / "data" / "pit_universe"
    build_snapshots(cfg_common(tmp_path))
    _, master, _ = _latest_master_for_test(tmp_path)

    # G2: 篡改快照加入未来成立日基金
    p = universe / "2014-03-31.csv"
    df = pd.read_csv(p, dtype=str)
    bad = df.iloc[[0]].copy()
    bad["code"] = "999999"
    bad["inception_date"] = "2020-01-01"
    bad["name"] = "未来基金"
    pd.concat([df, bad]).to_csv(p, index=False)
    qa = run_qa(universe, master)
    assert any(f.startswith("G2") for f in qa["fatal"])

    # G5: 恢复被去重的 C 份额
    p = universe / "2014-06-30.csv"
    df = pd.read_csv(p, dtype=str)
    dup = df.iloc[[0]].copy()
    dup["code"] = "100002"
    dup["name"] = "测试股票C"
    dup["share_class"] = "C"
    dup["share_class_group"] = "测试股票"
    dup["pit_level"] = "lite"
    pd.concat([df, dup]).to_csv(p, index=False)
    qa = run_qa(universe, master)
    assert any(f.startswith("G5") for f in qa["fatal"])
    # 恢复原样（幂等重建由下次调用负责）


def _latest_master_for_test(tmp_path):
    from pit.materialize import _latest_master
    return _latest_master(tmp_path / "data" / "pit_raw")
