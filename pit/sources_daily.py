# -*- coding: utf-8 -*-
"""数据源 R3：从今天开始的每日真实截面（路线 C）。

每天抓取并**不可变**归档：
  - ak.fund_name_em()           全部基金基础信息（代码/简称/类型）
  - ak.fund_open_fund_daily_em() 开放式基金当日申赎状态（申购状态/赎回状态/成立日期）
  原始响应 json + 归一化 universe.csv + meta.json + sha256。

用途：
  1. 保证 2026-xx-xx 之后的快照是真正 PIT（当天当下状态，非回填）；
  2. 作为后期的交叉验证。**禁止**用它回填历史快照（QA G8 检查）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from pit.common import (archive_immutable, code6, to_ts, utc_now_iso,
                        normalize_header, col)
from pit.schema import (CODE_COLS, NAME_COLS, TYPE_COLS, FOUND_COLS, STATUS_COLS,
                        share_class_of, purchase_state)

DAILY_COLUMNS = ["code", "name", "fund_type", "inception_date", "status",
                 "share_class", "share_class_group", "as_of", "known_at",
                 "source", "source_file", "source_sha256", "market", "nav_date"]


def _fetch_open_fund_daily() -> pd.DataFrame:
    import akshare as ak
    df = ak.fund_open_fund_daily_em()
    if df is None or df.empty:
        raise RuntimeError("ak.fund_open_fund_daily_em() 返回空表。")
    return df


def _fetch_fund_names() -> pd.DataFrame:
    import akshare as ak
    df = ak.fund_name_em()
    if df is None or df.empty:
        raise RuntimeError("ak.fund_name_em() 返回空表。")
    return df


def fetch_daily(raw_dir: Path, date_str: Optional[str] = None) -> tuple[Path, dict]:
    """抓当日真实截面并归档到 data/pit_raw/daily/<date>/。"""
    date_str = date_str or pd.Timestamp(utc_now_iso()[:10]).date().isoformat()
    target = Path(raw_dir) / "daily" / date_str
    target.mkdir(parents=True, exist_ok=True)

    open_df = _fetch_open_fund_daily()
    name_df = _fetch_fund_names()
    ts_now = utc_now_iso()

    payload = {
        "fund_open_fund_daily_em": {c: open_df[c].tolist() for c in open_df.columns},
        "fund_name_em": {c: name_df[c].tolist() for c in name_df.columns},
    }
    pbytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    rec = archive_immutable(target, "raw_response.json", pbytes, {
        "fetched_at": ts_now, "date": date_str, "source": "akshare",
        "apis": ["fund_open_fund_daily_em", "fund_name_em"], "kind": "daily_raw",
        "rows_open": int(len(open_df)), "rows_name": int(len(name_df)),
    })

    # ---------- 归一化 ----------
    open_df = normalize_header(open_df)
    name_df = normalize_header(name_df)
    c = col(open_df, CODE_COLS)
    if c is None:
        raise ValueError("fund_open_fund_daily_em 缺基金代码列。")
    o = pd.DataFrame()
    o["code"] = open_df[c].map(code6)
    n = col(open_df, NAME_COLS)
    if n:
        o["name"] = open_df[n].fillna("").astype(str)
    t = col(open_df, TYPE_COLS)
    if t:
        o["fund_type"] = open_df[t].fillna("").astype(str)
    f = col(open_df, FOUND_COLS)
    if f:
        o["inception_date"] = open_df[f].map(to_ts)
    st = col(open_df, STATUS_COLS)
    if st:
        o["status"] = open_df[st].fillna("").astype(str)
    else:
        o["status"] = ""
    o["market"] = "O"
    o["source"] = "akshare:fund_open_fund_daily_em"
    o["source_file"] = "raw_response.json"
    o["source_sha256"] = rec["sha256"]
    o["nav_date"] = ""

    # 名称全集（含场内外）补充类型/成立日；缺失代码静默跳过
    n2 = col(name_df, NAME_COLS)
    t2 = col(name_df, TYPE_COLS)
    if n2 and t2:
        extra = pd.DataFrame()
        extra["code"] = name_df[col(name_df, CODE_COLS)].map(code6)
        extra["name"] = name_df[n2].fillna("").astype(str)
        extra["fund_type"] = name_df[t2].fillna("").astype(str)
        if col(name_df, FOUND_COLS):
            extra["inception_date"] = name_df[col(name_df, FOUND_COLS)].map(to_ts)
        extra["status"] = ""
        extra["market"] = "E"
        extra["source"] = "akshare:fund_name_em"
        extra["source_file"] = "raw_response.json"
        extra["source_sha256"] = rec["sha256"]
        extra["nav_date"] = ""
        o = pd.concat([o, extra], ignore_index=True)

    for cm in DAILY_COLUMNS:
        if cm not in o.columns:
            o[cm] = ""
    o = o.dropna(subset=["code"]).drop_duplicates("code")
    # 名称尾部推断份额类别（数据源若无明确列，绝不臆造）
    o["share_class"], o["share_class_group"] = zip(*[
        share_class_of(nm) for nm in o["name"].fillna("").astype(str)])
    o["as_of"] = date_str
    o["known_at"] = date_str
    o["status_mode"] = o["status"].map(lambda s: purchase_state(s)[0])
    o = o[DAILY_COLUMNS + ["status_mode"]].sort_values("code").reset_index(drop=True)

    data = o.to_csv(index=False).encode("utf-8")
    daily_rec = archive_immutable(target, "universe.csv", data, meta=None)
    meta = {"date": date_str, "fetched_at": ts_now, "rows": int(len(o)),
            "raw": rec, "normalized": daily_rec,
            "note": "当日真实截面；不得用于回填历史快照（QA G8）"}
    return target, meta


def scan_daily(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """扫描已归档每日截面，返回 {date: universe.csv DataFrame}。"""
    out = {}
    import json
    root = Path(raw_dir) / "daily"
    if not root.is_dir():
        return out
    for meta_path in sorted(root.glob("*/meta*.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("kind") != "daily_normalized":
            continue
        p = meta_path.parent / "universe.csv"
        df = pd.read_csv(p, dtype=str)
        df["as_of"] = meta_path.parent.name
        df["known_at"] = meta_path.parent.name
        out[meta_path.parent.name] = df
    return out
