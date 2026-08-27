# -*- coding: utf-8 -*-
"""事件化历史数据库：生命周期 / 类型变更 / 申赎状态 / 名称份额事件。

事件表约定（data/pit_events/*.csv，UTF-8，UTF-8-sig 亦可）：

    event_id, code, event_type, effective_date, known_at,
    value, value_prev, source, source_file, source_sha256, confidence, note

event_type:
    inception / liquidation / transform / merge / name_change / type_change
    suspend_all / restore_all / suspend_limit:<单日限额元> / restore_limit
    share_class_add

区分两个时间：
  - effective_date：事件**什么时候生效**（回测只能按 effective_date <= signal 应用边界）
  - known_at      ：投资者**什么时候看到公告**（状态型断言要求 known_at <= signal）

事件只用于边界与状态推断；生命周期仍以全历史主表
(found_date/delist_date/due_date) 为准，事件与主表冲突时 QA 报警。

另支持 R0 供应商形态的“逐日申赎状态表”：
    data/pit_raw/purchase_status/<as_of>/purchase_status.csv
    (code, purchase_status, source, ... , known_at)
    这类**状态断言**是严格 PIT 的最强来源（Wind/Choice 逐日导出）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from pit.common import col, code6, sha256_file, to_ts, normalize_header
from pit.schema import purchase_state

EVENT_COLUMNS = ["event_id", "code", "event_type", "effective_date", "known_at",
                 "value", "value_prev", "source", "source_file", "source_sha256",
                 "confidence", "note"]

EVENT_DIR = Path("data/pit_events")
PURCHASE_STATE_DIR = Path("data/pit_raw/purchase_status")


def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码 {path}")


def load_event_tables(event_dir: Path = EVENT_DIR) -> pd.DataFrame:
    """合并 data/pit_events/*.csv 为一个事件表（保留每行来源与哈希）。"""
    event_dir = Path(event_dir)
    if not event_dir.is_dir():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    parts = []
    for path in sorted(event_dir.glob("*.csv")):
        df = _read_csv_any(path)
        df = normalize_header(df)
        if "code" not in df.columns:
            c = col(df, ("基金代码", "基金编号", "ts_code"))
            if c:
                df["code"] = df[c]
        if "code" not in df.columns:
            raise ValueError(f"事件表 {path} 缺少 code 列。")
        df["code"] = df["code"].map(code6)
        df = df.dropna(subset=["code"])
        if "event_type" not in df.columns:
            raise ValueError(f"事件表 {path} 缺少 event_type 列。")
        for src, dst in (("effective_date", "effective_date"), ("known_at", "known_at"),
                         ("value", "value"), ("source", "source")):
            pass
        if "source" not in df.columns:
            df["source"] = path.stem
        if "source_file" not in df.columns:
            df["source_file"] = path.name
        if "source_sha256" not in df.columns:
            df["source_sha256"] = sha256_file(path)
        for cm in EVENT_COLUMNS:
            if cm not in df.columns:
                df[cm] = ""
        df["effective_date"] = df["effective_date"].map(to_ts)
        df["known_at"] = df["known_at"].map(to_ts)
        df["confidence"] = df["confidence"].fillna("").astype(str)
        parts.append(df[EVENT_COLUMNS])
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _normalize_type(event_type: str, value: str) -> tuple[str, str]:
    et = str(event_type).strip().lower()
    v = str(value or "")
    if et in ("suspend", "暂停申购", "暂停全部申购", "停止申购"):
        return "suspend_all", v
    if et in ("restore", "恢复申购", "开放申购"):
        return "restore_all", v
    m = re.search(r"(?:限额|限购)\s*[:：]?\s*([\d,]+)", et + " " + v)
    if et.startswith("suspend_limit") or "大额" in et:
        return ("suspend_limit", m.group(1) if m else v)
    return et, v


def events_for(events: pd.DataFrame, code: str) -> pd.DataFrame:
    return events[events["code"] == code].sort_values(["effective_date", "known_at"])


def purchase_status_at(events: pd.DataFrame, code: str, t: pd.Timestamp,
                       default: str = "unknown") -> tuple[str, str, str]:
    """由暂停/恢复事件推断 (mode, 依据, pit_level)。

    返回 mode: open / suspend_all / suspend_limit:<n> / unknown
    pit_level: "event"（有事件覆盖）或 "unknown"（无事件 → 不宣称严格）。
    """
    ev = events_for(events, code)
    if ev.empty or not ev["event_type"].str.startswith(("suspend", "restore")).any():
        return default, "no_events", "unknown"
    mode, basis = default, ""
    for _, r in ev.iterrows():
        et, v = _normalize_type(r["event_type"], r["value"])
        eff = r["effective_date"]
        if pd.isna(eff) or eff > t:
            continue
        if et == "suspend_all":
            mode, basis = "suspend_all", f"event:{r['event_id']}@{eff.date()}"
        elif et == "suspend_limit":
            mode, basis = f"suspend_limit:{v}", f"event:{r['event_id']}@{eff.date()}"
        elif et in ("restore_all", "restore_limit"):
            mode, basis = "open", f"event:{r['event_id']}@{eff.date()}"
    return mode, basis, ("event" if basis else "unknown")


def type_at(events: pd.DataFrame, code: str, t: pd.Timestamp,
            fallback: Optional[str] = None) -> tuple[str, str, str]:
    """类型变更事件 → (type, basis, pit_level: event/fallback/unknown)。"""
    ev = events_for(events, code)
    if ev.empty or not ev["event_type"].str.startswith("type_change"):
        return fallback or "", "no_type_event", ("fallback" if fallback else "unknown")
    cur, basis = fallback or "", ""
    for _, r in ev.iterrows():
        eff = r["effective_date"]
        if pd.isna(eff) or eff > t:
            continue
        cur, basis = str(r["value"]), f"event:{r['event_id']}@{eff.date()}"
    return cur, basis, ("event" if basis else "fallback" if fallback else "unknown")


def load_purchase_state_files(root: Path = PURCHASE_STATE_DIR) -> list[dict]:
    """扫描 R0 状态表 data/pit_raw/purchase_status/<as_of>/*.csv。"""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for csv_path in sorted(root.glob("*/purchase_status.csv")):
        as_of = csv_path.parent.name
        meta_path = csv_path.parent / "meta.json"
        meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
        df = _read_csv_any(csv_path)
        df = normalize_header(df)
        c = col(df, ("code", "基金代码", "ts_code"))
        st = col(df, ("purchase_status", "申购状态", "状态", "交易状态"))
        if c is None or st is None:
            raise ValueError(f"{csv_path} 需含 code 与 purchase_status/申购状态 列。")
        df = df[[c, st]].copy()
        df.columns = ["code", "purchase_status"]
        df["code"] = df["code"].map(code6)
        df["as_of"] = as_of
        df["known_at"] = meta.get("known_at", as_of)
        df["source"] = meta.get("source", "vendor:purchase_status")
        df["source_file"] = csv_path.name
        df["source_sha256"] = sha256_file(csv_path)
        out.append(df.dropna(subset=["code"]).drop_duplicates("code"))
    return out


def purchase_state_at(state_files: list[pd.DataFrame], code: str, t: pd.Timestamp) -> tuple[str, str, str]:
    """R0 状态表 → 取 known_at <= t 的最新状态断言。mode/pit_level:
    ("unknown", "", "unknown") 表示无可用状态。"""
    usable = [df for df in state_files
              if pd.Timestamp(df["known_at"].iloc[0]) <= t]
    if not usable:
        return "unknown", "", "unknown"
    latest = max(usable, key=lambda d: pd.Timestamp(d["known_at"].iloc[0]))
    hit = latest[latest["code"] == code]
    if hit.empty:
        return "unknown", "", "unknown"
    row = hit.iloc[0]
    mode, _ = purchase_state(str(row["purchase_status"]))
    return mode, f"state:{row['source_file']}@known_{row['known_at']}", "state"
