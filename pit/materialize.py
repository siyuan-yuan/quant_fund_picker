# -*- coding: utf-8 -*-
"""物化月末不可变快照：企业主表(生命周期) × 类型截面 × 申赎状态 × 份额去重。

三种过滤严格分开，绝不混为一列：
  - lifecycle_ok：成立 ≤ t < 清盘/到期（基金池过滤）
  - type_ok      ：目标类型可证（基金池过滤）
  - purchase_ok  ：当时可申购（基金池过滤；大额限购按槽位资金判断）
  - history_ok   ：净值历史 ≥ 最低窗口（**模型可计算性**过滤，只标记不剔除）
  - target_ok    ：归一化后命中目标类型（含 coarse 口径粒度说明）

PIT 级别（逐行）：
  strict —— type 来自按期索引截面(known_at<=t) 且 申购状态来自状态表/事件(known_at<=t)
  lite   —— 任一状态断言缺失或来自“当前值假定”（lite_reason 写明缺什么）
快照级：全部行 strict 才写 strict；否则 lite。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from pit.common import (archive_immutable, atomic_write_bytes, code6,
                        month_ends, to_ts, utc_now_iso)
from pit.events import (purchase_state_at, purchase_status_at,
                        load_purchase_state_files, load_event_tables)
from pit.schema import (DEFAULT_NORMALIZED_TARGET, dedup_share_classes,
                        normalize_type, share_class_of)
from pit.sources_csrc import parse_archive
from pit.sources_daily import scan_daily
from pit.sources_master import MASTER_COLUMNS

SNAPSHOT_COLUMNS = [
    "code", "name", "fund_type", "fund_type_raw", "type_taxonomy",
    "status", "inception_date", "share_class", "share_class_group",
    "as_of", "known_at", "source", "source_file", "source_sha256",
    "lifecycle_ok", "type_ok", "purchase_ok", "target_ok",
    "purchase_status", "purchase_status_source", "purchase_status_basis",
    "history_ok", "history_rows", "history_reason",
    "pit_level", "lite_reason", "master_fetched_at", "build_time",
]


@dataclass
class BuildConfig:
    raw_dir: Path = Path("data/pit_raw")
    universe_dir: Path = Path("data/pit_universe")
    event_dir: Path = Path("data/pit_events")
    nav_dir: Path = Path("cache")
    start: Optional[str] = None          # YYYY-MM-DD（默认主表最早成立月）
    end: Optional[str] = None            # 默认今天
    target_types: set = field(default_factory=lambda: set(DEFAULT_NORMALIZED_TARGET))
    type_policy: str = "strict"          # strict | fallback
    purchase_policy: str = "strict"      # strict | unknown-keep
    markets: set = field(default_factory=lambda: {"O"})
    slot_capital: float = 0.0            # 单槽买入金额（大额限购判定）
    min_nav_rows: int = 0                # 0=不按净值长度过滤（只标记）
    require_history: bool = False        # True 时 history_ok=False 行不产出
    no_dedup: bool = False
    builder: str = f"pit-materialize {__import__('pit').__version__}"


def _latest_master(raw_dir: Path) -> tuple[Path, pd.DataFrame, dict]:
    root = Path(raw_dir) / "master"
    if not root.is_dir():
        raise FileNotFoundError(
            "未找到主表归档 data/pit_raw/master/<ts>/fund_basic.csv。"
            "先运行: python -m pit master  (Tushare) 或 --import-master <csv> (Wind/Choice)。")
    cands = sorted(root.glob("*/fund_basic.csv"), key=lambda p: p.parent.name)
    path = cands[-1]
    df = pd.read_csv(path, dtype=str)
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    return path.parent, df, meta


def _cover_type_at(csrc: Dict[str, pd.DataFrame], t: pd.Timestamp) -> tuple[Optional[pd.DataFrame], str, str]:
    """取 as_of<=t 且 known_at<=t 的最新索引截面。返回 (df|None, as_of, known_at)。"""
    usable = [(as_of, df) for as_of, df in csrc.items()
              if pd.Timestamp(as_of) <= t
              and bool(df["known_at"].iloc[0])
              and pd.Timestamp(df["known_at"].iloc[0]) <= t]
    if not usable:
        return None, "", ""
    as_of, df = max(usable, key=lambda x: pd.Timestamp(x[0]))
    return df, as_of, str(df["known_at"].iloc[0])


def _history_stats(code: str, t: pd.Timestamp, nav_dir: Path) -> tuple[bool, int, str]:
    """净值可计算性：cache/nav_<code>.csv 中 ≤t 的行数。只标记，不剔除。"""
    p = Path(nav_dir) / f"nav_{code}.csv"
    if not p.exists():
        return False, 0, "no_nav_cache"
    try:
        df = pd.read_csv(p, dtype=str)
    except Exception:
        return False, 0, "nav_unreadable"
    dcol = df.columns[0] if len(df.columns) else None
    if dcol is None:
        return False, 0, "no_date_col"
    d = pd.to_datetime(df[dcol], errors="coerce")
    n = int((d <= t).sum())
    return n > 0, n, ("ok" if n > 0 else "no_data_le_t")


def build_snapshots(cfg: BuildConfig, enforce_qa: bool = False) -> dict:
    """构建月末快照 + _manifest.json。返回汇总 {'snapshots': [...], 'qa': [...]}。"""
    from pit.quality import run_qa

    master_dir, master, master_meta = _latest_master(cfg.raw_dir)
    for cm in MASTER_COLUMNS:
        if cm not in master.columns:
            master[cm] = ""
    master["found_date"] = master["found_date"].map(to_ts)
    master["delist_date"] = master["delist_date"].map(to_ts)
    master["due_date"] = master["due_date"].map(to_ts)
    master["code"] = master["code"].map(code6)
    master = master.dropna(subset=["code"]).drop_duplicates("code")

    csrc = parse_archive(cfg.raw_dir)
    daily = scan_daily(cfg.raw_dir)
    events = load_event_tables(cfg.event_dir)
    state_files = load_purchase_state_files(cfg.raw_dir / "purchase_status")

    master_min = master["found_date"].dropna().min()
    if cfg.start:
        start = pd.Timestamp(cfg.start).normalize()
    elif pd.notna(master_min):
        start = pd.Timestamp(master_min).normalize()
    else:
        start = pd.Timestamp("2010-01-31")
    end = pd.Timestamp(cfg.end).normalize() if cfg.end else pd.Timestamp(utc_now_iso()[:10])
    dates = month_ends(start, end)
    if not dates:
        raise ValueError("空月份区间。")

    snapshots = []
    for t in dates:
        frame = _build_one(master, csrc, daily, events, state_files, t, cfg)
        if frame.empty:
            continue
        # 硬校验（构造即检）：
        inc_chk = pd.to_datetime(frame["inception_date"], errors="coerce", format="mixed")
        if (inc_chk.notna() & (inc_chk > t)).any():
            raise RuntimeError(f"{t.date()} 快照含成立日在未来的基金（G2）。")
        if (frame["known_at"].astype(str) > str(t.date())).any():
            raise RuntimeError(f"{t.date()} 快照含 known_at 晚于信号日（G3）。")
        # 份额去重
        if not cfg.no_dedup:
            frame = dedup_share_classes(frame)
        pit_level = "strict" if frame["pit_level"].eq("strict").all() else "lite"
        data = frame.to_csv(index=False).encode("utf-8")
        cfg.universe_dir.mkdir(parents=True, exist_ok=True)
        rec = archive_immutable(cfg.universe_dir, f"{t.date()}.csv", data,
                                meta=None)  # 溯源由 _manifest.json 统一记录
        snapshots.append({"file": f"{t.date()}.csv", "as_of": str(t.date()),
                          "rows": int(len(frame)), "pit_level": pit_level,
                          "sha256": rec["sha256"]})
    summary = {"snapshots": snapshots, "master": str(master_dir), "built_at": utc_now_iso(),
               "csrc_files": sorted(csrc.keys()), "daily_files": sorted(daily.keys()),
               "events_rows": int(len(events)), "purchase_state_files": len(state_files)}
    atomic_write_bytes(cfg.universe_dir / "_manifest.json",
                       json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
    qa = run_qa(cfg.universe_dir, master, summary)
    if enforce_qa and qa["fatal"]:
        raise RuntimeError(f"质量门禁未通过：{qa['fatal']}")
    summary["qa"] = qa
    return summary


def _build_one(master: pd.DataFrame, csrc: Dict[str, pd.DataFrame],
               daily: Dict[str, pd.DataFrame], events: pd.DataFrame,
               state_files: list[pd.DataFrame], t: pd.Timestamp,
               cfg: BuildConfig) -> pd.DataFrame:
    index_df, index_asof, index_known = _cover_type_at(csrc, t)
    index_rows = index_df.set_index("code") if index_df is not None else pd.DataFrame()
    daily_latest = _daily_latest(daily, t)

    recs = []
    for _, r in master.iterrows():
        code = r["code"]
        found = r["found_date"]
        delist = r["delist_date"]
        due = r["due_date"]
        if pd.isna(found) or found > t:
            continue
        if pd.notna(delist) and t >= delist:
            continue
        if pd.notna(due) and t >= due:
            continue
        market = str(r.get("market", "")).strip()
        if market and cfg.markets and market not in cfg.markets:
            continue

        # ---- 类型 / 名称 / 份额：索引截面优先；策略策略见 type_policy ----
        row: Dict[str, object] = {
            "code": code, "master_fetched_at": str(r.get("master_as_of", "")),
        }
        idx_hit = (not index_rows.empty) and code in index_rows.index
        if idx_hit:
            ir = index_rows.loc[code]
            row["name"] = str(ir["name"])
            raw_type = str(ir["fund_type"])
            norm_type, taxonomy = normalize_type(raw_type)
            row["fund_type"] = norm_type
            row["fund_type_raw"] = raw_type
            row["type_taxonomy"] = taxonomy
            row["share_class"] = str(ir.get("share_class", ""))
            row["share_class_group"] = str(ir.get("share_class_group", ""))
            row["status"] = str(ir.get("status", ""))
            inc2 = ir.get("inception_date")
            row["inception_date"] = str(inc2) if pd.notna(inc2) else str(found.date())
            row["source"] = str(ir["source"])
            row["source_file"] = str(ir["source_file"])
            row["source_sha256"] = str(ir["source_sha256"])
            row["known_at"] = index_known
            type_ok = norm_type in cfg.target_types
            row["type_ok"] = type_ok
            row["target_ok"] = type_ok
            type_pit = "index"
        else:
            if cfg.type_policy == "strict":
                continue          # 不在最近可证索引截面 → 不可证，严格模式剔除
            raw_type = str(r.get("fund_type", ""))
            norm_type, taxonomy = normalize_type(raw_type)
            row["name"] = str(r.get("name", ""))
            row["fund_type"] = norm_type
            row["fund_type_raw"] = raw_type
            row["type_taxonomy"] = taxonomy
            sc, sg = share_class_of(row["name"], "")
            row["share_class"], row["share_class_group"] = sc, sg
            row["status"] = str(r.get("status", ""))
            row["inception_date"] = str(found.date())
            row["source"] = "master-current-assumed"
            row["source_file"] = "fund_basic.csv"
            row["source_sha256"] = ""
            row["known_at"] = ""
            row["type_ok"] = norm_type in cfg.target_types
            row["target_ok"] = row["type_ok"]
            type_pit = "current-assumed"
        if not row["type_ok"]:
            continue

        # ---- 申购状态：每日真实截面(R3) > R0 状态表 > 事件 > （策略决定） ----
        mode, basis, pit = ("unknown", "", "unknown")
        if daily_latest is not None:
            drow = daily_latest[daily_latest["code"] == code]
            if not drow.empty:
                mode = str(drow.iloc[0].get("status_mode", "unknown"))
                basis = f"daily:{daily_latest['as_of'].iloc[0]}"
                pit = "state"
        if pit == "unknown" and code in _state_map(state_files):
            mode, basis, pit = purchase_state_at(state_files, code, t)
        if pit == "unknown" and not events.empty:
            mode, basis, pit = purchase_status_at(events, code, t)
        if pit == "unknown":
            if cfg.purchase_policy == "strict":
                continue
            mode, basis, pit = "unknown", "", "unknown"

        purchase_ok = True
        if mode == "suspend_all":
            purchase_ok = False
        elif mode.startswith("suspend_limit"):
            lim = mode.split(":", 1)[1] if ":" in mode else ""
            try:
                lim_n = float(lim.replace(",", "")) if lim else float("inf")
            except ValueError:
                lim_n = float("inf")
            purchase_ok = lim_n >= cfg.slot_capital
        row["purchase_status"] = mode
        row["purchase_status_source"] = pit
        row["purchase_status_basis"] = basis
        row["purchase_ok"] = purchase_ok
        if not purchase_ok:
            continue
        # 行级 known_at = 各来源“状态断言”可知日最大值（事件表有 known_at 列）
        state_known = _basis_known_at(basis, events, code, t)
        if str(row["known_at"]) < state_known:
            row["known_at"] = state_known

        # ---- 净值可计算性（独立标记） ----
        h_ok, h_rows, h_reason = _history_stats(code, t, cfg.nav_dir)
        if cfg.require_history and not h_ok:
            continue
        row["history_ok"] = h_ok
        row["history_rows"] = h_rows
        row["history_reason"] = h_reason

        # ---- PIT 级别 ----
        lite = []
        if type_pit != "index":
            lite.append(f"type:{type_pit}")
        if pit != "state" and pit != "event":
            lite.append(f"purchase:{pit}")
        shared = "share_class" in row and row.get("share_class", "")
        if not shared:
            lite.append("share_class:inferred")
        row["pit_level"] = "strict" if not lite else "lite"
        row["lite_reason"] = ";".join(lite)
        row["as_of"] = str(t.date())
        row["known_at"] = row["known_at"] or str(t.date())
        row["build_time"] = utc_now_iso()
        recs.append(row)

    if not recs:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    frame = pd.DataFrame(recs)
    for cm in SNAPSHOT_COLUMNS:
        if cm not in frame.columns:
            frame[cm] = ""
    return frame[SNAPSHOT_COLUMNS].sort_values("code").reset_index(drop=True)


def _basis_known_at(basis: str, events: pd.DataFrame, code: str,
                    t: pd.Timestamp) -> str:
    """从状态来源 basis 提取已知日（daily/state/event）；无法解析返回空串。"""
    import re as _re
    m = _re.search(r"known_(\d{4}-\d{2}-\d{2})", basis)
    if m:
        return m.group(1)
    m = _re.search(r"daily:(\d{4}-\d{2}-\d{2})", basis)
    if m:
        return m.group(1)
    m = _re.search(r"event:(\S+?)@(\d{4}-\d{2}-\d{2})", basis)
    if m:
        ev = events[(events["code"] == code) & (events["event_id"] == m.group(1))]
        if not ev.empty and pd.notna(ev["known_at"].iloc[0]):
            return str(ev["known_at"].iloc[0])[:10]
    return ""


_STATE_MAP_CACHE: Dict[int, set] = {}


def _state_map(state_files: list[pd.DataFrame]) -> set[str]:
    key = id(state_files)
    if key not in _STATE_MAP_CACHE:
        codes = set()
        for df in state_files:
            codes |= set(df["code"].dropna().astype(str))
        _STATE_MAP_CACHE[key] = codes
    return _STATE_MAP_CACHE[key]


def _daily_latest(daily: Dict[str, pd.DataFrame], t: pd.Timestamp) -> Optional[pd.DataFrame]:
    usable = [d for d in daily if pd.Timestamp(d) <= t]
    return daily[max(usable)] if usable else None
