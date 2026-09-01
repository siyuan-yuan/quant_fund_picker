# -*- coding: utf-8 -*-
"""数据源 R1：全历史公募基金主表（Tushare fund_basic / Wind / Choice 导出）。

- fetch_tushare_fund_basic(): 原样归档 API 响应（payload.json + meta.json），
  再归一化为 data/pit_raw/master/<ts>/fund_basic.csv。
- import_vendor_master(): 导入 Wind/Choice/CSMAR 等导出的任意列名主表
  （支持别名），同样归档原始文件并归一化。

诚实边界（写入 master_meta）：
  - status / fund_type 是数据提供方“当前”口径，未必保存历史每次变更；
  - purc_startdate 仅是“日常申购起始日”，不是暂停/恢复区间 → 不能据此判定
    历史上是否暂停申购；暂停状态必须来自 申赎状态明细/公告事件（见 events.py）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from pit.common import (archive_immutable, code6, sha256_bytes, to_ts, utc_now_iso,
                        normalize_header, col)
from pit.schema import (CODE_COLS, NAME_COLS, FULLNAME_COLS, TYPE_COLS, STATUS_COLS,
                        FOUND_COLS, DELIST_COLS, DUE_COLS, LIST_COLS, PURC_START_COLS,
                        MANAGEMENT_COLS, CUSTODIAN_COLS, MARKET_COLS)

MASTER_COLUMNS = [
    "code", "name", "full_name", "management", "custodian", "fund_type", "fund_type_raw",
    "invest_type", "status_code", "status", "found_date", "due_date", "list_date",
    "delist_date", "issue_date", "purc_startdate", "redm_startdate", "market",
    "source", "source_file", "source_sha256", "fetched_at", "master_as_of",
]


def load_token() -> str:
    import os
    tok = os.environ.get("TUSHARE_TOKEN", "")
    if not tok:
        cfg = Path("pit_config.json")
        if cfg.exists():
            tok = str(json.loads(cfg.read_text("utf-8")).get("tushare_token", ""))
    if not tok:
        raise FileNotFoundError(
            "未找到 Tushare token。请设置环境变量 TUSHARE_TOKEN，或在 pit_config.json 写入 "
            '{"tushare_token": "..."}。fund_basic 需要 ≥2000 积分。')
    return tok


def fetch_tushare_fund_basic(raw_dir: Path, token: Optional[str] = None,
                             fields: Optional[str] = None) -> tuple[Path, dict]:
    """抓取 Tushare 全历史公募基金主表并不可变归档。返回 (原始归档目录, meta)。"""
    token = token or load_token()
    import tushare as ts
    try:
        pro = ts.pro_api(token)
        df = pro.fund_basic(fields=fields or (
            "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,"
            "issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,"
            "min_amount,exp_return,benchmark,status,invest_type,type,trustee,"
            "purc_startdate,redm_startdate,market"))
    except Exception as exc:
        raise RuntimeError(
            f"Tushare fund_basic 调用失败：{exc}。确认 token 有效且积分 ≥2000。") from exc
    if df is None or df.empty:
        raise RuntimeError("Tushare fund_basic 返回空表。")

    ts_now = utc_now_iso()
    payload = {c: df[c].tolist() for c in df.columns}
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    rec_dir = raw_dir / "master" / ts_now
    rec = archive_immutable(rec_dir, "payload.json", payload_bytes, {
        "fetched_at": ts_now, "source": "tushare", "api": "fund_basic",
        "rows": int(len(df)), "fields": list(df.columns), "kind": "master_payload",
    })
    meta = {
        "fetched_at": ts_now, "source": "tushare", "api": "fund_basic",
        "rows": int(len(df)), "payload": rec,
        "limit_note": ("需要≥2000积分；status/fund_type为当前口径；"
                       "purc_startdate仅为首个日常申购起始日，不等于暂停区间"),
    }
    return rec_dir, meta


def normalize_master(df: pd.DataFrame, source: str, source_file: str,
                     source_sha256: str, fetched_at: str) -> pd.DataFrame:
    """任意列名主表 → 统一主表（保留原始列，仅追加归一化列）。"""
    df = normalize_header(df)
    out = pd.DataFrame(index=df.index)
    c = col(df, CODE_COLS)
    if c is None:
        raise ValueError("主表缺基金代码列（code/基金代码/ts_code）。")
    out["code"] = df[c].map(code6)
    n = col(df, NAME_COLS) or col(df, FULLNAME_COLS)
    if n:
        out["name"] = df[n].fillna("").astype(str)
    f = col(df, FULLNAME_COLS) or n
    if f:
        out["full_name"] = df[f].fillna("").astype(str)
    for src_col, dst in ((col(df, MANAGEMENT_COLS), "management"),
                         (col(df, CUSTODIAN_COLS), "custodian"),
                         (col(df, STATUS_COLS), "status"),
                         (col(df, MARKET_COLS), "market")):
        if src_col:
            out[dst] = df[src_col].fillna("").astype(str)
    t = col(df, TYPE_COLS)
    if t:
        out["fund_type"] = df[t].fillna("").astype(str)
        out["fund_type_raw"] = out["fund_type"]
    if "type" in df.columns and t != "type":
        out["invest_type"] = df["type"].fillna("").astype(str)
    if "invest_type" in df.columns and t != "invest_type":
        out["invest_type"] = df["invest_type"].fillna("").astype(str)
    if "status" in out:
        out["status_code"] = out["status"]
    for src_col, dst in ((col(df, FOUND_COLS), "found_date"),
                         (col(df, DELIST_COLS), "delist_date"),
                         (col(df, DUE_COLS), "due_date"),
                         (col(df, LIST_COLS), "list_date"),
                         (col(df, PURC_START_COLS), "purc_startdate")):
        if src_col:
            out[dst] = df[src_col].map(to_ts)
    out["source"] = source
    out["source_file"] = source_file
    out["source_sha256"] = source_sha256
    out["fetched_at"] = fetched_at
    out["master_as_of"] = fetched_at[:10]
    out = out.drop_duplicates("code").reset_index(drop=True)
    for cm in MASTER_COLUMNS:
        if cm not in out.columns:
            out[cm] = ""
    return out[MASTER_COLUMNS]


def archive_master(df: pd.DataFrame, raw_dir: Path, source: str, source_file: str,
                   source_sha256: str, fetched_at: str, note: str = "") -> tuple[Path, dict]:
    """归一化主表归档为 data/pit_raw/master/<ts>/fund_basic.csv。"""
    rec_dir = raw_dir / "master" / fetched_at
    data = df.to_csv(index=False).encode("utf-8")
    rec = archive_immutable(rec_dir, "fund_basic.csv", data, {
        "fetched_at": fetched_at, "source": source, "source_file": source_file,
        "source_sha256": source_sha256, "rows": int(len(df)), "kind": "normalized_master",
        "columns": list(df.columns), "note": note,
    })
    return rec_dir, rec


def import_vendor_master(csv_path: Path, raw_dir: Path, source: str,
                         fetched_at: Optional[str] = None) -> tuple[Path, dict]:
    """导入 Wind/Choice/CSMAR 等导出的 CSV 主表（原始件原样归档）。"""
    csv_path = Path(csv_path)
    raw = csv_path.read_bytes()
    fetched_at = fetched_at or utc_now_iso()
    digest = sha256_bytes(raw)
    archive_immutable(raw_dir / "vendor", csv_path.name, raw, {
        "fetched_at": fetched_at, "source": source, "kind": "vendor_master_raw",
    })
    df = pd.read_csv(csv_path, dtype=str)
    norm = normalize_master(df, source=source, source_file=csv_path.name,
                            source_sha256=digest, fetched_at=fetched_at)
    return archive_master(norm, raw_dir, source, csv_path.name, digest, fetched_at,
                          note=f"原始件 archive: {raw_dir}/vendor/{csv_path.name}")


def fetch_and_archive_tushare(raw_dir: Path, token: Optional[str] = None) -> tuple[Path, dict]:
    """抓原始 payload → 归一化 → 归档；返回 (master 目录, meta)。"""
    rec_dir, meta = fetch_tushare_fund_basic(raw_dir, token)
    payload = json.loads((rec_dir / "payload.json").read_text("utf-8"))
    df = pd.DataFrame(payload)
    digest = sha256_bytes((rec_dir / "payload.json").read_bytes())
    norm = normalize_master(df, source="tushare:fund_basic",
                            source_file="payload.json", source_sha256=digest,
                            fetched_at=meta["fetched_at"])
    return archive_master(norm, raw_dir, "tushare:fund_basic", "payload.json",
                          digest, meta["fetched_at"], note="原始 API 响应已归档")
