# -*- coding: utf-8 -*-
"""数据源 R2：证监会《公募基金产品索引》按期截面（真实历史快照）。

- csrc_download(manifest_csv, raw_dir): 从清单下载并不可变归档原始 xlsx，
  元数据记录 page_url / published_at(known_at) / as_of / sha256 / downloaded_at。
- parse_csrc_index(xlsx_path, meta): 解析为统一列：
  code, name, fund_type, inception_date, share_class, share_class_group,
  status, as_of, known_at, source, source_file, source_sha256

清单 data/pit_manifest/csrc_sources.csv 列：
  as_of,published_at,title,page_url,file_url
  （as_of/published_at 可为空，尽量从页面标题/页面日期解析；file_url 必填）
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from pit.common import (archive_immutable, col, normalize_header, sha256_bytes,
                        to_ts, utc_now_iso)
from pit.schema import (CODE_COLS, NAME_COLS, FULLNAME_COLS, TYPE_COLS, STATUS_COLS,
                        FOUND_COLS, SHARE_COLS, SHARE_GROUP_COLS, MANAGEMENT_COLS,
                        CUSTODIAN_COLS, share_class_of)

DEFAULT_MANIFEST = Path("data/pit_manifest/csrc_sources.csv")
TITLE_ASOF = re.compile(r"截至\s*(\d{4})\s*[年./-]?\s*(\d{1,2})\s*[月./-]?\s*(\d{1,2})")
TITLE_DATE = re.compile(r"(\d{4})\s*[年./-]\s*(\d{1,2})\s*[月./-]\s*(\d{1,2})")


def _asof_from_title(title: str) -> Optional[pd.Timestamp]:
    m = TITLE_ASOF.search(title)
    if not m:
        m = TITLE_DATE.search(title)
    if not m:
        return None
    try:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def download_from_manifest(manifest_csv: Path, raw_dir: Path,
                           only: Optional[Iterable[str]] = None) -> list[dict]:
    """逐行下载清单中的 xlsx，按 (as_of, published_at) 归档到
    data/pit_raw/csrc/<as_of>/。<文件原> + meta.json。返回记录列表。"""
    manifest_csv = Path(manifest_csv)
    raw_dir = Path(raw_dir)
    import requests
    out = []
    with open(manifest_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            file_url = (row.get("file_url") or "").strip()
            if not file_url:
                continue
            as_of = to_ts(row.get("as_of")) or _asof_from_title(row.get("title", ""))
            if as_of is None:
                raise ValueError(f"清单行无法确定 as_of（请填 as_of 列）：{row}")
            if only and str(as_of.date()) not in set(only):
                continue
            published = to_ts(row.get("published_at") or row.get("known_at") or "")
            if published is None:
                # 无页面日期绝不伪造：以抓取日期并标记 known_at_approx=True
                published = pd.Timestamp(to_ts(utc_now_iso())).normalize()
                approx = True
            else:
                approx = False
            title = (row.get("title") or Path(file_url).stem).strip()
            resp = requests.get(file_url, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0 (PIT archive)"})
            resp.raise_for_status()
            data = resp.content
            filename = title + ".xlsx"
            target = raw_dir / str(as_of.date())
            rec = archive_immutable(target, filename, data, {
                "as_of": str(as_of.date()), "known_at": str(published.date()),
                "known_at_approx": approx, "title": title,
                "page_url": (row.get("page_url") or "").strip(),
                "file_url": file_url, "downloaded_at": utc_now_iso(),
                "kind": "csrc_index_raw",
            })
            out.append({"as_of": str(as_of.date()), "known_at": str(published.date()),
                        "filename": filename, "path": str(target / filename),
                        "sha256": rec["sha256"],
                        "page_url": (row.get("page_url") or "").strip()})
        return out


def _find_header(df: pd.DataFrame) -> Optional[int]:
    """扫描前 15 行找表头：需同时含代码 & (名称) & (类型/成立)。"""
    for i in range(min(len(df), 15)):
        row = [str(x).strip() for x in df.iloc[i].tolist()]
        joined = "|".join(row)
        has_code = any(k in joined for k in ("基金代码", "代码", "ts_code"))
        has_name = any(k in joined for k in ("基金简称", "基金名称", "简称", "名称"))
        has_type = any(k in joined for k in ("基金类型", "投资类型", "产品类型", "类型"))
        if has_code and has_name and has_type:
            return i
        if has_code and has_name:
            return i
    return None


def parse_csrc_index(xlsx_path: Path, as_of: pd.Timestamp, known_at: pd.Timestamp,
                     source: str = "csrc:基金产品索引",
                     known_at_approx: bool = False) -> pd.DataFrame:
    """解析证监会产品索引 xlsx → 统一截面行。"""
    xlsx_path = Path(xlsx_path)
    try:
        df = pd.read_excel(xlsx_path, header=None, dtype=str)
    except Exception as exc:
        raise ValueError(f"无法读取证监会索引 {xlsx_path}: {exc}") from exc
    hdr = _find_header(df)
    if hdr is None:
        raise ValueError(
            f"{xlsx_path} 未找到表头（需要含基金代码 + 基金简称/名称 + 类型列）。"
            "请人工确认文件结构，或补充 manifest 的 parse_hint。")
    header = [str(x).strip() for x in df.iloc[hdr].tolist()]
    df = df.iloc[hdr + 1:].copy()
    df.columns = header
    df = normalize_header(df)
    code_c = col(df, CODE_COLS)
    name_c = col(df, NAME_COLS) or col(df, FULLNAME_COLS)
    type_c = col(df, TYPE_COLS)
    if code_c is None or name_c is None:
        raise ValueError(f"{xlsx_path} 表头缺 基金代码/基金简称 列。")
    out = pd.DataFrame()
    out["code"] = df[code_c].map(lambda x: str(x).strip().split(".")[0].zfill(6)
                                 if str(x).strip().isdigit() else None)
    out["name"] = df[name_c].fillna("").astype(str)
    if type_c:
        out["fund_type"] = df[type_c].fillna("").astype(str)
    f = col(df, FOUND_COLS)
    if f:
        out["inception_date"] = df[f].map(to_ts)
    share_c = col(df, SHARE_COLS)
    grp_c = col(df, SHARE_GROUP_COLS)
    if share_c:
        out["share_class"] = df[share_c].fillna("").astype(str).str.upper().str.strip()
    if grp_c:
        out["share_class_group_src"] = df[grp_c].fillna("").astype(str)
    mgr = col(df, MANAGEMENT_COLS)
    if mgr:
        out["management"] = df[mgr].fillna("").astype(str)
    cus = col(df, CUSTODIAN_COLS)
    if cus:
        out["custodian"] = df[cus].fillna("").astype(str)
    st = col(df, STATUS_COLS)
    if st:
        out["status"] = df[st].fillna("").astype(str)
    out = out.dropna(subset=["code"]).drop_duplicates("code")
    sc, sg = zip(*[share_class_of(n, c) for n, c in zip(out["name"], out.get("share_class", ""))])
    out["share_class"] = list(sc)
    out["share_class_group"] = list(sg)
    out["as_of"] = str(as_of.date())
    out["known_at"] = str(known_at.date())
    out["known_at_approx"] = str(bool(known_at_approx)).lower()
    out["source"] = source
    out["source_file"] = xlsx_path.name
    out["source_sha256"] = sha256_bytes(xlsx_path.read_bytes())
    return out


def scan_raw_csrc(raw_dir: Path) -> list[dict]:
    """扫描 data/pit_raw/csrc/ 下已归档索引（读取各 meta.json）。构建可用索引目录。"""
    import json
    out = []
    root = Path(raw_dir) / "csrc"
    if not root.is_dir():
        return out
    for meta_path in sorted(root.glob("*/meta*.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("kind") != "csrc_index_raw":
            continue
        p = meta_path.parent / meta["filename"]
        out.append({"as_of": meta["as_of"], "known_at": meta["known_at"],
                    "known_at_approx": meta.get("known_at_approx", False),
                    "title": meta.get("title", ""), "path": p,
                    "sha256": meta["sha256"]})
    return out


def parse_archive(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """返回 {as_of: DataFrame}（已解析截面，含来源/哈希/known_at 列）。"""
    res: Dict[str, pd.DataFrame] = {}
    for info in scan_raw_csrc(raw_dir):
        as_of = pd.Timestamp(info["as_of"]).normalize()
        df = parse_csrc_index(Path(info["path"]), as_of, pd.Timestamp(info["known_at"]),
                              known_at_approx=info["known_at_approx"])
        res[str(as_of.date())] = df
    return res
