# -*- coding: utf-8 -*-
"""公共工具：哈希、原子写、不可变归档、日期解析、月末序列。"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def archive_immutable(directory: Path, filename: str, data: bytes,
                      meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """不可变归档：同文件名且内容不同 → 抛错（绝不覆盖历史原始件）。

    返回 {filename, sha256, size, archived: bool}；meta 单独写为 meta.json，
    同样不覆盖已有 meta（内容不同则抛错）。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    target = directory / filename
    if target.exists():
        old = sha256_file(target)
        if old != digest:
            raise FileExistsError(
                f"不可变归档冲突：{target} 已存在且与本次内容不同（旧 {old[:12]} / 新 {digest[:12]}）。"
                "禁止覆盖历史原始数据，请另存新文件名（如带日期/来源后缀）。")
        archived = False
    else:
        atomic_write_bytes(target, data)
        archived = True
    rec = {"filename": filename, "sha256": digest, "size": len(data),
           "archived": archived, "path": str(target)}
    if meta is not None:
        meta_path = directory / "meta.json"
        if meta_path.exists():
            old_meta = json.loads(meta_path.read_text("utf-8"))
            if old_meta.get("sha256") != digest:
                # 同目录多文件（如 payload.json + universe.csv）：元数据按文件分开存，
                # 绝不覆盖已有 meta.json。
                meta_path = directory / f"meta.{target.name}.json"
                if meta_path.exists():
                    raise FileExistsError(f"元数据已存在且内容不同：{meta_path}")
        atomic_write_json(meta_path, {**meta, "sha256": digest, "file": target.name})
    rec["meta"] = str(meta_path) if meta is not None else ""
    return rec


_CODE_RE = re.compile(r"(\d{6})")


def code6(x: Any) -> Optional[str]:
    s = str(x).strip()
    m = _CODE_RE.search(s)
    return m.group(1) if m else None


def to_ts(value: Any) -> Optional[pd.Timestamp]:
    """宽容日期解析：20260630 / 2026-06-30 / 2026/6/30 / 2026年6月30日 / Excel datetime。"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize() if not pd.isna(value) else None
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isnan(value):
            return None
        v = float(value)
        # Excel 序列日期（1900 系统）
        if 20000 < v < 80000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))
        return pd.Timestamp(str(int(v))).normalize()
    s = str(value).strip()
    if not s or s.lower() in ("nan", "nat", "none", "-", "--", ""):
        return None
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", s)
    s = re.sub(r"(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", s)
    s = re.sub(r"[年月/.]", "-", s)
    s = s.split(" ")[0].strip("-")
    try:
        return pd.Timestamp(pd.to_datetime(s, errors="coerce")).normalize()
    except Exception:
        return None


def month_ends(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """闭区间内每个自然月月末（含 start/end 所在月）。"""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = calendar.monthrange(y, m)[1]
        out.append(pd.Timestamp(y, m, last))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text("utf-8"))


def col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    return next((x for x in names if x in df.columns), None)


def norm_str(s: Any) -> str:
    return "" if s is None or pd.isna(s) else str(s).strip()


def normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    """列名去空白、全角括号归一，便于别名匹配。"""
    df = df.copy()
    df.columns = [str(c).strip().replace("（", "(").replace("）", ")") for c in df.columns]
    return df
