# -*- coding: utf-8 -*-
"""严格 Point-in-Time（PiT）动态可投池。

严格模式**只**读取在当时已经落盘的基金名录快照，绝不从今天的
``fund_meta`` / ``rank_all`` 反推历史成分。这一点是避免幸存者偏差的前提：
已经清盘的基金必须仍在对应历史快照中。

快照约定（UTF-8 CSV）：
    data/pit_universe/2018-01-31.csv

必需列：``code``（或 ``基金代码``）。推荐同时保存 ``name``、``fund_type``、
``status``、``inception_date``、``known_at``。文件名日期或 ``as_of`` 列是快照
时点；若有 ``known_at``，其不得晚于信号日。快照可以早于信号日，但不能晚于。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}
CODE_COLS = ("code", "基金代码")
TYPE_COLS = ("fund_type", "基金类型", "ftype")
NAME_COLS = ("name", "基金简称", "基金名称")
STATUS_COLS = ("status", "状态", "基金状态")
INCEPTION_COLS = ("inception_date", "成立日期", "成立日")
KNOWN_COLS = ("known_at", "可得日期", "发布日期")
ASOF_COLS = ("as_of", "快照日期", "snapshot_date")


class PITUniverseError(RuntimeError):
    """历史成分不可验证时中止，而不是悄悄用今天的名单替代。"""


def _col(df: pd.DataFrame, names: Iterable[str]):
    return next((x for x in names if x in df.columns), None)


def _date_from_name(path: Path):
    m = re.search(r"(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)", path.stem)
    return pd.Timestamp("-".join(m.groups())).normalize() if m else None


@dataclass(frozen=True)
class Snapshot:
    path: Path
    as_of: pd.Timestamp
    known_at: pd.Timestamp


class PITUniverseStore:
    """不可向未来偷看历史基金成分的快照仓库。"""

    def __init__(self, directory: str | Path, target_types=TARGET_TYPES):
        self.directory = Path(directory)
        self.target_types = set(target_types)
        if not self.directory.is_dir():
            raise PITUniverseError(
                f"严格 PiT 模式需要历史可投池快照目录：{self.directory}。"
                "请导入含已清盘基金的历史快照；不要用今天的 fund_meta/rank_all 回填。"
            )
        self.snapshots = self._discover()
        if not self.snapshots:
            raise PITUniverseError(f"{self.directory} 中未找到带日期的 CSV 快照。")

    def _discover(self):
        out = []
        for path in sorted(self.directory.glob("*.csv")):
            try:
                head = pd.read_csv(path, nrows=2)
                date = _date_from_name(path)
                asof_col = _col(head, ASOF_COLS)
                if asof_col and head[asof_col].notna().any():
                    date = pd.Timestamp(head[asof_col].dropna().iloc[0]).normalize()
                if date is None:
                    continue
                known_col = _col(head, KNOWN_COLS)
                known = (pd.Timestamp(head[known_col].dropna().iloc[0]).normalize()
                         if known_col and head[known_col].notna().any() else date)
                out.append(Snapshot(path, date, known))
            except Exception as exc:
                raise PITUniverseError(f"无法读取历史快照 {path}: {exc}") from exc
        return sorted(out, key=lambda x: x.as_of)

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        for s in self.snapshots:
            h.update(f"{s.path.name}|{s.as_of.date()}|{s.known_at.date()}|".encode())
            h.update(s.path.read_bytes())
        return h.hexdigest()[:16]

    def _snapshot_for(self, signal_date) -> Snapshot:
        day = pd.Timestamp(signal_date).normalize()
        usable = [s for s in self.snapshots if s.as_of <= day and s.known_at <= day]
        if not usable:
            raise PITUniverseError(f"{day.date()} 前没有已知历史可投池快照，拒绝回测。")
        return usable[-1]

    def universe(self, signal_date) -> Tuple[pd.DataFrame, dict]:
        """返回当日可投基金及可审计元数据，不使用任何当前基金名录。"""
        day = pd.Timestamp(signal_date).normalize()
        snap = self._snapshot_for(day)
        df = pd.read_csv(snap.path, dtype=str)
        code_col = _col(df, CODE_COLS)
        if code_col is None:
            raise PITUniverseError(f"{snap.path} 缺少 code/基金代码 列。")
        df["code"] = df[code_col].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        df = df.dropna(subset=["code"]).drop_duplicates("code").copy()

        type_col, name_col = _col(df, TYPE_COLS), _col(df, NAME_COLS)
        status_col, inception_col = _col(df, STATUS_COLS), _col(df, INCEPTION_COLS)
        # 没有类型列时无法证明策略池类型，严格模式必须拒绝，而非用当前元数据补齐。
        if type_col is None:
            raise PITUniverseError(f"{snap.path} 缺少 fund_type/基金类型，不能验证历史策略范围。")
        df["fund_type"] = df[type_col].fillna("").astype(str)
        df["name"] = df[name_col].fillna(df["code"]).astype(str) if name_col else df["code"]
        df = df[df["fund_type"].isin(self.target_types)]
        # 同一策略的 C/E 份额不重复进入可投池。
        df = df[~df["name"].str.strip().str.endswith(("C", "E"))]
        if status_col:
            # 只有历史快照明确标为可申赎/存续时才纳入；空值由数据提供方定义，保留并审计。
            bad = r"清盘|终止|注销|到期|暂停申购|停止申购"
            df = df[~df[status_col].fillna("").astype(str).str.contains(bad, regex=True)]
        if inception_col:
            inc = pd.to_datetime(df[inception_col], errors="coerce")
            df = df[inc.isna() | (inc <= day)]
        df["snapshot_as_of"] = str(snap.as_of.date())
        df["snapshot_known_at"] = str(snap.known_at.date())
        audit = dict(signal_date=str(day.date()), snapshot_file=snap.path.name,
                     snapshot_as_of=str(snap.as_of.date()), known_at=str(snap.known_at.date()),
                     members=int(len(df)), manifest=self.manifest_hash())
        return df[["code", "name", "fund_type", "snapshot_as_of", "snapshot_known_at"]], audit
