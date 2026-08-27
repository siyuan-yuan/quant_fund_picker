# -*- coding: utf-8 -*-
"""严格 Point-in-Time（PiT）动态可投池读取器。

严格模式**只**读取在当时已经落盘、且由 `python -m pit build` 物化的历史快照，
绝不从今天的 ``fund_meta`` / ``rank_all`` 反推历史成分。这是避免幸存者偏差的前提：
已经清盘的基金必须仍在对应历史快照中（QA G4），A/C/E 不得重复（QA G5）。

快照约定（UTF-8 CSV，由 pit/materialize.py 生成）：
    data/pit_universe/2018-01-31.csv + _manifest.json
    data/pit_universe/2015-12-31.csv + ...

关键列：
    code / name / fund_type / status / inception_date
    as_of / known_at            —— 快照时点 / 状态断言可知日
    share_class / share_class_group —— A/C/E 去重（组级）
    lifecycle_ok / type_ok / purchase_ok —— 三类基金池过滤（分别保存）
    history_ok                  —— 模型可计算性（净值窗口），**不混入基金池过滤**
    pit_level = strict | lite   —— strict 才允许进入严格回测
    source / source_file / source_sha256 —— 来源与不可变原始件哈希

严格性与回填防线：
    - 快照级 manifest 记录 sha256 与 pit_level；
    - 默认只读 ``strict`` 快照，``lite`` 必须显式 ``allow_lite=True``；
    - 快照必须 as_of <= signal 且 known_at <= signal；
    - 文件哈希与 _manifest.json 不一致 → 拒绝（防篡改/手工改写）。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}
CODE_COLS = ("code", "基金代码")
TYPE_COLS = ("fund_type", "基金类型", "ftype")
NAME_COLS = ("name", "基金简称", "基金名称")
STATUS_COLS = ("status", "状态", "基金状态")
INCEPTION_COLS = ("inception_date", "成立日期", "成立日")
KNOWN_COLS = ("known_at", "可得日期", "发布日期")
ASOF_COLS = ("as_of", "快照日期", "snapshot_date")
SHARE_COLS = ("share_class", "份额类别")
GROUP_COLS = ("share_class_group", "主基金")
TARGET_OK_COLS = ("target_ok", "目标类型")
PIT_LEVEL_COLS = ("pit_level", "PIT级别")
HISTORY_OK_COLS = ("history_ok", "净值窗口达标")


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
    pit_level: str = ""
    sha256: str = ""
    rows: int = 0


def _load_manifest(directory: Path) -> dict:
    mp = directory / "_manifest.json"
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text("utf-8"))
    except Exception:
        return {}


class PITUniverseStore:
    """不可向未来偷看历史基金成分的快照仓库。

    allow_lite=True 时允许读取 PIT-lite 快照（回测报告必须标注降级）。
    """

    def __init__(self, directory: str | Path, target_types=TARGET_TYPES,
                 allow_lite: bool = False):
        self.directory = Path(directory)
        self.target_types = set(target_types)
        self.allow_lite = allow_lite
        self.manifest = _load_manifest(self.directory)
        if not self.directory.is_dir():
            raise PITUniverseError(
                f"严格 PiT 模式需要历史可投池快照目录：{self.directory}。"
                "请用 `python -m pit build` 生成含已清盘基金的历史快照；"
                "不要用今天的 fund_meta/rank_all 回填。")
        self.snapshots = self._discover()
        if not self.snapshots:
            raise PITUniverseError(f"{self.directory} 中未找到带日期的 CSV 快照。")

    def _discover(self):
        out = []
        manifest_rows = {m.get("file"): m for m in
                         (self.manifest.get("snapshots") or []) if m.get("file")}
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
                lvl_col = _col(head, PIT_LEVEL_COLS)
                lvl = (str(head[lvl_col].dropna().iloc[0]).strip()
                       if lvl_col and head[lvl_col].notna().any() else "")
                mrow = manifest_rows.get(path.name, {})
                # manifest（构建器）为准；无 manifest 的老快照退回首行 pit_level。
                out.append(Snapshot(path, date, known,
                                    pit_level=str(mrow.get("pit_level", "") or lvl),
                                    sha256=str(mrow.get("sha256", "")),
                                    rows=int(mrow.get("rows", 0) or 0)))
            except Exception as exc:
                raise PITUniverseError(f"无法读取历史快照 {path}: {exc}") from exc
        return sorted(out, key=lambda x: x.as_of)

    def _verify(self, snap: Snapshot) -> str:
        """校验 manifest 哈希与文件一致性，返回现行哈希。"""
        h = hashlib.sha256(snap.path.read_bytes()).hexdigest()
        if snap.sha256 and snap.sha256 != h:
            raise PITUniverseError(
                f"快照 {snap.path.name} 哈希与 _manifest.json 不一致（篡改/手工改写风险），拒绝读取。")
        if snap.pit_level == "lite" and not self.allow_lite:
            raise PITUniverseError(
                f"快照 {snap.path.name} 为 PIT-lite（类型/申购状态存在当日口径或缺失），"
                "严格模式拒绝。确要使用请 PITUniverseStore(allow_lite=True) 并在报告中标注降级。")
        return h

    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        for s in self.snapshots:
            self._verify(s)
            h.update(f"{s.path.name}|{s.as_of.date()}|{s.known_at.date()}|".encode())
            h.update(s.path.read_bytes())
        return h.hexdigest()[:16]

    def _snapshot_for(self, signal_date) -> Snapshot:
        day = pd.Timestamp(signal_date).normalize()
        usable = [s for s in self.snapshots if s.as_of <= day and s.known_at <= day]
        if not usable:
            raise PITUniverseError(f"{day.date()} 前没有已知历史可投池快照，拒绝回测。")
        return usable[-1]

    def universe(self, signal_date, require_history: bool = False,
                 require_purchase: bool = True) -> Tuple[pd.DataFrame, dict]:
        """返回 (可投基金, 审计元数据)。

        基金池过滤（lifecycle/type/purchase）已由构建期完成；
        此处仅防御性复核，并保留 history_ok 供打分层单独决定是否过滤。
        """
        day = pd.Timestamp(signal_date).normalize()
        snap = self._snapshot_for(day)
        self._verify(snap)
        df = pd.read_csv(snap.path, dtype=str)
        code_col = _col(df, CODE_COLS)
        if code_col is None:
            raise PITUniverseError(f"{snap.path} 缺少 code/基金代码 列。")
        df["code"] = df[code_col].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        df = df.dropna(subset=["code"]).drop_duplicates("code").copy()

        type_col = _col(df, TYPE_COLS)
        name_col = _col(df, NAME_COLS)
        status_col = _col(df, STATUS_COLS)
        inception_col = _col(df, INCEPTION_COLS)
        target_ok_col = _col(df, TARGET_OK_COLS)
        if type_col is None and target_ok_col is None:
            raise PITUniverseError(f"{snap.path} 缺少 fund_type/fund_type 或 target_ok，不能验证历史策略范围。")
        if type_col:
            df["fund_type"] = df[type_col].fillna("").astype(str)
        else:
            df["fund_type"] = ""
        df["name"] = df[name_col].fillna(df["code"]).astype(str) if name_col else df["code"]
        if target_ok_col:
            df["target_ok"] = df[target_ok_col].astype(str).str.lower().isin(["true", "1"])
            df = df[df["target_ok"]]
        else:
            df = df[df["fund_type"].isin(self.target_types)]

        # A/C/E 去重（构建期已做；此处对旧快照/手工快照兜底）。
        share_col = _col(df, SHARE_COLS)
        group_col = _col(df, GROUP_COLS)
        if group_col:
            df["__group"] = df[group_col].fillna(df["name"]).astype(str)
            if share_col:
                df["__cls"] = df[share_col].fillna("").astype(str).str.upper().str.strip()
            else:
                df["__cls"] = df["name"].str.strip().str[-1].where(
                    df["name"].str.strip().str[-1].isin(list("ACEFHIY")), "")
            rank = {"": 0, "A": 0, "E": 1, "I": 1, "B": 2, "H": 2, "R": 3, "C": 4, "Y": 5}
            df["__rk"] = df["__cls"].map(lambda c: rank.get(c, 9))
            df["__ci"] = df["code"].astype(int)
            df = (df.sort_values(["__rk", "__ci"]).groupby("__group", sort=False)
                    .head(1).sort_index().drop(columns=["__group", "__cls", "__rk", "__ci"]))
        else:
            df = df[~df["name"].str.strip().str.endswith(("C", "E"))]

        # 防御性：构建期已按 known_at<=t 选择快照，此处再核对状态断言时间。
        if "known_at" in df.columns:
            ka = pd.to_datetime(df["known_at"], errors="coerce")
            if (ka.notna() & (ka > day)).any():
                raise PITUniverseError(f"{snap.path} 存在 known_at 晚于信号日 {day.date()} 的行。")

        if status_col:
            bad = r"清盘|终止|注销|到期|暂停申购|停止申购"
            df = df[~df[status_col].fillna("").astype(str).str.contains(bad, regex=True)]
        if inception_col:
            inc = pd.to_datetime(df[inception_col], errors="coerce")
            df = df[inc.isna() | (inc <= day)]
        if require_purchase and "purchase_ok" in df.columns:
            df = df[df["purchase_ok"].astype(str).str.lower().isin(["true", "1"])]
        hist = None
        if "history_ok" in df.columns:
            df["history_ok"] = df["history_ok"].astype(str).str.lower().isin(["true", "1"])
            hist = int(df["history_ok"].sum())
            if require_history:
                df = df[df["history_ok"]]
        df["snapshot_as_of"] = str(snap.as_of.date())
        df["snapshot_known_at"] = str(snap.known_at.date())
        audit = dict(signal_date=str(day.date()), snapshot_file=snap.path.name,
                     snapshot_as_of=str(snap.as_of.date()), known_at=str(snap.known_at.date()),
                     pit_level=snap.pit_level, members=int(len(df)),
                     history_ok_true=hist if hist is not None else "",
                     manifest=self.manifest_hash())
        cols = ["code", "name", "fund_type", "snapshot_as_of", "snapshot_known_at"]
        for extra in ("share_class", "share_class_group", "purchase_status", "pit_level",
                      "history_ok", "status", "inception_date", "source"):
            if extra in df.columns:
                cols.append(extra)
        return df[cols], audit
