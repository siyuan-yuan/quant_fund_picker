# -*- coding: utf-8 -*-
"""PIT 数据模型：列别名、类型归一、份额类别、状态判定。"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

from pit.common import code6, norm_str

# ---------------- 列别名 ----------------
CODE_COLS = ("code", "基金代码", "基金编号", "ts_code")
NAME_COLS = ("name", "基金简称", "基金名称", "简称")
FULLNAME_COLS = ("full_name", "基金全称", "全称")
TYPE_COLS = ("fund_type", "基金类型", "类型", "投资类型", "产品类型", "ftype")
STATUS_COLS = ("status", "状态", "基金状态", "存续状态")
FOUND_COLS = ("inception_date", "found_date", "成立日期", "成立日", "基金成立日")
DELIST_COLS = ("delist_date", "终止日期", "清算日期", "清盘日期", "退市日期", "注销日期")
DUE_COLS = ("due_date", "到期日期", "到期日")
LIST_COLS = ("list_date", "上市日期", "上市时间", "上市日")
PURC_START_COLS = ("purc_startdate", "申购起始日", "日常申购起始日", "开放申购日")
SHARE_COLS = ("share_class", "份额类别", "类别", "基金份额类别", "份额")
SHARE_GROUP_COLS = ("share_class_group", "主基金", "主基金代码", "母基金", "主份额")
MANAGEMENT_COLS = ("management", "管理人", "基金管理人", "基金公司", "基金公司简称")
CUSTODIAN_COLS = ("custodian", "托管人", "基金托管人")
MARKET_COLS = ("market", "市场", "交易市场")
ASOF_COLS = ("as_of", "快照日期", "snapshot_date", "截止日期", "数据截止日")
KNOWN_COLS = ("known_at", "可得日期", "发布日期", "公开日期", "公告日期")
SOURCE_COLS = ("source", "来源", "数据来源")
SOURCE_FILE_COLS = ("source_file", "来源文件", "源文件", "原始文件")
SHA_COLS = ("source_sha256", "来源哈希", "文件哈希", "sha256")
PIT_LEVEL_COLS = ("pit_level", "PIT级别", "PIT等级")

# 申购/赎回状态关键词
SUSPEND_ALL = re.compile(r"暂停(全部|大额)?(申|认)购|停止(申|认)购|暂停申赎|暂停开放|暂停", re.I)
SUSPEND_LARGE = re.compile(r"暂停大额申(认)?购|大额申购限制|单日.*?限额|限额", re.I)
RESTORE = re.compile(r"恢复(申|认)购|恢复大额申购|开放申购|正常申购", re.I)
DELIST_KW = re.compile(r"清盘|终止|注销|到期|摘牌|退市|清算", re.I)
TRANSFORM_KW = re.compile(r"转型|合并|更名|变更", re.I)

# 基金类型 → 目标类型（口径归一）。记录 source_taxonomy 便于审计。
# 注：Tushare/证监会索引的“混合型/股票型”比天天基金“混合型-偏股”粗；
#     放宽匹配会引入（可能偏平衡的）混合型，反映在 type_taxonomy="coarse"。
TYPE_NORMALIZE = {
    "股票型": ("股票型", "coarse"),
    "股票型基金": ("股票型", "coarse"),
    "积极投资股票基金": ("股票型", "coarse"),
    "混合型": ("混合型", "coarse"),
    "混合型基金": ("混合型", "coarse"),
    "偏股混合基金": ("混合型-偏股", "fine"),
    "偏股混合型": ("混合型-偏股", "fine"),
    "混合型-偏股": ("混合型-偏股", "fine"),
    "混合型-灵活": ("混合型-灵活", "fine"),
    "灵活配置混合基金": ("混合型-灵活", "fine"),
    "混合型-平衡": ("混合型-平衡", "fine"),
    "指数型-股票": ("指数型-股票", "fine"),
    "指数型-海外股票": ("指数型-海外股票", "fine"),
    "指数股票型": ("指数型-股票", "fine"),
    "被动指数型": ("指数型-股票", "fine"),
    "QDII": ("QDII", "coarse"),
    "QDII-普通股票": ("QDII-普通股票", "fine"),
    "QDII-混合偏股": ("QDII-混合偏股", "fine"),
    "债券型": ("债券型", "coarse"),
    "货币型": ("货币型", "coarse"),
    "FOF": ("FOF", "coarse"),
    "REITs": ("REITs", "coarse"),
    "商品": ("商品", "coarse"),
}

# 扫描/回测目标类型（与 scan_market.TARGET_TYPES 保持一致；含可开关的海外指数 QDII）
DEFAULT_TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}
# 默认允许的归一化目标类型（含 coarse 股票型/混合型 —— 数据源口径更粗时的选择）
DEFAULT_NORMALIZED_TARGET = DEFAULT_TARGET_TYPES | {"混合型", "指数型-海外股票"}


def normalize_type(raw: str) -> tuple[str, str]:
    """返回 (规范化类型, taxonomy)。taxonomy: fine=份额级精细 / coarse=大类。"""
    s = norm_str(raw)
    key = s
    # 去掉括号备注，如 “股票型（QDII）”
    key = re.sub(r"[（(].*?[)）]", "", key).strip()
    if key in TYPE_NORMALIZE:
        return TYPE_NORMALIZE[key]
    for k, v in TYPE_NORMALIZE.items():
        if key.startswith(k):
            if re.match(r"^(股票|混合|指数|债券|货币|QDII|FOF|REITs|商品)", key):
                return v
        elif len(key) >= 3 and k.startswith(key) and re.match(r"^(股票|混合|指数|债券|货币|QDII|FOF)", key):
            return v
    return (s, "unknown")


def is_target_type(raw: str, target: Iterable[str]) -> bool:
    norm, _ = normalize_type(raw)
    return norm in set(target) or norm in DEFAULT_TARGET_TYPES


_SHARE_SUFFIX = re.compile(r"[-\s]?([A-Z](?:\d+)?)$")
_AMOUNT_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*(万|亿|千)?\s*元")


def _limit_yuan(text: str) -> str:
    m = _AMOUNT_RE.search(norm_str(text))
    if not m:
        return ""
    v = float(m.group(1).replace(",", ""))
    unit = {"万": 1e4, "亿": 1e8, "千": 1e3}.get(m.group(2) or "", 1)
    return str(int(v * unit))


def share_class_of(name: str, share_class: Optional[str] = None) -> tuple[str, str]:
    """返回 (share_class, share_class_group)。

    share_class 优先（数据源明确列）；否则从名称尾部提取 A/B/C/E/H/I 等。
    group 用于 A/C 去重：取去掉份额后缀后的名称（已去空白尾缀）。
    """
    s = norm_str(name)
    cls = norm_str(share_class).upper()
    if not cls:
        m = _SHARE_SUFFIX.search(s)
        if m and m.group(1)[0] in "ABCDEFHIY":
            cls = m.group(1)
    mv = _SHARE_SUFFIX.sub("", s).rstrip("- ").strip()
    return cls, mv


# 去重时保留的份额优先级（A/E/I 优先，C 次之；同为 A 保留代码小者）
CLASS_RANK = {"": 0, "A": 0, "E": 1, "I": 1, "B": 2, "H": 2, "R": 3, "C": 4, "Y": 5}


def class_rank(cls: str) -> int:
    return CLASS_RANK.get(norm_str(cls).upper(), 9)


def dedup_share_classes(df: pd.DataFrame, code_col="code", name_col="name",
                        share_col="share_class", group_col="share_class_group") -> pd.DataFrame:
    """同组（A/C/E）只保留一个份额，优先 A/E/I，其余保留代码最小。

    输入需已含 share_class / share_class_group 列（builder 负责生成；
    无 group 列时退化按名称去重）。
    """
    if df.empty:
        return df
    df = df.copy()
    if group_col not in df.columns:
        if name_col not in df.columns:
            return df
        df[group_col] = df[name_col].astype(str).str.strip()
    if share_col not in df.columns:
        df[share_col] = ""
    df["__rank"] = df[share_col].apply(class_rank)
    df["__code_int"] = df[code_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("999999").astype(int)
    keep = (df.sort_values(["__rank", "__code_int"])
              .groupby(group_col, sort=False).head(1)
              .sort_index())
    return keep.drop(columns=["__rank", "__code_int"])


def purchase_state(status_text: str) -> tuple[str, str]:
    """状态文本 → (mode, raw)。

    mode: open / suspend_all / suspend_large / closed / unknown
    """
    s = norm_str(status_text)
    if not s or s in ("nan", "-", "--", "未知", "暂停"):
        return ("unknown", s)
    if SUSPEND_LARGE.search(s):
        lim = _limit_yuan(s)
        return (("suspend_limit:" + lim) if lim else "suspend_limit", s)
    if SUSPEND_ALL.search(s):
        return ("suspend_all", s)
    if RESTORE.search(s):
        return ("open", s)
    if DELIST_KW.search(s):
        return ("closed", s)
    return ("open", s) if re.search(r"开放|正常|申购|交易", s) else ("unknown", s)


def code_only(df: pd.DataFrame) -> pd.DataFrame:
    """确保含 6 位 code 列并去重。"""
    df = df.copy()
    c = next((x for x in ("code", "基金代码", "基金编号", "ts_code") if x in df.columns), None)
    if c is None:
        raise ValueError("缺少基金代码列")
    df["code"] = df[c].map(code6)
    df = df.dropna(subset=["code"]).drop_duplicates("code")
    return df
