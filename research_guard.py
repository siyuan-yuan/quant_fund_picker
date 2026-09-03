#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究纪律护栏 · C8：标签窗前视硬断言 + 标签契约四元组（工程护栏，零数值影响）

C8（docs/V5_模型改进候选台账_2026-09.md §C8，2026-09-02 预登记）：
  在一切「打标脚本」（从原始净值构建 fwd 收益标签的产线）入口加硬断言——
  **标签起点（fwd 收益的基期 bar）必须 ≥ 特征快照日 + 执行延迟**；
  并把 {标签窗, 特征窗, 训练截止规则, 执行延迟} 四元组写进每份产物的 manifest 头。

证据锚（为什么值得做）：
  R3.5 fwd12_ab.csv —— `huber F2 fwd12_z` 仅把训练截止由 q−6M 改为 h 匹配的 q−12M，
  IC +0.0773 → −0.0842（符号翻转，HAC −2.00）。一处标签窗前视足以把「看似最佳」翻成显著为负。

护栏性质（候选台账明示）：
  - 本模块**不改变任何既有标签数值**（不改 base bar、不改 fwd 收益、不改训练样本）；
  - 只做两件事：① 在每次构建/落盘时对"标签起点是否可成交（无前视、无过早基期）"
    作**显式硬断言**（raise，不可被 -O 静默关闭）；② 把标签契约四元组写入产物 manifest 头，
    供审计按 manifest 复核。

约定（重要，避免把「执行延迟」与「标签口径」混为一谈）：
  - `exec_delay_days` = **标签口径**下从特征快照日到可成交 bar 的交易日数。
    本项目现状：IC/评分用 fwd 标签**以决策日(特征快照日)当月 bar 起算** → 标签口径 exec_delay=0
    （`p1_panel_build` / `_build_ml_panel` 的 base 即 asof(决策日)）。执行层 sim 的 T+1（D0.4）
    是**另一层**口径，二者分离并各自打标，禁止在本模块混改数值。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 标签契约四元组 {标签窗, 特征窗, 训练截止规则, 执行延迟}
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelSpec:
    """一份打标产线的标签契约（C8 四元组 + 标识）。"""
    pipeline: str                # 唯一产物标识，例如 "ml_panel" / "p1_panel_canonical"
    horizon_months: tuple        # 标签窗（月），例如 (3, 6, 12)
    feature_snapshot_rule: str   # 特征窗规则（特征自何日/窗口截断）
    train_cutoff_rule: str       # 训练截止规则（h 匹配：训练决策日 ≤ q − horizon_months）
    exec_delay_days: int         # 执行延迟（交易日），标签口径 base 偏移
    # base 起算规则：标签 base bar 相对特征快照日的取法
    base_rule: str = "标签以特征快照日(决策日)当月可成交 bar 为基期起算"

    def four_tuple(self) -> dict:
        """C8 四元组，作为产物 manifest 头的标准键值。"""
        return {
            "标签窗": list(self.horizon_months),
            "特征窗": self.feature_snapshot_rule,
            "训练截止规则": self.train_cutoff_rule,
            "执行延迟": self.exec_delay_days,
        }

    def header_lines(self) -> List[str]:
        """返回可直接写入 manifest/文件头的若干行（每行以 # label_contract: 前缀）。"""
        blob = json.dumps({"pipeline": self.pipeline, **self.four_tuple(),
                           "base_rule": self.base_rule},
                          ensure_ascii=False, sort_keys=True)
        # 折行便于人工审读，但仍保持单行可 json 解析的主键行
        pretty = json.dumps({"pipeline": self.pipeline, **self.four_tuple(),
                             "base_rule": self.base_rule},
                            ensure_ascii=False, indent=2, sort_keys=True)
        return ["# label_contract: " + blob,
                "# label_contract(pretty):"] + \
               ["#   " + ln for ln in pretty.splitlines()]


def default_contract(pipeline: str, horizon_months: tuple,
                     exec_delay_days: int = 0) -> LabelSpec:
    """便捷构造：默认特征窗/训练截止为项目标准（决策日截断 + h 匹配）。"""
    return LabelSpec(
        pipeline=pipeline,
        horizon_months=tuple(horizon_months),
        feature_snapshot_rule=f"决策日 d 当月截断（仅用 ≤ d 数据）",
        train_cutoff_rule=(
            "h 匹配：以决策日 d 为样本的行, 其 horizon 标签须在 d + horizon_months 前成熟; "
            "训练截止 ≤ q − horizon_months"),
        exec_delay_days=exec_delay_days,
    )


def write_contract_sidecar(out_csv_path: str, spec: LabelSpec,
                           manifest_path: Optional[str] = None) -> str:
    """把标签契约四元组写成 `<out_csv_path>.label_contract.json` 侧车 + manifest 头。

    - CSV 产物不写 `#` 注释头（会破坏 pandas read_csv），故用同名 `.label_contract.json` 侧车；
    - 若给出 manifest_path（v5 风格 jsonl），追加一行 label_contract 记录。
    返回侧车路径。
    """
    side = f"{out_csv_path}.label_contract.json"
    with open(side, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pipeline": spec.pipeline, **spec.four_tuple(),
                             "base_rule": spec.base_rule},
                            ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if manifest_path:
        rec = {"type": "label_contract", "pipeline": spec.pipeline,
               "artifact": os.path.basename(out_csv_path),
               **spec.four_tuple(), "sidecar": os.path.basename(side)}
        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return side


# --------------------------------------------------------------------------
# C8 硬断言：标签起点 ≥ 特征快照日 + 执行延迟（且不越过可成交基期 = 无前视）
# --------------------------------------------------------------------------
def _asof_pos(dates: pd.DatetimeIndex, snap: pd.Timestamp) -> int:
    """返回 dates 中 ≤ snap 的最后一个 bar 下标（即 asof(snap)），找不到返回 -1。"""
    pos = int(dates.searchsorted(snap, side="right")) - 1
    return pos


def resolve_first_tradeable_date(adj_dates: pd.DatetimeIndex, snap: pd.Timestamp,
                                 exec_delay_days: int) -> Optional[pd.Timestamp]:
    """标签口径下，特征快照日 snap 之后"第一个可成交 base bar"的日期。

    - exec_delay_days == 0：可成交基期 = asof(snap)（决策日当月最后一个已披露 bar）；
    - exec_delay_days == k>0：可成交基期 = snap 之后第 k 个交易日 bar
      （对应 D0.4 T+1/T+2 执行语义，标签须自成交 bar 起算才可归因）。
    越界（无足够未来 bar）返回 None。
    """
    pos = _asof_pos(adj_dates, snap)
    if pos < 0:
        return None
    target = pos + exec_delay_days
    if target >= len(adj_dates):
        return None
    return adj_dates[target]


def assert_label_start_not_earlier(rows: pd.DataFrame, context: str = "",
                                   raise_=True) -> int:
    """C8 硬断言：对每一行，标签起点(实际 base 日) ≥ 该行最小可成交基期。

    rows 须含列：
      label_start_date  : 本行实际用作 fwd 标签基期 bar 的日期（未产出标签为 NaT）
      min_label_start   : resolve_first_tradeable_date(...) 给出的最小可成交基期
    返回违规行数（raise_=False 时不抛，供诊断）。
    """
    valid = rows["label_start_date"].notna()
    if not valid.any():
        return 0
    v = rows.loc[valid]
    # 缺失 min 视为不可成交 → 违规
    has_min = v["min_label_start"].notna()
    too_early = pd.Series(False, index=v.index)
    too_early[has_min] = v.loc[has_min, "label_start_date"] < \
        v.loc[has_min, "min_label_start"]
    no_min = ~has_min
    n_bad = int((too_early | no_min).sum())
    if n_bad and raise_:
        bad = v.loc[too_early | no_min].head(5)
        detail = "\n".join(
            f"  code/snap={r['snapshot_date']} label_start={r['label_start_date']} "
            f"min_allowed={r['min_label_start']}"
            for _, r in bad.iterrows())
        raise AssertionError(
            f"[C8 标签起点 < 特征快照日+执行延迟] {context}: {n_bad} 行违规（越早基期 / "
            f"不可成交基期）。违反 F4 类前视/过早基期护栏。\n{detail}")
    return int(n_bad)


def assert_label_start_not_in_future(rows: pd.DataFrame, context: str = "",
                                     raise_=True) -> int:
    """F4 类防护（前视方向）：标签 base 不得越过 asof(snap)（决策日之后的信息）。

    仅当标签口径 exec_delay==0 时合法 base == asof(snap)；若实现把 base 取到
    snap 之后的 bar，说明"用决策日后价格做基期" → 违规。exec_delay>0 的产线
    由调用方自行用 resolve_first_tradeable_date 校准，此函数针对 delay==0 产线。
    """
    valid = rows["label_start_date"].notna()
    if not valid.any():
        return 0
    v = rows.loc[valid]
    snap = pd.to_datetime(v["snapshot_date"])
    fut = pd.to_datetime(v["label_start_date"]) > snap
    n_bad = int(fut.sum())
    if n_bad and raise_:
        bad = v.loc[fut].head(5)
        detail = "\n".join(
            f"  code/snap={r['snapshot_date']} label_start={r['label_start_date']}"
            for _, r in bad.iterrows())
        raise AssertionError(
            f"[C8 标签 base 越过决策日(前视)] {context}: {n_bad} 行违规。\n{detail}")
    return int(n_bad)


def validate_label_contract(rows: pd.DataFrame, spec: LabelSpec,
                            context: str = "") -> int:
    """一次跑满 C8 两个方向硬断言（先执行后裁，只 raise 违规行）。

    rows 列：snapshot_date / label_start_date / min_label_start（min 由调用方给出）。
    返回违规总数（正常为 0，违规即 raise）。
    """
    n1 = assert_label_start_not_earlier(rows, context)
    n2 = 0
    if spec.exec_delay_days == 0:
        # delay==0：base 必须 == asof(snap)，既不能早(上面)也不能越过决策日(这里)
        n2 = assert_label_start_not_in_future(rows, context)
    return n1 + n2


# --------------------------------------------------------------------------
# 训练截止 h 匹配（F4 复发闸，冗余于 _model_zoo.TARGET_HORIZON_MO 单测）
# --------------------------------------------------------------------------
def label_matures_on(decision_date, horizon_months) -> pd.Timestamp:
    """标签成熟日：决策日 d 的 horizon 标签到 d + horizon_months 才完全已知。"""
    return pd.Timestamp(decision_date) + pd.DateOffset(months=int(horizon_months))


def assert_train_cutoff_h_matched(train_decision_dates, cutoff_date,
                                  horizon_months, context: str = ""):
    """训练截止纪律：cutoff_date 处允许使用的"最新"决策日须满足其标签已成熟。

    即 max(train_decision_date) + horizon_months <= cutoff_date。
    违反 ⇒ 训练样本包含标签未成熟的未来（F4 复发）。
    """
    cutoff = pd.Timestamp(cutoff_date)
    arr = pd.to_datetime(list(train_decision_dates))
    if len(arr) == 0:
        return
    latest = arr.max()
    if label_matures_on(latest, horizon_months) > cutoff:
        raise AssertionError(
            f"[F4 训练截止未 h 匹配] {context}: 决策日 {latest.date()} 的 "
            f"horizon={horizon_months}M 标签到 "
            f"{label_matures_on(latest, horizon_months).date()} 才成熟, "
            f"但 cutoff={cutoff.date()} 即已使用 → 前视复发。")


if __name__ == "__main__":
    # 自检
    spec = default_contract("demo", (6, 12))
    print("\n".join(spec.header_lines()))
    assert len(spec.four_tuple()) == 4
    print("\n[OK] research_guard 自检通过")
