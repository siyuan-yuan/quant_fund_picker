#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #14（D0.5）推定型底稿（离线可跑）+ 用户侧完整版说明

分级来源（预登记）：
  a. Tushare fund_basic 历史字段        —— 需要用户侧联网（🔶 受阻项，见 D0.1）
  b. 静态推定型：成立日 + 首发类型静态推定 + 变更公告修正
本脚本交付 (b) 的离线底稿：
  - 宇宙 = cache/nav_*.csv 全池（仅含至今仍存活者——**本缓存 100% 为幸存者集，
    死亡基金为 0**，见审计文件物证节）；
  - fund_type = 今日名录类型（cache/fund_meta.csv）→ 打标 type_src=inferred_today；
    **该列不是历史类型，一切下游使用必须带 inferred 标**；
  - status：有缓存且最新净值 < 30 天前 → active；≥30 天 → stale_or_dead(推断)；
  - inception_date：净值序列首日（名录无成立日字段时）；
  - known_at：= 快照日（推定口径的保守值，满足 known_at ≤ as_of 的硬约束）。
快照粒度：月末，2006-01-31 → 2026-03-31。快照 membership 规则：
  code ∈ snapshot(m) ⟺ inception_date < m 且（m ≤ last_nav_date 或 m 在停披露容忍 61 日内）
产物：data/pit_universe/YYYY-MM-DD.csv + output/v5/d05_universe_audit.md
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

CACHE = "cache"
OUT_DIR = "data/pit_universe"
AUDIT = "output/v5/d05_universe_audit.md"
START, END = "2006-01-31", "2026-03-31"
STALE_DAYS = 30


def nav_span(fp):
    df = pd.read_csv(fp, usecols=["date"], parse_dates=["date"])
    return df.date.min(), df.date.max(), len(df)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("output/v5", exist_ok=True)

    meta = None
    for cand in [f"{CACHE}/fund_meta.csv", f"{CACHE}/fund_meta_full.csv"]:
        if os.path.exists(cand):
            meta = pd.read_csv(cand, dtype=str)
            break

    rows = []
    for fp in sorted(glob.glob(f"{CACHE}/nav_*.csv")):
        code = os.path.basename(fp)[4:-4]
        first, last, n = nav_span(fp)
        rows.append(dict(code=code, first=first, last=last, n_nav=n))
    sp = pd.DataFrame(rows).set_index("code")
    print(f"[D0.5b] 净值覆盖 {len(sp)} 只")

    if meta is not None:
        code_cols = [c for c in meta.columns if c in ("基金代码", "code")]
        type_cols = [c for c in meta.columns if c in ("基金类型", "fund_type")]
        name_cols = [c for c in meta.columns if c in ("基金简称", "基金名称", "name")]
        if code_cols:
            m = meta.set_index(code_cols[0])
            sp["fund_type"] = m[type_cols[0]].reindex(sp.index) if type_cols else np.nan
            sp["name"] = m[name_cols[0]].reindex(sp.index) if name_cols else np.nan
    for c in ["fund_type", "name"]:
        if c not in sp.columns:
            sp[c] = np.nan
    sp["inception"] = sp["first"]

    months = pd.date_range(START, END, freq="ME")
    n_snaps = 0
    stats_rows = []
    for m in months:
        mem = sp[(sp.inception < m)]
        mem = mem[(mem["last"] >= m) | ((m - mem["last"]).dt.days <= STALE_DAYS + 31)]
        g = pd.DataFrame(dict(
            code=mem.index, name=mem["name"].fillna("").values,
            fund_type=mem["fund_type"].fillna("").values,
            status=np.where((m - mem["last"]).dt.days <= STALE_DAYS, "active", "stale_or_dead(推断)"),
            inception_date=mem["inception"].dt.date.astype(str).values,
            known_at=str(m.date()),
            type_src="inferred_today"))
        fp = f"{OUT_DIR}/{m.date()}.csv"
        g.to_csv(fp, index=False, encoding="utf-8-sig")
        n_snaps += 1
        if m.month == 12:
            stats_rows.append((str(m.date()), len(g), int((g.status != "active").sum())))
    sample = "; ".join(f"{d}: {n}只(非active {k})" for d, n, k in stats_rows[-8:])

    # 幸存者偏差物证：全池最后净值日期分布
    last_all = pd.to_datetime(sp["last"])
    ev = (f"全池 {len(sp):,} 只 nav 最后净值日期：min={last_all.min().date()}、"
          f"P5={last_all.quantile(.05).date()}、P50={last_all.quantile(.50).date()}、"
          f"max={last_all.max().date()}；2020/2023/2024 年前停披露数 = "
          f"{(last_all < '2020-01-01').sum()}/{(last_all < '2023-01-01').sum()}/{(last_all < '2024-01-01').sum()}")

    audit = ["# D0.5 推定型 PiT 宇宙底稿审计", "",
             f"快照 {n_snaps} 份（月末，{START} → {END}），全部 type_src=inferred_today。", "",
             "**硬规则声明**：known_at=快照日（≤as_of 恒成立）；历史月份 membership 含晚近消失"
             "基金的最后披露月（本缓存中事实上不存在该类基金——见物证节）。", "",
             "## 抽样", "", f"年末样本点：{sample}", "",
             "## 关键物证（缓存层幸存者偏差直接量测）", "", ev, "",
             "⇒ 本仓库一切历史产物（含 P1-P4/V3.x/V4）的池层级**从未包含任何已消失基金**；"
             "『严格 PIT』标签历史上仅适用于时间截断维度，宇宙维度从来是幸存者并集投影。", "",
             "## 局限（真实名录合入前此底稿一律打标）",
             "1. fund_type 为今日名录类型，历史变更未追溯（推定型）；",
             "2. 仅覆盖有净值缓存者；真实死亡基金名录合入依赖 D0.1 探针阳性 + 用户侧 Tushare fund_basic；",
             "3. stale_or_dead(推断) = 最新净值距快照日 >30 天，未区分清盘/暂停披露/份额合并。",
             "> ⚠️ 本底稿不构成 FULL-PIT 宇宙；U4.x 全池重建在真实名录到位前冻结（预登记硬冻结）。"]
    open(AUDIT, "w", encoding="utf-8").write("\n".join(audit))
    print(f"[D0.5b] 完: {n_snaps} 份快照 → {OUT_DIR}/ ；审计 → {AUDIT}")


if __name__ == "__main__":
    main()
