#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #24（R3.5）前置：动物园评分缓存 _2e4ec0f5 复刻（SURV-ADJ）

output/bt_scores_cache/*_2e4ec0f5.csv 不随仓库分发；本脚本按原式复刻：
  池 = top100_history_pool.txt（216 只历史并集池，F3 已定性为幸存者偏差并集 →
  本重建产物的合法口径仅为 SURV-ADJ 勘误重跑，**禁作任何 FULL-PIT 裁决**）；
  打分 = backtest_local.harvest_date（score_fund bt=True + finalize，原式）；
  月份 = 2006-09 → 2026-03 全部月末（235 个月，覆盖动物园原始区间）。
断点续跑。
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from backtest_local import harvest_date, CACHE_DIR

SUF = "_2e4ec0f5"
START, END = "2006-09-30", "2026-03-31"


def main():
    codes = [l.strip().zfill(6) for l in open("top100_history_pool.txt", encoding="utf-8")
             if l.strip() and l.strip()[0].isdigit()]
    have_nav = sum(os.path.exists(f"cache/nav_{c}.csv") for c in codes)
    print(f"[R3.5-cache] 种子池 {len(codes)} 只；净值缓存覆盖 {have_nav}/{len(codes)}", flush=True)

    dates = [str(d.date()) for d in pd.date_range(START, END, freq="ME")]
    todo = [d for d in dates if not os.path.exists(f"{CACHE_DIR}/{d}{SUF}.csv")]
    print(f"[R3.5-cache] 月末 {len(dates)}，待打 {len(todo)} 个月", flush=True)
    for i, d in enumerate(todo, 1):
        g = harvest_date(d, codes)
        if not g.empty:
            g.to_csv(f"{CACHE_DIR}/{d}{SUF}.csv", index=False, encoding="utf-8-sig")
        else:
            print(f"  [warn] {d} 无评分（可能全池该月历史不足）", flush=True)
    print("[R3.5-cache] done", flush=True)


if __name__ == "__main__":
    main()
