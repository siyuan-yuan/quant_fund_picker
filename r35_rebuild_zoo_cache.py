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
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from backtest_local import harvest_date, CACHE_DIR

SUF = "_2e4ec0f5"
# 探针实证（2026-09-01 三轮 × 双进程日志一致）：2014-03-31 及之前全月评分=0 且空月不写盘 →
# 起点右移至 2014-04 不改变任何产出字节（早段只会产生"无评分"日志与零文件），节省 ~79 分钟 CPU。
START, END = "2014-04-30", "2026-03-31"


CODES = [l.strip().zfill(6) for l in open("top100_history_pool.txt", encoding="utf-8")
         if l.strip() and l.strip()[0].isdigit()]


def _harvest_one(d):
    """打一个月并原子落盘；返回 (date, n_rows)"""
    fp = f"{CACHE_DIR}/{d}{SUF}.csv"
    if os.path.exists(fp):            # 断点续跑（并发下二次确认）
        return d, -1
    g = harvest_date(d, CODES)
    if g.empty:
        return d, 0
    g.to_csv(fp + ".tmp", index=False, encoding="utf-8-sig")
    os.replace(fp + ".tmp", fp)
    return d, len(g)


def main(workers: int = 1):
    have_nav = sum(os.path.exists(f"cache/nav_{c}.csv") for c in CODES)
    print(f"[R3.5-cache] 种子池 {len(CODES)} 只；净值缓存覆盖 {have_nav}/{len(CODES)}", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    dates = [str(d.date()) for d in pd.date_range(START, END, freq="ME")]
    todo = [d for d in dates if not os.path.exists(f"{CACHE_DIR}/{d}{SUF}.csv")]
    print(f"[R3.5-cache] 月末 {len(dates)}，待打 {len(todo)} 个月（workers={workers}）", flush=True)
    if not todo:
        print("[R3.5-cache] 缓存已齐，跳过", flush=True)
        return

    done, empty = 0, 0
    if workers <= 1:
        for i, d in enumerate(todo, 1):
            _, n = _harvest_one(d)
            if n == 0:
                empty += 1
                print(f"  [warn] {d} 无评分（可能全池该月历史不足）", flush=True)
            done += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {d} rows={n}", flush=True)
    else:
        # 月级并行：每月独立落盘，语义与串行完全一致（harvest_date 为纯函数式月快照）；
        # max_tasks_per_child 回收进程，避免 provider 净值缓存逐月累积吃内存。
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=4) as ex:
            futs = {ex.submit(_harvest_one, d): d for d in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                d, n = fu.result()
                if n == 0:
                    empty += 1
                    print(f"  [warn] {d} 无评分（可能全池该月历史不足）", flush=True)
                done += 1
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {d} rows={n}", flush=True)
    print(f"[R3.5-cache] done（打完 {done} 月，空月 {empty}）", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1,
                    help="月级并行进程数（月间零耦合、按月断点续跑；单进程内存约数百 MB，建议 ≤ 核数/2）")
    main(ap.parse_args().workers)
