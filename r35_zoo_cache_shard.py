#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #24（R3.5）前置：动物园评分缓存分片构建器（**纯基建，零语义变更**）

与已删除的 `r35_rebuild_zoo_cache.py`（串行版）的关系
------------------------------------------------------
【2026-09-02 文件整理】串行版已删除：`--nshards 1 --shard 0` 与其**逐字节等价**
（同一月份枚举、同一断点续跑语义），保留两份等价脚本属冗余。历史版本见 git ≤ `36d7134`。

本脚本**不改变任何计算**：同一 `backtest_local.harvest_date`、同一种子池
（`top100_history_pool.txt`）、同一月末枚举（2014-04→2026-03，144 个月末）、
同一文件名与落盘格式（`output/bt_scores_cache/<date>_2e4ec0f5.csv`）。
唯一差异 = **把月份集合按 shard 切分到多进程**，因为：

  1. 各月末评分**彼此独立**（harvest_date 只读 ≤as_of 的净值，月与月之间无状态传递）；
  2. M11 修复后 `finalize` 按 code 稳定排序 ⇒ 单月产物对线程/进程调度顺序**不敏感**；
  3. 实测该任务为 **I/O bound**（单进程 CPU 占用仅 ~40%，瓶颈在 15k 净值 CSV 的反复读取），
     2 核机器上串行需 ~5.5h，分片后可压到 ~2h。

因此分片与串行的产物应**逐字节一致**；本脚本保留 `--verify` 模式用于抽样自证（见下）。

内存保护：每完成一个月清空 `provider._memo` 的 nav 条目（与 `p1_panel_build` 同做法）。
该缓存纯属读缓存，清理**不影响任何数值**，只防止多进程并跑时 OOM（本机 3GB）。

用法:
    python r35_zoo_cache_shard.py --shard 0 --nshards 3
    python r35_zoo_cache_shard.py --verify 2014-04-30   # 重算一月并与已落盘文件逐字节比对
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from backtest_local import harvest_date, CACHE_DIR

SUF = "_2e4ec0f5"
START, END = "2014-04-30", "2026-03-31"   # 与 r35_rebuild_zoo_cache.py 完全一致


def load_codes():
    return [l.strip().zfill(6) for l in open("top100_history_pool.txt", encoding="utf-8")
            if l.strip() and l.strip()[0].isdigit()]


def all_dates():
    return [str(d.date()) for d in pd.date_range(START, END, freq="ME")]


def _drop_nav_memo():
    for k in [k for k in list(provider._memo) if k.startswith("nav_")]:
        provider._memo.pop(k, None)


def run_shard(shard: int, nshards: int):
    codes = load_codes()
    dates = all_dates()
    mine = [d for i, d in enumerate(dates) if i % nshards == shard]
    todo = [d for d in mine if not os.path.exists(f"{CACHE_DIR}/{d}{SUF}.csv")]
    print(f"[shard {shard}/{nshards}] 分得 {len(mine)} 月，待打 {len(todo)} 月；"
          f"池 {len(codes)} 只", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for i, d in enumerate(todo, 1):
        g = harvest_date(d, codes)
        if not g.empty:
            tmp = f"{CACHE_DIR}/{d}{SUF}.csv.tmp"
            g.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, f"{CACHE_DIR}/{d}{SUF}.csv")   # 原子写，防半截文件
        else:
            print(f"  [warn] {d} 无评分", flush=True)
        _drop_nav_memo()
        print(f"  [shard {shard}] {i}/{len(todo)} {d} ok", flush=True)
    print(f"[shard {shard}] done", flush=True)


def verify(date: str):
    """把某月重算一遍，与已落盘产物比 sha256：证明分片/串行产物一致（determinism 自证）。"""
    path = f"{CACHE_DIR}/{date}{SUF}.csv"
    if not os.path.exists(path):
        print(f"[verify] 缺 {path}"); return 1
    old = hashlib.sha256(open(path, "rb").read()).hexdigest()
    g = harvest_date(date, load_codes())
    tmp = f"{CACHE_DIR}/_verify_{date}.csv"
    g.to_csv(tmp, index=False, encoding="utf-8-sig")
    new = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    os.remove(tmp)
    ok = old == new
    print(f"[verify] {date} 落盘 {old[:16]} vs 重算 {new[:16]} → {'一致 ✅' if ok else '不一致 ❌'}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--verify", default=None, help="重算指定月末并与落盘产物比 sha256")
    a = ap.parse_args()
    if a.verify:
        raise SystemExit(verify(a.verify))
    run_shard(a.shard, a.nshards)


if __name__ == "__main__":
    main()
