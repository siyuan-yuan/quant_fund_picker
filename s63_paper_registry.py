#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #35（S6.3）：前瞻纸面验证登记器

协议（冻结，执行期不得改动，改动即重置观察期）：
  1. 起点 = 下一个自然决策月（上月末最后交易日）；
  2. 每月固定动作：完整刷新净值缓存后运行本脚本一次，记录当时生产模型 V3.8
     的 Top10 名单 + S_total + 宇宙规模；**只记录，不改模型**；
  3. 12 个月台账齐后（fwd6 成熟才可以评估）：与历史 IC 分布对照；
     统计口径 §0.3 前瞻段规则（fwd6 月度 IC，n=12 → 仅报告分布与符号，不作 t 判定——
     n=12 下任何 t 都无意义，此为本协议的诚实边界，预登记封死）。
  4. 台账路径：data/s63_ledger/YYYY-MM-DD.csv —— 每行一基金（code, S_total, rank, 当月宇宙合格数）。
     台账一旦写入不可改写（只允许追加月份）；若某月漏跑，该月记 missing 并在终评说明。

运行：python s63_paper_registry.py [--note "..."]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from engine import score_fund, finalize

LEDGER_DIR = "data/s63_ledger"
PROTOCOL_SEAL = "V5-#35 冻结于 2026-09-01（commit 见 git log）；改动即重置观察期"


def latest_month_end() -> str:
    """以沪深300缓存日历定最近一个已收盘的月末交易日。"""
    cal = pd.to_datetime(pd.read_csv("cache/idx_sh000300.csv", usecols=["date"])["date"])
    today = pd.Timestamp(dt.date.today())
    past = cal[cal <= today]
    me = past.groupby(past.dt.to_period("M")).max()
    return str(me.iloc[-1].date())


def universe_codes() -> list[str]:
    """现存池（SURV；FULL-PIT 解锁后此函数须替换为 PiT 宇宙，替换即新协议版本号）。"""
    import json
    pool = json.load(open("cache/fund_pool.json", encoding="utf-8"))
    codes = pool if isinstance(pool, list) else pool.get("codes", list(pool))
    return sorted({str(c).zfill(6) for c in codes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--date", default=None, help="覆盖登记月（默认自动取最近月末交易日）")
    args = ap.parse_args()

    d = args.date or latest_month_end()
    os.makedirs(LEDGER_DIR, exist_ok=True)
    fp = os.path.join(LEDGER_DIR, f"{d}.csv")
    if os.path.exists(fp):
        raise SystemExit(f"[S6.3] 台账 {fp} 已存在——协议禁止改写历史月份。若为误跑请人工删除并在日志声明。")

    codes = universe_codes()
    rows = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(score_fund, c, as_of=d, bt=True): c for c in codes}
        for fut in as_completed(futs):
            r = fut.result()
            if not r.get("error"):
                rows.append(r)
    df = finalize(rows, as_of=d)
    ok = df.dropna(subset=["S_total"]).copy()
    ok["rank"] = ok["S_total"].rank(ascending=False, method="first").astype(int)
    out = ok.sort_values("rank")[["rank", "code", "S_total"]]
    out["universe_n"] = len(codes)
    out["protocol"] = PROTOCOL_SEAL
    out["note"] = args.note
    out.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"[S6.3] 登记 {d}: 宇宙 {len(codes)} 只, 有评分 {len(out)} 只 → {fp}")
    print(out.head(10).to_string(index=False))
    print("\n[协议] 本月记录已封存。12 个月台账齐后按 §0.3 前瞻段规则对照历史 IC 分布。")


if __name__ == "__main__":
    main()
