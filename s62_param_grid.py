#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #34（S6.2）：规则参数族的邻域稳健性（仅报告，禁止用于再调参）

网格（预登记自封，事后不得增删）：
  买入线 buy ∈ {65, 70, 75}；卖出线 sell ∈ {40, 45, 50}；
  CPPI 档位三档同步 ±5pp ∈ {(-0.10,-0.15,-0.20), (-0.15,-0.20,-0.25), (-0.20,-0.25,-0.30)}；
  trail ∈ {0.15, 0.20, 0.25}；槽位 slots ∈ {8, 10, 12}；合计 3×3×3×3×3 = 243 格。

口径：SURV-ADJ（幸存池 × 修复执行口径），与 B2.1 R3 完成态基线一致：
  panel = P1-0 重建面板（2014-03→2026-03）；成交价 navadj；exec_delay=1（T+1）；
  成本 = 买入 0.15% + 持有期阶梯（<7d 1.5% / <365d 0.5% / <730d 0.25% / ≥730d 0）。
  （实施注记：#34 预登记未固化执行口径，采用 B2.1-R3 当前唯一基线口径——此处登记，事后不再改。）

报告口径（预登记）：基线格（买70/卖45/CPPI 基准/trail 0.20/slots 10）在 243 格邻域分布中的
  位置（分位）+ 各单维边际分布；不平判、不择优。任何"邻域内更优点"严禁进入裁决——
  那是过拟合陷阱，本实验全程视邻域最优为噪声。

产物：output/v5/s62_param_grid.csv / s62_summary.md
"""
from __future__ import annotations

import itertools
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

import sim_core as SC
from b21_baseline import ARGS, LazyNavs, NAV_ADJ, bench_series, load_panel

OUT = "output/v5"
BASE = dict(buy=70.0, sell=45.0, cppi=(-0.15, -0.20, -0.25), trail=0.20, slots=10)


def ladder_cost(hold_days, is_ov):
    if is_ov:
        return 0.005
    if hold_days < 7:
        return 0.015
    if hold_days < 365:
        return 0.005
    if hold_days < 730:
        return 0.0025
    return 0.0


def run_one(panel, bench, buy, sell, cppi, trail, slots):
    args = types.SimpleNamespace(**{**ARGS, "buy": buy, "sell": sell, "slots": slots,
                                    "trail_stop": trail,
                                    "cppi_dd1": cppi[0], "cppi_dd2": cppi[1],
                                    "cppi_dd3": cppi[2]})
    navs = LazyNavs(NAV_ADJ)
    dates = [str(d) for d in sorted(panel.date.unique())]
    ec, tr = SC.simulate(panel, navs, bench, dates, args, exec_delay_days=1,
                         cost_out_fn=ladder_cost, label=f"s62_{buy}_{sell}")
    yrs = (ec.index[-1] - ec.index[0]).days / 365.25
    tot = ec.equity.iloc[-1] / ARGS["capital"] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    dd = ec.drawdown.min()
    return dict(buy=buy, sell=sell, cppi=repr(cppi), trail=trail, slots=slots,
                cagr=float(cagr), maxdd=float(dd),
                calmar=float(cagr / abs(dd)) if dd else np.nan,
                n_trades=len(tr), cost=float(ec.attrs.get("total_cost", np.nan)))


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_panel()
    bench = bench_series()
    grid = list(itertools.product([65.0, 70.0, 75.0], [40.0, 45.0, 50.0],
                                  [(-0.10, -0.15, -0.20), (-0.15, -0.20, -0.25),
                                   (-0.20, -0.25, -0.30)],
                                  [0.15, 0.20, 0.25], [8, 10, 12]))
    assert len(grid) == 243 and tuple(BASE[k] for k in ("buy", "sell")) == (70.0, 45.0)
    rows = []
    for i, (buy, sell, cppi, trail, slots) in enumerate(grid, 1):
        r = run_one(panel, bench, buy, sell, cppi, trail, slots)
        rows.append(r)
        if i % 20 == 0 or i == 1:
            print(f"[S6.2 {i}/243] buy={buy} sell={sell} trail={trail} slots={slots} "
                  f"CAGR={r['cagr']:+.2%} MaxDD={r['maxdd']:+.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/s62_param_grid.csv", index=False)

    base = df[(np.isclose(df.buy, 70)) & (np.isclose(df.sell, 45)) &
              (df.cppi == repr(BASE["cppi"])) & (np.isclose(df.trail, 0.20)) &
              (df.slots == 10)].iloc[0]
    pct = (df.cagr < base.cagr).mean()
    lines = [
        "# S6.2 规则参数邻域稳健性（243 格，仅报告；SURV-ADJ 口径，禁作再调参依据）",
        "",
        f"全域 CAGR ∈ [{df.cagr.min():+.2%}, {df.cagr.max():+.2%}]；MaxDD ∈ [{df.maxdd.min():+.1%}, {df.maxdd.max():+.1%}]",
        f"基线格 CAGR={base.cagr:+.2%}，位于邻域分位 **{pct:.0%}**（=CAGR 低于基线的格占比）",
        "",
        "## 单维边际分布（CAGR 中位数）",
        "",
    ]
    for col in ["buy", "sell", "cppi", "trail", "slots"]:
        g = df.groupby(col).cagr.agg(["min", "median", "max"]).round(4)
        lines.append(f"### {col}\n\n{g.to_markdown()}\n")
    lines += [
        "## 判定说明",
        "",
        "- 本实验只测平坦性：邻域分布越集中 → 基线越不依赖精调；分位 ~50% 为健康基线特征。",
        "- **邻域内任何更优格一律视为噪声**，不得据此改生产参数（预登记硬规则）。",
    ]
    with open(f"{OUT}/s62_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[S6.2 done] 基线分位 {pct:.0%}，CAGR 全域 [{df.cagr.min():+.2%},{df.cagr.max():+.2%}]")


if __name__ == "__main__":
    main()
