#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #21（B2.2）成本敏感性网格 + #22（B2.3）执行时滞敏感性（仅报告，不裁决）

中枢口径 = B2.1 R3（navadj + T+1 + 阶梯成本 + V3.8 默认参数，SURV-ADJ）。

B2.2 网格（预登记 3×3×3=27 格，事后不得增删）：
  申购 ∈ {0, 0.15%, 1.5%} × 赎回阶梯整体倍率 ∈ {0.5×, 1×, 2×} × 滑点 ∈ {0, 10bp, 20bp}
  滑点实现：一阶费率等价（成交价±滑点 ≈ 费率加项；买入 cin+slip、卖出 cout+slip），
  处理口径在此披露，不改变成交时序。
B2.3：exec_delay ∈ {0,1,2}（其余口径与 R3 同），并做逐笔"信号日→成交日"价滑分布。

产物：output/v5/b22_cost_grid.csv / b23_delay_diff.csv / b23_slippage.csv / b22_b23_summary.md
"""
from __future__ import annotations

import itertools
import os
import types

import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True

import sim_core as SC
from b21_baseline import (ARGS, START, END, LazyNavs, NAV_ADJ, load_panel,
                          bench_series)

OUT = "output/v5"
LADDER_BASE = [(7, 0.015), (365, 0.005), (730, 0.0025), (10**9, 0.0)]


def ladder_scaled(scale, slip):
    def f(gross, hd):
        for cap, rate in LADDER_BASE:
            if hd < cap:
                return rate * scale + slip
        return slip
    return f


def run(panel, navs, bench, dates, delay=1, cin=0.0015, cout_fn=None, tag=""):
    args = types.SimpleNamespace(**{**ARGS, "cost_in": cin})
    ec, tr = SC.simulate(panel, navs, bench, dates, args,
                         exec_delay_days=delay,
                         cost_in_fn=(lambda amt: cin), cost_out_fn=cout_fn, label=tag)
    yrs = (ec.index[-1] - ec.index[0]) / pd.Timedelta(days=365.25)
    tot = ec.equity.iloc[-1] / ARGS["capital"] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    dd = ec.drawdown.min()
    return dict(tag=tag, delay=delay, cagr=round(float(cagr), 6),
                maxdd=round(float(dd), 6),
                calmar=round(float(cagr / abs(dd)), 4) if dd else np.nan,
                total_ret=round(float(tot), 6), n_trades=len(tr),
                cost=round(float(ec.attrs.get("total_cost", np.nan)), 2),
                turn=round(float(ec.attrs.get("rebal_turnover", 0.0)), 2)), ec, tr


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_panel()
    bench = bench_series()
    dates = [str(d) for d in sorted(panel.date.unique())]
    navs = LazyNavs(NAV_ADJ)
    print(f"[B2.2/B2.3] 面板 {len(dates)} 月", flush=True)

    # ---------- B2.2 成本 27 格 ----------
    rows = []
    grid = list(itertools.product([0.0, 0.0015, 0.015], [0.5, 1.0, 2.0], [0.0, 1e-3, 2e-3]))
    for i, (cin, scale, slip) in enumerate(grid):
        tag = f"cin={cin:.4f}_ladder={scale:g}x_slip={int(slip*1e4)}bp"
        r, _, _ = run(panel, navs, bench, dates, delay=1, cin=cin,
                      cout_fn=ladder_scaled(scale, slip), tag=tag)
        r.update(cost_in=cin, ladder_scale=scale, slip=slip)
        rows.append(r)
        print(f"[B2.2 {i+1}/27] {tag:32s} CAGR={r['cagr']:+.2%} MaxDD={r['maxdd']:.1%} "
              f"cost={r['cost']:,.0f}", flush=True)
    g = pd.DataFrame(rows)[["cost_in", "ladder_scale", "slip", "cagr", "maxdd",
                            "calmar", "total_ret", "n_trades", "cost", "turn", "tag"]]
    g.to_csv(f"{OUT}/b22_cost_grid.csv", index=False, encoding="utf-8-sig")

    # ---------- B2.3 时滞 {0,1,2} ----------
    drows, slips_all = [], []
    for delay in [0, 1, 2]:
        tag = f"delay={delay}"
        r, ec, tr = run(panel, navs, bench, dates, delay=delay, cin=0.0015,
                        cout_fn=ladder_scaled(1.0, 0.0), tag=tag)
        drows.append(r)
        for _, t in tr.iterrows():
            s = navs[t.code]
            if s is None:
                continue
            pos = s.index.get_indexer([pd.Timestamp(t.entry_date)], method="ffill")
            if pos[0] - delay >= 0:
                px_sig = float(s.iloc[pos[0] - delay])
                px_exe = float(s.iloc[pos[0]])
                slips_all.append(dict(delay=delay, code=t.code, entry=t.entry_date,
                                      slip=px_exe / px_sig - 1))
        ec.reset_index().to_csv(f"{OUT}/b23_equity_delay{delay}.csv",
                                index=False, encoding="utf-8-sig")
        print(f"[B2.3] {tag}: CAGR={r['cagr']:+.2%} MaxDD={r['maxdd']:.1%} "
              f"trades={r['n_trades']}", flush=True)
    dd = pd.DataFrame(drows)
    dd.to_csv(f"{OUT}/b23_delay_diff.csv", index=False, encoding="utf-8-sig")
    sp = pd.DataFrame(slips_all)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    summ = (sp.groupby("delay")["slip"]
              .agg(["count", "mean", "std"] + [lambda x, q=q: x.quantile(q) for q in qs]))
    summ.columns = (["n", "mean", "std"] + [f"p{int(q*100)}" for q in qs])
    sp.to_csv(f"{OUT}/b23_slippage.csv", index=False, encoding="utf-8-sig")

    base = g[(g.cost_in == 0.0015) & (g.ladder_scale == 1.0) & (g.slip == 0.0)].iloc[0]
    L = ["# B2.2 成本敏感性 + B2.3 时滞敏感性（SURV-ADJ，仅报告不裁决）", "",
         f"中枢（R3 口径）：CAGR={base.cagr:+.2%} MaxDD={base.maxdd:.1%} Calmar={base.calmar}", "",
         f"## B2.2 27 格：CAGR 区间 [{g.cagr.min():+.2%}, {g.cagr.max():+.2%}]，"
         f"全距 {g.cagr.max()-g.cagr.min():.2%}", "",
         g.round(4).to_markdown(index=False), "",
         "## B2.3 时滞", "",
         dd.round(4).to_markdown(index=False), "",
         "信号日→成交日 价滑分布（买侧，正值=成交劣于信号日价）：", "",
         summ.round(4).to_markdown(), ""]
    open(f"{OUT}/b22_b23_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[done] {OUT}/b22_cost_grid.csv / b23_delay_diff.csv / b23_slippage.csv / b22_b23_summary.md")


if __name__ == "__main__":
    main()
