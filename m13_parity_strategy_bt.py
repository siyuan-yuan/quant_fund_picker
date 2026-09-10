#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1.3 对拍：strategy_bt(sim_core 薄封装) vs strategy_bt._simulate_legacy（冻结旧实现）

协议（预登记 #18 对拍级）：合成面板（含周末决策日映射、跌出面板、S=NaN、trail/ma20/hi_water/
water_gate 触发情形）+ 合成净值 + 合成基准；5 个变体配置逐一对比：
  1) 权益曲线 max|Δ| ≤ 1e-8；
  2) 交易台账 [code, entry_date, exit_date, exit_reason, hold_days] 逐行一致
     （entry_date 历史原式 = 面板日期 → 封装件按同映射回填）；
  3) net_ret/gross_ret ≤ 1.5e-4（sim_core 台账 px 列 4 位四舍五入的传播上限，文档化容差）。
通过 → output/v5/m13_parity_strategy_bt.md 落盘。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import strategy_bt

OUT = "output/v5"


def synth():
    # 交易日历: 2006-01-02 起 2.5 年工作日
    days = pd.date_range("2006-01-02", "2008-06-30", freq="B")
    bench = pd.Series(1000 * (1.0004 ** np.arange(len(days))), index=days)
    rng = np.random.default_rng(7)
    navs, pts = {}, []
    for c, drift in [("000001", 1.0006), ("000002", 1.0002), ("000003", 1.0008),
                     ("000004", 1.0001), ("000005", 0.9995)]:
        start = pd.Timestamp("2006-02-01") if c != "000003" else pd.Timestamp("2007-01-15")
        idx = days[days >= start]
        s = pd.Series(1.0 * (drift ** np.arange(len(idx))) * (1 + rng.normal(0, 0.004, len(idx)).cumsum() * 0.1),
                      index=idx)
        navs[c] = s.sort_index()
        pts.append(c)
    qdates = ["2006-03-31", "2006-06-30", "2006-09-30", "2006-12-31",  # 含 09-30(六)/12-31(日)
              "2007-03-31", "2007-06-29", "2007-09-28", "2007-12-31",
              "2008-03-31", "2008-06-30"]
    scores = {  # 设计: 000004 于 2007Q1 跌出面板; 000005 于 2007Q3 S=NaN; 高水位在 2007Q4
        "2006-03-31": [("000001", 80), ("000002", 72), ("000004", 60), ("000005", 40)],
        "2006-06-30": [("000001", 82), ("000002", 66), ("000004", 74), ("000005", 41)],
        "2006-09-30": [("000001", 85), ("000002", 48), ("000004", 76), ("000005", 60)],
        "2006-12-31": [("000001", 90), ("000002", 77), ("000004", 78), ("000005", 62)],
        "2007-03-31": [("000001", 88), ("000002", 74), ("000005", 66)],   # 000004 跌出
        "2007-06-29": [("000001", 86), ("000003", 91), ("000004", 75), ("000005", 64)],
        "2007-09-28": [("000001", 84), ("000002", 79), ("000003", 90), ("000005", np.nan)],
        "2007-12-31": [("000001", 89), ("000002", 81), ("000003", 93), ("000004", 76), ("000005", 63)],
        "2008-03-31": [("000001", 44), ("000002", 80), ("000003", 92), ("000004", 74)],
        "2008-06-30": [("000001", 83), ("000002", 82), ("000003", 94), ("000004", 77)],
    }
    water = {"2006-03-31": 0.3, "2006-06-30": 0.4, "2006-09-30": 0.5, "2006-12-31": 0.6,
             "2007-03-31": 0.6, "2007-06-29": 0.7, "2007-09-28": 0.8, "2007-12-31": 0.95,  # 触发 hi_water/gate
             "2008-03-31": 0.95, "2008-06-30": 0.5}
    rows = []
    for d, lst in scores.items():
        for c, s in lst:
            rows.append(dict(date=d, code=c, S=s, water=water[d]))
    panel = pd.DataFrame(rows)
    return panel, navs, bench


VARIANTS = [
    ("A0", dict(buy_th=70, sell_th=50)),
    ("A", dict(buy_th=70, sell_th=45)),
    ("D", dict(buy_th=70, sell_th=45, ma20_exit=True)),
    ("E", dict(buy_th=70, sell_th=45, trail_stop=0.15)),
    ("H", dict(buy_th=70, sell_th=45, hi_water=(0.90, 5), water_gate=0.92)),
]

TCOLS = ["code", "entry_date", "exit_date", "entry_S", "exit_S", "exit_reason", "hold_days"]


def compare(name, a_ec, a_tr, b_ec, b_tr, dmap):
    lines, ok = [], True
    da = float(np.abs(a_ec.equity.values - b_ec.equity.values).max())
    same_len = len(a_ec) == len(b_ec)
    lines.append(f"| {name} | ec 行数 {len(a_ec)} vs {len(b_ec)} | max\\|Δequity\\| = {da:.2e} |")
    if not same_len or da > 1e-8:
        ok = False
    ta, tb = a_tr.copy(), b_tr.copy()
    # 封装件 entry_date 回填为面板日期（历史原式口径）+ hold_days 同步重算
    ta["entry_date"] = ta["entry_date"].map(lambda d: dmap.get(d, d))
    ta["hold_days"] = (pd.to_datetime(ta.exit_date) - pd.to_datetime(ta.entry_date)).dt.days
    ta, tb = ta.sort_values(["exit_date", "code"]).reset_index(drop=True), \
        tb.sort_values(["exit_date", "code"]).reset_index(drop=True)
    if len(ta) != len(tb):
        lines.append(f"| {name} | 交易笔数 {len(ta)} vs {len(tb)} 不一致 |")
        return lines, False
    for col in TCOLS:
        sa, sb = ta[col], tb[col]
        eq = ((sa == sb) | (sa.isna() & sb.isna())).values   # NaN 视为相等（pandas3 astype(str) 保留缺失，不可用字符串比对）
        if not eq.all():
            bad = np.where(~eq)[0][:3]
            lines.append(f"| {name} | 台账列 {col} 不一致行: {bad.tolist()} |")
            ok = False
    for col in ["net_ret", "gross_ret"]:
        d = float(np.abs(ta[col].values - tb[col].values).max())
        if d > 1.5e-4:
            lines.append(f"| {name} | {col} maxΔ = {d:.2e} 超过容差 1.5e-4 |")
            ok = False
    return lines, ok


def main():
    panel, navs, bench = synth()
    days = bench.loc[pd.Timestamp("2006-03-31"):pd.Timestamp("2008-06-30")].index
    dmap = {}   # 交易日 → 面板日
    for d in sorted(panel.date.unique()):
        elig = days[days <= pd.Timestamp(d)]
        if len(elig):
            dmap[str(elig[-1].date())] = d
    all_ok, md = True, ["# M1.3 对拍报告：strategy_bt ↔ sim_core 薄封装（合成数据）", ""]
    for name, kw in VARIANTS:
        ec_new, tr_new, _ = strategy_bt.simulate(panel, navs, bench, label=name, **kw)
        ec_old, tr_old, _ = strategy_bt._simulate_legacy(panel, navs, bench, label=name, **kw)
        lines, ok = compare(name, ec_new, tr_new, ec_old, tr_old, dmap)
        # 注: compare 需要 交易日→面板日 映射 dmap（封装件 entry=交易日，回填至面板日口径）
        all_ok &= ok
        md += lines
    md += ["", f"**总判定：{'全绿 ✅（薄封装与旧实现合成对拍一致）' if all_ok else '存在分叉 ❌——禁止切换'}**", ""]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/m13_parity_strategy_bt.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
