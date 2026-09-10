#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #20（B2.1）：SURV-ADJ 基线重立 + 口径修复差异分解

口径标签：SURV-ADJ（幸存池 + 复权价 + 官方ret标签）。
本产物是对账基线，**禁作 FULL-PIT 裁决证据**；结论若与 FULL-PIT 冲突以 FULL-PIT 为准。

分解设计（旧口径 → 逐项修复，预登记顺序不得调整）：
  R0  旧口径封存档案：raw nav 成交 + T+0@close + flat 成本(0.15%/0.5%) + 价格基准对比
  R1  ＋复权成交：成交价改为 navadj（官方 ret 复权序列）
  R2  ＋T+1 执行：exec_delay_days=1（信号日次一交易日净值成交）
  R3  ＋阶梯赎回成本：cost_out_fn(hd<7d→1.5%; <365d→0.5%; <730d→0.25%; ≥730d→0)
      （行业通行档；公募强制 7 日惩罚档 1.5% 为监管要求，其余档位为预登记默认值）
  R4  ＋全收益基准：组合口径不变，仅在基准列用 H00300 TR 对比
      （若 cache/bench_tr_h00300.csv 不存在 → TR_PENDING 打标，基准列继续用价格指数，
        并在报告中明示"M1 价基基准高估 ~2pp/yr"的档案量级）

参数与历史基线一致：V3.8 执行层默认（slots=10、buy70/sell45、CPPI、危机、现金2.5%、
20% 移动止损、季度再平衡、本金 100k）。面板 = P1-0 幸存池重建产物（2014-03→2026-03）。

产物：output/v5/b21_decomposition.csv / b21_baseline_summary.md /
      b21_equity_<tag>.csv / b21_trades_<tag>.csv
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True

import sim_core as SC

PANEL_DIR = "output/p1_panel"
OUT = "output/v5"
NAV_RAW, NAV_ADJ = "cache/nav_%s.csv", "cache/navadj_%s.csv"
TR_CSV = "cache/bench_tr_h00300.csv"
START, END = "2014-03-31", "2026-03-31"

ARGS = dict(capital=100000.0, cost_in=0.0015, cost_out=0.005, slots=10,
            buy=70.0, sell=45.0, pool_mode="default", legacy=False,
            cash_yield=0.025, trail_stop=0.20, rebalance="quarterly",
            crisis=True, cppi=True, crisis_ma=200, crisis_vol_window=20,
            crisis_vol_q=0.80)


class LazyNavs:
    """按需加载净值序列（缓存文件懒读），接口与模拟器需要的 dict 一致。"""

    def __init__(self, pattern):
        self.pattern, self._d = pattern, {}

    def _load(self, code):
        fp = self.pattern % code
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp, parse_dates=["date"])
        col = "adj_nav" if "adj_nav" in df.columns else "nav"
        s = df.set_index("date")[col].sort_index()
        return s[~s.index.duplicated(keep="last")].dropna()

    def __contains__(self, code):
        if code not in self._d:
            self._d[code] = self._load(code)
        return self._d[code] is not None and len(self._d[code]) > 60

    def __getitem__(self, code):
        if code not in self._d:
            self._d[code] = self._load(code)
        return self._d[code]

    def get(self, code):
        return self[code] if code in self else None


def load_panel():
    files = sorted(f for f in os.listdir(PANEL_DIR)
                   if f.endswith(".csv") and f[0].isdigit())
    parts = []
    for f in files:
        d = f[:-4]
        if not (START <= d <= END):
            continue
        g = pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str},
                        usecols=["code", "S_total", "water", "R_MDD"])
        g = g.dropna(subset=["S_total"])
        g["date"] = d
        parts.append(g.rename(columns={"S_total": "S"}))
    return pd.concat(parts, ignore_index=True)


def ladder_fn(gross, hold_days):
    if hold_days < 7:
        return 0.015
    if hold_days < 365:
        return 0.005
    if hold_days < 730:
        return 0.0025
    return 0.0


def bench_series():
    df = pd.read_csv("cache/idx_sh000300.csv", parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def run_config(tag, navs, delay, cost_out_fn, panel, bench, keep_trades=True):
    args = types.SimpleNamespace(**ARGS)
    ec, tr = SC.simulate(panel, navs, bench, [str(d) for d in sorted(panel.date.unique())],
                         args, exec_delay_days=delay,
                         cost_out_fn=cost_out_fn, label=tag)
    yrs = (ec.index[-1] - ec.index[0]).days / 365.25
    tot = ec.equity.iloc[-1] / ARGS["capital"] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    dd = ec.drawdown.min()
    row = dict(config=tag, final=float(ec.equity.iloc[-1]), total_ret=tot, cagr=cagr,
               maxdd=dd, calmar=cagr / abs(dd) if dd else np.nan,
               n_trades=len(tr), cost=float(ec.attrs.get("total_cost", np.nan)))
    return row, ec, tr


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_panel()
    bench = bench_series()
    navs_raw, navs_adj = LazyNavs(NAV_RAW), LazyNavs(NAV_ADJ)
    print(f"[B2.1] 面板 {panel.date.nunique()} 月 × {len(panel)} 行; universe={len(set(panel.code))}",
          flush=True)

    tr_bench = pd.read_csv(TR_CSV, parse_dates=["date"]).set_index("date")["tr_close"] \
        if os.path.exists(TR_CSV) else None
    tr_status = "OK" if tr_bench is not None else "TR_PENDING(H00300 未构建,D0.3 用户侧待执行)"

    configs = [
        ("R0_legacy_rawnav_T0_flat_pxbench", navs_raw, 0, None),
        ("R1_+navadj", navs_adj, 0, None),
        ("R2_+T1", navs_adj, 1, None),
        ("R3_+ladder_cost", navs_adj, 1, ladder_fn),
    ]
    rows, curves = [], {}
    for tag, navs, delay, cof in configs:
        r, ec, tr = run_config(tag, navs, delay, cof, panel, bench)
        bb = bench.loc[ec.index[0]:ec.index[-1]]
        r["bench_px_ret"] = float(bb.iloc[-1] / bb.iloc[0] - 1)
        if tr_bench is not None:
            tb = tr_bench.loc[ec.index[0]:ec.index[-1]]
            r["bench_tr_ret"] = float(tb.iloc[-1] / tb.iloc[0] - 1)
        else:
            r["bench_tr_ret"] = np.nan
        r["excess_vs_px"] = r["total_ret"] - r["bench_px_ret"]
        r["excess_vs_tr"] = (r["total_ret"] - r["bench_tr_ret"]) if tr_bench is not None else np.nan
        rows.append(r)
        curves[tag] = ec
        ec.reset_index().to_csv(f"{OUT}/b21_equity_{tag}.csv", index=False, encoding="utf-8-sig")
        tr.to_csv(f"{OUT}/b21_trades_{tag}.csv", index=False, encoding="utf-8-sig")
        print(f"[B2.1] {tag:42s} CAGR={r['cagr']:+.2%} MaxDD={r['maxdd']:.1%} "
              f"trades={r['n_trades']} cost={r['cost']:,.0f}", flush=True)

    # R4: 组合口径不变（=R3 曲线），仅基准列换 TR
    r3 = [r for r in rows if r["config"] == "R3_+ladder_cost"][0].copy()
    r4 = dict(r3)
    r4["config"] = "R4_+TR_bench(仅基准列)"
    rows.append(r4)

    dec = pd.DataFrame(rows)
    for m in ["cagr", "maxdd", "calmar", "total_ret"]:
        dec[f"d_{m}"] = dec[m].diff()
    dec.to_csv(f"{OUT}/b21_decomposition.csv", index=False, encoding="utf-8-sig")

    L = ["# B2.1 SURV-ADJ 基线重立 + 口径修复差异分解", "",
         f"**口径标签: SURV-ADJ（禁作 FULL-PIT 裁决证据）**  |  TR 基准状态: {tr_status}", "",
         f"区间 {START} → {END}；V3.8 默认参数；面板=P1-0 幸存池重建（{panel.date.nunique()} 个月）。", "",
         dec.round(4).to_markdown(index=False), "",
         "阶梯赎回成本（预登记默认档）：<7d 1.5% / <365d 0.5% / <730d 0.25% / ≥730d 0；申购 0.15%。", ""]
    if tr_bench is None:
        L += ["⚠️ TR_PENDING：基准列仍为价格指数。档案结论（M1）：价基基准约高估超额 ~2pp/yr。",
              "H00300 本机执行 `python d03_build_tr.py` 并由对账闸门放行后自动生效。", ""]
    L.append("> 旧口径 R0 行仅为对账展示；历史 6.16% 口径自此封存为档案，不再引用。")
    open(f"{OUT}/b21_baseline_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[B2.1] 产物: {OUT}/b21_decomposition.csv, b21_baseline_summary.md")


if __name__ == "__main__":
    main()
