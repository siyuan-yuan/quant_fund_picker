#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预登记 #37(C1) / #38(C2) 执行器 —— 变体面板端到端（M11 后权威面板）

变体构造（仅面板层变换，不打分引擎）：
  A = 对照 = S_total 原样（= b21_baseline R3，已跑）；
  C1-B = MDD-惩罚剔除版：base 按 finalize.total 原式重算 →
         risk.apply_penalties(base, penalties 中除 mdd_smooth 项外全部保留)；
  C2   = V3 动量 M2：F_momentum := momentum_score_smooth_m2(rank4,rank7)（rank 用回面板固有列）→ total 重算。

判门（预登记 #37/#38，跑前冻结）：
  #37: H1 = OOS(2024-06+) fwd6 配对 IC 差(C1B−A) HAC t ≥ 2；
       H2 = 端到端 MaxDD 劣化 ≤ 2pp 且 ΔCalmar ≥ −0.05。
  #38: H1 = OOS fwd6 配对 IC 差(M2−M1) HAC t ≥ 2（ΔIC≥0）；
       H2 = 端到端 CAGR 差 ≥ 0 且 MaxDD 劣化 ≤ 2pp。
  双门独立，任一不过 → 维持现状。

自检门（先验，不达标当即 abort）：
  1. 2020-12 月按 finalize.total 原式复算 = S_total（max|Δ| ≤ 0.05）；
  2. C1-B ≥ A 按行成立（惩罚只能减分）；
产物：output/v5/c_experiments_{config,verdict}.csv；口径 SURV-ADJ。
"""
from __future__ import annotations

import ast
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

import risk
import factors
import sim_core as SC
from b21_baseline import (load_panel, ladder_fn, bench_series, LazyNavs,
                          NAV_ADJ, ARGS, run_config)
from stats_hac import nw_tstat

PANEL_DIR = "output/p1_panel"
OUT = "output/v5"
OOS_START = "2024-06-01"
LAGS = {"fwd6": 5}


def parse_penalties(s) -> list:
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        return v if isinstance(v, list) else []
    except Exception:
        return []


def reload_base(g: pd.DataFrame) -> pd.Series:
    """= finalize.total 的 base 部分：缺失因子按剩余腿归一化。"""
    fv = g["F_value"].where(g["F_value"].notna()).clip(upper=100) * g["w_value"]
    fa = g["F_alpha"].where(g["F_alpha"].notna()) * g["w_alpha"]
    fm = g["F_momentum"].where(g["F_momentum"].notna()) * g["w_mom"]
    num = fv.fillna(0) + fa.fillna(0) + fm.fillna(0)
    den = (g["F_value"].notna() * g["w_value"] + g["F_alpha"].notna() * g["w_alpha"]
           + g["F_momentum"].notna() * g["w_mom"])
    return pd.Series(np.where(den > 1e-9, num / den, 0.0), index=g.index)


def repenalize(g: pd.DataFrame, base: pd.Series, keep) -> pd.Series:
    """面板 penalties 列仅存数值链（名字在入库时剥离），逐值连乘。"""
    pens = [[p for p in parse_penalties(s) if keep(p)] for s in g["penalties"]]
    out = []
    for b, ps in zip(base, pens):
        s = b
        for p in ps:
            s *= (1 - float(p))
        out.append(round(max(0.0, min(100.0, s)), 1))
    return pd.Series(out, index=g.index).clip(0, 100)


def variant_no_mdd(g: pd.DataFrame) -> pd.Series:
    """剔除 MDD 平滑惩罚：用 (R_MDD, water) 精确重算该罚值 → 从数值链中删去一次严格首次出现。"""
    base = reload_base(g)
    out = []
    for b, pen, rm, w in zip(base, g["penalties"], g["R_MDD"], g["water"]):
        ps = [float(p) for p in parse_penalties(pen)]
        p_mdd = risk.mdd_smooth_penalty(rm if rm == rm else None, w if w == w else None)
        if p_mdd > 0:                       # 剔除首次严格相等的那个（可能与其他罚值同值，删谁无差）
            try:
                ps.remove(p_mdd)
            except ValueError:
                ps = [x for x in ps if abs(x - p_mdd) > 1e-12]
        s = b
        for p in ps:
            s *= (1 - p)
        out.append(round(max(0.0, min(100.0, s)), 1))
    return pd.Series(out, index=g.index).clip(0, 100)


def variant_m2(g: pd.DataFrame) -> pd.Series:
    fm = [factors.momentum_score_smooth_m2(a, b) for a, b in zip(g["rank4"], g["rank7"])]
    g2 = g.assign(F_momentum=fm)
    return repenalize(g2, reload_base(g2), lambda p: True)


def self_check():
    g = pd.read_csv(f"{PANEL_DIR}/2020-12-31.csv", dtype={"code": str})
    S_reb = repenalize(g, reload_base(g), lambda p: True)
    dev = (S_reb - g["S_total"]).abs()
    if float(dev.max()) > 0.05:
        raise SystemExit(f"自检门1 abort: 2020-12 复算 max|Δ|={dev.max():.4f} > 0.05 → 权重/惩罚复算口径不符, 停止执行")
    v = variant_no_mdd(g)
    if not (v >= g["S_total"] - 1e-9).all():
        raise SystemExit("自检门2 abort: C1-B 出现 < S_total")
    print(f"[c-exp] 自检通过: 2020-12 复算 max|Δ|={dev.max():.4f}; C1-B 方向 monotonic OK", flush=True)


FULL_COLS = ["code", "name", "ftype", "is_passive", "n_days", "val_pct", "val_cov",
             "valuation_blind", "trend_ok", "ma20_dist", "macd_dif", "wr", "dc",
             "F_value", "F_alpha", "R_MDD", "smallcap_exp", "top3_conc", "penalties",
             "water", "top1_style", "top1_w", "r4", "r7", "r3s1", "r3s0", "r4s0",
             "r4s2", "r7s2", "r12s1", "fwd1", "fwd3", "fwd6", "rank4", "rank7",
             "F_momentum", "w_value", "w_alpha", "w_mom", "weights_mode", "S_total",
             "rating", "date"]


def load_full_panel():
    parts = []
    for f in sorted(os.listdir(PANEL_DIR)):
        if not (f.endswith(".csv") and f[0].isdigit() and "2014-03-31" <= f[:-4] <= "2026-03-31"):
            continue
        g = pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str}, usecols=FULL_COLS)
        g = g.dropna(subset=["S_total"]).assign(date_col=f[:-4])
        if "date" not in g or g["date"].isna().all():
            g["date"] = f[:-4]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def paired_ic_diff(df, col_b, col_a="S_total", h="fwd6", scope="ALL"):
    from scipy import stats as st
    diffs = {}
    for m, g in df.groupby("date"):
        a, b = g[col_a].values.astype(float), g[col_b].values.astype(float)
        y = g[h].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        diffs[m] = st.spearmanr(b[ok], y[ok])[0] - st.spearmanr(a[ok], y[ok])[0]
    ser = pd.Series(diffs).sort_index()
    if len(ser) < 8:
        return dict(scope=scope, n=len(ser), mean=np.nan, t_naive=np.nan, t_hac=np.nan)
    lag = LAGS[h] if scope == "OOS" else max(int(round(len(ser) ** (1 / 3))), LAGS[h])
    band = min(lag, len(ser) // 3)
    r = nw_tstat(ser.values, lag=band)
    return dict(scope=scope, n=len(ser), mean=float(ser.mean()),
                t_naive=float(r["t_naive"]), t_hac=float(r["t_hac"]), bw=band)


def main():
    os.makedirs(OUT, exist_ok=True)
    self_check()
    df = load_full_panel()
    print(f"[c-exp] 面板 {df.date.nunique()} 月 × {len(df)} 行", flush=True)

    df["S_C1B"] = variant_no_mdd(df)
    df["S_C2"] = variant_m2(df)

    rows_ic = []
    for tag, col in [("C1_BminusA", "S_C1B"), ("C2_M2minusM1", "S_C2")]:
        rows_ic.append(dict(tag=tag, **paired_ic_diff(df, col, h="fwd6", scope="ALL")))
        rows_ic.append(dict(tag=tag, **paired_ic_diff(df[df.date.astype(str) >= OOS_START],
                                                      col, h="fwd6", scope="OOS")))
    ic_df = pd.DataFrame(rows_ic)
    ic_df.to_csv(f"{OUT}/c_experiments_pair_ic.csv", index=False, encoding="utf-8-sig")
    print(ic_df.to_string(index=False), flush=True)

    bench = bench_series()
    navs = LazyNavs(NAV_ADJ)
    rows_bt = {}
    for tag, col in [("A_ref(=R3)", "S_total"), ("C1B_noMDD", "S_C1B"), ("C2_M2", "S_C2")]:
        panel = df[["date", "code", col, "water", "R_MDD"]].rename(columns={col: "S"}).copy()
        panel["water"] = df["water"]
        r, ec, tr = run_config(tag, navs, 1, ladder_fn, panel, bench)
        rows_bt[tag] = r
        print(f"[c-exp] BT {tag}: CAGR={r['cagr']:+.2%} MaxDD={r['maxdd']:.1%} trades={r['n_trades']}",
              flush=True)
    bt_df = pd.DataFrame(list(rows_bt.values()))
    bt_df.to_csv(f"{OUT}/c_experiments_e2e.csv", index=False, encoding="utf-8-sig")

    A = rows_bt["A_ref(=R3)"]
    verdicts = []
    for tag, ic_tag, mcol in [("C1(#37)", "C1_BminusA", "C1B_noMDD"),
                              ("C2(#38)", "C2_M2minusM1", "C2_M2")]:
        o = ic_df[(ic_df.tag == ic_tag) & (ic_df.scope == "OOS")].iloc[0]
        B = rows_bt[mcol]
        dcagr = B["cagr"] - A["cagr"]
        dmaxdd_pp = (B["maxdd"] - A["maxdd"]) * 100
        dcal = B["calmar"] - A["calmar"] if A.get("calmar") and not pd.isna(A["calmar"]) else np.nan
        if tag.startswith("C1"):
            h1, h2 = o.t_hac >= 2, (dmaxdd_pp <= 2.0) and (dcal >= -0.05)
        else:
            h1, h2 = o.t_hac >= 2, (dcagr >= 0) and (dmaxdd_pp <= 2.0)
        verdicts.append(dict(rule=tag, h1_t_hac=round(float(o.t_hac), 2), h1_pass=bool(h1),
                             dcagr_pp=round(dcagr * 100, 2), dmaxdd_pp=round(dmaxdd_pp, 2),
                             dcalmar=round(float(dcal), 3) if dcal == dcal else None,
                             h2_pass=bool(h2),
                             verdict=("H1+H2 双门过 → 候选改进成立(SURV-ADJ 通道, 待 FULL-PIT 终裁)"
                                      if (h1 and h2)
                                      else "未过双门 → 维持现状(预登记否决条款生效)")))
        print(f"[{tag}] H1 t_hac={o.t_hac:.2f}({'过' if h1 else '未过'}) "
              f"H2 ΔCAGR={dcagr * 100:+.2f}pp ΔMaxDD={dmaxdd_pp:+.2f}pp({'过' if h2 else '未过'}) "
              f"→ {verdicts[-1]['verdict']}", flush=True)
    pd.DataFrame(verdicts).to_csv(f"{OUT}/c_experiments_verdict.csv", index=False, encoding="utf-8-sig")
    print("[c-experiments done] 产物: c_experiments_pair_ic.csv / c_experiments_e2e.csv / c_experiments_verdict.csv", flush=True)


if __name__ == "__main__":
    main()
