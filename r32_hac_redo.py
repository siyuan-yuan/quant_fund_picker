#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #23（R3.2）：P1-2/P1-3/P1-4 历史裁决的 HAC 复核

复核对象（历史报告中的三条裁决）：
  P1-2  秩空间正交化否决（原配对差 t_naive = −3.05, fwd6, S1_正交_原权 − S0）
  P1-3  动量窗口维持（3 个备选 M1 组合 vs base r4r7 的 fwd6 配对差）
  P1-4  MDD 乘法惩罚保留（S_without_mdd − S_total 配对差，原 t_naive = −2.16）
        + 集中度惩罚消融配对差 + 3 个惩罚因子的增量 IC

协议：只重算统计。序列生成与 p1_analysis 原文同式（取数/滤月/秩/rank 合成逐一复现），
差分项一律对同一月份集合做配对差后再上 nw_tstat。翻转判定 |t|≥2。
产物：output/v5/r32_*.csv / r32_summary.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

import provider
provider.STALE_OK = True

import p1_analysis as P1
from stats_hac import series_judgement, HORIZON_LAGS

OUT = "output/v5"
LM = {"fwd1": HORIZON_LAGS["fwd1_monthly"], "fwd3": HORIZON_LAGS["fwd3_monthly"],
      "fwd6": HORIZON_LAGS["fwd6_monthly"]}


def monthly_ic(d, score_col, h, dirn=+1):
    out = {}
    for m, g in d.groupby("date"):
        v = P1._ic_row(g, score_col, h, dirn)
        if v == v:
            out[m] = v
    return pd.Series(out).sort_index()


def pair_t(ic_a: pd.Series, ic_b: pd.Series, h: str, name: str, extra=None):
    idx = ic_a.index.intersection(ic_b.index)
    dser = (ic_a.loc[idx] - ic_b.loc[idx]).dropna()
    row = dict(item=name, horizon=h, **series_judgement(dser.values, LM[h]))
    if extra:
        row.update(extra)
    return row


def redo_p12(df, min_n=30, regen=True):
    fp = os.path.join(P1.OUT_DIR, "f1_2_scores.csv")
    if regen or not os.path.exists(fp):
        print("[R3.2/P1-2] 重建 f1_2_scores.csv（p1_2 原式）…", flush=True)
        P1.p1_2(df, min_n)
    long_df = pd.read_csv(fp, dtype={"code": str}, parse_dates=["date"])
    rows = []
    variants = ["S1_正交_原权", "S2_正交_IC权", "S3_原始_IC权"]
    for h in ["fwd1", "fwd3", "fwd6"]:
        ic = {}
        for v in ["S0"] + variants:
            sub = long_df[long_df["variant"] == v]
            ic[v] = monthly_ic(sub.rename(columns={"S": "S_variant"}), "S_variant", h)
        for v in variants:
            rows.append(pair_t(ic[v], ic["S0"], h, f"P12_{v}-S0", dict(subwindow="full")))
        cut = pd.Timestamp("2023-04-01")
        for v in variants:
            rows.append(pair_t(ic[v][ic[v].index >= cut], ic["S0"][ic["S0"].index >= cut],
                               h, f"P12_{v}-S0", dict(subwindow="2023+")))
    return pd.DataFrame(rows)


def redo_p13(df, min_n=30):
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()

    def pair_score(g, wa, wb):
        ra = pd.to_numeric(g[wa], errors="coerce").rank(pct=True)
        rb = pd.to_numeric(g[wb], errors="coerce").rank(pct=True)
        return (100 * (0.6 * ra + 0.4 * rb).clip(0, 1)).values

    rows = []
    for name, a, b in P1.MOM_PAIRS[1:]:           # alts vs base r4r7
        diffs = {}
        for m, g in d.groupby("date"):
            s_b = pair_score(g, "r4", "r7")
            s_a = pair_score(g, a, b)
            y = g["fwd6"].values.astype(float)
            ok = np.isfinite(s_b) & np.isfinite(s_a) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            diffs[m] = stats.spearmanr(s_a[ok], y[ok])[0] - stats.spearmanr(s_b[ok], y[ok])[0]
        ser = pd.Series(diffs).sort_index()
        rows.append(dict(item=f"P13_pair_diff_{name}", horizon="fwd6",
                         **series_judgement(ser.values, LM["fwd6"])))
    return pd.DataFrame(rows)


def redo_p14(df, min_n=30):
    import risk as R
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    d["p_mdd"] = [R.mdd_smooth_penalty(rm, w) if rm == rm else 0.0
                  for rm, w in zip(d["R_MDD"], d["water"])]
    d["S_without_mdd"] = np.where(d["p_mdd"] > 0,
                                  np.clip(d["S_total"] / (1 - d["p_mdd"].clip(0, 0.99)), 0, 100),
                                  d["S_total"])
    d["p_conc"] = np.where((~d["is_passive"]) & (d["top3_conc"] > 0.70)
                           & (d["wr"] < 0.5) & d["top3_conc"].notna() & d["wr"].notna(), 0.4, 0.0)
    d["S_without_conc"] = np.where(d["p_conc"] > 0,
                                   np.clip(d["S_total"] / (1 - d["p_conc"]), 0, 100),
                                   d["S_total"])
    ic_s = monthly_ic(d, "S_total", "fwd6")
    ic_w = monthly_ic(d, "S_without_mdd", "fwd6")
    ic_c = monthly_ic(d, "S_without_conc", "fwd6")
    rows = [pair_t(ic_w, ic_s, "fwd6", "P14_diff_SwithoutMdd-S"),
            pair_t(ic_c, ic_s, "fwd6", "P14_diff_SwithoutConc-S")]
    for fname in ["risk_mdd", "risk_conc", "risk_small"]:
        col, dirn = P1.FACTORS[fname]
        ic = {}
        for m, g in d.groupby("date"):
            x = (g[col].values * dirn).astype(float)
            s = g["S_total"].values.astype(float)
            y = g["fwd6"].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(s) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            res_s = P1._ols_resid(y[ok], np.column_stack([np.ones(ok.sum()), s[ok]]))
            xr = x[ok]
            if np.std(xr) > 1e-12 and np.std(res_s) > 1e-12:
                ic[m] = stats.spearmanr(xr, res_s)[0]
        ser = pd.Series(ic).sort_index().dropna()
        rows.append(dict(item=f"P14_incremental_{fname}", horizon="fwd6",
                         **series_judgement(ser.values, LM["fwd6"])))
    return pd.DataFrame(rows)


def main(min_n=30, no_regen=False):
    os.makedirs(OUT, exist_ok=True)
    df = P1.load_panel(min_n=min_n)
    t12 = redo_p12(df, min_n, regen=not no_regen)
    t12.to_csv(f"{OUT}/r32_p12_orth_hac.csv", index=False, encoding="utf-8-sig")
    t13 = redo_p13(df, min_n)
    t13.to_csv(f"{OUT}/r32_p13_momwin_hac.csv", index=False, encoding="utf-8-sig")
    t14 = redo_p14(df, min_n)
    t14.to_csv(f"{OUT}/r32_p14_penalty_hac.csv", index=False, encoding="utf-8-sig")

    allr = pd.concat([t12, t13, t14], ignore_index=True)
    flips = allr[allr.flip]
    L = ["# R3.2 P1-2/P1-3/P1-4 裁决 HAC 复核（SURV-ADJ，统计-only）", "",
         f"共复核 {len(allr)} 条统计项；翻转（naive vs HAC 显著性门 |t|≥2 跨界）{len(flips)} 条。", ""]
    for tag, t in [("P1-2 正交化否决", t12), ("P1-3 动量窗口维持", t13), ("P1-4 惩罚消融", t14)]:
        L.append(f"## {tag}")
        L.append(t.to_markdown(index=False))
        L.append("")
    if len(flips):
        L.append("## 翻转清单（须按 §0.3 翻案规则处置，历史报告不改原文）")
        L.append(flips.to_markdown(index=False))
    open(f"{OUT}/r32_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[R3.2] 翻转 {len(flips)}/{len(allr)}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(allr.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--no-regen", action="store_true")
    a = ap.parse_args()
    main(a.min_n, a.no_regen)
