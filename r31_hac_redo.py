#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #23（R3.1）：P1 因子体检 HAC 重判

只重算统计、不改模型（协议原文）。IC 序列生成逐字复用 p1_analysis.IC 机制
（同一 _ic_row/load_panel/min_n=30 口径），仅将 t 值由朴素公式换为
stats_hac.nw_tstat（Bartlett, L 按 §0.3：fwd6→5、fwd3→3、fwd1→1 月度）。

翻转判定（§0.3）：显著性门 |t| ≥ 2。naive-side 与 HAC-side 显著性结论不一致 → 翻转，
翻案追加段自动写入 r31_summary.md（历史报告原文不改）。

产物：output/v5/r31_factor_ic_hac.csv / r31_subgroup_ic_hac.csv / r31_summary.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True

from stats_hac import nw_tstat, series_judgement, HORIZON_LAGS
import p1_analysis as P1   # 复用 FACTORS/_ic_row/load_panel，保证 IC 序列一字不差

LAG = {"fwd1": HORIZON_LAGS["fwd1_monthly"], "fwd3": HORIZON_LAGS["fwd3_monthly"],
       "fwd6": HORIZON_LAGS["fwd6_monthly"]}
OUT = "output/v5"


def ic_series(d, col, h, dirn):
    ic = [v for v in d.groupby("date").apply(
        lambda g: P1._ic_row(g, col, h, dirn), include_groups=False).values if v == v]
    ic = np.array(ic, dtype=float)
    return ic[ic == ic]


def judge(ic, L):
    return series_judgement(ic, L)


def main(min_n: int = 30):
    os.makedirs(OUT, exist_ok=True)
    df = P1.load_panel(min_n=min_n)
    d = df[df.groupby("date")["code"].transform("size") >= min_n]
    print(f"[R3.1] 面板月份 {d.date.nunique()}，行 {len(d)}（min_n={min_n}，与 P1-1 同口径）")

    # ---- 全样本 13 因子 × 3 horizon ----
    rows = []
    for fname, (col, dirn) in P1.FACTORS.items():
        if col not in d.columns:
            continue
        for h in P1.HORIZONS:
            ic = ic_series(d, col, h, dirn)
            rows.append(dict(factor=fname, horizon=h, **judge(ic, LAG[h])))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{OUT}/r31_factor_ic_hac.csv", index=False, encoding="utf-8-sig")

    # ---- 5 子群 × (CORE6+eng_S) × fwd6（P1-1 分组定义原样复用） ----
    d2 = d.copy()
    d2["is_ov"] = d2["top1_style"].isin(
        {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}).fillna(False)
    srows = []
    for tag, sub in [("all", d2), ("active", d2[~d2["is_passive"]]),
                     ("passive", d2[d2["is_passive"]]),
                     ("ashare", d2[~d2["is_ov"]]), ("overseas", d2[d2["is_ov"]])]:
        for fname in P1.CORE6 + ["eng_S"]:
            col, dirn = P1.FACTORS[fname]
            ic = ic_series(sub, col, "fwd6", dirn)
            if len(ic) > 3:
                srows.append(dict(group=tag, factor=fname, **judge(ic, LAG["fwd6"])))
    stbl = pd.DataFrame(srows)
    stbl.to_csv(f"{OUT}/r31_subgroup_ic_hac.csv", index=False, encoding="utf-8-sig")

    # ---- 翻转清单与摘要 ----
    flips = tbl[tbl.flip].copy()
    sflips = stbl[stbl.flip].copy() if len(stbl) else pd.DataFrame()
    L = ["# R3.1 P1 因子体检 HAC 复核（统计-only，SURV-ADJ 面板口径不变）",
         "",
         "判定规则（预登记 §0.3）：显著性门 |t|≥2；L: fwd6→5, fwd3→3, fwd1→1（月度 Bartlett NW）。",
         f"月份数区间 n∈[{tbl.n_months.min()}, {tbl.n_months.max()}]。",
         "",
         f"## 全样本（13 因子×3 horizon = {len(tbl)} 行）：翻转 {len(flips)} 行",
         ""]
    if len(flips):
        L.append(flips[["factor", "horizon", "ic_mean", "t_naive", "t_hac",
                        "sig_naive", "sig_hac"]].to_markdown(index=False))
    else:
        L.append("（无翻转）")
    L += ["", f"## 子群（fwd6，{len(stbl)} 行）：翻转 {len(sflips)} 行", ""]
    if len(sflips):
        L.append(sflips[["group", "factor", "ic_mean", "t_naive", "t_hac"]].to_markdown(index=False))
    else:
        L.append("（无翻转）")
    L += ["",
          "## 边界项点名（历史裁决上下沿 ±0.5 内者必须人工过目）",
          ""]
    edge = tbl[(tbl.t_hac.abs() - 2.0).abs() < 0.5]
    if len(edge):
        L.append(edge[["factor", "horizon", "ic_mean", "t_naive", "t_hac",
                       "sig_hac"]].to_markdown(index=False))
    L += ["", "> 口径标签：SURV-ADJ（幸存池+官方ret标签）。本重判不得作为 FULL-PIT 裁决证据。"]
    open(f"{OUT}/r31_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print("[R3.1] 已写 r31_factor_ic_hac.csv / r31_subgroup_ic_hac.csv / r31_summary.md")
    print(f"[R3.1] 全样本翻转 {len(flips)} / {len(tbl)}；子群翻转 {len(sflips)} / {len(stbl)}")
    if len(flips):
        print(flips[["factor", "horizon", "t_naive", "t_hac"]].to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()
    main(args.min_n)
