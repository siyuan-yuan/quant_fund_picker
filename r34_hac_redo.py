#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #23（R3.4）：V3.6/V3.7 平滑化采纳证据链 HAC 复核

复核对象：factor_study 变体裁决（季度面板 × fwd6）。历史方式 = 各变体 IC 各自
t 值互比（§0.3 已作废）；本脚本统一为 **配对差 HAC t**（季度 × fwd6 → L=2）。

采纳链逐项（对 V0现行 的配对差）：
  生产V3.7(S_eng)(=fv_old+fa_new+fm_m1, 原组合 naive t=3.46)、
  V1-alpha平滑(原 t=3.42)、V2-mom平滑M1、V3-mom平滑M2、S4-value平滑、V5-全平滑
判定规则（预登记）：配对差 HAC |t|≥2 且方向与历史裁决一致 → 维持；显著反向 → 翻转；
不显著 → 证据不足标记（不擅自改判生产，留 A5 动议）。

产物：output/v5/r34_adoption_chain_hac.csv / r34_summary.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from stats_hac import series_judgement, HORIZON_LAGS

OUT = "output/v5"
LQ = HORIZON_LAGS["fwd6_quarterly"]      # 季度 × fwd6 → L=2
ROWS = "output/factor_variant_rows.csv"

VARIANTS = [("V1", "V1-alpha平滑"), ("V2", "V2-mom平滑M1"), ("V3", "V3-mom平滑M2"),
            ("S4", "S4-value平滑"), ("V5", "V5-全平滑")]
DECISION = {"V1": "采纳(V3.7, 原naive t=3.42)", "V2": "采纳(M1)",
            "V3": "未进生产（历史无显式否决记录）",
            "S4": "否决(原t=3.02<3.20,各自t互比口径已作废)",
            "V5": "参考",
            "S_eng": "生产 V3.7 组合(=fv_old+fa_new+fm_m1, 原组合 naive t=3.46)"}


def q_ic(df, scol, h="fwd6"):
    dd = df.dropna(subset=[h, scol])
    rs = dd.groupby("date").apply(
        lambda g: stats.spearmanr(g[scol], g[h])[0], include_groups=False).dropna()
    return rs


def main():
    if not os.path.exists(ROWS):
        print(f"❌ 缺 {ROWS}；先跑 factor_study.py 重建变体面板。")
        sys.exit(2)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(ROWS, dtype={"code": str}, parse_dates=["date"])
    df = df.dropna(subset=["S_eng"])
    print(f"[R3.4] 变体面板 {len(df)} 行 × {df.date.nunique()} 季")

    ic0 = q_ic(df, "Sx_V0")
    ic_prod = q_ic(df, "S_eng")
    base = series_judgement(ic0.values, LQ)
    rows = [dict(item="V0现行", horizon="fwd6", kind="ic 自身", n_quarters=base["n_months"],
                 ic_mean=base["ic_mean"], t_naive=base["t_naive"], t_hac=base["t_hac"],
                 sig_hac=base["sig_hac"], verdict="")]
    ownp = series_judgement(ic_prod.values, LQ)
    pj0 = series_judgement((ic_prod - ic0.reindex(ic_prod.index)).dropna().values, LQ)
    rows.append(dict(item="生产V3.7(S_eng) 自身", horizon="fwd6", kind="ic 自身",
                     n_quarters=ownp["n_months"], ic_mean=ownp["ic_mean"],
                     t_naive=ownp["t_naive"], t_hac=ownp["t_hac"],
                     sig_hac=ownp["sig_hac"], verdict=DECISION["S_eng"]))
    rows.append(dict(item="生产V3.7(S_eng) − V0", horizon="fwd6", kind="配对差",
                     n_quarters=pj0["n_months"], ic_mean=pj0["ic_mean"],
                     t_naive=pj0["t_naive"], t_hac=pj0["t_hac"],
                     sig_hac=pj0["sig_hac"], verdict=DECISION["S_eng"], flip=pj0["flip"]))
    for tag, name in VARIANTS:
        icv = q_ic(df, f"Sx_{tag}")
        own = series_judgement(icv.values, LQ)
        dser = (icv - ic0.reindex(icv.index)).dropna()
        pj = series_judgement(dser.values, LQ)
        rows.append(dict(item=f"{name} 自身", horizon="fwd6", kind="ic 自身",
                         n_quarters=own["n_months"], ic_mean=own["ic_mean"],
                         t_naive=own["t_naive"], t_hac=own["t_hac"],
                         sig_hac=own["sig_hac"], verdict=DECISION[tag]))
        rows.append(dict(item=f"{name} − V0", horizon="fwd6", kind="配对差",
                         n_quarters=pj["n_months"], ic_mean=pj["ic_mean"],
                         t_naive=pj["t_naive"], t_hac=pj["t_hac"],
                         sig_hac=pj["sig_hac"], verdict=DECISION[tag],
                         flip=pj["flip"]))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{OUT}/r34_adoption_chain_hac.csv", index=False, encoding="utf-8-sig")

    pdser = tbl[tbl.kind == "配对差"]
    L = ["# R3.4 V3.6/V3.7 采纳证据链 HAC 复核（季度 × fwd6，L=2 Bartlett NW）", "",
         "判定规则（预登记 §0.3）：变体间一律**配对差** HAC t；|t|≥2 为显著。",
         "历史裁决用的『各自 t 互比』口径作废；本表为唯一有效复核账。", "",
         tbl.to_markdown(index=False), "",
         "## 处置预案（不自动改生产）",
         "- 配对差显著且方向=正向 → 历史采纳维持为『成立』；",
         "- M2 注记：历史仅『未采纳』而无显式否决记录；本次量化=相对 V0 有显著增量，"
         "但未与 M1 直接配对；若未来动议比较 M1/M2，须按 M2−M1 配对差另裁。",
         "- 配对差不显著 → 历史采纳降级为『证据不足』，入 A5 复核动议；",
         "- 配对差显著反向 → 翻转条目入勘误。",
         "> 口径：SURV-ADJ（幸存池+今日名录类型）；重建池覆盖以 factor_study 日志为准，已披露。"]
    open(f"{OUT}/r34_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
