#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #17（M1.2）余项：HP4A-3 配对段 与 rbsa_ew 配对差的 HAC 复核

输入（均由刚刷新的确定性管线产出，严禁复用旧产物）：
  - output/p4/hp4a_pit_monthly_ic.csv  （p4_alpha_decay.py 重跑产物）
  - output/rbsa_ew_verdict.csv         （rbsa_ew_study.py 重跑产物）
口径：fwd6 月度 L=5（§0.3）；判定门 |t|≥2；双向假设，不预设方向。
历史锚点：HP4A-3 OOS 配对差 t=2.50（24月，naive）；rbsa_ew 配对差 t=…（运行后登记）。
产物：output/v5/m12b_hac_remainder.csv / m12b_summary.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stats_hac import nw_tstat

OUT = "output/v5"
GATE = 2.0


def pair_hac(a: pd.Series, b: pd.Series, L: int):
    diff = (a - b).dropna()
    n = len(diff)
    if n < max(3 * L, 8) or diff.std(ddof=1) == 0:
        return dict(n=n, mean=float(diff.mean()) if n else np.nan,
                    t_naive=np.nan, t_hac=np.nan, sig=np.nan)
    r = nw_tstat(diff.values, lag=L)
    t_n, t_h = float(r["t_naive"]), float(r["t_hac"])
    return dict(n=n, mean=float(diff.mean()), t_naive=t_n, t_hac=t_h,
                sig=bool(abs(t_h) >= GATE))


def main():
    rows = []

    # ---- HP4A-3: S3(PIT重加权) − S0(基线) 配对差, 全期 + OOS(2024+) ----
    fp = "output/p4/hp4a_pit_monthly_ic.csv"
    assert os.path.exists(fp), "缺 output/p4/hp4a_pit_monthly_ic.csv：先跑 p4_alpha_decay.py"
    wide = pd.read_csv(fp, index_col=0, parse_dates=True)   # 宽表: date × {S0_基线, 面板S_total, S3_PIT重加权, S3r_36m滚动}
    assert {"S0_基线", "S3_PIT重加权"}.issubset(wide.columns), f"variant 列不符: {list(wide.columns)}"
    for name, sub in [("HP4A3_全期_S3-S0", wide),
                      ("HP4A3_OOS2024+_S3-S0", wide[wide.index >= "2024-01-01"])]:
        r = pair_hac(sub["S3_PIT重加权"], sub["S0_基线"], L=5)
        rows.append(dict(item=name, kind="pair_diff_ic", hist_t=2.50 if "OOS" in name else -0.61,
                         **r))
    if "S3r_36m滚动" in wide.columns:
        for name, sub in [("HP4A3_OOS2024+_S3r-S0", wide[wide.index >= "2024-01-01"])]:
            r = pair_hac(sub["S3r_36m滚动"], sub["S0_基线"], L=5)
            rows.append(dict(item=name, kind="pair_diff_ic", hist_t=3.03, **r))

    # ---- rbsa_ew: EW 与 baseline 配对 ----
    fp2 = "output/rbsa_ew_verdict.csv"
    assert os.path.exists(fp2), "缺 output/rbsa_ew_verdict.csv：先跑 rbsa_ew_study.py"
    v = pd.read_csv(fp2)
    r = pair_hac(v["ic_ew"], v["ic_base"], L=5)
    rows.append(dict(item="rbsa_ew_vs_base", kind="pair_diff_ic", hist_t=np.nan, **r))

    df = pd.DataFrame(rows)

    def _flip(x):
        if pd.isna(x.t_hac):
            return False
        sig_now = abs(x.t_hac) >= GATE
        if pd.isna(x.hist_t):
            return False                     # 历史未登记 t → 只报现行口径，不判翻转
        return (abs(x.hist_t) >= GATE) != sig_now

    df["flip"] = df.apply(_flip, axis=1)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/m12b_hac_remainder.csv", index=False)
    md = ["# M1.2 余项 HAC 复核：HP4A-3 / rbsa_ew 配对段", "",
          "| item | 窗 | n | 差均值 | t_naive | t_HAC | HAC显著(\|t\|≥2) | 历史t(naive) | 翻转 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, x in df.iterrows():
        item, win = x["item"], "_".join(x["item"].split("_")[1:-1]) or "全期"
        md.append(f"| {item} | {win} | {x.n} | "
                  f"{x['mean']:+.4f} | {x.t_naive:.2f} | {x.t_hac:.2f} | {x.sig} | "
                  f"{x.hist_t if pd.notna(x.hist_t) else '未登记'} | {x.flip} |")
    md += ["", "判定规则：§0.3（fwd6 月度 L=5，NW-Bartlett，双向门 |t|≥2）；",
           "HP4A-4 终局裁决为模型级双门（独立于本中间门），即便中间门翻转终局亦不变（审计报告 §136 已登记）。"]
    with open(f"{OUT}/m12b_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
