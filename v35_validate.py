# -*- coding: utf-8 -*-
"""V3.5(条款2+5) 变体回测: 28季面板, 采纳参数重算 vs V3.4现行"""
import numpy as np
import pandas as pd
from scipy import stats

rows = pd.read_csv("output/veto_analysis_rows.csv", dtype={"code": str})
rows["pct6"] = rows.groupby("date")["fwd6"].rank(pct=True)

def f35(r):
    rm, w = r.R_MDD, r.water
    if pd.isna(rm): return 1.0
    if rm > 1.5:
        return 0.5 if (pd.notna(w) and w <= 0.35) else 0.0
    if rm > 1.2: return 0.75
    return 1.0

rows["S34"] = rows.S
rows["S35"] = (rows.base * rows.other_pen * rows.apply(f35, axis=1)).clip(0, 100).round(1)
rows["rank35"] = rows.groupby("date")["S35"].rank(pct=True)

def sig(d, lab):
    d = d.dropna(subset=["fwd6"])
    if not len(d):
        print(f"  {lab}: 无"); return dict()
    print(f"  {lab}: n={len(d)} fwd6{d.fwd6.mean():+.1%} 胜率{(d.fwd6>0).mean():.0%} "
          f"分位{d.pct6.mean():.3f} fwd12{d.fwd12.mean():+.1%}")
    return dict(n=len(d), fwd6=d.fwd6.mean(), win=(d.fwd6 > 0).mean(), pct=d.pct6.mean())

print("===== Buy线(S≥70) =====")
b34 = rows[rows.S34 >= 70]; b35 = rows[rows.S35 >= 70]
sig(b34, "V3.4现行")
sig(b35, "V3.5新规")
sig(rows[(rows.S35 >= 70) & (rows.S34 < 70)], "  V3.5新增")
sig(rows[(rows.S34 >= 70) & (rows.S35 < 70)], "  V3.5移除")

print("\n===== ⚔️动量战术 =====")
m34 = (rows.F_mom >= 100) & (rows.rank_pct >= 0.95) & (rows.S34 >= 50)
m35 = (rows.F_mom >= 100) & (rows.rank35 >= 0.95) & (rows.S35 >= 50)
sig(rows[m34], "V3.4现行")
sig(rows[m35], "V3.5新规")
sig(rows[m35 & ~m34], "  V3.5新增")
sig(rows[m34 & ~m35], "  V3.5移除")

print("\n===== 条款5专属: 低水位否决降级区样本 =====")
lw = rows[(rows.R_MDD > 1.5) & (rows.water <= 0.35)]
sig(lw, "水位≤35%&R_MDD>1.5 全体(原-100%→现-50%)")
sig(lw[lw.S35 >= 50], "  其中降级后S≥50")
sig(lw[lw.S35 >= 70], "  其中降级后S≥70")

print("\n===== 全局面相: S 与 fwd6 截面分位的 Spearman 相关 =====")
for c, lab in [("S34", "V3.4"), ("S35", "V3.5")]:
    d = rows.dropna(subset=["fwd6", c])
    rho, p = stats.spearmanr(d[c], d.fwd6)
    # 按日期分组求相关再取均值时点t (更像wf口径)
    rs = d.groupby("date").apply(lambda g: stats.spearmanr(g[c], g.fwd6)[0]).dropna()
    t = rs.mean() / (rs.std() / np.sqrt(len(rs)))
    print(f"  {lab}: 全体ρ={rho:+.4f} | 分期均值{rs.mean():+.4f} t={t:.2f} (n期={len(rs)})")

# 保存 V3.5 背书摘要
with open("output/v35_wf_verdict.md", "w", encoding="utf-8") as f:
    f.write("# V3.5 变体回测背书 (条款2+5, 28季×217池, n=%d)\n\n" % len(rows))
    f.write("见终端输出与 veto_analysis_rows.csv\n")
print("\n[saved] output/v35_wf_verdict.md")
