# -*- coding: utf-8 -*-
"""用已落盘的轨迹CSV重画 008528 图 (CJK字体修复)"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import provider
provider.STALE_OK = True

tdf = pd.read_csv("output/case_008528_trajectory.csv")
SIG_COLOR = {"🟢价值重仓": "#2ebd85", "🟢价值建仓": "#7fd6a8", "⚔️动量战术": "#f0b90b",
             "🟡分批观望": "#c9a227", "⚪回避": "#8b94a8", "🚫否决池": "#f6465d"}

nav = provider.get_fund_nav("008528").set_index("date")
seg = nav.loc["2024-07-01":"2026-07-31"]
base = seg["nav"].iloc[0]
trough_idx = seg["nav"].idxmin()
after = seg.loc[trough_idx:]
peak_idx = after["nav"].idxmax()
mult = seg["nav"].loc[peak_idx] / seg["nav"].loc[trough_idx]

fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1]})
ax = axes[0]
ax.plot(seg.index, seg["nav"] / base, color="#e8b64c", lw=1.8, label="008528 净值（2024-07=1）")
ax.scatter([trough_idx], [seg["nav"].loc[trough_idx] / base], color="#f6465d", zorder=5)
ax.scatter([peak_idx], [seg["nav"].loc[peak_idx] / base], color="#2ebd85", zorder=5)
ax.annotate(f"低点 {trough_idx.date()}", (trough_idx, seg["nav"].loc[trough_idx] / base),
            textcoords="offset points", xytext=(8, -14), fontsize=8.5, color="#f6465d")
ax.annotate(f"×{mult:.1f} {peak_idx.date()}", (peak_idx, seg["nav"].loc[peak_idx] / base),
            textcoords="offset points", xytext=(-95, 6), fontsize=8.5, color="#2ebd85")
ax.axvline(pd.Timestamp("2025-08-29"), color="#4c9aff", ls="--", lw=1)
ax.text(pd.Timestamp("2025-09-03"), ax.get_ylim()[0] * 1.1, "一年前 (2025-08)", fontsize=8.5, color="#4c9aff")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("008528 华泰柏瑞质量成长A — 净值 vs V3.5 打分/信号轨迹（低点起 ×6.9，全程 0 信号）", fontsize=11)

ax2 = axes[1]
x = pd.to_datetime(tdf["date"])
ax2.plot(x, tdf["S"], color="#4c9aff", lw=1.4, zorder=1)
for _, r in tdf.dropna(subset=["S"]).iterrows():
    c = SIG_COLOR.get(r["signal"], "#8b94a8")
    ax2.scatter(pd.Timestamp(r["date"]), r["S"], color=c, s=44, zorder=3,
                edgecolors="#0b0e11", linewidths=.6)
ax2.axhline(70, color="#2ebd85", ls=":", lw=1)
ax2.axhline(50, color="#f0b90b", ls=":", lw=1)
ax2.text(x.iloc[0], 71.5, "Buy 70", fontsize=8, color="#2ebd85")
ax2.text(x.iloc[0], 51.5, "战术线 50", fontsize=8, color="#f0b90b")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=7)
           for k, c in SIG_COLOR.items()]
ax2.legend(handles=handles, loc="upper left", fontsize=7.5, ncol=3, framealpha=.3)
ax2.set_ylim(0, 105)
ax2.set_ylabel("S_total")
ax3 = ax2.twinx()
ax3.plot(x, tdf["rank_pct"] * 100, color="#8b94a8", lw=1, ls="--")
ax3.set_ylabel("池分位 (%)", color="#8b94a8")
ax3.tick_params(axis="y", colors="#8b94a8")

fig.patch.set_facecolor("#0a0d12")
for a in list(axes) + [ax3]:
    a.set_facecolor("#0a0d12")
    a.tick_params(colors="#8b94a8")
    a.xaxis.label.set_color("#8b94a8")
    a.yaxis.label.set_color("#8b94a8")
    for sp in a.spines.values():
        sp.set_color("#2a3348")
ax.title.set_color("#e8e8e8")
for a in axes:
    leg = a.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_color("#c8cfdd")
plt.tight_layout()
plt.savefig("output/case_008528.png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print("replot OK")
