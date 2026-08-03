# -*- coding: utf-8 -*-
"""V3.5 单点事件检验: 008528 华泰柏瑞质量成长A — 模型在一年前(2025-08)是否发买入信号?
方法同 014193 案: 月末 PiT 打分轨迹 + V3.5 作战信号还原 + 前瞻收益对账"""
import time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import provider
provider.STALE_OK = True

import rbsa
from engine import score_fund, finalize
from backtest import build_pool, forward_returns

TARGET = "008528"
DATES = [str(d.date()) for d in pd.date_range("2024-07-31", "2026-07-31", freq="ME")]


def signal_of(s, rank_pct, f_mom, pens):
    ps = pens or ""
    if "-100%" in ps:
        return "🚫否决池"
    if s >= 85:
        return "🟢价值重仓"
    if s >= 70:
        return "🟢价值建仓"
    if s >= 50 and rank_pct >= 0.95 and f_mom >= 100:
        return "⚔️动量战术"
    if s >= 50:
        return "🟡分批观望"
    return "⚪回避"


SIG_COLOR = {"🟢价值重仓": "#2ebd85", "🟢价值建仓": "#7fd6a8", "⚔️动量战术": "#f0b90b",
             "🟡分批观望": "#c9a227", "⚪回避": "#8b94a8", "🚫否决池": "#f6465d"}


def main():
    t0 = time.time()
    pool = build_pool(random_n=800)
    if TARGET not in pool:
        pool.append(TARGET)
    print(f"[池] {len(pool)} 只", flush=True)

    traj = []
    for d in DATES:
        rows = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True): c for c in pool}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if not r.get("error"):
                        rows.append(r)
                except Exception:
                    pass
        df = finalize(rows, as_of=d)
        ok = df.dropna(subset=["S_total"])
        hit = df[df.code == TARGET]
        if len(hit) == 0:
            traj.append({"date": d, "S": np.nan, "rank_pct": np.nan, "note": "无法评分"})
            continue
        h = hit.iloc[0]
        f6, f12 = forward_returns(TARGET, d)
        rank_pct = float((ok.S_total <= h.S_total).mean())
        sig = signal_of(h.S_total, rank_pct, h.F_momentum if pd.notna(h.F_momentum) else 0,
                        h.penalty_str)
        traj.append({"date": d, "water": h.water, "S": h.S_total, "rating": h.rating,
                     "rank_pct": round(rank_pct, 3), "signal": sig,
                     "F_value": h.F_value, "val_pct": h.val_pct,
                     "F_alpha": h.F_alpha, "ir": h.ir_winrate, "dc": h.down_capture,
                     "F_mom": h.F_momentum, "r4": h.mom_4m1m, "r7": h.mom_7m1m,
                     "top3": h.penalty_detail.get("top3_conc"),
                     "R_MDD": h.penalty_detail.get("R_MDD"),
                     "pens": h.penalty_str,
                     "fwd6": None if f6 != f6 else round(f6, 4),
                     "fwd12": None if f12 != f12 else round(f12, 4)})
        print(f"  {d} S={h.S_total:5.1f} 池分位={rank_pct:.0%} {sig} "
              f"F_v={h.F_value} F_a={h.F_alpha} F_m={h.F_momentum} pens={h.penalty_str or '—'} "
              f"fwd6={traj[-1]['fwd6']} ({time.time()-t0:.0f}s)", flush=True)

    tdf = pd.DataFrame(traj)
    tdf.to_csv("output/case_008528_trajectory.csv", index=False, encoding="utf-8-sig")

    # ---- 净值 + 打分/信号轨迹图 ----
    nav = provider.get_fund_nav(TARGET).set_index("date")
    seg = nav.loc["2024-07-01":"2026-07-31"]
    base = seg["nav"].iloc[0]
    # 翻三倍核验: 区间内最低点 → 其后最高点
    trough_idx = seg["nav"].idxmin()
    after = seg.loc[trough_idx:]
    peak_idx = after["nav"].idxmax()
    mult = seg["nav"].loc[peak_idx] / seg["nav"].loc[trough_idx]
    print(f"\n[翻三倍核验] 区间低点 {trough_idx.date()} nav={seg['nav'].loc[trough_idx]:.4f} "
          f"→ 其后高点 {peak_idx.date()} nav={seg['nav'].loc[peak_idx]:.4f} = ×{mult:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(seg.index, seg["nav"] / base, color="#e8b64c", lw=1.8,
            label="008528 NAV (norm, 2024-07=1)")
    ax.scatter([trough_idx], [seg["nav"].loc[trough_idx] / base], color="#f6465d", zorder=5)
    ax.scatter([peak_idx], [seg["nav"].loc[peak_idx] / base], color="#2ebd85", zorder=5)
    ax.annotate(f"低点 {trough_idx.date()}", (trough_idx, seg["nav"].loc[trough_idx] / base),
                textcoords="offset points", xytext=(6, -12), fontsize=8, color="#f6465d")
    ax.annotate(f"×{mult:.1f} 高点 {peak_idx.date()}", (peak_idx, seg["nav"].loc[peak_idx] / base),
                textcoords="offset points", xytext=(-90, 8), fontsize=8, color="#2ebd85")
    ax.axvline(pd.Timestamp("2025-08-29"), color="#4c9aff", ls="--", lw=1)
    ax.text(pd.Timestamp("2025-09-02"), ax.get_ylim()[0] * 1.05, "一年前(2025-08)",
            fontsize=8, color="#4c9aff")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("008528 华泰柏瑞质量成长A — NAV vs V3.5 模型打分/信号轨迹", fontsize=11)

    ax2 = axes[1]
    x = pd.to_datetime(tdf["date"])
    ax2.plot(x, tdf["S"], color="#4c9aff", lw=1.4, zorder=1)
    for _, r in tdf.dropna(subset=["S"]).iterrows():
        c = SIG_COLOR.get(r["signal"], "#8b94a8")
        ax2.scatter(pd.Timestamp(r["date"]), r["S"], color=c, s=42, zorder=3,
                    edgecolors="#0b0e11", linewidths=.6)
    ax2.axhline(70, color="#2ebd85", ls=":", lw=1)
    ax2.axhline(50, color="#f0b90b", ls=":", lw=1)
    ax2.text(x.iloc[0], 71, "Buy 70", fontsize=8, color="#2ebd85")
    ax2.text(x.iloc[0], 51, "战术线 50", fontsize=8, color="#f0b90b")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=7)
               for k, c in SIG_COLOR.items()]
    ax2.legend(handles=handles, loc="upper left", fontsize=7.5, ncol=3, framealpha=.3)
    ax2.set_ylim(0, 105)
    ax3 = ax2.twinx()
    ax3.plot(x, tdf["rank_pct"] * 100, color="#8b94a8", lw=1, ls="--")
    ax3.set_ylabel("池分位 (%)", color="#8b94a8")
    ax3.tick_params(axis="y", colors="#8b94a8")
    fig.patch.set_facecolor("#0b0e11")
    for a in list(axes) + [ax3]:
        a.set_facecolor("#0b0e11")
        a.tick_params(colors="#8b94a8")
        for sp in a.spines.values():
            sp.set_color("#2a3348")
    for a in axes:
        a.yaxis.label.set_color("#8b94a8")
        a.title.set_color("#e8e8e8")
        leg = a.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_color("#c8cfdd")
    plt.tight_layout()
    plt.savefig("output/case_008528.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    json.dump(traj, open("output/case_008528.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[saved] output/case_008528.png / .csv / .json")


if __name__ == "__main__":
    main()
