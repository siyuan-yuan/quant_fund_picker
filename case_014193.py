# -*- coding: utf-8 -*-
"""
单点事件检验: 模型在2026年初能否预判014193(汇添富芯片产业指数增强A)未来大涨?
方法: 2025-01 ~ 2026-07 每月末 PiT 打分轨迹 + 前瞻收益验证
"""
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

TARGET = "014193"
DATES = ["2025-01-27","2025-02-28","2025-03-31","2025-04-30","2025-05-30",
         "2025-06-30","2025-07-31","2025-08-29","2025-09-30","2025-10-31",
         "2025-11-28","2025-12-31","2026-01-30","2026-02-27","2026-03-31",
         "2026-04-30","2026-05-29","2026-06-30","2026-07-31"]


def main():
    t0 = time.time()
    pool = build_pool(random_n=800)
    if TARGET not in pool:
        pool.append(TARGET)
    print(f"[池] {len(pool)} 只")

    traj = []
    for d in DATES:
        rows = []
        with ThreadPoolExecutor(max_workers=8) as ex:
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
        if len(hit) == 0 or pd.isna(hit.iloc[0].get("F_alpha")) and hit.iloc[0].get("error"):
            err = hit.iloc[0].get("error") if len(hit) else "unknown"
            traj.append({"date": d, "S": np.nan, "rank_pct": np.nan, "note": str(err)})
            print(f"  {d}: 目标基金无法评分 ({err})")
            continue
        h = hit.iloc[0]
        f6, f12 = forward_returns(TARGET, d)
        rank_pct = float((ok.S_total <= h.S_total).mean())   # 总分在池内分位(1=最高)
        traj.append({"date": d, "water": h.water, "S": h.S_total, "rating": h.rating,
                     "rank_pct": round(rank_pct, 3),
                     "F_value": h.F_value, "val_pct": h.val_pct,
                     "F_alpha": h.F_alpha, "ir": h.ir_winrate, "dc": h.down_capture,
                     "F_mom": h.F_momentum, "r4": h.mom_4m1m, "r7": h.mom_7m1m,
                     "top3": h.penalty_detail.get("top3_conc"),
                     "R_MDD": h.penalty_detail.get("R_MDD"),
                     "pens": h.penalty_str, "fwd6": None if f6 != f6 else round(f6, 4),
                     "fwd12": None if f12 != f12 else round(f12, 4)})
        print(f"  {d} S={h.S_total:5.1f} 池分位={rank_pct:.0%} F_v={h.F_value} F_a={h.F_alpha} "
              f"F_m={h.F_momentum} pens={h.penalty_str or '—'} fwd6={traj[-1]['fwd6']} ({time.time()-t0:.0f}s)", flush=True)
    tdf = pd.DataFrame(traj)
    tdf.to_csv("output/case_014193_trajectory.csv", index=False, encoding="utf-8-sig")

    # ---- 净值曲线 & 打分轨迹图 ----
    nav = provider.get_fund_nav(TARGET).set_index("date")
    seg = nav.loc["2025-01-01":"2026-07-31"]
    base = seg["nav"].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(seg.index, seg["nav"] / base, color="#e8b64c", lw=1.8,
            label="014193 NAV (norm, 2025-01=1)")
    ax.axvline(pd.Timestamp("2025-12-31"), color="#f6465d", ls="--", lw=1)
    ax.annotate("2025-12-31 checkpoint", (pd.Timestamp("2025-12-31"), ax.get_ylim()[0]),
                rotation=90, fontsize=8, color="#f6465d", va="bottom")
    ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("NAV (normalized)")
    ax.set_title("014193 CSI Chip Enhanced A — NAV vs Model Score trajectory", fontsize=11)

    ax2 = axes[1]
    x = pd.to_datetime(tdf["date"])
    ax2.plot(x, tdf["S"], marker="o", color="#4c9aff", lw=1.8, label="S_total (model score)")
    ax2.plot(x, tdf["F_value"], marker=".", color="#2ebd85", lw=1, label="F_value")
    ax2.axhline(70, color="#2ebd85", ls=":", lw=1); ax2.axhline(50, color="#f0b90b", ls=":", lw=1)
    ax2.text(x.iloc[1], 71, "Buy line 70", fontsize=8, color="#2ebd85")
    ax2.set_ylim(0, 105); ax2.legend(loc="upper left", fontsize=9, ncol=2)
    ax2.set_ylabel("score")
    ax3 = ax2.twinx()
    ax3.plot(x, tdf["rank_pct"] * 100, color="#8b94a8", lw=1, ls="--", label="pool rank %")
    ax3.set_ylabel("rank percentile in pool (%)", color="#8b94a8")
    ax3.tick_params(axis="y", colors="#8b94a8")
    ax3.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("output/case_014193.png", dpi=130, bbox_inches="tight")
    print("\n[saved] output/case_014193.png, case_014193_trajectory.csv")

    print("\n===== 关键日期快照 =====")
    for d in ["2025-12-31", "2026-01-30"]:
        r = tdf[tdf.date == d]
        if len(r):
            print(r[["date","S","rank_pct","F_value","val_pct","F_alpha","ir","dc",
                     "F_mom","r4","pens","fwd6","fwd12"]].to_string(index=False))
    json.dump(traj, open("output/case_014193.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
