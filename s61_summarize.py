#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6.1 汇总（#33）：种子×maxn 网格 11 组面板 vs canonical 基准的组间分布（仅报告不平判）

比较对象（预登记）：S_total 的 fwd1/3/6 IC 全期均值与 HAC t（§0.3 同口径；L=5/3/1）。
canonical = output/p1_panel（原式种子、maxn=500、当前权威面板）。
产物：output/v5/s61_robustness.csv / s61_summary.md
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from p1_analysis import _ic_row
from stats_hac import nw_tstat

BASE_DIR = "output/p1_panel"
GRID_DIR = "output/s61_panel"
MIN_N = 30
LAGS = {"fwd6": 5, "fwd3": 3, "fwd1": 1}


def load_all(panel_dir: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(panel_dir, "20*.csv")))
    parts = []
    for f in fs:
        g = pd.read_csv(f, dtype={"code": str},
                        usecols=["code", "date", "S_total", "fwd1", "fwd3", "fwd6"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def ic_stats(df: pd.DataFrame, h: str) -> dict:
    ic = df.groupby("date", sort=True).apply(
        lambda g: _ic_row(g, "S_total", h, 1.0) if len(g) >= MIN_N else np.nan,
        include_groups=False).dropna()
    if len(ic) < 8:
        return dict(n=len(ic), ic_mean=np.nan, t_naive=np.nan, t_hac=np.nan)
    r = nw_tstat(ic.values, lag=LAGS[h])
    return dict(n=len(ic), ic_mean=float(ic.mean()),
                t_naive=float(r["t_naive"]), t_hac=float(r["t_hac"]))


def main():
    base = load_all(BASE_DIR)
    print(f"[S6.1汇总] canonical: {base.date.nunique()} 月 × {len(base)} 行", flush=True)
    rows = []
    res_base = {h: ic_stats(base, h) for h in LAGS}
    for h, r in res_base.items():
        rows.append(dict(tag="CANONICAL", horizon=h, **r))

    for d in sorted(glob.glob(os.path.join(GRID_DIR, "*"))):
        tag = os.path.basename(d)
        n_months = len([f for f in os.listdir(d) if f[:4].isdigit()])
        if n_months < 100:
            print(f"[skip] {tag}: 仅 {n_months} 月(<100), 可能仍在构建", flush=True)
            continue
        df = load_all(d)
        for h in LAGS:
            r = ic_stats(df, h)
            rows.append(dict(tag=tag, horizon=h, **r))
        print(f"[done] {tag}: {df.date.nunique()} 月", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs("output/v5", exist_ok=True)
    out.to_csv("output/v5/s61_robustness.csv", index=False)

    lines = ["# S6.1 种子×maxn 抽样稳健性（#33 子维度，批内 rank 通道）",
             "",
             f"canonical vs {len(glob.glob(os.path.join(GRID_DIR, '*')))} 组变体面板的 S_total IC 组间分布",
             "", "| tag | horizon | n | IC均值 | t_naive | t_HAC |", "|---|---|---|---|---|---|---|"]
    for _, x in out.iterrows():
        lines.append(f"| {x.tag} | {x.horizon} | {x.n} | {x.ic_mean:+.4f} | "
                     f"{x.t_naive:.2f} | {x.t_hac:.2f} |")
    lines += ["",
              "## 组间离散度（不含 canonical 的 11 组变体内）", ""]
    for h in LAGS:
        sub = out[(out.tag != "CANONICAL") & (out.horizon == h)].ic_mean
        if len(sub):
            lines.append(f"- {h}: IC均值 跨组范围 [{sub.min():+.4f}, {sub.max():+.4f}], "
                         f"组间 std {sub.std():.4f}（canonical "
                         f"{res_base[h]['ic_mean']:+.4f} {'在' if sub.min() <= res_base[h]['ic_mean'] <= sub.max() else '超出'}范围内）")
    lines += ["", "**仅报告，不平判；不得据此调参（#33 自封）。**", ""]
    with open("output/v5/s61_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
