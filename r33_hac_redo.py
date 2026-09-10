#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #23（R3.3）：P2-A / P2-2 历史裁决 HAC 复核

复核对象：
  P2-A 价值因子体制条件化：V1(水位门控)/V2(三档)/V3(全去价值) 对 V0 的配对差
        （× fwd1/3/6 × full/2023-04+ 窗口 + V1 高水位子样本诊断）
  P2-2 低波因子族：-vol126 / -vol252 / -tuw252（含 tuw252 原 t=3.97 边界项）三个 horizon

协议：只重算统计。V0-V3 合成权重与 synth_rows 逐字复用 p2_analysis；
vol 面板 = output/p2/vol_panel.csv（p2_vol_build.py 重建，方向 col 取负与原式一致）。
产物：output/v5/r33_p2a_hac.csv / r33_p22_hac.csv / r33_summary.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True

import p1_analysis as P1
import p2_analysis as P2
from stats_hac import series_judgement, HORIZON_LAGS
from engine import resolve_weights

OUT = "output/v5"
LM = {"fwd1": HORIZON_LAGS["fwd1_monthly"], "fwd3": HORIZON_LAGS["fwd3_monthly"],
      "fwd6": HORIZON_LAGS["fwd6_monthly"]}
RECENT = pd.Timestamp("2023-04-01")


def month_weights(m, variant, water_of):
    """p2_a.month_weights 原式复制（改动权重=改模型,故必须一字不差）。"""
    wtr = water_of[m]
    (wv0, wa0, wm0), _ = resolve_weights(wtr)
    if variant == "V0":
        return [wv0, wa0, wm0], True
    if variant == "V1":
        if wtr == wtr and wtr > 0.35:
            return [0.0, 0.35, 0.25], False
        return [wv0, wa0, wm0], True
    if variant == "V2":
        if wtr != wtr:
            wtr = 0.5
        if wtr <= 0.20:
            return [0.55, 0.45 * 0.35 / 0.60, 0.45 * 0.25 / 0.60], True
        if wtr <= 0.35:
            return [0.35, 0.65 * 0.35 / 0.60, 0.65 * 0.25 / 0.60], True
        return [0.20, 0.80 * 0.35 / 0.60, 0.80 * 0.25 / 0.60], True
    if variant == "V3":
        return [0.0, 0.35, 0.25], False
    raise ValueError(variant)


def redo_p2a(df, min_n=30):
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    for c in ["F_value", "F_alpha", "F_momentum"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    months = sorted(d["date"].unique())
    water_of = {m: float(g["water"].iloc[0]) for m, g in d.groupby("date")}

    for variant in ["V0", "V1", "V2", "V3"]:
        cols = []
        for m, g in d.groupby("date", sort=True):
            w, uv = month_weights(m, variant, water_of)
            cols.append(P2.synth_rows(g, w, uv))
        d[f"S_{variant}"] = np.concatenate(cols)
    diff = (d["S_V0"] - d["S_total"]).abs()
    print(f"[R3.3/P2-A] 校验 S_V0 vs 面板 S_total: max|diff|={diff.max():.4f} (同原校验门)", flush=True)

    ics = {v: {h: P2.monthly_ic_series(d, f"S_{v}", h) for h in ["fwd1", "fwd3", "fwd6"]}
           for v in ["V0", "V1", "V2", "V3"]}
    rows = []
    for v in ["V0", "V1", "V2", "V3"]:
        for h in ["fwd1", "fwd3", "fwd6"]:
            s = ics[v][h].dropna()
            rows.append(dict(item=f"P2A_{v}_{h}", kind="ic", window="full",
                             **series_judgement(s.values, LM[h])))
    for window, lo in [("full", pd.Timestamp("2000-01-01")), ("recent_2023_04+", RECENT)]:
        for h in ["fwd6", "fwd3", "fwd1"]:
            for v in ["V1", "V2", "V3"]:
                a = ics[v][h]; b = ics["V0"][h]
                dd = (a[a.index >= lo] - b[b.index >= lo]).dropna()
                rows.append(dict(item=f"P2A_{v}-V0_{h}", kind="pair_diff", window=window,
                                 **series_judgement(dd.values, LM[h])))
    mask_hi = pd.Series({m: (water_of[m] == water_of[m] and water_of[m] > 0.35) for m in months})
    for h in ["fwd6", "fwd3"]:
        dd = (ics["V1"][h][mask_hi] - ics["V0"][h][mask_hi]).dropna()
        rows.append(dict(item=f"P2A_V1-V0_{h}", kind="pair_diff", window="water_gt_0.35_only",
                         **series_judgement(dd.values, LM[h])))
    return pd.DataFrame(rows)


def redo_p22(df, min_n=30):
    vp = pd.read_csv(P2.VOL_CSV, dtype={"code": str}, parse_dates=["date"])
    d = df[df.groupby("date")["code"].transform("size") >= min_n].merge(
        vp, on=["date", "code"], how="left")
    rows = []
    for name, col, sign in [("nvol126", "vol126", -1.0), ("nvol252", "vol252", -1.0),
                            ("ntuw252", "tuw252", -1.0)]:
        d[name] = sign * pd.to_numeric(d[col], errors="coerce")
        for h in ["fwd1", "fwd3", "fwd6"]:
            s = P2.monthly_ic_series(d, name, h).dropna()
            rows.append(dict(item=f"P22_{name}_{h}", kind="ic", window="full",
                             **series_judgement(s.values, LM[h])))
    return pd.DataFrame(rows)


def main(min_n=30):
    os.makedirs(OUT, exist_ok=True)
    df = P1.load_panel(min_n=min_n)
    ta = redo_p2a(df, min_n)
    ta.to_csv(f"{OUT}/r33_p2a_hac.csv", index=False, encoding="utf-8-sig")
    t2 = redo_p22(df, min_n)
    t2.to_csv(f"{OUT}/r33_p22_hac.csv", index=False, encoding="utf-8-sig")
    allr = pd.concat([ta, t2], ignore_index=True)
    flips = allr[allr.flip]
    L = ["# R3.3 P2-A / P2-2 裁决 HAC 复核（SURV-ADJ，统计-only）", "",
         f"共 {len(allr)} 条；翻转 {len(flips)} 条（|t|≥2 门，L 按 §0.3）。", ""]
    for t, ttl in [(ta, "P2-A 价值体制条件化"), (t2, "P2-2 低波因子族")]:
        L.append(f"## {ttl}")
        L.append(t.to_markdown(index=False)); L.append("")
    if len(flips):
        L.append("## 翻转清单")
        L.append(flips.to_markdown(index=False))
    open(f"{OUT}/r33_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"[R3.3] 翻转 {len(flips)}/{len(allr)}")
    pf = allr[(allr.kind == "pair_diff") | allr.item.str.contains("ntuw")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pf.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30)
    main(ap.parse_args().min_n)
