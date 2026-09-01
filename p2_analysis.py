#!/usr/bin/env python3
"""Phase 2 (本地波次) 分析:
  p2_a — P2-A 价值因子体制条件化: V0(现行)/V1(水位门控)/V2(三档水位权重)/V3(全去价值)
         预登记见 docs/优化计划_V4_2026-08.md §2 预登记#2。
  p2_2 — P2-2 低波因子族: -vol126 / -vol252 / -tuw252 的 IC/五分位价差/增量IC/子群。

全部离线, 基于 P1-0 严格 PIT 面板 (output/p1_panel) + 辅助 vol 面板 (output/p2/vol_panel.csv)。
产物: output/p2/p2a_*.csv, p2_2/p22_*.csv
复现: ./.venv/bin/python p2_vol_build.py 后
      ./.venv/bin/python p2_analysis.py --only p2_a
      ./.venv/bin/python p2_analysis.py --only p2_2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output", "p2")
VOL_CSV = os.path.join(OUT_DIR, "vol_panel.csv")

from p1_analysis import load_panel, _parse_pens, _ic_row, quintile_spread  # noqa: E402

RECENT_START = pd.Timestamp("2023-04-01")  # 近 3 年窗 (2023-04 → 2026-03)


# ---------------- 通用统计工具 ----------------

def _t(mean, sd, n):
    if n < 5 or sd == 0 or sd != sd:
        return np.nan
    return float(mean / (sd / np.sqrt(n)))


def monthly_ic_series(d: pd.DataFrame, col: str, hcol: str, direction: float = 1.0):
    """逐月截面 Spearman IC 序列 (对齐 date)"""
    rows = []
    for dt, g in d.groupby("date", sort=True):
        rows.append((dt, _ic_row(g, col, hcol, direction)))
    s = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    return s


def paired_diff_t(ic_a: pd.Series, ic_b: pd.Series):
    """逐月配对差 t: mean(a-b) / (sd/√n)"""
    j = pd.concat([ic_a, ic_b], axis=1, join="inner", keys=["a", "b"]).dropna()
    d = (j["a"] - j["b"]).values
    n = len(d)
    if n < 5:
        return dict(n=n, mean_diff=np.nan, t=np.nan, pct_better=np.nan)
    return dict(n=n, mean_diff=float(np.mean(d)),
                t=_t(np.mean(d), np.std(d, ddof=1), n),
                pct_better=float((d > 0).mean()))


def ic_summary(ic: pd.Series):
    v = ic.dropna()
    return dict(n_months=int(len(v)),
                ic_mean=float(v.mean()) if len(v) else np.nan,
                ic_t=_t(v.mean(), v.std(ddof=1), len(v)) if len(v) else np.nan,
                icir=float(v.mean() / v.std(ddof=1)) if len(v) and v.std(ddof=1) > 0 else np.nan,
                pct_pos=float((v > 0).mean()) if len(v) else np.nan)


def _save(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → {path}")
    return path


# ================= P2-A 价值体制条件化 =================

def synth_rows(g: pd.DataFrame, weights, use_value: bool):
    """engine 同式合成: 缺失因子按权重归一 + 乘法惩罚链 (与 p1_2 synth 一致)"""
    fv, fa, fm = (g[c].values for c in ["F_value", "F_alpha", "F_momentum"])
    pens_l = g["penalties"].apply(_parse_pens).tolist()
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        num = den = 0.0
        items = [(weights[1], fa[i]), (weights[2], fm[i])]
        if use_value:
            items = [(weights[0], fv[i])] + items
        for w, v in items:
            if v is None or (isinstance(v, float) and v != v):
                continue
            num += w * v
            den += w
        base = num / den if den > 1e-9 else 0.0
        s = base
        for p in (pens_l[i] or []):
            s *= (1 - p)
        out[i] = max(0.0, min(100.0, s))
    return out


def p2_a(df: pd.DataFrame, min_n: int = 30) -> None:
    from engine import resolve_weights

    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    for c in ["F_value", "F_alpha", "F_momentum"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    months = sorted(d["date"].unique())
    # 逐月水位 (每行都有, 取首行)
    water_of = {m: float(g["water"].iloc[0]) for m, g in d.groupby("date")}

    # 预登记权重方案 (§2 预登记#2)
    def month_weights(m, variant):
        wtr = water_of[m]
        (wv0, wa0, wm0), _ = resolve_weights(wtr)
        if variant == "V0":
            return [wv0, wa0, wm0], True
        if variant == "V1":  # water>0.35 价值腿门控, alpha:mom 按 0.35:0.25 归一
            if wtr == wtr and wtr > 0.35:
                return [0.0, 0.35, 0.25], False
            return [wv0, wa0, wm0], True
        if variant == "V2":  # 三档水位价值权重
            if wtr != wtr:
                wtr = 0.5
            if wtr <= 0.20:
                return [0.55, 0.45 * 0.35 / 0.60, 0.45 * 0.25 / 0.60], True
            if wtr <= 0.35:
                return [0.35, 0.65 * 0.35 / 0.60, 0.65 * 0.25 / 0.60], True
            return [0.20, 0.80 * 0.35 / 0.60, 0.80 * 0.25 / 0.60], True
        if variant == "V3":  # 全去价值
            return [0.0, 0.35, 0.25], False
        raise ValueError(variant)

    S = {}
    for variant in ["V0", "V1", "V2", "V3"]:
        cols = []
        for m, g in d.groupby("date", sort=True):
            w, uv = month_weights(m, variant)
            cols.append(synth_rows(g, w, uv))
        d[f"S_{variant}"] = np.concatenate(cols)

    # 校验: V0 应复现面板 S_total (与引擎离线重建口径一致, 允许 0.1 内舍入差)
    diff = (d["S_V0"] - d["S_total"]).abs()
    print(f"[p2_a] 校验 S_V0 vs 面板 S_total: max|diff| = {diff.max():.4f}, "
          f"mean|diff| = {diff.mean():.5f}", flush=True)
    _save(pd.DataFrame([dict(metric="max_abs_diff", value=float(diff.max())),
                        dict(metric="mean_abs_diff", value=float(diff.mean())),
                        dict(metric="rows", value=int(len(d)))]), "p2a_verify.csv")

    # 逐变体 × 逐 horizon IC
    ic_rows, paired_rows, split_rows = [], [], []
    ics = {}
    for variant in ["V0", "V1", "V2", "V3"]:
        ics[variant] = {}
        for hcol in ["fwd1", "fwd3", "fwd6"]:
            s = monthly_ic_series(d, f"S_{variant}", hcol)
            ics[variant][hcol] = s
            ics[variant][hcol].name = f"{variant}_{hcol}"
            row = dict(variant=variant, horizon=hcol, **ic_summary(s))
            ic_rows.append(row)

    # 配对差 vs V0 (fwd6 主 horizon; fwd1/3 复核)
    for window, lo, hi in [("full", pd.Timestamp("2013-01-01"), pd.Timestamp("2200-01-01")),
                           ("recent_2023_04+", RECENT_START, pd.Timestamp("2200-01-01"))]:
        for hcol in ["fwd6", "fwd3", "fwd1"]:
            for variant in ["V1", "V2", "V3"]:
                a = ics[variant][hcol]
                b = ics["V0"][hcol]
                res = paired_diff_t(a[a.index >= lo], b[b.index >= lo])
                paired_rows.append(dict(window=window, horizon=hcol, variant=variant,
                                        **res))
    # V1 诊断: 仅 water>0.35 月份 (门控生效子样本)
    mask_hi = pd.Series({m: (water_of[m] == water_of[m] and water_of[m] > 0.35)
                         for m in months})
    for hcol in ["fwd6", "fwd3"]:
        res = paired_diff_t(ics["V1"][hcol][mask_hi], ics["V0"][hcol][mask_hi])
        paired_rows.append(dict(window="water_gt_0.35_only", horizon=hcol,
                                variant="V1", **res))

    # 水位分层诊断: 每变体在 低水位(≤0.35)/高水位(>0.35) 月份的 IC
    for band, cond in [("water_le_0.35", lambda w: w == w and w <= 0.35),
                       ("water_gt_0.35", lambda w: w == w and w > 0.35)]:
        keep = pd.Series({m: cond(water_of[m]) for m in months})
        for variant in ["V0", "V1", "V2", "V3"]:
            for hcol in ["fwd6"]:
                s = ics[variant][hcol][keep]
                split_rows.append(dict(band=band, variant=variant, horizon=hcol,
                                       **ic_summary(s)))

    _save(pd.DataFrame(ic_rows), "p2a_ic_table.csv")
    _save(pd.DataFrame(paired_rows), "p2a_paired.csv")
    _save(pd.DataFrame(split_rows), "p2a_water_split.csv")
    print("[p2_a] 完成", flush=True)


# ================= P2-2 低波因子族 =================

# (因子列名, 源列, 符号) 低波=好 → 存储列取负, 之后统一 direction=1.0
VOL_FACTORS = [
    ("nv126", "vol126", -1.0),
    ("nv252", "vol252", -1.0),
    ("ntuw252", "tuw252", -1.0),
]
DIRN = 1.0  # 存储列已负值化


def p2_2(df: pd.DataFrame, min_n: int = 30) -> None:
    if not os.path.exists(VOL_CSV):
        raise SystemExit("缺少 vol 面板, 先运行: ./.venv/bin/python p2_vol_build.py")
    vol = pd.read_csv(VOL_CSV, dtype={"code": str}, parse_dates=["date"])
    d = df.merge(vol, on=["date", "code"], how="left")
    d = d[d.groupby("date")["code"].transform("size") >= min_n].copy()
    for fname, src, sign in VOL_FACTORS:
        d[fname] = -d[src].astype(float)
    n126 = int(d["vol126"].notna().sum())
    print(f"[p2_2] 合并后行数 {len(d)}, vol126 非空 {n126}", flush=True)

    ic_rows, spread_rows, incr_rows, sub_rows = [], [], [], []
    ics = {}
    for fname, src, _sign in VOL_FACTORS:
        for hcol in ["fwd1", "fwd3", "fwd6"]:
            s = monthly_ic_series(d, fname, hcol, DIRN)
            ics[(fname, hcol)] = s
            row = dict(factor=fname, horizon=hcol, **ic_summary(s))
            if hcol == "fwd6":
                row["gate"] = (abs(row["ic_mean"] or 0) > 0.05) and (
                    row["ic_t"] == row["ic_t"] and abs(row["ic_t"]) > 2)
            ic_rows.append(row)
        # 五分位价差 (fwd6)
        sp = []
        for dt, g in d.groupby("date", sort=True):
            sp.append((dt, quintile_spread(g, fname, "fwd6", DIRN)))
        sps = pd.Series([r[1] for r in sp], index=[r[0] for r in sp]).dropna()
        spread_rows.append(dict(factor=fname, n_months=int(len(sps)),
                                q5q1_mean=float(sps.mean()) if len(sps) else np.nan))

    # 增量 IC vs S_total (秩空间 OLS 残差, fwd6)
    d["S_total"] = pd.to_numeric(d["S_total"], errors="coerce")
    for fname, src, _sign in VOL_FACTORS:
        res_ids = []
        for dt, g in d.groupby("date", sort=True):
            x = g[fname].values
            y = g["fwd6"].values
            z = g["S_total"].values
            m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            if m.sum() < 30:
                res_ids.append((dt, np.nan))
                continue
            xr = pd.Series(x[m]).rank(pct=True).values
            zr = pd.Series(z[m]).rank(pct=True).values
            ones = np.ones_like(xr)
            beta, *_ = np.linalg.lstsq(np.column_stack([ones, zr]), xr, rcond=None)
            resid = xr - (ones * beta[0] + zr * beta[1])
            res_ids.append((dt, _spearman(resid, y[m])))
        s = pd.Series([r[1] for r in res_ids], index=[r[0] for r in res_ids])
        incr_rows.append(dict(factor=fname, **ic_summary(s)))

    # 子群 (主动/被动, A股/海外) fwd6
    d["is_ov"] = d["top1_style"].isin(
        {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}).fillna(False)
    for tag, sub in [("all", d), ("active", d[~d["is_passive"]]),
                     ("passive", d[d["is_passive"]]),
                     ("ashare", d[~d["is_ov"]]), ("overseas", d[d["is_ov"]])]:
        for fname, src, _sign in VOL_FACTORS:
            s = monthly_ic_series(sub, fname, "fwd6", DIRN)
            sub_rows.append(dict(group=tag, factor=fname, **ic_summary(s)))

    _save(pd.DataFrame(ic_rows), "p22_ic_table.csv")
    _save(pd.DataFrame(spread_rows), "p22_spreads.csv")
    _save(pd.DataFrame(incr_rows), "p22_incr_ic.csv")
    _save(pd.DataFrame(sub_rows), "p22_subgroup_ic.csv")
    print("[p2_2] 完成", flush=True)


def _spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10:
        return np.nan
    return stats.spearmanr(x[m], y[m])[0]


# ================= main =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["p2_a", "p2_2"], default=None)
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("加载 P1-0 面板 ...", flush=True)
    df = load_panel()
    print(f"面板行数 {len(df)}, 月份 {df['date'].nunique()}", flush=True)

    if args.only in (None, "p2_a"):
        p2_a(df, args.min_n)
    if args.only in (None, "p2_2"):
        p2_2(df, args.min_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
