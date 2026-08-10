# -*- coding: utf-8 -*-
"""全历史 ML 面板构建: bt_scores_cache(235季×217只, 2006-09→2026-03) + 净值缓存 → 特征 + fwd 收益标签"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import provider

SRC = "output/bt_scores_cache"
SUF = "_2e4ec0f5"
OUT = "output/ml_panel.csv"

# ---------- 1) 原始特征面板 ----------
files = sorted(f for f in os.listdir(SRC) if f.endswith(SUF + ".csv"))
parts = []
for f in files:
    g = pd.read_csv(os.path.join(SRC, f), dtype={"code": str})
    g["date"] = f[:10]
    parts.append(g)
df = pd.concat(parts, ignore_index=True)
print(f"[1] 面板 {len(df)} 行, {df.date.nunique()} 个月, {df.code.nunique()} 只")

# ---------- 2) 净值 → ma20_dist + fwd 标签 ----------
codes = sorted(df.code.unique())
adj_map, nav_map = {}, {}
for c in codes:
    try:
        raw = provider.get_fund_nav(c)
        if raw is None or not len(raw):
            continue
        raw = raw.set_index("date")
        r = raw["ret"].fillna(0)
        r = r[~r.index.duplicated(keep="last")].sort_index()
        adj = (1 + r).cumprod()
        adj_map[c] = adj
        nav_map[c] = raw["nav"]
    except Exception:
        pass
print(f"[2] 净值可用 {len(adj_map)}/{len(codes)}")

def ma20_dist_asof(adj, d):
    idx = adj.index[adj.index <= pd.Timestamp(d)]
    if len(idx) < 20:
        return np.nan
    w = idx[-20:]
    return float(adj.loc[idx[-1]] / adj.loc[w].mean() - 1)

def fwd_ret(adj, d, months):
    q = pd.Timestamp(d)
    t1, t2 = q + pd.DateOffset(months=months), q
    if t1 > adj.index[-1] or q < adj.index[0]:
        return np.nan
    v0, v1 = adj.asof(q), adj.asof(t1)
    if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
        return np.nan
    return float(v1 / v0 - 1)

df["ma20_dist"] = [ma20_dist_asof(adj_map[c], d) if c in adj_map else np.nan
                   for c, d in zip(df.code, df.date)]
for m in (3, 6, 12):
    df[f"fwd{m}"] = [fwd_ret(adj_map[c], d, m) if c in adj_map else np.nan
                     for c, d in zip(df.code, df.date)]

# 净值期间覆盖过滤(评分时点必须有净值)
last = {c: a.index[-1] for c, a in adj_map.items()}
df["nav_last"] = [last.get(c, pd.NaT) for c in df.code]
df = df[df.nav_last >= pd.to_datetime(df.date) + pd.Timedelta(days=60)]

# ---------- 3) 截面 rank 特征 ----------
for col, rk in [("r4", "r4_rk"), ("r7", "r7_rk"), ("wr", "wr_rk"), ("dc", "dc_rk"),
                ("val_pct", "val_rk")]:
    df[rk] = df.groupby("date")[col].rank(pct=True)

# V3.6 平滑回撤惩罚 + 趋势确认度(V4 口径)
rmdd = np.minimum(0.5 * np.maximum(df["R_MDD"].fillna(0) - 1.2, 0), 1.0)
low = df.groupby("date")["water"].transform("first") <= 0.35
rmdd = rmdd.where(~low, rmdd * 0.5)
df["rmdd_pen"] = rmdd
df["trend_t"] = np.clip((df["ma20_dist"].fillna(0) + 0.02) / 0.06, 0, 1)

# ---------- 4) 目标: 截面 z / rank ----------
for m in (3, 6, 12):
    z = df.groupby("date")[f"fwd{m}"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    df[f"fwd{m}_z"] = z
    df[f"fwd{m}_rk"] = df.groupby("date")[f"fwd{m}"].rank(pct=True)

# V4 特征集(与 model_v4.build_features 同口径)
df["value_z"] = 1.0 - df["val_pct"]
df["mom_pure"] = 0.5 * df["r4_rk"] + 0.5 * df["r7_rk"]
df["quality"] = 0.5 * df["wr_rk"] + 0.5 * (1.0 - df["dc_rk"])
df["safety"] = 1.0 - df["rmdd_pen"]
df["macro_state"] = df["water"] - 0.5
df["val_x_mom"] = df["value_z"] * df["mom_pure"]

keep = ["date", "code", "val_pct", "val_cov", "trend_ok", "water", "R_MDD",
        "other_pen", "r4", "r7", "wr", "dc", "ma20_dist", "rmdd_pen", "trend_t",
        "r4_rk", "r7_rk", "wr_rk", "dc_rk", "val_rk",
        "value_z", "mom_pure", "quality", "safety", "macro_state", "val_x_mom",
        "fwd3", "fwd6", "fwd12", "fwd3_z", "fwd6_z", "fwd12_z", "fwd3_rk", "fwd6_rk", "fwd12_rk"]
df[keep].to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[4] 已存 {OUT}: {len(df)} 行 | fwd6 有效 {df.fwd6.notna().sum()} | "
      f"最新可训练季 {df.loc[df.fwd6.notna(), 'date'].max()}")
print(f"[5] 各时期行数: {df.groupby(df.date.str[:4]).size().to_dict()}")
