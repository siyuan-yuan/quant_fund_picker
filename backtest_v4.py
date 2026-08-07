# -*- coding: utf-8 -*-
"""
backtest_v4.py — 用 V4 Huber 模型打分，跑 V3.8 执行层做 A/B 对比。

流程：
  1. 从 output/factor_rows/*.csv 加载所有 PiT 原始量
  2. 对每个季点，用严格 walk-forward：以该时点之前所有数据训练 Huber，再对当期基金打分
  3. 把分数映射到 0~100 截面百分位（与 S_total 同口径），写入 panel["S"]
  4. 调用 backtest_local.simulate() 复用 V3.8 执行层（10槽、CPPI、危机过滤等）
  5. 输出 tag=v4_* 的三本账与净值图，并和 V3.7 现行模型对比

用法:
    python backtest_v4.py                          # 默认 2019Q1→2025Q4
    python backtest_v4.py --buy 70 --sell 45
    python backtest_v4.py --start 2018-03-31 --end 2025-12-31
"""
import os, sys, argparse, time, glob
import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backtest_local import (load_names, simulate, report, chart,
                             mdd_factor)


def load_navs_offline(codes):
    """直接从 cache/nav_*.csv 读，不走 akshare 重抓。"""
    navs = {}
    for c in codes:
        path = f"cache/nav_{c}.csv"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            if len(df) < 200:
                continue
            s = df.set_index("date")["nav"].sort_index()
            s = s[~s.index.duplicated(keep="last")].dropna()
            if len(s) > 200:
                navs[c] = s
        except Exception:
            pass
    return navs


FACTOR_ROWS = "output/factor_rows"
DECAY_HALFLIFE_YEARS = 2.0
HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.01
MIN_TRAIN = 8                      # 至少 8 季训练


def load_raw_panel():
    files = sorted(glob.glob(f"{FACTOR_ROWS}/*.csv"))
    rows = []
    for f in files:
        d = pd.read_csv(f, dtype={"code": str})
        d["date"] = pd.to_datetime(d["date"])
        # 截面 rank
        for c in ["r4", "r7", "wr", "dc", "R_MDD"]:
            d[c + "_rk"] = d.groupby("date")[c].rank(pct=True)
        d["trend_t"] = np.clip((d["ma20_dist"] + 0.02) / 0.06, 0, 1)
        rp = np.minimum(0.5 * np.maximum(d["R_MDD"] - 1.2, 0), 1.0)
        rp[d.water <= 0.35] *= 0.5
        d["rmdd_pen"] = rp
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    return panel


def build_features(val_pct, r4_rk, r7_rk, wr_rk, dc_rk, rmdd_pen, water, trend_t):
    value_z = 1.0 - val_pct
    mom_pure = 0.5 * r4_rk + 0.5 * r7_rk
    quality = 0.5 * wr_rk + 0.5 * (1.0 - dc_rk)
    safety = 1.0 - rmdd_pen
    macro_state = water - 0.5
    val_x_mom = value_z * mom_pure
    return np.column_stack([value_z, mom_pure, quality, safety,
                            macro_state, trend_t, val_x_mom])


def score_v4_walkforward(panel):
    """每个季点用之前所有数据训练 Huber，然后对当期打分（严格 PiT）。"""
    dates = sorted(panel.date.unique())
    out = []
    t0 = time.time()
    for i, d in enumerate(dates):
        g = panel[panel.date == d].copy()
        if i < MIN_TRAIN:
            g["S"] = np.nan
        else:
            tr = panel[panel.date < d]
            feat_tr = build_features(tr.val_pct.values, tr.r4_rk.values,
                                     tr.r7_rk.values, tr.wr_rk.values,
                                     tr.dc_rk.values, tr.rmdd_pen.values,
                                     tr.water.values, tr.trend_t.values)
            # 截面 z-score y
            y = tr.groupby("date")["fwd6"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)).values
            years = (d - pd.to_datetime(tr.date)).dt.days / 365.25
            sw = np.exp(-np.log(2) * years / DECAY_HALFLIFE_YEARS)
            m = Pipeline([("sc", StandardScaler()),
                          ("h", HuberRegressor(epsilon=HUBER_EPSILON,
                                               alpha=HUBER_ALPHA,
                                               max_iter=1000))])
            # 去 NaN
            mask = ~np.isnan(feat_tr).any(axis=1) & ~np.isnan(y)
            m.fit(feat_tr[mask], y[mask], h__sample_weight=sw[mask])

            feat_te = build_features(g.val_pct.values, g.r4_rk.values,
                                     g.r7_rk.values, g.wr_rk.values,
                                     g.dc_rk.values, g.rmdd_pen.values,
                                     g.water.values, g.trend_t.values)
            z = m.named_steps["h"].predict(
                m.named_steps["sc"].transform(feat_te))
            g["S"] = pd.Series(z, index=g.index).rank(pct=True) * 100
        out.append(g[["date", "code", "S", "water", "R_MDD"]])
        if (i + 1) % 4 == 0:
            print(f"  [v4] {pd.Timestamp(d).date()} 打分完成 "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-03-31")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--buy", type=float, default=70.0)
    ap.add_argument("--sell", type=float, default=45.0)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--cost-in", dest="cost_in", type=float, default=0.0015)
    ap.add_argument("--cost-out", dest="cost_out", type=float, default=0.005)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--cash-yield", type=float, default=0.025)
    ap.add_argument("--trail-stop", type=float, default=0.20)
    ap.add_argument("--rebalance", choices=["none", "quarterly"], default="quarterly")
    ap.add_argument("--no-crisis", dest="crisis", action="store_false")
    ap.set_defaults(crisis=True)
    ap.add_argument("--no-cppi", dest="cppi", action="store_false")
    ap.set_defaults(cppi=True)
    args = ap.parse_args()

    print("[1/4] 加载原始因子面板 ...")
    raw = load_raw_panel()
    raw = raw[(raw.date >= pd.Timestamp(args.start)) &
              (raw.date <= pd.Timestamp(args.end))]
    print(f"  日期范围 {raw.date.min().date()} → {raw.date.max().date()}, "
          f"n={len(raw)}, 基金={raw.code.nunique()}")

    print("[2/4] Walk-forward 训练 V4 Huber 并打分 ...")
    panel = score_v4_walkforward(raw)
    panel = panel.dropna(subset=["S"])
    # 关键: simulate 用 day_str=str(day.date()) 作 key，必须保证 date 是 'YYYY-MM-DD' 字符串
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(panel.date.unique())
    print(f"  可回测 {len(dates)} 个季点, S>{args.buy:.0f} 信号 {int((panel.S>args.buy).sum())}")

    print("[3/4] 加载净值 ...")
    navs = load_navs_offline(sorted(panel.code.unique()))
    print(f"  可用净值 {len(navs)}/{panel.code.nunique()}")
    bench = provider.get_close_by_src("sina", "sh000300").dropna().sort_index()

    print("[4/4] 跑 V3.8 执行层 ...")
    ec, tr = simulate(panel, navs, bench, dates, args)
    names = load_names()
    if tr is None:
        tr = pd.DataFrame(columns=["code", "name", "entry_date", "exit_date",
                                   "entry_px", "exit_px", "entry_S", "exit_S",
                                   "exit_reason", "hold_days", "alloc_yuan",
                                   "net_ret", "pnl_yuan"])
    if len(tr):
        if "name" not in tr.columns:
            tr.insert(1, "name", tr.code.map(lambda c: names.get(c, "")))
        tr = tr.sort_values(["entry_date", "code"])
    tag = (f"v4_b{args.buy:.0f}s{args.sell:.0f}n{args.slots}"
           f"k{int(args.capital/10000)}w")
    tr.to_csv(f"output/bt_trades_{tag}.csv", index=False, encoding="utf-8-sig")
    ec.reset_index().to_csv(f"output/bt_daily_{tag}.csv", index=False, encoding="utf-8-sig")
    report(ec, tr, bench, args, dates, names, tag)
    chart(ec, bench, args, dates, tag)
    print(f"\n[done] tag={tag}")


if __name__ == "__main__":
    main()
