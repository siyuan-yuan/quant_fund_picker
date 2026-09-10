# -*- coding: utf-8 -*-
"""全历史模型动物园 (Model Zoo) —— 多模型 × 多特征 × 多目标 × 多时间窗口搜索

协议: 严格 walk-forward, 训练样本仅用 决策日 q-6个月 之前且 fwd 标签已实现的季度
      (标签在决策时点已知, 无前视偏差)。IC 用 OOS 预测 vs 实际 fwd6 计算。

用法:
  python _model_zoo.py screen    # 粗筛: 每4季重训, 全配置网格
  python _model_zoo.py verify    # 精验: 每季重训, 决赛配置 + 分年代 + 迁移
  python _model_zoo.py backtest  # 端到端: 决赛配置走 PiT Top100 回测(多窗口)
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import (HuberRegressor, Ridge, Lasso, ElasticNet,
                                  LinearRegression)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

WORK = "output/ml_work"
os.makedirs(WORK, exist_ok=True)
SEED = 42

# ---------------- 数据 ----------------
def load_panel():
    df = pd.read_csv("output/ml_panel.csv", dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"])
    df["trend_ok01"] = df["trend_ok"].astype(int)
    return df.sort_values(["date", "code"]).reset_index(drop=True)

# ---------------- 特征集 ----------------
FEATS = {
    "F2": ["value_z", "mom_pure", "quality", "safety", "macro_state", "trend_t", "val_x_mom"],
    "F3": ["value_z", "mom_pure", "quality", "safety", "macro_state", "trend_t",
           "val_x_mom", "val_cov", "other_pen", "R_MDD", "water", "trend_ok01", "ma20_dist"],
    "F5": ["val_rk", "val_cov", "trend_t", "trend_ok01", "r4_rk", "r7_rk", "wr_rk",
           "dc_rk", "R_MDD", "other_pen", "water", "ma20_dist"],
}
TARGETS = {"fwd6_z": "fwd6_z", "fwd6_rk": "fwd6_rk", "fwd12_z": "fwd12_z", "fwd3_z": "fwd3_z"}

# ---------------- 模型族 ----------------
def make_model(name):
    if name == "huber":
        return Pipeline([("sc", StandardScaler()),
                         ("m", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000))])
    if name == "ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if name == "lasso":
        return Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.002, max_iter=3000))])
    if name == "enet":
        return Pipeline([("sc", StandardScaler()),
                         ("m", ElasticNet(alpha=0.002, l1_ratio=0.5, max_iter=3000))])
    if name == "rf":
        return RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=25,
                                     n_jobs=-1, random_state=SEED)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=80, max_depth=6, min_samples_leaf=25,
                                   n_jobs=-1, random_state=SEED)
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=150, max_depth=4,
                                             l2_regularization=0.1, learning_rate=0.08,
                                             random_state=SEED)
    if name == "svrlin":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LinearSVR(C=1.0, epsilon=0.05, max_iter=3000, random_state=SEED))])
    if name == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                            max_iter=600, early_stopping=True,
                                            random_state=SEED))])
    raise ValueError(name)

class EnsHgbHuber:
    """集成: HGB + Huber 预测均值(需 fit 两个模型)"""
    def __init__(self):
        self.hgb = make_model("hgb")
        self.hub = make_model("huber")
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.hgb.fit(X, y)
            self.hub.fit(X, y)
        else:
            self.hgb.fit(X, y, sample_weight=sample_weight)
            self.hub.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X):
        return 0.5 * self.hgb.predict(X) + 0.5 * self.hub.predict(X)

MODELS = ["huber", "ridge", "lasso", "enet", "rf", "et", "gb", "hgb", "svrlin", "mlp", "ens_hgb_huber"]

# ---------------- Walk-forward ----------------
def impute_features(df, cols):
    """R3.5 修复版：逐日截面中位数填充；再缺则用**扩展窗历史月中位数**（严格 < 当月；
    原实现的全样本中位数兜底有前向成分，审计 F4 / 预登记 #24）。首日仍缺 → 保留 NaN。"""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in cols:
        med = out.groupby("date")[c].transform(lambda s: s.median())
        month_med = out.groupby("date")[c].median().sort_index()
        hist_med = month_med.shift(1).expanding().median()
        hist_row = out["date"].map(hist_med)
        out[c] = out[c].fillna(med).fillna(hist_row)
    return out

# 标签成熟期（月）：训练样本必须满足 date ≤ q − hM（R3.5 修复：原统一 6M，
# 对 fwd12 构成未来 6 个月标签前视，审计 F4 / 预登记 #24）。
TARGET_HORIZON_MO = {"fwd3_z": 3, "fwd6_z": 6, "fwd6_rk": 6, "fwd12_z": 12}


def oos_predictions(df, model_name, feats, target, retrain_every=1, min_q=8):
    """返回 df 副本 + pred 列(全部 OOS)。训练样本: date <= q−hM 且该目标标签有效。"""
    cols = FEATS[feats]
    ycol = TARGETS[target]
    h_mo = TARGET_HORIZON_MO[target]
    df = impute_features(df, cols)
    df = df.copy()
    df["pred"] = np.nan
    dates = sorted(df.date.unique())
    train_dates = [d for d in dates if (d <= dates[-1] - pd.DateOffset(months=h_mo))]
    n_tr = len(train_dates)
    mdl = None
    last_train = None
    for i, q in enumerate(dates):
        cutoff = q - pd.DateOffset(months=h_mo)
        usable = [d for d in train_dates if d <= cutoff]
        if len(usable) >= min_q and (last_train is None or i - last_train >= retrain_every or q > train_dates[-1]):
            tr = df[df.date.isin(usable) & df[ycol].notna()]
            X, y = tr[cols].astype(float).values, tr[ycol].values
            m = EnsHgbHuber() if model_name == "ens_hgb_huber" else make_model(model_name)
            m.fit(X, y)
            mdl, last_train = m, i
        if mdl is not None:
            msk = df.date == q
            df.loc[msk, "pred"] = mdl.predict(df.loc[msk, cols].astype(float).values)
    return df

def ic_metrics(df, pred_col="pred", ycol="fwd6"):
    """按日期 Spearman IC; 返回指标 dict + 每季 IC 序列"""
    recs = []
    for d, g in df.groupby("date"):
        gg = g[[pred_col, ycol]].dropna()
        if len(gg) < 30 or gg[pred_col].nunique() < 5 or gg[ycol].nunique() < 5:
            continue
        ic = gg[pred_col].corr(gg[ycol], method="spearman")
        if pd.notna(ic):
            recs.append((d, ic))
    if not recs:
        return None
    s = pd.DataFrame(recs, columns=["date", "ic"]).set_index("date")["ic"]
    ic_mean, ic_std = s.mean(), s.std(ddof=1)
    return {"n_q": len(s), "ic": float(ic_mean), "t": float(ic_mean / (ic_std / np.sqrt(len(s))) if ic_std > 0 else np.nan),
            "icir": float(ic_mean / ic_std) if ic_std > 0 else np.nan,
            "ic_pos": float((s > 0).mean()), "series": s}

def quintile_spread(df, pred_col="pred", ycol="fwd6", q=5):
    """逐日截面内分5组, 再跨日平均 Q5-Q1 实际 fwd6 差(修正: 不跨日混排)"""
    parts = []
    for _, g in df.groupby("date"):
        gg = g[[pred_col, ycol]].dropna()
        if len(gg) < 30:
            continue
        gg["b"] = pd.qcut(gg[pred_col].rank(method="first"), q, labels=False)
        m = gg.groupby("b")[ycol].mean()
        parts.append((m.iloc[-1], m.iloc[0]))
    if not parts:
        return np.nan, np.nan
    top = np.mean([p[0] for p in parts])
    bot = np.mean([p[1] for p in parts])
    return float(top - bot), float(top)

# ---------------- 阶段 ----------------
def run_screen(df):
    rows = []
    for model in MODELS:
        for feats in ["F2", "F3", "F5"]:
            for target in ["fwd6_z", "fwd6_rk"]:
                t0 = time.time()
                d = oos_predictions(df, model, feats, target, retrain_every=4)
                m = ic_metrics(d)
                sp, topq = quintile_spread(d)
                rows.append(dict(model=model, feats=feats, target=target,
                                 ic=m["ic"] if m else np.nan, t=m["t"] if m else np.nan,
                                 icir=m["icir"] if m else np.nan, ic_pos=m["ic_pos"] if m else np.nan,
                                 q5q1=sp, topq=topq, n_q=m["n_q"] if m else 0,
                                 sec=round(time.time() - t0, 1)))
                print(f"  [{model} {feats} {target}] IC={rows[-1]['ic']:.4f} t={rows[-1]['t']:.2f} "
                      f"Q5Q1={sp*100:.2f}% {rows[-1]['sec']}s", flush=True)
                d[["date", "code", "pred"]].to_csv(f"{WORK}/pred4_{model}_{feats}_{target}.csv", index=False)
    out = pd.DataFrame(rows).sort_values("ic", ascending=False)
    out.to_csv(f"{WORK}/screen.csv", index=False)
    print("\n===== SCREEN TOP15 =====")
    print(out.head(15).to_string(index=False))

def run_verify(df, top_n=8):
    scr = pd.read_csv(f"{WORK}/screen.csv")
    finalists = []
    for _, r in scr.head(top_n).iterrows():
        finalists.append((r.model, r.feats, r.target))
    # 固定加入: V4 原版(huber F2 fwd6_z) + 目标窗口变化(huber F2 fwd3_z / fwd12_z)
    for extra in [("huber", "F2", "fwd3_z"), ("huber", "F2", "fwd12_z")]:
        if extra not in finalists:
            finalists.append(extra)
    rows, series_all = [], {}
    for model, feats, target in finalists:
        t0 = time.time()
        d = oos_predictions(df, model, feats, target, retrain_every=1)
        m = ic_metrics(d)
        sp, topq = quintile_spread(d)
        # 分年代
        yr = d["date"].dt.year
        era = pd.cut(yr, bins=[2005, 2012, 2018, 2027], labels=["2006-12", "2013-18", "2019-26"])
        era_ic = {}
        for e, g in d.groupby(era):
            mm = ic_metrics(g)
            era_ic[str(e)] = mm["ic"] if mm else np.nan
        cfg = f"{model}__{feats}__{target}"
        d[["date", "code", "pred"]].to_csv(f"{WORK}/pred1_{cfg}.csv", index=False)
        series_all[cfg] = m["series"]
        rows.append(dict(cfg=cfg, model=model, feats=feats, target=target,
                         ic=m["ic"], t=m["t"], icir=m["icir"], ic_pos=m["ic_pos"],
                         q5q1=sp, topq=topq, n_q=m["n_q"],
                         **{f"era_{k}": v for k, v in era_ic.items()},
                         sec=round(time.time() - t0, 1)))
        print(f"  [{cfg}] IC={m['ic']:.4f} t={m['t']:.2f} era={ {k: round(v,3) for k,v in era_ic.items()} } "
              f"Q5Q1={sp*100:.2f}% {rows[-1]['sec']}s", flush=True)
    out = pd.DataFrame(rows).sort_values("ic", ascending=False)
    out.to_csv(f"{WORK}/verify.csv", index=False)
    print("\n===== VERIFY =====")
    print(out.to_string(index=False))
    # 迁移实验(决赛前4)
    transfer_rows = []
    for cfg in out.head(4).cfg.tolist():
        model, feats, target = cfg.split("__")
        for name, lo, hi in [("old→new", "2006-09-30", "2015-12-31"),
                             ("new→old", "2016-01-31", "2026-03-31")]:
            tr = df[(df.date >= lo) & (df.date <= hi) & df.fwd6.notna()]
            cutoff = pd.Timestamp(hi) if name == "new→old" else pd.Timestamp("2015-12-31")
            # R3.5 修复：切分训练集截止也按目标成熟期（原统一 6M，fwd12 前视）
            tr = tr[tr.date <= cutoff - pd.DateOffset(months=TARGET_HORIZON_MO[target])]
            te = df[(df.date > cutoff) & (df.date <= "2026-03-31")] if name == "old→new" else \
                 df[(df.date >= "2006-09-30") & (df.date <= cutoff)]
            if name == "new→old":
                te = df[(df.date >= "2006-09-30") & (df.date <= "2015-12-31")]
            m = make_model(model)
            if model == "ens_hgb_huber":
                m = EnsHgbHuber()
            m.fit(tr[FEATS[feats]].astype(float).values, tr[TARGETS[target]].values)
            te = te.copy()
            te["pred"] = m.predict(te[FEATS[feats]].astype(float).values)
            mm = ic_metrics(te)
            transfer_rows.append(dict(cfg=cfg, split=name, ic=mm["ic"] if mm else np.nan,
                                      n_q=mm["n_q"] if mm else 0))
            print(f"  [transfer {cfg} {name}] IC={transfer_rows[-1]['ic']:.4f} (n_q={transfer_rows[-1]['n_q']})", flush=True)
    pd.DataFrame(transfer_rows).to_csv(f"{WORK}/transfer.csv", index=False)

def run_backtest(df, top_n=6):
    FINALISTS = ["lasso__F2__fwd6_rk", "enet__F2__fwd6_rk", "svrlin__F2__fwd6_z",
                 "lasso__F3__fwd6_rk", "huber__F2__fwd6_z", "huber__F2__fwd12_z",
                 "et__F5__fwd6_rk", "enet__F3__fwd6_rk"]
    cfgs = [c for c in FINALISTS if os.path.exists(f"{WORK}/pred1_{c}.csv")]
    # 基线: V3.7 纯分(无ML) 与 V4 原版 0.5 混合
    base_cfgs = [("V3.7_pure", None), ("V4huber_F2_fwd6_z", None)]
    results = []
    # 预载净值与基准
    import provider, backtest_local as bl
    codes = sorted(df.code.unique())
    navs = {}
    for c in codes:
        try:
            raw = provider.get_fund_nav(c)
            if raw is None or not len(raw):
                continue
            s = raw.set_index("date")["nav"].sort_index()
            s = s[~s.index.duplicated(keep="last")].dropna()
            if len(s) > 200:
                navs[c] = s
        except Exception:
            pass
    bench = provider.get_close_by_src("sina", "sh000300").dropna().sort_index()
    print(f"[bt] 净值 {len(navs)}/{len(codes)} | 基准 {bench.index[-1].date()}")

    import argparse
    def mkargs(**kw):
        a = argparse.Namespace(**dict(capital=100000.0, slots=10, cost_in=0.0015,
            cost_out=0.005, buy=70.0, sell=45.0, legacy=False, cash_yield=0.025,
            trail_stop=0.20, rebalance="quarterly", crisis=True, cppi=True,
            crisis_ma=200, crisis_vol_window=20, crisis_vol_q=0.80,
            pool_mode="pit-top", pit_top_n=100))
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    # S_v37
    from engine import resolve_weights
    s37 = {}
    for d, g in df.groupby(df.date.dt.strftime("%Y-%m-%d"), sort=True):
        water = g["water"].iloc[0]
        (wv, wa, wm), _ = resolve_weights(water if water == water else None)
        gg = g[["code", "r4", "r7", "val_cov", "val_pct", "trend_ok", "wr", "dc",
                "other_pen", "R_MDD", "water"]].copy()
        gg["wv"], gg["wa"], gg["wm"] = wv, wa, wm
        r = bl.score_from_raw(gg)
        s37[d] = dict(zip(r.code.astype(str), r.S_v37))
    df["S37"] = [s37.get(d, {}).get(c, np.nan) for d, c in zip(df.date.dt.strftime("%Y-%m-%d"), df.code)]

    def run_one(panel, qs, args):
        ec, tr = bl.simulate(panel, navs, bench, qs, args)
        yrs = (pd.Timestamp(qs[-1]) - pd.Timestamp(qs[0])).days / 365.25
        tot = ec.equity.iloc[-1] / args.capital - 1
        cagr = (1 + tot) ** (1 / yrs) - 1
        dd = ec.drawdown.min()
        win = (tr.net_ret > 0).mean() if len(tr) else np.nan
        return dict(tot=tot, cagr=cagr, dd=dd, calmar=cagr / abs(dd) if dd < 0 else np.nan,
                    trades=len(tr), win=win, hold=tr.hold_days.mean() if len(tr) else np.nan)

    def make_panel(pred_df, w_blend):
        p = pred_df.copy()
        p["pred100"] = p.groupby("date")["pred"].rank(pct=True) * 100
        p["S"] = (w_blend * p["pred100"] + (1 - w_blend) * p["S37"]).round(1)
        p["water"] = p.groupby("date")["water"].transform("first")
        p["date"] = p["date"].dt.strftime("%Y-%m-%d")   # simulate 用字符串键
        return p[["date", "code", "S", "water", "R_MDD"]]

    qs_all = [str(d.date()) for d in sorted(df.date.unique())]
    windows = {"full": qs_all,
               "w2015": [q for q in qs_all if q >= "2015-01-31"],
               "w2019": [q for q in qs_all if q >= "2019-01-31"]}

    for cfg in cfgs:
        pred = pd.read_csv(f"{WORK}/pred1_{cfg}.csv", parse_dates=["date"], dtype={"code": str})
        pred = pred.merge(df[["date", "code", "S37", "water", "R_MDD"]], on=["date", "code"], how="left")
        for wb in (0.5, 1.0, 0.3):
            panel = make_panel(pred, wb)
            for wname, qs in windows.items():
                args = mkargs()
                r = run_one(panel, qs, args)
                results.append(dict(cfg=f"{cfg}_w{wb}", window=wname, **r))
                print(f"  [bt {cfg} w{wb} {wname}] tot={r['tot']:+.1%} cagr={r['cagr']:+.2%} "
                      f"dd={r['dd']:.1%} calmar={r['calmar']:.2f} trades={r['trades']} win={r['win']:.0%}", flush=True)
    # V3.7 纯分基线
    p = df[["date", "code", "S37", "water", "R_MDD"]].rename(columns={"S37": "S"})
    p["date"] = p["date"].dt.strftime("%Y-%m-%d")
    for wname, qs in windows.items():
        r = run_one(p, qs, mkargs())
        results.append(dict(cfg="V3.7_pure", window=wname, **r))
        print(f"  [bt V3.7_pure {wname}] tot={r['tot']:+.1%} cagr={r['cagr']:+.2%} "
              f"dd={r['dd']:.1%} calmar={r['calmar']:.2f} trades={r['trades']} win={r['win']:.0%}", flush=True)
    out = pd.DataFrame(results)
    out.to_csv(f"{WORK}/backtest.csv", index=False)
    print("\n===== BACKTEST (按 full 窗口 Calmar 排序) =====")
    full = out[out.window == "full"].sort_values("calmar", ascending=False)
    print(full.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["screen", "verify", "backtest"])
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    df = load_panel()
    print(f"[data] {len(df)} 行 | fwd6 有效 {df.fwd6.notna().sum()} | 用时 {time.time()-t0:.0f}s")
    if args.phase == "screen":
        run_screen(df)
    elif args.phase == "verify":
        run_verify(df, top_n=args.top)
    else:
        run_backtest(df, top_n=args.top)
    print(f"[done] 总用时 {(time.time()-t0)/60:.1f} min")
