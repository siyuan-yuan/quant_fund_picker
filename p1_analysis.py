# -*- coding: utf-8 -*-
"""
Phase 1 因子诊断与正交化分析（P1-1 / P1-2 / P1-3 / P1-4）
输入: output/p1_panel/*.csv（P1-0 严格 PIT 全池面板）
协议: docs/优化计划_V4_2026-08.md §0.3 与 §2 P1 各节（全部变体预登记）
输出: output/p1/*.csv + docs/P1_*_报告.md
复现: ./.venv/bin/python p1_analysis.py [--only f1_1] [--min-n 30]
"""
import os, argparse, json, ast
import numpy as np
import pandas as pd
from scipy import stats

PANEL_DIR = os.path.join("output", "p1_panel")
OUT_DIR = os.path.join("output", "p1")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 因子登记（方向: 已乘 -1 的统一为"越高越好"） ----------------
FACTORS = {
    "value":      ("val_pct", -1),
    "trend_ma20": ("ma20_dist", +1),
    "alpha_ir":   ("wr", +1),
    "alpha_dc":   ("dc", -1),
    "mom4":       ("r4", +1),
    "mom7":       ("r7", +1),
    "risk_mdd":   ("R_MDD", -1),
    "risk_conc":  ("top3_conc", -1),
    "risk_small": ("smallcap_exp", -1),
    "eng_value":  ("F_value", +1),
    "eng_alpha":  ("F_alpha", +1),
    "eng_mom":    ("F_momentum", +1),
    "eng_S":      ("S_total", +1),
}
HORIZONS = ["fwd1", "fwd3", "fwd6"]
CORE6 = ["value", "trend_ma20", "alpha_ir", "alpha_dc", "mom4", "mom7"]


def load_panel(min_n=30):
    files = sorted(f for f in os.listdir(PANEL_DIR) if f.endswith(".csv") and f[0:2].isdigit())
    df = pd.concat([pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str}) for f in files],
                   ignore_index=True)
    if "date" not in df.columns:
        raise RuntimeError("面板缺少 date 列")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _parse_pens(x):
    """penalties 列: 内存中是 list[float], 经 CSV 往返后是 '[0.3, 0.15]' 字符串"""
    if x is None or (isinstance(x, float) and x != x):
        return []
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return [float(p) for p in v] if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    if isinstance(x, (list, tuple)):
        return [float(p) for p in x]
    return []


def _ic_row(g, fcol, hcol, direction):
    if len(g) < 10:
        return np.nan
    x, y = g[fcol].values * direction, g[hcol].values
    m = ~(np.isnan(x) | np.isnan(y)) if x.dtype != object else ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return stats.spearmanr(x[m], y[m])[0]


def quintile_spread(g, fcol, hcol, direction):
    x, y = g[fcol].values * direction, g[hcol].values
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 30:
        return np.nan
    q = pd.qcut(pd.Series(x[m]).rank(method="first"), 5, labels=False)
    yy = pd.Series(y[m])
    return float(yy[q == 4].mean() - yy[q == 0].mean())


# ================= P1-1 三因子体检 =================
def p1_1(df, min_n):
    d = df[df.groupby("date")["code"].transform("size") >= min_n]
    rows, spr = [], []
    for fname, (col, dirn) in FACTORS.items():
        if col not in d.columns:
            continue
        for h in HORIZONS:
            ic = [v for v in d.groupby("date").apply(
                lambda g: _ic_row(g, col, h, dirn), include_groups=False).values if v == v]
            ic = np.array(ic, dtype=float)
            ic = ic[ic == ic]
            sp = [v for v in d.groupby("date").apply(
                lambda g: quintile_spread(g, col, h, dirn), include_groups=False).values if v == v]
            sp = np.array(sp, dtype=float)
            sp = sp[sp == sp]
            rows.append(dict(factor=fname, horizon=h, n_months=len(ic),
                             ic_mean=round(float(ic.mean()), 4) if len(ic) else np.nan,
                             ic_sd=round(float(ic.std()), 4) if len(ic) else np.nan,
                             t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                             if len(ic) > 2 and ic.std() > 0 else np.nan,
                             pct_pos=round(float((ic > 0).mean()), 3) if len(ic) else np.nan,
                             spread_q5q1=round(float(sp.mean()), 4) if len(sp) else np.nan))
            for v in sp:
                spr.append(dict(factor=fname, horizon=h, spread=v))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, "f1_1_ic_table.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(spr).to_csv(os.path.join(OUT_DIR, "f1_1_spreads.csv"), index=False, encoding="utf-8-sig")

    # 相关矩阵（核心6因子, 分时期）
    def corr_mat(sub):
        mats = []
        for _, g in sub.groupby("date"):
            if len(g) < min_n:
                continue
            row = {}
            for a in CORE6:
                for b in CORE6:
                    ca, da = FACTORS[a]
                    cb, db = FACTORS[b]
                    x, y = g[ca].values * da, g[cb].values * db
                    m = ~(np.isnan(x) | np.isnan(y))
                    if m.sum() >= min_n and np.std(x[m]) > 1e-12 and np.std(y[m]) > 1e-12:
                        row[f"{a}|{b}"] = stats.spearmanr(x[m], y[m])[0]
            if row:
                mats.append(row)
        mm = pd.DataFrame(mats).mean()
        idx = [(s[0], s[1]) for s in mm.index]
        return mm.to_frame("corr_mean")

    c_all = corr_mat(d)
    c_all.to_csv(os.path.join(OUT_DIR, "f1_1_corr_all.csv"), encoding="utf-8-sig")
    c1 = corr_mat(d[d["date"] < "2020-01-01"])
    c1.to_csv(os.path.join(OUT_DIR, "f1_1_corr_2015_2019.csv"), encoding="utf-8-sig")
    c2 = corr_mat(d[d["date"] >= "2020-01-01"])
    c2.to_csv(os.path.join(OUT_DIR, "f1_1_corr_2020_2026.csv"), encoding="utf-8-sig")

    # 分年度 IC（fwd6）
    dy = d.copy()
    dy["year"] = dy["date"].dt.year
    yrows = []
    for fname in FACTORS:
        col, dirn = FACTORS[fname]
        if col not in dy.columns:
            continue
        for y, g in dy.groupby("year"):
            ic = [v for v in g.groupby("date").apply(
                lambda gg: _ic_row(gg, col, "fwd6", dirn), include_groups=False).values if v == v]
            ic = np.array([v for v in ic if v == v])
            if len(ic):
                yrows.append(dict(factor=fname, year=y, n_months=len(ic),
                                  ic_mean=round(float(ic.mean()), 4),
                                  t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                                  if ic.std() > 0 else np.nan))
    pd.DataFrame(yrows).to_csv(os.path.join(OUT_DIR, "f1_1_yearly_ic.csv"),
                               index=False, encoding="utf-8-sig")

    # 分组: 主动 vs 被动 / A股 vs 海外
    sub_rows = []
    d2 = d.copy()
    d2["is_ov"] = d2["top1_style"].isin(
        {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}).fillna(False)
    for tag, sub in [("all", d2), ("active", d2[~d2["is_passive"]]),
                     ("passive", d2[d2["is_passive"]]),
                     ("ashare", d2[~d2["is_ov"]]), ("overseas", d2[d2["is_ov"]])]:
        for fname in CORE6 + ["eng_S"]:
            col, dirn = FACTORS[fname]
            ic = [v for v in sub.groupby("date").apply(
                lambda g: _ic_row(g, col, "fwd6", dirn), include_groups=False).values if v == v]
            ic = np.array([v for v in ic if v == v])
            if len(ic) > 3:
                sub_rows.append(dict(group=tag, factor=fname, n_months=len(ic),
                                     ic_mean=round(float(ic.mean()), 4),
                                     t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                                     if ic.std() > 0 else np.nan))
    pd.DataFrame(sub_rows).to_csv(os.path.join(OUT_DIR, "f1_1_subgroup_ic.csv"),
                                  index=False, encoding="utf-8-sig")
    return tbl


# ================= P1-2 正交化与权重裁决 =================
def _ols_resid(y, X):
    """y: (n,), X: (n,k) 含常数项 → 残差"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def p1_2(df, min_n):
    from engine import resolve_weights
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    for c in ["F_value", "F_alpha", "F_momentum"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    def synth(weights, fv, fa, fm, pens):
        """engine 同式: 缺失因子归一化 + 乘法惩罚"""
        out = np.full(len(fv), np.nan)
        for i in range(len(fv)):
            num = den = 0.0
            for w, v in zip(weights, (fv[i], fa[i], fm[i])):
                if v is None or (isinstance(v, float) and v != v):
                    continue
                num += w * v
                den += w
            base = num / den if den > 1e-9 else 0.0
            s = base
            for p in (pens[i] or []):
                s *= (1 - p)
            out[i] = max(0.0, min(100.0, s))
        return out

    months = sorted(d["date"].unique())
    idx_of = {m: i for i, m in enumerate(months)}
    # 每决策月 regime 权重（与面板构建同式: 水位≤0.20 → 左侧 0.55/0.35/0.10）
    w_base_month = {}
    for m, g in d.groupby("date"):
        wtr = g["water"].iloc[0]
        wtr = float(wtr) if wtr == wtr else np.nan
        (wv, wa, wm), _ = resolve_weights(wtr)
        w_base_month[m] = [wv, wa, wm]

    # 扩展窗 IC 权重（≥24 月前窗, fwd6, 基线因子层, 无前视）
    base_ic = {m: {} for m in months}
    for m, g in d.groupby("date"):
        for fn in ["eng_value", "eng_alpha", "eng_mom"]:
            col, dirn = FACTORS[fn]
            base_ic[m][fn] = _ic_row(g, col, "fwd6", dirn)

    def ic_weights(i, m):
        if i < 24:
            b = w_base_month[m]
            return {"eng_value": b[0], "eng_alpha": b[1], "eng_mom": b[2]}
        w = {}
        for fn in ["eng_value", "eng_alpha", "eng_mom"]:
            vals = [base_ic[months[j]][fn] for j in range(i)
                    if base_ic[months[j]][fn] == base_ic[months[j]][fn]]
            w[fn] = abs(float(np.mean(vals))) if vals else 1e-6
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    recs = []
    for m, g in d.groupby("date"):
        i = idx_of[m]
        g = g.reset_index(drop=True)
        fv, fa, fm = (g[c].values for c in ["F_value", "F_alpha", "F_momentum"])
        pens_l = g["penalties"].apply(_parse_pens).tolist()
        # 秩空间正交化 (全非缺失子集)
        msk = ~(np.isnan(fv) | np.isnan(fa) | np.isnan(fm))
        a_perp = np.full(len(g), np.nan)
        m_perp = np.full(len(g), np.nan)
        if msk.sum() >= 30:
            rv = pd.Series(fv[msk]).rank(pct=True).values
            ra = pd.Series(fa[msk]).rank(pct=True).values
            rm = pd.Series(fm[msk]).rank(pct=True).values
            X1 = np.column_stack([np.ones_like(rv), rv])
            rp_a = _ols_resid(ra, X1)
            X2 = np.column_stack([np.ones_like(rv), rv, rp_a])
            rp_m = _ols_resid(rm, X2)
            # 残差 → 0-100 截面分位
            a_perp[msk] = 100 * pd.Series(rp_a).rank(pct=True).values
            m_perp[msk] = 100 * pd.Series(rp_m).rank(pct=True).values
        w_ic = ic_weights(i, m)
        for name, (fvv, fvv_a, fvv_m), w in [
                ("S0", (fv, fa, fm), None),
                ("S1_正交_原权", (fv, a_perp, m_perp), None),
                ("S2_正交_IC权", (fv, a_perp, m_perp), w_ic),
                ("S3_原始_IC权", (fv, fa, fm), w_ic)]:
            ws = [w["eng_value"], w["eng_alpha"], w["eng_mom"]] if w is not None \
                else w_base_month[m]
            sc = synth(ws, fvv, fvv_a, fvv_m, pens_l)
            recs.append(pd.DataFrame(dict(date=m, code=g["code"], variant=name,
                                          S=sc, fwd1=g["fwd1"], fwd3=g["fwd3"], fwd6=g["fwd6"])))
    long_df = pd.concat(recs, ignore_index=True)
    # 校验: S0(重建) vs 面板 S_total (regime权重同式)
    s0 = long_df[long_df["variant"] == "S0"].set_index(["date", "code"])["S"]
    key = d.set_index(["date", "code"])
    align = key["S_total"].index.join(s0.index, how="inner")
    diff = float(np.nanmax(np.abs(s0.loc[align].values - key.loc[align, "S_total"].values)))
    print(f"[P1-2] S0重建 vs 面板S_total 最大差 = {diff:.2f} (应≈0.05, 仅四舍五入)", flush=True)
    long_df.to_csv(os.path.join(OUT_DIR, "f1_2_scores.csv"), index=False, encoding="utf-8-sig")

    # 月度 IC 与配对差 t
    piv = {}
    for h in HORIZONS:
        ics = {}
        for v in ["S0", "S1_正交_原权", "S2_正交_IC权", "S3_原始_IC权"]:
            sub = long_df[long_df["variant"] == v]
            s = sub.groupby("date").apply(
                lambda g: _ic_row(g, "S", h, +1), include_groups=False).dropna()
            ics[v] = s
        all_s = pd.DataFrame(ics)
        all_s[f"pair_S1"] = all_s["S1_正交_原权"] - all_s["S0"]
        all_s[f"pair_S2"] = all_s["S2_正交_IC权"] - all_s["S0"]
        all_s[f"pair_S3"] = all_s["S3_原始_IC权"] - all_s["S0"]
        piv[h] = all_s
    rows = []
    for h, all_s in piv.items():
        for v in ["S0", "S1_正交_原权", "S2_正交_IC权", "S3_原始_IC权"]:
            s = all_s[v].dropna()
            rows.append(dict(horizon=h, variant=v, n_months=len(s),
                             ic_mean=round(float(s.mean()), 4),
                             t=round(float(s.mean() / (s.std() / np.sqrt(len(s)))), 2)
                             if s.std() > 0 else np.nan))
        for pv in ["pair_S1", "pair_S2", "pair_S3"]:
            s = all_s[pv].dropna()
            rows.append(dict(horizon=h, variant=pv + "_配对差", n_months=len(s),
                             ic_mean=round(float(s.mean()), 4),
                             t=round(float(s.mean() / (s.std() / np.sqrt(len(s)))), 2)
                             if s.std() > 0 else np.nan))
        # 近3年子窗口 (2023-04+)
        s3 = all_s[all_s.index >= pd.Timestamp("2023-04-01")]
        for pv in ["pair_S1", "pair_S2", "pair_S3"]:
            s = s3[pv].dropna()
            rows.append(dict(horizon=h + "_2023+", variant=pv + "_配对差", n_months=len(s),
                             ic_mean=round(float(s.mean()), 4) if len(s) else np.nan,
                             t=round(float(s.mean() / (s.std() / np.sqrt(len(s)))), 2)
                             if len(s) > 2 and s.std() > 0 else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "f1_2_stats.csv"),
                              index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


# ================= P1-3 动量窗口敏感性 =================
MOMV = ["r4", "r7", "r3s1", "r3s0", "r4s0", "r4s2", "r7s2", "r12s1"]
MOM_PAIRS = [
    ("base_r4r7", "r4", "r7"),
    ("alt_r4s2r7s2", "r4s2", "r7s2"),
    ("alt_r3s1r7", "r3s1", "r7"),
    ("alt_r12s1r7", "r12s1", "r7"),
]


def p1_3(df, min_n):
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    rows = []
    # 单窗口
    for w in MOMV:
        for h in HORIZONS:
            ic = [v for v in d.groupby("date").apply(
                lambda g: _ic_row(g, w, h, +1), include_groups=False).values if v == v]
            ic = np.array([v for v in ic if v == v])
            sp = [v for v in d.groupby("date").apply(
                lambda g: quintile_spread(g, w, h, +1), include_groups=False).values if v == v]
            rows.append(dict(item=f"win_{w}", horizon=h, n_months=len(ic),
                             ic_mean=round(float(ic.mean()), 4) if len(ic) else np.nan,
                             t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                             if len(ic) > 2 and ic.std() > 0 else np.nan,
                             spread_q5q1=round(float(np.mean(sp)), 4) if len(sp) else np.nan))
    # 双窗组合 M1 公式
    def pair_score(g, wa, wb):
        ra = pd.to_numeric(g[wa], errors="coerce").rank(pct=True)
        rb = pd.to_numeric(g[wb], errors="coerce").rank(pct=True)
        m = 0.6 * ra + 0.4 * rb
        return 100 * m.clip(0, 1)
    for name, a, b in MOM_PAIRS:
        for h in HORIZONS:
            ic = []
            for _, g in d.groupby("date"):
                s = pair_score(g, a, b).values
                y = g[h].values
                mm = ~(np.isnan(s) | np.isnan(y))
                if mm.sum() < 30:
                    continue
                ic.append(stats.spearmanr(s[mm], y[mm])[0])
            ic = np.array([v for v in ic if v == v])
            rows.append(dict(item=f"pair_{name}", horizon=h, n_months=len(ic),
                             ic_mean=round(float(ic.mean()), 4) if len(ic) else np.nan,
                             t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                             if len(ic) > 2 and ic.std() > 0 else np.nan))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, "f1_3_momentum.csv"), index=False, encoding="utf-8-sig")
    # 配对差 t (vs base, fwd6) — 逐组直接计算, 不用 groupby.apply
    for name, a, b in MOM_PAIRS[1:]:
        diffs = []
        for _, g in d.groupby("date"):
            s_b = pair_score(g, "r4", "r7").values
            s_a = pair_score(g, a, b).values
            y = g["fwd6"].values.astype(float)
            ok = np.isfinite(s_b) & np.isfinite(s_a) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            diffs.append(stats.spearmanr(s_a[ok], y[ok])[0] - stats.spearmanr(s_b[ok], y[ok])[0])
        diffs = np.array([v for v in diffs if v == v])
        rows.append(dict(item=f"pair_diff_{name}", horizon="fwd6", n_months=len(diffs),
                         ic_mean=round(float(diffs.mean()), 4) if len(diffs) else np.nan,
                         t=round(float(diffs.mean() / (diffs.std() / np.sqrt(len(diffs)))), 2)
                         if len(diffs) > 2 and diffs.std() > 0 else np.nan))
    tbl2 = pd.DataFrame(rows)
    tbl2.to_csv(os.path.join(OUT_DIR, "f1_3_momentum.csv"), index=False, encoding="utf-8-sig")
    return tbl2


# ================= P1-4 惩罚项因子化复核 =================
def p1_4(df, min_n):
    import risk as R
    d = df[df.groupby("date")["code"].transform("size") >= min_n].copy()
    # (a) 惩罚量作为因子的 IC
    rows = []
    for fname in ["risk_mdd", "risk_conc", "risk_small"]:
        col, dirn = FACTORS[fname]
        for h in HORIZONS:
            ic = [v for v in d.groupby("date").apply(
                lambda g: _ic_row(g, col, h, dirn), include_groups=False).values if v == v]
            ic = np.array([v for v in ic if v == v])
            rows.append(dict(item=f"factor_{fname}", horizon=h, n_months=len(ic),
                             ic_mean=round(float(ic.mean()), 4) if len(ic) else np.nan,
                             t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                             if len(ic) > 2 and ic.std() > 0 else np.nan))
    # (b) 增量 IC: 对 S_total 残差再测
    for fname in ["risk_mdd", "risk_conc", "risk_small"]:
        col, dirn = FACTORS[fname]
        ic = []
        for m, g in d.groupby("date"):
            x = (g[col].values * dirn).astype(float)
            s = g["S_total"].values.astype(float)
            y = g["fwd6"].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(s) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            res_s = _ols_resid(y[ok], np.column_stack([np.ones(ok.sum()), s[ok]]))
            xr = x[ok]
            if np.std(xr) > 1e-12 and np.std(res_s) > 1e-12:
                ic.append(stats.spearmanr(xr, res_s)[0])
        ic = np.array([v for v in ic if v == v])
        rows.append(dict(item=f"incremental_{fname}", horizon="fwd6", n_months=len(ic),
                         ic_mean=round(float(ic.mean()), 4) if len(ic) else np.nan,
                         t=round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2)
                         if len(ic) > 2 and ic.std() > 0 else np.nan))
    # (c) MDD 乘法惩罚价值: S_total vs S_without_mdd
    d2 = d.copy()
    d2["p_mdd"] = [R.mdd_smooth_penalty(rm, w)
                   if rm == rm else 0.0
                   for rm, w in zip(d2["R_MDD"], d2["water"])]
    d2["S_without_mdd"] = np.where(
        d2["p_mdd"] > 0,
        np.clip(d2["S_total"] / (1 - d2["p_mdd"].clip(0, 0.99)), 0, 100),
        d2["S_total"])
    ic_s, ic_w = [], []
    for m, g in d2.groupby("date"):
        y = g["fwd6"].values.astype(float)
        for c in ["S_total", "S_without_mdd"]:
            x = g[c].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 30:
                (ic_s if c == "S_total" else ic_w).append(stats.spearmanr(x[ok], y[ok])[0])
    ic_s, ic_w = np.array(ic_s), np.array(ic_w)
    diff = ic_w - ic_s
    rows.append(dict(item="S_total_fwd6", horizon="fwd6", n_months=len(ic_s),
                     ic_mean=round(float(ic_s.mean()), 4),
                     t=round(float(ic_s.mean() / (ic_s.std() / np.sqrt(len(ic_s)))), 2)))
    rows.append(dict(item="S_without_mdd_fwd6", horizon="fwd6", n_months=len(ic_w),
                     ic_mean=round(float(ic_w.mean()), 4),
                     t=round(float(ic_w.mean() / (ic_w.std() / np.sqrt(len(ic_w)))), 2)))
    rows.append(dict(item="diff_Sw-S", horizon="fwd6", n_months=len(diff),
                     ic_mean=round(float(diff.mean()), 4),
                     t=round(float(diff.mean() / (diff.std() / np.sqrt(len(diff)))), 2)
                     if diff.std() > 0 else np.nan))
    # (d) 集中度惩罚消融: 引擎规则 top3>0.70 且 wr<0.5 且主动 → 乘(1-0.4)
    d2["p_conc"] = np.where(
        (~d2["is_passive"]) & (d2["top3_conc"] > 0.70)
        & (d2["wr"] < 0.5) & d2["top3_conc"].notna() & d2["wr"].notna(), 0.4, 0.0)
    d2["S_without_conc"] = np.where(
        d2["p_conc"] > 0, np.clip(d2["S_total"] / (1 - d2["p_conc"]), 0, 100), d2["S_total"])
    ic_c = []
    for m, g in d2.groupby("date"):
        y = g["fwd6"].values.astype(float)
        x = g["S_without_conc"].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 30:
            ic_c.append(stats.spearmanr(x[ok], y[ok])[0])
    ic_c = np.array(ic_c)
    diff_c = ic_c - ic_s
    rows.append(dict(item="S_without_conc_fwd6", horizon="fwd6", n_months=len(ic_c),
                     ic_mean=round(float(ic_c.mean()), 4),
                     t=round(float(ic_c.mean() / (ic_c.std() / np.sqrt(len(ic_c)))), 2)))
    rows.append(dict(item="diff_conc_Sw-S", horizon="fwd6", n_months=len(diff_c),
                     ic_mean=round(float(diff_c.mean()), 4),
                     t=round(float(diff_c.mean() / (diff_c.std() / np.sqrt(len(diff_c)))), 2)
                     if diff_c.std() > 0 else np.nan))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, "f1_4_penalty.csv"), index=False, encoding="utf-8-sig")
    return tbl


def _report(tbl, path, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n> 自动生成于 p1_analysis.py。数据: output/p1_panel 严格PIT全池面板。\n\n")
        f.write("```\n" + tbl.to_string(index=False) + "\n```\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="f1_1|f1_2|f1_3|f1_4")
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()
    df = load_panel()
    n_months = df["date"].nunique()
    n_ok = int((df.groupby("date")["code"].size() >= args.min_n).sum())
    print(f"[panel] 决策月={n_months}  有效截面(≥{args.min_n})月数≈{n_ok}  总行={len(df)}", flush=True)
    if args.only in (None, "f1_1"):
        t = p1_1(df, args.min_n)
        print(t[t["horizon"] == "fwd6"].to_string(index=False))
    if args.only in (None, "f1_2"):
        p1_2(df, args.min_n)
    if args.only in (None, "f1_3"):
        p1_3(df, args.min_n)
    if args.only in (None, "f1_4"):
        p1_4(df, args.min_n)
    print("[saved] output/p1/*", flush=True)


if __name__ == "__main__":
    main()
