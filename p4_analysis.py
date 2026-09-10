#!/usr/bin/env python3
"""Phase 4 波次1：多重检验审计（DSR + CSCV + 配对t Holm/BH）与 OOS 健康度快照。

预登记: docs/优化计划_V4_2026-08.md §2 预登记 #6（运行前写入）。
这是审计而非裁决: 仅当某被否决变体 DSR>0.97 且 Holm 显著 且 双窗 Calmar/MaxDD 双改善时
才推翻既往裁决(预期不发生; 发生则如实改判)。

产物: output/p4/p4_ds.csv, p4_cscv.csv, p4_pair_audit.csv, p4_oos_ic.csv
复现: ./.venv/bin/python p4_analysis.py
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output", "p4")

CURVES_CSV = os.path.join(ROOT, "output", "p3", "p3_curves.csv")
PANEL_DIR = os.path.join(ROOT, "output", "p1_panel")
BASELINE = "v38_cost"
EXPERIMENTAL = ["vol_target", "no_overlay", "conviction", "cap25", "cap30", "cap35",
                "p90", "sell40", "xm_fix3", "xm_dyn", "roll_hwm", "cap_hyst"]
ALL_CONFIGS = ["base_zero", BASELINE] + EXPERIMENTAL
N_TRIALS_P3 = len(ALL_CONFIGS)          # 14
N_TRIALS_ALL = 34                       # + P1-2×3, P1-3×11, P1-4×3, P2-A×3, P2-2×3
G_EULER = 0.5772156649015329


def monthly_returns(curves: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    eq = curves[names]
    m = eq.resample("ME").last()
    return m.pct_change().dropna()


# ---------------- DSR (Bailey & López de Prado) ----------------

def dsr_metrics(r: pd.Series, n_trials: int) -> dict:
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 0:
        return dict(T=T, SR=np.nan, DSR=np.nan)
    sr = mu / sd
    g3 = stats.skew(r, bias=True)
    g4 = stats.kurtosis(r, bias=True) + 3.0   # 非超额峰度 (E[r^4]/σ^4)
    var_factor = 1.0 - g3 * sr + (g4 - 1.0) * sr ** 2 / 4.0
    var_factor = max(var_factor, 1e-9)
    sigma_sr = math.sqrt(var_factor / T)
    # 零假设下 N 个试验的最大 SR 的期望 (正态近似)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0 = sigma_sr * ((1 - G_EULER) * z1 + G_EULER * z2)
    dsr = stats.norm.cdf((sr - sr0) * math.sqrt(T) / math.sqrt(var_factor))
    return dict(T=T, SR=round(sr, 4), skew=round(float(g3), 3), kurt=round(float(g4), 3),
                sigma_SR=round(sigma_sr, 4), SR0=round(sr0, 4), DSR=round(float(dsr), 4))


def cscv_pbo(rets: pd.DataFrame, n_groups: int = 8) -> dict:
    """C(8,4) 平衡划分的 PBO (Bailey & López de Prado)。"""
    n = len(rets)
    if n % n_groups != 0:
        n = (n // n_groups) * n_groups
        rets = rets.iloc[:n]
    idx = np.arange(n).reshape(n_groups, -1)
    folds = [rets.iloc[idx[i]] for i in range(n_groups)]
    combos = list(itertools.combinations(range(n_groups), n_groups // 2))
    overfit_flags, chosen_counts = [], {}
    for combo in combos:
        tr, te = np.array(combo), np.array(sorted(set(range(n_groups)) - set(combo)))
        tr_ret = pd.concat([folds[i] for i in tr]).mean()
        te_all = pd.concat([folds[i] for i in te])
        te_means = te_all.mean()
        tr_means = tr_ret
        best = tr_means.idxmax()
        chosen_counts[best] = chosen_counts.get(best, 0) + 1
        overfit_flags.append(bool(te_means[best] < te_means.median()))
    pbo = float(np.mean(overfit_flags))
    return dict(n=len(rets), n_groups=n_groups, n_partitions=len(combos),
                fold_len=n // n_groups, PBO=round(pbo, 4),
                chosen_counts=chosen_counts)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def bh_q(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    q = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvals[i] * m / (rank + 1))
        q[i] = running
    return q


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = pd.read_csv(CURVES_CSV, parse_dates=["date"]).set_index("date")
    have = [c for c in ALL_CONFIGS if c in curves.columns]
    rets = monthly_returns(curves[have], have)
    print(f"月度收益矩阵: {rets.shape[0]} 月 × {rets.shape[1]} 配置")

    # ---------- DSR ----------
    rows = []
    for name in have:
        for tag, ntr in [("N=14", N_TRIALS_P3), ("N=34", N_TRIALS_ALL)]:
            m = dsr_metrics(rets[name], ntr)
            m.update(variant=name, N=tag)
            rows.append(m)
    ds = pd.DataFrame(rows)
    ds.to_csv(os.path.join(OUT_DIR, "p4_ds.csv"), index=False, encoding="utf-8-sig")
    print("DSR (N=14):")
    print(ds[ds["N"] == "N=14"][["variant", "SR", "SR0", "DSR"]]
          .sort_values("DSR", ascending=False).to_string(index=False))

    # ---------- CSCV ----------
    c = cscv_pbo(rets[have], n_groups=8)
    cc = c.pop("chosen_counts")
    pd.DataFrame([dict(c, chosen_counts=str(cc))]).to_csv(
        os.path.join(OUT_DIR, "p4_cscv.csv"), index=False, encoding="utf-8-sig")
    print(f"CSCV: PBO={c['PBO']}, 分区={c['n_partitions']}, 被选中次数={cc}")

    # ---------- 配对 t 审计 ----------
    rows = []
    for name in EXPERIMENTAL:
        if name not in rets.columns:
            continue
        d = (rets[name] - rets[BASELINE]).dropna()
        t = stats.ttest_1samp(d, 0.0).statistic
        p = float(stats.t.sf(abs(t), len(d) - 1) * 2)
        rows.append(dict(variant=name, n=int(len(d)), mean_diff_bps=round(d.mean() * 1e4, 1),
                         t=round(t, 3), p=round(p, 5)))
    audit = pd.DataFrame(rows)
    pv = audit["p"].tolist()
    audit["holm_p"] = [round(x, 5) for x in holm(pv)]
    audit["bh_q"] = [round(x, 5) for x in bh_q(pv)]
    audit = audit.sort_values("t", ascending=False)
    audit.to_csv(os.path.join(OUT_DIR, "p4_pair_audit.csv"), index=False, encoding="utf-8-sig")
    print("\n配对 t 审计 (vs v38_cost, 月度):")
    print(audit.to_string(index=False))

    # ---------- P4-2 OOS 健康度 ----------
    sys.path.insert(0, ROOT)
    from p1_analysis import _ic_row
    files = sorted(f for f in os.listdir(PANEL_DIR) if f.endswith(".csv") and f[0:2].isdigit())
    parts = [pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str}) for f in files]
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.groupby("date")["code"].transform("size") >= 30]

    def raw_ic(col: str) -> pd.Series:
        ics = []
        for dt, g in panel.groupby("date", sort=True):
            ics.append((dt, _ic_row(g, col, "fwd6", 1.0)))
        return pd.Series([v for _, v in ics], index=[dt for dt, _ in ics])

    oos_rows = []
    for col in ["S_total", "F_momentum", "F_alpha"]:
        r_raw = raw_ic(col)
        full_mean = r_raw.dropna().mean()      # 全期原始 IC 均值 (分母, 同 P1 口径)
        r = r_raw.rolling(12).mean()
        valid = r.dropna()
        # 尾部月份 fwd6 未成熟 → 取最后一个有效 12 月窗 (as_of = 该窗最后一月的决策日)
        last12, as_of = valid.iloc[-1], valid.index[-1]
        oos_rows.append(dict(factor=col, full_mean=round(full_mean, 4),
                             rolling_12m_latest=round(last12, 4),
                             ratio=round(last12 / full_mean, 3) if full_mean else np.nan,
                             alert=bool(last12 / full_mean < 0.5) if full_mean else np.nan,
                             as_of=str(as_of.date())))
    oos = pd.DataFrame(oos_rows)
    oos.to_csv(os.path.join(OUT_DIR, "p4_oos_ic.csv"), index=False, encoding="utf-8-sig")
    print("\nOOS 健康度 (滚动12月 IC fwd6):")
    print(oos.to_string(index=False))

    # ---------- OOS 衰减分解（告警触发时的标准诊断, 记录不改模型） ----------
    panel["is_ov"] = panel["top1_style"].isin(
        {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}).fillna(False)

    def raw_ic(col: str, dirn: float, sub: pd.DataFrame) -> pd.Series:
        ics = []
        for dt, g in sub.groupby("date", sort=True):
            ics.append((dt, _ic_row(g, col, "fwd6", dirn)))
        return pd.Series([v for _, v in ics], index=[dt for dt, _ in ics])

    dec_rows = []
    for col, dirn in [("F_value", 1.0), ("wr", 1.0), ("dc", -1.0), ("F_alpha", 1.0),
                      ("F_momentum", 1.0), ("S_total", 1.0)]:
        s = raw_ic(col, dirn, panel).dropna()
        dec_rows.append(dict(factor=col, direction=dirn,
                             full=round(s.mean(), 4),
                             last24m=round(s[s.index >= "2024-01-01"].mean(), 4),
                             last12m_2025=round(s[s.index >= "2025-01-01"].mean(), 4)))
    for tag, sub in [("active", panel[~panel.is_passive]), ("passive", panel[panel.is_passive]),
                     ("ashare", panel[~panel.is_ov]), ("overseas", panel[panel.is_ov])]:
        for col in ["F_alpha", "wr"]:
            s = raw_ic(col, 1.0, sub).dropna()
            dec_rows.append(dict(factor=f"{col}@{tag}", direction=1.0,
                                 full=round(s.mean(), 4),
                                 last24m=round(s[s.index >= "2024-01-01"].mean(), 4),
                                 last12m_2025=round(s[s.index >= "2025-01-01"].mean(), 4)))
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(os.path.join(OUT_DIR, "p4_oos_decay.csv"), index=False, encoding="utf-8-sig")
    print("\nOOS 衰减分解 (IC fwd6):")
    print(dec.to_string(index=False))
    print("完成 → output/p4/")


if __name__ == "__main__":
    main()
