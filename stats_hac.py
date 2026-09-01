# -*- coding: utf-8 -*-
"""V5 预登记 #17（M1.2）：Newey-West HAC 统计包（全仓库历史 t 值复核的唯一口径）

设计锚点（预登记 §0.3）：
  - Bartlett 核，滞后阶 L 规则：fwd6 月度重叠收益 → L=5；fwd3 → L=3；fwd1 → L=1；
    季度频率下 fwd6（≈2 季重叠）→ L=2；fwd12 → L=11。
  - 变体/模型间一律**配对差序列**做 HAC t；严禁"两条序列各自 t 值互比"。
  - 显著性门：|t| ≥ 2（简易双侧 95%）；翻案需 HAC 口径翻转且双窗同向。

实现要点：均值推断的 HAC 方差 Var(x̄) = [γ0 + 2Σ_{k=1..L}(1−k/(L+1))γk] / T；
对 fwd6 月度重叠 IC 序列，这为重叠结构提供了正确的 O(h) 膨胀修正
（蒙特卡洛 + 闭式已知答案验证，见 tests/test_research_invariants.py）。
"""
from __future__ import annotations

import numpy as np

# 预登记滞后阶表（唯一真相源）
HORIZON_LAGS = {
    ("fwd6", "monthly"): 5,
    ("fwd3", "monthly"): 3,
    ("fwd1", "monthly"): 1,
    ("fwd12", "monthly"): 11,
    ("fwd6", "quarterly"): 2,
}
# 以键串访问的别名表（r31-r35 脚本用）
HORIZON_LAGS.update({f"{h}_{f}": HORIZON_LAGS[(h, f)] for h, f in
                     [("fwd6", "monthly"), ("fwd3", "monthly"), ("fwd1", "monthly"),
                      ("fwd12", "monthly"), ("fwd6", "quarterly")]})


def autocov(x: np.ndarray, k: int) -> float:
    n = len(x)
    xm = x - x.mean()
    return float(np.dot(xm[k:], xm[:n - k]) / n)


def nw_var_mean(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett) 均值方差估计：var(mean) = S*(0:L)/T。可为负时截断到 0。"""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 3:
        return np.nan
    s = autocov(x, 0)
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * autocov(x, k)
    return max(s, 0.0) / T


def nw_tstat(x: np.ndarray, lag: int) -> dict:
    """返回 dict(n, mean, t_naive, t_hac, lag)。t_naive 为经典 t（对照展示 t 膨胀）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return dict(n=T, mean=np.nan, t_naive=np.nan, t_hac=np.nan, lag=lag)
    mu = float(x.mean())
    sd_naive = float(x.std(ddof=1))
    t_naive = mu / (sd_naive / np.sqrt(T)) if sd_naive > 0 else np.nan
    v_hac = nw_var_mean(x, lag)
    t_hac = mu / np.sqrt(v_hac) if v_hac > 0 else np.nan
    return dict(n=T, mean=mu, t_naive=t_naive, t_hac=t_hac, lag=lag)


def paired_diff_hac_t(ic_a: np.ndarray, ic_b: np.ndarray, horizon: str = "fwd6",
                      freq: str = "monthly") -> dict:
    """变体间裁决的唯一合法统计量：逐日 IC 差序列的 HAC t（NaN 对齐丢弃）。"""
    a, b = np.asarray(ic_a, dtype=float), np.asarray(ic_b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    lag = HORIZON_LAGS.get((horizon, freq), HORIZON_LAGS.get(f"{horizon}_{freq}", 1))
    return nw_tstat(d, lag)


def monthly_ic_lag(horizon_months: int, freq: str = "monthly") -> int:
    """按前瞻月数推滞后阶：L = h − 1（重叠收益需要覆盖到第 h−1 阶自相关）。"""
    if freq == "quarterly":
        # 季度样本量小，预登记取保守 L = h/3（fwd6→2, 较 MA(1) 最少要求多带 1 阶）
        return max(0, horizon_months // 3)
    return max(0, horizon_months - 1)


def series_judgement(x, lag: int, gate: float = 2.0) -> dict:
    """V5 §0.3 统一判定行：IC/配对差序列 → naive t 与 HAC t 并列 + 翻转标记。

    gate=显著性门（预登记 2.0）。返回 dict 可直接入表。"""
    st = nw_tstat(x, lag)
    t_n, t_h = st["t_naive"], st["t_hac"]
    sig_n = bool(np.isfinite(t_n) and abs(t_n) >= gate)
    sig_h = bool(np.isfinite(t_h) and abs(t_h) >= gate)
    return dict(n_months=st["n"], ic_mean=round(st["mean"], 4) if st["mean"] == st["mean"] else np.nan,
                t_naive=round(t_n, 2) if t_n == t_n else np.nan,
                t_hac=round(t_h, 2) if t_h == t_h else np.nan,
                sig_naive=sig_n, sig_hac=sig_h, flip=sig_n != sig_h)
