# -*- coding: utf-8 -*-
"""
V6 / T1.1 —— 动态 RBSA：时变风格暴露路径 β_{i,t}

预登记依据：docs/优化计划_V6_底层表示重构_2026-09.md §3 T1

现行生产口径（对照臂 B0，`rbsa.rbsa_weights` + `engine.py:122`）：
    r^bench_t = ŵ_q' f_t ,  ∀t ∈ [q-3y, q]
即用**决策日那一组 60 日窗权重**去解释**过去三年**的基金收益。这不是 look-ahead
（ŵ_q 只用 ≤q 的数据），但是模型错设：基金的风格漂移收益会成片掉进残差，
被 F_alpha 记成"经理 alpha"。

本模块给出时变替代：
    r_{i,t} = α_{i,t} + β_{i,t}' f_t + ε_{i,t}
    β_{i,t} = β_{i,t-1} + η_{i,t},   η ~ N(0, Q)

【PIT 硬约束（违反即产物作废）】
本模块只提供**滤波值 filtered**（β_{t|t}，仅用 ≤t 的信息）。
RTS smoother (`rts_smooth`) 仅供诊断作图，其输出**禁止**进入任何特征/标签；
函数会在返回对象上打 `pit_safe=False` 标记，下游 assert 可据此拦截。

三臂（与 B0 并列评测，见 §3 T1）：
    A1 kalman_beta_path   —— 主推臂
    A2 rolling_beta_path  —— 廉价对照臂（区分"时变收益" vs "KF 技巧收益"）
    A3 break_test         —— 诊断臂（回答"固定暴露假设是否被数据拒绝"）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "project_simplex", "constrained_ls",
    "kalman_beta_path", "rolling_beta_path", "rts_smooth",
    "break_test", "abnormal_return", "residual_style_r2",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# 约束工具：公募不可做空 ⇒ β ≥ 0；风格权重归一 ⇒ Σβ = 1
# --------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    """欧氏投影到概率单纯形 {β ≥ 0, Σβ = 1}（Duchi et al. 2008）。

    KF 每步更新后调用，保证暴露路径始终落在"可实现的隐形仓位"集合内。
    投影前后的残差差异由 `kalman_beta_path` 以 `proj_shift` 诊断量披露。
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.arange(1, n + 1)
    cond = u - (css - 1.0) / idx > 0
    if not cond.any():          # 数值极端情形：退化为等权
        return np.full(n, 1.0 / n)
    rho = idx[cond][-1]
    theta = (css[cond][-1] - 1.0) / float(rho)
    return np.maximum(v - theta, 0.0)


def constrained_ls(y: np.ndarray, X: np.ndarray, n_iter: int = 800,
                   lr: float | None = None) -> np.ndarray:
    """单纯形约束下的最小二乘（投影梯度）。

    用途 1：KF 的 burn-in 初值 β_0。
    用途 2：**不变量自检**——σ_η → 0 时 KF 的终端 β 应收敛到全样本约束 LS 解
            （见 tests/test_v6_dyn_rbsa.py::test_kf_converges_to_static_when_q_zero）。
            这是计划书 §5-R1 登记的双实现对账。
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    k = X.shape[1]
    b = np.full(k, 1.0 / k)
    G = X.T @ X / max(len(y), 1)
    c = X.T @ y / max(len(y), 1)
    if lr is None:
        ev = np.linalg.eigvalsh(G).max() if k else 1.0
        lr = 1.0 / max(ev, _EPS)
    for _ in range(n_iter):
        b = project_simplex(b - lr * (G @ b - c))
    return b


# --------------------------------------------------------------------------
# A1 —— Kalman Filter 时变暴露
# --------------------------------------------------------------------------
def kalman_beta_path(fund_ret: pd.Series,
                     idx_ret: pd.DataFrame,
                     q: float = 0.03,
                     burn_in: int = 60,
                     p0: float = 1e-2,
                     simplex: bool = True,
                     obs_var: float | None = None) -> pd.DataFrame:
    """随机游走状态空间的**滤波**暴露路径 β_{t|t}。

    Parameters
    ----------
    q : **无量纲信噪比**，状态噪声按 Q = q²·σ_ε² / Var(f) 标定。

        【T1-A1 修订，2026-09-07，见 docs/V6_T1_动态基准报告.md §1】
        初版用绝对尺度 `sigma_eta` 参数化，有两个缺陷（合成数据实测）：
          (1) σ_η 的合适取值随基金残差波动率与因子波动率变化，**跨基金不可比**，
              同一档在低波基金上过慢、在高波基金上过噪；
          (2) 观测噪声 R 的 EWMA 自适应在风格突变处被大新息抬高 → 卡尔曼增益被
              压制 → **漂移越大跟得越慢**（负反馈），与 A1 的设计意图相反。
              实测：真实 β 已切至 idx2 且噪声极小时，终端仍给出 idx2=0.275。
        现改为信噪比参数化 + R 固定为 burn-in 估计。q 有直接可读的物理含义
        （断点后收敛到 0.7 所需交易日数，合成数据实测）：
              q=0.01 → ~120 日 ； q=0.03 → ~39 日 ； q=0.1 → ~13 日
        且该映射在残差噪声 0.001/0.004 两档下几乎不变（尺度无关性达成）。

        预登记三档由 {1e-6,1e-5,1e-4} 修订为 **{0.01, 0.03, 0.1}**（§3 T1 稳健门
        "至少两档同向"不变）。本修订发生在 T1 产出任何裁决数字**之前**，
        属参数化口径修正而非结果导向调参，按 V5 预登记纪律记入修订说明。
    burn_in : 用前 burn_in 个观测做约束 LS 得 β_0 与观测噪声估计；
        该段的 β 受初值影响（**不得**用于下游统计，调用方按需截断）。
    obs_var : 观测噪声方差 σ_ε²。None 则由 burn-in 残差估计后**固定**
        （不再 EWMA 自适应，理由见上 (2)）。

    Returns
    -------
    DataFrame，index = fund_ret.index，columns = idx_ret.columns，
    另附 `.attrs`：pit_safe=True / q / obs_var / mean_proj_shift / n_obs。
    """
    names = list(idx_ret.columns)
    k = len(names)
    df = pd.DataFrame({"y": fund_ret}).join(idx_ret, how="inner")
    df = df.dropna(subset=["y"])
    df[names] = df[names].fillna(0.0)
    n = len(df)
    if n == 0 or k == 0:
        out = pd.DataFrame(index=fund_ret.index, columns=names, dtype=float)
        out.attrs.update(pit_safe=True, q=float(q), n_obs=0,
                         obs_var=np.nan, mean_proj_shift=np.nan)
        return out

    y = df["y"].to_numpy(float)
    X = df[names].to_numpy(float)

    nb = int(min(max(burn_in, k + 5), n))
    beta = constrained_ls(y[:nb], X[:nb]) if nb >= k + 2 else np.full(k, 1.0 / k)
    resid0 = y[:nb] - X[:nb] @ beta
    R = float(obs_var) if obs_var is not None else float(max(np.var(resid0), 1e-10))

    # 信噪比标定：Q = q²·R / Var(f)。除以因子方差使 q 无量纲 ⇒ 跨基金可比。
    var_f = float(np.mean(np.var(X, axis=0)))
    var_f = var_f if var_f > _EPS else 1.0
    P = np.eye(k) * p0
    Q = np.eye(k) * (float(q) ** 2 * R / var_f)

    B = np.empty((n, k))
    shifts = np.empty(n)
    for t in range(n):
        f = X[t]
        # --- predict ---
        P = P + Q
        # --- update（标量观测） ---
        Pf = P @ f
        S = float(f @ Pf) + R
        if S > _EPS:
            K = Pf / S
            v = y[t] - float(f @ beta)
            beta = beta + K * v
            P = P - np.outer(K, Pf)
            P = 0.5 * (P + P.T)
        # --- 约束投影 ---
        if simplex:
            b_new = project_simplex(beta)
            shifts[t] = float(np.abs(b_new - beta).sum())
            beta = b_new
        else:
            shifts[t] = 0.0
        B[t] = beta

    out = pd.DataFrame(B, index=df.index, columns=names)
    out = out.reindex(fund_ret.index)
    out.attrs.update(pit_safe=True, q=float(q), obs_var=float(R), n_obs=int(n),
                     burn_in=int(nb), mean_proj_shift=float(np.mean(shifts)))
    return out


def rts_smooth(*_args, **_kwargs):
    """【禁止入产】RTS 平滑器占位。

    平滑值 β_{t|T} 使用了 t 之后的信息，进入特征/标签即构成前视。
    计划书 §3 T1 规定其只可用于诊断作图；本函数刻意不实现，
    以免被误用——需要作图时请在诊断脚本内局部实现并显式标注 pit_safe=False。
    """
    raise NotImplementedError(
        "RTS smoother 仅供诊断；禁止用于特征/标签（V6 §3 T1 PIT 硬约束）")


# --------------------------------------------------------------------------
# A2 —— 滚动 Ridge 暴露路径（廉价对照臂）
# --------------------------------------------------------------------------
def rolling_beta_path(fund_ret: pd.Series,
                      idx_ret: pd.DataFrame,
                      window: int = 60,
                      step: int = 21,
                      ridge_alpha: float = 1.0) -> pd.DataFrame:
    """每 `step` 个交易日用最近 `window` 日重估一次权重，**只用于其后区间**。

    与 B0 的唯一区别：B0 把决策日的一组权重反套整条历史；本函数把每段权重
    只应用到它自己那一段之后 → 得到分段常数的暴露路径。
    若 A1 显著优于 B0 而 A2 与 B0 无异，说明增益来自 KF 的具体设定而非"时变"
    本身（§3 T1 稳健门据此判 KF 特有拟合）。
    """
    from sklearn.linear_model import Ridge

    names = list(idx_ret.columns)
    df = pd.DataFrame({"y": fund_ret}).join(idx_ret, how="inner").dropna(subset=["y"])
    df[names] = df[names].fillna(0.0)
    n = len(df)
    B = pd.DataFrame(np.nan, index=df.index, columns=names)
    if n < window + 1:
        B.attrs.update(pit_safe=True, window=window, step=step)
        return B.reindex(fund_ret.index)

    y = df["y"].to_numpy(float)
    X = df[names].to_numpy(float)
    last = None
    for t0 in range(window, n, step):
        seg_y, seg_X = y[t0 - window:t0], X[t0 - window:t0]
        if np.nanstd(seg_y) > 1e-8:
            m = Ridge(alpha=ridge_alpha, positive=True, fit_intercept=True)
            m.fit(seg_X, seg_y)
            w = np.clip(m.coef_, 0, None)
            s = w.sum()
            if s > 1e-6:
                last = w / s
        if last is not None:
            B.iloc[t0:t0 + step] = last
    B = B.ffill()
    B.attrs.update(pit_safe=True, window=int(window), step=int(step))
    return B.reindex(fund_ret.index)


# --------------------------------------------------------------------------
# 派生量
# --------------------------------------------------------------------------
def abnormal_return(fund_ret: pd.Series, beta_path: pd.DataFrame,
                    idx_ret: pd.DataFrame) -> pd.Series:
    """AR_t = r_t − β_{t|t}' f_t  （真正的历史 abnormal return）。

    与现行 `active = ret − bench`（bench 由单一 ŵ_q 生成）对位替换。
    """
    names = list(beta_path.columns)
    b = beta_path.reindex(fund_ret.index)
    f = idx_ret.reindex(fund_ret.index)[names].fillna(0.0)
    bench = (b[names] * f).sum(axis=1, skipna=False)
    return (fund_ret - bench).dropna()


def residual_style_r2(ar: pd.Series, idx_ret: pd.DataFrame,
                      window: int = 252) -> float:
    """H1-1 检验量：AR 序列对同期风格指数的滚动回归 R²（越低 ⇒ 风格残留越少）。

    返回各窗口 R² 的均值。B0 的 active return 因风格漂移未被吸收，预期该值偏高。
    """
    names = list(idx_ret.columns)
    df = pd.DataFrame({"a": ar}).join(idx_ret, how="inner").dropna()
    if len(df) < window + 10:
        return np.nan
    y_all = df["a"].to_numpy(float)
    X_all = df[names].to_numpy(float)
    r2s = []
    for t0 in range(0, len(df) - window + 1, window // 2):
        y, X = y_all[t0:t0 + window], X_all[t0:t0 + window]
        Xc = np.column_stack([np.ones(len(y)), X])
        try:
            coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - Xc @ coef
        sst = float(((y - y.mean()) ** 2).sum())
        if sst > _EPS:
            r2s.append(1.0 - float((resid ** 2).sum()) / sst)
    return float(np.mean(r2s)) if r2s else np.nan


# --------------------------------------------------------------------------
# A3 —— 结构突变诊断
# --------------------------------------------------------------------------
def break_test(fund_ret: pd.Series, idx_ret: pd.DataFrame,
               min_seg: int = 126, n_boot: int = 0,
               seed: int = 0) -> dict:
    """单突变点 sup-Wald（Quandt likelihood ratio）检验"固定暴露"假设。

    H0: β 在样本内恒定；H1: 存在一个断点 τ 使前后 β 不同。
    统计量：sup_τ F(τ)，F 为分段回归 vs 全样本回归的 Chow F。
    `n_boot > 0` 时用同方差自助给出经验 p 值（sup 型统计量的渐近分布非标准，
    自助更稳；n_boot=0 则只返回统计量与断点位置，不给 p）。
    """
    names = list(idx_ret.columns)
    df = pd.DataFrame({"y": fund_ret}).join(idx_ret, how="inner").dropna()
    n, k = len(df), len(names) + 1
    out = {"n": n, "sup_F": np.nan, "break_idx": None, "break_date": None,
           "p_boot": np.nan}
    if n < 2 * min_seg + k + 5:
        return out

    y = df["y"].to_numpy(float)
    X = np.column_stack([np.ones(n), df[names].to_numpy(float)])

    def _ssr(yy, XX):
        coef, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        r = yy - XX @ coef
        return float(r @ r)

    def _supF(yy):
        s_all = _ssr(yy, X)
        best, bi = -np.inf, None
        for t in range(min_seg, n - min_seg):
            s = _ssr(yy[:t], X[:t]) + _ssr(yy[t:], X[t:])
            if s <= _EPS:
                continue
            F = ((s_all - s) / k) / (s / (n - 2 * k))
            if F > best:
                best, bi = F, t
        return best, bi

    supF, bi = _supF(y)
    out["sup_F"] = float(supF) if np.isfinite(supF) else np.nan
    if bi is not None:
        out["break_idx"] = int(bi)
        out["break_date"] = df.index[bi]

    if n_boot and np.isfinite(supF):
        rng = np.random.default_rng(seed)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fit, res = X @ coef, y - X @ coef
        cnt = 0
        for _ in range(int(n_boot)):
            yb = fit + rng.permutation(res)
            fb, _bi = _supF(yb)
            if np.isfinite(fb) and fb >= supF:
                cnt += 1
        out["p_boot"] = (cnt + 1) / (int(n_boot) + 1)
    return out
