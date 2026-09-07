# -*- coding: utf-8 -*-
"""
V6 / T2 —— 目标函数重构：raw / style / skill 三套 target + 排序化

预登记依据：docs/优化计划_V6_底层表示重构_2026-09.md §3 T2

现行口径的问题（`_build_ml_panel.py`）：
    标签 = 复权净值直接算 R^fund_{t→t+h}，再做截面 z / rank。
    若未来半年 AI 大涨，一只 AI 暴露高的**平庸基金**也会成为正样本 →
    模型学到的可能是"哪种风格会涨"，而不是"同风险暴露下谁更优秀"。
    这不等于错（style timing 本身可以是 alpha 来源），但必须**拆开**。

本模块给出三套并行 target：
    Y^raw   = R^fund_{t→t+h}
    Y^style = β_{i,t}' R^index_{t→t+h}        （可预测的赛道分量）
    Y^skill = Y^raw − Y^style                 （同赛道内的选基能力）

【β 冻结口径（关键，防前视）】
Y^style / Y^skill 用的 β 是**决策日 t 的滤波值**（dyn_rbsa 输出），
并在整个持有期 [t, t+h] 内**冻结**。理由：决策日只能知道决策日的 β。
持有期内 β 漂移造成的偏差是**已知近似**，由 `style_drift_diagnostic` 量化披露
（§3 T2 要求）。禁止使用持有期内实现的 β 路径均值——那是前视。

【标签成熟纪律】
训练样本必须满足 date ≤ q − hM（承 `_model_zoo.py` R3.5 修复，
原统一 6M 对 fwd12 构成 6 个月前视）。本模块由 `mature_mask` 统一提供，
并对**每个 horizon** 各自计算，不再有硬编码。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "forward_return", "index_forward_return",
    "build_targets", "cross_sectional_transforms",
    "mature_mask", "style_drift_diagnostic",
    "assert_no_lookahead", "blend_scores",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# 前瞻收益
# --------------------------------------------------------------------------
def forward_return(adj: pd.Series, d, months: int) -> float:
    """复权净值序列 adj 在 [d, d+months] 的持有期收益。

    与 `_build_ml_panel.fwd_ret` 同口径（asof 取值），保留以便逐位对账。
    """
    if adj is None or len(adj) == 0:
        return np.nan
    q = pd.Timestamp(d)
    t1 = q + pd.DateOffset(months=int(months))
    if t1 > adj.index[-1] or q < adj.index[0]:
        return np.nan
    v0, v1 = adj.asof(q), adj.asof(t1)
    if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
        return np.nan
    return float(v1 / v0 - 1.0)


def index_forward_return(idx_close: pd.DataFrame, d, months: int) -> pd.Series:
    """各风格指数在 [d, d+months] 的持有期收益（与 forward_return 同口径）。"""
    q = pd.Timestamp(d)
    t1 = q + pd.DateOffset(months=int(months))
    out = {}
    for c in idx_close.columns:
        s = idx_close[c].dropna()
        if len(s) == 0 or t1 > s.index[-1] or q < s.index[0]:
            out[c] = np.nan
            continue
        v0, v1 = s.asof(q), s.asof(t1)
        out[c] = float(v1 / v0 - 1.0) if (pd.notna(v0) and pd.notna(v1)
                                          and v0 > 0) else np.nan
    return pd.Series(out)


# --------------------------------------------------------------------------
# 三套 target
# --------------------------------------------------------------------------
def build_targets(dates, codes, adj_map: dict, beta_at: dict,
                  idx_close: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    """构建 (date, code) × {raw, style, skill} × horizons 的标签面板。

    Parameters
    ----------
    beta_at : {(date, code): pd.Series(β, index=风格名)}
        **决策日 t 的滤波暴露**（来自 `dyn_rbsa.kalman_beta_path`，pit_safe=True）。
        缺失则该行 style/skill 为 NaN（不做任何回填——回填等于编造暴露）。
    idx_close : 风格指数收盘价（列名须与 β 的 index 对齐）。

    Notes
    -----
    style 用 Σ_j β_j · R^index_j。因 β 已归一到单纯形（Σβ=1），
    style 可读作"一个与该基金同等风格配置的被动组合"的持有期收益。
    """
    fwd_idx_cache = {}
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        for h in horizons:
            key = (d, h)
            if key not in fwd_idx_cache:
                fwd_idx_cache[key] = index_forward_return(idx_close, d, h)
        for c in codes:
            adj = adj_map.get(c)
            rec = {"date": d, "code": c}
            b = beta_at.get((d, c))
            for h in horizons:
                raw = forward_return(adj, d, h) if adj is not None else np.nan
                rec[f"raw{h}"] = raw
                if b is None or not isinstance(b, pd.Series) or b.isna().all():
                    rec[f"style{h}"] = np.nan
                    rec[f"skill{h}"] = np.nan
                    continue
                fi = fwd_idx_cache[(d, h)].reindex(b.index)
                ok = fi.notna() & b.notna()
                if not ok.any() or float(b[ok].sum()) <= _EPS:
                    rec[f"style{h}"] = np.nan
                    rec[f"skill{h}"] = np.nan
                    continue
                # 覆盖不足则不出数（与 F_value 的估值盲区同精神）
                cov = float(b[ok].sum()) / float(max(b.sum(), _EPS))
                if cov < 0.5:
                    rec[f"style{h}"] = np.nan
                    rec[f"skill{h}"] = np.nan
                    continue
                st = float((b[ok] * fi[ok]).sum() / b[ok].sum())
                rec[f"style{h}"] = st
                rec[f"skill{h}"] = np.nan if pd.isna(raw) else float(raw - st)
            rows.append(rec)
    return pd.DataFrame(rows)


def cross_sectional_transforms(df: pd.DataFrame, cols, by="date") -> pd.DataFrame:
    """逐截面 z / rank / 二分类标签。

    - `{c}_z`   : 截面标准化（回归目标）
    - `{c}_rk`  : 截面百分位（learning-to-rank 目标）
    - `{c}_pos` : 1[c > 0]（分类目标，仅对 skill 有直接含义 ——
                  "该基金是否跑赢了自己的动态基准"）

    全部**逐截面**计算，不使用任何跨期统计量（承 V5 §0.2-5）。
    """
    out = df.copy()
    for c in cols:
        g = out.groupby(by)[c]
        out[f"{c}_z"] = g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        out[f"{c}_rk"] = g.rank(pct=True)
        out[f"{c}_pos"] = np.where(out[c].notna(), (out[c] > 0).astype(float),
                                   np.nan)
    return out


# --------------------------------------------------------------------------
# 成熟期与前视自检
# --------------------------------------------------------------------------
def mature_mask(dates: pd.Series, as_of, horizon_months: int) -> pd.Series:
    """训练样本资格：date ≤ as_of − hM（标签在 as_of 时已完整成熟）。

    对每个 horizon 各自计算 —— 修复 `_model_zoo.py` 曾对 fwd12 硬编码 6M 的错误。
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=int(horizon_months))
    return (d <= cutoff).to_numpy()


def assert_no_lookahead(panel: pd.DataFrame, horizon_months: int,
                        date_col="date", label_col=None) -> dict:
    """标签窗前视断言（承 C8 护栏，扩展到新标签族）。

    检查：面板中**最后一个有效标签**的决策日 + h 个月不得超过面板最后决策日。
    若超过，说明标签用到了数据尾部之外的信息（或标签根本无法成熟）。
    返回诊断字典；违规抛 AssertionError。
    """
    d = pd.to_datetime(panel[date_col])
    last_decision = d.max()
    out = {"last_decision": last_decision, "horizon_months": horizon_months}
    if label_col is not None:
        valid = panel[panel[label_col].notna()]
        if len(valid) == 0:
            out["last_labeled"] = None
            return out
        last_lab = pd.to_datetime(valid[date_col]).max()
        out["last_labeled"] = last_lab
        matures = last_lab + pd.DateOffset(months=int(horizon_months))
        out["matures_at"] = matures
        assert matures <= last_decision + pd.Timedelta(days=1), (
            f"标签前视：{label_col} 最后有效决策日 {last_lab.date()} "
            f"+{horizon_months}M = {matures.date()} > 面板末日 {last_decision.date()}")
    return out


def style_drift_diagnostic(beta_at: dict, beta_paths: dict,
                           horizon_months: int, sample=None) -> pd.DataFrame:
    """量化"持有期内冻结 β"这一已知近似的偏差（§3 T2 要求披露）。

    对每个 (date, code)，比较：
      - 冻结 β（决策日滤波值，**实际使用**）
      - 持有期内实现的 β 路径均值（**仅诊断，含未来信息，禁止入模**）
    输出 L1 距离，用于回答"冻结假设在多大程度上失真"。
    """
    recs = []
    items = list(beta_at.items())
    if sample is not None and len(items) > sample:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.choice(len(items), sample, replace=False)]
    for (d, c), b0 in items:
        path = beta_paths.get(c)
        if path is None or b0 is None or not isinstance(b0, pd.Series):
            continue
        seg = path.loc[(path.index > pd.Timestamp(d)) &
                       (path.index <= pd.Timestamp(d) + pd.DateOffset(
                           months=int(horizon_months)))]
        if len(seg) == 0:
            continue
        b1 = seg.mean()
        common = b0.index.intersection(b1.index)
        if len(common) == 0:
            continue
        recs.append({"date": pd.Timestamp(d), "code": c,
                     "l1_drift": float(np.abs(b0[common] - b1[common]).sum()),
                     "n_days": int(len(seg))})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# style / skill 组合
# --------------------------------------------------------------------------
def blend_scores(score_style: pd.Series, score_skill: pd.Series,
                 lam) -> pd.Series:
    """Score = λ·Score_style + (1−λ)·Score_skill。

    λ 的三种预登记设定（§3 T2，**必须全部报告，禁止事后择优**）：
      (i)  常数 0.5
      (ii) 由 water/regime 状态映射（调用方传入逐行 Series）
      (iii) 由两腿近 12 月 OOS IC 动态加权（调用方传入逐行 Series）
    """
    a = pd.to_numeric(score_style, errors="coerce")
    b = pd.to_numeric(score_skill, errors="coerce")
    if np.isscalar(lam):
        w = pd.Series(float(lam), index=a.index)
    else:
        w = pd.to_numeric(pd.Series(lam), errors="coerce").reindex(a.index)
    w = w.clip(0.0, 1.0)
    return w * a + (1.0 - w) * b
