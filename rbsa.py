# -*- coding: utf-8 -*-
"""
Ridge-RBSA 底层穿透: 带L2正则与非负约束的风格归因
Sharpe(1992) Returns-Based Style Analysis 的机构级变体:
  - 60日滚动窗口 × 3个错位窗口取平均 = 平滑隐形仓位
  - L2 岭惩罚压制共线性(上证50 vs 沪深300)
  - 非负约束 = 公募基金不可做空
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import (RBSA_INDICES, RBSA_WINDOW, RBSA_SMOOTH_OFFSETS,
                    RIDGE_ALPHA, PE_PERCENTILE_WINDOW, SMALLCAP_STYLE)
import provider


_FULL_MAT = {}
_PEP_CACHE = {}


def index_return_matrix(dates: pd.DatetimeIndex, indices=None) -> pd.DataFrame:
    """因子收益矩阵, 按基金净值日期对齐; indices 可指定面板(回测用风格6)
    矩阵全局缓存: 月度频率回测时避免重复构建"""
    idxs = indices or RBSA_INDICES
    key = tuple(x[1] for x in idxs)
    if key not in _FULL_MAT:
        cols = {}
        for src, code, name, _, _ in idxs:
            close = provider.get_close_by_src(src, code)
            cols[name] = close.pct_change()
        _FULL_MAT[key] = pd.DataFrame(cols).sort_index()
    full = _FULL_MAT[key]
    return full.reindex(full.index.union(dates)).ffill().reindex(dates)


def rbsa_weights(fund_ret: pd.Series, idx_ret: pd.DataFrame) -> pd.Series:
    """
    错位多窗口岭回归取平均 → 平滑隐形仓位权重 (和为1)
    fund_ret / idx_ret 索引一致(交易日)
    """
    names = list(idx_ret.columns)
    W = []
    for off in RBSA_SMOOTH_OFFSETS:
        end = len(fund_ret) - off
        if end < RBSA_WINDOW + 10:
            continue
        y = fund_ret.iloc[end - RBSA_WINDOW:end].values
        X = idx_ret.iloc[end - RBSA_WINDOW:end][names].fillna(0).values
        if np.nanstd(y) < 1e-8:
            continue
        m = Ridge(alpha=RIDGE_ALPHA, positive=True, fit_intercept=True)
        m.fit(X, y)
        w = np.clip(m.coef_, 0, None)
        s = w.sum()
        if s > 1e-6:
            W.append(w / s)
    if not W:
        return pd.Series(np.nan, index=names)
    return pd.Series(np.mean(W, axis=0), index=names)


def index_pe_percentile(window: int = PE_PERCENTILE_WINDOW, indices=None,
                        as_of=None) -> dict:
    """各指数滚动PE分位 → {指数名: 0~1}
    as_of=None → 当前; 回测时截断到回测日(纯PiT)
    结果按 (面板, window, as_of) 记忆化 —— 月度回测必要优化"""
    idxs = indices or RBSA_INDICES
    cache_key = (tuple(x[1] for x in idxs), window, str(as_of))
    if as_of is None:
        cached = None
    else:
        cached = _PEP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    out = {}
    for src, _, name, pe_key, _ in idxs:
        try:
            pe = provider.get_pe_by_key(pe_key).dropna()
            if len(pe) == 0:           # "none"=境外无PE源 → 估值盲区权重
                out[name] = np.nan
                continue
            if as_of is not None:
                pe = pe[pe.index <= pd.Timestamp(as_of)]
            if len(pe) < 250:          # 历史过短(指数发布晚)则弃用该指数
                out[name] = np.nan
                continue
            hist = pe.iloc[-window:] if len(pe) > window else pe
            cur = pe.iloc[-1]
            out[name] = float((hist < cur).mean())
        except Exception:
            out[name] = np.nan
    if as_of is not None:
        _PEP_CACHE[cache_key] = dict(out)
    return out


def fund_valuation_percentile(weights: pd.Series, indices=None, as_of=None) -> tuple:
    """RBSA隐形仓位加权 → (基金底层资产估值分位 0~1, PE覆盖率 0~1)
    覆盖率 = 有PE分位的指数权重之和 / 总权重; 境外'none'腿拉低覆盖率"""
    pe_pct = index_pe_percentile(indices=indices, as_of=as_of)
    w = weights.fillna(0)
    if w.sum() <= 0:
        return np.nan, 0.0
    covered = {n: pe_pct.get(n, np.nan) for n in w.index}
    valid = [n for n in w.index
             if covered.get(n) is not None and not pd.isna(covered.get(n))]
    cov_w = float(sum(w[n] for n in valid))
    if cov_w <= 0 or not valid:
        return np.nan, 0.0
    pct = float(sum(w[n] * covered[n] for n in valid) / cov_w)
    return pct, cov_w / float(w.sum())


def smallcap_exposure(weights: pd.Series) -> float:
    """中小盘/高换手风格暴露 (规模反噬阀)"""
    tag = {x[2]: x[4] for x in RBSA_INDICES}
    return float(sum(weights.get(n, 0) for n in weights.index
                     if tag.get(n) in SMALLCAP_STYLE))


def market_water_level(indices=None, as_of=None) -> float:
    """大盘水位计: 6风格指数等权PE分位 (0~1, 越低越便宜)
    as_of=None → 当前; 回测截断(纯PiT)"""
    panel = indices or [x for x in RBSA_INDICES if x[0] == "sina"]
    pct = index_pe_percentile(indices=panel, as_of=as_of)
    vals = [v for v in pct.values() if v == v]
    return float(np.mean(vals)) if vals else np.nan
