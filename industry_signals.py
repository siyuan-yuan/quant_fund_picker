# -*- coding: utf-8 -*-
"""
行业前视性信号：盈利动量 (F_earn_momentum)

实验依据（exp_comprehensive.py, exp_implementation.py）:
  54,583个观察点, 476只基金, 2011-2026 (覆盖牛/熊/震荡)
  Walk-forward验证（训练集归一化，无前瞻偏差）:
  - earn_momentum: test_IC=+0.102, 稳定度=71%
  - 加入后最优方案 WF_IC=+0.351 (当前系统+0.238), 正IC率=83%

信号定义:
  earnings = close / pe       # 隐含盈利
  earnings_growth_3m = earnings.pct_change(63)  # 63交易日≈3个月增速
  earnings_momentum = earnings_growth_3m.diff(63)  # 增速的加速度
  F_earn_momentum = RBSA加权各行业earnings_momentum
"""
import numpy as np
import pandas as pd
from config import RBSA_INDICES
import provider

# 进程内缓存
_EARN_MOM_CACHE = {}


def _compute_industry_earn_momentum(pe_series: pd.Series, close_series: pd.Series) -> pd.Series:
    """
    从单个行业的PE和价格序列计算盈利动量
    严格Point-in-Time: 只使用过去数据
    """
    # 隐含盈利 = 价格 / PE
    earnings = close_series / pe_series
    
    # 3个月盈利增速（63交易日）
    earnings_growth_3m = earnings.pct_change(63)
    
    # 盈利动量 = 增速的变化（加速度）
    earnings_momentum = earnings_growth_3m.diff(63)
    
    return earnings_momentum


def get_industry_earn_momentum(as_of: str = None) -> dict:
    """
    获取各行业在指定日期的盈利动量值
    
    返回: {行业名称: earnings_momentum值}
    
    缓存策略: 按最新PE数据日期缓存（日内不重复计算）
    """
    cache_key = as_of or "latest"
    if cache_key in _EARN_MOM_CACHE:
        return _EARN_MOM_CACHE[cache_key]
    
    result = {}
    
    for src, _, name, pe_key, _ in RBSA_INDICES:
        if pe_key.startswith("none") or not pe_key:
            continue  # 无PE数据的跳过
        
        try:
            pe = provider.get_pe_by_key(pe_key)
            if pe is None or len(pe) < 200:
                continue
            
            # 获取价格数据
            if src == "csindex":
                close = provider.get_index_close_csindex(pe_key.split(":")[-1] if ":" in pe_key else pe_key)
            elif src in ("us_sina", "hk_sina"):
                # 境外指数用价格
                code = pe_key.replace("csi:", "")
                close = provider.get_index_close(pe_key)
                if close is None or len(close) < 200:
                    continue
            else:
                close = provider.get_index_close(pe_key)
            
            if close is None or len(close) < 200:
                continue
            
            # 截取到as_of日期
            if as_of is not None:
                as_of_ts = pd.Timestamp(as_of)
                pe = pe[pe.index <= as_of_ts]
                close = close[close.index <= as_of_ts]
            
            if len(pe) < 200 or len(close) < 200:
                continue
            
            # 对齐日期
            common_idx = pe.index.intersection(close.index)
            if len(common_idx) < 200:
                continue
            
            pe_aligned = pe.reindex(common_idx)
            close_aligned = close.reindex(common_idx)
            
            # 计算盈利动量
            earn_mom = _compute_industry_earn_momentum(pe_aligned, close_aligned)
            
            # 取最新值
            latest = earn_mom.dropna()
            if len(latest) > 0:
                result[name] = float(latest.iloc[-1])
        
        except Exception:
            continue
    
    _EARN_MOM_CACHE[cache_key] = result
    return result


def fund_earn_momentum(rbsa_weights: pd.Series, as_of: str = None) -> float:
    """
    RBSA加权行业盈利动量 → 基金的F_earn_momentum
    
    参数:
        rbsa_weights: 基金的RBSA权重 {行业名: 权重}
        as_of: 截止日期
    
    返回: float (加权盈利动量), 或 np.nan (数据不足)
    """
    ind_mom = get_industry_earn_momentum(as_of)
    
    if not ind_mom:
        return np.nan
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for sector, weight in rbsa_weights.items():
        if sector in ind_mom and not np.isnan(ind_mom[sector]):
            weighted_sum += weight * ind_mom[sector]
            total_weight += weight
    
    if total_weight < 0.1:
        return np.nan
    
    return weighted_sum / total_weight


def earn_momentum_score(raw_value: float, cross_section: pd.Series) -> float:
    """
    将原始盈利动量值转换为0-100分
    
    使用截面百分位排名（与F_momentum类似）
    
    参数:
        raw_value: 基金的原始盈利动量值
        cross_section: 全市场基金的盈利动量值序列
    
    返回: 0-100分
    """
    if np.isnan(raw_value) or cross_section is None or len(cross_section.dropna()) < 30:
        return np.nan
    
    valid = cross_section.dropna()
    percentile = (valid < raw_value).mean()
    return percentile * 100


def clear_cache():
    """清除缓存"""
    global _EARN_MOM_CACHE
    _EARN_MOM_CACHE = {}
