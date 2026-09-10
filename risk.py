# -*- coding: utf-8 -*-
"""
风险乘数与动态过滤阀 —— 乘法惩罚, 一票否决
"""
import numpy as np
import pandas as pd

from config import (TENURE_MIN_DAYS, TENURE_SMOOTH_MAX,
                    MDD_SMOOTH_FREE, MDD_SMOOTH_K,
                    MDD_VETO_LOW_WATER, MDD_LOWWATER_DAMP,
                    AUM_STYLE_LIMIT_YI, CONCENTRATION_LIMIT)


def tenure_smooth_penalty(days) -> float:
    """V3.6: 任期归因平滑折价 — 0年任→TENURE_SMOOTH_MAX, 线性衰减至3年任→0"""
    if days is None or days >= TENURE_MIN_DAYS:
        return 0.0
    return round(TENURE_SMOOTH_MAX * (1 - days / TENURE_MIN_DAYS), 3)


def mdd_smooth_penalty(r_mdd: float, water=None) -> float:
    """V3.6: R_MDD 平滑惩罚 — ≤1.0免罚, p=clip(0.5*(R-1),0,1); 底部区(水≤35%)减半"""
    if r_mdd is None or r_mdd != r_mdd or r_mdd <= MDD_SMOOTH_FREE:
        return 0.0
    p = min(MDD_SMOOTH_K * (r_mdd - MDD_SMOOTH_FREE), 1.0)
    if water is not None and water == water and water <= MDD_VETO_LOW_WATER:
        p *= MDD_LOWWATER_DAMP
    return round(p, 3)


def max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.dropna()).cumprod()
    dd = nav / nav.cummax() - 1
    return float(dd.min())


def build_penalties(*, tenure_days, is_passive, fund_ret, bench_ret,
                    weights, smallcap_exp, cur_scale, ir_winrate,
                    skip_tenure=False, skip_aum=False, water=None):
    """
    返回 (penalty_list, detail)
      penalty_list: [(名称, PenaltyRate)] —— 乘法链
      回测模式: skip_tenure/skip_aum=True (经理任期/AUM为当前快照, 非PiT)
      V3.5: water 用于条款5 低水位否决降级
    """
    pens, d = [], {}

    # 1) V3.6: 任期归因平滑折价 — 仅主动型基金介入 (被动指数豁免: 看的是赛道不是经理)
    if not is_passive and not skip_tenure:
        d["tenure_days"] = tenure_days
        tp = tenure_smooth_penalty(tenure_days)
        if tp >= 0.005:
            pens.append((f"经理任期{int(tenure_days)//365}年(归因平滑{tp:.0%})", tp))

    # 2) V3.6: 相对动态基准超额回撤 — 平滑惩罚 (≤1.0免罚, 斜率0.5, 底部区减半)
    f3 = fund_ret.dropna()
    b3 = bench_ret.reindex(f3.index).fillna(0)
    mdd_f, mdd_b = max_drawdown(f3.iloc[-756:]), max_drawdown(b3.iloc[-756:])
    # V3.7.1: 基准近乎零回撤(债基/货币类, 3年权益基准MDD必然>0.5%) → 标尺失效, R_MDD=None不适用
    #         (旧逻辑给 np.inf → 罚满100%且污染JSON, 007195债基整批陪葬事故)
    if abs(mdd_b) < 0.005:
        r_mdd = None
    else:
        r_mdd = abs(mdd_f) / abs(mdd_b)
    d["mdd_fund"], d["mdd_bench"] = round(mdd_f, 4), round(mdd_b, 4)
    d["R_MDD"] = round(r_mdd, 3) if r_mdd is not None else None
    mp = mdd_smooth_penalty(r_mdd, water)
    if mp >= 0.005:
        tag = "(底部减半)" if (water is not None and water == water and water <= MDD_VETO_LOW_WATER) else ""
        pens.append((f"回撤比值{r_mdd:.2f}(平滑{mp:.0%}{tag})", mp))

    # 3) 动态规模反噬: 中小盘/高换手风格 且 规模>150亿
    if not skip_aum and smallcap_exp > 0.5 and cur_scale and cur_scale > AUM_STYLE_LIMIT_YI:
        pens.append((f"小盘风格暴露{smallcap_exp:.0%}且规模{cur_scale:.0f}亿>150亿", 0.3))
    d["smallcap_exp"] = round(smallcap_exp, 3)
    d["cur_scale"] = cur_scale

    # 4) 无效极端集中度: RBSA前三大板块>70% 且 IR胜率<50%
    #    (被动指数基金豁免: 集中跟踪标的指数是其法定义务, "Alpha不足"为范畴错误)
    top3 = float(weights.nlargest(3).sum()) if len(weights) >= 3 else 0.0
    d["top3_conc"] = round(top3, 3)
    if (not is_passive) and top3 > CONCENTRATION_LIMIT and (not np.isnan(ir_winrate)) and ir_winrate < 0.5:
        pens.append((f"无效集中度{top3:.0%}且胜率{ir_winrate:.0%}<50%", 0.4))

    return pens, d


def apply_penalties(base_score: float, pens: list) -> float:
    s = base_score
    for _, p in pens:
        s *= (1 - p)
    return max(0.0, min(100.0, s))
