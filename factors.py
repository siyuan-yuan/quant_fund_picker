# -*- coding: utf-8 -*-
"""
三大收益因子:
  F_value    —— 估值分位 + 趋势确认滤网 + 缩量加分信号
  F_alpha    —— 动态基准滚动IR胜率 + 下行捕获率
  F_momentum —— 滞后截面动量(4M-1M, 7M-1M)相对排名
"""
import numpy as np
import pandas as pd

from config import (ALPHA_LOOKBACK_DAYS, IR_WINDOW, IR_STEP, IR_THRESHOLD,
                    MOM_4M_START, MOM_4M_END, MOM_7M_START, MOM_7M_END,
                    AUM_SHRINK_TRIGGER, AUM_SHRINK_RANGE, BONUS_POINTS,
                    TENURE_MIN_DAYS)


# ---------------- F_value ----------------
def macd_dif(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 35:
        return np.nan
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    return float((ema12 - ema26).iloc[-1])


def ma_reversal_ok(adj_nav: pd.Series) -> bool:
    """
    V3.2 右侧反转触发器 (补MACD之迟滞, 抓924式拐点):
      净值站上20日均线 且 高于5个交易日前
    """
    s = adj_nav.dropna()
    if len(s) < 25:
        return False
    ma20 = s.rolling(20).mean().iloc[-1]
    return bool(s.iloc[-1] >= ma20 and s.iloc[-1] > s.iloc[-6])


def valuation_base_score(percentile: float, trend_confirmed: bool) -> float:
    """
    模型原规则:
      ≤10%: 动量未破位→100; 极度下降通道(价值陷阱)→上限60
      10%-30%: 99 → 70 线性递减
      30%-70%: 69 → 30 线性递减
      ≥70%: 0
    """
    if np.isnan(percentile):
        return np.nan
    p = percentile * 100
    if p <= 10:
        if trend_confirmed:
            return 100.0
        # 价值陷阱: 基准99封顶锁60
        raw = 99 - (99 - 99) * 0  # 基线给99分潜力
        return min(99.0, 60.0)
    if p <= 30:
        return 99 - (p - 10) / 20 * (99 - 70)
    if p <= 70:
        return 69 - (p - 30) / 40 * (69 - 30)
    return 0.0


def drawdown_bonus_signal(dossier: dict, tenure_days: int) -> tuple:
    """
    抗流动性死亡螺旋缩水信号 (近似实现, 3维共振):
      1. AUM较高点缩水>50% 且 当前规模 5~20亿
      2. 基金经理(任职期覆盖窗口, 视为未变更)
      3. 流动性滤网代理: 股票占净比 最新vs一年前 变动<10pp
         (未被迫抛售优质资产)
    返回 (bonus, detail)
    """
    sh = dossier.get("scale_hist", [])
    if len(sh) < 3:
        return 0, {"pass": False, "reason": "规模史不足"}
    scales = pd.Series([x["scale"] for x in sh if x["scale"]], dtype=float).dropna()
    if scales.empty:
        return 0, {"pass": False, "reason": "规模史为空"}
    cur, peak = scales.iloc[-1], scales.max()
    shrank = peak > 0 and (1 - cur / peak) > AUM_SHRINK_TRIGGER
    in_range = AUM_SHRINK_RANGE[0] <= cur <= AUM_SHRINK_RANGE[1]
    mgr_ok = tenure_days >= TENURE_MIN_DAYS

    liq_ok = False
    alloc = dossier.get("asset_alloc", {}).get("股票占净比")
    if alloc and alloc["values"]:
        v = pd.Series(alloc["values"], dtype=float).dropna()
        if len(v) >= 2 and abs(v.iloc[-1] - v.iloc[0]) < 10:
            liq_ok = True

    ok = bool(shrank and in_range and mgr_ok and liq_ok)
    return (BONUS_POINTS if ok else 0), {
        "pass": ok, "cur_scale": round(float(cur), 2),
        "peak": round(float(peak), 2), "shrank": bool(shrank),
        "in_range": bool(in_range), "mgr_ok": bool(mgr_ok), "liq_ok": liq_ok}


# ---------------- F_alpha ----------------
def rolling_ir_winrate(active: pd.Series, window=IR_WINDOW, step=IR_STEP,
                       th=IR_THRESHOLD) -> float:
    """6个月滚动窗口 IR>0.3 的胜率"""
    a = active.dropna().iloc[-ALPHA_LOOKBACK_DAYS:]
    if len(a) < window:
        return np.nan
    irs = []
    for st in range(0, len(a) - window + 1, step):
        seg = a.iloc[st:st + window]
        sd = seg.std()
        if sd > 1e-10:
            irs.append(seg.mean() / sd * np.sqrt(252))
    if not irs:
        return np.nan
    return float(np.mean([ir > th for ir in irs]))


def ir_score(winrate: float) -> float:
    if np.isnan(winrate):
        return np.nan
    if winrate > 0.70:
        return 100.0
    if winrate < 0.50:
        return 0.0
    return 60.0 + (winrate - 0.50) / 0.20 * 39.0


def downside_capture(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    """动态基准下跌月份: 基金月均跌幅 / 基准月均跌幅"""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    m = df.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    down = m[m["b"] < 0]
    if len(down) < 4 or down["b"].mean() >= 0:
        return np.nan
    return float(down["f"].mean() / down["b"].mean())


def dc_score(ratio: float) -> float:
    if np.isnan(ratio):
        return np.nan
    if ratio < 0.80:
        return 100.0
    if ratio > 1.10:
        return 0.0
    return float(100.0 * (1.10 - ratio) / 0.30)


# ---------------- F_momentum ----------------
def lagged_momentum_returns(adj_nav: pd.Series) -> tuple:
    """(4M-1M收益, 7M-1M收益) —— 剔除最近1个月规避短期反转"""
    s = adj_nav.dropna()
    if len(s) < MOM_7M_START + 2:
        return np.nan, np.nan
    r4 = float(s.iloc[-MOM_4M_END] / s.iloc[-MOM_4M_START] - 1)
    r7 = float(s.iloc[-MOM_7M_END] / s.iloc[-MOM_7M_START] - 1)
    return r4, r7


def momentum_score(rank4: float, rank7: float) -> float:
    """rank 为同类中分位 (1.0=最高收益)。
    双前30%→100; 双前50%→60; 4M跌出前50%→0(动量破位); 其余→30"""
    if np.isnan(rank4) or np.isnan(rank7):
        return np.nan
    if rank4 >= 0.70 and rank7 >= 0.70:
        return 100.0
    if rank4 >= 0.50 and rank7 >= 0.50:
        return 60.0
    if rank4 < 0.50:
        return 0.0
    return 30.0


# ================= V3.7 平滑因子候选 (变体裁决后生效) =================
def ma20_distance(adj_nav: pd.Series) -> float:
    """净值相对MA20的距离(小数) — 平滑趋势门的连续输入"""
    s = adj_nav.dropna()
    if len(s) < 20:
        return np.nan
    return float(s.iloc[-1] / s.rolling(20).mean().iloc[-1] - 1)


def trend_gate_smooth(dist: float) -> float:
    """MA20距离 → 趋势确认度 t∈[0,1]: -2%以下→0, +4%以上→1, 线性过渡"""
    if np.isnan(dist):
        return 0.0
    return float(np.clip((dist + 0.02) / 0.06, 0, 1))


def value_score_smooth(pct: float, trend_t: float) -> float:
    """F_value 平滑版: ≤10%分位 60+40t 连续趋势门; 70-90%分位 30→0 渐进(杀30分悬崖)"""
    if np.isnan(pct):
        return np.nan
    p = pct * 100
    if p <= 10:
        return 60.0 + 40.0 * trend_t
    if p <= 30:
        return 99 - (p - 10) / 20 * (99 - 70)
    if p <= 70:
        return 69 - (p - 30) / 40 * (69 - 30)
    if p <= 90:
        return 30 * (90 - p) / 20
    return 0.0


def ir_score_smooth(w: float) -> float:
    """IR胜率分 平滑版 (杀0.50处60分悬崖): 0.45→20 / 0.55→55 / 0.65→88"""
    if np.isnan(w):
        return np.nan
    return float(100.0 / (1.0 + np.exp(-(w - 0.53) / 0.06)))


def dc_score_smooth(ratio: float) -> float:
    """下行捕获分 平滑版: 0.85→89 / 0.95→50 / 1.05→19"""
    if np.isnan(ratio):
        return np.nan
    return float(100.0 / (1.0 + np.exp((ratio - 0.95) / 0.07)))


def momentum_score_smooth_m1(rank4: float, rank7: float) -> float:
    """动量 平滑版M1(线性混合): m=0.6r4+0.4r7; m≥0.70→100, ≤0.38→0"""
    if np.isnan(rank4) or np.isnan(rank7):
        return np.nan
    m = 0.6 * rank4 + 0.4 * rank7
    return float(100 * np.clip((m - 0.38) / 0.32, 0, 1))


def momentum_score_smooth_m2(rank4: float, rank7: float) -> float:
    """动量 平滑版M2(短门×长势): 保留4M破位语义 but 平滑"""
    if np.isnan(rank4) or np.isnan(rank7):
        return np.nan
    g = lambda x: float(np.clip((x - 0.35) / 0.35, 0, 1))
    return 100 * g(rank4) * (0.55 + 0.45 * g(rank7))
