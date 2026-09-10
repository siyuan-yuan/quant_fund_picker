#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正式策略实验脚本：资金分配 / 再平衡 / 截面排名 / 止损 / 大盘趋势过滤
======================================================================

设计目标
--------
用仓库内已经落盘的 Point-in-Time 打分缓存与日频净值缓存，系统性对比策略构建层变体，
避免只凭直觉判断 Cash Drag、截面排名、再平衡、移动止损等改动的优劣。

默认复刻对象
------------
默认读取 `output/bt_scores_cache/*_2e4ec0f5.csv` 一类月频缓存，并在 2006-09-30 →
2026-03-31 区间复刻 `output/bt_summary_b70s45n10k10w_monthly.md` 的核心基线。
如果不指定 `--score-suffix`，脚本会在给定日期区间内自动选择“文件数×行数”最大的
缓存后缀，通常就是历史 217 池的 `_2e4ec0f5`。

重要说明
--------
1. 本脚本只依赖 Python 标准库，不依赖 pandas/numpy，便于在干净环境里复现。
2. 因子分数 S 会从缓存原料重建，公式对齐 `backtest_local.py::score_from_raw`。
3. 默认 `--decision-calendar exact` 用“评分日期必须刚好是交易日才交易”的旧本地回测口径，
   以便对齐已有报告；更真实的口径可用 `--decision-calendar previous`，即月末休市则落到
   之前最后一个交易日。
4. 再平衡为研究用近似实现：按目标权重卖出超配、买入低配；未建模 T+N 到账、短期赎回费、
   最小申赎金额、真实申赎确认价等细节。
5. 幸存者偏差不由本脚本解决；本脚本仍受输入基金池限制。

输出
----
output/strategy_experiment_summary.csv
output/strategy_experiment_report.md
output/strategy_experiment_equity_<variant>.csv
output/strategy_experiment_trades_<variant>.csv

用法
----
python strategy_experiment.py
python strategy_experiment.py --start 2006-09-30 --end 2026-03-31 --score-suffix _2e4ec0f5
python strategy_experiment.py --decision-calendar previous
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple


# ============================ 基础参数 ============================

DEFAULT_START = "2006-09-30"
DEFAULT_END = "2026-03-31"
DEFAULT_CAPITAL = 100000.0
DEFAULT_BUY = 70.0
DEFAULT_SELL = 45.0
DEFAULT_SLOTS = 10
DEFAULT_COST_IN = 0.0015
DEFAULT_COST_OUT = 0.005
MIN_ALLOC_PCT = 0.02

CACHE_DIR = "cache"
SCORE_CACHE_DIR = "output/bt_scores_cache"
OUT_DIR = "output"
BENCH_FILE = "cache/idx_sh000300.csv"


# ============================ 工具函数 ============================


def parse_date(s: str) -> date:
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def fmt_date(d: date) -> str:
    return d.isoformat()


def ffloat(x, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return default
        return float(s)
    except Exception:
        return default


def safe_exp(x: float) -> float:
    if x > 700:
        return math.exp(700)
    if x < -700:
        return math.exp(-700)
    return math.exp(x)


def pct(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "—"
    return f"{x:+.2%}"


def num(x: Optional[float], nd: int = 2) -> str:
    if x is None or math.isnan(x):
        return "—"
    return f"{x:.{nd}f}"


def ensure_out() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def read_csv_dicts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ============================ S 分重建：对齐 backtest_local.py ============================


def valuation_base_score(percentile: float, trend_confirmed: bool) -> float:
    if math.isnan(percentile):
        return math.nan
    p = percentile * 100.0
    if p <= 10:
        return 100.0 if trend_confirmed else 60.0
    if p <= 30:
        return 99.0 - (p - 10.0) / 20.0 * (99.0 - 70.0)
    if p <= 70:
        return 69.0 - (p - 30.0) / 40.0 * (69.0 - 30.0)
    return 0.0


def ir_score_smooth(w: float) -> float:
    if math.isnan(w):
        return math.nan
    return 100.0 / (1.0 + safe_exp(-(w - 0.53) / 0.06))


def dc_score_smooth(ratio: float) -> float:
    if math.isnan(ratio):
        return math.nan
    return 100.0 / (1.0 + safe_exp((ratio - 0.95) / 0.07))


def momentum_score_smooth_m1(rank4: float, rank7: float) -> float:
    if math.isnan(rank4) or math.isnan(rank7):
        return math.nan
    m = 0.6 * rank4 + 0.4 * rank7
    return 100.0 * min(max((m - 0.38) / 0.32, 0.0), 1.0)


def mdd_factor(rm: float, water: float) -> float:
    if math.isnan(rm) or rm <= 1.2:
        return 1.0
    p = min(0.5 * (rm - 1.2), 1.0)
    if not math.isnan(water) and water <= 0.35:
        p *= 0.5
    return 1.0 - p


def rank_pct(values: List[float]) -> List[float]:
    """复刻 pandas Series.rank(pct=True) 默认逻辑：升序、并列取平均、NaN 保持 NaN。"""
    out = [math.nan] * len(values)
    valid = sorted((v, i) for i, v in enumerate(values) if not math.isnan(v))
    n = len(valid)
    if n == 0:
        return out
    j = 0
    rank_start = 1
    while j < n:
        k = j + 1
        while k < n and valid[k][0] == valid[j][0]:
            k += 1
        avg_rank = (rank_start + (rank_start + (k - j) - 1)) / 2.0
        for _, idx in valid[j:k]:
            out[idx] = avg_rank / n
        rank_start += (k - j)
        j = k
    return out


def rebuild_scores(raw_rows: List[dict], as_of: str) -> List[dict]:
    r4 = [ffloat(r.get("r4")) for r in raw_rows]
    r7 = [ffloat(r.get("r7")) for r in raw_rows]
    rank4 = rank_pct(r4)
    rank7 = rank_pct(r7)

    scored: List[dict] = []
    for i, r in enumerate(raw_rows):
        code = str(r.get("code", "")).zfill(6)
        val_cov = ffloat(r.get("val_cov"))
        val_pct = ffloat(r.get("val_pct"))
        trend_ok = str(r.get("trend_ok", "")).strip().lower() == "true"

        if math.isnan(val_cov) or val_cov < 0.5 or math.isnan(val_pct):
            fv = math.nan
        else:
            fv = valuation_base_score(val_pct, trend_ok)

        alpha_parts: List[float] = []
        wr = ffloat(r.get("wr"))
        dc = ffloat(r.get("dc"))
        if not math.isnan(wr):
            alpha_parts.append(ir_score_smooth(wr))
        if not math.isnan(dc):
            alpha_parts.append(dc_score_smooth(dc))
        fa = sum(alpha_parts) / len(alpha_parts) if alpha_parts else math.nan
        fm = momentum_score_smooth_m1(rank4[i], rank7[i])

        wv = ffloat(r.get("wv"), 0.0)
        wa = ffloat(r.get("wa"), 0.0)
        wm = ffloat(r.get("wm"), 0.0)
        water = ffloat(r.get("water"))
        rm = ffloat(r.get("R_MDD"))
        other_pen = ffloat(r.get("other_pen"), 1.0)
        pen = other_pen * mdd_factor(rm, water)

        nume = 0.0
        deno = 0.0
        if not math.isnan(fv):
            nume += wv * min(max(fv, 0.0), 100.0)
            deno += wv
        if not math.isnan(fa):
            nume += wa * fa
            deno += wa
        if not math.isnan(fm):
            nume += wm * fm
            deno += wm
        score = ((nume / deno) if deno else 0.0) * pen
        score = min(max(score, 0.0), 100.0)

        rr = dict(r)
        rr.update({
            "date": as_of,
            "code": code,
            "S": score,
            "rank4": rank4[i],
            "rank7": rank7[i],
            "F_value": fv,
            "F_alpha": fa,
            "F_mom": fm,
        })
        scored.append(rr)
    return scored


# ============================ 数据读取 ============================


def score_cache_suffix_from_path(path: str) -> Tuple[str, str]:
    """返回 (YYYY-MM-DD, suffix)。suffix 包含前导下划线；无后缀则为空字符串。"""
    name = os.path.basename(path)
    stem = name[:-4] if name.endswith(".csv") else name
    if "_" in stem:
        d, suffix = stem.split("_", 1)
        return d, "_" + suffix
    return stem, ""


def choose_score_suffix(start: date, end: date) -> str:
    candidates: Dict[str, dict] = {}
    for fp in glob.glob(os.path.join(SCORE_CACHE_DIR, "*.csv")):
        d_str, suffix = score_cache_suffix_from_path(fp)
        try:
            d = parse_date(d_str)
        except Exception:
            continue
        if not (start <= d <= end):
            continue
        try:
            n = max(0, sum(1 for _ in open(fp, "r", encoding="utf-8-sig")) - 1)
        except Exception:
            n = 0
        rec = candidates.setdefault(suffix, {"files": 0, "rows": 0})
        rec["files"] += 1
        rec["rows"] += n
    if not candidates:
        raise FileNotFoundError(f"{SCORE_CACHE_DIR} 中找不到 {start}~{end} 的评分缓存")
    # 先最大总行数，再最大文件数。
    return max(candidates.items(), key=lambda kv: (kv[1]["rows"], kv[1]["files"]))[0]


def load_panel(start: date, end: date, suffix: str) -> Tuple[Dict[date, Dict[str, dict]], List[date], int]:
    panel: Dict[date, Dict[str, dict]] = {}
    rows_total = 0
    for fp in sorted(glob.glob(os.path.join(SCORE_CACHE_DIR, f"*{suffix}.csv"))):
        d_str, sf = score_cache_suffix_from_path(fp)
        if sf != suffix:
            continue
        try:
            d = parse_date(d_str)
        except Exception:
            continue
        if not (start <= d <= end):
            continue
        raw = read_csv_dicts(fp)
        if not raw:
            continue
        scored = rebuild_scores(raw, d_str)
        panel[d] = {r["code"]: r for r in scored}
        rows_total += len(scored)
    dates = sorted(panel)
    if not dates:
        raise RuntimeError(f"未能加载评分面板：suffix={suffix!r}, {start}~{end}")
    return panel, dates, rows_total


def load_bench(start: date, end: date) -> List[Tuple[date, float]]:
    rows = read_csv_dicts(BENCH_FILE)
    out: List[Tuple[date, float]] = []
    for r in rows:
        if "date" not in r:
            continue
        d = parse_date(r["date"])
        if start <= d <= end:
            close = ffloat(r.get("close", r.get("nav", "")))
            if not math.isnan(close):
                out.append((d, close))
    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError(f"未能加载基准：{BENCH_FILE}, {start}~{end}")
    return out


def load_navs(codes: Iterable[str]) -> Dict[str, List[Tuple[date, float]]]:
    navs: Dict[str, List[Tuple[date, float]]] = {}
    for code in sorted(set(codes)):
        fp = os.path.join(CACHE_DIR, f"nav_{code}.csv")
        if not os.path.exists(fp):
            continue
        arr: List[Tuple[date, float]] = []
        try:
            for r in read_csv_dicts(fp):
                d = parse_date(r.get("date", ""))
                nav = ffloat(r.get("nav"))
                if not math.isnan(nav):
                    arr.append((d, nav))
        except Exception:
            continue
        arr.sort(key=lambda x: x[0])
        # 对齐原本地回测：净值太短的剔除。
        if len(arr) > 200:
            navs[code] = arr
    return navs


def px(navs: Dict[str, List[Tuple[date, float]]], code: str, day: date) -> float:
    arr = navs.get(code)
    if not arr:
        return math.nan
    i = bisect.bisect_right(arr, (day, float("inf"))) - 1
    if i < 0:
        return math.nan
    return arr[i][1]


def first_nav_date(navs: Dict[str, List[Tuple[date, float]]], code: str) -> Optional[date]:
    arr = navs.get(code)
    return arr[0][0] if arr else None


def bench_ma(bench: List[Tuple[date, float]], window: int = 200) -> Dict[date, float]:
    out: Dict[date, float] = {}
    q: List[float] = []
    s = 0.0
    for d, close in bench:
        q.append(close)
        s += close
        if len(q) > window:
            s -= q.pop(0)
        out[d] = s / len(q) if q else math.nan
    return out


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def bench_realized_vol_and_threshold(
    bench: List[Tuple[date, float]],
    window: int = 20,
    q: float = 0.80,
    min_history: int = 60,
) -> Tuple[Dict[date, float], Dict[date, float]]:
    """20日实现波动率与历史分位阈值。

    阈值只使用当前日之前已经观测到的 realized vol，避免未来函数。
    """
    vols: Dict[date, float] = {}
    thresholds: Dict[date, float] = {}
    rets: List[float] = []
    hist_vols: List[float] = []
    for i, (d, close) in enumerate(bench):
        if i > 0 and bench[i - 1][1] > 0:
            rets.append(close / bench[i - 1][1] - 1.0)
        vol = math.nan
        if len(rets) >= window:
            xs = rets[-window:]
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
            vol = math.sqrt(var) * math.sqrt(252.0)
        vols[d] = vol
        thresholds[d] = quantile(sorted(hist_vols), q) if len(hist_vols) >= min_history else math.nan
        if not math.isnan(vol):
            hist_vols.append(vol)
    return vols, thresholds


# ============================ 实验配置与模拟 ============================


@dataclass
class Variant:
    name: str
    description: str
    allocation: str = "fixed_slot"       # fixed_slot / cash_empty
    selection: str = "threshold"         # threshold / rank
    rank_pct: float = 0.10
    rank_abs_floor: Optional[float] = None
    buy_th: float = DEFAULT_BUY
    sell_th: float = DEFAULT_SELL
    rebalance: str = "none"              # none / monthly / quarterly
    trail_stop: Optional[float] = None
    macro_ma200: str = "none"            # none / buy_filter / half_slots
    cash_yield: float = 0.0              # 闲置现金年化收益；如 0.02 表示货基/短债年化 2%
    water_cap_gate: Optional[float] = None   # 大盘水位阈值；触发后限制权益槽位
    water_cap_slots: Optional[int] = None    # 触发水位阈值后的最大持仓槽位数
    crisis_filter: bool = False              # 趋势破位 + 20日实现波动率高分位：危机期禁止新开权益仓
    crisis_ma_window: int = 200              # 趋势均线窗口；200≈MA200，150≈更敏感快线
    crisis_vol_window: int = 20
    crisis_vol_quantile: float = 0.80
    crisis_cap_slots: Optional[int] = None   # 危机期主动降仓到最多N槽；None=只禁买不主动卖
    cppi: bool = False                       # 组合级动态风险预算/回撤熔断
    cppi_dd1: float = -0.10
    cppi_slots1: int = 6
    cppi_dd2: float = -0.15
    cppi_slots2: int = 3
    cppi_dd3: float = -0.20
    cppi_slots3: int = 0
    cppi_hwm_mode: str = "reset"          # reset / global / decay / partial
    cppi_hwm_halflife_years: Optional[float] = None
    cppi_partial_reset_floor: Optional[float] = None  # partial模式：解锁时 risk_hwm = equity / floor，如0.85代表复活即处于-15%预算位
    cppi_rerisk: bool = False             # 熔断后按波动率分位逐步恢复，而非一次性放开


@dataclass
class Position:
    code: str
    units: float
    cost_basis: float              # 当前剩余份额对应的现金成本，含申购成本。
    entry_px: float
    entry_date: date
    signal_date: date
    entry_S: float
    peak: float


@dataclass
class SimResult:
    variant: Variant
    equity_rows: List[dict]
    trade_rows: List[dict]
    summary: dict


@dataclass
class SimulatorConfig:
    capital: float = DEFAULT_CAPITAL
    slots: int = DEFAULT_SLOTS
    cost_in: float = DEFAULT_COST_IN
    cost_out: float = DEFAULT_COST_OUT
    min_alloc_pct: float = MIN_ALLOC_PCT
    decision_calendar: str = "exact"     # exact / previous


def default_variants(buy: float, sell: float) -> List[Variant]:
    return [
        Variant(
            name="baseline_fixed_slot",
            description="旧口径：S>买入阈值 / S<卖出阈值；单笔买入 min(cash, equity/slots)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
        ),
        Variant(
            name="cash_to_empty_slots",
            description="修复候选：新开仓时把现金均分给剩余空槽位",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell,
        ),
        Variant(
            name="monthly_rebalance_cash_empty",
            description="现金均分空槽 + 每个决策日等权再平衡",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, rebalance="monthly",
        ),
        Variant(
            name="quarterly_rebalance_cash_empty",
            description="现金均分空槽 + 季末等权再平衡",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, rebalance="quarterly",
        ),
        Variant(
            name="rank_top10_cash_empty",
            description="截面前10%强制候选 + 现金均分空槽；不设绝对分数底线",
            allocation="cash_empty", selection="rank", rank_pct=0.10, rank_abs_floor=None, buy_th=buy, sell_th=sell,
        ),
        Variant(
            name="rank_top10_S45_cash_empty",
            description="截面前10% 且 S>45 + 现金均分空槽",
            allocation="cash_empty", selection="rank", rank_pct=0.10, rank_abs_floor=45.0, buy_th=buy, sell_th=sell,
        ),
        Variant(
            name="rank_top20_S45_cash_empty",
            description="截面前20% 且 S>45 + 现金均分空槽",
            allocation="cash_empty", selection="rank", rank_pct=0.20, rank_abs_floor=45.0, buy_th=buy, sell_th=sell,
        ),
        Variant(
            name="trail15_cash_empty",
            description="现金均分空槽 + 15%移动止损",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, trail_stop=0.15,
        ),
        Variant(
            name="trail20_cash_empty",
            description="现金均分空槽 + 20%移动止损",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, trail_stop=0.20,
        ),
        Variant(
            name="hs300_ma200_buy_filter",
            description="现金均分空槽 + 沪深300低于200日均线时禁止新开仓",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, macro_ma200="buy_filter",
        ),
        Variant(
            name="hs300_ma200_half_slots",
            description="现金均分空槽 + 沪深300低于200日均线时最大槽位降为5并卖弱留强",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell, macro_ma200="half_slots",
        ),
        Variant(
            name="baseline_cash_yield_2p0",
            description="原始固定槽位 + 闲置现金年化2.0%",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell, cash_yield=0.020,
        ),
        Variant(
            name="fixedslot_qreb_trail20_cash_yield_2p0",
            description="保留空槽：固定槽位买入 + 季度等权再平衡 + 20%移动止损 + 闲置现金年化2.0%",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.020,
        ),
        Variant(
            name="fixedslot_qreb_trail20_cash_yield_2p5",
            description="保留空槽：固定槽位买入 + 季度等权再平衡 + 20%移动止损 + 闲置现金年化2.5%",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
        ),
        Variant(
            name="cash_empty_qreb_trail20_cash_yield_2p0",
            description="对照：现金均分空槽 + 季度等权再平衡 + 20%移动止损 + 闲置现金年化2.0%",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.020,
        ),
        Variant(
            name="cash_empty_qreb_trail20_cash_yield_2p5",
            description="对照：现金均分空槽 + 季度等权再平衡 + 20%移动止损 + 闲置现金年化2.5%",
            allocation="cash_empty", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water70_cap5",
            description="敏感性测试：最强候选 + 大盘水位>70%时强制最多5槽（约50%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.70, water_cap_slots=5,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water70_cap3",
            description="敏感性测试：最强候选 + 大盘水位>70%时强制最多3槽（约30%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.70, water_cap_slots=3,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water85_cap5",
            description="最强候选 + 大盘水位>85%时强制最多5槽（约50%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.85, water_cap_slots=5,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water90_cap5",
            description="最强候选 + 大盘水位>90%时强制最多5槽（约50%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.90, water_cap_slots=5,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water85_cap3",
            description="最强候选 + 大盘水位>85%时强制最多3槽（约30%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.85, water_cap_slots=3,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_water90_cap3",
            description="最强候选 + 大盘水位>90%时强制最多3槽（约30%权益上限）",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            water_cap_gate=0.90, water_cap_slots=3,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_crisis_ma200_vol80",
            description="最强候选 + 危机过滤：沪深300<MA200 且 20日实现波动率>历史80%分位时禁止新开仓",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_vol_window=20, crisis_vol_quantile=0.80,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_cppi_10_15_20",
            description="最强候选 + 组合级CPPI：自身回撤-10%限6槽、-15%限3槽、-20%清仓后等待右侧信号重启",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            cppi=True,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_crisis_cppi",
            description="最强候选 + 危机过滤(MA200&Vol80) + 组合级CPPI回撤预算",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_cppi_15_20_25",
            description="敏感性测试：较宽CPPI：自身回撤-15%限6槽、-20%限3槽、-25%清仓重启",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="fixedslot_qreb_trail20_y25_crisis_cppi_15_20_25",
            description="敏感性测试：危机过滤 + 较宽CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma200_v80_cap6_cppi15",
            description="主动危机防守：MA200破位+Vol80时强制最多6槽 + 较宽CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80, crisis_cap_slots=6,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma200_v80_cap3_cppi15",
            description="主动危机防守：MA200破位+Vol80时强制最多3槽 + 较宽CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80, crisis_cap_slots=3,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma150_v80_cap6_cppi15",
            description="均线敏感度：MA150破位+Vol80时强制最多6槽 + 较宽CPPI",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=150, crisis_vol_window=20, crisis_vol_quantile=0.80, crisis_cap_slots=6,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma150_v80_cap3_cppi15",
            description="均线敏感度：MA150破位+Vol80时强制最多3槽 + 较宽CPPI",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=150, crisis_vol_window=20, crisis_vol_quantile=0.80, crisis_cap_slots=3,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma200_v75_cap6_cppi15",
            description="波动率水位：MA200破位+Vol75时强制最多6槽 + 较宽CPPI",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.75, crisis_cap_slots=6,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="active_crisis_ma200_v85_cap6_cppi15",
            description="波动率水位：MA200破位+Vol85时强制最多6槽 + 较宽CPPI",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.85, crisis_cap_slots=6,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="global_hwm_crisis_cppi_15_20_25",
            description="压力测试：全局HWM不重置 + 危机禁买 + CPPI(-15/-20/-25)，检验死锁风险",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="global",
        ),
        Variant(
            name="global_hwm_rerisk_crisis_cppi_15_20_25",
            description="压力测试：全局HWM不重置 + 危机禁买 + CPPI + 波动率分位逐步复活",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="global", cppi_rerisk=True,
        ),
        Variant(
            name="decay_hwm_2y_crisis_cppi_15_20_25",
            description="Time-decay HWM：2年半衰期 + 危机禁买 + CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="decay", cppi_hwm_halflife_years=2.0,
        ),
        Variant(
            name="decay_hwm_3y_crisis_cppi_15_20_25",
            description="Time-decay HWM：3年半衰期 + 危机禁买 + CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="decay", cppi_hwm_halflife_years=3.0,
        ),
        Variant(
            name="decay_hwm_4y_crisis_cppi_15_20_25",
            description="Time-decay HWM：4年半衰期 + 危机禁买 + CPPI(-15/-20/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="decay", cppi_hwm_halflife_years=4.0,
        ),
        Variant(
            name="decay_hwm_3y_rerisk_crisis_cppi_15_20_25",
            description="复活机制：3年衰减HWM + 危机禁买 + CPPI + 波动率分位逐步恢复仓位",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="decay", cppi_hwm_halflife_years=3.0, cppi_rerisk=True,
        ),
        Variant(
            name="decay_hwm_4y_rerisk_crisis_cppi_15_20_25",
            description="复活机制：4年衰减HWM + 危机禁买 + CPPI + 波动率分位逐步恢复仓位",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="decay", cppi_hwm_halflife_years=4.0, cppi_rerisk=True,
        ),
        Variant(
            name="grid_reset_crisis_cppi_14_19_24",
            description="阈值网格：重置HWM + 危机禁买 + CPPI(-14/-19/-24)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.14, cppi_slots1=6, cppi_dd2=-0.19, cppi_slots2=3, cppi_dd3=-0.24, cppi_slots3=0,
        ),
        Variant(
            name="grid_reset_crisis_cppi_15_20_24",
            description="阈值网格：重置HWM + 危机禁买 + CPPI(-15/-20/-24)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.24, cppi_slots3=0,
        ),
        Variant(
            name="grid_reset_crisis_cppi_16_21_25",
            description="阈值网格：重置HWM + 危机禁买 + CPPI(-16/-21/-25)",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.16, cppi_slots1=6, cppi_dd2=-0.21, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
        ),
        Variant(
            name="partial_hwm85_crisis_cppi_15_20_25",
            description="部分重置HWM：熔断复活时 risk_hwm=equity/0.85，保留15%历史压力 + 危机禁买",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="partial", cppi_partial_reset_floor=0.85,
        ),
        Variant(
            name="partial_hwm80_crisis_cppi_15_20_25",
            description="部分重置HWM：熔断复活时 risk_hwm=equity/0.80，保留20%历史压力 + 危机禁买",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="partial", cppi_partial_reset_floor=0.80,
        ),
        Variant(
            name="partial_hwm85_rerisk_crisis_cppi_15_20_25",
            description="部分重置HWM 0.85 + 波动率分位逐步恢复仓位 + 危机禁买",
            allocation="fixed_slot", selection="threshold", buy_th=buy, sell_th=sell,
            rebalance="quarterly", trail_stop=0.20, cash_yield=0.025,
            crisis_filter=True, crisis_ma_window=200, crisis_vol_window=20, crisis_vol_quantile=0.80,
            cppi=True, cppi_dd1=-0.15, cppi_slots1=6, cppi_dd2=-0.20, cppi_slots2=3, cppi_dd3=-0.25, cppi_slots3=0,
            cppi_hwm_mode="partial", cppi_partial_reset_floor=0.85, cppi_rerisk=True,
        ),
    ]


def build_decision_map(
    score_dates: List[date],
    bench_days: List[date],
    mode: str,
) -> Dict[date, List[date]]:
    """返回 trading_day -> [score_date...]。exact 复刻旧口径；previous 更贴近真实月末休市处理。"""
    bset = set(bench_days)
    out: Dict[date, List[date]] = {}
    if mode == "exact":
        for sd in score_dates:
            if sd in bset:
                out.setdefault(sd, []).append(sd)
        return out
    if mode == "previous":
        for sd in score_dates:
            i = bisect.bisect_right(bench_days, sd) - 1
            if i >= 0:
                out.setdefault(bench_days[i], []).append(sd)
        return out
    raise ValueError("decision_calendar must be exact or previous")


def is_rebalance_day(signal_date: date, freq: str) -> bool:
    if freq == "none":
        return False
    if freq == "monthly":
        return True
    if freq == "quarterly":
        return signal_date.month in {3, 6, 9, 12}
    raise ValueError(f"unknown rebalance freq: {freq}")


def position_value(navs: Dict[str, List[Tuple[date, float]]], pos: Position, day: date) -> float:
    p = px(navs, pos.code, day)
    return 0.0 if math.isnan(p) else pos.units * p


def equity_now(navs: Dict[str, List[Tuple[date, float]]], positions: Dict[str, Position], cash: float, day: date) -> float:
    return cash + sum(position_value(navs, p, day) for p in positions.values())


def max_drawdown(equities: List[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        if peak > 0:
            mdd = min(mdd, e / peak - 1.0)
    return mdd


def simulate_variant(
    variant: Variant,
    panel: Dict[date, Dict[str, dict]],
    score_dates: List[date],
    navs: Dict[str, List[Tuple[date, float]]],
    bench: List[Tuple[date, float]],
    cfg: SimulatorConfig,
    start: date,
    end: date,
) -> SimResult:
    bench_days = [d for d, _ in bench]
    bench_close = {d: c for d, c in bench}
    ma200 = bench_ma(bench, 200)
    crisis_ma = bench_ma(bench, variant.crisis_ma_window)
    vol20, vol80 = bench_realized_vol_and_threshold(bench, variant.crisis_vol_window, variant.crisis_vol_quantile)
    _, rerisk_vol60 = bench_realized_vol_and_threshold(bench, variant.crisis_vol_window, 0.60)
    _, rerisk_vol80 = bench_realized_vol_and_threshold(bench, variant.crisis_vol_window, 0.80)
    decision_map = build_decision_map(score_dates, bench_days, cfg.decision_calendar)

    cash = cfg.capital
    positions: Dict[str, Position] = {}
    trade_rows: List[dict] = []
    equity_rows: List[dict] = []
    total_cost = 0.0
    cash_interest_total = 0.0
    blocked_buys_macro = 0
    skipped_full = 0
    skipped_small = 0
    rebal_turnover = 0.0
    water_cap_hits = 0
    water_cap_sells = 0
    crisis_days = 0
    crisis_blocked_buys = 0
    crisis_cap_sells = 0
    cppi_sells = 0
    cppi_hard_stops = 0
    cppi_unlocks = 0
    cppi_recovery_steps = 0
    cppi_locked = False
    cppi_hwm = cfg.capital
    cppi_last_dd = 0.0
    cppi_recovery_cap: Optional[int] = None
    last_scores: Optional[Dict[str, dict]] = None
    prev_day: Optional[date] = None

    def sell(code: str, day: date, signal_date: Optional[date], reason: str, exit_s: Optional[float] = None) -> None:
        nonlocal cash, total_cost
        pos = positions.pop(code)
        price = px(navs, code, day)
        if math.isnan(price):
            # 无价则把仓位放回，避免凭空消失。
            positions[code] = pos
            return
        gross = pos.units * price
        fee = gross * cfg.cost_out
        net = gross - fee
        cash += net
        total_cost += fee
        net_ret = net / pos.cost_basis - 1.0 if pos.cost_basis > 0 else math.nan
        trade_rows.append({
            "variant": variant.name,
            "code": code,
            "entry_date": fmt_date(pos.entry_date),
            "entry_signal_date": fmt_date(pos.signal_date),
            "exit_date": fmt_date(day),
            "exit_signal_date": fmt_date(signal_date) if signal_date else "",
            "entry_px": f"{pos.entry_px:.6f}",
            "exit_px": f"{price:.6f}",
            "entry_S": f"{pos.entry_S:.4f}",
            "exit_S": f"{exit_s:.4f}" if exit_s is not None and not math.isnan(exit_s) else "",
            "exit_reason": reason,
            "hold_days": (day - pos.entry_date).days,
            "cost_basis": f"{pos.cost_basis:.6f}",
            "net_out": f"{net:.6f}",
            "net_ret": f"{net_ret:.8f}",
            "pnl_yuan": f"{net - pos.cost_basis:.6f}",
        })

    def rebalance(day: date, target_slots: int) -> None:
        """研究用等权再平衡：卖超配、补低配。保留原 entry_date；部分卖出不作为完整交易。"""
        nonlocal cash, total_cost, rebal_turnover
        if not positions or target_slots <= 0:
            return
        eq = equity_now(navs, positions, cash, day)
        if eq <= 0:
            return
        target = eq / target_slots

        # 先卖超配，1% 容忍带，降低无意义碎片交易。
        for code in list(positions):
            pos = positions[code]
            price = px(navs, code, day)
            if math.isnan(price) or pos.units <= 0:
                continue
            val = pos.units * price
            if val <= target * 1.01:
                continue
            sell_value = val - target
            sell_units = sell_value / price
            if sell_units <= 0 or sell_units >= pos.units:
                continue
            ratio = sell_units / pos.units
            cost_reduce = pos.cost_basis * ratio
            pos.units -= sell_units
            pos.cost_basis -= cost_reduce
            gross = sell_units * price
            fee = gross * cfg.cost_out
            cash += gross - fee
            total_cost += fee
            rebal_turnover += gross

        # 再补低配，1% 容忍带。
        for code in list(positions):
            if cash <= 0:
                break
            pos = positions[code]
            price = px(navs, code, day)
            if math.isnan(price) or price <= 0:
                continue
            val = pos.units * price
            if val >= target * 0.99:
                continue
            need_value = target - val
            # alloc 是现金流出，包含申购费。
            alloc = min(cash, need_value * (1.0 + cfg.cost_in))
            if alloc <= 0:
                continue
            units = alloc / (price * (1.0 + cfg.cost_in))
            fee = alloc - units * price
            pos.units += units
            pos.cost_basis += alloc
            cash -= alloc
            total_cost += fee
            rebal_turnover += alloc

    def trim_to_slot_cap(day: date, cap_slots: int, scores: Optional[Dict[str, dict]], reason: str) -> int:
        """只降风险、不补仓：卖弱留强到 cap_slots，并把剩余持仓卖回每槽约10%目标。"""
        nonlocal cash, total_cost, rebal_turnover
        sold_full = 0
        cap_slots = max(0, int(cap_slots))
        if cap_slots == 0:
            for code in list(positions):
                exit_s = ffloat(scores[code].get("S")) if scores and code in scores else math.nan
                sell(code, day, None, reason, exit_s)
                sold_full += 1
            return sold_full

        if len(positions) > cap_slots:
            order = sorted(
                positions.keys(),
                key=lambda c: ffloat(scores[c].get("S")) if scores and c in scores else -1.0,
                reverse=True,
            )
            for code in order[cap_slots:]:
                exit_s = ffloat(scores[code].get("S")) if scores and code in scores else math.nan
                sell(code, day, None, reason, exit_s)
                sold_full += 1

        # 剩余仓位若因上涨超过 1/10 目标，只卖不买，保证 cap 是权益上限而非持仓数量装饰。
        eq = equity_now(navs, positions, cash, day)
        if eq > 0:
            target = eq / cfg.slots
            for code in list(positions):
                pos = positions[code]
                price = px(navs, code, day)
                if math.isnan(price) or price <= 0 or pos.units <= 0:
                    continue
                val = pos.units * price
                if val <= target * 1.01:
                    continue
                sell_units = (val - target) / price
                if sell_units <= 0 or sell_units >= pos.units:
                    continue
                ratio = sell_units / pos.units
                pos.units -= sell_units
                pos.cost_basis *= (1.0 - ratio)
                gross = sell_units * price
                fee = gross * cfg.cost_out
                cash += gross - fee
                total_cost += fee
                rebal_turnover += gross
        return sold_full

    def candidates_for(g: Dict[str, dict], v: Variant) -> List[dict]:
        rows = [r for r in g.values() if not math.isnan(ffloat(r.get("S")))]
        rows.sort(key=lambda r: ffloat(r.get("S")), reverse=True)
        if v.selection == "threshold":
            return [r for r in rows if ffloat(r.get("S")) > v.buy_th]
        if v.selection == "rank":
            k = max(1, int(math.ceil(len(rows) * v.rank_pct)))
            top = rows[:k]
            if v.rank_abs_floor is not None:
                top = [r for r in top if ffloat(r.get("S")) > v.rank_abs_floor]
            return top
        raise ValueError(f"unknown selection: {v.selection}")

    def market_water(g: Dict[str, dict]) -> float:
        """评分缓存中 water 是同一决策日的大盘水位；取首个有效值即可。"""
        for r in g.values():
            w = ffloat(r.get("water"))
            if not math.isnan(w):
                return w
        return math.nan

    def rerisk_step(day: date) -> int:
        """波动率越低，熔断后恢复仓位越快。"""
        vol = vol20.get(day, math.nan)
        q60 = rerisk_vol60.get(day, math.nan)
        q80 = rerisk_vol80.get(day, math.nan)
        if math.isnan(vol) or math.isnan(q60) or math.isnan(q80):
            return 1
        if vol < q60:
            return 3
        if vol < q80:
            return 2
        return 1

    for day in bench_days:
        # 0) 闲置现金收益：按相邻交易日之间的自然日差，对当时现金做日度单利计提。
        #    这近似货基/短债每日收益入账；收益入账后成为现金余额的一部分。
        elapsed_days = max(0, (day - prev_day).days) if prev_day is not None else 0
        if prev_day is not None and variant.cash_yield:
            if elapsed_days:
                interest = cash * variant.cash_yield * elapsed_days / 365.25
                cash += interest
                cash_interest_total += interest
        if variant.cppi and variant.cppi_hwm_mode == "decay" and elapsed_days and variant.cppi_hwm_halflife_years:
            decay = math.exp(-math.log(2.0) * elapsed_days / (365.25 * variant.cppi_hwm_halflife_years))
            cppi_hwm *= decay
        prev_day = day

        # 1) 日度移动止损。
        if variant.trail_stop is not None:
            for code in list(positions):
                pos = positions[code]
                price = px(navs, code, day)
                if math.isnan(price):
                    continue
                pos.peak = max(pos.peak, price)
                if price < pos.peak * (1.0 - variant.trail_stop):
                    sell(code, day, None, f"trail_{variant.trail_stop:.0%}")

        # 1a2) 日度危机状态：趋势破位 + 波动率高分位。
        daily_crisis_active = False
        if variant.crisis_filter:
            close = bench_close.get(day, math.nan)
            ma = crisis_ma.get(day, math.nan)
            vol = vol20.get(day, math.nan)
            vth = vol80.get(day, math.nan)
            daily_crisis_active = (
                (not math.isnan(close)) and (not math.isnan(ma)) and close < ma
                and (not math.isnan(vol)) and (not math.isnan(vth)) and vol > vth
            )
            if daily_crisis_active:
                crisis_days += 1
                if variant.crisis_cap_slots is not None and positions:
                    sold = trim_to_slot_cap(day, int(variant.crisis_cap_slots), last_scores, f"crisis_cap{variant.crisis_cap_slots}")
                    crisis_cap_sells += sold

        # 1b) 组合级动态风险预算 / CPPI：根据策略自身净值回撤强制降权益上限。
        daily_cppi_cap: Optional[int] = None
        if variant.cppi:
            eq_for_cppi = equity_now(navs, positions, cash, day)
            if (not cppi_locked) and eq_for_cppi > cppi_hwm:
                cppi_hwm = eq_for_cppi
            cppi_dd = eq_for_cppi / cppi_hwm - 1.0 if cppi_hwm > 0 else 0.0
            cppi_last_dd = cppi_dd
            if cppi_locked:
                daily_cppi_cap = 0
            elif cppi_dd <= variant.cppi_dd3:
                daily_cppi_cap = variant.cppi_slots3
                cppi_locked = True
                cppi_recovery_cap = None
                cppi_hard_stops += 1
            elif cppi_dd <= variant.cppi_dd2:
                daily_cppi_cap = variant.cppi_slots2
            elif cppi_dd <= variant.cppi_dd1:
                daily_cppi_cap = variant.cppi_slots1

            if daily_cppi_cap is not None and positions:
                sold = trim_to_slot_cap(day, daily_cppi_cap, last_scores, f"cppi_cap{daily_cppi_cap}")
                cppi_sells += sold

        # 2) 决策日：可能有多个评分日期映射到同一天；按评分日期顺序执行。
        if day in decision_map:
            for signal_date in sorted(decision_map[day]):
                g = panel[signal_date]
                last_scores = g

                crisis_active = daily_crisis_active

                risk_off = False
                target_slots = cfg.slots
                if variant.macro_ma200 != "none":
                    close = bench_close.get(day, math.nan)
                    ma = ma200.get(day, math.nan)
                    risk_off = (not math.isnan(close)) and (not math.isnan(ma)) and close < ma
                    if risk_off and variant.macro_ma200 == "half_slots":
                        target_slots = max(1, cfg.slots // 2)

                wlevel = market_water(g)
                water_cap_active = (
                    variant.water_cap_gate is not None
                    and variant.water_cap_slots is not None
                    and not math.isnan(wlevel)
                    and wlevel > variant.water_cap_gate
                )
                if water_cap_active:
                    target_slots = min(target_slots, max(1, int(variant.water_cap_slots)))
                    water_cap_hits += 1

                crisis_cap_active = crisis_active and variant.crisis_cap_slots is not None
                if crisis_cap_active:
                    target_slots = min(target_slots, max(0, int(variant.crisis_cap_slots)))

                if variant.cppi and daily_cppi_cap is not None:
                    target_slots = min(target_slots, max(0, int(daily_cppi_cap)))

                if variant.cppi and cppi_locked:
                    # 硬熔断后等待右侧信号 + 非危机。reset模式重置HWM；global/decay模式必须等风险HWM约束自然放松。
                    can_unlock = bool(candidates_for(g, variant)) and not crisis_active
                    if variant.cppi_hwm_mode in {"global", "decay"}:
                        can_unlock = can_unlock and cppi_last_dd > variant.cppi_dd3
                    if can_unlock:
                        cppi_locked = False
                        if variant.cppi_hwm_mode == "reset":
                            cppi_hwm = equity_now(navs, positions, cash, day)
                            cppi_last_dd = 0.0
                        elif variant.cppi_hwm_mode == "partial":
                            eq_unlock = equity_now(navs, positions, cash, day)
                            floor = variant.cppi_partial_reset_floor or 0.85
                            cppi_hwm = max(eq_unlock / max(floor, 1e-9), eq_unlock)
                            cppi_last_dd = eq_unlock / cppi_hwm - 1.0 if cppi_hwm > 0 else 0.0
                            if cppi_last_dd <= variant.cppi_dd2:
                                target_slots = min(target_slots or cfg.slots, variant.cppi_slots2)
                            elif cppi_last_dd <= variant.cppi_dd1:
                                target_slots = min(target_slots or cfg.slots, variant.cppi_slots1)
                        cppi_unlocks += 1
                        if variant.cppi_rerisk:
                            cppi_recovery_cap = min(cfg.slots, rerisk_step(day))
                            target_slots = min(target_slots or cfg.slots, cppi_recovery_cap)
                            cppi_recovery_steps += 1
                        elif target_slots == 0:
                            target_slots = cfg.slots
                    else:
                        target_slots = 0

                if variant.cppi and (not cppi_locked) and variant.cppi_rerisk and cppi_recovery_cap is not None:
                    if candidates_for(g, variant) and not crisis_active:
                        cppi_recovery_cap = min(cfg.slots, cppi_recovery_cap + rerisk_step(day))
                        cppi_recovery_steps += 1
                    if cppi_recovery_cap >= cfg.slots:
                        cppi_recovery_cap = None
                    else:
                        target_slots = min(target_slots, cppi_recovery_cap)

                # 2a) 先卖出：跌出面板 / S 跌破卖出阈值。
                for code in list(positions):
                    if code not in g:
                        sell(code, day, signal_date, "drop_panel")
                        continue
                    score = ffloat(g[code].get("S"))
                    if math.isnan(score) or score < variant.sell_th:
                        sell(code, day, signal_date, f"S<{variant.sell_th:.0f}", score)

                # 2b) 宏观 half_slots：风险关闭时卖弱留强。
                if variant.macro_ma200 == "half_slots" and risk_off and len(positions) > target_slots:
                    order = sorted(
                        positions.keys(),
                        key=lambda c: ffloat(g[c].get("S")) if c in g else -1.0,
                        reverse=True,
                    )
                    for code in order[target_slots:]:
                        exit_s = ffloat(g[code].get("S")) if code in g else math.nan
                        sell(code, day, signal_date, "macro_ma200_half_slots", exit_s)

                # 2b2) 主动危机防守：宏观趋势+波动率共振时直接限制全局权益上限。
                water_rebalanced = False
                if crisis_cap_active:
                    sold = trim_to_slot_cap(day, target_slots, g, f"crisis_cap{target_slots}")
                    crisis_cap_sells += sold
                    water_rebalanced = True

                # 2b3) 估值水位风控阀：高水位时限制持仓槽位，卖弱留强；随后按10槽目标权重再平衡，
                #      使 5槽≈50%权益、3槽≈30%权益，而不是把剩余槽位放大成满仓。
                if water_cap_active:
                    if len(positions) > target_slots:
                        order = sorted(
                            positions.keys(),
                            key=lambda c: ffloat(g[c].get("S")) if c in g else -1.0,
                            reverse=True,
                        )
                        for code in order[target_slots:]:
                            exit_s = ffloat(g[code].get("S")) if code in g else math.nan
                            sell(code, day, signal_date, f"water>{variant.water_cap_gate:.0%}_cap{target_slots}", exit_s)
                            water_cap_sells += 1
                    rebalance(day, cfg.slots)
                    water_rebalanced = True

                # 2c) 再平衡。
                if (not water_rebalanced) and is_rebalance_day(signal_date, variant.rebalance):
                    rebalance(day, cfg.slots)

                # 2d) 买入。
                if variant.macro_ma200 == "buy_filter" and risk_off:
                    blocked_buys_macro += len(candidates_for(g, variant))
                    continue
                if crisis_active:
                    crisis_blocked_buys += len(candidates_for(g, variant))
                    continue

                for row in candidates_for(g, variant):
                    code = str(row.get("code", "")).zfill(6)
                    if code in positions:
                        continue
                    if len(positions) >= target_slots:
                        skipped_full += 1
                        continue
                    if code not in navs:
                        continue
                    first_dt = first_nav_date(navs, code)
                    if first_dt and first_dt > day:
                        continue
                    price = px(navs, code, day)
                    if math.isnan(price) or price <= 0:
                        continue

                    eq = equity_now(navs, positions, cash, day)
                    if eq <= 0 or cash <= 0:
                        continue
                    empty_slots = max(1, target_slots - len(positions))
                    if variant.allocation == "fixed_slot":
                        alloc = min(cash, eq / cfg.slots)
                    elif variant.allocation == "cash_empty":
                        alloc = cash / empty_slots
                    else:
                        raise ValueError(f"unknown allocation: {variant.allocation}")

                    if alloc < eq * cfg.min_alloc_pct:
                        skipped_small += 1
                        continue
                    alloc = min(alloc, cash)
                    units = alloc / (price * (1.0 + cfg.cost_in))
                    fee = alloc - units * price
                    cash -= alloc
                    total_cost += fee
                    score = ffloat(row.get("S"))
                    positions[code] = Position(
                        code=code,
                        units=units,
                        cost_basis=alloc,
                        entry_px=price,
                        entry_date=day,
                        signal_date=signal_date,
                        entry_S=score,
                        peak=price,
                    )

        # 3) 日频盯市。
        eq = equity_now(navs, positions, cash, day)
        invested_value = eq - cash
        equity_rows.append({
            "date": fmt_date(day),
            "equity": f"{eq:.8f}",
            "cash": f"{cash:.8f}",
            "invested_value": f"{invested_value:.8f}",
            "n_pos": len(positions),
            "cash_ratio": f"{(cash / eq) if eq > 0 else math.nan:.8f}",
            "exposure": f"{(invested_value / eq) if eq > 0 else math.nan:.8f}",
            "cash_interest_cum": f"{cash_interest_total:.8f}",
        })

    # 4) 期末研究清算：交易账记录净赎回；权益曲线为了对齐旧口径，默认不回写最后一天。
    for code in list(positions):
        sell(code, end, None, "end_liquidation")

    equities = [ffloat(r["equity"]) for r in equity_rows]
    n_pos = [int(r["n_pos"]) for r in equity_rows]
    cash_ratios = [ffloat(r["cash_ratio"]) for r in equity_rows if not math.isnan(ffloat(r["cash_ratio"]))]
    exposures = [ffloat(r["exposure"]) for r in equity_rows if not math.isnan(ffloat(r["exposure"]))]
    final_equity_curve = equities[-1] if equities else math.nan
    liquidation_equity = cash

    yrs = (end - start).days / 365.25
    total_return = final_equity_curve / cfg.capital - 1.0 if cfg.capital else math.nan
    cagr = (final_equity_curve / cfg.capital) ** (1.0 / yrs) - 1.0 if yrs > 0 and final_equity_curve > 0 else math.nan
    dd = max_drawdown(equities)

    bench_return = bench[-1][1] / bench[0][1] - 1.0 if bench and bench[0][1] else math.nan
    bench_dd = max_drawdown([c for _, c in bench])

    net_rets = [ffloat(r["net_ret"]) for r in trade_rows if not math.isnan(ffloat(r["net_ret"]))]
    hold_days = [int(r["hold_days"]) for r in trade_rows if str(r.get("hold_days", "")).strip() != ""]
    wins = [x for x in net_rets if x > 0]
    losses = [x for x in net_rets if x <= 0]
    avg_win = sum(wins) / len(wins) if wins else math.nan
    avg_loss = sum(losses) / len(losses) if losses else math.nan

    summary = {
        "variant": variant.name,
        "description": variant.description,
        "allocation": variant.allocation,
        "selection": variant.selection,
        "rank_pct": variant.rank_pct if variant.selection == "rank" else "",
        "rank_abs_floor": variant.rank_abs_floor if variant.rank_abs_floor is not None else "",
        "rebalance": variant.rebalance,
        "trail_stop": variant.trail_stop if variant.trail_stop is not None else "",
        "macro_ma200": variant.macro_ma200,
        "cash_yield": variant.cash_yield,
        "water_cap_gate": variant.water_cap_gate if variant.water_cap_gate is not None else "",
        "water_cap_slots": variant.water_cap_slots if variant.water_cap_slots is not None else "",
        "water_cap_hits": water_cap_hits,
        "water_cap_sells": water_cap_sells,
        "crisis_filter": variant.crisis_filter,
        "crisis_ma_window": variant.crisis_ma_window if variant.crisis_filter else "",
        "crisis_vol_quantile": variant.crisis_vol_quantile if variant.crisis_filter else "",
        "crisis_days": crisis_days,
        "crisis_blocked_buys": crisis_blocked_buys,
        "crisis_cap_slots": variant.crisis_cap_slots if variant.crisis_cap_slots is not None else "",
        "crisis_cap_sells": crisis_cap_sells,
        "cppi": variant.cppi,
        "cppi_hwm_mode": variant.cppi_hwm_mode if variant.cppi else "",
        "cppi_hwm_halflife_years": variant.cppi_hwm_halflife_years if variant.cppi_hwm_halflife_years is not None else "",
        "cppi_partial_reset_floor": variant.cppi_partial_reset_floor if variant.cppi_partial_reset_floor is not None else "",
        "cppi_rerisk": variant.cppi_rerisk,
        "cppi_sells": cppi_sells,
        "cppi_hard_stops": cppi_hard_stops,
        "cppi_unlocks": cppi_unlocks,
        "cppi_recovery_steps": cppi_recovery_steps,
        "final_equity_curve": final_equity_curve,
        "liquidation_equity_after_cost": liquidation_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": dd,
        "calmar": cagr / abs(dd) if dd < 0 and not math.isnan(cagr) else math.nan,
        "bench_return": bench_return,
        "bench_max_drawdown": bench_dd,
        "excess_return": total_return - bench_return if not math.isnan(total_return) and not math.isnan(bench_return) else math.nan,
        "trades": len(trade_rows),
        "win_rate": len(wins) / len(net_rets) if net_rets else math.nan,
        "avg_net_ret": sum(net_rets) / len(net_rets) if net_rets else math.nan,
        "median_net_ret": sorted(net_rets)[len(net_rets) // 2] if net_rets else math.nan,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": abs(avg_win / avg_loss) if not math.isnan(avg_win) and not math.isnan(avg_loss) and avg_loss != 0 else math.nan,
        "avg_hold_days": sum(hold_days) / len(hold_days) if hold_days else math.nan,
        "avg_n_pos": sum(n_pos) / len(n_pos) if n_pos else math.nan,
        "avg_cash_ratio": sum(cash_ratios) / len(cash_ratios) if cash_ratios else math.nan,
        "avg_exposure": sum(exposures) / len(exposures) if exposures else math.nan,
        "total_cost_est": total_cost,
        "cash_interest_total": cash_interest_total,
        "rebalance_turnover": rebal_turnover,
        "blocked_buys_macro": blocked_buys_macro,
        "skipped_full": skipped_full,
        "skipped_small": skipped_small,
    }
    return SimResult(variant=variant, equity_rows=equity_rows, trade_rows=trade_rows, summary=summary)


# ============================ 输出报告 ============================


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def write_outputs(results: List[SimResult], meta: dict) -> None:
    ensure_out()
    summary_fields = [
        "variant", "description", "allocation", "selection", "rank_pct", "rank_abs_floor",
        "rebalance", "trail_stop", "macro_ma200", "cash_yield", "water_cap_gate", "water_cap_slots", "water_cap_hits", "water_cap_sells",
        "crisis_filter", "crisis_ma_window", "crisis_vol_quantile", "crisis_days", "crisis_blocked_buys", "crisis_cap_slots", "crisis_cap_sells", "cppi", "cppi_hwm_mode", "cppi_hwm_halflife_years", "cppi_partial_reset_floor", "cppi_rerisk", "cppi_sells", "cppi_hard_stops", "cppi_unlocks", "cppi_recovery_steps",
        "final_equity_curve", "liquidation_equity_after_cost", "total_return", "cagr",
        "max_drawdown", "calmar", "bench_return", "bench_max_drawdown", "excess_return",
        "trades", "win_rate", "avg_net_ret", "median_net_ret", "avg_win", "avg_loss",
        "profit_loss_ratio", "avg_hold_days", "avg_n_pos", "avg_cash_ratio", "avg_exposure",
        "total_cost_est", "cash_interest_total", "rebalance_turnover", "blocked_buys_macro", "skipped_full", "skipped_small",
    ]
    summary_rows = []
    for res in results:
        row = {}
        for k in summary_fields:
            v = res.summary.get(k, "")
            if isinstance(v, float):
                row[k] = f"{v:.10f}" if not math.isnan(v) else ""
            else:
                row[k] = v
        summary_rows.append(row)

        name = sanitize_name(res.variant.name)
        eq_fields = ["date", "equity", "cash", "invested_value", "n_pos", "cash_ratio", "exposure", "cash_interest_cum"]
        tr_fields = [
            "variant", "code", "entry_date", "entry_signal_date", "exit_date", "exit_signal_date",
            "entry_px", "exit_px", "entry_S", "exit_S", "exit_reason", "hold_days",
            "cost_basis", "net_out", "net_ret", "pnl_yuan",
        ]
        write_csv_dicts(os.path.join(OUT_DIR, f"strategy_experiment_equity_{name}.csv"), res.equity_rows, eq_fields)
        write_csv_dicts(os.path.join(OUT_DIR, f"strategy_experiment_trades_{name}.csv"), res.trade_rows, tr_fields)

    write_csv_dicts(os.path.join(OUT_DIR, "strategy_experiment_summary.csv"), summary_rows, summary_fields)
    write_report(results, meta)


def write_report(results: List[SimResult], meta: dict) -> None:
    baseline = next((r for r in results if r.variant.name == "baseline_fixed_slot"), results[0])
    rows_sorted = sorted(results, key=lambda r: ffloat(r.summary.get("cagr")), reverse=True)

    lines: List[str] = []
    lines.append("# 策略构建层正式实验报告")
    lines.append("")
    lines.append("> 本报告由 `strategy_experiment.py` 生成；仅用于量化研究，不构成投资建议。")
    lines.append("")
    lines.append("## 一、实验元数据")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for k in [
        "start", "end", "score_suffix", "score_dates", "score_rows", "navs_loaded",
        "bench_days", "decision_calendar", "capital", "buy_th", "sell_th", "slots", "cost_in", "cost_out",
    ]:
        lines.append(f"| {k} | {meta.get(k, '')} |")
    lines.append("")
    lines.append("## 二、核心对照表")
    lines.append("")
    lines.append("| 变体 | 总收益 | CAGR | 最大回撤 | Calmar | 平均仓位数 | 平均现金 | 现金收益 | 水位触发/卖出 | 危机日/禁买/降仓 | CPPI卖出/熔断/重启/恢复步 | 交易数 | 胜率 | 平均持有 | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows_sorted:
        s = r.summary
        lines.append(
            f"| {r.variant.name} | {pct(s['total_return'])} | {pct(s['cagr'])} | {pct(s['max_drawdown'])} | "
            f"{num(s['calmar'], 2)} | {num(s['avg_n_pos'], 2)} | {pct(s['avg_cash_ratio'])} | "
            f"{s['cash_interest_total']:,.0f} | {int(s.get('water_cap_hits', 0))}/{int(s.get('water_cap_sells', 0))} | "
            f"{int(s.get('crisis_days', 0))}/{int(s.get('crisis_blocked_buys', 0))}/{int(s.get('crisis_cap_sells', 0))} | "
            f"{int(s.get('cppi_sells', 0))}/{int(s.get('cppi_hard_stops', 0))}/{int(s.get('cppi_unlocks', 0))}/{int(s.get('cppi_recovery_steps', 0))} | "
            f"{int(s['trades'])} | {pct(s['win_rate'])} | {num(s['avg_hold_days'], 0)} | {r.variant.description} |"
        )
    lines.append("")

    lines.append("## 三、相对基线增量")
    lines.append("")
    b = baseline.summary
    lines.append("| 变体 | Δ总收益 | ΔCAGR | Δ最大回撤 | Δ平均现金 | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows_sorted:
        s = r.summary
        lines.append(
            f"| {r.variant.name} | {pct(s['total_return'] - b['total_return'])} | "
            f"{pct(s['cagr'] - b['cagr'])} | {pct(s['max_drawdown'] - b['max_drawdown'])} | "
            f"{pct(s['avg_cash_ratio'] - b['avg_cash_ratio'])} | {r.variant.description} |"
        )
    lines.append("")

    lines.append("## 四、读数提示")
    lines.append("")
    lines.append("1. `baseline_fixed_slot` 用于复刻旧逻辑：单次买入金额不超过 `equity / slots`。")
    lines.append("2. `cash_to_empty_slots` 是对 Cash Drag 指控的直接检验：新开仓时把现金均分给剩余空槽。")
    lines.append("3. `rank_top10_*` / `rank_top20_*` 检验“用截面排名替代绝对阈值”的效果。")
    lines.append("4. `monthly_rebalance_*` / `quarterly_rebalance_*` 为研究用近似再平衡，正式实盘仍需申赎到账和短赎费建模。")
    lines.append("5. `trail15_*` / `trail20_*` 检验移动止损是否真的改善组合，而不是只改善个案。")
    lines.append("6. `hs300_ma200_*` 是一个简单宏观趋势过滤雏形，不代表最终宏观择时模型。")
    lines.append("7. `*_crisis_ma200_vol80` 检验 MA200 破位 + 20日实现波动率突破历史80%分位的危机禁买逻辑。")
    lines.append("8. `*_cppi_*` 检验组合自身回撤预算：-10%限6槽、-15%限3槽、-20%清仓并等待右侧信号重启。")
    lines.append("9. 本实验仍继承输入基金池的幸存者偏差；要回答真实 Alpha，必须补全已清盘/合并基金的 PiT 数据。")
    lines.append("")
    lines.append("## 五、文件索引")
    lines.append("")
    lines.append("- 汇总：`output/strategy_experiment_summary.csv`")
    lines.append("- 各变体日频权益：`output/strategy_experiment_equity_<variant>.csv`")
    lines.append("- 各变体交易账：`output/strategy_experiment_trades_<variant>.csv`")

    with open(os.path.join(OUT_DIR, "strategy_experiment_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================ CLI ============================


def main() -> None:
    ap = argparse.ArgumentParser(description="正式策略构建层实验：资金分配/再平衡/截面排名/止损/趋势过滤")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--score-suffix", default="auto", help="如 _2e4ec0f5；auto=自动选择总行数最多的缓存后缀")
    ap.add_argument("--decision-calendar", choices=["exact", "previous"], default="exact",
                    help="exact=评分日必须是交易日才交易，复刻旧口径；previous=落到之前最后一个交易日")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--buy", type=float, default=DEFAULT_BUY)
    ap.add_argument("--sell", type=float, default=DEFAULT_SELL)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--cost-in", type=float, default=DEFAULT_COST_IN)
    ap.add_argument("--cost-out", type=float, default=DEFAULT_COST_OUT)
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    if end <= start:
        raise ValueError("end must be greater than start")

    suffix = choose_score_suffix(start, end) if args.score_suffix == "auto" else args.score_suffix
    panel, score_dates, score_rows = load_panel(start, end, suffix)
    codes = sorted({c for g in panel.values() for c in g})
    navs = load_navs(codes)
    bench = load_bench(start, end)

    cfg = SimulatorConfig(
        capital=args.capital,
        slots=args.slots,
        cost_in=args.cost_in,
        cost_out=args.cost_out,
        decision_calendar=args.decision_calendar,
    )
    variants = default_variants(args.buy, args.sell)

    print(f"[实验] {start} → {end} | suffix={suffix or '(empty)'} | decision={args.decision_calendar}")
    print(f"[数据] 评分日期 {len(score_dates)} 个 | 评分行 {score_rows} | 净值 {len(navs)}/{len(codes)} | 基准交易日 {len(bench)}")

    results: List[SimResult] = []
    for v in variants:
        print(f"  -> {v.name} ...", flush=True)
        res = simulate_variant(v, panel, score_dates, navs, bench, cfg, start, end)
        results.append(res)
        s = res.summary
        print(
            f"     总收益 {s['total_return']:+.2%} | CAGR {s['cagr']:+.2%} | "
            f"DD {s['max_drawdown']:.2%} | 平均现金 {s['avg_cash_ratio']:.2%} | 交易 {s['trades']}"
        )

    meta = {
        "start": args.start,
        "end": args.end,
        "score_suffix": suffix,
        "score_dates": len(score_dates),
        "score_rows": score_rows,
        "navs_loaded": f"{len(navs)}/{len(codes)}",
        "bench_days": len(bench),
        "decision_calendar": args.decision_calendar,
        "capital": args.capital,
        "buy_th": args.buy,
        "sell_th": args.sell,
        "slots": args.slots,
        "cost_in": args.cost_in,
        "cost_out": args.cost_out,
    }
    write_outputs(results, meta)
    print("[输出] output/strategy_experiment_summary.csv")
    print("[输出] output/strategy_experiment_report.md")
    print("[输出] output/strategy_experiment_equity_<variant>.csv / trades_<variant>.csv")


if __name__ == "__main__":
    main()
