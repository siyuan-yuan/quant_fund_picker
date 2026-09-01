#!/usr/bin/env python3
"""Phase 3 波次1：V3.8 执行层参数化回测 harness（P1-0 严格 PIT 面板）。

预登记: docs/优化计划_V4_2026-08.md §2 预登记 #3（实验前写入）。

变体（主基线 = v38_cost）:
  base_zero  V3.8 规则 + 零成本（诊断参考）
  v38_cost   V3.8 规则 + realistic 成本阶梯（主基线）
  vol_target CPPI 替换为连续波动率目标 (bench 126d vol, 目标 15%, clip[0.3,1.0])
  no_overlay 无风险叠加（10 槽固定, 对照）
  conviction S>85 基金目标权重 1.2×槽位
  cap25/30/35 单一 top1_style 组合权重上限 25/30/35%

协议要点（与 backtest_local.simulate 同式）:
  决策日=月末最后交易日; 买入 S>70 降序 min(cash, eq*wc/10) 不填满空槽; 卖出 S<45;
  移动止损 20% 逐日; 危机禁买 bench<MA200 且 20d vol>历史80分位(min60);
  现金 2.5% 年化逐日计息; 季末等权再平衡(只调存量);
  CPPI -15/-20/-25→6/3/0 槽 + 硬熔断锁 + 右侧信号解锁(HWM 重置);
  持有基金缺席当月样本池 → engine.score_fund(bt=True) 重打分并入批 rank（确定性缓存）。

产物: output/p3/p3_summary.csv, p3_curves.csv, p3_trades.csv, p3_cost_dist.csv
复现: ./.venv/bin/python p3_backtest.py            # 全变体
      ./.venv/bin/python p3_backtest.py --only v38_cost
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import provider  # noqa: E402

provider.STALE_OK = True
import p1_panel_build as PB  # noqa: E402  (score_one, aggregate, 常量)

PANEL_DIR = os.path.join(ROOT, "output", "p1_panel")
OUT_DIR = os.path.join(ROOT, "output", "p3")
NAV_DIR = os.path.join(ROOT, "cache")
BENCH_CSV = os.path.join(NAV_DIR, "idx_sh000300.csv")
SCORE_CACHE_CSV = os.path.join(OUT_DIR, "score_cache.csv")

# ---------------- V3.8 常数（与 config 同值, 便于单文件复现） ----------------
SLOTS = 10
BUY_TH, SELL_TH = 70.0, 45.0
CASH_YIELD = 0.025
TRAIL_STOP = 0.20
CRISIS_MA = 200
CRISIS_VOL_WINDOW, CRISIS_VOL_Q, CRISIS_VOL_MINH = 20, 0.80, 60
CPPI_DD1, CPPI_S1 = -0.15, 6
CPPI_DD2, CPPI_S2 = -0.20, 3
CPPI_DD3, CPPI_S3 = -0.25, 0
OVERSEAS_STYLES = {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}
RECENT_START = "2023-04-01"

# 成本（预登记 #3 P3-1）
COST_IN = 0.0015


def cost_out(hold_days: float, is_ov: bool) -> float:
    if is_ov:
        return 0.005
    if hold_days < 7:
        return 0.015
    if hold_days < 30:
        return 0.0075
    if hold_days < 180:
        return 0.005
    if hold_days < 730:
        return 0.0025
    return 0.0


# ---------------- 数据 ----------------

class NavLib:
    """按需加载净值: code → Series(nav, index=date)"""

    def __init__(self):
        self._s: dict[str, pd.Series] = {}

    def series(self, code: str) -> pd.Series | None:
        if code in self._s:
            return self._s[code]
        p = os.path.join(NAV_DIR, f"nav_{code}.csv")
        if not os.path.exists(p):
            self._s[code] = None
            return None
        df = pd.read_csv(p, parse_dates=["date"])
        s = (df.set_index("date")["nav"].astype(float).sort_index()
             [~df.set_index("date").index.duplicated(keep="last")])
        s = s[~s.index.duplicated(keep="last")].dropna()
        s = s[s > 0]  # 脏数据: 净值<=0 视为缺失
        self._s[code] = s if len(s) else None
        return self._s[code]

    def px(self, code: str, day) -> float:
        s = self.series(code)
        if s is None:
            return np.nan
        v = s.asof(pd.Timestamp(day))
        return float(v) if (v == v and v > 0) else np.nan


def load_bench() -> pd.Series:
    df = pd.read_csv(BENCH_CSV, parse_dates=["date"])
    return df.set_index("date")["close"].astype(float).sort_index()


def load_panel() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(os.listdir(PANEL_DIR)):
        if not (f.endswith(".csv") and f[:2].isdigit()):
            continue
        d = pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str})
        if "date" in d.columns and len(d) > 0:
            out[str(d["date"].iloc[0])] = d
    return out


RAW_COLS = ["code", "name", "ftype", "is_passive", "n_days", "F_value", "F_alpha",
            "R_MDD", "smallcap_exp", "top3_conc", "penalties", "r4", "r7",
            "top1_style", "top1_w"]


def _parse_pens(x):
    if x is None or (isinstance(x, float) and x != x):
        return []
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return [float(p) for p in v] if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    return [float(p) for p in x] if isinstance(x, (list, tuple)) else []


class ScoreCache:
    """(date, code) → score_one 结果, 落盘共享（跨变体）"""

    def __init__(self):
        self.tab: dict[tuple, dict] = {}
        if os.path.exists(SCORE_CACHE_CSV):
            df = pd.read_csv(SCORE_CACHE_CSV, dtype={"code": str})
            if "date" not in df.columns:
                print("score_cache 缺 date 列(已损坏), 重建缓存")
                os.remove(SCORE_CACHE_CSV)
                return
            for r in df.itertuples():
                self.tab[(r.date, r.code)] = {
                    "date": r.date, "code": r.code, "name": r.name, "ftype": r.ftype,
                    "is_passive": bool(r.is_passive), "n_days": r.n_days,
                    "F_value": r.F_value, "F_alpha": r.F_alpha,
                    "R_MDD": r.R_MDD, "smallcap_exp": r.smallcap_exp,
                    "top3_conc": r.top3_conc, "r4": r.r4, "r7": r.r7,
                    "top1_style": r.top1_style, "top1_w": r.top1_w,
                    "penalties": _parse_pens(r.penalties),
                }
            print(f"score_cache: 载入 {len(self.tab)} 行")

    def get(self, date: str, code: str) -> dict | None:
        return self.tab.get((date, code))

    def put(self, rec: dict):
        self.tab[(str(rec["date"]), rec["code"])] = rec

    def save(self):
        rows = [v for _, v in sorted(self.tab.items())]
        if rows:
            pd.DataFrame(rows).to_csv(SCORE_CACHE_CSV, index=False, encoding="utf-8-sig")


def month_batch(date: str, panel: dict, sc: ScoreCache, held: list[str]) -> pd.DataFrame:
    """当月决策批 = 面板样本 + 缺席样本池的持有基金重打分; aggregate 同式重算 S_total"""
    base = panel[date]
    have = set(base["code"])
    missing = [c for c in held if c not in have]
    extra = []
    for c in missing:
        rec = sc.get(date, c)
        if rec is None:
            provider._memo.clear()
            rec = PB.score_one(c, date)
            if rec is not None:
                rec["date"] = date
                sc.put(rec)
        if rec is not None:
            extra.append(rec)
    df = base[RAW_COLS].copy()
    df["penalties"] = df["penalties"].apply(_parse_pens)
    if extra:
        ex = pd.DataFrame(extra)
        ex["penalties"] = [list(p) for p in ex["penalties"]]
        df = pd.concat([df, ex[RAW_COLS]], ignore_index=True)
    return PB.aggregate(df, date)


# ---------------- 模拟器 ----------------

# ---------------- H-P4A (预登记 #7): PIT IC 重加权评分变体 ----------------
def _reweight_s(df: pd.DataFrame, wv: float, wa: float, wm: float) -> list:
    """与 PB._total / PB.aggregate 同式, 但因子权重换成外部传入 (S_pit 世界)"""
    def _t(r):
        num = den = 0.0
        fv, fa, fm = r.F_value, r.F_alpha, r.F_momentum
        if fv is not None and fv == fv:
            num += wv * min(fv, 100); den += wv
        if fa is not None and fa == fa:
            num += wa * fa; den += wa
        if fm is not None and fm == fm:
            num += wm * fm; den += wm
        base = num / den if den > 1e-9 else 0.0
        s = base
        for p in (r.penalties or []):
            s *= (1 - p)
        return round(max(0.0, min(100.0, s)), 1)
    return [_t(r) for r in df.itertuples()]


SPIT_W: dict[str, tuple[float, float, float]] = {}   # date → (wv, wa, wm), 由 main 载入


def load_spit_weights() -> None:
    """载入 H-P4A 预登记的 S3 PIT 权重表 (扩展窗 |IC(fwd6)| 归一化, 完全 PIT)"""
    fp = os.path.join(ROOT, "output", "p4", "hp4a_pit_weights.csv")
    df = pd.read_csv(fp, dtype={"date": str})
    for r in df.itertuples():
        SPIT_W[r.date] = (float(r.w_value), float(r.w_alpha), float(r.w_mom))
    print(f"spit 权重表: 载入 {len(SPIT_W)} 个月 ({min(SPIT_W)}→{max(SPIT_W)})", flush=True)


def load_ov_dyn_cap() -> dict[str, int]:
    """P3-6 预登记 #4: diff = ret252(INX) - ret252(沪深300), 严格 PIT(取决策日前一个日历日的 INX 收盘)
    → 上限 4(diff>+15%) / 3(|diff|<=15%) / 2(diff<-15%); 信号缺失=3"""
    us = pd.read_csv(os.path.join(NAV_DIR, "idx_us__INX.csv"), parse_dates=["date"])
    us = us.set_index("date")["close"].astype(float).sort_index()
    out = {}
    for d in sorted(panel_dates_global):
        ts = pd.Timestamp(d)
        a = bench_global.asof(ts)
        u = us.asof(ts - pd.Timedelta(days=1))  # 美股同日历日收盘在 A 股决策时点未知
        if a != a or u != u:
            out[d] = 3
            continue
        ia = bench_global.asof(ts - pd.Timedelta(days=253))
        iu = us.asof(ts - pd.Timedelta(days=1 + 253))
        if ia != ia or iu != iu or ia <= 0 or iu <= 0:
            out[d] = 3
            continue
        diff = (u / iu - 1) - (a / ia - 1)
        out[d] = 4 if diff > 0.15 else (2 if diff < -0.15 else 3)
    return out


bench_global: pd.Series | None = None
panel_dates_global: list[str] = []


def simulate(cfg: dict, bench: pd.Series, panel: dict, sc: ScoreCache,
             navs: NavLib, first_day: str, last_day: str,
             ov_dyn_cap: dict[str, int] | None = None):
    buy_mode = cfg.get("buy_mode", "abs")
    sell_th = float(cfg.get("sell_th", SELL_TH))
    ov_mode = cfg.get("ov_mode", "none")
    cash = 100_000.0
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    curve: list[dict] = []
    total_cost = 0.0
    rebal_turnover = 0.0
    last_scores: pd.DataFrame | None = None
    prev_day = None
    cppi_hwm, cppi_locked = cash, False
    cppi_hard_stops = cppi_unlocks = 0

    # 危机信号（legacy 同式, 分位只用当日之前历史）
    ret = bench.pct_change()
    from collections import deque
    recent_eq: deque = deque(maxlen=252)   # roll_hwm 用: 过去 252d 组合权益
    eff_cap, prev_dec_raw, hyst_upshifts = SLOTS, None, 0   # cap_hyst 用
    ma = bench.rolling(CRISIS_MA).mean()
    vol20 = ret.rolling(CRISIS_VOL_WINDOW).std() * np.sqrt(252)
    vals, th = [], []
    for v in vol20.values:
        hist = [x for x in vals if x == x]
        th.append(float(np.quantile(hist, CRISIS_VOL_Q)) if len(hist) >= CRISIS_VOL_MINH else np.nan)
        if v == v:
            vals.append(float(v))
    vol_th = pd.Series(th, index=bench.index)
    vol126 = ret.rolling(126).std() * np.sqrt(252)

    def is_crisis(day):
        if day not in bench.index:
            return False
        b, m, v, t = (bench.asof(day), ma.asof(day), vol20.asof(day), vol_th.asof(day))
        return bool(pd.notna(m) and pd.notna(v) and pd.notna(t) and b < m and v > t)

    def equity(day):
        return cash + sum(p["units"] * navs.px(c, day) for c, p in positions.items())

    def sell(code, day, reason, exit_s=np.nan):
        nonlocal cash, total_cost
        p = positions.pop(code)
        price = navs.px(code, day)
        if price != price:
            positions[code] = p
            return False
        gross = p["units"] * price
        fee = gross * (cost_out((pd.Timestamp(day) - pd.Timestamp(p["entry_date"])).days, p["is_ov"])
                       if cfg["cost"] == "ladder" else 0.0)
        cash += gross - fee
        total_cost += fee
        trades.append(dict(variant=cfg["name"], code=code,
                           entry_date=p["entry_date"], exit_date=str(pd.Timestamp(day).date()),
                           hold_days=(pd.Timestamp(day) - pd.Timestamp(p["entry_date"])).days,
                           is_ov=p["is_ov"], style=p["style"],
                           entry_S=p["entry_S"],
                           exit_S=round(exit_s, 1) if exit_s == exit_s else np.nan,
                           reason=reason, cost=round(fee, 2),
                           pnl=round(gross - fee - p["alloc"], 2)))
        return True

    def trim_to_cap(day, cap, scores, reason):
        nonlocal cash, total_cost, rebal_turnover
        cap = max(0, int(cap))
        if cap == 0:
            for c in list(positions):
                s_ = scores.loc[c, "S_total"] if scores is not None and c in scores.index else np.nan
                sell(c, day, reason, exit_s=s_)
            return
        if len(positions) > cap:
            def s_of(c):
                if scores is not None and c in scores.index:
                    s_ = scores.loc[c, "S_total"]
                    return float(s_) if s_ == s_ else -1.0
                return -1.0
            for c in sorted(positions, key=s_of, reverse=True)[cap:]:
                sell(c, day, reason, exit_s=s_of(c))
        eq = equity(day)
        if eq <= 0:
            return
        for c in list(positions):
            target = eq * positions[c]["wc"] / SLOTS
            price = navs.px(c, day)
            if price != price or target <= 0:
                continue
            val = positions[c]["units"] * price
            if val <= target * 1.01:
                continue
            sell_units = (val - target) / price
            if sell_units <= 0 or sell_units >= positions[c]["units"]:
                continue
            ratio = sell_units / positions[c]["units"]
            fee = sell_units * price * (
                cost_out((pd.Timestamp(day) - pd.Timestamp(positions[c]["entry_date"])).days,
                         positions[c]["is_ov"]) if cfg["cost"] == "ladder" else 0.0)
            positions[c]["units"] -= sell_units
            positions[c]["alloc"] *= (1 - ratio)
            cash += sell_units * price - fee
            total_cost += fee
            rebal_turnover += sell_units * price

    def rebalance(day):
        nonlocal cash, total_cost, rebal_turnover
        if not positions:
            return
        eq = equity(day)
        for c in list(positions):
            target = eq * positions[c]["wc"] / SLOTS
            price = navs.px(c, day)
            if price != price:
                continue
            val = positions[c]["units"] * price
            if val > target * 1.01:
                sell_units = (val - target) / price
                if 0 < sell_units < positions[c]["units"]:
                    fee = sell_units * price * (
                        cost_out((pd.Timestamp(day) - pd.Timestamp(positions[c]["entry_date"])).days,
                                 positions[c]["is_ov"]) if cfg["cost"] == "ladder" else 0.0)
                    positions[c]["units"] -= sell_units
                    positions[c]["alloc"] *= 1 - sell_units / positions[c]["units"]
                    cash += sell_units * price - fee
                    total_cost += fee
                    rebal_turnover += sell_units * price
        for c in list(positions):
            if cash <= 0:
                break
            target = eq * positions[c]["wc"] / SLOTS
            price = navs.px(c, day)
            if price != price:
                continue
            val = positions[c]["units"] * price
            if val >= target * 0.99:
                continue
            ci = COST_IN if cfg["cost"] == "ladder" else 0.0
            alloc = min(cash, (target - val) * (1 + ci))
            if alloc <= 0:
                continue
            positions[c]["units"] += alloc / (price * (1 + ci))
            positions[c]["alloc"] += alloc
            cash -= alloc
            total_cost += alloc * ci / (1 + ci)
            rebal_turnover += alloc

    def style_weight(day, style):
        return sum(p["units"] * navs.px(c, day) for c, p in positions.items() if p["style"] == style)

    start_b = bench.index[bench.index <= pd.Timestamp(first_day)]
    if not len(start_b):
        raise SystemExit(f"bench 早于首个决策月 {first_day}")
    day_grid = bench.loc[start_b[-1]: pd.Timestamp(last_day)].index
    # 决策日映射: 面板决策月(自然月末) → 其当日或之前最后一个交易日执行
    # (P1-0 的 as_of 为自然月末, 数据严格截断; 周末落空的月末在此映射回最后交易日,
    #  保证每个决策月都执行 —— 与 legacy 静默跳过周末月末不同, 已在预登记中披露)
    decision_map: dict[pd.Timestamp, str] = {}
    for pd_ in sorted(panel.keys()):
        ts = pd.Timestamp(pd_)
        cands = bench.index[bench.index <= ts]
        if len(cands):
            exec_day = cands[-1]
            if exec_day in day_grid:
                decision_map[exec_day] = pd_
    for day in day_grid:
        if prev_day is not None:
            dt = max(0, (day - prev_day).days)
            if dt:
                cash += cash * CASH_YIELD * dt / 365.25
        prev_day = day

        # 移动止损（逐日）
        for c in list(positions):
            price = navs.px(c, day)
            if price == price:
                positions[c]["peak"] = max(positions[c]["peak"], price)
                if price < positions[c]["peak"] * (1 - TRAIL_STOP):
                    sell(c, day, f"trail{TRAIL_STOP:.0%}")

        crisis = is_crisis(day)

        # 风险叠加 → daily_cap（预登记 #5: base / roll_hwm / cap_hyst）
        daily_cap = SLOTS
        if cfg["overlay"] == "cppi":
            eq_now = equity(day)
            mode = cfg.get("cppi_mode", "base")
            if mode == "roll_hwm":
                hwm = eq_now
                if recent_eq:
                    hwm = max(hwm, max(recent_eq))
                dd = eq_now / hwm - 1 if hwm > 0 else 0.0
            else:
                if not cppi_locked and eq_now > cppi_hwm:
                    cppi_hwm = eq_now
                dd = eq_now / cppi_hwm - 1 if cppi_hwm > 0 else 0.0
            if cppi_locked:
                daily_cap = 0
            elif dd <= CPPI_DD3:
                daily_cap, cppi_locked = CPPI_S3, True
                cppi_hard_stops += 1
                eff_cap, prev_dec_raw = CPPI_S3, None
            elif mode == "cap_hyst":
                raw_cap = CPPI_S2 if dd <= CPPI_DD2 else (CPPI_S1 if dd <= CPPI_DD1 else SLOTS)
                if raw_cap < eff_cap:
                    eff_cap = raw_cap          # 下调当日立即生效
                elif raw_cap > eff_cap and day in decision_map:
                    if prev_dec_raw == raw_cap:  # 连续 2 决策日同一更高档 → 上调
                        eff_cap = raw_cap
                        hyst_upshifts += 1
                    prev_dec_raw = raw_cap
                daily_cap = eff_cap
            else:
                if dd <= CPPI_DD2:
                    daily_cap = CPPI_S2
                elif dd <= CPPI_DD1:
                    daily_cap = CPPI_S1
        elif cfg["overlay"] == "voltgt":
            v = vol126.asof(day)
            if v == v and v > 0:
                daily_cap = int(round(SLOTS * float(np.clip(0.15 / v, 0.3, 1.0))))
            else:
                daily_cap = SLOTS
        if daily_cap < len(positions):
            trim_to_cap(day, daily_cap, last_scores, f"cap{daily_cap}")

        if day in decision_map:
            day_str = decision_map[day]
            batch = month_batch(day_str, panel, sc, list(positions))
            if cfg.get("score_spit"):
                if day_str not in SPIT_W:
                    raise RuntimeError(f"spit 权重表缺决策月 {day_str}")
                batch["S_total"] = _reweight_s(batch, *SPIT_W[day_str])
            last_scores = batch

            # 买入线（预登记 #4: abs=S>70 现行 / p90=max(60, P90) 相对线）
            if buy_mode == "p90":
                p90 = float(batch["S_total"].quantile(0.90))
                gate = max(60.0, p90)
            else:
                gate = BUY_TH

            # CPPI 锁/解锁（右侧信号: 有 S 过线候选且非危机）
            if cfg["overlay"] == "cppi" and cppi_locked:
                hit = batch["S_total"] >= gate if buy_mode == "p90" else batch["S_total"] > gate
                if hit.any() and not crisis:
                    cppi_locked = False
                    cppi_hwm = equity(day)
                    daily_cap = SLOTS
                    cppi_unlocks += 1
                    eff_cap, prev_dec_raw = SLOTS, None  # 状态迁移, 不受迟滞约束(预登记 #5)
                else:
                    daily_cap = 0

            # S<sell_th 卖出（预登记 #4: 45 现行 / 40 放宽迟滞带）
            for c in list(positions):
                if c in batch["code"].values:
                    s_ = batch.loc[batch["code"] == c, "S_total"].iloc[0]
                    if s_ == s_ and s_ < sell_th:
                        sell(c, day, f"S<{sell_th:.0f}", exit_s=s_)

            # 风格预算 trim
            cap_style = cfg.get("style_cap")
            if cap_style is not None:
                eq = equity(day)
                for style in sorted({p["style"] for p in positions.values()}):
                    w = style_weight(day, style)
                    if w > cap_style * eq + 1e-9:
                        members = [c for c in positions if positions[c]["style"] == style]

                        def s_of(c):
                            if c in batch["code"].values:
                                s_ = batch.loc[batch["code"] == c, "S_total"].iloc[0]
                                return float(s_) if s_ == s_ else -1.0
                            return -1.0
                        for c in sorted(members, key=s_of, reverse=True)[1:]:
                            if style_weight(day, style) <= cap_style * eq + 1e-9:
                                break
                            sell(c, day, f"style_cap{int(cap_style*100)}", exit_s=s_of(c))
                        # 剩余超限部分减持
                        if style_weight(day, style) > cap_style * eq + 1e-9:
                            for c in positions:
                                if positions[c]["style"] != style:
                                    continue
                                price = navs.px(c, day)
                                if price != price:
                                    continue
                                excess = style_weight(day, style) - cap_style * equity(day)
                                val = positions[c]["units"] * price
                                if val <= excess:
                                    sell(c, day, f"style_cap{int(cap_style*100)}", exit_s=s_of(c))
                                else:
                                    ratio = excess / val
                                    fee = val * ratio * (
                                        cost_out((pd.Timestamp(day) - pd.Timestamp(positions[c]["entry_date"])).days,
                                                 positions[c]["is_ov"]) if cfg["cost"] == "ladder" else 0.0)
                                    positions[c]["units"] *= (1 - ratio)
                                    positions[c]["alloc"] *= (1 - ratio)
                                    cash += val * ratio - fee
                                    total_cost += fee
                                    break

            if cfg["overlay"] == "cppi" and daily_cap < len(positions):
                trim_to_cap(day, daily_cap, last_scores, f"cppi_cap{daily_cap}")

            # 海外槽位上限（预登记 #4: fix3 固定3 / dyn 2-3-4 随美A 252d 相对趋势）
            ov_cap = np.inf
            if ov_mode == "fix3":
                ov_cap = 3
            elif ov_mode == "dyn" and ov_dyn_cap is not None:
                ov_cap = int(ov_dyn_cap.get(day_str, 3))
            ov_pos = [c for c, p in positions.items() if p["is_ov"]]
            if np.isfinite(ov_cap) and len(ov_pos) > ov_cap:
                def s_of(c):
                    if c in batch["code"].values:
                        s_ = batch.loc[batch["code"] == c, "S_total"].iloc[0]
                        return float(s_) if s_ == s_ else -1.0
                    return -1.0
                for c in sorted(ov_pos, key=s_of, reverse=True)[int(ov_cap):]:
                    sell(c, day, f"ovcap{int(ov_cap)}", exit_s=s_of(c))

            # 季末等权再平衡
            if day.month in (3, 6, 9, 12):
                rebalance(day)

            # 买入
            if not crisis and daily_cap > 0:
                if buy_mode == "p90":
                    cands = batch[batch["S_total"] >= gate].sort_values("S_total", ascending=False)
                else:
                    cands = batch[batch["S_total"] > gate].sort_values("S_total", ascending=False)
                for _, row in cands.iterrows():
                    c = row["code"]
                    if c in positions or len(positions) >= min(SLOTS, daily_cap):
                        break
                    eq_now = equity(day)
                    wc = 1.2 if (cfg["conviction"] and row["S_total"] > 85) else 1.0
                    alloc = min(cash, eq_now * wc / SLOTS)
                    if alloc < eq_now * 0.02:
                        continue
                    if cap_style is not None:
                        st = row["top1_style"] if isinstance(row["top1_style"], str) else ""
                        if st and style_weight(day, st) + alloc > cap_style * eq_now:
                            continue
                    price = navs.px(c, day)
                    if price != price:
                        continue
                    ci = COST_IN if cfg["cost"] == "ladder" else 0.0
                    is_ov = row["top1_style"] in OVERSEAS_STYLES if isinstance(row["top1_style"], str) else False
                    if is_ov and np.isfinite(ov_cap) and \
                            sum(1 for p in positions.values() if p["is_ov"]) >= ov_cap:
                        continue
                    positions[c] = dict(units=alloc / (price * (1 + ci)),
                                        entry_date=str(day.date()), entry_px=price, entry_S=float(row["S_total"]),
                                        alloc=alloc, peak=price, wc=wc, is_ov=is_ov,
                                        style=row["top1_style"] if isinstance(row["top1_style"], str) else "")
                    cash -= alloc
                    total_cost += alloc * ci / (1 + ci)

        eq = equity(day)
        recent_eq.append(eq)
        curve.append(dict(date=day, equity=eq, n_pos=len(positions),
                          cash=cash, cash_ratio=cash / eq if eq else np.nan,
                          crisis=crisis, locked=cppi_locked, cap_used=daily_cap))

    # 期末清算
    last_d = day_grid[-1]
    for c in list(positions):
        sell(c, last_d, "期末清算")
    eq = equity(last_d)
    curve.append(dict(date=last_d, equity=eq, n_pos=0, cash=cash,
                      cash_ratio=1.0, crisis=False, locked=cppi_locked, cap_used=daily_cap))

    ec = pd.DataFrame(curve).set_index("date")
    ec["drawdown"] = ec.equity / ec.equity.cummax() - 1
    return ec, pd.DataFrame(trades)


# ---------------- 指标 ----------------

def perf(eq: pd.Series, tr: pd.DataFrame, window: str) -> dict:
    if window == "full":
        e = eq
        t = tr
    else:
        e = eq[eq.index >= RECENT_START]
        t = tr[tr["exit_date"] >= RECENT_START] if len(tr) else tr
    t0, t1 = e.index[0], e.index[-1]
    yrs = (t1 - t0).days / 365.25
    if yrs <= 0 or e.iloc[0] <= 0:
        return dict(window=window)
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    r = e.pct_change().dropna()
    vol = float(r.std() * np.sqrt(252)) if len(r) else np.nan
    mdd = float((e / e.cummax() - 1).min())
    monthly = e.resample("ME").last().pct_change().dropna()
    return dict(window=window, start=str(t0.date()), end=str(t1.date()), yrs=round(yrs, 2),
                CAGR=round(cagr, 4), ann_vol=round(vol, 4), MaxDD=round(mdd, 4),
                Calmar=round(cagr / abs(mdd), 3) if mdd < 0 else np.nan,
                win_rate_m=round(float((monthly > 0).mean()), 3),
                n_trades=int(len(t)), avg_hold_days=round(float(t["hold_days"].mean()), 1) if len(t) else np.nan,
                total_cost=round(float(t["cost"].sum()), 0) if len(t) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑一个变体名")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载数据 ...", flush=True)
    bench = load_bench()
    panel = load_panel()
    dates = sorted(panel.keys())
    first_day, last_day = dates[0], dates[-1]
    global bench_global, panel_dates_global
    bench_global, panel_dates_global = bench, dates
    sc = ScoreCache()
    navs = NavLib()
    print(f"面板 {len(dates)} 个月 ({first_day}→{last_day}), bench {len(bench)} 交易日")

    VARIANTS = [
        dict(name="base_zero", cost="zero", overlay="cppi", conviction=False, style_cap=None),
        dict(name="v38_cost", cost="ladder", overlay="cppi", conviction=False, style_cap=None),
        dict(name="vol_target", cost="ladder", overlay="voltgt", conviction=False, style_cap=None),
        dict(name="no_overlay", cost="ladder", overlay="none", conviction=False, style_cap=None),
        dict(name="conviction", cost="ladder", overlay="cppi", conviction=True, style_cap=None),
        dict(name="cap25", cost="ladder", overlay="cppi", conviction=False, style_cap=0.25),
        dict(name="cap30", cost="ladder", overlay="cppi", conviction=False, style_cap=0.30),
        dict(name="cap35", cost="ladder", overlay="cppi", conviction=False, style_cap=0.35),
        # ---- 波次2（预登记 #4）----
        dict(name="p90", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="p90", sell_th=45.0, ov_mode="none"),
        dict(name="sell40", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=40.0, ov_mode="none"),
        dict(name="xm_fix3", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=45.0, ov_mode="fix3"),
        dict(name="xm_dyn", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=45.0, ov_mode="dyn"),
        # ---- 波次3（预登记 #5: H-P3B CPPI 棘轮陷阱）----
        dict(name="roll_hwm", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=45.0, ov_mode="none", cppi_mode="roll_hwm"),
        dict(name="cap_hyst", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=45.0, ov_mode="none", cppi_mode="cap_hyst"),
        # ---- H-P4A 条件分支（预登记 #7: HP4A-4）----
        # 评分层 = P1-2 S3 规则的 PIT IC 重加权（权重表 output/p4/hp4a_pit_weights.csv,
        # 仅用决策月之前数据）; 执行层与 v38_cost 完全相同（ladder 成本 + CPPI + 现行全部规则）
        dict(name="spit", cost="ladder", overlay="cppi", conviction=False, style_cap=None,
             buy_mode="abs", sell_th=45.0, ov_mode="none", score_spit=True),
    ]
    if args.only:
        VARIANTS = [v for v in VARIANTS if v["name"] == args.only]

    if any(v.get("score_spit") for v in VARIANTS):
        load_spit_weights()
    ov_dyn_cap = load_ov_dyn_cap() if any(v.get("ov_mode") == "dyn" for v in VARIANTS) else None
    if ov_dyn_cap is not None:
        from collections import Counter
        print("xm_dyn 信号: 上限分布 =", dict(Counter(ov_dyn_cap.values())))

    curves, all_trades = {}, []
    for cfg in VARIANTS:
        t0 = time.time()
        tag = f"{cfg['overlay']}/{cfg['cost']}"
        if cfg["conviction"]:
            tag += "+conv"
        if cfg["style_cap"]:
            tag += f"+cap{int(cfg['style_cap'] * 100)}"
        if cfg.get("buy_mode") == "p90":
            tag += "+p90"
        if float(cfg.get("sell_th", 45.0)) != 45.0:
            tag += f"+s{cfg['sell_th']:.0f}"
        if cfg.get("ov_mode") not in (None, "none"):
            tag += f"+ov{cfg['ov_mode']}"
        if cfg.get("cppi_mode") not in (None, "base"):
            tag += f"+{cfg['cppi_mode']}"
        if cfg.get("score_spit"):
            tag += "+spit"
        print(f"== {cfg['name']} ({tag}) ==", flush=True)
        ec, tr = simulate(cfg, bench, panel, sc, navs, first_day, last_day, ov_dyn_cap)
        ec.attrs["n_pos_mean"] = float(ec.n_pos.mean())
        curves[cfg["name"]] = ec
        if len(tr):
            all_trades.append(tr)
        sc.save()
        p_full = perf(ec["equity"], tr, "full")
        print(f"   full: CAGR={p_full.get('CAGR')} MaxDD={p_full.get('MaxDD')} "
              f"Calmar={p_full.get('Calmar')} trades={p_full.get('n_trades')} "
              f"({time.time() - t0:.0f}s)", flush=True)

    pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT_DIR, "p3_trades.csv"),
                                                    index=False, encoding="utf-8-sig")
    wide = pd.DataFrame({k: v["equity"] for k, v in curves.items()})
    wide.to_csv(os.path.join(OUT_DIR, "p3_curves.csv"), encoding="utf-8-sig")

    rows = []
    for name, ec in curves.items():
        parts = [t for t in all_trades if (t["variant"] == name).all()]
        tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for w in ["full", "recent"]:
            p = perf(ec["equity"], tr, w)
            p["variant"] = name
            p["mean_n_pos"] = round(float(ec.n_pos.mean()), 1)
            p["mean_cash"] = round(float(ec.cash_ratio.mean()), 3)
            rows.append(p)
    # 基准
    be = bench.loc[first_day:last_day]
    for w, b in [("full", be), ("recent", be[be.index >= RECENT_START])]:
        yrs = (b.index[-1] - b.index[0]).days / 365.25
        cagr = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
        mdd = float((b / b.cummax() - 1).min())
        rows.append(dict(variant="BENCH_sh000300", window=w, start=str(b.index[0].date()),
                         end=str(b.index[-1].date()), yrs=round(yrs, 2), CAGR=round(cagr, 4),
                         ann_vol=round(float(b.pct_change().std() * np.sqrt(252)), 4),
                         MaxDD=round(mdd, 4), Calmar=round(cagr / abs(mdd), 3)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR, "p3_summary.csv"), index=False, encoding="utf-8-sig")

    # P3-1 持有期分布（v38_cost）
    parts = [t for t in all_trades if (t["variant"] == "v38_cost").all()]
    tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(tr):
        bins = [0, 7, 30, 180, 365, 730, 10**9]
        lab = ["<7d", "7-30d", "30-180d", "180d-1y", "1-2y", ">2y"]
        dist = pd.cut(tr["hold_days"], bins=bins, labels=lab).value_counts().reindex(lab)
        out = dist.rename("n_trades").reset_index().rename(columns={"index": "hold_bin"})
        out["pct"] = (out["n_trades"] / out["n_trades"].sum()).round(3)
        out.to_csv(os.path.join(OUT_DIR, "p3_cost_dist.csv"), index=False, encoding="utf-8-sig")
    print("完成 → output/p3/p3_summary.csv, p3_curves.csv, p3_trades.csv, p3_cost_dist.csv")


if __name__ == "__main__":
    main()
