# -*- coding: utf-8 -*-
"""V5 预登记 #16/#13（M1.3/D0.4）：统一执行模拟器 sim_core

从 backtest_local.simulate 逐行移植，新增参数化的执行制度：
  exec_delay_days: 成交时滞（交易日）。0 = 旧口径（决策日收盘成交，T+0@close）；
                   1 = 预登记口径（决策日 D 收盘出信号，D 后第 1 个交易日净值成交，T+1）。
                   对 买/卖/再平衡/CPPI 降仓/移动止损 全部交易一致生效。
  cost_in_fn(amount_yuan) -> rate   申购费率函数；None = 旧平铺 args.cost_in
  cost_out_fn(gross_yuan, hold_days) -> rate  赎回阶梯费率；None = 旧平铺 args.cost_out
  qdii_delay_days / qdii_codes: QDII 确认时滞（预登记 D0.4 固定假设 T+2）
  stale_block_days: 最新净值距当日 >N 自然日即视为不可交易（买入与信号卖出被挡；
                   持仓继续以最后复权净值盯市；trail/CPPI 队列按日重试天然等待；
                   期末清算豁免——清盘按清算净值分配）。

严格不变量：exec_delay_days=0 且费率函数为 None 且扩展参数缺省时，逐日净值必须与
backtest_local.simulate 完全一致（M1.3 对拍测试 max|Δ|=0，位级一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (STRAT_CASH_YIELD, STRAT_REBALANCE, STRAT_TRAIL_STOP,
                    STRAT_CRISIS_MA, STRAT_CRISIS_VOL_WINDOW, STRAT_CRISIS_VOL_Q,
                    STRAT_CPPI, STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1,
                    STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2, STRAT_CPPI_DD3,
                    STRAT_CPPI_SLOTS3)
from backtest_local import px, realized_vol_regime, is_quarter_end_signal


def simulate(panel, navs, bench, dates, args,
             exec_delay_days: int = 1,
             cost_in_fn=None, cost_out_fn=None, label: str = "sim_core",
             qdii_delay_days: int | None = None, qdii_codes: set | None = None,
             stale_block_days: int | None = None):
    by_date = {d: g.set_index("code") for d, g in panel.groupby("date")}
    eligible_by_date = {}
    if getattr(args, "pool_mode", "default") == "pit-top":
        pit_top_n = getattr(args, "pit_top_n", 100)
        for d, g in by_date.items():
            eligible_by_date[d] = set(g.nlargest(pit_top_n, "S").index)

    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    day_grid = bench.loc[T0:T1].index
    day_pos = {d: i for i, d in enumerate(day_grid)}

    def exec_day_of(day, dly=None):
        d = exec_delay_days if dly is None else dly
        return day_grid[min(day_pos[day] + d, len(day_grid) - 1)]

    def last_px_date(code, day):
        s = navs.get(code)
        if s is None:
            return None
        idx = s.index.searchsorted(day, side="right") - 1
        return s.index[idx] if idx >= 0 else None

    def tradable(code, day):
        if stale_block_days is None:
            return True
        ld = last_px_date(code, day)
        return ld is not None and (pd.Timestamp(day) - pd.Timestamp(ld)).days <= stale_block_days

    def delay_of(code):
        if qdii_delay_days is not None and qdii_codes and code in qdii_codes:
            return qdii_delay_days
        return exec_delay_days

    use_v38 = not getattr(args, "legacy", False)
    cash_yield = float(getattr(args, "cash_yield", STRAT_CASH_YIELD) or 0.0) if use_v38 else 0.0
    trail_stop = float(getattr(args, "trail_stop", STRAT_TRAIL_STOP) or 0.0) if use_v38 else 0.0
    rebalance_freq = getattr(args, "rebalance", STRAT_REBALANCE) if use_v38 else "none"
    crisis_on = bool(getattr(args, "crisis", True)) if use_v38 else False
    cppi_on = bool(getattr(args, "cppi", STRAT_CPPI)) if use_v38 else False

    ma = bench.rolling(int(getattr(args, "crisis_ma", STRAT_CRISIS_MA))).mean() if crisis_on else None
    vol, vol_th = realized_vol_regime(
        bench,
        window=int(getattr(args, "crisis_vol_window", STRAT_CRISIS_VOL_WINDOW)),
        q=float(getattr(args, "crisis_vol_q", STRAT_CRISIS_VOL_Q)),
    ) if crisis_on else (None, None)

    cin_rate = (lambda amt: args.cost_in) if cost_in_fn is None else cost_in_fn
    cout_rate = (lambda gross, hd: args.cost_out) if cost_out_fn is None else cost_out_fn

    cash, positions, trades, curve = float(args.capital), {}, [], []
    last_scores = None
    total_cost, cash_interest_total, rebal_turnover = 0.0, 0.0, 0.0
    prev_day = None
    crisis_days = crisis_blocked = 0
    cppi_hwm, cppi_locked = float(args.capital), False
    cppi_sells = cppi_hard_stops = cppi_unlocks = 0

    queue = []   # [(exec_day, seq, action)]；按追加顺序回放

    def pos_value(code, day):
        return positions[code]["units"] * px(navs[code], day)

    def equity(day):
        return cash + sum(pos_value(c, day) for c in positions)

    def sell(code, day, reason, exit_s=np.nan, price=None):
        nonlocal cash, total_cost
        p = positions.pop(code)
        price = price if price is not None else px(navs[code], day)
        if pd.isna(price):
            positions[code] = p
            return False
        gross = p["units"] * price
        hd = (pd.Timestamp(day) - pd.Timestamp(p["entry_date"])).days
        fee = gross * cout_rate(gross, hd)
        net_out = gross - fee
        cash += net_out
        total_cost += fee
        ret = net_out / p["alloc"] - 1
        trades.append(dict(
            code=code, entry_date=p["entry_date"], exit_date=str(pd.Timestamp(day).date()),
            entry_px=round(p["entry_px"], 4), exit_px=round(price, 4),
            entry_S=round(p["entry_S"], 1), exit_S=round(exit_s, 1) if pd.notna(exit_s) else np.nan,
            exit_reason=reason,
            hold_days=hd,
            alloc_yuan=round(p["alloc"], 2), net_ret=round(ret, 4),
            pnl_yuan=round(net_out - p["alloc"], 2)))
        return True

    def trim_to_cap(day, cap_slots, scores, reason):
        nonlocal cash, total_cost, rebal_turnover, cppi_sells
        sold = 0
        cap_slots = max(0, int(cap_slots))
        if cap_slots == 0:
            for c in list(positions):
                exit_s = scores.loc[c, "S"] if scores is not None and c in scores.index else np.nan
                sold += int(sell(c, day, reason, exit_s=exit_s))
            cppi_sells += sold
            return sold
        if len(positions) > cap_slots:
            order = sorted(positions,
                           key=lambda c: (scores.loc[c, "S"] if scores is not None and c in scores.index else -1),
                           reverse=True)
            for c in order[cap_slots:]:
                exit_s = scores.loc[c, "S"] if scores is not None and c in scores.index else np.nan
                sold += int(sell(c, day, reason, exit_s=exit_s))
        eq = equity(day)
        target = eq / args.slots if eq > 0 else 0
        for c in list(positions):
            price = px(navs[c], day)
            if pd.isna(price) or target <= 0:
                continue
            val = positions[c]["units"] * price
            if val <= target * 1.01:
                continue
            sell_units = (val - target) / price
            if sell_units <= 0 or sell_units >= positions[c]["units"]:
                continue
            ratio = sell_units / positions[c]["units"]
            positions[c]["units"] -= sell_units
            positions[c]["alloc"] *= (1 - ratio)
            gross = sell_units * price
            hd = (pd.Timestamp(day) - pd.Timestamp(positions[c]["entry_date"])).days
            fee = gross * cout_rate(gross, hd)
            cash += gross - fee
            total_cost += fee
            rebal_turnover += gross
        cppi_sells += sold
        return sold

    def rebalance_to_10slots(day):
        nonlocal cash, total_cost, rebal_turnover
        if not positions:
            return
        eq = equity(day)
        target = eq / args.slots
        for c in list(positions):
            price = px(navs[c], day)
            if pd.isna(price):
                continue
            val = positions[c]["units"] * price
            if val <= target * 1.01:
                continue
            sell_units = (val - target) / price
            if sell_units <= 0 or sell_units >= positions[c]["units"]:
                continue
            ratio = sell_units / positions[c]["units"]
            positions[c]["units"] -= sell_units
            positions[c]["alloc"] *= (1 - ratio)
            gross = sell_units * price
            hd = (pd.Timestamp(day) - pd.Timestamp(positions[c]["entry_date"])).days
            fee = gross * cout_rate(gross, hd)
            cash += gross - fee
            total_cost += fee
            rebal_turnover += gross
        for c in list(positions):
            if cash <= 0:
                break
            price = px(navs[c], day)
            if pd.isna(price):
                continue
            val = positions[c]["units"] * price
            if val >= target * 0.99:
                continue
            alloc = min(cash, (target - val) * (1 + args.cost_in))
            if alloc <= 0:
                continue
            positions[c]["units"] += alloc / (price * (1 + cin_rate(alloc)))
            positions[c]["alloc"] += alloc
            cash -= alloc
            total_cost += alloc * cin_rate(alloc) / (1 + cin_rate(alloc))
            rebal_turnover += alloc

    def buy_wave(day, cand_rows, daily_cap, eligible):
        """cand_rows: list[(code, S)] 按 S 降序；在成交日用当日价格撮合。"""
        nonlocal cash, total_cost
        for c, s in cand_rows:
            if eligible is not None and c not in eligible:
                continue
            if c in positions or len(positions) >= min(args.slots, daily_cap) or c not in navs:
                continue
            if not tradable(c, day):
                continue
            price = px(navs[c], day)
            if pd.isna(price):
                continue
            eq_now = equity(day)
            alloc = min(cash, eq_now / args.slots)
            if alloc < eq_now * 0.02:
                continue
            positions[c] = dict(units=alloc / (price * (1 + cin_rate(alloc))),
                                entry_px=price, entry_date=str(pd.Timestamp(day).date()),
                                entry_S=s, alloc=alloc, peak=price)
            cash -= alloc
            total_cost += alloc * cin_rate(alloc) / (1 + cin_rate(alloc))

    def flush(day):
        """执行所有 exec_day <= day 的队列动作（按决策先后保序）。"""
        due = [a for a in queue if a[0] <= day]
        if not due:
            return
        queue[:] = [a for a in queue if a[0] > day]
        for _, _, act in due:
            kind = act[0]
            if kind == "sell":
                _, code, reason, exit_s = act
                if code in positions and tradable(code, day):
                    sell(code, day, reason, exit_s=exit_s)
            elif kind == "trim":
                _, cap_slots, scores, reason = act
                trim_to_cap(day, cap_slots, scores, reason)
            elif kind == "rebalance":
                rebalance_to_10slots(day)
            elif kind == "buy_wave":
                _, cand_rows, daily_cap, eligible = act
                buy_wave(day, cand_rows, daily_cap, eligible)

    seq = 0

    def q(day_decided, action):
        nonlocal seq
        dly = delay_of(action[1]) if len(action) > 1 and isinstance(action[1], str) else None
        queue.append((exec_day_of(day_decided, dly), seq, action))
        seq += 1

    def is_crisis(day):
        if not crisis_on:
            return False
        if day not in bench.index:
            return False
        return (pd.notna(ma.asof(day)) and pd.notna(vol.asof(day)) and pd.notna(vol_th.asof(day))
                and bench.asof(day) < ma.asof(day) and vol.asof(day) > vol_th.asof(day))

    for day in day_grid:
        if prev_day is not None and cash_yield > 0:
            dt = max(0, (pd.Timestamp(day) - pd.Timestamp(prev_day)).days)
            if dt:
                interest = cash * cash_yield * dt / 365.25
                cash += interest
                cash_interest_total += interest
        prev_day = day

        flush(day)   # 先消化早前决策、今日到期的成交

        # 单基金 trailing stop（决策于当日，成交/exec 由 exec_delay 决定）
        if trail_stop > 0:
            for c in list(positions):
                price = px(navs[c], day)
                if pd.notna(price):
                    positions[c]["peak"] = max(positions[c].get("peak", positions[c]["entry_px"]), price)
                    if price < positions[c]["peak"] * (1 - trail_stop):
                        q(day, ("sell", c, f"trail_{trail_stop:.0%}", np.nan))
            flush(day)

        crisis_active = is_crisis(day)
        if crisis_active:
            crisis_days += 1

        daily_cap = args.slots
        if cppi_on:
            eq_now = equity(day)
            if (not cppi_locked) and eq_now > cppi_hwm:
                cppi_hwm = eq_now
            dd = eq_now / cppi_hwm - 1 if cppi_hwm > 0 else 0
            if cppi_locked:
                daily_cap = 0
            elif dd <= STRAT_CPPI_DD3:
                daily_cap = STRAT_CPPI_SLOTS3
                cppi_locked = True
                cppi_hard_stops += 1
            elif dd <= STRAT_CPPI_DD2:
                daily_cap = STRAT_CPPI_SLOTS2
            elif dd <= STRAT_CPPI_DD1:
                daily_cap = STRAT_CPPI_SLOTS1
            if daily_cap < len(positions):
                q(day, ("trim", daily_cap, last_scores, f"cppi_cap{daily_cap}"))
                flush(day)

        day_str = str(day.date())
        if day_str in by_date:
            g = by_date[day_str]
            last_scores = g
            if getattr(args, "pool_mode", "default") == "pit-top":
                eligible = eligible_by_date.get(day_str, set())
                candidates = g[(g.S > args.buy) & (g.index.isin(eligible))].sort_values("S", ascending=False)
            else:
                eligible = None
                candidates = g[(g.S > args.buy)].sort_values("S", ascending=False)

            if cppi_on and cppi_locked:
                if len(candidates) and not crisis_active:
                    cppi_locked = False
                    cppi_hwm = equity(day)
                    daily_cap = args.slots
                    cppi_unlocks += 1
                else:
                    daily_cap = 0

            for c in list(positions):
                if c not in g.index:
                    continue
                score = g.loc[c, "S"]
                if pd.notna(score) and score < args.sell:
                    q(day, ("sell", c, f"S<{args.sell:.0f}", score))
            flush(day)

            if cppi_on and daily_cap < len(positions):
                q(day, ("trim", daily_cap, g, f"cppi_cap{daily_cap}"))
                flush(day)

            if use_v38 and rebalance_freq == "quarterly" and is_quarter_end_signal(day_str):
                q(day, ("rebalance",))
                flush(day)

            if crisis_active:
                crisis_blocked += len(candidates)
            elif daily_cap > 0:
                cand_rows = [(c, r.S) for c, r in candidates.iterrows()]
                q(day, ("buy_wave", cand_rows, daily_cap, eligible))
                flush(day)

        eq = equity(day)
        curve.append((day, eq, len(positions), cash, cash / eq if eq else np.nan,
                      crisis_active, cppi_hwm, cppi_locked))

    for c in list(positions):
        sell(c, T1, "期末清算")

    ec = pd.DataFrame(curve, columns=["date", "equity", "n_pos", "cash", "cash_ratio",
                                      "crisis", "cppi_hwm", "cppi_locked"]).set_index("date")
    ec["drawdown"] = ec.equity / ec.equity.cummax() - 1
    ec.attrs.update(dict(model=label, exec_delay_days=exec_delay_days,
                         qdii_delay_days=qdii_delay_days, stale_block_days=stale_block_days,
                         cash_interest=cash_interest_total,
                         total_cost=total_cost, rebal_turnover=rebal_turnover,
                         crisis_days=crisis_days, crisis_blocked=crisis_blocked,
                         cppi_sells=cppi_sells, cppi_hard_stops=cppi_hard_stops,
                         cppi_unlocks=cppi_unlocks))
    return ec, pd.DataFrame(trades)
