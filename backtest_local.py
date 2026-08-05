# -*- coding: utf-8 -*-
"""
本地自助回测 —— 你只管敲命令，它帮你出三本账
=================================================================
用法(在 quant_fund_picker 目录下):
    python backtest_local.py                          # 默认: V3.8 最优执行层, 70买45卖
    python backtest_local.py --legacy                 # 旧版纯S阈值回测
    python backtest_local.py --sell 50                # 改动一条线对比
    python backtest_local.py --capital 200000 --slots 8
    python backtest_local.py --codes 自选.txt         # 对自定义基金池回测(每行一个6位代码)
    python backtest_local.py --rebuild                # 强制重打新季度的分(换模型后要这个)

产出(output/ 下, tag 为参数签名):
    bt_trades_<tag>.csv   逐笔买卖台账: 买卖点/价/分/持有天数/净收益/年化/盈亏元
    bt_daily_<tag>.csv    逐日净值+回撤+持仓数
    bt_summary_<tag>.md   汇总报告: 总收益/CAGR/回撤/胜率/盈亏比/分年表/前5后5
    bt_equity_<tag>.png   净值与回撤图(vs 沪深300)

评分面板缓存 output/bt_scores_cache/: 每个季点打一次分永久缓存, 二次运行秒级
严格口径: 默认必须提供历史可投池快照（含清盘基金），每月动态建池；信号 T+2 真实交易日成交；回测豁免无法获得历史档案的任期/AUM 惩罚。
=================================================================
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

import provider
import rbsa, factors
from engine import score_fund, finalize
from pit_universe import PITUniverseStore, PITUniverseError
from config import (STRAT_BUY_TH, STRAT_SELL_TH, STRAT_SLOTS, STRAT_VERSION,
                    STRAT_CASH_YIELD, STRAT_REBALANCE, STRAT_TRAIL_STOP,
                    STRAT_CRISIS_MA, STRAT_CRISIS_VOL_WINDOW, STRAT_CRISIS_VOL_Q,
                    STRAT_CPPI, STRAT_CPPI_DD1, STRAT_CPPI_SLOTS1,
                    STRAT_CPPI_DD2, STRAT_CPPI_SLOTS2, STRAT_CPPI_DD3,
                    STRAT_CPPI_SLOTS3, STRAT_CPPI_HWM_MODE,
                    STRAT_HI_WATER, STRAT_HI_WATER_SLOTS)

CACHE_DIR = "output/bt_scores_cache"


# ============ 1. 打分面板: 每个季点一份(优先复用已有, 缺失则现打现存) ============
def mdd_factor(rm, w):
    if pd.isna(rm) or rm <= 1.2:
        return 1.0
    p = min(0.5 * (rm - 1.2), 1.0)
    if pd.notna(w) and w <= 0.35:
        p *= 0.5
    return 1 - p


def score_from_raw(g):
    """原料行 → V3.7.2 分 (fv旧 + alpha平滑 + mom M1 × 惩罚链), 与引擎同源"""
    g = g.copy()
    g["rank4"] = g["r4"].rank(pct=True)
    g["rank7"] = g["r7"].rank(pct=True)
    out = []
    for _, r in g.iterrows():
        fv = (np.nan if (r.val_cov < 0.5 or pd.isna(r.val_pct))
              else factors.valuation_base_score(r.val_pct, r.trend_ok))
        al = [x for x in [factors.ir_score_smooth(r.wr) if pd.notna(r.wr) else np.nan,
                          factors.dc_score_smooth(r.dc) if pd.notna(r.dc) else np.nan]
              if pd.notna(x)]
        fa = float(np.mean(al)) if al else np.nan
        fm = factors.momentum_score_smooth_m1(r.rank4, r.rank7)
        pen = r.other_pen * mdd_factor(r.R_MDD, r.water)
        num = ((r.wv * min(max(fv, 0), 100)) if pd.notna(fv) else 0) \
            + (r.wa * fa if pd.notna(fa) else 0) + (r.wm * fm if pd.notna(fm) else 0)
        den = (r.wv if pd.notna(fv) else 0) + (r.wa if pd.notna(fa) else 0) \
            + (r.wm if pd.notna(fm) else 0)
        out.append(min(max((num / den if den else 0) * pen, 0), 100))
    g["S"] = out
    return g


def harvest_date(d, universe):
    """仅对该日历史快照的成员打分；pit_meta 阻断今天元数据泄漏。"""
    t0 = time.time()
    rows = []
    meta = universe.set_index("code")[["name", "fund_type"]].to_dict("index")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(score_fund, c, as_of=d, bt=True, pit_meta=meta[c]): c for c in meta}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if not r.get("error"):
                    rows.append(r)
            except Exception:
                pass
    fdf = finalize(rows, as_of=d)
    recs = []
    for h in fdf.dropna(subset=["S_total"]).itertuples():
        pdt = h.penalty_detail or {}
        op = 1.0
        for n, p in (h.penalties or []):
            if "回撤比值" not in n:
                op *= (1 - p)
        recs.append(dict(code=h.code, water=h.water, wv=h.w_value, wa=h.w_alpha,
            wm=h.w_mom, val_pct=h.val_pct, val_cov=h.val_coverage,
            trend_ok=bool(h.trend_ok), wr=h.ir_winrate, dc=h.down_capture,
            r4=h.mom_4m1m, r7=h.mom_7m1m, R_MDD=pdt.get("R_MDD"), other_pen=op))
    df = pd.DataFrame(recs)
    print(f"  [PiT harvest] {d} 快照成员 {len(universe)}，有效评分 {len(df)} ({time.time()-t0:.0f}s)", flush=True)
    return df


def build_panel(dates, universe_store, rebuild=False):
    """严格动态可投池：每个信号日独立取历史快照，不复用旧研究池原料。"""
    root = os.path.join(CACHE_DIR, "strict_" + universe_store.manifest_hash())
    os.makedirs(root, exist_ok=True)
    parts, audits = [], []
    for d in dates:
        universe, audit = universe_store.universe(d)
        ck = f"{root}/{d}.csv"
        if (not rebuild) and os.path.exists(ck):
            g = pd.read_csv(ck, dtype={"code": str})
        else:
            g = harvest_date(d, universe)
            if not g.empty:
                g.to_csv(ck, index=False, encoding="utf-8-sig")
        audit.update(scored_members=int(len(g)), cache_file=ck)
        audits.append(audit)
        if g.empty:
            print(f"  [warn] {d} 没有可用的 PiT 评分，跳过")
            continue
        g = score_from_raw(g)
        g["date"] = d
        parts.append(g[["date", "code", "S", "water", "R_MDD"]])
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), audits)


# ============ 2. 数据 ============
def load_navs(codes):
    navs = {}
    for c in codes:
        try:
            df = provider.get_fund_nav(c)          # DataFrame(date,nav,ret), 带缓存自动刷新
            if df is None or not len(df):
                continue
            s = df.set_index("date")["nav"].sort_index()
            s = s[~s.index.duplicated(keep="last")].dropna()
            if len(s) > 200:
                navs[c] = s
        except Exception:
            pass
    return navs


def load_names():
    try:
        m = provider.get_fund_meta()
        return m["基金简称"].to_dict()
    except Exception:
        return {}


def px(s, dt):
    if s is None:
        return np.nan
    v = s.asof(pd.Timestamp(dt))
    return float(v) if pd.notna(v) else np.nan


def realized_vol_regime(bench, window=20, q=0.80, min_hist=60):
    """返回 (ann_vol, rolling_prior_quantile)。分位阈值只使用当日之前历史，避免未来函数。"""
    ret = bench.pct_change()
    vol = ret.rolling(window).std() * np.sqrt(252)
    vals, th = [], []
    for v in vol.values:
        hist = [x for x in vals if pd.notna(x)]
        th.append(float(np.quantile(hist, q)) if len(hist) >= min_hist else np.nan)
        if pd.notna(v):
            vals.append(float(v))
    return vol, pd.Series(th, index=bench.index)


def is_quarter_end_signal(day_str):
    return pd.Timestamp(day_str).month in (3, 6, 9, 12)


# ============ 3. 逐日模拟(金额记账) ============
def simulate(panel, navs, bench, dates, args):
    """V3.8 Calmar 版逐日模拟。

    默认执行 `config.STRAT_VERSION`：固定10槽、季度再平衡、20%单基移动止损、
    现金2.5%收益、MA200&Vol80危机禁买、组合自身CPPI(-15/-20/-25)。
    用 `--legacy` 可退回旧的 S>buy / S<sell / fixed-slot 逻辑。
    """
    by_date = {d: g.set_index("code") for d, g in panel.groupby("date")}
    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    day_grid = bench.loc[T0:T1].index

    # ===== V3.8 开关 =====
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

    cash, positions, trades, curve = float(args.capital), {}, [], []
    last_scores = None
    total_cost, cash_interest_total, rebal_turnover = 0.0, 0.0, 0.0
    prev_day = None
    crisis_days = crisis_blocked = 0
    cppi_hwm, cppi_locked = float(args.capital), False
    cppi_sells = cppi_hard_stops = cppi_unlocks = 0

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
        fee = gross * args.cost_out
        net_out = gross - fee
        cash += net_out
        total_cost += fee
        ret = net_out / p["alloc"] - 1
        trades.append(dict(
            code=code, entry_date=p["entry_date"], exit_date=str(pd.Timestamp(day).date()),
            entry_px=round(p["entry_px"], 4), exit_px=round(price, 4),
            entry_S=round(p["entry_S"], 1), exit_S=round(exit_s, 1) if pd.notna(exit_s) else np.nan,
            exit_reason=reason,
            hold_days=(pd.Timestamp(day) - pd.Timestamp(p["entry_date"])).days,
            alloc_yuan=round(p["alloc"], 2), net_ret=round(ret, 4),
            pnl_yuan=round(net_out - p["alloc"], 2)))
        return True

    def trim_to_cap(day, cap_slots, scores, reason):
        """只降风险、不补仓：卖弱留强到 cap_slots，并把剩余仓位卖回每槽10%左右。"""
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
            fee = gross * args.cost_out
            cash += gross - fee
            total_cost += fee
            rebal_turnover += gross
        cppi_sells += sold
        return sold

    def rebalance_to_10slots(day):
        """季度等权再平衡：每只目标约等于组合1/10。"""
        nonlocal cash, total_cost, rebal_turnover
        if not positions:
            return
        eq = equity(day)
        target = eq / args.slots
        # 先卖超配
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
            fee = gross * args.cost_out
            cash += gross - fee
            total_cost += fee
            rebal_turnover += gross
        # 再补低配；不会增加新标的数量
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
            positions[c]["units"] += alloc / (price * (1 + args.cost_in))
            positions[c]["alloc"] += alloc
            cash -= alloc
            total_cost += alloc * args.cost_in / (1 + args.cost_in)
            rebal_turnover += alloc

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

        # 单基金 20% trailing stop
        if trail_stop > 0:
            for c in list(positions):
                price = px(navs[c], day)
                if pd.notna(price):
                    positions[c]["peak"] = max(positions[c].get("peak", positions[c]["entry_px"]), price)
                    if price < positions[c]["peak"] * (1 - trail_stop):
                        sell(c, day, f"trail_{trail_stop:.0%}", price=price)

        crisis_active = is_crisis(day)
        if crisis_active:
            crisis_days += 1

        # 组合级 CPPI：自身净值回撤预算
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
                trim_to_cap(day, daily_cap, last_scores, f"cppi_cap{daily_cap}")

        day_str = str(day.date())
        if day_str in by_date:
            g = by_date[day_str]
            last_scores = g
            candidates = g[g.S > args.buy].sort_values("S", ascending=False)

            # 熔断后：仅在右侧信号出现且非危机时复活，并重置HWM为当前净值
            if cppi_on and cppi_locked:
                if len(candidates) and not crisis_active:
                    cppi_locked = False
                    cppi_hwm = equity(day)
                    daily_cap = args.slots
                    cppi_unlocks += 1
                else:
                    daily_cap = 0

            # 卖出：跌破 S 阈值
            for c in list(positions):
                if c not in g.index:
                    continue
                score = g.loc[c, "S"]
                if pd.notna(score) and score < args.sell:
                    sell(c, day, f"S<{args.sell:.0f}", exit_s=score)

            if cppi_on and daily_cap < len(positions):
                trim_to_cap(day, daily_cap, g, f"cppi_cap{daily_cap}")

            # 季度再平衡：先卖后买之后的风险预算仍按10槽目标权重执行
            if use_v38 and rebalance_freq == "quarterly" and is_quarter_end_signal(day_str):
                rebalance_to_10slots(day)

            # 危机期禁买；旧MA200单因子过滤不再使用
            if crisis_active:
                crisis_blocked += len(candidates)
            elif daily_cap > 0:
                for c, row in candidates.iterrows():
                    if c in positions or len(positions) >= min(args.slots, daily_cap) or c not in navs:
                        continue
                    price = px(navs[c], day)
                    if pd.isna(price):
                        continue
                    eq_now = equity(day)
                    alloc = min(cash, eq_now / args.slots)  # 最新最优模型保留固定槽位，不强行填满空槽
                    if alloc < eq_now * 0.02:
                        continue
                    positions[c] = dict(units=alloc / (price * (1 + args.cost_in)),
                                        entry_px=price, entry_date=day_str,
                                        entry_S=row.S, alloc=alloc, peak=price)
                    cash -= alloc
                    total_cost += alloc * args.cost_in / (1 + args.cost_in)

        eq = equity(day)
        curve.append((day, eq, len(positions), cash, cash / eq if eq else np.nan,
                      crisis_active, cppi_hwm, cppi_locked))

    # 期末清仓
    for c in list(positions):
        sell(c, T1, "期末清算")

    ec = pd.DataFrame(curve, columns=["date", "equity", "n_pos", "cash", "cash_ratio",
                                      "crisis", "cppi_hwm", "cppi_locked"]).set_index("date")
    ec["drawdown"] = ec.equity / ec.equity.cummax() - 1
    ec.attrs.update(dict(model=STRAT_VERSION if use_v38 else "legacy", cash_interest=cash_interest_total,
                         total_cost=total_cost, rebal_turnover=rebal_turnover,
                         crisis_days=crisis_days, crisis_blocked=crisis_blocked,
                         cppi_sells=cppi_sells, cppi_hard_stops=cppi_hard_stops,
                         cppi_unlocks=cppi_unlocks))
    return ec, pd.DataFrame(trades)


# ============ 4. 汇总与报告 ============
def report(ec, tr, bench, args, dates, names, tag):
    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    yrs = (T1 - T0).days / 365.25
    tot = ec.equity.iloc[-1] / args.capital - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    dd = ec.drawdown.min()
    bb = bench.loc[T0:T1]; btot = bb.iloc[-1] / bb.iloc[0] - 1
    win = (tr.net_ret > 0).mean() if len(tr) else np.nan
    aw = tr.loc[tr.net_ret > 0, "net_ret"].mean() if (tr.net_ret > 0).any() else np.nan
    al = tr.loc[tr.net_ret <= 0, "net_ret"].mean() if (tr.net_ret <= 0).any() else np.nan
    invested = (ec.n_pos > 0).mean()

    S = []
    model_name = ec.attrs.get("model", "legacy")
    S.append(f"# 本地回测报告  区间 {dates[0]} → {dates[-1]}  tag={tag}\n")
    S.append(f"模型: **{model_name}**\n")
    v38_on = not getattr(args, 'legacy', False)
    S.append(f"参数: 买 S>{args.buy:.0f} / 卖 S<{args.sell:.0f} / 槽位 {args.slots} / 本金 {args.capital:,.0f} 元 / "
             f"成本 申购{args.cost_in:.2%}+赎回{args.cost_out:.2%} / "
             f"现金年化{(getattr(args, 'cash_yield', STRAT_CASH_YIELD) if v38_on else 0):.2%} / "
             f"20%止损 {'开' if v38_on else '关'} / "
             f"CPPI {'开' if v38_on else '关'}\\n")
    S.append("## 总账")
    S.append(f"| 指标 | 数值 |\n|---|---|")
    S.append(f"| 期末资产 | **{ec.equity.iloc[-1]:,.0f} 元** |")
    S.append(f"| 组合总收益 | **{tot:+.1%}** ({args.capital * tot:+,.0f} 元) |")
    S.append(f"| 年化 CAGR | {cagr:+.2%} |")
    S.append(f"| 最大回撤(日频) | {dd:.1%} |")
    S.append(f"| Calmar | {cagr / abs(dd):.2f} |")
    S.append(f"| 沪深300 同期 | {btot:+.1%} (超额 {tot - btot:+.1%}) |")
    S.append(f"| 有持仓天数占比 | {invested:.0%} |")
    S.append(f"| 平均现金占比 | {ec.cash_ratio.mean():.1%} |" if "cash_ratio" in ec else "| 平均现金占比 | — |")
    S.append(f"| 现金收益累计 | {ec.attrs.get('cash_interest', 0):,.0f} 元 |")
    S.append(f"| 危机日/禁买信号 | {ec.attrs.get('crisis_days', 0)} / {ec.attrs.get('crisis_blocked', 0)} |")
    S.append(f"| CPPI卖出/熔断/重启 | {ec.attrs.get('cppi_sells', 0)} / {ec.attrs.get('cppi_hard_stops', 0)} / {ec.attrs.get('cppi_unlocks', 0)} |\n")
    if len(tr):
        S.append("## 交易账")
        S.append(f"| 指标 | 数值 |\n|---|---|")
        S.append(f"| 交易笔数 | {len(tr)} |")
        S.append(f"| 胜率 | **{win:.1%}** ({(tr.net_ret > 0).sum()}胜{(tr.net_ret <= 0).sum()}负) |")
        S.append(f"| 平均净收益/笔 | {tr.net_ret.mean():+.1%} |")
        pf = f"{abs(aw / al):.2f}" if pd.notna(al) and al != 0 else "—(无亏损笔)"
        S.append(f"| 盈亏比 | {pf} (均盈{aw:+.1%} / 均亏{al if pd.notna(al) else 0:+.1%}) |")
        S.append(f"| 平均持有 | {tr.hold_days.mean():.0f} 天 |")
        S.append(f"| 累计盈亏 | {tr.pnl_yuan.sum():+,.0f} 元 |")
        # 交易成本以模拟器逐笔/再平衡实际扣除为准；旧账本缺少再平衡成本时才回退估算。
        cost_est = ec.attrs.get("total_cost")
        if cost_est is None:
            cost_in_total = (tr["alloc_yuan"] * args.cost_in).sum() if "alloc_yuan" in tr.columns else 0
            cost_out_total = ((tr["alloc_yuan"] + tr["pnl_yuan"]) * args.cost_out).sum() if "alloc_yuan" in tr.columns else 0
            cost_est = cost_in_total + cost_out_total
        S.append(f"| 累计交易成本 | {cost_est:,.0f} 元 |\n")
        S.append("### 退出原因")
        ### 退出原因
        if len(tr) > 0 and "exit_reason" in tr.columns:
            S.append(tr.groupby("exit_reason").agg(n=("net_ret", "size"),
                                                   均净收益=("net_ret", "mean"),
                                                   胜率=("net_ret", lambda x: (x > 0).mean())
                                                   ).round(3).to_markdown() + "\n")
        else:
            S.append("（无交易记录）\n")
        S.append("### 按入场年")
        tr2 = tr.copy(); tr2["yr"] = pd.to_datetime(tr2.entry_date).dt.year
        S.append(tr2.groupby("yr").agg(n=("net_ret", "size"),
              均净收益=("net_ret", "mean"), 胜率=("net_ret", lambda x: (x > 0).mean())
              ).round(3).to_markdown() + "\n")
        S.append("### 最佳 5 笔")
        S.append(tr.nlargest(5, "net_ret")[["code", "entry_date", "exit_date", "entry_S",
              "net_ret", "pnl_yuan", "hold_days"]].round(3).to_markdown(index=False) + "\n")
        S.append("### 最差 5 笔")
        S.append(tr.nsmallest(5, "net_ret")[["code", "entry_date", "exit_date", "entry_S",
              "net_ret", "pnl_yuan", "hold_days"]].round(3).to_markdown(index=False) + "\n")
    S.append("> 严格 PiT 口径：每月只使用当时已知的历史可投池快照，信号日后延迟成交；快照审计见 `bt_pit_audit.json`。历史经理任期/AUM未有逐日档案，评分中已豁免，不能解释为该两项通过验证。仅量化研究用途，不构成投资建议。\n")
    txt = "\n".join(S)
    open(f"output/bt_summary_{tag}.md", "w", encoding="utf-8").write(txt)
    # 控制台精简版
    print(f"\n========== 回测结果 {dates[0]} → {dates[-1]} ==========")
    print(f"期末资产 {ec.equity.iloc[-1]:,.0f} 元 | 总收益 {tot:+.1%} | CAGR {cagr:+.2%} | "
          f"最大回撤 {dd:.1%} | 基准 {btot:+.1%}")
    if len(tr):
        print(f"交易 {len(tr)} 笔 | 胜率 {win:.0%} | 均净收益 {tr.net_ret.mean():+.1%} | "
              f"盈亏比 {'—(无亏损笔)' if pd.isna(al) or al == 0 else f'{abs(aw / al):.2f}'} | "
              f"平均持有 {tr.hold_days.mean():.0f} 天")
    return txt


def chart(ec, bench, args, dates, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP", "DejaVu Sans"]
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    bb = bench.loc[T0:T1]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax[0].plot(ec.index, ec.equity / args.capital, color="#f5b83d", lw=2.2, label="策略")
    ax[0].plot(bb.index, bb / bb.iloc[0], color="#888", lw=1.3, ls="--", label="沪深300")
    ax[0].set_title(f"本地回测 {dates[0]}→{dates[-1]}  买S>{args.buy:.0f} 卖S<{args.sell:.0f} "
                    f"{args.slots}槽 本金{args.capital:,.0f}元")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)
    ax[1].fill_between(ec.index, ec.drawdown, 0, color="#f5b83d", alpha=0.4, label="策略回撤")
    ax[1].fill_between(bb.index, bb / bb.cummax() - 1, 0, color="#888", alpha=0.3, label="沪深300回撤")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"output/bt_equity_{tag}.png", dpi=130)
    print(f"[图] output/bt_equity_{tag}.png")


# ============ 5. 入口 ============
def main():
    ap = argparse.ArgumentParser(description="月度打分 + 日度再平衡回测")
    ap.add_argument("--start", default="2006-03-31", help="起始日期")
    ap.add_argument("--end", default=None, help="结束日期")
    ap.add_argument("--buy", type=float, default=70.0)
    ap.add_argument("--sell", type=float, default=45.0)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--cost-in", dest="cost_in", type=float, default=0.0015)
    ap.add_argument("--cost-out", dest="cost_out", type=float, default=0.005)
    ap.add_argument("--codes", default=None, help="严格模式禁用：请把历史成分写入 --universe-dir 快照")
    ap.add_argument("--universe-dir", default="data/pit_universe",
                    help="历史可投池快照目录（严格模式必需，含清盘基金）")
    ap.add_argument("--execution-lag", type=int, default=2,
                    help="信号日后延迟的基准交易日数；默认2：T+1披露净值后按下一日净值成交")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="关闭V3.8执行层，回到旧版纯S阈值回测")
    ap.add_argument("--cash-yield", type=float, default=STRAT_CASH_YIELD, help="V3.8闲置现金年化收益")
    ap.add_argument("--trail-stop", type=float, default=STRAT_TRAIL_STOP, help="V3.8单基移动止损阈值")
    ap.add_argument("--rebalance", choices=["none", "quarterly"], default=STRAT_REBALANCE)
    ap.add_argument("--no-crisis", dest="crisis", action="store_false", help="关闭MA+Vol危机禁买过滤")
    ap.set_defaults(crisis=True)
    ap.add_argument("--no-cppi", dest="cppi", action="store_false", help="关闭组合级CPPI风险预算")
    ap.set_defaults(cppi=STRAT_CPPI)
    ap.add_argument("--crisis-ma", type=int, default=STRAT_CRISIS_MA)
    ap.add_argument("--crisis-vol-window", type=int, default=STRAT_CRISIS_VOL_WINDOW)
    ap.add_argument("--crisis-vol-q", type=float, default=STRAT_CRISIS_VOL_Q)
    args = ap.parse_args()

    if args.codes:
        ap.error("--codes 是静态自选池，会破坏动态可投池；严格 PiT 回测请提供历史快照。")
    if args.execution_lag < 1:
        ap.error("--execution-lag 必须至少为 1，禁止信号日同价成交。")
    try:
        universe_store = PITUniverseStore(args.universe_dir)
    except PITUniverseError as exc:
        ap.error(str(exc))

    # 每月末仅使用截至该日已知的信息打分；交易在净值披露后的后续交易日执行。
    end = args.end or str((pd.Timestamp.today() - pd.offsets.MonthEnd(1)).date())
    signal_dates = [str(d.date()) for d in pd.date_range(args.start, end, freq="ME")]
    if not signal_dates:
        ap.error("回测区间内没有月末信号日。")
    print(f"[严格 PiT] 信号 {signal_dates[0]} → {signal_dates[-1]}，共 {len(signal_dates)} 个；"
          f"快照指纹 {universe_store.manifest_hash()}")

    panel, audits = build_panel(signal_dates, universe_store, args.rebuild)
    if panel.empty:
        print("❌ 没有任何可用 PiT 评分月份，程序退出")
        return
    bench = provider.get_close_by_src("sina", "sh000300").dropna().sort_index()
    # 防止月末节假日和同日成交：每个信号映射到至少 T+1，默认 T+2 的真实基准交易日。
    trade_map = {}
    for d in sorted(panel.date.unique()):
        pos = bench.index.searchsorted(pd.Timestamp(d), side="right") + args.execution_lag - 1
        if pos < len(bench.index):
            trade_map[d] = str(bench.index[pos].date())
    panel["signal_date"] = panel["date"]
    panel["date"] = panel["date"].map(trade_map)
    panel = panel.dropna(subset=["date"])
    if panel.empty:
        print("❌ 信号日之后没有可用于成交的基准交易日，程序退出")
        return
    pd.DataFrame(audits).to_csv("output/bt_pit_universe_audit.csv", index=False, encoding="utf-8-sig")
    import json
    with open("output/bt_pit_audit.json", "w", encoding="utf-8") as fh:
        json.dump({"mode": "strict_point_in_time_dynamic_universe", "execution_lag_sessions": args.execution_lag,
                   "manifest": universe_store.manifest_hash(), "snapshots": audits}, fh,
                  ensure_ascii=False, indent=2)
    print(f"[面板] 严格动态评分行 {len(panel)} | S>{args.buy:.0f} 信号 {int((panel.S > args.buy).sum())} 个")

    qs = sorted(panel["date"].unique())
    print(f"[有效执行月] {len(qs)} 个（信号日后 T+{args.execution_lag} 交易日成交）")
    navs = load_navs(list(set(panel.code)))
    print(f"[净值] 可用 {len(navs)}/{panel.code.nunique()}")

    model_tag = "legacy" if args.legacy else "v38"
    tag = f"pit_{model_tag}_b{args.buy:.0f}s{args.sell:.0f}n{args.slots}k{int(args.capital/10000)}w"

    ec, tr = simulate(panel, navs, bench, qs, args)
    names = load_names()
    if len(tr):
        tr.insert(1, "name", tr.code.map(lambda c: names.get(c, "")))
        tr = tr.sort_values(["entry_date", "code"])

    tr.to_csv(f"output/bt_trades_{tag}.csv", index=False, encoding="utf-8-sig")
    ec.reset_index().to_csv(f"output/bt_daily_{tag}.csv", index=False, encoding="utf-8-sig")
    report(ec, tr, bench, args, qs, names, tag)
    chart(ec, bench, args, qs, tag)
    print(f"[输出] output/bt_trades_{tag}.csv | bt_daily_{tag}.csv | bt_summary_{tag}.md")

if __name__ == "__main__":
    main()
