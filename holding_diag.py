# -*- coding: utf-8 -*-
"""
持仓诊断：单基入场高点回撤止损 + 组合级 CPPI 动态槽位 + 回补提示
================================================================
输入（支付宝/天天基金持仓页最容易获得的真实数据）：
  - 基金代码
  - 当前市值（持有金额）
  - 买入日期（交易记录） 或 持仓成本金额 或 持有收益率%
输出（全部由真实净值历史自动计算，天天基金复权日收益）：
  1) 单基金：自入场日起最高复权净值 → 当前回撤
     - 回撤 > STRAT_TRAIL_STOP(20%) → 清仓提示（移动止损）
     - 回撤 > 15% → 接近止损提示
  2) 组合级 CPPI：重建组合净值曲线（成本×复权净值折算 + 现金）
     - HWM（历史最高水位）→ 当前回撤 → 触发档位
     - 档位规则来自 config.STRAT_CPPI_DD1/2/3: -15%→6槽 / -20%→3槽 / -25%→清仓
  3) 回补提示：带滞回带的回升线（触发线 + hysteresis）与创出新高
     - 回撤回升至 -13% → 恢复 10 槽；-18% → 6 槽；-23% → 3 槽；创新高 → 满槽
     并按日序模拟状态机，给出历史上"何时触发/何时回补"的真实事件与日期。
"""

import numpy as np
import pandas as pd


def adj_series(nav_df):
    """复权净值序列：由天天基金官方日增长率(复权)累乘，date 升序"""
    df = nav_df.copy()
    if "date" not in df or "ret" not in df or len(df) == 0:
        return pd.Series(dtype=float, name="adj")
    df = df.dropna(subset=["ret"]).sort_values("date")
    adj = (1 + df["ret"].astype(float).fillna(0)).cumprod()
    return pd.Series(adj.values, index=pd.DatetimeIndex(df["date"]), name="adj")


def parse_user_inputs(rec):
    """解析单只持仓的用户输入 → dict(amount市值, buy_date, cost, ret_pct)"""
    out = dict(amount=None, buy_date=None, cost=None, ret_pct=None)
    try:
        v = rec.get("amount")
        if v is not None:
            out["amount"] = float(v) if np.isfinite(float(v)) else None
    except Exception:
        pass
    d = rec.get("buy_date") or rec.get("date") or rec.get("entry_date") or rec.get("buyDate")
    if d:
        try:
            out["buy_date"] = pd.Timestamp(str(d)).date()
        except Exception:
            out["buy_date"] = None
    c = rec.get("cost")
    if c is not None:
        try:
            c = float(c)
            out["cost"] = c if np.isfinite(c) and c > 0 else None
        except Exception:
            out["cost"] = None
    r = rec.get("ret_pct")
    if r is None:
        r = rec.get("return_pct") or rec.get("ret")
    if r is not None:
        try:
            r = float(r)
            out["ret_pct"] = r if np.isfinite(r) else None
        except Exception:
            out["ret_pct"] = None
    # 市值 / (1 + 收益率) = 成本（支付宝"持有收益率"口径）
    if out["cost"] is None and out["ret_pct"] is not None and out["amount"]:
        cost = out["amount"] / (1.0 + out["ret_pct"] / 100.0)
        if cost > 0:
            out["cost"] = cost
    return out


def infer_entry_date(adj, ret_pct, tol_base=0.03):
    """由持有收益率反推买入日。

    复权净值比 adj_now/adj_d - 1 应 ≈ 持有收益率。找误差最小的日期；
    误差容忍 max(3pp, 3×最小误差+0.5pp)，多个近优候选取**最晚**（最近期
    入场，对止损提示最不易误报；如有多笔/定投，近似为最近一笔）。
    返回 (date, err)；无匹配返回 (None, None)。
    """
    if ret_pct is None or adj is None or len(adj) < 10:
        return None, None
    try:
        r = float(ret_pct) / 100.0
    except Exception:
        return None, None
    now = float(adj.iloc[-1])
    if now <= 0:
        return None, None
    err = now / adj.values - (1.0 + r)
    abs_err = np.abs(err)
    i_min = int(np.nanargmin(abs_err))
    min_err = float(abs_err[i_min])
    if min_err > tol_base:
        return None, None
    # 候选 = 误差 ≤ min(最优误差+2pp, 3pp) 的日期，取其中**最晚**者：
    # 净值走平/多笔买入时存在多个同日收益率的日期，按最近一笔近似（对止损判断偏保守不误报）
    cand = np.where(abs_err <= min(min_err + 0.02, tol_base))[0]
    if len(cand) == 0:
        cand = np.array([i_min])
    i = int(cand.max())
    return adj.index[i], float(err[i])


def fund_stop_diag(rec, nav_df, stop=0.20, warn_dd=0.15):
    """单基金入场高点回撤 → 移动止损状态。

    返回 dict：
      computable / status(ok|near|triggered|no_data|need_entry)
      entry_date / entry_nav / inferred(收益率反推) / days_held
      peak / peak_date / dd(自入场高点回撤) / trigger_nav(触发清仓净值)
      ret_held(真实持有收益) / ret_user(用户报的收益率) / reason
    """
    adj = adj_series(nav_df)
    ui = parse_user_inputs(rec)
    last_date = None
    if nav_df is not None and len(nav_df):
        last_date = pd.Timestamp(nav_df["date"].iloc[-1]).date()
    out = dict(
        computable=False, status="no_data", reason="",
        entry_date=None, entry_nav=None, inferred=False, days_held=None,
        peak=None, peak_date=None, dd=None, trigger_nav=None,
        ret_held=None, ret_user=ui["ret_pct"], stop=float(stop), last_date=last_date,
        amount=ui["amount"], cost=ui["cost"],
    )
    if len(adj) < 30:
        out["status"] = "no_data"
        out["reason"] = "净值历史不足，无法计算入场高点回撤"
        return out
    entry = ui["buy_date"]
    if entry is None:
        # 用持有收益率反推入场日；未给收益率但给了市值+成本 → 隐含收益率 = 市值/成本-1
        r_infer = ui["ret_pct"]
        if r_infer is None and ui["amount"] and ui["cost"] and ui["cost"] > 0:
            r_infer = (ui["amount"] / ui["cost"] - 1.0) * 100.0
            out["ret_user"] = round(r_infer, 4)
        if r_infer is None:
            out["status"] = "need_entry"
            out["reason"] = "缺少买入日期/成本/收益率，无法定位入场高点（输入 代码 市值 买入日期，或 代码 市值 收益率%，或 代码 市值 成本）"
            return out
        entry, infer_err = infer_entry_date(adj, r_infer)
        if entry is None:
            out["status"] = "need_entry"
            out["reason"] = "持有收益率与净值曲线无法匹配（多笔买入/转换/申赎费差异），请在支付宝交易记录中查买入日期后填写"
            return out
        out["inferred"] = True
    entry = pd.Timestamp(entry)
    if entry < adj.index[0]:
        entry = adj.index[0]
    if entry > adj.index[-1]:
        entry = adj.index[-1]
    seg = adj[adj.index >= entry]
    if len(seg) < 2:
        out["status"] = "no_data"
        out["reason"] = "入场日之后净值数据不足"
        return out
    now = float(adj.iloc[-1])
    entry_nav = float(seg.iloc[0])
    peak = float(seg.max())
    peak_date = seg.idxmax()
    dd = now / peak - 1.0
    ret_held = now / entry_nav - 1.0
    out.update(
        computable=True,
        entry_date=str(entry.date()),
        entry_nav=round(entry_nav, 4),
        inferred=out["inferred"],
        days_held=int((adj.index[-1] - entry).days),
        peak=round(peak, 4),
        peak_date=str(peak_date.date()),
        dd=float(dd),
        trigger_nav=round(peak * (1.0 - stop), 4),
        ret_held=float(ret_held),
    )
    if dd <= -stop:
        out["status"] = "triggered"      # 清仓
    elif dd <= -warn_dd:
        out["status"] = "near"           # 接近止损
    else:
        out["status"] = "ok"
    return out


def cppi_tier_sim(value, rules, full_slots, hysteresis=0.02):
    """按日序模拟 CPPI 状态机（触发/回补带滞回），返回 (events, slots_now, dd_now)。

    value: 组合净值序列(ndarray, 时间升序)
    rules: [(dd线, 槽位), ...] 回撤加深排序，如 [(-0.15,6),(-0.20,3),(-0.25,0)]
    events: [(date_idx, kind, dd线, 槽位)] kind: trigger|restore|newhigh
    触发（下行）: dd ≤ 线      → 降档（-15→6槽 / -20→3槽 / -25→清仓）
    回补（上行）: dd > 下一档线+滞回带 → 升档（-23→3槽 / -18→6槽 / -13→满槽）；
                创出新高 → 直接满槽（HWM重置纪律）
    """
    n = len(value)
    events = []
    hwm = -np.inf
    slots = full_slots
    rules = sorted(rules, key=lambda r: -r[0])         # 由浅入深: [(-0.15,6),(-0.20,3),(-0.25,0)]
    if not rules:
        return events, slots, 0.0, hwm
    full_restore_line = rules[0][0] + hysteresis       # 满槽回补线 = 最浅触发线+滞回带（-15%+2pp=-13%）
    for i in range(n):
        v = float(value[i])
        if v > hwm:
            hwm = v
            if slots < full_slots:
                slots = full_slots
                events.append((i, "newhigh", None, full_slots))
        if hwm <= 0:
            continue
        dd_i = v / hwm - 1.0
        # 触发（下行）：跌破线且槽位更低才降
        for line, s in rules:
            if dd_i <= line and s < slots:
                slots = s
                events.append((i, "trigger", line, s))
        # 回补（上行）：一次到位，取可恢复的最高档
        if slots < full_slots:
            best, best_line = slots, None
            for j, (line, s) in enumerate(rules):
                if s <= slots:
                    continue
                deeper_line = rules[j + 1][0] if j + 1 < len(rules) else line
                if dd_i > deeper_line + hysteresis and s > best:
                    best, best_line = s, deeper_line + hysteresis
            if dd_i > full_restore_line:
                best, best_line = full_slots, full_restore_line
            if best > slots:
                slots = best
                events.append((i, "restore", best_line, best))
    dd_now = (float(value[-1]) / hwm - 1.0) if hwm > 0 else 0.0
    return events, slots, dd_now, hwm


def portfolio_cppi(fund_series, cash=0.0, rules=None, full_slots=10,
                   hysteresis=0.02, max_points=160):
    """组合级 CPPI：重建组合净值曲线 → HWM → 当前回撤 → 档位/槽位/回补提示。

    fund_series: [(entry_date, scale_value, adj_series), ...]
      entry_date 可为推断日；scale_value = 当前市值（无市值时可用成本兜底），
      曲线按 市值×adj(t)/adj(now) 重建 —— 与 成本×adj(t)/adj(entry) 完全等价，
      因此只要"市值+买入日期（或收益率反推日期）"即可，无需用户维护净值曲线。
    cash: 可用现金（按常数并入组合净值，近似处理）
    返回 dict（computable=False 时仅 reason）：
      hwm / current / dd / slots / tier_name / full_slots
      events: 历史触发/回补事件（含日期、类型）
      next_trigger: 当前档位下一触发线(净值与金额)
      restore: 当前档位回补线（回升到多少恢复多少槽，含金额与最后触及日）
      chart: 抽样后的 {dates, value, hwm} 供前端画曲线
    """
    valid = []
    for entry, scale, adj in fund_series:
        if entry is None or not scale or scale <= 0 or adj is None or len(adj) < 5:
            continue
        entry = pd.Timestamp(entry)
        if float(adj.iloc[-1]) <= 0 or float(adj.iloc[0]) <= 0:
            continue
        valid.append((entry, float(scale), adj))
    if not valid:
        return dict(computable=False,
                    reason="持仓缺少 市值+买入日期（或 成本/收益率），无法重建组合净值曲线（提供后自动计算真实回撤与槽位）")
    # 并集交易日
    idx = valid[0][2].index
    for _, _, a in valid[1:]:
        idx = idx.union(a.index)
    idx = pd.DatetimeIndex(sorted(idx))
    contrib = np.zeros(len(idx))
    for entry, scale, adj in valid:
        now_adj = float(adj.iloc[-1])
        vals = adj.reindex(idx).values / now_adj * scale
        vals[idx < entry] = np.nan          # 入场前不计
        vals = np.where(np.isnan(vals), 0.0, vals)
        contrib = contrib + vals
    value = contrib + cash                  # 现金常数并入
    if rules is None:
        rules = [(-0.15, 6), (-0.20, 3), (-0.25, 0)]
    events, slots, dd_now, hwm = cppi_tier_sim(value, rules, full_slots, hysteresis)
    current = float(value[-1])
    # 抽样曲线
    step = max(1, len(idx) // max_points)
    pts = list(range(0, len(idx), step))
    if pts[-1] != len(idx) - 1:
        pts.append(len(idx) - 1)
    hwm_line = np.maximum.accumulate(value)
    chart = dict(
        dates=[str(idx[i].date()) for i in pts],
        value=[round(float(value[i]), 2) for i in pts],
        hwm=[round(float(hwm_line[i]), 2) for i in pts],
    )
    # 事件转日期
    ev = []
    for i, kind, line, s in events:
        ev.append(dict(date=str(idx[i].date()), kind=kind,
                       dd=line, slots=s))
    # 档位名
    tier_names = {full_slots: "正常·满仓", 6: "风控档·减槽", 3: "防御档·深度减槽", 0: "清仓档·禁权益"}
    tier_name = tier_names.get(slots, f"{slots} 槽")
    # 当前档位的下一触发线 & 回补线
    rules_by_slots = {s: line for line, s in rules}
    next_trigger = None
    if slots > 0:
        for line, s in sorted(rules, key=lambda r: -r[1]):   # 6,3,0
            if s < slots:
                next_trigger = dict(dd=line, slots=s,
                                    value=round(hwm * (1 + line), 2))
                break
    restore = None
    if slots < full_slots:
        # 回补目标 = 上一更高档位的槽位（若无更高档则满仓）
        higher = sorted([s for _, s in rules if s > slots] + [full_slots])
        target = higher[0] if higher else full_slots
        if target == full_slots:
            line_restore = rules[0][0] if rules else -0.15   # 最浅触发线
        else:
            line_restore = next(line for line, s in rules if s == target)
        ratio = 1 + line_restore + hysteresis
        restore_val = hwm * ratio
        # 回补线最近一次被站上（历史）→ 给用户时间参照
        last_on = None
        for i in range(len(idx) - 1, -1, -1):
            if value[i] >= restore_val:
                last_on = str(idx[i].date())
                break
        restore = dict(
            slots=target, dd=round(line_restore + hysteresis, 4),
            value=round(restore_val, 2), last_on=last_on,
            new_high_value=round(hwm, 2),
        )
    return dict(
        computable=True,
        n_funds=len(valid),
        cash=cash,
        hwm=round(hwm, 2),
        current=round(current, 2),
        dd=float(dd_now),
        slots=int(slots),
        full_slots=int(full_slots),
        tier_name=tier_name,
        events=ev,
        next_trigger=next_trigger,
        restore=restore,
        chart=chart,
        rules=[dict(dd=line, slots=s) for line, s in rules],
        hysteresis=hysteresis,
    )
