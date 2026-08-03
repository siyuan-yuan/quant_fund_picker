# -*- coding: utf-8 -*-
"""
评分引擎: 单基金全流水线 + 跨截面动量排名 + 总分合成
"""
import numpy as np
import pandas as pd

from config import (W_VALUE, W_ALPHA, W_MOMENTUM, RATING_BANDS,
                    REGIME_LOW_WATER, REGIME_HIGH_WATER,
                    W_VALUE_LOW, W_ALPHA_LOW, W_MOM_LOW,
                    RBSA_INDICES, OVERSEAS_NAMES, OVERSEAS_SRCS,
                    OVERSEAS_SWITCH_THRESHOLD, TENURE_CAP_DAYS, YOUNG_MAX_DAYS)
import provider, rbsa, factors, risk


def is_overseas_fund(code: str, name: str) -> bool:
    ftype = provider.fund_type(code) or ""
    return ("海外" in ftype) or ("QDII" in ftype) or ("QDII" in name.upper())


def market_water(as_of=None) -> float:
    """大盘水位计 (6风格等权PE分位)"""
    return rbsa.market_water_level(as_of=as_of)


def resolve_weights(water: float):
    """V3.2 regime自适应: 水位≤20% → 左侧模式(估值↑ 动量↓)"""
    if water is not None and water == water and water <= REGIME_LOW_WATER:
        return (W_VALUE_LOW, W_ALPHA_LOW, W_MOM_LOW), "左侧低估区"
    return (W_VALUE, W_ALPHA, W_MOMENTUM), "标准"


def score_fund(code: str, as_of: str = None, bt: bool = False, indices=None) -> dict:
    """
    单基金流水线(动量截面排名在 universe 层完成)
      as_of: 回测截止日 —— 净值/估值全部截断至该日(Point-in-Time)
      bt=True: 回测模式 —— 跳过非PiT数据(经理档案/AUM/缩水加分)与其挂钩惩罚
      indices: 因子面板(默认 config 全局; 回测传风格6面板)
    """
    nav = provider.get_fund_nav(code)
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        nav = nav[nav["date"] <= as_of_ts]
    dossier = {} if bt else provider.get_fund_dossier(code)
    if bt:
        try:
            name = str(provider.get_fund_meta().loc[code, "基金简称"])
        except Exception:
            name = code
    else:
        name = dossier.get("name", code)
    nav = nav.set_index("date")
    ret = nav["ret"]
    adj = (1 + ret.fillna(0)).cumprod()

    out = {"code": code, "name": name, "ftype": provider.fund_type(code),
           "n_days": len(nav), "last_date": str(nav.index[-1].date() if len(nav) else None)}

    min_days = 800 if bt else 200        # 回测要求≥3年alpha窗口+缓冲
    if len(nav) < min_days:
        out["error"] = f"净值历史不足({len(nav)}d)"
        return out

    # --- RBSA 穿透 (V3.3: 海外基金"借壳"修正) ---
    indices_use = indices or RBSA_INDICES
    idx_ret = rbsa.index_return_matrix(nav.index, indices=indices_use)
    w = rbsa.rbsa_weights(ret, idx_ret)
    panel_mode = "unified"
    # 海外类型基金: 母市场就是其法定benchmark → 直接切纯境外面板(无条件)
    if not bt and is_overseas_fund(code, name):
        ov_idx = [x for x in RBSA_INDICES if x[0] in OVERSEAS_SRCS]
        ov_ret = rbsa.index_return_matrix(nav.index, indices=ov_idx)
        w_ov = rbsa.rbsa_weights(ret, ov_ret)
        if w_ov.notna().all() and w_ov.sum() > 0.5:   # 拟合退化则回退统一面板
            idx_ret, w, indices_use, panel_mode = ov_ret, w_ov, ov_idx, "overseas"
    out["rbsa"] = {k: round(float(v), 3) for k, v in w.items()}
    out["panel_mode"] = panel_mode

    # 动态基准 (当期隐形仓位 × 指数收益)
    bench = (idx_ret[w.index] * w).sum(axis=1).fillna(0)

    # --- F_value ---
    pct, val_cov = rbsa.fund_valuation_percentile(w, indices=indices_use, as_of=as_of)
    dif = factors.macd_dif(adj)
    ma_sig = factors.ma_reversal_ok(adj)              # V3.2 右侧反转触发器
    ma20_dist = factors.ma20_distance(adj)            # V3.7 平滑趋势门输入
    trend_ok = ((not np.isnan(dif)) and dif > 0) or ma_sig
    if bt:
        tenure_max, bonus, bonus_dt = 0, 0, {"pass": False}
    else:
        tenures = [provider.parse_worktime_days(m["workTime"]) for m in dossier.get("managers", [])]
        tenure_max = max(tenures) if tenures else 0
        bonus, bonus_dt = factors.drawdown_bonus_signal(dossier, tenure_max)
    # V3.3 估值盲区: PE覆盖率<50% (如美股QDII) → F_value不计入
    valuation_blind = (val_cov < 0.5) or (pct != pct)
    if valuation_blind:
        v_base, f_value = np.nan, None
    else:
        v_base = factors.valuation_base_score(pct, trend_ok)
        f_value = min((v_base if not np.isnan(v_base) else 0) + bonus, 100)

    # --- F_alpha --- (V3.7: 平滑评分, 面板裁决 t 3.20→3.42)
    active = (ret - bench).dropna()
    wr = factors.rolling_ir_winrate(active)
    dc = factors.downside_capture(ret, bench)
    s_ir, s_dc = factors.ir_score_smooth(wr), factors.dc_score_smooth(dc)
    subs = [x for x in (s_ir, s_dc) if not np.isnan(x)]
    f_alpha = float(np.mean(subs)) if subs else np.nan

    # --- 动量原始收益 (截面排名在 universe 层完成) ---
    r4, r7 = factors.lagged_momentum_returns(adj)

    # --- 风控 ---
    cur_scale = bonus_dt.get("cur_scale")
    is_passive = provider.is_passive_fund(code, name)
    pens, pdt = risk.build_penalties(
        tenure_days=tenure_max, is_passive=is_passive,
        fund_ret=ret, bench_ret=bench, weights=w,
        smallcap_exp=rbsa.smallcap_exposure(w), cur_scale=cur_scale,
        ir_winrate=wr, skip_tenure=bt, skip_aum=bt,
        water=market_water(as_of))          # V3.5条款5: 水位感知否决

    out.update({
        "f_value_base": None if np.isnan(v_base) else round(v_base, 1),
        "val_pct": None if np.isnan(pct) else round(pct, 4),
        "val_coverage": round(val_cov, 3),
        "valuation_blind": bool(valuation_blind),
        "macd_dif": None if np.isnan(dif) else round(dif, 5),
        "ma20_dist": None if np.isnan(ma20_dist) else round(ma20_dist, 5),
        "trend_ma20": ma_sig,
        "trend_ok": trend_ok, "bonus": bonus, "bonus_detail": bonus_dt,
        "F_value": None if f_value is None else round(f_value, 1),
        "ir_winrate": None if np.isnan(wr) else round(wr, 3),
        "s_ir": None if np.isnan(s_ir) else round(s_ir, 1),
        "down_capture": None if np.isnan(dc) else round(dc, 3),
        "s_dc": None if np.isnan(s_dc) else round(s_dc, 1),
        "F_alpha": None if np.isnan(f_alpha) else round(f_alpha, 1),
        "mom_4m1m": None if np.isnan(r4) else round(r4, 4),
        "mom_7m1m": None if np.isnan(r7) else round(r7, 4),
        "tenure_days": tenure_max, "is_passive": is_passive,
        "penalties": pens, "penalty_detail": pdt,
        "scale": cur_scale,
    })
    return out


def finalize(rows: list, as_of: str = None) -> pd.DataFrame:
    """截面动量排名 → F_momentum → S_total(regime权重) × 惩罚 → 评级
    as_of: 回测日(水位计PiT); None → 实时"""
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = None
    for col, default in [("mom_4m1m", np.nan), ("mom_7m1m", np.nan),
                         ("F_value", np.nan), ("F_alpha", np.nan),
                         ("penalties", None)]:
        if col not in df:
            df[col] = default
    df["penalties"] = df["penalties"].apply(lambda p: p if isinstance(p, list) else [])

    water = market_water(as_of)
    (wv, wa, wm), mode = resolve_weights(water)
    df["rank4"] = df["mom_4m1m"].rank(pct=True)
    df["rank7"] = df["mom_7m1m"].rank(pct=True)
    df["F_momentum"] = df.apply(
        lambda r: factors.momentum_score_smooth_m1(r["rank4"], r["rank7"]), axis=1).round(1)   # V3.7 平滑M1 (t 3.20→3.33)

    def total(r):
        # V3.3: 缺失因子(如估值盲区) → 按剩余因子归一化, 不做惩罚性置零
        num, den = 0.0, 0.0
        if not pd.isna(r["F_value"]):
            num += wv * min(r["F_value"], 100); den += wv
        if not pd.isna(r["F_alpha"]):
            num += wa * r["F_alpha"]; den += wa
        if not pd.isna(r["F_momentum"]):
            num += wm * r["F_momentum"]; den += wm
        base = (num / den) if den > 1e-9 else 0.0
        return round(risk.apply_penalties(base, r["penalties"] or []), 1)

    df["S_total"] = df.apply(total, axis=1)
    df["water"] = None if water != water else round(water, 4)
    df["weights_mode"] = mode
    df["w_value"], df["w_alpha"], df["w_mom"] = wv, wa, wm

    def rate(s):
        for th, lab in RATING_BANDS:
            if s >= th:
                return lab
        return RATING_BANDS[-1][1]

    def rate_row(r):
        """V3.5 评级封顶规则"""
        s = r["S_total"]
        nd = r.get("n_days")
        # 条款4: 新星基金(净值<3年) → 🌱观察仓封顶 (S≥50才发徽章)
        if isinstance(nd, (int, float)) and nd == nd and nd < YOUNG_MAX_DAYS:
            return "🌱观察仓" if s >= 50 else rate(s)
        # 条款3: 经理任期<2年 → 封顶 Buy (bt模式 tenure=0 不触发)
        td = r.get("tenure_days")
        if td and s >= 85:
            return "Buy 浅绿(任期<2年封顶)" if td < TENURE_CAP_DAYS else rate(s)
        return rate(s)

    df["rating"] = df.apply(rate_row, axis=1)
    df["penalty_str"] = df["penalties"].apply(
        lambda p: "; ".join(f"{n}(-{x:.0%})" for n, x in p) if p else "")
    return df.sort_values("S_total", ascending=False).reset_index(drop=True)
