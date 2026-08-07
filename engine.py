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
                    OVERSEAS_SWITCH_THRESHOLD, TENURE_CAP_DAYS, YOUNG_MAX_DAYS,
                    USE_V4_MODEL, W_V4)
import provider, rbsa, factors, risk

# V4 Huber 模型（可选依赖：未训练或未装 sklearn 时优雅降级到 V3.7）
model_v4 = None
_V4_BUNDLE = None
try:
    if USE_V4_MODEL:
        import model_v4 as _mv4
        _V4_BUNDLE = _mv4.load_model()
        if _V4_BUNDLE is not None:
            model_v4 = _mv4
except Exception as _e:
    import sys
    print(f"[engine] V4 模型加载失败，回退 V3.7: {_e}", file=sys.stderr)


def is_overseas_fund(code: str, name: str, ftype: str = None) -> bool:
    """判断必须可注入历史类型，PiT 回测不可回查今天的基金名录。"""
    ftype = ftype if ftype is not None else (provider.fund_type(code) or "")
    return ("海外" in ftype) or ("QDII" in ftype) or ("QDII" in name.upper())


def market_water(as_of=None) -> float:
    """大盘水位计 (6风格等权PE分位)"""
    return rbsa.market_water_level(as_of=as_of)


def resolve_weights(water: float):
    """V3.2 regime自适应: 水位≤20% → 左侧模式(估值↑ 动量↓)"""
    if water is not None and water == water and water <= REGIME_LOW_WATER:
        return (W_VALUE_LOW, W_ALPHA_LOW, W_MOM_LOW), "左侧低估区"
    return (W_VALUE, W_ALPHA, W_MOMENTUM), "标准"


def score_fund(code: str, as_of: str = None, bt: bool = False, indices=None,
               pit_meta: dict = None) -> dict:
    """
    单基金流水线(动量截面排名在 universe 层完成)
      as_of: 回测截止日 —— 净值/估值全部截断至该日(Point-in-Time)
      bt=True: 回测模式 —— 跳过非PiT数据(经理档案/AUM/缩水加分)与其挂钩惩罚
      pit_meta: 历史快照里的 ``name``/``fund_type``。严格回测传入后，类型、
                被动基金识别及海外判断均不会查询今天的 fund_meta。
      indices: 因子面板(默认 config 全局; 回测传风格6面板)
    """
    nav = provider.get_fund_nav(code)
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        nav = nav[nav["date"] <= as_of_ts]
    # 档案拉取失败不阻断评分：降级为空档案，后续因子与风控按缺失处理
    if bt:
        dossier = {}
    else:
        try:
            dossier = provider.get_fund_dossier(code)
        except Exception as e:
            dossier = {"code": code, "name": code, "scale_hist": [], "asset_alloc": {}, "managers": [], "_err": str(e)[:80]}
    if bt:
        # pit_meta 是严格回测的唯一元数据来源；缺失时保留旧兼容路径，调用方须披露。
        if pit_meta is not None:
            name = str(pit_meta.get("name") or code)
            ftype = str(pit_meta.get("fund_type") or "")
        else:
            try:
                name = str(provider.get_fund_meta().loc[code, "基金简称"])
            except Exception:
                name = code
            ftype = provider.fund_type(code)
    else:
        name = dossier.get("name", code)
        # 档案降级为空时（_stale），尝试从全市场名录补全名称
        if not name or name == code:
            try:
                name = str(provider.get_fund_meta().loc[code, "基金简称"])
            except Exception:
                name = dossier.get("name", code)
        ftype = provider.fund_type(code)
    nav = nav.set_index("date")
    ret = nav["ret"]
    adj = (1 + ret.fillna(0)).cumprod()

    out = {"code": code, "name": name, "ftype": ftype,
           "n_days": len(nav), "last_date": str(nav.index[-1].date() if len(nav) else None)}

    min_days = 800 if bt else 200        # 回测要求≥3年alpha窗口+缓冲
    if len(nav) < min_days:
        # 新基友好提示：区分成立不足200天的新基与异常短历史
        if len(nav) < 200:
            out["error"] = f"成立仅{len(nav)}天，历史不足200天暂无法按V3完整评分（需≥200日）；建议作为观察仓≤5%或等满200日后再评"
        else:
            out["error"] = f"净值历史不足({len(nav)}d)"
        return out
    # 历史快照即使误收录了已停止披露的份额，也不能把很久以前的净值当作当日可交易价格。
    if bt and as_of is not None and nav.index[-1] < pd.Timestamp(as_of) - pd.Timedelta(days=7):
        out["error"] = f"截至回测日净值过期({nav.index[-1].date()})"
        return out

    # --- RBSA 穿透 (V3.3: 海外基金"借壳"修正) ---
    indices_use = indices or RBSA_INDICES
    idx_ret = rbsa.index_return_matrix(nav.index, indices=indices_use)
    w = rbsa.rbsa_weights(ret, idx_ret)
    panel_mode = "unified"
    # 海外类型基金: 母市场就是其法定benchmark → 直接切纯境外面板(无条件)
    if not bt and is_overseas_fund(code, name, ftype):
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
    # 历史类型进入严格 PiT 回测时，不能调用 provider.is_passive_fund（其读取今日名录）。
    is_passive = (("指数" in ftype) or ("指数" in name) or ("ETF" in name) or ("联接" in name)) \
        if (bt and pit_meta is not None) else provider.is_passive_fund(code, name)
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


def get_global_momentum_ref(as_of: str = None):
    """
    获取全局动量参照标尺 (Global Momentum Reference Universe)
    为单基金透视、批量评分(如自选池/支付宝持仓)、持仓诊断提供同源唯一参考分布，
    避免小样本被内部直接 rank(pct=True) 导致打分严重被排挤与产生前后背离。
    返回: (ref_p4: pd.Series, ref_p7: pd.Series)
    """
    import glob, os
    from config import OUTPUT_DIR
    ref_p4 = ref_p7 = None
    if as_of is None:
        # 实时环境优先：近期生成的全市场扫描候选池
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "scan_*.csv")))
        if files:
            try:
                df_ref = pd.read_csv(files[-1], dtype={"code": str})
                p4 = pd.to_numeric(df_ref.get("mom_4m1m"), errors="coerce").dropna()
                p7 = pd.to_numeric(df_ref.get("mom_7m1m"), errors="coerce").dropna()
                if len(p4) >= 30 and len(p7) >= 30:
                    ref_p4, ref_p7 = p4, p7
            except Exception:
                pass
    # 备用方案（兜底缓存）：若不存在近期 scan 文件，则取最新一期全量时点表 (如 2026-06-30.csv)
    if ref_p4 is None or ref_p7 is None:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "bt_scores_cache", "*.csv")))
        if files:
            try:
                df_ref = pd.read_csv(files[-1], dtype={"code": str})
                p4 = pd.to_numeric(df_ref.get("mom_4m1m"), errors="coerce").dropna()
                p7 = pd.to_numeric(df_ref.get("mom_7m1m"), errors="coerce").dropna()
                if len(p4) >= 30 and len(p7) >= 30:
                    ref_p4, ref_p7 = p4, p7
            except Exception:
                pass
    return ref_p4, ref_p7


def finalize(rows: list, as_of: str = None, use_global_ref: bool = False) -> pd.DataFrame:
    """截面动量排名 → F_momentum → S_total(regime权重) × 惩罚 → 评级
    as_of: 回测日(水位计PiT); None → 实时
    use_global_ref: 局部/小样本测算(单基、自选池、持仓诊断)采用全市场统一动量参照标尺"""
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
    if use_global_ref:
        ref_p4, ref_p7 = get_global_momentum_ref(as_of)
        if ref_p4 is not None and ref_p7 is not None and len(ref_p4) >= 30 and len(ref_p7) >= 30:
            df["rank4"] = df["mom_4m1m"].apply(lambda m: float((ref_p4 <= m).mean()) if pd.notna(m) else np.nan)
            df["rank7"] = df["mom_7m1m"].apply(lambda m: float((ref_p7 <= m).mean()) if pd.notna(m) else np.nan)
        else:
            df["rank4"] = df["mom_4m1m"].rank(pct=True)
            df["rank7"] = df["mom_7m1m"].rank(pct=True)
    else:
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

    df["S_v37"] = [total(r) for _, r in df.iterrows()]

    # ---------- V4: Huber 稳健回归 + 截面 rank ----------
    # 训练: 28季 PiT 面板, 7 经济学特征, 2 年衰减, Huber ε=1.35
    # 验证: WF IC 0.198 vs V3.7 0.116 (+71%); strict OOS IC 0.171 vs 0.093 (+85%)
    if model_v4 is not None and _V4_BUNDLE is not None:
        bundle = _V4_BUNDLE
        # 截面 rank 各原始量 (估值分位 val_pct 已是 5 年分位，不再 rank)
        df["wr_rk"] = df["ir_winrate"].rank(pct=True)
        df["dc_rk"] = df["down_capture"].rank(pct=True)
        df["r4_rk"] = df["mom_4m1m"].rank(pct=True)
        df["r7_rk"] = df["mom_7m1m"].rank(pct=True)
        # V3.6 平滑回撤惩罚 (与训练时口径一致；底部区减半)
        rmdd_pen = np.zeros(len(df))
        pdt = df["penalty_detail"].apply(lambda d: d or {})
        r_mdd = pdt.apply(lambda d: d.get("R_MDD"))
        mask = r_mdd.notna() & (r_mdd > 1.2)
        rmdd_pen[mask] = np.minimum(0.5 * (r_mdd[mask] - 1.2), 1.0)
        if pd.notna(water):
            if water <= 0.35:
                rmdd_pen[mask] *= 0.5
        # 趋势确认度 (与训练时同口径)
        trend_t = np.clip((df["ma20_dist"].fillna(0) + 0.02) / 0.06, 0, 1)

        # 估值盲区：V4 训练时 val_pct 是有效输入；盲区基金用截面中位数补
        val_pct = df["val_pct"].astype(float).copy()
        val_pct = val_pct.fillna(val_pct.median())

        w_arr = np.full(len(df), water if water == water else 0.43)  # 截面均值兜底
        X = model_v4.build_features(
            val_pct.values, df["r4_rk"].values, df["r7_rk"].values,
            df["wr_rk"].values, df["dc_rk"].values, rmdd_pen,
            w_arr, trend_t.values)
        m = bundle["model"]
        z = m.named_steps["hub"].predict(m.named_steps["sc"].transform(X))
        df["S_v4_raw"] = z
        # 映射到 0~100 截面百分位
        df["S_v4"] = (pd.Series(z).rank(pct=True) * 100).values
        # V4 只乘非 R_MDD 惩罚 (任期归因/规模反噬/集中度)；R_MDD 已在模型内
        non_rmdd_penalties = df["penalties"].apply(
            lambda ps: [(n, p) for n, p in (ps or []) if "回撤比值" not in n])
        df["S_v4_penalized"] = [
            round(risk.apply_penalties(s, pl), 1) for s, pl in
            zip(df["S_v4"], non_rmdd_penalties)
        ]
        # V4/V3.7 混合：V4 IC 高但在高水位偏激进，V3.7 的估值悬崖是安全垫
        wv4 = float(W_V4) if W_V4 is not None else 0.5
        df["S_total"] = (wv4 * df["S_v4_penalized"] +
                         (1 - wv4) * df["S_v37"]).round(1)
        df["model_version"] = f"{bundle.get('version', 'V4')}+V3.7×{1-wv4:.1f}"
    else:
        df["S_v4"] = np.nan
        df["S_v4_penalized"] = np.nan
        df["S_v4_raw"] = np.nan
        df["S_total"] = df["S_v37"]
        df["model_version"] = "V3.7"

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
