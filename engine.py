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


def _safe_list(v):
    """把 pandas 的 NaN / None / 其他脏值统一收敛成 list。"""
    return v if isinstance(v, list) else []


def _safe_dict(v):
    """把 pandas 的 NaN / None / 其他脏值统一收敛成 dict。"""
    return v if isinstance(v, dict) else {}


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
        # 任期未知(None) ≠ 0年：档案拉取失败/无经理信息时宁豁免平滑折价，
        # 也不按"0年任期"顶格罚-30%(旧版限流时随机扣分的不一致性来源)
        tenure_max = max(tenures) if tenures else None
        bonus, bonus_dt = factors.drawdown_bonus_signal(dossier, tenure_max or 0)
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

    # 档案降级标记: provider 彻底失败时返回带 _err 的空档案(可能复用旧缓存的也无此标记)
    dossier_degraded = bool(isinstance(dossier, dict) and dossier.get("_err"))
    data_incomplete = bool((not bt) and (dossier_degraded or tenure_max is None))

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
        "data_incomplete": data_incomplete,
    })
    return out


_REF_CACHE = {"key": None, "ref": None}   # 进程内参照快照缓存 (path, mtime) 单槽


def _load_ref_file(path: str, stamp: str):
    """从参照快照文件解析各因子全市场分布。列别名兼容两类来源：
      扫描榜单 scan_*.csv      → mom_4m1m / mom_7m1m / ir_winrate / down_capture / val_pct / S_v4_raw
      回测时点 bt_scores_cache → r4 / r7 / wr / dc / val_pct  (无 z 列 → 仅支撑 V3.7 腿)
    任一分布样本<30 视为不可用(None)。"""
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except Exception:
        return None
    alias = {"mom4": ["mom_4m1m", "r4"], "mom7": ["mom_7m1m", "r7"],
             "wr": ["ir_winrate", "wr"], "dc": ["down_capture", "dc"],
             "val_pct": ["val_pct"], "z": ["S_v4_raw"]}

    def col(key):
        for c in alias[key]:
            if c in df:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) >= 30:
                    return s
        return None

    vp = col("val_pct")
    return {"stamp": stamp,
            "mom4": col("mom4"), "mom7": col("mom7"),
            "wr": col("wr"), "dc": col("dc"),
            "val_median": float(vp.median()) if vp is not None else np.nan,
            "z": col("z")}


def get_global_ref_universe(as_of: str = None):
    """
    全市场参照宇宙快照 (Global Reference Universe)
    为单基透视、批量评分、持仓诊断等小样本测算提供同源唯一参照分布：
      mom4/mom7   —— V3.7 动量腿的 ECDF 标尺
      wr/dc       —— V4 质量腿输入 (ir胜率/下行捕获) 的 ECDF 标尺
      val_median  —— V4 估值盲区填充中位数(替代批内中位数)
      z           —— V4 原始预测值的全市场分位映射(替代批内 rank→0~100)
    返回 dict(stamp, mom4, mom7, wr, dc, val_median, z) 或 None。
    批次无关性：同一基金在同一快照下，无论与 1 只还是 5000 只基金同批计算，结果严格一致。
    """
    import glob, os
    from config import OUTPUT_DIR

    def _newest(pattern):
        fs = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
        return fs[-1] if fs else None

    cands = []
    if as_of is None:
        cands.append(_newest("scan_*.csv"))                      # 实时：最新全市场扫描
    cands.append(_newest(os.path.join("bt_scores_cache", "*.csv")))  # 兜底：最新全量时点表
    for f in cands:
        if not f:
            continue
        try:
            key = (f, os.path.getmtime(f))
        except OSError:
            continue
        if _REF_CACHE["key"] != key:
            stamp = os.path.basename(f).rsplit(".", 1)[0]
            _REF_CACHE.update(key=key, ref=_load_ref_file(f, stamp))
        ref = _REF_CACHE["ref"]
        if ref and ref["mom4"] is not None and ref["mom7"] is not None:
            return ref
    return None


def get_global_momentum_ref(as_of: str = None):
    """兼容旧接口：仅取动量两标尺。新代码请用 get_global_ref_universe。"""
    ref = get_global_ref_universe(as_of)
    if ref:
        return ref["mom4"], ref["mom7"]
    return None, None


def finalize(rows: list, as_of: str = None, use_global_ref: bool = False) -> pd.DataFrame:
    """截面动量排名 → F_momentum → S_total(regime权重) × 惩罚 → 评级
    as_of: 回测日(水位计PiT); None → 实时
    use_global_ref: 局部/小样本测算(单基、自选池、持仓诊断)采用全市场统一参照快照 —
        V3.7 动量腿与 V4 的全部 rank 输入 / z 分位统一对参照宇宙做 ECDF 映射，
        结果与批次大小和批次构成完全无关(单基=批量=诊断)；
        快照缺失 V4 所需分布时按一致性闸门整体降级 V3.7，绝不退回批内 rank。"""
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = None
    # 批量扫描/自选/调仓会混入异常行；pandas 会把缺失 dict/list 列补成 float('nan')。
    # 若后续直接 d.get(...)，就会触发经典事故：'float' object has no attribute 'get'。
    # 这里把 finalize 依赖的列一次性补齐并做类型归一化，让错误行保留但绝不拖垮整批。
    for col, default in [("mom_4m1m", np.nan), ("mom_7m1m", np.nan),
                         ("F_value", np.nan), ("F_alpha", np.nan),
                         ("val_pct", np.nan), ("ma20_dist", np.nan),
                         ("ir_winrate", np.nan), ("down_capture", np.nan),
                         ("penalties", None), ("penalty_detail", None)]:
        if col not in df:
            df[col] = default
    df["penalties"] = df["penalties"].apply(_safe_list)
    df["penalty_detail"] = df["penalty_detail"].apply(_safe_dict)

    water = market_water(as_of)
    (wv, wa, wm), mode = resolve_weights(water)
    ref = get_global_ref_universe(as_of) if use_global_ref else None
    global_mom_ok = ref is not None
    if global_mom_ok:
        df["rank4"] = df["mom_4m1m"].apply(lambda m: float((ref["mom4"] <= m).mean()) if pd.notna(m) else np.nan)
        df["rank7"] = df["mom_7m1m"].apply(lambda m: float((ref["mom7"] <= m).mean()) if pd.notna(m) else np.nan)
        df["ref_stamp"] = ref["stamp"]
    elif use_global_ref:
        # 参照缺失时的批次无关性兜底：宁可关闭动量腿(NaN→总分按剩余因子归一化)，
        # 也绝不退回批内 rank —— 否则单基透视的动量恒为满分位，与批量场景系统性背离
        df["rank4"] = np.nan
        df["rank7"] = np.nan
        df["ref_stamp"] = "无全市场参照(动量腿关闭)"
    else:
        df["rank4"] = df["mom_4m1m"].rank(pct=True)
        df["rank7"] = df["mom_7m1m"].rank(pct=True)
        df["ref_stamp"] = None
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
        # ---- 一致性闸门 (V4.1) ----
        # 小样本模式(use_global_ref)下，V4 必须有全市场快照做 ECDF 映射；
        # 否则批内 rank 会随批次构成漂移(单基时 S_v4 恒=100，模型信息被丢弃) ——
        # 宁可整体回退 V3.7，也绝不退回批内 rank。全域批(扫描/回测)批内即训练域，保持原口径。
        v4_global_ok = (global_mom_ok
                        and ref.get("z") is not None
                        and ref.get("wr") is not None
                        and ref.get("dc") is not None
                        and ref.get("val_median") == ref.get("val_median"))  # NaN 防御
        v4_mode = ("global" if v4_global_ok else "off") if use_global_ref else "batch"

        if v4_mode == "off":
            df["S_v4"] = np.nan
            df["S_v4_penalized"] = np.nan
            df["S_v4_raw"] = np.nan
            df["S_total"] = df["S_v37"]
            df["model_version"] = "V3.7(无V4全市场快照·闸门降级)"
        else:
            if v4_mode == "global":
                # 与 V3.7 动量腿同一把尺：全部 rank 输入对参照宇宙做 ECDF
                df["wr_rk"] = df["ir_winrate"].apply(lambda v: float((ref["wr"] <= v).mean()) if pd.notna(v) else np.nan)
                df["dc_rk"] = df["down_capture"].apply(lambda v: float((ref["dc"] <= v).mean()) if pd.notna(v) else np.nan)
                df["r4_rk"] = df["rank4"]
                df["r7_rk"] = df["rank7"]
                val_fill = ref["val_median"]
            else:
                # 全市场批次: 批内即全截面，与训练同域
                df["wr_rk"] = df["ir_winrate"].rank(pct=True)
                df["dc_rk"] = df["down_capture"].rank(pct=True)
                df["r4_rk"] = df["mom_4m1m"].rank(pct=True)
                df["r7_rk"] = df["mom_7m1m"].rank(pct=True)
                val_fill = df["val_pct"].astype(float).median()
            # V3.6 平滑回撤惩罚 (与训练时口径一致；底部区减半)
            rmdd_pen = np.zeros(len(df))
            pdt = df["penalty_detail"].apply(_safe_dict)
            r_mdd = pdt.apply(lambda d: d.get("R_MDD"))
            mask = r_mdd.notna() & (r_mdd > 1.2)
            rmdd_pen[mask] = np.minimum(0.5 * (r_mdd[mask] - 1.2), 1.0)
            if pd.notna(water):
                if water <= 0.35:
                    rmdd_pen[mask] *= 0.5
            # 趋势确认度 (与训练时同口径)
            trend_t = np.clip((df["ma20_dist"].fillna(0) + 0.02) / 0.06, 0, 1)

            # 估值盲区：V4 训练时 val_pct 是有效输入；盲区基金用参照中位数(global)或全域中位数(batch)补
            val_pct = df["val_pct"].astype(float).copy()
            val_pct = val_pct.fillna(val_fill)

            w_arr = np.full(len(df), water if water == water else 0.43)  # 截面均值兜底
            X = model_v4.build_features(
                val_pct.values, df["r4_rk"].values, df["r7_rk"].values,
                df["wr_rk"].values, df["dc_rk"].values, rmdd_pen,
                w_arr, trend_t.values)
            m = bundle["model"]
            # 逐行防御：任一特征非有限(NaN/±Inf)的行不喂模型，该行 V4 静默回退 V3.7
            # (修复旧版单基估值盲区基 X 含 NaN 直接抛 ValueError → 接口400 的事故)
            finite = np.isfinite(X).all(axis=1)
            z = np.full(len(df), np.nan)
            if finite.any():
                z[finite] = m.named_steps["hub"].predict(m.named_steps["sc"].transform(X[finite]))
            df["S_v4_raw"] = z
            # 映射到 0~100：global 模式对快照 z 分布做 ECDF；batch 模式批内即全截面
            if v4_mode == "global":
                df["S_v4"] = [round(float((ref["z"] <= zz).mean()) * 100, 1)
                              if pd.notna(zz) else np.nan for zz in z]
            else:
                df["S_v4"] = (pd.Series(z).rank(pct=True) * 100).values
            # V4 只乘非 R_MDD 惩罚 (任期归因/规模反噬/集中度)；R_MDD 已在模型内
            non_rmdd_penalties = df["penalties"].apply(
                lambda ps: [(n, p) for n, p in (ps or []) if "回撤比值" not in n])
            df["S_v4_penalized"] = [
                round(risk.apply_penalties(s, pl), 1) if pd.notna(s) else np.nan
                for s, pl in zip(df["S_v4"], non_rmdd_penalties)
            ]
            # V4/V3.7 混合：V4 IC 高但在高水位偏激进，V3.7 的估值悬崖是安全垫
            wv4 = float(W_V4) if W_V4 is not None else 0.5
            st = df["S_v37"].astype(float).copy()
            okb = df["S_v4_penalized"].notna()
            st.loc[okb] = (wv4 * df.loc[okb, "S_v4_penalized"] +
                           (1 - wv4) * df.loc[okb, "S_v37"]).round(1)
            df["S_total"] = st
            tag = "·全市场ECDF" if v4_mode == "global" else ""
            df["model_version"] = f"{bundle.get('version', 'V4')}+V3.7×{1-wv4:.1f}{tag}"
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
        # 条款3: 经理任期<2年 → 封顶 Buy (bt模式 tenure=0 不触发;
        #         tenure=None 表示档案缺失任期未知 → 不封顶，由 data_incomplete 打标披露)
        td = r.get("tenure_days")
        if td and s >= 85:
            return "Buy 浅绿(任期<2年封顶)" if td < TENURE_CAP_DAYS else rate(s)
        return rate(s)

    df["rating"] = df.apply(rate_row, axis=1)
    df["penalty_str"] = df["penalties"].apply(
        lambda p: "; ".join(f"{n}(-{x:.0%})" for n, x in p) if p else "")
    return df.sort_values("S_total", ascending=False).reset_index(drop=True)