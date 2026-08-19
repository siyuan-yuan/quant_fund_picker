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

try:
    import config
    _BUY_FEE_DEFAULT = float(config.BUY_FEE_RATE)
    _BUY_FEE_BY_FUND = getattr(config, "BUY_FEE_BY_FUND", {}) or {}
except Exception:
    _BUY_FEE_DEFAULT = 0.0015
    _BUY_FEE_BY_FUND = {}


def buy_fee_rate(code):
    """取某基金申购费率（折后实扣费率），优先级：

    1. config.BUY_FEE_BY_FUND 按代码覆盖（用户显式指定，最高优先）；
    2. 自动查天天基金（provider.get_fund_buy_fee）该基金的折后申购费率；
    3. 兜底 config.BUY_FEE_RATE 默认值。
    任何失败都返回非 0 的小数费率，绝不阻断计算。
    """
    code = str(code or "")
    try:
        if code in _BUY_FEE_BY_FUND:
            return float(_BUY_FEE_BY_FUND[code])
    except (TypeError, ValueError):
        pass
    try:
        import provider  # 惰性导入，避免 holding_diag 顶部依赖 provider
        info = provider.get_fund_buy_fee(code)
        if isinstance(info, dict) and "rate" in info:
            return float(info.get("rate") or 0)
    except Exception:
        pass
    try:
        return float(_BUY_FEE_DEFAULT)
    except (TypeError, ValueError):
        return 0.0


def adj_series(nav_df):
    """复权净值序列：由天天基金官方日增长率(复权)累乘，date 升序"""
    df = nav_df.copy()
    if "date" not in df or "ret" not in df or len(df) == 0:
        return pd.Series(dtype=float, name="adj")
    df = df.dropna(subset=["ret"]).sort_values("date")
    adj = (1 + df["ret"].astype(float).fillna(0)).cumprod()
    return pd.Series(adj.values, index=pd.DatetimeIndex(df["date"]), name="adj")


def unit_series(nav_df):
    """单位净值序列（账户市值口径：确认份额 × 单位净值）。"""
    if nav_df is None or len(nav_df) == 0 or "date" not in nav_df or "nav" not in nav_df:
        return pd.Series(dtype=float, name="nav")
    df = nav_df.dropna(subset=["nav"]).sort_values("date")
    return pd.Series(df["nav"].astype(float).values,
                     index=pd.DatetimeIndex(df["date"]), name="nav")


def confirm_nav_pos(idx, d, delay=0):
    """下单日 → 确认净值在序列中的位置。

    规则（用户填的是北京 15:00 切日后的下单日）：
      1. 下单日若无净值（周末/法定休市/海外休市）→ 顺延到下一交易日；
      2. delay=1（QDII）再往后一个交易日（T 日下单，按 T+1 净值确认）；
      3. 下单日晚于最新净值 → 暂用最新净值（在途/缓存未更新）。
    """
    if idx is None or len(idx) == 0:
        return 0
    p = int(idx.searchsorted(pd.Timestamp(d)))
    n = len(idx)
    if p >= n:
        return n - 1
    p = min(max(p, 0) + int(delay or 0), n - 1)
    return max(p, 0)


def nav_confirm_delay(code):
    """QDII / 海外基金：下单日 T 按 T+1 净值确认；A 股为 0（当天净值）。"""
    code = str(code or "").zfill(6) if code else ""
    if not code or code == "000000":
        return 0
    try:
        import provider
        from config import OVERSEAS_FUND_TYPES
        ft = str(provider.fund_type(code) or "")
        if ft in OVERSEAS_FUND_TYPES or "QDII" in ft or "海外" in ft:
            return 1
    except Exception:
        pass
    return 0


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


def infer_entry_date(adj, ret_pct, tol_base=0.03, return_info=False):
    """由持有收益率反推买入日。

    复权净值比 adj_now/adj_d - 1 应 ≈ 持有收益率。找误差最小的日期；
    误差容忍 max(3pp, 3×最小误差+0.5pp)，多个近优候选取**最晚**（最近期
    入场，对止损提示最不易误报；如有多笔/定投，近似为最近一笔）。
    返回 (date, err)；无匹配返回 (None, None)。
    return_info=True 时返回 (date, err, span_days)：
      span_days = 近优候选日期区间的跨度(天)。多笔买入/定投/中途卖出时，同一
      持有收益率往往在净值曲线上有多个相距甚远的匹配日 → 反推的单一日期的
      "入场高点回撤"不可靠，调用方应提示用户改用操作台账(每笔买卖)精确诊断。
    """
    if ret_pct is None or adj is None or len(adj) < 10:
        return (None, None, 0) if return_info else (None, None)
    try:
        r = float(ret_pct) / 100.0
    except Exception:
        return (None, None, 0) if return_info else (None, None)
    now = float(adj.iloc[-1])
    if now <= 0:
        return (None, None, 0) if return_info else (None, None)
    err = now / adj.values - (1.0 + r)
    abs_err = np.abs(err)
    i_min = int(np.nanargmin(abs_err))
    min_err = float(abs_err[i_min])
    if min_err > tol_base:
        return (None, None, 0) if return_info else (None, None)
    # 候选 = 误差 ≤ min(最优误差+2pp, 3pp) 的日期，取其中**最晚**者：
    # 净值走平/多笔买入时存在多个同日收益率的日期，按最近一笔近似（对止损判断偏保守不误报）
    cand = np.where(abs_err <= min(min_err + 0.02, tol_base))[0]
    if len(cand) == 0:
        cand = np.array([i_min])
    i = int(cand.max())
    span = int((adj.index[int(cand.max())] - adj.index[int(cand.min())]).days) if len(cand) > 1 else 0
    if return_info:
        return adj.index[i], float(err[i]), span
    return adj.index[i], float(err[i])


# 反推入场日不可靠阈值：近优候选日期跨度超过该天数 → 判定"疑似多笔/定投/中途卖出"
# （此时只输入一个总市值+收益率的单日近似，入场高点回撤会系统性失真）
INFER_AMBIGUOUS_SPAN_DAYS = 60


# ================= 定投 (DCA) 生成与混合持仓反推 =================
DCA_FREQ_LABELS = {"daily": "每日", "monthly": "每月", "biweekly": "每两周", "weekly": "每周"}
_DCA_FREQ_DAYS = {"daily": 1, "monthly": None, "biweekly": 14, "weekly": 7}
# 反推时每月扣款日的候选日（覆盖发薪日/常见定投日；均 ≤28，避免大小月问题）
DCA_DAY_OF_MONTH_GRID = (1, 5, 10, 15, 20, 25, 28)


def dca_dates(start, freq="monthly", end=None):
    """定投扣款日序列（自然日，升序）。

    monthly: 每月与开始日同一天；该月无此日(如31号)取当月最后一天。
    end 缺省 = 今天；返回 ≥ start 且 ≤ end 的期次。
    """
    start = pd.Timestamp(str(start)).normalize()
    end = pd.Timestamp(str(end)).normalize() if end else pd.Timestamp.today().normalize()
    if start > end:
        return []
    if freq == "monthly":
        # 锚定"开始日所在月"的1号，保证首月期次不丢（如 1/31 开始 → 首期=1/31）
        anchor = start.to_period("M").start_time
        months = pd.date_range(anchor, end, freq="MS")
        out = []
        for m in months:
            d = m + pd.Timedelta(days=min(start.day, m.days_in_month) - 1)
            if start <= d <= end:          # 双向钳制：首期不早于开始日，末期不晚于结束日
                out.append(d.date())
        return out
    step = _DCA_FREQ_DAYS.get(freq, 7)     # daily=1 / weekly=7 / biweekly=14
    return [d.date() for d in pd.date_range(start, end, freq=f"{step}D")]


def dca_lots(start, amount, freq="monthly", end=None, adj=None, include_pending=False):
    """生成定投买入记录 [(date_str, 'buy', amount), ...]。

    adj 提供时自动跳过休市：
      - 已公布净值的历史区间只保留真实净值日（周末/法定休市自动排除）；
      - ``include_pending=True`` 时，最新净值日之后至 ``end`` 的工作日也保留为
        **待确认扣款**。这对 QDII 尤其重要：其净值通常滞后 1～2 天，不能因为今日
        净值尚未发布，就把用户今日已经发生的定投从台账中漏掉。

    include_pending 默认关闭以保持历史回测/反推的严格净值日口径；台账生成与“补齐至
    今日”应显式开启。待确认区间只能可靠排除周末，之后净值发布时诊断会自动使用真实
    确认净值；若平台因特殊休市未实际扣款，用户仍可在台账删除该笔。
    """
    amount = float(amount)
    if amount <= 0:
        return []
    dates = dca_dates(start, freq, end)
    if adj is not None and len(adj):
        trading = set(adj.index.date)      # 有净值的日子 = 该基金真实交易日
        last_nav_date = max(trading)
        dates = [d for d in dates
                 if d in trading or (include_pending and d > last_nav_date and d.weekday() < 5)]
    return [(str(d), "buy", amount) for d in dates]


def infer_dca(manual_lots, total_mv, adj, freqs=("monthly", "biweekly", "weekly"),
              top_n=5, max_history_days=10 * 365):
    """混合持仓反推定投参数：已知当前总市值 + 主动买入/卖出记录 → 反推(频率, 开始日, 每期金额)。

    manual_lots: [(date, side, amount), ...]（定投之外的主动买卖，卖出为负贡献）。
    原理: 定投贡献市值 V_d = 总市值 - 主动记录贡献；对每个候选(频率, 开始日)，
    每期金额 a = V_d / Σ(now/adj(扣款日)) 可精确求解；再按金额整数度(百元整)、
    频率先验(每月>两周>每周)、期数打分排序 → 返回 top_n 候选。
    返回 dict: {ok, reason, candidates:[{freq, freq_label, start_date, amount,
                periods, total_invested, score}...]}
    """
    out = dict(ok=False, reason="", candidates=[])
    if total_mv is None or float(total_mv) <= 0:
        out["reason"] = "需要当前总市值（支付宝持仓页可查）"
        return out
    if adj is None or len(adj) < 60:
        out["reason"] = "净值历史不足，无法反推定投参数"
        return out
    now = float(adj.iloc[-1])
    if now <= 0:
        out["reason"] = "净值数据异常"
        return out
    idx = adj.index
    n = len(idx)

    def _mv(d, side, amt):
        pos = int(min(max(int(idx.searchsorted(pd.Timestamp(d))), 0), n - 1))
        return (amt if side == "buy" else -amt) * now / float(adj.iloc[pos])

    v_manual = sum(_mv(d, s, a) for d, s, a in manual_lots if a and a > 0)
    v_dca = float(total_mv) - v_manual
    if v_dca < -max(1.0, 0.01 * float(total_mv)):
        out["reason"] = (f"主动买卖按净值折算的市值约 {v_manual:,.0f} 元已超过总市值 "
                         f"{float(total_mv):,.0f} 元，无法反推定投（请核对主动买入记录或市值）")
        return out
    if v_dca <= 0:
        out["reason"] = "总市值≈主动买卖市值，未检测到定投贡献，无需反推"
        return out
    lo_date = max(idx[0].date(), (pd.Timestamp.today() - pd.Timedelta(days=max_history_days)).date())
    hi_date = idx[-1].date()
    if lo_date > hi_date:
        out["reason"] = "净值历史不足"
        return out

    freq_prior = {"monthly": 1.0, "biweekly": 0.85, "weekly": 0.70}
    common_amounts = (500, 1000, 1500, 2000, 3000, 5000, 8000, 10000)
    cands = []
    for freq in freqs:
        if freq == "monthly":
            starts = []
            for m in pd.date_range(pd.Timestamp(lo_date), pd.Timestamp(hi_date), freq="MS"):
                for day in DCA_DAY_OF_MONTH_GRID:
                    d = m + pd.Timedelta(days=min(day, m.days_in_month) - 1)
                    if d.date() >= lo_date:
                        starts.append(d)
        else:
            starts = pd.date_range(lo_date, hi_date, freq=f"{_DCA_FREQ_DAYS.get(freq, 7)}D")
        for s in starts:
            dates = dca_dates(s, freq, hi_date)
            if len(dates) < 2:
                continue
            pos = np.clip(idx.searchsorted(pd.DatetimeIndex(dates)), 0, n - 1)
            W = float(np.sum(now / adj.values[pos]))
            if W <= 0:
                continue
            a = v_dca / W
            if a < 10 or a > 500000:
                continue
            # 金额整数度: 离整百越近越可信（定投金额习惯整百）→ 整百=1.0, 半百=0.0
            frac = (a / 100.0) - np.floor(a / 100.0)
            round_score = 1.0 - 2.0 * min(frac, 1.0 - frac)
            score = round_score * freq_prior.get(freq, 1.0)
            if any(abs(a - c) / c < 0.01 for c in common_amounts):
                score += 0.05 * freq_prior.get(freq, 1.0)
            k = len(dates)
            if k < 3:
                score *= 0.8
            cands.append(dict(freq=freq, freq_label=DCA_FREQ_LABELS.get(freq, freq),
                              start_date=str(s.date()), amount=round(a, 2),
                              periods=k, total_invested=round(a * k, 2),
                              score=round(score, 4)))
    if not cands:
        out["reason"] = "未找到合理的定投参数组合（请核对市值与主动记录）"
        return out
    cands.sort(key=lambda c: -c["score"])
    dedup = []
    for c in cands:
        dup = False
        for d0 in dedup:
            if (c["freq"] == d0["freq"] and abs(c["amount"] - d0["amount"]) / d0["amount"] < 0.02
                    and abs((pd.Timestamp(c["start_date"]) - pd.Timestamp(d0["start_date"])).days) <= 31):
                dup = True
                break
        if not dup:
            dedup.append(c)
        if len(dedup) >= top_n:
            break
    out.update(ok=True, candidates=dedup)
    return out


def exposure_overlap(cand_expo, port_expo, min_port_weight=0.15):
    """候选基金与(组合+已选)的暴露相似度 0~1。

    cand_expo: {"风格": 权重, ...}（候选基金 RBSA 暴露）
    port_expo: {"风格": 权重, ...}（组合加总暴露，或已选候选的暴露）
    相似度 = 候选暴露中落在"组合实质配置"风格上的权重占比：
      - 组合实质配置 = 权重 ≥ min_port_weight 的风格（防止分散型/宽基持仓把
        所有风格都"覆盖"一遍导致误杀一切候选）；
      - 候选侧按全部正权重计（权重越集中在被覆盖的风格上，相似度越高）。
    用于买入候选的"高度类似"排除（配合 exposure_dup 的两档规则使用）。
    """
    if not cand_expo or not port_expo:
        return 0.0
    cw = {}
    for k, v in cand_expo.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            cw[k] = fv
    if not cw:
        return 0.0
    meaningful = {}
    for k, v in port_expo.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv >= min_port_weight:
            meaningful[k] = fv
    if not meaningful:
        return 0.0
    covered = sum(v for k, v in cw.items() if k in meaningful)
    return covered / sum(cw.values())


def exposure_dup(cand_expo, port_expo, skip=0.70, index_top1=0.40, min_port_weight=0.15):
    """候选是否与(组合+已选)高度类似 → (是否重复, 重叠度, 理由)。

    两档规则（防误杀主动基金）：
      Tier1 指数级重复: 候选是单一指数/主题产品(top1风格权重≥index_top1)且该风格
        组合已实质持有(≥min_port_weight) → 重复（如 2 只纳指100、2 只白酒指数）。
      Tier2 近同质主动基金: 整体暴露重叠 ≥ skip(0.70) → 重复（RBSA 轮廓几乎相同）。
    返回的 overlap 即曝光给用户的重叠度。
    """
    if not cand_expo or not port_expo:
        return False, 0.0, ""
    cw = {k: float(v) for k, v in cand_expo.items() if v and float(v) > 0}
    if not cw:
        return False, 0.0, ""
    top1_style, top1_w = max(cw.items(), key=lambda kv: kv[1])
    if top1_w >= index_top1 and port_expo.get(top1_style, 0.0) >= min_port_weight:
        return True, round(top1_w, 3), f"同指数重复（{top1_style} 已持有）"
    ov = exposure_overlap(cw, port_expo, min_port_weight=min_port_weight)
    if ov >= skip:
        return True, round(ov, 3), "暴露高度重叠"
    return False, round(ov, 3), ""


def rbsa_l1_dist(a_expo, b_expo):
    """两条 RBSA 暴露向量的 L1 距离（Σ|a_k − b_k|，键取并集、缺省补 0）。

    用于"复制盘"识别：同一团队/同一策略的孪生产品，逐格暴露几乎一致
    （如 001801 汇添富达欣 vs 001417 汇添富医疗服务：L1≈0.003，
    近1年日收益相关 0.9997），而仅同风格的不同基金 L1 通常 ≥0.05。
    返回值恒为 float；任一输入为空返回 None（表示无法判定）。
    """
    if not a_expo or not b_expo:
        return None
    def _clean(expo):
        out = {}
        for k, v in (expo or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            out[k] = fv
        return out
    a, b = _clean(a_expo), _clean(b_expo)
    if not a or not b:
        return None
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def find_clone_exposure(cand_expo, ref_list, max_l1=0.02):
    """候选是否与参照池中的某只基金构成"复制盘"（暴露轮廓逐格几乎一致）。

    与 exposure_dup 的"聚合暴露重叠"思路互补：这里做逐对(pairwise)向量距离，
    专检同策略孪生产品——它们的 top1 往往未达指数级阈值(≥0.40)，
    聚合重叠度也不高(弥散型主动基)，但 16 因子暴露逐格相同。

    cand_expo: 候选基金 RBSA 暴露 dict
    ref_list:  [(code, name, expo), ...] 持仓 + 已选候选
    max_l1:    判定阈值（默认对齐 config.STRAT_CLONE_L1=0.02）
    返回: (ref_dict, l1) —— 命中的最近参照；未命中返回 (None, None)
    """
    best_ref, best_l1 = None, None
    for ref in ref_list or []:
        try:
            code, name, expo = ref
        except (TypeError, ValueError):
            continue
        l1 = rbsa_l1_dist(cand_expo, expo)
        if l1 is None:
            continue
        if best_l1 is None or l1 < best_l1:
            best_ref, best_l1 = {"code": code, "name": name}, l1
    if best_l1 is not None and best_l1 <= max_l1:
        return best_ref, best_l1
    return None, None


def fund_lots_diag(lots, nav_df, anchor_amount=None, stop=0.20, warn_dd=0.15, code=None,
                   buy_fee=None):
    """多笔买入/卖出 → 单基金持仓诊断（FIFO 成本 + 持仓市值曲线 + 入场高点回撤）。

    lots: [(date, side, amount), ...]  side ∈ {'buy','sell'}，amount 为金额（元）。
      买入金额 = 用户填写的**扣款总金额（含申购费）**，系统自动扣费后折份额：
        净申购 = 金额 × (1 - 申购费率)，份额 = 净申购 ÷ 当日净值（费率见 config）。
      卖出金额 = 卖出净到账（已扣赎回费，赎回费默认 0，见 config）。
    code: 基金代码（可选），用于查该基金的申购费率覆盖（BUY_FEE_BY_FUND）。
    buy_fee: 申购费率（可选）。给定时直接使用；缺省 None 时按 code 查 config。
    anchor_amount: 用户当前市值（可选）。份额×现值与用户市值存在小数差/费用差时，
      按 k=市值/计算值 整体缩放成本与曲线，保证"当前市值=用户所填"且持有收益率不变。
    返回 dict（字段与 fund_stop_diag 兼容的超集）：
      computable / status(ok|near|triggered|no_data|flat) / reason
      entry_date / entry_nav / days_held / peak / peak_date / dd / trigger_nav
      ret_held / mv_now(当前市值) / basis(FIFO剩余成本) / shares_now / over_sell
      total_bought / total_sold / lots_n / flat / curve(持仓市值曲线 Series)
    """
    adj = adj_series(nav_df)
    unit = unit_series(nav_df)
    navs = unit if len(unit) else adj
    out = dict(
        computable=False, status="no_data", reason="", entry_date=None, entry_nav=None,
        inferred=False, days_held=None, peak=None, peak_date=None, dd=None,
        trigger_nav=None, ret_held=None, mv_now=None, basis=None, shares_now=None,
        over_sell=False, total_bought=0.0, total_sold=0.0, lots_n=0,
        flat=False, curve=None, stop=float(stop),
        nav_asof=None, buy_fee=None, confirm_delay=0, pending_buy=0.0, pending_n=0,
    )
    if len(navs) < 30:
        out["reason"] = "净值历史不足，无法计算入场高点回撤"
        return out
    lots = sorted([(pd.Timestamp(d), str(side).lower(), float(a))
                   for d, side, a in lots if a is not None and a > 0])
    lots = [(d, s, a) for d, s, a in lots if s in ("buy", "sell")]
    # 当天及以后的单 = 待确认（支付宝「持有金额」不含在途），不计入当前市值
    today0 = pd.Timestamp(pd.Timestamp.today().date())
    pending = [(d, s, a) for d, s, a in lots if d.normalize() >= today0]
    lots = [(d, s, a) for d, s, a in lots if d.normalize() < today0]
    out["pending_n"] = len(pending)
    out["pending_buy"] = round(sum(a for d, s, a in pending if s == "buy"), 2)
    if not lots:
        out["status"] = "need_entry"
        out["reason"] = "缺少买入记录，无法定位入场高点（在操作台账中添加买入，或输入 代码 市值 买入日期）"
        if pending:
            out["reason"] = "仅有待确认交易（下单日≥今天），支付宝持有金额尚未计入"
        return out
    out["lots_n"] = len(lots)
    n = len(navs)
    idx = navs.index
    delay = 0

    def _pos(d):
        return confirm_nav_pos(idx, d, delay)

    fee = buy_fee_rate(code) if buy_fee is None else float(buy_fee)
    out["buy_fee"] = round(float(fee), 6)
    out["confirm_delay"] = 0
    out["nav_asof"] = str(idx[-1].date())
    deltas = np.zeros(n)
    run = 0.0
    for d, side, amt in lots:
        pxv = float(navs.iloc[_pos(d)])
        if pxv <= 0:
            continue
        if side == "buy":
            sh = amt / (1.0 + fee) / pxv
            deltas[_pos(d)] += sh
            run += sh
        else:
            sh = amt / pxv
            if sh > run + 1e-6:
                out["over_sell"] = True
            deltas[_pos(d)] -= sh
            run = max(0.0, run - sh)
    shares_t = np.maximum(np.cumsum(deltas), 0.0)

    q = []
    total_bought = total_sold = 0.0
    for d, side, amt in lots:
        pxv = float(navs.iloc[_pos(d)])
        if pxv <= 0:
            continue
        if side == "buy":
            net = amt / (1.0 + fee)
            q.append([net / pxv, net])
            total_bought += amt
        else:
            sh_sell = amt / pxv
            total_sold += amt
            while sh_sell > 1e-9 and q:
                take = min(q[0][0], sh_sell)
                q[0][1] *= (1.0 - take / q[0][0])
                q[0][0] -= take
                sh_sell -= take
                if q[0][0] <= 1e-9:
                    q.pop(0)
    shares_now = float(sum(x[0] for x in q))
    basis = float(sum(x[1] for x in q))
    if shares_now <= 1e-9:
        out.update(status="flat", computable=True, flat=True, shares_now=0.0,
                   basis=0.0, total_bought=total_bought, total_sold=total_sold,
                   reason="台账中该基金已全部卖出，不计入组合")
        return out
    mv = shares_now * float(navs.iloc[-1])
    k = 1.0
    if anchor_amount and anchor_amount > 0 and mv > 0:
        k = anchor_amount / mv
    curve = pd.Series(shares_t * navs.values * k, index=idx, name="value")
    curve[shares_t <= 1e-9] = np.nan
    basis *= k
    mv = float(curve.iloc[-1])
    first_pos = int(np.argmax(shares_t > 1e-9))
    peak_adj = float(navs.iloc[first_pos:].max())
    peak_adj_date = navs.iloc[first_pos:].idxmax()
    peak_val = float(curve.iloc[first_pos:].max())
    dd = float(mv / peak_val - 1.0)
    entry_date = idx[first_pos]
    entry_nav = float(navs.iloc[first_pos])
    ret_held = float(mv / basis - 1.0) if basis > 0 else None
    out.update(
        computable=True,
        status="ok",
        entry_date=str(entry_date.date()),
        entry_nav=round(entry_nav, 4),
        inferred=False,
        days_held=int((idx[-1] - entry_date).days),
        peak=round(peak_adj, 4),
        peak_date=str(peak_adj_date.date()),
        dd=float(dd),
        trigger_nav=round(peak_adj * (1.0 - stop), 4),
        ret_held=ret_held,
        mv_now=round(mv, 2),
        basis=round(basis, 2),
        shares_now=round(shares_now, 4),
        total_bought=round(total_bought, 2),
        total_sold=round(total_sold, 2),
        curve=curve,
    )
    if dd <= -stop:
        out["status"] = "triggered"      # 清仓
    elif dd <= -warn_dd:
        out["status"] = "near"           # 接近止损
    else:
        out["status"] = "ok"
    return out


def fund_stop_diag(rec, nav_df, stop=0.20, warn_dd=0.15):
    """单基金入场高点回撤 → 移动止损状态（单笔买入 = 多笔引擎的特例）。

    返回 dict：
      computable / status(ok|near|triggered|no_data|need_entry|flat)
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
        entry_date=None, entry_nav=None, inferred=False, infer_ambiguous=False,
        days_held=None, peak=None, peak_date=None, dd=None, trigger_nav=None,
        ret_held=None, ret_user=ui["ret_pct"], stop=float(stop), last_date=last_date,
        amount=ui["amount"], cost=ui["cost"],
    )
    if len(adj) < 30:
        out["reason"] = "净值历史不足，无法计算入场高点回撤"
        return out
    ambig_reason = None
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
        entry, infer_err, infer_span = infer_entry_date(adj, r_infer, return_info=True)
        if entry is None:
            out["status"] = "need_entry"
            out["reason"] = "持有收益率与净值曲线无法匹配（多笔买入/转换/申赎费差异），请在支付宝交易记录中查买入日期后填写"
            return out
        out["inferred"] = True
        # 多笔买入/定投/中途卖出：同一收益率在净值曲线上有多个相距很远的匹配日，
        # 单日近似会把"入场高点回撤"算歪（可能严重低估，止损形同虚设）→ 明确提示
        if infer_span > INFER_AMBIGUOUS_SPAN_DAYS:
            out["infer_ambiguous"] = True
            ambig_reason = (f"持有收益率在净值曲线上匹配到跨度约 {infer_span} 天的多个日期"
                            f"（疑似多笔买入/定投/中途卖出），入场日与回撤按最近一笔近似、"
                            f"仅供粗略参考；建议在操作台账记录每笔买卖（代码 买/卖 日期 金额）"
                            f"后重新诊断，可获得精确的 FIFO 成本与真实回撤")
        else:
            ambig_reason = None
    # 单笔买入 → 复用多笔引擎（曲线锚定到用户市值，成本=市值或成本）
    cost = ui["cost"] if (ui["cost"] or 0) > 0 else (ui["amount"] or 0)
    lots = [(pd.Timestamp(entry), "buy", cost)]
    d = fund_lots_diag(lots, nav_df, anchor_amount=ui["amount"] if (ui["amount"] or 0) > 0 else None,
                       stop=stop, warn_dd=warn_dd, code=rec.get("code"))
    out.update(d)
    out["inferred"] = bool(ui["buy_date"] is None and out.get("entry_date"))
    # fund_lots_diag 成功路径 reason="" 会覆盖反推警告 → 重新挂上
    if ambig_reason:
        out["reason"] = ambig_reason
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

    fund_series: 每个元素为一条持仓曲线，两种形式：
      - (curve,)：pd.Series 直接给出持仓市值曲线（多笔买卖台账用，index=日期，
        入场前为 NaN）；已锚定市值，不再缩放。
      - (entry_date, scale_value, adj_series)：单笔买入用，曲线按
        市值×adj(t)/adj(now) 重建（与 成本×adj(t)/adj(entry) 完全等价）。
    cash: 可用现金（按常数并入组合净值，近似处理）
    返回 dict（computable=False 时仅 reason）：
      hwm / current / dd / slots / tier_name / full_slots
      events: 历史触发/回补事件（含日期、类型）
      next_trigger: 当前档位下一触发线(净值与金额)
      restore: 当前档位回补线（回升到多少恢复多少槽，含金额与最后触及日）
      chart: 抽样后的 {dates, value, hwm} 供前端画曲线
    """
    curves = []
    for ent in fund_series:
        try:
            if isinstance(ent, (tuple, list)) and len(ent) == 1 and isinstance(ent[0], pd.Series):
                c = ent[0]
                if c is None or len(c) < 5 or float(c.iloc[-1]) <= 0:
                    continue
                curves.append(c)
            elif isinstance(ent, (tuple, list)) and len(ent) == 3:
                entry, scale, adj = ent
                if entry is None or not scale or scale <= 0 or adj is None or len(adj) < 5:
                    continue
                entry = pd.Timestamp(entry)
                if float(adj.iloc[-1]) <= 0 or float(adj.iloc[0]) <= 0:
                    continue
                now_adj = float(adj.iloc[-1])
                c = pd.Series(adj.values / now_adj * float(scale), index=adj.index, name="value")
                c[adj.index < entry] = np.nan
                curves.append(c)
        except Exception:
            continue
    if not curves:
        return dict(computable=False,
                    reason="持仓缺少 市值+买入日期（或 成本/收益率），无法重建组合净值曲线（提供后自动计算真实回撤与槽位）")
    # ---- 自适应起点：自动检测最早买入时间 ----
    # 基金复权净值从成立日起就有数据，持仓曲线在买入前为 0/NaN；
    # 取所有曲线中第一个有效值(>0)日期的最早者作为起点，裁掉前面全 0 的前缀段
    starts = []
    for c in curves:
        arr = np.asarray(c.values, dtype=float)
        v = np.where(np.isfinite(arr) & (arr > 0))[0]
        if len(v):
            starts.append(c.index[v[0]])
    start = min(starts) if starts else curves[0].index[0]
    # 并集交易日（从最早买入日起）
    idx = curves[0].index[curves[0].index >= start]
    for c in curves[1:]:
        idx = idx.union(c.index[c.index >= start])
    idx = pd.DatetimeIndex(sorted(idx))
    contrib = np.zeros(len(idx))
    for c in curves:
        # 基金自身日历内的 NaN = 入场前/空仓 → 市值 0（结构性空仓）
        own = c.fillna(0.0)
        # 并集日历上该基金缺净值的日期（如 QDII 与 A 股披露时差、节假日错位）
        # → 沿用最近一期净值(forward-fill)，绝不能按 0 计（否则组合值瞬间塌陷，
        #   会伪造出巨幅回撤、连续击穿 CPPI 触发线的假信号）
        vals = own.reindex(idx).ffill().fillna(0.0).values.astype(float)
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
        n_funds=len(curves),
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
