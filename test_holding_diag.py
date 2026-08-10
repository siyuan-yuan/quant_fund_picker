# -*- coding: utf-8 -*-
"""holding_diag 回归测试：入场高点回撤止损 / CPPI 状态机 / 组合曲线重建。

纯本地合成数据（不联网）；另有 webapp /api/rebalance 端到端用例（需 cache/ 净值文件）。
运行: python test_holding_diag.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holding_diag as hd  # noqa: E402


def mk_nav(dates, rets):
    """按 日增长率 构造 provider.get_fund_nav 同构 DataFrame"""
    nav = [1.0]
    for r in rets[1:]:
        nav.append(nav[-1] * (1 + r))
    return pd.DataFrame({"date": pd.to_datetime(dates), "nav": nav, "ret": rets})


def test_adj_series():
    df = mk_nav(["2021-01-01", "2021-01-04", "2021-01-05"], [0.0, -0.10, 0.05])
    adj = hd.adj_series(df)
    assert abs(adj.iloc[-1] - 1.0 * 0.9 * 1.05) < 1e-9
    print("test_adj_series OK")


def test_infer_entry_date():
    # 净值: 1.0 → 1.5(高点) → 1.2 → 0.9；当前0.9。持有收益率 -40%（自1.5起）
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    adj = pd.Series(np.linspace(1.0, 1.5, 10).tolist() + np.linspace(1.5, 0.9, 10).tolist(),
                    index=dates)
    d, err = hd.infer_entry_date(adj, -40.0)
    assert d is not None and abs(err) < 0.02
    # 收益率无法匹配 → None
    d2, _ = hd.infer_entry_date(adj, 999.0)
    assert d2 is None
    print("test_infer_entry_date OK, inferred:", d.date())


def test_fund_stop_diag():
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    # 前60日爬升到2.0，后60日跌到1.2 → 自高点回撤 -40%
    vals = np.linspace(1.0, 2.0, 60).tolist() + np.linspace(2.0, 1.2, 60).tolist()
    rets = [0.0] + [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
    df = mk_nav([str(d.date()) for d in dates], rets)

    # 给出买入日期（高点2.0在买入日之后）→ 精确计算
    d = hd.fund_stop_diag(dict(code="X", amount=10000, buy_date="2020-02-01"), df)
    assert d["computable"] and d["status"] == "triggered"
    assert d["entry_date"] == "2020-02-01" and not d["inferred"]
    assert abs(d["dd"] - (1.2 / 2.0 - 1)) < 1e-6
    assert abs(d["trigger_nav"] - 2.0 * 0.8) < 1e-6

    # 只给收益率 → 推断入场日
    d2 = hd.fund_stop_diag(dict(code="X", amount=10000, ret_pct=-40.0), df)
    assert d2["computable"] and d2["inferred"] and d2["status"] == "triggered"

    # 市值+成本 → 隐含收益率 → 推断入场
    d3 = hd.fund_stop_diag(dict(code="X", amount=10000, cost=16666.0), df)
    assert d3["computable"] and d3["inferred"]

    # 什么都不给 → 不误报
    d4 = hd.fund_stop_diag(dict(code="X", amount=10000), df)
    assert d4["status"] == "need_entry" and not d4["computable"]

    # 买入日期=近期低点 → 正常（2020-04-28 之后净值≈1.2 走平）
    d5 = hd.fund_stop_diag(dict(code="X", amount=10000, buy_date="2020-04-28"), df)
    assert d5["computable"] and d5["status"] == "ok"
    assert abs(d5["dd"]) < 0.03
    print("test_fund_stop_diag OK")


def test_infer_ambiguous():
    """多笔买入+中途卖出：只输总市值+总收益率 → 反推的单日入场不可靠 → infer_ambiguous 告警。"""
    # 净值: 爬升至1.5见顶 → 回落到1.19（与真实案例同构）
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    vals = np.linspace(1.0, 1.5, 100).tolist() + np.linspace(1.5, 1.19, 100).tolist()
    rets = [0.0] + [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
    df = mk_nav([str(d.date()) for d in dates], rets)

    # 真实场景: 两笔买入 + 一笔部分卖出 → FIFO 真实持有收益/市值
    lots = [("2024-02-19", "buy", 10000.0), ("2024-05-29", "buy", 10000.0),
            ("2024-06-08", "sell", 5000.0)]
    true = hd.fund_lots_diag(lots, df)
    assert true["computable"]

    # 只输 总市值+总收益率 → 必须给出"不可靠"告警（单日近似会严重低估真实回撤）
    d = hd.fund_stop_diag(dict(code="X", amount=true["mv_now"],
                               ret_pct=round(true["ret_held"] * 100, 2)), df)
    assert d["computable"] and d["inferred"] and d["infer_ambiguous"], d
    assert "台账" in (d.get("reason") or "")

    # 提供明确买入日期 → 不告警（用户自查过交易记录）
    d2 = hd.fund_stop_diag(dict(code="X", amount=true["mv_now"], buy_date="2024-02-19"), df)
    assert d2["computable"] and not d2["infer_ambiguous"]

    # return_info 向后兼容: 默认二元组, 显式开启才返回三元素
    adj = hd.adj_series(df)
    r2 = hd.infer_entry_date(adj, -5.6)
    assert isinstance(r2, tuple) and len(r2) == 2
    r3 = hd.infer_entry_date(adj, -5.6, return_info=True)
    assert len(r3) == 3 and r3[2] > hd.INFER_AMBIGUOUS_SPAN_DAYS
    print("test_infer_ambiguous OK")


def test_dca_dates_lots_infer():
    """定投序列生成（月末钳制/周频）+ 混合持仓反推定投参数。"""
    # 净值: 先涨后跌（与其它用例同构）
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    vals = np.linspace(1.0, 1.5, 100).tolist() + np.linspace(1.5, 1.19, 100).tolist()
    rets = [0.0] + [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
    df = mk_nav([str(d.date()) for d in dates], rets)
    adj = hd.adj_series(df)

    # 月末31号定投 → 无31号的月份自动取月末（2月29 / 4月30）
    d2 = hd.dca_dates("2024-01-31", "monthly", "2024-04-30")
    assert [str(d) for d in d2] == ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"], d2
    # 末期钳制: 结束日早于当月扣款日 → 该期不生成（防超界多一期）
    d5 = hd.dca_dates("2024-03-10", "monthly", "2024-05-06")
    assert [str(d) for d in d5] == ["2024-03-10", "2024-04-10"], d5
    # 周频: 2024-03-01(周五) 起 5 期
    d3 = hd.dca_dates("2024-03-01", "weekly", "2024-03-31")
    assert len(d3) == 5 and str(d3[0]) == "2024-03-01"
    # 每日定投: 连续自然日
    d6 = hd.dca_dates("2024-03-01", "daily", "2024-03-05")
    assert [str(d) for d in d6] == ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"], d6
    # 两周期
    d4 = hd.dca_dates("2024-03-01", "biweekly", "2024-04-30")
    assert len(d4) == 5
    # 生成记录 + 净值范围裁剪
    lots = hd.dca_lots("2024-02-15", 1000, "monthly", adj=adj)
    assert all(lt[1] == "buy" and lt[2] == 1000.0 for lt in lots)
    assert lots[0][0] >= str(adj.index[0].date())
    # 混合持仓: 每月1号定投1000(2024-03-01起) + 主动买入一笔10000 → 反推应还原 ≈1000/月
    dca = hd.dca_lots("2024-03-01", 1000, "monthly", adj=adj)
    manual = [("2024-04-10", "buy", 10000.0)]
    mv = hd.fund_lots_diag(dca + manual, df)["mv_now"]
    r = hd.infer_dca(manual, mv, adj, freqs=("monthly",))
    assert r["ok"] and r["candidates"], r
    top = r["candidates"][0]
    assert top["freq"] == "monthly"
    assert abs(top["amount"] - 1000) / 1000 < 0.10, top
    assert abs((pd.Timestamp(top["start_date"]) - pd.Timestamp("2024-03-01")).days) <= 35, top
    # 主动买卖市值超过总市值 → 明确失败原因，不瞎给结果
    r2 = hd.infer_dca([("2024-04-10", "buy", 10000.0)], 1000.0, adj, freqs=("monthly",))
    assert not r2["ok"] and "超过" in r2["reason"]
    print("test_dca_dates_lots_infer OK")


def test_exposure_overlap():
    """RBSA 暴露重叠度：用于买入候选的重复度过滤（与组合/已选重叠≥60%判为类似）。"""
    nasdaq = {"纳斯达克100": 0.55, "标普500": 0.25, "恒生科技": 0.20}
    a_tech = {"沪深300": 0.3, "中证1000": 0.2, "全指信息": 0.5}
    port_nasdaq = {"纳斯达克100": 0.6, "标普500": 0.3}   # 已持有纳指ETF等
    # 纳指候选 vs 纳指组合 → 高重叠（应跳过）
    assert hd.exposure_overlap(nasdaq, port_nasdaq) >= 0.8
    # A股科技 vs 纳指组合 → 低重叠（不跳过）
    assert hd.exposure_overlap(a_tech, port_nasdaq) < 0.05
    # 空暴露 → 0
    assert hd.exposure_overlap({}, port_nasdaq) == 0.0
    assert hd.exposure_overlap(nasdaq, {}) == 0.0
    # 组合暴露=0.55纳指（部分覆盖候选的0.55）→ 覆盖度=0.55/1.0=0.55 <0.6 不跳过
    partial = {"纳斯达克100": 0.55}
    ov = hd.exposure_overlap(nasdaq, partial)
    assert abs(ov - 0.55) < 1e-9, ov
    # 分散型持仓不误杀: 组合全风格小额暴露(实质配置=仅≥15%的风格) → 医疗候选不被判重复
    diversified = {"沪深300": 0.20, "中证500": 0.13, "全指信息": 0.11,
                   "全指消费": 0.10, "全指医药": 0.09, "中证1000": 0.08}
    medical = {"全指医药": 0.60, "全指消费": 0.20, "创业板50": 0.10, "沪深300": 0.05}
    assert hd.exposure_overlap(medical, diversified) < 0.10, "分散持仓不应挡住医疗候选"
    # 但组合实质持有医药(≥15%) → 医疗候选判重复
    med_port = {"全指医药": 0.45, "全指消费": 0.20, "沪深300": 0.15}
    assert hd.exposure_overlap(medical, med_port) >= 0.8
    # 两档规则: 指数级重复(候选top1≥40%且组合已实质持有该风格) → 必排除
    nasdaq2 = {"纳斯达克100": 0.85, "标普500": 0.10, "恒生科技": 0.05}
    is_dup, ov, reason = hd.exposure_dup(nasdaq2, port_nasdaq)
    assert is_dup and "同指数" in reason and ov >= 0.8
    # 主动基金近同质: 整体暴露重叠≥0.70 → 排除; 不同风格主动基金不排除
    tech_a = {"全指信息": 0.30, "创业板50": 0.25, "中证500": 0.20, "中证1000": 0.15, "沪深300": 0.10}
    tech_b = {"全指信息": 0.32, "创业板50": 0.27, "中证500": 0.18, "中证1000": 0.13, "沪深300": 0.10}
    is_dup2, ov2, _ = hd.exposure_dup(tech_b, tech_a)
    assert is_dup2 and ov2 >= 0.9, (is_dup2, ov2)
    is_dup3, ov3, _ = hd.exposure_dup(medical, tech_a)
    assert not is_dup3 and ov3 < 0.15
    print("test_exposure_overlap OK")


def test_cppi_tier_sim():
    # 全路径: 触发-15(6槽) → 触发-20(3槽) → 回补-18(6槽) → 新高(满槽)
    v = np.array([100.0, 88.0, 90.0, 84.0, 86.0, 78.0, 80.0, 81.0, 82.0, 83.0, 84.0, 86.0, 100.5])
    ev, slots, dd, hwm = hd.cppi_tier_sim(v, [(-0.15, 6), (-0.20, 3), (-0.25, 0)], 10, 0.02)
    assert slots == 10 and hwm == 100.5
    assert [e[1] for e in ev] == ["trigger", "trigger", "restore", "newhigh"], ev
    assert abs(ev[2][2] - (-0.18)) < 1e-9 and ev[2][3] == 6

    # 触发-25(清仓) → 逐级回补 -23(3槽) → -18(6槽) → -13(满槽)
    # （首日单日 -26% 会级联触发三档，属正常；随后逐级回补）
    v2 = np.array([100.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0])
    ev2, slots2, _, _ = hd.cppi_tier_sim(v2, [(-0.15, 6), (-0.20, 3), (-0.25, 0)], 10, 0.02)
    assert slots2 == 10
    tr2 = [e for e in ev2 if e[1] == "trigger"]
    rs2 = [e for e in ev2 if e[1] == "restore"]
    assert tr2[-1][2] == -0.25 and tr2[-1][3] == 0
    assert [round(e[2], 2) for e in rs2] == [-0.23, -0.18, -0.13]

    # 滞回: 回撤收窄但未过回补线 → 不升档
    v3 = np.array([100.0, 84.0, 87.0])
    ev3, slots3, _, _ = hd.cppi_tier_sim(v3, [(-0.15, 6), (-0.20, 3), (-0.25, 0)], 10, 0.02)
    assert slots3 == 6 and [e[1] for e in ev3] == ["trigger"]
    print("test_cppi_tier_sim OK")


def test_portfolio_cppi():
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    idx = pd.DatetimeIndex(dates)
    # 基金A: 先涨后跌（2021-02-01入场，之后高点=2.0 → 现值1.2）
    a = pd.Series(np.linspace(1.0, 2.0, 60).tolist() + np.linspace(2.0, 1.2, 40).tolist(), index=idx)
    # 基金B: 稳步上涨（2021-01-15入场，现值为1.8）
    b = pd.Series(np.linspace(1.0, 1.8, 100), index=idx)
    r = hd.portfolio_cppi([("2021-02-01", 10000.0, a), ("2021-01-15", 20000.0, b)],
                          cash=5000.0, rules=[(-0.15, 6), (-0.20, 3), (-0.25, 0)],
                          full_slots=10, hysteresis=0.02)
    assert r["computable"] and r["n_funds"] == 2
    assert r["current"] == r["chart"]["value"][-1]
    assert abs(r["current"] - (10000 * 1.2 / 1.2 + 20000 * 1.8 / 1.8 + 5000)) < 1e-6
    assert r["hwm"] > r["current"] and r["dd"] < 0
    assert 0 <= r["slots"] <= 10
    assert r["restore"] is None or r["restore"]["slots"] > r["slots"]
    # 无有效基金 → computable=False
    r2 = hd.portfolio_cppi([(None, 10000.0, a)])
    assert not r2["computable"]
    print("test_portfolio_cppi OK, dd=%.1f%% slots=%d hwm=%.0f" % (r["dd"] * 100, r["slots"], r["hwm"]))


def test_fund_lots_diag():
    # 合成净值（40天，平台+跳变保证数学干净）:
    #   d0..d9 = 1.0 | d10 = 1.5 | d11..d29 = 1.5 | d30 = 1.2 | d31..d39 = 1.2
    n = 40
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    vals = np.array([1.0] * 10 + [1.5] * 20 + [1.2] * 10)
    rets = [0.0] + [vals[i] / vals[i - 1] - 1 for i in range(1, n)]
    df = mk_nav([str(d.date()) for d in idx], rets)
    d10, d30 = str(idx[10].date()), str(idx[30].date())
    # 两笔买入 + 一笔部分卖出（FIFO）
    lots = [
        ("2021-01-01", "buy", 10000.0),   # 10000 股 @1.0
        (d10, "buy", 15000.0),             # 10000 股 @1.5
        (d30, "sell", 1200.0),             # 1000 股 @1.2 → FIFO 从第一笔扣
    ]
    d = hd.fund_lots_diag(lots, df)
    assert d["computable"] and d["status"] == "triggered"
    assert d["entry_date"] == "2021-01-01"
    assert abs(d["mv_now"] - 22800.0) < 1e-6          # 19000股 × 1.2
    assert abs(d["basis"] - 24000.0) < 1e-6           # 9000 + 15000（FIFO扣1000股成本）
    assert abs(d["shares_now"] - 19000.0) < 1e-6
    assert abs(d["ret_held"] - (22800 / 24000 - 1)) < 1e-9
    assert abs(d["dd"] - (22800 / 30000 - 1)) < 1e-9  # 持仓曲线高点=20000股×1.5 @d10
    assert abs(d["trigger_nav"] - 1.5 * 0.8) < 1e-9   # 净值口径触发价
    assert d["peak_date"] == str(idx[10].date()) and d["peak"] == 1.5
    assert d["lots_n"] == 3 and not d["over_sell"] and not d["flat"]
    assert d["curve"] is not None and abs(d["curve"].iloc[-1] - 22800.0) < 1e-6
    assert d["curve"].max() == 30000.0
    # 锚定市值：整体缩放，收益率不变
    d2 = hd.fund_lots_diag(lots, df, anchor_amount=18240.0)
    assert abs(d2["mv_now"] - 18240.0) < 1e-6
    assert abs(d2["basis"] - 19200.0) < 1e-6
    assert abs(d2["ret_held"] - (22800 / 24000 - 1)) < 1e-9
    # 卖出超过持有 → over_sell + flat
    d3 = hd.fund_lots_diag([("2021-01-01", "buy", 10000.0), (d10, "sell", 50000.0)], df)
    assert d3["over_sell"] and d3["flat"] and d3["status"] == "flat"
    # 全部卖完 → flat
    d4 = hd.fund_lots_diag([("2021-01-01", "buy", 10000.0), (d30, "sell", 15000.0)], df)
    assert d4["flat"]
    # 空记录 → need_entry
    d5 = hd.fund_lots_diag([], df)
    assert d5["status"] == "need_entry"
    print("test_fund_lots_diag OK")


def test_portfolio_cppi_curve_input():
    # (curve,) 直接传持仓市值曲线（多笔买卖台账口径）
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    # 场景1: A 从 10000 跌到 2500（-75%），B 平稳 20000 → 组合 -21.4% → 3 槽
    c1 = pd.Series(np.linspace(10000, 2500, 10), index=idx)
    c2 = pd.Series(np.full(10, 20000.0), index=idx)
    r = hd.portfolio_cppi([(c1,), (c2,)], cash=5000.0,
                          rules=[(-0.15, 6), (-0.20, 3), (-0.25, 0)], full_slots=10)
    assert r["computable"] and r["n_funds"] == 2
    assert r["hwm"] == 35000.0
    assert abs(r["current"] - 27500.0) < 1e-6
    assert abs(r["dd"] - (27500 / 35000 - 1)) < 1e-9
    assert r["slots"] == 3 and r["tier_name"] == "防御档·深度减槽"
    # 场景2: 跌更深 → -25.7% → 清仓档
    c1b = pd.Series(np.linspace(10000, 1000, 10), index=idx)
    r2 = hd.portfolio_cppi([(c1b,), (c2,)], cash=5000.0,
                           rules=[(-0.15, 6), (-0.20, 3), (-0.25, 0)], full_slots=10)
    assert r2["slots"] == 0 and r2["tier_name"] == "清仓档·禁权益"
    # 场景3: 横盘微跌 → 满槽
    r3 = hd.portfolio_cppi([(pd.Series(np.linspace(10000, 9500, 10), index=idx),)],
                           cash=0.0, rules=[(-0.15, 6), (-0.20, 3), (-0.25, 0)], full_slots=10)
    assert r3["slots"] == 10
    print("test_portfolio_cppi_curve_input OK, dd=%.1f%% slots=%d | %s" % (r["dd"] * 100, r["slots"], r["tier_name"]))


def test_portfolio_cppi_auto_start():
    """组合曲线自适应起点：最早买入日前全 0/NaN 的前缀段应被裁掉"""
    n = 60
    idx = pd.date_range("2008-01-01", periods=n, freq="D")
    # 基金A: 前 20 天未持有（0/NaN），从第 20 天起有值（2021 年式的“后入场”）
    a = pd.Series([np.nan] * 20 + np.linspace(10000, 8000, 40).tolist(), index=idx)
    # 基金B: 前 40 天未持有，从第 40 天起有值
    b = pd.Series([np.nan] * 40 + np.linspace(20000, 24000, 20).tolist(), index=idx)
    r = hd.portfolio_cppi([(a,), (b,)], cash=0.0,
                          rules=[(-0.15, 6), (-0.20, 3), (-0.25, 0)], full_slots=10)
    assert r["computable"] and r["n_funds"] == 2
    # 起点 = 最早入场日（A 的第 20 天），不是 2008 年的第 0 天
    assert r["chart"]["dates"][0] == str(idx[20].date()), r["chart"]["dates"][0]
    assert len(r["chart"]["dates"]) == n - 20
    # 组合值 = A + B（B 在 A 入场后仍有 20 天为 0，属真实历史，保留）
    assert abs(r["current"] - (8000 + 24000)) < 1e-6
    # 全部未入场 → 起点兜底为第一只曲线的起点
    r2 = hd.portfolio_cppi([(pd.Series([np.nan] * 10, index=idx[:10]),)], cash=0.0)
    assert not r2["computable"] or r2["chart"]["dates"][0] == str(idx[0].date())
    print("test_portfolio_cppi_auto_start OK, start=%s → end=%s" % (r["chart"]["dates"][0], r["chart"]["dates"][-1]))


def test_webapp_endpoint():
    """端到端: /api/rebalance 解析 代码 市值 买入日期|成本|收益率 并输出 stop/cppi（需 cache/ 净值）"""
    if not os.path.exists("cache/nav_161725.csv"):
        print("test_webapp_endpoint SKIP (no cache)")
        return
    import warnings
    warnings.filterwarnings("ignore")
    import webapp
    c = webapp.app.test_client()
    c.delete("/api/ledger")   # 隔离：清空台账，避免污染断言
    r = c.post("/api/rebalance", json={
        "total_capital": "10万", "cash": "20000",
        "holdings_text": "161725 2.5万 2021-06-01\n110011 1.8万 2023-01-03\n005827 1.2万 +15.3%",
    })
    j = r.get_json()
    assert r.status_code == 200 and j["ok"]
    stops = [h["stop"] for h in j["holdings"]]
    assert all(s and s.get("computable") for s in stops), stops
    assert j["cppi"]["computable"] and j["cppi"]["n_funds"] == 3
    assert "slots_eff" in j["summary"]
    # 旧格式兼容
    r2 = c.post("/api/rebalance", json={"holdings_text": "161725 2.5万"})
    j2 = r2.get_json()
    assert j2["ok"] and j2["cppi"]["computable"] is False
    print("test_webapp_endpoint OK")


if __name__ == "__main__":
    for fn in [test_adj_series, test_infer_entry_date, test_infer_ambiguous, test_fund_stop_diag, test_fund_lots_diag, test_portfolio_cppi_curve_input, test_portfolio_cppi_auto_start,
               test_dca_dates_lots_infer, test_exposure_overlap, test_cppi_tier_sim, test_portfolio_cppi, test_webapp_endpoint]:
        fn()
    print("\nALL TESTS PASSED")
