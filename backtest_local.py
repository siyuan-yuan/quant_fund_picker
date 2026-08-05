# -*- coding: utf-8 -*-
"""
本地自助回测 —— 你只管敲命令，它帮你出三本账
=================================================================
用法(在 quant_fund_picker 目录下):
    python backtest_local.py                          # 默认: 2019Q1→最近完整季, 70买45卖
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
诚实边界: 默认使用全市场池（scan_market.py 生成的 rank_all.csv）; 回测豁免任期惩罚
=================================================================
"""
import os, sys, argparse, hashlib, time
import numpy as np
import pandas as pd

import provider
import rbsa, factors
from engine import score_fund, finalize
from config import (STRAT_BUY_TH, STRAT_SELL_TH, STRAT_SLOTS,
                    STRAT_HI_WATER, STRAT_HI_WATER_SLOTS)

CACHE_DIR = "output/bt_scores_cache"
FACTOR_ROWS = "output/factor_rows"


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


def harvest_date(d, codes):
    """对一个季点现打分(慢, 打一次即缓存)"""
    t0 = time.time()
    rows = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(score_fund, c, as_of=d, bt=True): c for c in codes}
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
    print(f"  [harvest] {d} 打分 {len(df)} 只 ({time.time()-t0:.0f}s)", flush=True)
    return df


def build_panel(dates, codes, default_universe, rebuild=False):
    """返回 DataFrame: date × code 的原始量+S; 逐季点缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    uid = "" if default_universe else "_" + hashlib.md5(",".join(sorted(codes)).encode()).hexdigest()[:8]
    parts = []
    for d in dates:
        fr = f"{FACTOR_ROWS}/{d}.csv"
        ck = f"{CACHE_DIR}/{d}{uid}.csv"
        if default_universe and os.path.exists(fr):          # 研究池已有原料行
            g = pd.read_csv(fr, dtype={"code": str})
            g = g[g.code.isin(codes)].copy()
        elif (not rebuild) and os.path.exists(ck):
            g = pd.read_csv(ck, dtype={"code": str})
        else:
            g = harvest_date(d, codes)
            if not g.empty:
                g.to_csv(ck, index=False, encoding="utf-8-sig")
        if g.empty:
            print(f"  [warn] {d} 无评分数据, 跳过")
            continue
        g = score_from_raw(g)
        g["date"] = d
        parts.append(g[["date", "code", "S", "water", "R_MDD"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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


# ============ 3. 逐日模拟(金额记账) ============
def simulate(panel, navs, bench, dates, args):
    by_date = {d: g.set_index("code") for d, g in panel.groupby("date")}
    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    day_grid = bench.loc[T0:T1].index

    cash, positions, trades, curve = float(args.capital), {}, [], []

    def sell(code, day, reason, exit_s=np.nan, price=None):
        nonlocal cash
        p = positions.pop(code)
        price = price if price is not None else px(navs[code], day)
        net_out = p["units"] * price * (1 - args.cost_out)
        cash += net_out
        ret = net_out / p["alloc"] - 1
        trades.append(dict(
            code=code, entry_date=p["entry_date"], exit_date=str(pd.Timestamp(day).date()),
            entry_px=round(p["entry_px"], 4), exit_px=round(price, 4),
            entry_S=round(p["entry_S"], 1), exit_S=round(exit_s, 1) if pd.notna(exit_s) else np.nan,
            hold_days=(pd.Timestamp(day) - pd.Timestamp(p["entry_date"])).days,
            net_ret=round(ret, 4), pnl_yuan=round(net_out - p["alloc"], 2)))

    for day in day_grid:
        day_str = str(day.date())

        # ========== 每天检查卖出 ==========
        for c in list(positions):
            if c not in by_date.get(day_str, pd.DataFrame()).index:
                continue
            score = by_date[day_str].loc[c, "S"]
            if pd.notna(score) and score < args.sell:
                sell(c, day, f"S<{args.sell:.0f}", exit_s=score)

        # ========== 每天检查买入（如果当天有打分）==========
        if day_str in by_date:
            g = by_date[day_str]
            equity_now = cash + sum(p["units"] * px(navs[c], day) for c, p in positions.items())

            for c, row in g[g.S > args.buy].sort_values("S", ascending=False).iterrows():
                if c in positions or len(positions) >= args.slots or c not in navs:
                    continue
                price = px(navs[c], day)
                if pd.isna(price):
                    continue
                alloc = min(cash, equity_now / args.slots)
                if alloc < equity_now * 0.02:
                    continue
                positions[c] = dict(units=alloc / (price * (1 + args.cost_in)),
                                    entry_px=price, entry_date=day_str,
                                    entry_S=row.S, alloc=alloc)
                cash -= alloc

        eq = cash + sum(p["units"] * px(navs[c], day) for c, p in positions.items())
        curve.append((day, eq, len(positions)))

    # 期末清仓
    for c in list(positions):
        sell(c, T1, "期末清算")

    ec = pd.DataFrame(curve, columns=["date", "equity", "n_pos"]).set_index("date")
    ec["drawdown"] = ec.equity / ec.equity.cummax() - 1
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
    S.append(f"# 本地回测报告  区间 {dates[0]} → {dates[-1]}  tag={tag}\n")
    S.append(f"参数: 买 S>{args.buy:.0f} / 卖 S<{args.sell:.0f} / 槽位 {args.slots} / 本金 {args.capital:,.0f} 元 / "
             f"成本 申购{args.cost_in:.2%}+赎回{args.cost_out:.2%} / 高位瘦身 {'开' if getattr(args, 'hi_water', False) else '关'}\\n")
    S.append("## 总账")
    S.append(f"| 指标 | 数值 |\n|---|---|")
    S.append(f"| 期末资产 | **{ec.equity.iloc[-1]:,.0f} 元** |")
    S.append(f"| 组合总收益 | **{tot:+.1%}** ({args.capital * tot:+,.0f} 元) |")
    S.append(f"| 年化 CAGR | {cagr:+.2%} |")
    S.append(f"| 最大回撤(日频) | {dd:.1%} |")
    S.append(f"| Calmar | {cagr / abs(dd):.2f} |")
    S.append(f"| 沪深300 同期 | {btot:+.1%} (超额 {tot - btot:+.1%}) |")
    S.append(f"| 有持仓天数占比 | {invested:.0%} |\n")
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
        cost_in_total = (tr["alloc_yuan"] * args.cost_in).sum() if "alloc_yuan" in tr.columns else 0
        cost_out_total = (
                    (tr["alloc_yuan"] + tr["pnl_yuan"]) * args.cost_out).sum() if "alloc_yuan" in tr.columns else 0
        S.append(f"| 累计交易成本 | {cost_in_total + cost_out_total:,.0f} 元 |\n")
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
    S.append("> 仅量化研究用途，不构成投资建议。默认池含幸存者偏差(收益上偏); 回测豁免任期惩罚。\n")
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
    ap.add_argument("--codes", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    # 读取基金池
    if args.codes:
        codes = [l.strip().zfill(6) for l in open(args.codes, encoding="utf-8")
                 if l.strip() and l.strip()[0].isdigit()]
        default_universe = False
        print(f"[池] 自定义 {len(codes)} 只: {args.codes}")
    else:
        codes = sorted(set(pd.concat([pd.read_csv(f"{FACTOR_ROWS}/{f}", dtype={"code": str})["code"]
                                      for f in os.listdir(FACTOR_ROWS) if f.endswith(".csv")]).tolist()))
        default_universe = True
        print(f"[池] 全市场池 {len(codes)} 只")

    # ========== 关键修改：每月末打分 ==========
    end = args.end or str((pd.Timestamp.today() - pd.offsets.MonthEnd(1)).date())
    qs = [str(d.date()) for d in pd.date_range(args.start, end, freq="ME")]
    print(f"[打分区间] {qs[0]} → {qs[-1]} 共 {len(qs)} 个月末")

    panel = build_panel(qs, codes, default_universe, args.rebuild)
    print(f"[面板] 评分行 {len(panel)} | S>{args.buy:.0f} 信号 {int((panel.S > args.buy).sum())} 个")

    # 过滤掉没有打分数据的月份
    available_dates = set(panel["date"].unique())
    qs = [d for d in qs if d in available_dates]
    print(f"[有效打分月] 实际可用 {len(qs)} 个月")

    if not qs:
        print("❌ 没有任何可用打分月份，程序退出")
        return

    navs = load_navs(list(set(panel.code)))
    print(f"[净值] 可用 {len(navs)}/{panel.code.nunique()}")

    bench = provider.get_close_by_src("sina", "sh000300").dropna().sort_index()

    tag = (f"b{args.buy:.0f}s{args.sell:.0f}n{args.slots}k{int(args.capital/10000)}w" +
           ("_monthly" if not default_universe else ""))

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
