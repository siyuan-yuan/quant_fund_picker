# -*- coding: utf-8 -*-
"""
V3.7.1 策略级回测 —— 信号驱动的真交易模拟
规则(用户指定基线): S>70 买入, S<50 卖出, 50~70 持有(迟滞带), 10 等权仓位槽
数据: 2006Q1 ~ 2026Q1 全市场池 (PiT 因子原料)
     + 缓存日频净值 (真实成交价) + 沪深300 基准
成本: 申购0.15% + 赎回0.5% (往返0.65%)
诚实边界: 池为存活至今基金(幸存者偏差上偏), 任期惩罚因非PiT在回测中豁免
"""
import os, json
import numpy as np
import pandas as pd
import factors

CACHE, OUT = "cache", "output"
COST_IN, COST_OUT = 0.0015, 0.005
N_SLOTS = 10

# ============ 1. 重建 V3.7.1 采纳分 (fv旧 + alpha平滑 + mom M1) ============
def mdd_factor(rm, w):
    if pd.isna(rm) or rm <= 1.2:
        return 1.0
    p = min(0.5 * (rm - 1.2), 1.0)
    if pd.notna(w) and w <= 0.35:
        p *= 0.5
    return 1 - p

def rebuild_scores():
    df = pd.concat([pd.read_csv(f"{OUT}/factor_rows/{f}", dtype={"code": str})
                    for f in os.listdir(f"{OUT}/factor_rows") if f.endswith(".csv")])
    df = df.dropna(subset=["S_eng"]).copy()
    df["rank4"] = df.groupby("date")["r4"].rank(pct=True)
    df["rank7"] = df.groupby("date")["r7"].rank(pct=True)
    fv = df.apply(lambda r: (np.nan if (r.val_cov < 0.5 or pd.isna(r.val_pct))
                             else factors.valuation_base_score(r.val_pct, r.trend_ok)), axis=1)
    fa = df.apply(lambda r: np.nanmean([factors.ir_score_smooth(r.wr) if pd.notna(r.wr) else np.nan,
                                        factors.dc_score_smooth(r.dc) if pd.notna(r.dc) else np.nan])
                  if (pd.notna(r.wr) or pd.notna(r.dc)) else np.nan, axis=1)
    fm = df.apply(lambda r: factors.momentum_score_smooth_m1(r.rank4, r.rank7), axis=1)
    pen = df.apply(lambda r: r.other_pen * mdd_factor(r.R_MDD, r.water), axis=1)
    num = (df.wv * fv.clip(0, 100)).where(fv.notna(), 0) \
        + (df.wa * fa).where(fa.notna(), 0) + (df.wm * fm).where(fm.notna(), 0)
    den = df.wv * fv.notna() + df.wa * fa.notna() + df.wm * fm.notna()
    df["S"] = ((num / den.replace(0, np.nan)).fillna(0) * pen).clip(0, 100)
    return df

def ic_check(df):
    from scipy import stats
    dd = df.dropna(subset=["fwd6", "S"])
    rs = dd.groupby("date").apply(lambda g: stats.spearmanr(g.S, g.fwd6)[0],
                                  include_groups=False).dropna()
    t = rs.mean() / (rs.std() / np.sqrt(len(rs)))
    print(f"[校验] V3.7.1 重建: IC均值 {rs.mean():+.4f}  t={t:.2f}  (期望≈3.46, 因子研究同源)")
    return t

# ============ 2. 净值与基准 ============
def load_navs(codes):
    navs = {}
    for c in codes:
        fp = f"{CACHE}/nav_{c}.csv"
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp, parse_dates=["date"]).set_index("date")["nav"].sort_index()
        d = d[~d.index.duplicated(keep="last")].dropna()
        if len(d) > 200:
            navs[c] = d
    return navs

def load_bench():
    d = pd.read_csv(f"{CACHE}/idx_sh000300.csv")
    d.columns = [str(c).lower() for c in d.columns]
    col = "close" if "close" in d.columns else ("nav" if "nav" in d.columns else d.columns[1])
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[col].sort_index()

def px(s, dt):
    """asof 取价: 最后一个 ≤ dt 的净值"""
    if s is None:
        return np.nan
    v = s.asof(pd.Timestamp(dt))
    return float(v) if pd.notna(v) else np.nan

# ============ 3. 交易模拟 ============
def simulate(df, navs, bench, buy_th=70.0, sell_th=45.0, water_gate=None,
             ma20_exit=False, trail_stop=None, hi_water=None, label="base"):
    """hi_water=(门槛, 高位槽位上限): 水位≥门槛 → 决策日强制瘦身(留强去弱)并封新仓上限"""
    dates = sorted(df["date"].unique())
    T0, T1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    by_date = {d: g.set_index("code") for d, g in df.groupby("date")}
    water = df.groupby("date")["water"].first()
    ma20 = {c: s.rolling(20).mean() for c, s in navs.items()} if ma20_exit else {}

    day_grid = bench.loc[T0:T1].index                       # 逐日盯市(基准交易日历)
    cash, positions, trades = 1.0, {}, []                   # 初始净值1.0
    equity_curve, blocked_buys, skipped_full = [], 0, 0

    def sell(code, dt, reason, exit_S=np.nan, price=None):
        nonlocal cash
        p = positions.pop(code)
        price = price if price is not None else px(navs[code], dt)
        val = p["units"] * price * (1 - COST_OUT)
        cash += val
        net_ret = (price * (1 - COST_OUT)) / (p["entry_px"] * (1 + COST_IN)) - 1
        trades.append(dict(code=code, entry_date=p["entry_date"], exit_date=str(pd.Timestamp(dt).date()),
                           entry_S=round(p["entry_S"], 1),
                           exit_S=round(exit_S, 1) if pd.notna(exit_S) else np.nan,
                           exit_reason=reason, hold_days=(pd.Timestamp(dt) - pd.Timestamp(p["entry_date"])).days,
                           gross_ret=price / p["entry_px"] - 1, net_ret=net_ret))

    # 季末若遇休市 → 决策落在之前最后一个交易日
    dmap = {}
    for d in dates:
        ts = pd.Timestamp(d)
        elig = day_grid[day_grid <= ts]
        if len(elig):
            dmap[str(elig[-1].date())] = d

    for day in day_grid:
        # 3a. MA20 日内纪律 (变体D)
        if ma20_exit:
            for c in list(positions):
                p, m = px(navs[c], day), px(ma20[c], day)
                if pd.notna(p) and pd.notna(m) and p < m:
                    sell(c, day, "ma20破位")
        # 3a2. 移动止损 (变体E: 自入场后最高价回撤>15%离场)
        if trail_stop is not None:
            for c in list(positions):
                p = px(navs[c], day)
                if pd.notna(p):
                    pos = positions[c]
                    pos["peak"] = max(pos.get("peak", pos["entry_px"]), p)
                    if p < pos["peak"] * (1 - trail_stop):
                        sell(c, day, f"回撤{trail_stop:.0%}止损", price=p)
        # 3b. 决策日: 先卖后买
        if str(day.date()) in dmap:
            dkey = dmap[str(day.date())]
            g = by_date[dkey]
            for c in list(positions):
                if c not in g.index or pd.isna(g.loc[c, "S"]):
                    sell(c, day, "跌出面板")
                elif g.loc[c, "S"] < sell_th:
                    sell(c, day, f"S<{sell_th:.0f}", exit_S=g.loc[c, "S"])
            # 批判清单⑦: 持仓侧水位降杠杆 — 高位(≥gate)强制瘦身, 留强去弱
            max_slots = N_SLOTS
            if hi_water is not None and water.get(dkey, 0) >= hi_water[0]:
                max_slots = hi_water[1]
                if len(positions) > max_slots:
                    order = sorted(positions,
                                   key=lambda c: (g.loc[c, "S"] if c in g.index else -1),
                                   reverse=True)
                    for c in order[max_slots:]:
                        sell(c, day, f"水位≥{hi_water[0]:.0%}瘦身",
                             exit_S=g.loc[c, "S"] if c in g.index else np.nan)
            equity_now = cash + sum(p["units"] * px(navs[c], day) for c, p in positions.items())
            buys = g[g.S > buy_th].sort_values("S", ascending=False)
            if water_gate is not None and water.get(dkey, 1) >= water_gate:
                blocked_buys += len(buys)
            else:
                for c, row in buys.iterrows():
                    if c in positions or len(positions) >= max_slots:
                        skipped_full += 1 if c not in positions else 0
                        continue
                    price = px(navs.get(c), day)
                    if pd.isna(price) or pd.Timestamp(navs[c].index[0]) > day:
                        continue                          # 当时尚无净值(新基金)
                    alloc = min(cash, equity_now / N_SLOTS)
                    if alloc < equity_now * 0.02:
                        skipped_full += 1
                        continue
                    units = alloc / (price * (1 + COST_IN))
                    cash -= alloc
                    positions[c] = dict(units=units, entry_px=price,
                                        entry_date=dkey, entry_S=row.S)
        # 3c. 逐日盯市
        eq = cash + sum(p["units"] * px(navs.get(c), day) for c, p in positions.items())
        equity_curve.append((day, eq, len(positions)))

    for c in list(positions):                             # 期末强制平仓
        sell(c, T1, "期末清算")
    equity_curve[-1] = (equity_curve[-1][0], cash, 0)

    ec = pd.DataFrame(equity_curve, columns=["date", "equity", "n_pos"]).set_index("date")
    return ec, pd.DataFrame(trades), dict(blocked=blocked_buys, skipped=skipped_full)

# ============ 4. 指标 ============
def metrics(ec, tr, bench, extra, label, T0, T1):
    yrs = (T1 - T0).days / 365.25
    tot = ec.equity.iloc[-1] / ec.equity.iloc[0] - 1
    cagr = (ec.equity.iloc[-1] / ec.equity.iloc[0]) ** (1 / yrs) - 1
    dd = (ec.equity / ec.equity.cummax() - 1).min()
    bb = bench.loc[T0:T1]
    btot = bb.iloc[-1] / bb.iloc[0] - 1
    bdd = (bb / bb.cummax() - 1).min()
    invested_ratio = (ec.n_pos > 0).mean()
    r = dict(变体=label, 交易数=len(tr), 胜率=(tr.net_ret > 0).mean() if len(tr) else np.nan,
             平均净收益=tr.net_ret.mean() if len(tr) else np.nan,
             中位净收益=tr.net_ret.median() if len(tr) else np.nan,
             平均盈利=tr.loc[tr.net_ret > 0, "net_ret"].mean() if (tr.net_ret > 0).any() else np.nan,
             平均亏损=tr.loc[tr.net_ret <= 0, "net_ret"].mean() if (tr.net_ret <= 0).any() else np.nan,
             盈亏比=abs(tr.loc[tr.net_ret > 0, "net_ret"].mean() /
                       tr.loc[tr.net_ret <= 0, "net_ret"].mean()) if (tr.net_ret <= 0).any() and (tr.net_ret > 0).any() else np.nan,
             平均持有天数=tr.hold_days.mean() if len(tr) else np.nan,
             组合总收益=tot, CAGR=cagr, 最大回撤=dd, Calmar=cagr / abs(dd) if dd else np.nan,
             仓位利用率=invested_ratio,
             基准同期=btot, 基准回撤=bdd, 超额收益=tot - btot)
    r.update(extra)
    return r

def main():
    os.makedirs(OUT, exist_ok=True)
    df = rebuild_scores()
    t_ic = ic_check(df)
    navs = load_navs(df.code.unique())
    missing = sorted(set(df.code.unique()) - set(navs))
    print(f"[数据] 面板 {df.shape} | 净值可用 {len(navs)}/{df.code.nunique()} 缺 {missing}")
    bench = load_bench()
    T0 = pd.Timestamp(df.date.min()); T1 = pd.Timestamp(df.date.max())
    sig = df[df.S > 70]
    print(f"[信号] S>70: {len(sig)} fund-quarter, 每季均值 {sig.groupby('date').size().mean():.1f} 只, "
          f"零信号季度 {(28 - sig.groupby('date').size().shape[0])} 个")

    variants = [
        ("A0 旧基线 <50卖(历史参考)",     dict(buy_th=70, sell_th=50)),
        ("A 新基线 >70买/<45卖(推荐)",    dict(buy_th=70, sell_th=45)),
        ("H A+水位≥90%持仓瘦身(10→5槽)", dict(buy_th=70, sell_th=45, hi_water=(0.90, 5))),
    ]
    rows, curves, ledgers = [], {}, {}
    for label, kw in variants:
        ec, tr, extra = simulate(df, navs, bench, label=label, **kw)
        m = metrics(ec, tr, bench, extra, label, T0, T1)
        rows.append(m); curves[label] = ec; ledgers[label] = tr
        print(f"[{label}] 交易{len(tr)}笔 胜率{m['胜率']:.0%} 均净{m['平均净收益']:+.1%} "
              f"总收益{m['组合总收益']:+.1%} CAGR{m['CAGR']:+.1%} DD{m['最大回撤']:.1%}")

    rep = pd.DataFrame(rows)
    rep.to_csv(f"{OUT}/strategy_bt_variants.csv", index=False, encoding="utf-8-sig")
    BL = "A 新基线 >70买/<45卖(推荐)"
    ledgers[BL].to_csv(f"{OUT}/strategy_bt_trades.csv", index=False, encoding="utf-8-sig")
    df[["date", "code", "S", "water", "R_MDD", "fwd6"]].to_csv(
        f"{OUT}/strategy_bt_scores.csv", index=False, encoding="utf-8-sig")

    # 退出原因分布 & 年度拆解(新基线)
    tr = ledgers[BL]
    print("\n[基线退出原因]"); print(tr.groupby("exit_reason").agg(n=("net_ret", "size"),
          均净=("net_ret", "mean"), 胜率=("net_ret", lambda x: (x > 0).mean())).round(3))
    tr["entry_yr"] = pd.to_datetime(tr.entry_date).dt.year
    print("\n[基线按入场年]"); print(tr.groupby("entry_yr").agg(n=("net_ret", "size"),
          均净=("net_ret", "mean"), 胜率=("net_ret", lambda x: (x > 0).mean())).round(3))

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
    mk = {"A0 旧基线 <50卖(历史参考)": ("#9aa3b2", 1.2),
          "A 新基线 >70买/<45卖(推荐)": ("#f5b83d", 2.4),
          "H A+水位≥90%持仓瘦身(10→5槽)": ("#6fe3a5", 1.4)}
    for lb, ec in curves.items():
        c, lw = mk[lb]
        ax[0].plot(ec.index, ec.equity / ec.equity.iloc[0], color=c, lw=lw, label=lb)
    bb = bench.loc[T0:T1]
    ax[0].plot(bb.index, bb / bb.iloc[0], color="#666", lw=1.2, ls="--", label="沪深300")
    ax[0].legend(fontsize=9); ax[0].set_title("V3.7.2 策略回测 (>70买/<45卖, 10槽等权, 成本0.65%)  2019Q1–2025Q4")
    ax[0].grid(alpha=0.25)
    ec0 = curves[BL]
    ax[1].fill_between(ec0.index, ec0.equity / ec0.equity.cummax() - 1, 0,
                       color="#f5b83d", alpha=0.4, label="策略回撤")
    ax[1].fill_between(bb.index, bb / bb.cummax() - 1, 0, color="#888", alpha=0.3, label="沪深300回撤")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25); ax[1].set_ylabel("回撤")
    fig.tight_layout(); fig.savefig(f"{OUT}/strategy_bt_equity.png", dpi=130)
    print("\n[输出] strategy_bt_variants.csv / strategy_bt_trades.csv / strategy_bt_scores.csv / strategy_bt_equity.png")

if __name__ == "__main__":
    main()
