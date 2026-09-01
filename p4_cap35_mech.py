# -*- coding: utf-8 -*-
"""
P4-4 cap35 机制核查（预登记 #8, 纯诊断, 无新回测）
问题: 旧面板 P3-4 判定 STRAT_STYLE_CAP=0.35 为"现金拖累"（n_pos 4.5-5.2, 现金 48-55%）;
     新面板上 cap35 双窗越双门 → 机制到底是 A 现金拖累 / B 保护性分散 / C 混合?
输入: output/p3/{p3_summary.csv, p3_trades.csv, p3_curves.csv}（新面板全矩阵重跑产物）
判据: 预登记 #8（docs/优化计划_V4_2026-08.md §2）
输出: output/p4/p44_cap35_mech.csv + 控制台分类结论
复现: ./.venv/bin/python p4_cap35_mech.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(ROOT, "output", "p3")
OUT = os.path.join(ROOT, "output", "p4")
BASE, ALT = "v38_cost", "cap35"


def dd_episodes(equity: pd.Series, k: int = 3):
    """最深 k 个回撤 episode: (起点=前高日, 谷底日, 深度)"""
    dd = equity / equity.cummax() - 1
    episodes, in_dd, start = [], False, None
    for d, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, d
        elif v >= 0 and in_dd:
            in_dd = False
            seg = dd.loc[start:dd.index[dd.index.get_loc(d) - 1]]
            trough = seg.idxmin()
            peak = equity.loc[:trough].idxmax()
            episodes.append((peak, trough, float(seg.min())))
    if in_dd:
        seg = dd.loc[start:]
        trough = seg.idxmin()
        peak = equity.loc[:trough].idxmax()
        episodes.append((peak, trough, float(seg.min())))
    episodes.sort(key=lambda x: x[2])
    return episodes[:k]


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = pd.read_csv(os.path.join(P3, "p3_summary.csv"))
    trades = pd.read_csv(os.path.join(P3, "p3_trades.csv"), dtype={"code": str})
    curves = pd.read_csv(os.path.join(P3, "p3_curves.csv"), index_col=0, parse_dates=True)

    rows = []
    for name in [BASE, ALT]:
        for w in ["full", "recent"]:
            r = summ[(summ.variant == name) & (summ.window == w)].iloc[0]
            rows.append(dict(variant=name, window=w, CAGR=round(r.CAGR, 4), MaxDD=round(r.MaxDD, 4),
                             Calmar=round(r.Calmar, 3), n_trades=int(r.n_trades),
                             avg_hold=round(r.avg_hold_days, 0), cost=int(r.total_cost),
                             n_pos=round(r.mean_n_pos, 2), cash=round(r.mean_cash, 3)))
    mech = pd.DataFrame(rows)

    # --- ① 暴露与现金 ---
    d_np = mech[(mech.variant == BASE) & (mech.window == "full")].n_pos.iloc[0] - \
           mech[(mech.variant == ALT) & (mech.window == "full")].n_pos.iloc[0]
    d_cash = mech[(mech.variant == ALT) & (mech.window == "full")].cash.iloc[0] - \
             mech[(mech.variant == BASE) & (mech.window == "full")].cash.iloc[0]

    # --- ② style-cap trim 与入场风格 ---
    tr = trades.copy()
    tr["entry_dt"] = pd.to_datetime(tr.entry_date)
    tr["exit_dt"] = pd.to_datetime(tr.exit_date)
    tr["ey"] = tr.entry_dt.dt.year
    tr["xy"] = tr.exit_dt.dt.year
    trims = tr[(tr.variant == ALT) & (tr.reason == "style_cap35")]
    trim_by_yr = trims.groupby("xy").size().to_dict()
    trim_by_style = trims["style"].value_counts().to_dict()
    ent_base = tr[tr.variant == BASE].groupby("style").size()
    ent_alt = tr[tr.variant == ALT].groupby("style").size()
    top_b = ent_base.idxmax()
    top_a = ent_alt.idxmax()
    ent_share_b = ent_b = (ent_base / ent_base.sum())
    ent_share_a = (ent_alt / ent_alt.sum())
    top_share_b = float(ent_share_b.iloc[0])
    top_share_a = float(ent_share_a.iloc[0])

    # --- ③ 回撤 episode 与年度收益 ---
    eq = curves[[BASE, ALT]]
    ep = {n: dd_episodes(eq[n]) for n in eq.columns}
    yearly = {}
    for n in eq.columns:
        s = eq[n].resample("YE").last()
        r = s.pct_change().dropna()
        r.index = r.index.year
        yearly[n] = r
    ydf = pd.DataFrame(yearly)

    # 2022 年（普跌体制）对比
    r22 = float(ydf.loc[2022, ALT] - ydf.loc[2022, BASE]) if 2022 in ydf.index else np.nan

    # v38 最深回撤发生期, cap35 同期表现
    base_ep0 = ep[BASE][0]
    a0 = eq[ALT].loc[base_ep0[0]:base_ep0[1]]
    cap35_dd_in_base_ep = float((a0 / a0.cummax() - 1).min()) if len(a0) > 1 else np.nan

    # --- 预登记 #8 分类判据 ---
    n_trims = len(trims)
    cond_A = (d_np >= 1.0) and (d_cash >= 0.10)
    cond_B = (n_trims >= 20) and (cap35_dd_in_base_ep > base_ep0[2]) and (r22 >= 0)
    if cond_A and cond_B:
        label = "A+B 均满足（异常, 人工复核）"
    elif cond_A:
        label = "A 现金拖累"
    elif cond_B:
        label = "B 保护性分散"
    else:
        label = "C 混合/未定"

    # --- 落盘 ---
    out_rows = [
        dict(metric="d_n_pos (v38-cap35, full)", value=round(d_np, 2)),
        dict(metric="d_cash (cap35-v38, full)", value=round(d_cash, 4)),
        dict(metric="n_style_cap_trims (full)", value=n_trims),
        dict(metric="cap35_maxDD在v38最深回撤期内的深度", value=round(cap35_dd_in_base_ep, 4)),
        dict(metric="v38最深回撤深度", value=round(base_ep0[2], 4)),
        dict(metric="r2022_cap35_minus_v38", value=round(r22, 4)),
        dict(metric="cond_A (d_n_pos>=1.0 & d_cash>=10pp)", value=cond_A),
        dict(metric="cond_B (trims>=20 & 最深期更浅 & 2022不劣)", value=cond_B),
        dict(metric="机制标签", value=label),
    ]
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, "p44_cap35_mech.csv"), index=False,
                                  encoding="utf-8-sig")

    print("① 暴露/现金 (全期):")
    print(mech[mech.window == "full"].to_string(index=False))
    print(f"\n② Δn_pos = {d_np:+.2f}, Δcash = {d_cash*100:+.1f}pp")
    print(f"   style_cap35 trim 共 {n_trims} 笔; 按年 = {dict(sorted(trim_by_yr.items()))}")
    print(f"   trim 风格分布 = {trim_by_style}")
    print(f"   入场风格集中度: v38 最高风格 {top_b} 占 {top_share_b*100:.1f}%; "
          f"cap35 最高风格 {top_a} 占 {top_share_a*100:.1f}%")
    print(f"   入场 top5 风格 (v38): {ent_share_b.sort_values(ascending=False).head(5).round(3).to_dict()}")
    print(f"   入场 top5 风格 (cap35): {ent_share_a.sort_values(ascending=False).head(5).round(3).to_dict()}")
    print("\n③ 最深 3 回撤 episode (前高日→谷底日, 深度):")
    for n in [BASE, ALT]:
        for pk, tg, dv in ep[n]:
            print(f"   {n}: {pk.date()}→{tg.date()}, {dv*100:.2f}%")
    print(f"\n   v38 最深回撤期内 cap35 自身最大回撤: {cap35_dd_in_base_ep*100:.2f}%")
    print("\n   年度收益 (cap35 − v38):")
    ydf["diff"] = ydf[ALT] - ydf[BASE]
    print((ydf * 100).round(2).to_string())
    print(f"\n[预登记 #8 判据] cond_A={cond_A}, cond_B={cond_B} → **{label}**")
    print(f"完成 → {OUT}/p44_cap35_mech.csv")


if __name__ == "__main__":
    main()
