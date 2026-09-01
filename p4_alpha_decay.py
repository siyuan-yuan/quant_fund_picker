# -*- coding: utf-8 -*-
"""
H-P4A：F_alpha 衰减机制调查（预登记 #7, docs/优化计划_V4_2026-08.md §2）
背景: P4-2 OOS 告警——S_total 滚动12月 IC 0.042 = 全期 0.131 的 32%；
      F_alpha 2024-2025 跨两种行情、全基金群翻负；F_momentum 同期偏强。
输入: output/p1_panel/*.csv（P1-0 严格 PIT 全池面板, 2013-01→2026-03）
实验:
  HP4A-1  年度×子群 IC 轨迹（F_alpha / wr / dc, 全池+4子群; top1_style×2025 诊断表）
  HP4A-2  衰减 vs 行情分类（预登记符号判据: 2024/2025 全池 F_alpha 年度 IC）
  HP4A-3  P1-2 IC 重加权否决的 PIT 重审:
          S_pit  = 严格复现 P1-2 S3 规则（扩展窗 |IC(fwd6)| 均值归一化权重,
                   仅用决策月之前的数据, 完全 PIT; engine 同式合成）
          S3r    = 36 月滚动窗变体（稳健性, 仅报告不入裁决）
          评测: (a) 全期 S_pit vs S_total 配对差（应复现 P1-2 t≈−3.9, 实现校验）
                (b) 真 OOS 段 2024-01→（fwd6 成熟月）: 权重全部由 2024 前数据推得
          预登记判据: (b) 段配对差均值 > 0 → 否决为体制条件性 → 携带 S_pit 至
                       HP4A-4 模型级双门; ≤ 0 → 否决维持, F_alpha 登记监控因子。
  HP4A-4  条件分支（p3_backtest 评分变体）——本脚本不执行, 由判据触发与否决定。
输出: output/p4/hp4a_yearly_ic.csv, hp4a_decay_class.csv, hp4a_pit_reweight.csv
复现: ./.venv/bin/python p4_alpha_decay.py
"""
import os
import numpy as np
import pandas as pd

from p1_analysis import load_panel, _ic_row, _parse_pens
from engine import resolve_weights

PANEL_DIR = os.path.join("output", "p1_panel")
OUT_DIR = os.path.join("output", "p4")
MIN_N = 30
OV_STYLES = {"纳斯达克100", "标普500", "恒生指数", "恒生科技"}
# P1-2 同式: IC 权重所用的三个基线因子（engine 层 0-100 分数, 方向均已"越高越好"）
IC_FACTORS = {"eng_value": "F_value", "eng_alpha": "F_alpha", "eng_mom": "F_momentum"}
WARMUP = 24          # P1-2: 前 24 个月用基线 regime 权重
ROLL = 36            # S3r: 滚动窗长度（月）
OOS_START = "2024-01-01"


def _month_ic_series(d: pd.DataFrame, fcol: str, dirn: float, hcol: str = "fwd6") -> pd.Series:
    """月度 IC（成熟月, 与 p4_analysis 同口径: _ic_row 内部要求 ≥10 只有效样本）"""
    out = {}
    for dt, g in d.groupby("date", sort=True):
        out[dt] = _ic_row(g, fcol, hcol, dirn)
    return pd.Series(out)


def _stat(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, t=np.nan)
    t = s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
    return dict(n=len(s), mean=round(float(s.mean()), 4), t=round(float(t), 2))


def synth(weights, fv, fa, fm, pens):
    """engine 同式: 缺失因子归一化 + 乘法惩罚 + [0,100] 截断（与 p1_analysis.p1_2 完全一致）"""
    out = np.full(len(fv), np.nan)
    for i in range(len(fv)):
        num = den = 0.0
        for w, v in zip(weights, (fv[i], fa[i], fm[i])):
            if v is None or (isinstance(v, float) and v != v):
                continue
            num += w * v
            den += w
        base = num / den if den > 1e-9 else 0.0
        s = base
        for p in (pens[i] or []):
            s *= (1 - p)
        out[i] = max(0.0, min(100.0, s))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_panel(min_n=MIN_N)
    for c in ["F_value", "F_alpha", "F_momentum", "wr", "dc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df.copy()
    d["is_ov"] = d["top1_style"].isin(OV_STYLES).fillna(False)
    months = sorted(d["date"].unique())
    idx_of = {m: i for i, m in enumerate(months)}
    print(f"[H-P4A] 面板: {d['date'].min().date()}→{d['date'].max().date()}, "
          f"{len(months)} 个月 n≥{MIN_N}, {len(d)} 行", flush=True)

    # ================= HP4A-1 年度×子群 IC 轨迹 =================
    groups = {
        "full": d,
        "active": d[~d.is_passive],
        "passive": d[d.is_passive],
        "ashare": d[~d.is_ov],
        "overseas": d[d.is_ov],
    }
    facs = [("F_alpha", 1.0), ("wr", 1.0), ("dc", -1.0)]
    rows = []
    for tag, sub in groups.items():
        sub = sub.copy()
        sub["year"] = sub["date"].dt.year
        for fcol, dirn in facs:
            ics = _month_ic_series(sub, fcol, dirn)
            ics = ics.dropna()
            for yr, g in ics.groupby(ics.index.year):
                rows.append(dict(subgroup=tag, factor=fcol, direction=dirn, year=int(yr),
                                 n_months=len(g),
                                 ic_mean=round(float(g.mean()), 4),
                                 t=round(float(g.mean() / (g.std() / np.sqrt(len(g)))), 2)
                                 if g.std() > 0 else np.nan))
    yearly = pd.DataFrame(rows)
    yearly.to_csv(os.path.join(OUT_DIR, "hp4a_yearly_ic.csv"), index=False, encoding="utf-8-sig")

    # top1_style × 2025 诊断表（仅报告: A 股风格轮动 vs 基金 alpha 机制衰减）
    sub25 = d[(d["date"] >= "2025-01-01")]
    sty_rows = []
    for sty, g in sub25.groupby("top1_style"):
        ics_a = _month_ic_series(g, "F_alpha", 1.0).dropna()
        ics_w = _month_ic_series(g, "wr", 1.0).dropna()
        if len(ics_a) >= 3:
            sty_rows.append(dict(top1_style=sty, n_months=len(ics_a), n_fund_mo=len(g),
                                 alpha_ic_2025=round(float(ics_a.mean()), 4),
                                 wr_ic_2025=round(float(ics_w.mean()), 4) if len(ics_w) else np.nan))
    style25 = pd.DataFrame(sty_rows).sort_values("n_fund_mo", ascending=False)
    style25.to_csv(os.path.join(OUT_DIR, "hp4a_style2025.csv"), index=False, encoding="utf-8-sig")

    # ================= HP4A-2 衰减 vs 行情分类（预登记符号判据） =================
    fa_full = _month_ic_series(d, "F_alpha", 1.0).dropna()
    ic24 = fa_full[fa_full.index.year == 2024]
    ic25 = fa_full[fa_full.index.year == 2025]
    m24, m25 = (float(ic24.mean()) if len(ic24) else np.nan,
                float(ic25.mean()) if len(ic25) else np.nan)
    if m24 != m24 or m25 != m25:
        label = "未证实(数据不足)"
    elif m24 < 0 and m25 < 0:
        label = "连续衰减"
    elif m24 < 0 or m25 < 0:
        label = "行情性/单年"
    else:
        label = "未证实"
    decay_class = pd.DataFrame([dict(ic_2024=round(m24, 4), n_2024=len(ic24),
                                     ic_2025=round(m25, 4), n_2025=len(ic25),
                                     label=label)])
    decay_class.to_csv(os.path.join(OUT_DIR, "hp4a_decay_class.csv"), index=False,
                       encoding="utf-8-sig")

    # ================= HP4A-3 PIT IC 重加权重审 =================
    # 每决策月基线 regime 权重（与面板构建同式）
    w_base_month = {}
    for m, g in d.groupby("date"):
        wtr = g["water"].iloc[0]
        wtr = float(wtr) if wtr == wtr else np.nan
        (wv, wa, wm), _ = resolve_weights(wtr)
        w_base_month[m] = [wv, wa, wm]

    # 扩展窗所需: 每决策月三因子 IC(fwd6)（含未成熟月 → NaN, 窗口内跳过）
    base_ic = {m: {} for m in months}
    for m, g in d.groupby("date"):
        for fn, col in IC_FACTORS.items():
            base_ic[m][fn] = _ic_row(g, col, "fwd6", 1.0)

    def ic_weights_expanding(i, m):
        """P1-2 原式: i<24 → 基线 regime 权重; 否则扩展窗 |IC| 均值归一化（完全 PIT）"""
        if i < WARMUP:
            b = w_base_month[m]
            return [b[0], b[1], b[2]]
        w = {}
        for fn in IC_FACTORS:
            vals = [base_ic[months[j]][fn] for j in range(i)
                    if base_ic[months[j]][fn] == base_ic[months[j]][fn]]
            w[fn] = abs(float(np.mean(vals))) if vals else 1e-6
        s = sum(w.values())
        return [w["eng_value"] / s, w["eng_alpha"] / s, w["eng_mom"] / s]

    def ic_weights_rolling(i, m):
        """S3r 稳健性变体: 36 月滚动窗 |IC| 均值归一化（仅用 i 之前数据）"""
        w = {}
        for fn in IC_FACTORS:
            vals = [base_ic[months[j]][fn] for j in range(max(0, i - ROLL), i)
                    if base_ic[months[j]][fn] == base_ic[months[j]][fn]]
            w[fn] = abs(float(np.mean(vals))) if vals else 1e-6
        s = sum(w.values())
        return [w["eng_value"] / s, w["eng_alpha"] / s, w["eng_mom"] / s]

    recs = []
    for m, g in d.groupby("date"):
        i = idx_of[m]
        g = g.reset_index(drop=True)
        fv, fa, fm = (g[c].values for c in ["F_value", "F_alpha", "F_momentum"])
        pens_l = g["penalties"].apply(_parse_pens).tolist()
        for name, ws in [("S0_基线", w_base_month[m]),
                         ("S3_PIT重加权", ic_weights_expanding(i, m)),
                         ("S3r_36m滚动", ic_weights_rolling(i, m))]:
            sc = synth(ws, fv, fa, fm, pens_l)
            recs.append(pd.DataFrame(dict(date=m, code=g["code"], variant=name, S=sc,
                                          fwd6=g["fwd6"])))
    long_df = pd.concat(recs, ignore_index=True)

    # 实现校验: S0(重建) vs 面板 S_total（P1-2 已验证 max|diff|≈0.05, 复跑防回归）
    s0 = long_df[long_df["variant"] == "S0_基线"].set_index(["date", "code"])["S"]
    key = d.set_index(["date", "code"])
    align = key["S_total"].index.join(s0.index, how="inner")
    maxdiff = float(np.nanmax(np.abs(s0.loc[align].values - key.loc[align, "S_total"].values)))
    print(f"[H-P4A] 实现校验: S0重建 vs 面板S_total 最大差 = {maxdiff:.2f} (应≈0.05)", flush=True)
    if maxdiff > 1.0:
        raise RuntimeError("S0 重建与面板 S_total 不一致, 检查实现")

    # 月度 IC 序列（fwd6, 成熟月）
    def monthly_ic(variant: str) -> pd.Series:
        sub = long_df[long_df["variant"] == variant]
        out = {}
        for dt, g in sub.groupby("date", sort=True):
            out[dt] = _ic_row(g, "S", "fwd6", 1.0)
        return pd.Series(out)

    ic_s0 = monthly_ic("S0_基线").dropna()
    ic_s3 = monthly_ic("S3_PIT重加权").dropna()
    ic_s3r = monthly_ic("S3r_36m滚动").dropna()
    common = ic_s0.index.intersection(ic_s3.index)
    pair_s3 = ic_s3[common] - ic_s0[common]
    common_r = ic_s0.index.intersection(ic_s3r.index)
    pair_s3r = ic_s3r[common_r] - ic_s0[common_r]

    # 面板 S_total 的月度 IC（与 S0 应几乎相同, 供对照）
    ic_panel = _month_ic_series(d, "S_total", 1.0).dropna()
    pcommon = ic_panel.index.intersection(ic_s0.index)
    diff_s0_panel = (ic_s0[pcommon] - ic_panel[pcommon])

    oos = common >= pd.Timestamp(OOS_START)
    rows = []
    for name, s in [("S0_基线(重建)", ic_s0), ("面板S_total", ic_panel),
                    ("S3_PIT重加权", ic_s3), ("S3r_36m滚动", ic_s3r)]:
        rows.append(dict(series=name, segment="全期", **_stat(s)))
        rows.append(dict(series=name, segment=f"OOS({OOS_START[0:7]}+)",
                         **_stat(s[s.index >= pd.Timestamp(OOS_START)])))
    for name, p in [("S3−S0 配对差", pair_s3), ("S3r−S0 配对差", pair_s3r),
                    ("S0−面板S_total (实现漂移)", diff_s0_panel)]:
        rows.append(dict(series=name, segment="全期", **_stat(p)))
        rows.append(dict(series=name, segment=f"OOS({OOS_START[0:7]}+)",
                         **_stat(p[p.index >= pd.Timestamp(OOS_START)])))
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUT_DIR, "hp4a_pit_reweight.csv"), index=False,
                    encoding="utf-8-sig")

    # 月度明细（供报告绘图/复核）
    pd.DataFrame({
        "S0_基线": ic_s0, "面板S_total": ic_panel,
        "S3_PIT重加权": ic_s3, "S3r_36m滚动": ic_s3r,
    }).to_csv(os.path.join(OUT_DIR, "hp4a_pit_monthly_ic.csv"), encoding="utf-8-sig")

    # S_pit 评分表（date, code, S_pit）——供 p3_backtest spit 变体交叉验证
    pit = long_df[long_df["variant"] == "S3_PIT重加权"]
    pit[["date", "code", "S"]].rename(columns={"S": "S_pit"}).to_csv(
        os.path.join(OUT_DIR, "hp4a_spit_scores.csv"), index=False, encoding="utf-8-sig")

    # 每决策月权重轨迹（S3, 报告用）
    wrows = []
    for m in months:
        i = idx_of[m]
        w = ic_weights_expanding(i, m)
        # 全精度落盘: p3_backtest spit 变体逐值复现 S_pit（3 位舍入会造成 ~0.08 分漂移）
        wrows.append(dict(date=m, w_value=w[0], w_alpha=w[1], w_mom=w[2],
                          warmup=i < WARMUP))
    wts = pd.DataFrame(wrows)
    wts.to_csv(os.path.join(OUT_DIR, "hp4a_pit_weights.csv"), index=False, encoding="utf-8-sig")

    # ================= 打印与预登记判据 =================
    print("\n[HP4A-1] 年度×子群 IC(fwd6) 网格:")
    pv = yearly.pivot_table(index=["subgroup", "factor"], columns="year",
                            values="ic_mean", aggfunc="first")
    print(pv.to_string())
    print("\n[HP4A-1] top1_style × 2025 (诊断, 仅报告):")
    print(style25.to_string(index=False))
    print("\n[HP4A-2] 衰减 vs 行情分类:")
    print(decay_class.to_string(index=False))
    print("\n[HP4A-3] S0/S3/S3r vs S_total 月度 IC 统计 (fwd6):")
    print(stats_df.to_string(index=False))
    oos_mean = float(pair_s3[pair_s3.index >= pd.Timestamp(OOS_START)].mean())
    n_oos = int((pair_s3.index >= pd.Timestamp(OOS_START)).sum())
    print(f"\n[HP4A-3] OOS 段 (n={n_oos} 月) S3−S0 配对差 IC 均值 = {oos_mean:.4f}")
    if oos_mean > 0:
        verdict = ("判据通过: P1-2 否决为体制条件性 → 携带 S_pit 至 HP4A-4 "
                   "（p3_backtest 评分变体, 模型级双门对决 v38_cost）")
    else:
        verdict = ("判据未通过: 否决维持（衰减体制下静态 0.40/0.35/0.25 仍占优）→ "
                   "正确响应 = 因子层监控而非调权; F_alpha 登记为监控因子（月度 IC 盯守, 不退役）")
    print(f"[HP4A-3] 预登记判据结论: {verdict}")
    print(f"\n完成 → {OUT_DIR}/ (hp4a_*.csv)")


if __name__ == "__main__":
    main()
