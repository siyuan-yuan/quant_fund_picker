# -*- coding: utf-8 -*-
"""V3.7 因子平滑变体裁决: 28季×217池重打分(取原始量), 5个变体离线对比"""
import os, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import provider
provider.STALE_OK = True
import rbsa, factors, risk
from engine import score_fund, finalize
from backtest import build_pool, forward_returns
from scipy import stats

OUT = "output/factor_rows"
os.makedirs(OUT, exist_ok=True)
DATES = [str(d.date()) for d in pd.date_range("2006-03-31", "2026-03-31", freq="QE")]


def harvest():
    """一次打分, 留存原始量 (F的输入, 非F本身)"""
    if os.path.exists(f"{OUT}/_done"):
        return
    pool = build_pool(random_n=800)
    t0 = time.time()
    for d in DATES:
        fp = f"{OUT}/{d}.csv"
        if os.path.exists(fp):
            continue
        rows = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True): c for c in pool}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if not r.get("error"):
                        rows.append(r)
                except Exception:
                    pass
        df = finalize(rows, as_of=d)
        ok = df.dropna(subset=["S_total"]).copy()
        recs = []
        for _, h in ok.iterrows():
            f6, f12 = forward_returns(h.code, d)
            pdt = h.penalty_detail or {}
            # 非R_MDD惩罚乘数(bt模式只剩集中度)
            op = 1.0
            for n, p in (h.penalties or []):
                if "回撤比值" not in n:
                    op *= (1 - p)
            recs.append(dict(date=d, code=h.code, S_eng=h.S_total,
                water=h.water, wv=h.w_value, wa=h.w_alpha, wm=h.w_mom,
                val_pct=h.val_pct, val_cov=h.val_coverage, trend_ok=bool(h.trend_ok),
                ma20_dist=h.get("ma20_dist"), wr=h.ir_winrate, dc=h.down_capture,
                r4=h.mom_4m1m, r7=h.mom_7m1m, R_MDD=pdt.get("R_MDD"),
                other_pen=op, F_value_eng=h.F_value, F_alpha_eng=h.F_alpha,
                F_mom_eng=h.F_momentum,
                fwd6=None if f6 != f6 else round(f6, 4),
                fwd12=None if f12 != f12 else round(f12, 4)))
        pd.DataFrame(recs).to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"[done] {d} n={len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    open(f"{OUT}/_done", "w").write("1")


def mdd_factor(rm, w):
    """现行 V3.6 平滑: ≤1.2免罚, k=0.5, 底部减半"""
    if pd.isna(rm) or rm <= 1.2:
        return 1.0
    p = min(0.5 * (rm - 1.2), 1.0)
    if pd.notna(w) and w <= 0.35:
        p *= 0.5
    return 1 - p


def build_variants(df):
    """离线重算 6 个版本的 S"""
    df = df.copy()
    df["rank4"] = df.groupby("date")["r4"].rank(pct=True)
    df["rank7"] = df.groupby("date")["r7"].rank(pct=True)
    df["t_trend"] = df.ma20_dist.apply(factors.trend_gate_smooth)

    # F_value: 旧=阶梯+布尔门(引擎复刻) vs 新=平滑
    fv_old = df.apply(lambda r: (np.nan if (r.val_cov < 0.5 or pd.isna(r.val_pct))
                                 else factors.valuation_base_score(r.val_pct, r.trend_ok)), axis=1)
    fv_new = df.apply(lambda r: (np.nan if (r.val_cov < 0.5 or pd.isna(r.val_pct))
                                 else factors.value_score_smooth(r.val_pct, r.t_trend)), axis=1)
    # F_alpha: 旧=台阶 vs 新=平滑
    def a_old(r):
        s = [factors.ir_score(r.wr) if pd.notna(r.wr) else np.nan,
             factors.dc_score(r.dc) if pd.notna(r.dc) else np.nan]
        s = [x for x in s if pd.notna(x)]
        return np.mean(s) if s else np.nan
    def a_new(r):
        s = [factors.ir_score_smooth(r.wr) if pd.notna(r.wr) else np.nan,
             factors.dc_score_smooth(r.dc) if pd.notna(r.dc) else np.nan]
        s = [x for x in s if pd.notna(x)]
        return np.mean(s) if s else np.nan
    fa_old, fa_new = df.apply(a_old, axis=1), df.apply(a_new, axis=1)
    # F_momentum: ranks → 旧阶梯 vs M1 vs M2
    fm_old = df.apply(lambda r: factors.momentum_score(r.rank4, r.rank7), axis=1)
    fm_m1 = df.apply(lambda r: factors.momentum_score_smooth_m1(r.rank4, r.rank7), axis=1)
    fm_m2 = df.apply(lambda r: factors.momentum_score_smooth_m2(r.rank4, r.rank7), axis=1)

    pen = df.apply(lambda r: r.other_pen * mdd_factor(r.R_MDD, r.water), axis=1)

    def synth(fv, fa, fm):
        num = (df.wv * fv.clip(0, 100)).where(fv.notna(), 0) \
            + (df.wa * fa).where(fa.notna(), 0) \
            + (df.wm * fm).where(fm.notna(), 0)
        den = df.wv * fv.notna() + df.wa * fa.notna() + df.wm * fm.notna()
        return ((num / den.replace(0, np.nan)).fillna(0) * pen).clip(0, 100)

    out = {
        "V0现行": synth(fv_old, fa_old, fm_old),
        "V1-alpha平滑": synth(fv_old, fa_new, fm_old),
        "V2-mom平滑M1": synth(fv_old, fa_old, fm_m1),
        "V3-mom平滑M2": synth(fv_old, fa_old, fm_m2),
        "S4-value平滑": synth(fv_new, fa_old, fm_old),
        "V5-全平滑": None,
    }
    return out, dict(fv_new=fv_new, fa_new=fa_new, fm_m1=fm_m1, fm_m2=fm_m2, synth=synth, fa_old=fa_old)


def stats_table(df):
    print(f"[校验] V0复建 vs 引擎S: 相关 {df.Sx_V0.corr(df.S_eng):.4f} 最大差 {(df.Sx_V0-df.S_eng).abs().max():.1f}")
    res = []
    for v in ["V0现行", "V1-alpha平滑", "V2-mom平滑M1", "V3-mom平滑M2", "S4-value平滑", "V5-全平滑"]:
        sc = f"Sx_{v[:2]}"
        df[f"rk_{v[:2]}"] = df.groupby("date")[sc].rank(pct=True)
        dd = df.dropna(subset=["fwd6", sc])
        rs = dd.groupby("date").apply(lambda g: stats.spearmanr(g[sc], g.fwd6)[0], include_groups=False).dropna()
        t = rs.mean() / (rs.std() / np.sqrt(len(rs)))
        by = dd[dd[sc] >= 70]
        print(f"{v:14s} IC均值{rs.mean():+.4f} t={t:5.2f} | Buy n={len(by)} fwd6{by.fwd6.mean():+.1%} 胜率{(by.fwd6>0).mean():.0%}")
        res.append(dict(variant=v, ic=rs.mean(), t=t, buy_n=len(by),
                        buy_fwd6=by.fwd6.mean(), buy_win=(by.fwd6 > 0).mean()))
    return pd.DataFrame(res)


def main():
    harvest()
    df = pd.concat([pd.read_csv(f"{OUT}/{f}", dtype={"code": str})
                    for f in os.listdir(OUT) if f.endswith(".csv")])
    df = df.dropna(subset=["S_eng"])
    outs, aux = build_variants(df)
    df["Sx_V0"] = outs["V0现行"]; df["Sx_V1"] = outs["V1-alpha平滑"]
    df["Sx_V2"] = outs["V2-mom平滑M1"]; df["Sx_V3"] = outs["V3-mom平滑M2"]
    df["Sx_S4"] = outs["S4-value平滑"]
    # V5 = value平滑 + alpha平滑 + M1/M2里IC更好的那个, 先算M1版
    df["Sx_V5"] = aux["synth"](aux["fv_new"], aux["fa_new"], aux["fm_m1"])
    tbl = stats_table(df)
    df.to_csv("output/factor_variant_rows.csv", index=False, encoding="utf-8-sig")
    tbl.to_csv("output/factor_variant_summary.csv", index=False, encoding="utf-8-sig")
    # 悬崖创伤统计: 旧阶梯两侧样本在平滑版下的分差
    print("\n[悬崖带实况] ir胜率∈[0.45,0.55]样本:")
    band = df[df.wr.between(0.45, 0.55)]
    print(f"  n={len(band)} 旧F_alpha均值{band.apply(lambda r: np.nanmean([factors.ir_score(r.wr) if pd.notna(r.wr) else np.nan]), axis=1).mean():.1f} "
          f"→ 新{factors.ir_score_smooth(0.50):.0f}附近 旧规则下同邻居可差60分")
    print("\n[saved] output/factor_variant_summary.csv / factor_variant_rows.csv")


if __name__ == "__main__":
    main()
