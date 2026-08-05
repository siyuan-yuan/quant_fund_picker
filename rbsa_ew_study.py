# -*- coding: utf-8 -*-
"""
V3.7.2 批判清单④: EW加权Ridge-RBSA vs 常规平权 —— 子样本裁决实验
设计: 10个跨regime季点 × 现存217池, EW(半衰期20交易日)重打分 → 重建V3.7.1分S_ew
     与基线(factor_rows)在同一(date,code)对上配对: Spearman IC 对 fwd6 逐日对比
达标线: EW 的 IC t 必须击败基线(t≈3.46量级), 且 Buy组不恶化, 才准进全量28季重跑
"""
import os, time, sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import provider
provider.STALE_OK = True
import rbsa, factors
from engine import score_fund
from scipy import stats

HL = 20                     # EW半衰期(交易日): 批判方"一周捕捉漂移"的折中(太快=噪声)
DATES = ["2019-03-31", "2019-12-31", "2020-12-31", "2021-03-31", "2021-12-31",
         "2022-06-30", "2022-12-31", "2023-09-30", "2024-09-30", "2025-06-30"]
OUT = "output/rbsa_ew_rows.csv"


def mdd_factor(rm, w):
    if pd.isna(rm) or rm <= 1.2:
        return 1.0
    p = min(0.5 * (rm - 1.2), 1.0)
    if pd.notna(w) and w <= 0.35:
        p *= 0.5
    return 1 - p


def s_score(g):
    """与 strategy_bt.rebuild_scores 同源: fv旧 + alpha平滑 + mom M1 × 惩罚链"""
    rank4 = g["r4"].rank(pct=True); rank7 = g["r7"].rank(pct=True)
    rows = []
    for (_, r), r4, r7 in zip(g.iterrows(), rank4, rank7):
        fv = (np.nan if (r.val_cov < 0.5 or pd.isna(r.val_pct))
              else factors.valuation_base_score(r.val_pct, r.trend_ok))
        al = [x for x in [factors.ir_score_smooth(r.wr) if pd.notna(r.wr) else np.nan,
                          factors.dc_score_smooth(r.dc) if pd.notna(r.dc) else np.nan]
              if pd.notna(x)]
        fa = np.mean(al) if al else np.nan
        fm = factors.momentum_score_smooth_m1(r4, r7)
        pen = r.other_pen * mdd_factor(r.R_MDD, r.water)
        num = ((r.wv * min(max(fv, 0), 100)) if pd.notna(fv) else 0) \
            + (r.wa * fa if pd.notna(fa) else 0) + (r.wm * fm if pd.notna(fm) else 0)
        den = (r.wv if pd.notna(fv) else 0) + (r.wa if pd.notna(fa) else 0) \
            + (r.wm if pd.notna(fm) else 0)
        rows.append(min(max((num / den if den else 0) * pen, 0), 100))
    # 截面rank的pct用于mom——注意动量rank必须基于全池原始r4/r7, 上面逐行rank(pct)已含
    return pd.Series(rows, index=g.index), (rank4, rank7)


def harvest_ew(codes):
    recs = []
    for d in DATES:
        t0 = time.time(); rows = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True): c for c in codes}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if not r.get("error"):
                        rows.append(r)
                except Exception:
                    pass
        from engine import finalize
        fdf = finalize(rows, as_of=d)
        for h in fdf.dropna(subset=["S_total"]).itertuples():
            pdt = h.penalty_detail or {}
            op = 1.0
            for n, p in (h.penalties or []):
                if "回撤比值" not in n:
                    op *= (1 - p)
            recs.append(dict(date=d, code=h.code, water=h.water,
                wv=h.w_value, wa=h.w_alpha, wm=h.w_mom,
                val_pct=h.val_pct, val_cov=h.val_coverage,
                trend_ok=bool(h.trend_ok), wr=h.ir_winrate,
                dc=h.down_capture, r4=h.mom_4m1m, r7=h.mom_7m1m,
                R_MDD=pdt.get("R_MDD"), other_pen=op))
        print(f"[harvest] {d} n={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(recs)


def main():
    base = pd.concat([pd.read_csv(f"output/factor_rows/{f}", dtype={"code": str})
                      for f in os.listdir("output/factor_rows") if f.endswith(".csv")])
    base = base[base.date.isin(DATES)].dropna(subset=["S_eng"]).copy()
    codes = sorted(base.code.unique())
    print(f"[plan] {len(DATES)} dates × {len(codes)} codes, EW半衰期={HL}d")

    rbsa.EW_HALFLIFE = HL     # 全局挂钩: 本进程内所有rbsa_weights走EW
    ew = harvest_ew(codes)
    ew.to_csv(OUT, index=False, encoding="utf-8-sig")

    # 逐日重建S并配对IC
    res = []
    for d in DATES:
        gb = base[base.date == d].set_index("code")
        ge = ew[ew.date == d].copy()
        if ge.empty:
            continue
        s_ix = ge.groupby("date", group_keys=False)
        ge["rank4"] = ge["r4"].rank(pct=True)
        ge["rank7"] = ge["r7"].rank(pct=True)
        _s, _ = s_score(ge)
        ge["S_ew"] = _s.values
        m = gb[["fwd6", "water"]].join(ge.set_index("code")[["S_ew"]], how="inner").dropna()
        # 基线IC (strategy_bt重建口径, 与EW同一配对样本, 保证公平)
        gbase = base[base.date == d].copy()
        gbase["rank4"] = gbase["r4"].rank(pct=True); gbase["rank7"] = gbase["r7"].rank(pct=True)
        sb_, _ = s_score(gbase); gbase["S_b"] = sb_.values
        mb = gbase.set_index("code")[["fwd6"]].join(ge.set_index("code")[["S_ew"]], how="inner").dropna()
        ic_b = stats.spearmanr(mb.index.map(gbase.set_index("code")["S_b"]), mb.fwd6)[0]
        ic_e = stats.spearmanr(mb.S_ew, mb.fwd6)[0]
        be = mb[mb.S_ew > 70]
        res.append(dict(date=d, n=len(mb), ic_base=ic_b, ic_ew=ic_e,
                        ew_buy_n=len(be), ew_buy_fwd6=be.fwd6.mean(),
                        ew_buy_win=(be.fwd6 > 0).mean() if len(be) else np.nan))
        print(f"{d}: n={len(mb)} IC_base={ic_b:+.4f} IC_ew={ic_e:+.4f} "
              f"EWBuy n={len(be)} fwd6{be.fwd6.mean() if len(be) else np.nan:+.1%}", flush=True)
    r = pd.DataFrame(res)
    tb = r.ic_base.mean() / (r.ic_base.std() / np.sqrt(len(r)))
    te = r.ic_ew.mean() / (r.ic_ew.std() / np.sqrt(len(r)))
    diff = r.ic_ew - r.ic_base
    td = diff.mean() / (diff.std() / np.sqrt(len(r)))
    print("\n===== 裁决 =====")
    print(f"基线: IC均值 {r.ic_base.mean():+.4f} t={tb:.2f}")
    print(f"EW{HL}: IC均值 {r.ic_ew.mean():+.4f} t={te:.2f}")
    print(f"配对差: 均值 {diff.mean():+.4f} t={td:.2f}  胜率(EW>base) {(diff>0).mean():.0%}")
    allb = pd.concat([pd.read_csv("output/factor_rows/" + d + ".csv", dtype={"code": str})
                      for d in DATES])
    print(f"EW Buy组汇总: n={r.ew_buy_n.sum()} 加权fwd6 "
          f"{np.nansum(r.ew_buy_n * r.ew_buy_fwd6) / max(r.ew_buy_n.sum(),1):+.1%}" if r.ew_buy_n.sum() else "EW Buy组为空")
    r.to_csv("output/rbsa_ew_verdict.csv", index=False, encoding="utf-8-sig")
    print("[输出] rbsa_ew_verdict.csv / rbsa_ew_rows.csv")

if __name__ == "__main__":
    main()
