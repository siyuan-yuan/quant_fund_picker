# -*- coding: utf-8 -*-
"""
提案裁决: 季度频率(2016-2025, 38个点)三方案对比
  base    = 现行 V3.2 (硬切估值 + 水位regime权重)
  v1      = 提案① 单基金动量覆盖 (F_mom>=85 & 站上MA20 → 权重0.15/0.35/0.50)
  v3      = 提案③ sigmoid估值 (中点55%ile, 替代线性)
"""
import time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True
from engine import score_fund, finalize, market_water, resolve_weights
from backtest import STYLE6, build_pool, forward_returns
from config import OUTPUT_DIR


def quarter_ends():
    close = provider.get_index_close("sh000300")
    m = close.loc["2016-01-01":"2025-07-31"]
    g = m.groupby([m.index.year, m.index.quarter])
    return [d.strftime("%Y-%m-%d") for d in g.apply(lambda x: x.index.max(), include_groups=False)]


def pen_prod(pens):
    return float(np.prod([1 - p for _, p in (pens or [])])) if pens else 1.0


def sigmoid_fv(p):
    return np.nan if pd.isna(p) else 100.0 / (1.0 + np.exp((p - 0.55) / 0.08))


def variants(row, base_w):
    """row: 引擎原始因子; base_w: (wv,wa,wm) 当日水位权重"""
    wv, wa, wm = base_w
    pp = pen_prod(row.get("penalties"))
    fv, fa, fm = row.get("F_value"), row.get("F_alpha"), row.get("F_momentum")
    def assemble(w1, w2, w3, fv_use):
        num, den = 0.0, 0.0
        if fv_use == fv_use and fv_use is not None: num += w1*min(fv_use,100); den += w1
        if fa == fa and fa is not None: num += w2*fa; den += w2
        if fm == fm and fm is not None: num += w3*fm; den += w3
        return (num/den if den else 0.0) * pp
    s_base = assemble(wv, wa, wm, fv)
    # 提案① 动量覆盖
    if fm == fm and fm is not None and fm >= 85 and row.get("trend_ma20"):
        s_v1 = assemble(0.15, 0.35, 0.50, fv)
    else:
        s_v1 = s_base
    # 提案③ sigmoid
    s_v3 = assemble(wv, wa, wm, sigmoid_fv(row.get("val_pct")))
    return s_base, s_v1, s_v3


def ic(df, col):
    d = df[[col, "fwd6"]].dropna()
    return float(d[col].corr(d["fwd6"], method="spearman")) if len(d) >= 20 else np.nan


def ls(df, col):
    d = df[[col, "fwd6"]].dropna()
    if len(d) < 30:
        return np.nan
    d = d.assign(q=pd.qcut(d[col].rank(method="first"), 5, labels=False) + 1)
    g = d.groupby("q")["fwd6"].mean()
    return float(g.get(5, np.nan) - g.get(1, np.nan))


def main():
    t0 = time.time()
    pool = build_pool(random_n=800)
    recs = []
    ledger = []
    for i, d in enumerate(quarter_ends(), 1):
        rows = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True, indices=STYLE6): c for c in pool}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if not r.get("error"):
                        f6, _ = forward_returns(r["code"], d)
                        if f6 == f6:
                            r["fwd6"] = round(f6, 4)
                            rows.append(r)
                except Exception:
                    pass
        df = finalize(rows, as_of=d)
        ok = df.dropna(subset=["S_total"]).copy()
        water = float(ok.water.iloc[0]) if len(ok) else np.nan
        (wv, wa, wm), _ = resolve_weights(water)
        res = ok.apply(lambda r: variants(r, (wv, wa, wm)), axis=1, result_type="expand")
        ok["v1"], ok["v3"] = res[1], res[2]
        ic_b, ic1, ic3 = ic(ok, "S_total"), ic(ok, "v1"), ic(ok, "v3")
        ls_b, ls1, ls3 = ls(ok, "S_total"), ls(ok, "v1"), ls(ok, "v3")
        recs.append({"date": d, "n": len(ok), "water": water,
                     "ic_base": ic_b, "ic_v1": ic1, "ic_v3": ic3,
                     "ls_base": ls_b, "ls_v1": ls1, "ls_v3": ls3})
        hot = ok[ok.v1 >= 70]
        for _, h in hot.iterrows():
            ledger.append({"date": d, "code": h.code, "name": h["name"], "water": water,
                           "v1": round(h.v1,1), "base": round(h.S_total,1), "fwd6": h.fwd6})
        print(f"  {d} n={len(ok)} IC b/v1/v3 = {ic_b:+.3f}/{ic1:+.3f}/{ic3:+.3f} "
              f"v1过线 {len(hot)} ({time.time()-t0:.0f}s)", flush=True)

    sdf = pd.DataFrame(recs)
    sdf.to_csv(f"{OUTPUT_DIR}/wf_variants.csv", index=False, encoding="utf-8-sig")
    ldf = pd.DataFrame(ledger)
    ldf.to_csv(f"{OUTPUT_DIR}/wf_v1_ledger.csv", index=False, encoding="utf-8-sig")

    def stat(col):
        x = sdf[col].dropna()
        return f"均值{x.mean():+.4f} t={x.mean()/(x.std()/np.sqrt(len(x))):+.2f} 正占比{(x>0).mean():.0%}"
    print("\n===== 季度Walk-Forward (2016-2025, n=%d) =====" % len(sdf))
    print("IC6  base:", stat("ic_base"), "| v1:", stat("ic_v1"), "| v3:", stat("ic_v3"))
    print("L/S  base:", stat("ls_base"), "| v1:", stat("ls_v1"), "| v3:", stat("ls_v3"))
    if len(ldf):
        print(f"\n===== 提案①'过线单'总账: {len(ldf)} 次买入信号 =====")
        print(f"平均随后6个月收益: {ldf.fwd6.mean():+.1%} | 中位 {ldf.fwd6.median():+.1%} | 胜率 {(ldf.fwd6>0).mean():.0%}")
        w = ldf.copy()
        w["wm"] = pd.cut(w.water, [0,0.2,0.7,1.01], labels=["低估","中性","高估"])
        print(w.groupby("wm", observed=True).fwd6.agg(["mean","count"]).round(3).to_string())
        print("\n最惨5单:"); print(w.nsmallest(5,"fwd6")[["date","name","v1","base","fwd6"]].to_string(index=False))
        print("最好5单:"); print(w.nlargest(5,"fwd6")[["date","name","v1","base","fwd6"]].to_string(index=False))


if __name__ == "__main__":
    main()
