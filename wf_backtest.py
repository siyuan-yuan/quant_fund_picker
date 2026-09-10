# -*- coding: utf-8 -*-
"""
月度频率 Walk-Forward 连续回测
2016-01 ~ 2025-07 每月最后一个交易日打分 → 前瞻6M/12M
产出: IC时序/显著性(t/ICIR)、按水位分组的regime表现、五分位多空价差
"""
import os, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import provider
provider.STALE_OK = True            # 复用现有缓存(历史净值/PE不变)

import rbsa
from engine import score_fund, finalize, market_water
from backtest import STYLE6, build_pool, forward_returns
from config import OUTPUT_DIR

START, END = "2016-01", "2025-07"
H6, H12 = 126, 252
os.makedirs(OUTPUT_DIR, exist_ok=True)


def month_end_points():
    close = provider.get_index_close("sh000300")
    m = close.loc[START + "-01":END + "-31"]
    return [(d.strftime("%Y-%m-%d"), d.strftime("%Y-%m"))
            for d in m.groupby([m.index.year, m.index.month]).apply(
                lambda g: g.index.max(), include_groups=False)]


def ic(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 15:
        return np.nan
    return float(spearmanr(d.a, d.b)[0])


def quintile_means(df, fwd):
    d = df[["S_total", fwd]].dropna()
    if len(d) < 25:
        return {}
    d = d.assign(q=pd.qcut(d["S_total"].rank(method="first"), 5, labels=False) + 1)
    return d.groupby("q")[fwd].mean().to_dict()


def main(workers=8):
    t0 = time.time()
    pool = build_pool(random_n=800)        # 确定性抽样 → 同160只池
    points = month_end_points()
    print(f"[wf] 月度点 {len(points)} 个 ({points[0][1]} ~ {points[-1][1]}), 基金池 {len(pool)}")

    date_rows, per_date = [], []
    for pi, (d, ym) in enumerate(points, 1):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True, indices=STYLE6): c for c in pool}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r.get("error"):
                        continue
                    f6, f12 = forward_returns(r["code"], d)
                    if f6 == f6:
                        r["fwd6"] = round(f6, 4)
                        date_rows.append(r)
                except Exception:
                    continue
        if not date_rows:
            continue
        df = finalize(date_rows, as_of=d)
        ok = df.dropna(subset=["S_total"])
        w = ok["water"].iloc[0] if len(ok) else np.nan
        rec = {"date": d, "ym": ym, "n": len(ok), "water": w,
               "avg_score": round(ok.S_total.mean(), 1),
               "avg_fvalue": round(ok.F_value.mean(), 1),
               "ic6": ic(ok.S_total, ok.fwd6),
               "ic_fv": ic(ok.F_value, ok.fwd6),
               "ic_fm": ic(ok.F_momentum, ok.fwd6),
               "ic_fa": ic(ok.F_alpha, ok.fwd6)}
        q6 = quintile_means(ok, "fwd6")
        rec["ls_q5q1"] = (q6.get(5, np.nan) - q6.get(1, np.nan)) if 5 in q6 and 1 in q6 else np.nan
        per_date.append(rec)
        if pi % 10 == 0 or pi == len(points):
            print(f"    {d} n={len(ok)} IC6={rec['ic6']:+.3f} 累计 {time.time()-t0:.0f}s", flush=True)
        date_rows = []

    sdf = pd.DataFrame(per_date)
    sdf.to_csv(f"{OUTPUT_DIR}/wf_dates.csv", index=False, encoding="utf-8-sig")

    def stat(x):
        x = x.dropna()
        n = len(x)
        if n < 5:
            return {}
        return {"n": n, "mean": round(x.mean(), 4), "std": round(x.std(), 4),
                "t": round(x.mean() / (x.std() / np.sqrt(n)), 2),
                "pct_pos": round((x > 0).mean(), 3)}

    regimes = {"低估区(≤20%)": sdf[sdf.water <= 0.20],
               "中性区(20-70%)": sdf[(sdf.water > 0.20) & (sdf.water <= 0.70)],
               "高估区(>70%)": sdf[sdf.water > 0.70]}
    regime_rows = [{"regime": k, "months": len(v), **{f"ic6_{kk}": vv for kk, vv in stat(v.ic6).items()},
                    "ls_mean": round(v.ls_q5q1.mean(), 4) if len(v.ls_q5q1.dropna()) else None,
                    "ls_t": stat(v.ls_q5q1).get("t")} for k, v in regimes.items()]
    rdf = pd.DataFrame(regime_rows)
    rdf.to_csv(f"{OUTPUT_DIR}/wf_regimes.csv", index=False, encoding="utf-8-sig")

    print("\n===== 全期 IC6 统计 =====")
    print(stat(sdf.ic6))
    print("\n===== 分水位 regime =====")
    print(rdf.to_string(index=False))
    json.dump({"dates": per_date, "overall_ic6": stat(sdf.ic6),
               "overall_ls": stat(sdf.ls_q5q1), "runtime_s": round(time.time() - t0)},
              open(f"{OUTPUT_DIR}/wf_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[saved] output/wf_*.csv/json | 用时 {round(time.time()-t0)}s")


if __name__ == "__main__":
    main()
