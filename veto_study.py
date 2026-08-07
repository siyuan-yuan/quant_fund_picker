# -*- coding: utf-8 -*-
"""
否决研究 Part A: R_MDD 一票否决/腰斩 是否过度?
季度末 PiT 打分 2019Q1~2025Q4 × 全池, 落盘 output/veto_rows/{date}.csv
"""
import os, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import provider
provider.STALE_OK = True

import rbsa
from engine import score_fund, finalize
from backtest import build_pool, forward_returns

OUT = "output/veto_rows"
os.makedirs(OUT, exist_ok=True)
DATES = [str(d.date()) for d in pd.date_range("2006-03-31", "2026-03-31", freq="QE")]


def main():
    pool = build_pool(random_n=800)          # 与 014193 案同一池 (seed 相同, 缓存热)
    t0 = time.time()
    for d in DATES:
        fp = f"{OUT}/{d}.csv"
        if os.path.exists(fp):
            print(f"[skip] {d}", flush=True)
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
        ok["rank_pct"] = ok.S_total.rank(pct=True)
        recs = []
        for _, h in ok.iterrows():
            f6, f12 = forward_returns(h.code, d)
            pdt = h.penalty_detail or {}
            recs.append(dict(
                date=d, code=h.code, name=h.name, S=h.S_total,
                rank_pct=round(float(h.rank_pct), 4),
                F_value=h.F_value, F_alpha=h.F_alpha, F_mom=h.F_momentum,
                water=h.water, wv=h.w_value, wa=h.w_alpha, wm=h.w_mom,
                R_MDD=pdt.get("R_MDD"), mdd_f=pdt.get("mdd_fund"),
                mdd_b=pdt.get("mdd_bench"), top3=pdt.get("top3_conc"),
                pens=h.penalty_str,
                fwd6=None if f6 != f6 else round(f6, 4),
                fwd12=None if f12 != f12 else round(f12, 4)))
        pd.DataFrame(recs).to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"[done] {d} n={len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
