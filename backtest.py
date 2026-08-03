# -*- coding: utf-8 -*-
"""
Step 2D — 历史回测 (Point-in-Time)
验证打分在 2018/2021/2024 拐点与未来6/12个月收益的关系
判定指标: Rank IC + 五分位组合收益 + 拐点择时叙事
注意: 采用风格6面板(PE历史自2005年起充分);
      经理任期/AUM/缩水加分为当前快照, 回测模式禁用并披露
"""
import os, time, json, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import provider, rbsa
from engine import score_fund, finalize
from config import RBSA_INDICES, CACHE_DIR, OUTPUT_DIR

STYLE6 = [x for x in RBSA_INDICES if x[0] == "sina"]
STYLE6_NAMES = [x[2] for x in STYLE6]

POINTS = [
    ("2018-01-26", "2018 熊顶·3587"),
    ("2018-10-19", "2018 政策底·2449"),
    ("2019-01-04", "2019 双底·2440"),
    ("2021-02-10", "2021 茅指数顶"),
    ("2021-12-13", "2021 岁末双顶"),
    ("2022-04-26", "2022 疫情底·2863"),
    ("2024-02-05", "2024 崩盘底·2635"),
    ("2024-05-20", "2024 反弹顶·3174"),
    ("2024-09-23", "2024·924前夜·2748"),
]
FIRST_NAV_MAX = "2014-12-31"   # 基金须在2018回测点前满3年
LAST_NAV_MIN = "2025-09-23"    # 保证最后一个回测点有12个月前瞻
H6, H12 = 126, 252
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- 无幸存者偏差的随机池 ----------------
def build_pool(random_n=500, seed=7, workers=6):
    rank = pd.read_csv(f"{CACHE_DIR}/rank_all.csv", dtype={"基金代码": str})
    meta = provider.get_fund_meta()
    df = rank.join(meta[["基金类型"]], on="基金代码")
    df = df[df["基金类型"].isin({"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"})]
    df = df[~df["基金简称"].str.strip().str.endswith(("C", "E"))]
    df["近3年"] = pd.to_numeric(df["近3年"], errors="coerce")
    base = df.dropna(subset=["近3年"])["基金代码"].tolist()

    extra = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("scan_") and f.endswith(".csv"):
            extra += pd.read_csv(f"{OUTPUT_DIR}/{f}", dtype={"code": str})["code"].tolist()
    cand = list(dict.fromkeys(extra + list(np.random.RandomState(seed).choice(base, random_n, replace=False))))
    print(f"[池] 候选 {len(cand)} 只, 开始拉取净值...")

    ok_codes = []
    t0 = time.time()
    def load(c):
        try:
            return c, provider.get_fund_nav(c)
        except Exception:
            return c, None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (c, nav) in enumerate(ex.map(load, cand), 1):
            if nav is not None and len(nav) and \
               str(nav["date"].min().date()) <= FIRST_NAV_MAX and \
               str(nav["date"].max().date()) >= LAST_NAV_MIN:
                ok_codes.append(c)
            if i % 100 == 0:
                print(f"    nav {i}/{len(cand)} 合格 {len(ok_codes)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[池] 满足 2014-12-31 前成立 & 数据贯穿至2025-09 的基金: {len(ok_codes)} 只")
    return ok_codes


# ---------------- 前瞻收益 ----------------
def forward_returns(code, as_of):
    nav = provider.get_fund_nav(code).set_index("date")
    adj = (1 + nav["ret"].fillna(0)).cumprod()
    ts = pd.Timestamp(as_of)
    pos = adj.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return np.nan, np.nan
    p0 = adj.iloc[pos]
    f6 = adj.iloc[pos + H6] / p0 - 1 if pos + H6 < len(adj) else np.nan
    f12 = adj.iloc[pos + H12] / p0 - 1 if pos + H12 < len(adj) else np.nan
    return f6, f12


def hs300_forward(as_of):
    close = provider.get_index_close("sh000300")
    ts = pd.Timestamp(as_of)
    pos = close.index.searchsorted(ts, side="right") - 1
    p0 = close.iloc[pos]
    f6 = close.iloc[pos + H6] / p0 - 1 if pos + H6 < len(close) else np.nan
    f12 = close.iloc[pos + H12] / p0 - 1 if pos + H12 < len(close) else np.nan
    return round(f6, 4), round(f12, 4)


# ---------------- 指标 ----------------
def ic(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 15:
        return np.nan, len(d)
    r, _ = spearmanr(d[x], d[y])
    return round(r, 3), len(d)


def quintiles(df, fwd):
    d = df[["S_total", fwd]].dropna()
    if len(d) < 25:
        return {}
    d = d.copy()
    d["q"] = pd.qcut(d["S_total"].rank(method="first"), 5, labels=False) + 1
    g = d.groupby("q")[fwd].agg(["mean", "count"])
    return {int(q): (round(m, 4), int(c)) for q, (m, c) in g.iterrows()}


# ---------------- 主流程 ----------------
def main():
    t0 = time.time()
    pool = build_pool()
    results = {}
    for d, label in POINTS:
        rows = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(score_fund, c, as_of=d, bt=True, indices=STYLE6): c for c in pool}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r.get("error"):
                        continue
                    f6, f12 = forward_returns(r["code"], d)
                    r["fwd6"], r["fwd12"] = round(f6, 4) if f6 == f6 else np.nan, \
                                            round(f12, 4) if f12 == f12 else np.nan
                    rows.append(r)
                except Exception:
                    continue
        df = finalize(rows, as_of=d)          # V3.2: 回测日水位计(PiT)自适应权重
        df["date"], df["label"] = d, label
        results[d] = df
        ok = df.dropna(subset=["S_total"])
        print(f"[{d} {label}] 有效基金 {len(ok)} | 均分 {ok.S_total.mean():.1f} | "
              f"F_value均值 {ok.F_value.mean():.1f} | 用时累计 {time.time()-t0:.0f}s", flush=True)

    # ---- 汇总指标 ----
    summary, factor_ics, quint_rows = [], [], []
    for d, label in POINTS:
        df = results[d]
        m6, m12 = hs300_forward(d)
        ic6, n6 = ic(df, "S_total", "fwd6")
        ic12, n12 = ic(df, "S_total", "fwd12")
        pe_pct = rbsa.index_pe_percentile(indices=STYLE6, as_of=d)
        avg_pepct = float(np.nanmean(list(pe_pct.values())))
        ok = df.dropna(subset=["S_total"])
        summary.append({"date": d, "label": label, "n": len(ok),
                        "avg_score": round(ok.S_total.mean(), 1),
                        "pct_below30": round((ok.S_total < 30).mean(), 3),
                        "avg_pe_pct": round(avg_pepct, 3),
                        "hs300_fwd6": m6, "hs300_fwd12": m12,
                        "ic6": ic6, "ic12": ic12})
        for fac in ["F_value", "F_alpha", "F_momentum"]:
            v6, _ = ic(df, fac, "fwd6")
            v12, _ = ic(df, fac, "fwd12")
            factor_ics.append({"date": d, "label": label, "factor": fac, "ic6": v6, "ic12": v12})
        q6, q12 = quintiles(df, "fwd6"), quintiles(df, "fwd12")
        for q in range(1, 6):
            quint_rows.append({"date": d, "label": label, "Q": q,
                               "fwd6": q6.get(q, (np.nan, 0))[0], "n6": q6.get(q, (0, 0))[1],
                               "fwd12": q12.get(q, (np.nan, 0))[0], "n12": q12.get(q, (0, 0))[1]})

    sdf = pd.DataFrame(summary)
    fidf = pd.DataFrame(factor_ics)
    qdf = pd.DataFrame(quint_rows)
    sdf.to_csv(f"{OUTPUT_DIR}/bt_summary.csv", index=False, encoding="utf-8-sig")
    fidf.to_csv(f"{OUTPUT_DIR}/bt_factor_ic.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(f"{OUTPUT_DIR}/bt_quintiles.csv", index=False, encoding="utf-8-sig")
    for d in results:
        results[d].to_json(f"{OUTPUT_DIR}/bt_raw_{d}.json", orient="records",
                           force_ascii=False, default_handler=str)

    # ---- 拐点叙事 ----
    print("\n===== Rank IC (S_total vs 未来收益) =====")
    print(sdf[["date", "label", "n", "avg_score", "avg_pe_pct",
               "hs300_fwd6", "ic6", "ic12"]].to_string(index=False))
    print("\n===== 因子IC均值 =====")
    print(fidf.groupby("factor")[["ic6", "ic12"]].mean().round(3).to_string())
    print("\n===== 五分位组合(全日期聚合平均) =====")
    agg = qdf.groupby("Q")[["fwd6", "fwd12"]].mean().round(4)
    print(agg.to_string())
    spread6 = agg.loc[5, "fwd6"] - agg.loc[1, "fwd6"]
    spread12 = agg.loc[5, "fwd12"] - agg.loc[1, "fwd12"]
    print(f"\nQ5-Q1 价差: 6M {spread6:+.2%} | 12M {spread12:+.2%}")
    print(f"IC均值: 6M {sdf.ic6.mean():+.3f} | 12M {sdf.ic12.mean():+.3f}")

    meta = {"summary": summary, "factor_ic": factor_ics, "quintiles": quint_rows,
            "pool_n": len(pool), "runtime_s": round(time.time() - t0)}
    json.dump(meta, open(f"{OUTPUT_DIR}/bt_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[saved] output/bt_*.csv, bt_meta.json | 总用时 {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
