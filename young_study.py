# -*- coding: utf-8 -*-
"""
否决研究 Part C: "不满3年直接排除" 的机会成本
young组 = 同类型、近3年为空但近1年非空 (年龄1~3岁, 被漏斗 dropna(近3年) 砍掉)
对照组 = 当前550只榜单池 (老基金, 缓存热)
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import provider
provider.STALE_OK = True
import factors
from config import CACHE_DIR

W2 = ("2024-07-31", "2026-07-31")   # 2年窗口 (young组当时1~3岁)
W1 = ("2025-07-31", "2026-07-31")   # 1年窗口
TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}


def win_ret(adj, t0, t1):
    adj = adj.dropna()
    if len(adj) < 30:
        return np.nan
    i0 = adj.index.searchsorted(pd.Timestamp(t0), side="right") - 1
    i1 = adj.index.searchsorted(pd.Timestamp(t1), side="right") - 1
    if i0 < 0 or i1 <= i0 or i1 >= len(adj):
        i1 = min(i1, len(adj) - 1)
        if i0 < 0 or i1 <= i0:
            return np.nan
    return float(adj.iloc[i1] / adj.iloc[i0] - 1)


def first_date(adj):
    return adj.index[0] if len(adj) else None


def main():
    t0 = time.time()
    rank = pd.read_csv(f"{CACHE_DIR}/rank_all.csv", dtype={"基金代码": str})
    meta = provider.get_fund_meta()
    df = rank.join(meta[["基金类型"]], on="基金代码")
    df = df[df["基金类型"].isin(TYPES)]
    df = df[~df["基金简称"].str.strip().str.endswith(("C", "E"))]
    for c in ["近3年", "近1年", "近6月"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    old = df.dropna(subset=["近3年"])
    young = df[df["近3年"].isna() & df["近1年"].notna() & df["近6月"].notna()]
    print(f"[宇宙] 同类型老基金(≥3年) {len(old)} 只 | 新基金(1~3岁, 被排除) {len(young)} 只", flush=True)

    samp = young.sample(min(160, len(young)), random_state=11)
    navs = {}

    def get(c):
        try:
            return c, provider.get_fund_nav(c)
        except Exception:
            return c, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(get, c) for c in samp["基金代码"]]
        for i, fut in enumerate(as_completed(futs), 1):
            c, nav = fut.result()
            if nav is not None and len(nav):
                navs[c] = nav.set_index("date")["ret"]
            if i % 40 == 0:
                print(f"  拉净值 {i}/{len(futs)} ({time.time()-t0:.0f}s)", flush=True)

    name_map = dict(zip(samp["基金代码"], samp["基金简称"]))
    rows = []
    for c, ret in navs.items():
        adj = (1 + ret.fillna(0)).cumprod()
        rows.append(dict(code=c, name=name_map.get(c, c),
                         start=str(first_date(adj).date()),
                         w2=win_ret(adj, *W2), w1=win_ret(adj, *W1),
                         r4_24=(lambda x: x[0])(factors.lagged_momentum_returns(adj[adj.index <= "2024-07-31"])) if len(adj[adj.index <= "2024-07-31"]) > 130 else np.nan,
                         r4_25=(lambda x: x[0])(factors.lagged_momentum_returns(adj[adj.index <= "2025-07-31"])) if len(adj[adj.index <= "2025-07-31"]) > 130 else np.nan))
    ydf = pd.DataFrame(rows)

    # 老基金对照池
    old_rows = []
    scan = pd.read_csv("output/scan_20260802.csv", dtype={"code": str})
    for c in scan["code"]:
        try:
            nav = provider.get_fund_nav(c).set_index("date")
            adj = (1 + nav["ret"].fillna(0)).cumprod()
            old_rows.append(dict(code=c, w2=win_ret(adj, *W2), w1=win_ret(adj, *W1),
                                 r4_24=factors.lagged_momentum_returns(adj[adj.index <= "2024-07-31"])[0],
                                 r4_25=factors.lagged_momentum_returns(adj[adj.index <= "2025-07-31"])[0]))
        except Exception:
            pass
    odf = pd.DataFrame(old_rows)

    hs = provider.get_index_close("sh000300")
    hs2, hs1 = win_ret(hs, *W2), win_ret(hs, *W1)

    print(f"\n===== 窗口 {W2[0]}~{W2[1]} (2年) | 沪深300 {hs2:+.1%} =====")
    for lab, d in [("新基金(被排除)", ydf.w2.dropna()), ("老基金榜单池", odf.w2.dropna())]:
        print(f"  {lab} n={len(d)}: 均值{d.mean():+.1%} 中位{d.median():+.1%} "
              f"P25 {d.quantile(.25):+.1%} P75 {d.quantile(.75):+.1%} "
              f">0占比{(d>0).mean():.0%} >+50%占比{(d>0.5).mean():.0%} 跑赢HS300占比{(d>hs2).mean():.0%}")
    print(f"\n===== 窗口 {W1[0]}~{W1[1]} (1年) | 沪深300 {hs1:+.1%} =====")
    for lab, d in [("新基金(被排除)", ydf.w1.dropna()), ("老基金榜单池", odf.w1.dropna())]:
        print(f"  {lab} n={len(d)}: 均值{d.mean():+.1%} 中位{d.median():+.1%} "
              f">0占比{(d>0).mean():.0%} >+30%占比{(d>0.3).mean():.0%} 跑赢HS300占比{(d>hs1).mean():.0%}")

    # 动量通道能否提前锁定新基金赢家: 4M-1M动量在合并宇宙中的分位
    for tag, dc, dr in [("2024-07-31", "r4_24", "w2"), ("2025-07-31", "r4_25", "w1")]:
        allr = pd.concat([ydf[["code", dc]].dropna().assign(grp="young"),
                          odf[["code", dc]].dropna().assign(grp="old")])
        allr["pct"] = allr[dc].rank(pct=True)
        yp = allr[allr.grp == "young"]
        print(f"\n[动量@{tag}] 新基金 4M-1M动量合并宇宙分位: 均值{yp['pct'].mean():.2f} "
              f"前30%占比{(yp['pct']>0.7).mean():.0%} 前10%占比{(yp['pct']>0.9).mean():.0%}")

    print("\n===== 新基金中的大赢家 (2年窗口Top10) =====")
    top = ydf.dropna(subset=["w2"]).nlargest(10, "w2")[["code", "name", "start", "w2", "w1"]]
    print(top.to_string(index=False))
    ydf.to_csv("output/young_rows.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] output/young_rows.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
