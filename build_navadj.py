#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #11（D0.2）：复权口径统一 + 分红/折算事件对账

口径规则（预登记，先于执行结果固定）:
  adj_nav_0 = nav_0 ;  adj_nav_t = adj_nav_{t-1} * (1 + ret_t)   # ret = 东财官方复权日增长率
  事件日判定: |ret_t - nav.pct_change_t| > 0.005 (0.5pp)
  事件分类（v2 修订，修订理由与计数同时入库；对结论零影响——adj 一律由官方 ret 构建）:
    nav 变动 < 0 且 ret > nav 变动            → "分红/折算(官方收益回补)"
    nav 变动 > +15% 且官方 ret 未同步跳变      → "份额折算(上拆, 分级上折/基金拆分)"
    其余                                      → "数据异常(待审)"
处置规则: 事件不参与任何裁决；adj 序列只由官方 ret 构建；不一致天数仅登记。
产物:
  cache/navadj_<code>.csv   (date, adj_nav)  仅研究池成员, 不覆盖原始文件
  output/v5/d02_audit_events.csv             全部事件日线
  output/v5/d02_fund_summary.csv             每只基金汇总
  output/v5/d02_summary.md                   全局摘要(事件数/影响分布/口径差异年化估计)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

CACHE, OUT = "cache", "output/v5"
UNIVERSE = "output/p1_panel/universe.csv"
EVENT_EPS = 0.005
os.makedirs(OUT, exist_ok=True)


def build_one(code: str):
    """返回 (adj_series, events_df, fund_row, error)"""
    p = f"{CACHE}/nav_{code}.csv"
    if not os.path.exists(p):
        return None, None, dict(code=code, error="无净值文件"), "无净值文件"
    try:
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df[~df["date"].duplicated(keep="last")]
        df = df[df["nav"] > 0]
        if len(df) < 2:
            return None, None, dict(code=code, error="行数不足"), "行数不足"
        ret = df["ret"].fillna(0.0).values
        nav = df["nav"].values
        adj = np.empty(len(df))
        adj[0] = nav[0]
        np.multiply.accumulate(1.0 + ret[1:], out=adj[1:])
        adj[1:] *= adj[0]

        nav_chg = np.empty(len(df)); nav_chg[0] = 0.0
        nav_chg[1:] = nav[1:] / nav[:-1] - 1.0
        implied = ret - nav_chg
        ev_mask = np.abs(implied) > EVENT_EPS
        n_ev = int(ev_mask.sum())
        events = None
        if n_ev:
            kind = np.where((nav_chg[ev_mask] < 0) & (ret[ev_mask] > nav_chg[ev_mask]),
                            "分红/折算(官方收益回补)",
                            np.where((nav_chg[ev_mask] > 0.15)
                                     & (np.abs(ret[ev_mask] - nav_chg[ev_mask]) > 0.15),
                                     "份额折算(上拆)", "数据异常(待审)"))
            events = pd.DataFrame(dict(
                code=code, date=df["date"].values[ev_mask],
                nav_chg=np.round(nav_chg[ev_mask], 6), off_ret=np.round(ret[ev_mask], 6),
                implied=np.round(implied[ev_mask], 6), kind=kind))
        yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        tot_nav = nav[-1] / nav[0] - 1.0
        tot_adj = adj[-1] / adj[0] - 1.0
        drag = ((1 + tot_adj) ** (1 / yrs) - (1 + tot_nav) ** (1 / yrs)) if yrs > 0 else np.nan
        row = dict(code=code, n=len(df), first=str(df["date"].iloc[0].date()),
                   last=str(df["date"].iloc[-1].date()), n_events=n_ev, yrs=round(yrs, 2),
                   ann_drag=round(drag, 6) if yrs >= 1 else np.nan)
        return pd.Series(adj, index=df["date"], name="adj_nav"), events, row, None
    except Exception as e:
        return None, None, dict(code=code, error=str(e)[:80]), str(e)[:80]


def _worker(code):
    adj, ev, row, err = build_one(code)
    if adj is not None:
        out = pd.DataFrame(dict(date=adj.index, adj_nav=np.round(adj.values, 6)))
        out.to_csv(f"{CACHE}/navadj_{code}.csv", index=False)
    return row, ev


def main(workers: int = 4):
    # 研究池（与 p1_panel_build 同源；文件缺则按原式重建）
    if os.path.exists(UNIVERSE):
        uni = pd.read_csv(UNIVERSE, dtype={"code": str})
        codes = uni["code"].tolist()
    else:
        import p1_panel_build
        uni = p1_panel_build.build_universe()
        codes = uni.index.tolist()
        print(f"[D0.2] universe 重建: {len(codes)} 只")
    print(f"[D0.2] 构建复权序列: {len(codes)} 只 …", flush=True)

    rows, evs, errs = [], [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (row, ev) in enumerate(ex.map(_worker, codes, chunksize=32), 1):
            rows.append(row)
            if ev is not None:
                evs.append(ev)
            if "error" in row and row["error"]:
                errs += 1
            if i % 500 == 0:
                print(f"  {i}/{len(codes)} ({errs} err)", flush=True)

    fs = pd.DataFrame(rows)
    fs.to_csv(f"{OUT}/d02_fund_summary.csv", index=False, encoding="utf-8-sig")
    ev_all = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
    if len(ev_all):
        ev_all.to_csv(f"{OUT}/d02_audit_events.csv", index=False, encoding="utf-8-sig")

    ok = fs[fs.get("error").isna()] if "error" in fs.columns else fs
    n_ok = int((fs["error"].isna()).sum()) if "error" in fs.columns else len(fs)
    kinds = ev_all["kind"].value_counts().to_dict() if len(ev_all) else {}
    drag = ok["ann_drag"].dropna() if len(ok) else pd.Series(dtype=float)
    d3 = ok.loc[ok["yrs"] >= 3, "ann_drag"].dropna()
    L = ["# D0.2 复权口径统一 · 全池对账摘要", "",
         f"- 池：{len(codes)} 只；成功构建 adj：{n_ok}（err {errs}）",
         f"- 事件日总数：{len(ev_all):,}（覆盖 {ev_all.code.nunique() if len(ev_all) else 0} 只）",
         f"- 事件分类：{kinds}",
         f"- raw nav vs adj 年化差（n={len(d3):,}, ≥3y 序列）：均值 {d3.mean():+.4%}/yr、"
         f"中位 {d3.median():+.3%}、P95 {d3.quantile(.95):+.2%}、最大 {d3.max():+.2%}",
         "",
         "口径声明：adj 序列（cache/navadj_*）自此为 V5 唯一入账口径（D0.6 硬规则 1）；",
         "事件分类仅登记不裁决；『数据异常(待审)』类按 D0.6 立案并规则化处理。"]
    open(f"{OUT}/d02_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
