#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #12（D0.3）：全收益基准构建（H00300 沪深300全收益）——用户本机执行

为什么必须用户侧：执行环境网络出口封锁；且现有 pe 缓存仅 (date, pe) 两列、
无股息率序列，TR 无法在沙箱离线重建。本脚本在本机联网环境运行，产出落入仓库，
随后全链路复跑即可。

口径规则（预登记）：
  基准 = 中证指数公司官方 H00300（沪深300全收益，日线收盘价）
  数据源优先级：① akshare stock_zh_index_hist_csindex("H00300")
                ② akshare index_zh_a_hist(symbol="H00300")（东财镜像）
                ③ 若 ①② 皆不可得 → 脚本退出码 2，绝不静默降级为价格指数拟合
  对账闸门（任一不过 → 退出码 3，产物标注 REJECTED 不得使用）：
    G1 覆盖：2006-01-01 → 2026-03-31（允许起点晚于 2006-01 但不晚于 2006-07）
    G2 隐含股息率 = TR/价格指数累计偏离年化 ∈ [0.5%, 5.0%]（沪深300 历史经验带）
    G3 与价格指数日收益差异 |r_tr − r_px| 的 P99 < 1.5%（防错码/串列污染）
    G4 无负值、无重复日期、单调日期索引

产出：
  cache/bench_tr_h00300.csv          (date, tr_close)          ← 唯一入账 TR 口径
  output/v5/d03_tr_reconciliation.md 对账报告（闸门结果 + 隐含股息率逐年表）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

OUT_MD = "output/v5/d03_tr_reconciliation.md"
OUT_CSV = "cache/bench_tr_h00300.csv"
PX_CACHE = "cache/idx_sh000300.csv"     # 已有价格指数缓存（用于 G2/G3 对账）


def fetch_h00300(start: str, end: str) -> pd.DataFrame | None:
    """按优先级尝试数据源；返回 (date, close) 或 None。"""
    try:
        import akshare as ak
    except ImportError:
        print("❌ 需要 akshare（pip install akshare）"); sys.exit(2)
    try:
        df = ak.stock_zh_index_hist_csindex(symbol="H00300", start_date=start.replace("-", ""),
                                            end_date=end.replace("-", ""))
        if df is not None and len(df) > 100:
            dcol = "日期" if "日期" in df.columns else "trade_date"
            ccol = "收盘" if "收盘" in df.columns else "close"
            out = df[[dcol, ccol]].rename(columns={dcol: "date", ccol: "tr_close"})
            out["date"] = pd.to_datetime(out["date"])
            return out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [源①csindex] 失败: {e}")
    try:
        df = ak.index_zh_a_hist(symbol="H00300", period="daily",
                                start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        if df is not None and len(df) > 100:
            out = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "tr_close"})
            out["date"] = pd.to_datetime(out["date"])
            return out.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [源②东财镜像] 失败: {e}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default="2026-03-31")
    args = ap.parse_args()

    os.makedirs("output/v5", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    tr = fetch_h00300(args.start, args.end)
    if tr is None:
        print("❌ 全部数据源不可得。按预登记不静默降级，退出码 2。")
        return 2

    tr = tr.dropna().drop_duplicates("date", keep="last").sort_values("date")
    tr = tr[(tr.date >= args.start) & (tr.date <= args.end)]
    report = ["# D0.3 全收益基准对账报告 (H00300)", ""]
    fails = []

    first, last = tr.date.iloc[0], tr.date.iloc[-1]
    g1 = first <= pd.Timestamp("2006-07-01") and last >= pd.Timestamp("2026-03-01")
    report.append(f"- G1 覆盖 {first.date()} → {last.date()} ({len(tr)} 行) → {'✅' if g1 else '❌'}")
    if not g1: fails.append("G1")

    g4 = (tr.tr_close > 0).all() and tr.date.is_monotonic_increasing and not tr.date.duplicated().any()
    report.append(f"- G4 卫生（正价/单调/无重复）→ {'✅' if g4 else '❌'}")
    if not g4: fails.append("G4")

    px = None
    if os.path.exists(PX_CACHE):
        raw = pd.read_csv(PX_CACHE)
        dc = [c for c in raw.columns if "date" in c.lower() or c == "日期"][0]
        cc = [c for c in raw.columns if c.lower() in ("close", "收盘")][0]
        px = raw.rename(columns={dc: "date", cc: "px"})[["date", "px"]]
        px["date"] = pd.to_datetime(px["date"])
    if px is None:
        report.append("- ⚠️ 缺价格指数缓存，G2/G3 跳过（随后在有缓存环境复核）")
    else:
        m = tr.merge(px, on="date", how="inner").dropna()
        r_tr = m.tr_close.pct_change(); r_px = m.px.pct_change()
        p99 = (r_tr - r_px).abs().quantile(0.99)
        g3 = p99 < 0.015
        report.append(f"- G3 |r_tr − r_px| P99 = {p99:.4%} → {'✅' if g3 else '❌'}")
        if not g3: fails.append("G3")
        m["ratio"] = m.tr_close / m.px
        m["year"] = m.date.dt.year
        yrows = []
        for y, g in m.groupby("year"):
            if len(g) > 40:
                yrs = (g.date.iloc[-1] - g.date.iloc[0]).days / 365.25
                rd = (g.ratio.iloc[-1] / g.ratio.iloc[0]) ** (1 / max(yrs, 1e-9)) - 1
                yrows.append((int(y), rd))
        overall_yrs = (m.date.iloc[-1] - m.date.iloc[0]).days / 365.25
        overall = (m["ratio"].iloc[-1] / m["ratio"].iloc[0]) ** (1 / overall_yrs) - 1
        g2 = 0.005 <= overall <= 0.05
        report.append(f"- G2 隐含股息率(全样本年化) = {overall:.2%} ∈ [0.5%, 5%] → {'✅' if g2 else '❌'}")
        if not g2: fails.append("G2")
        report.append("\n| 年份 | 隐含股息率(年化) |\n|---|---|")
        for y, rd in yrows:
            report.append(f"| {y} | {rd:.2%} |")

    if fails:
        report.insert(1, f"\n**REJECTED：闸门 {fails} 未过，产物不得作为入账口径**\n")
        open(OUT_MD, "w", encoding="utf-8").write("\n".join(report))
        print("\n".join(report)); return 3

    tr.to_csv(OUT_CSV, index=False)
    report.insert(1, "\n**PASS：产物 cache/bench_tr_h00300.csv 为唯一入账 TR 口径**\n")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(report))
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
