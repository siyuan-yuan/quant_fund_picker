# -*- coding: utf-8 -*-
"""
Step 2A — 全市场扫描器
三层漏斗 + 双通道选股池 + 6线程并行深算
用法: python scan_market.py [右侧池大小=400] [左侧池大小=150]
"""
import sys, time, datetime as dt, os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import akshare as ak

import provider
from engine import score_fund, finalize
from config import OUTPUT_DIR, CACHE_DIR, YOUNG_TOP_N

TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_universe(right_n=400, left_n=150) -> pd.DataFrame:
    """漏斗: 名录 ∩ 全市场排行大表(单次调用) → 双通道池"""
    rank_path = f"{CACHE_DIR}/rank_all.csv"
    if provider._fresh(rank_path):
        rank = pd.read_csv(rank_path, dtype={"基金代码": str})
    else:
        rank = ak.fund_open_fund_rank_em(symbol="全部")
        rank["基金代码"] = rank["基金代码"].astype(str).str.zfill(6)
        rank.to_csv(rank_path, index=False)

    meta = provider.get_fund_meta()
    df = rank.join(meta[["基金类型"]], on="基金代码")
    df = df[df["基金类型"].isin(TARGET_TYPES)]
    # 剔除C/E类份额(同策略重复), 保留A或基础份额
    df = df[~df["基金简称"].str.strip().str.endswith(("C", "E"))]
    for c in ["近3年", "近1年", "近6月"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # V3.5 条款4: 🌱新星观察池 —— 近3年为空(成立<3年)但近1年非空(≥12个月), 取近1年TopN
    young_src = df[df["近3年"].isna() & df["近1年"].notna()]
    ypool = young_src.nlargest(YOUNG_TOP_N, "近1年").assign(channel="新星")

    df = df.dropna(subset=["近3年"])          # 主池要求≥3年完整周期数据
    df = df[df["近3年"] != 0]

    right = df.nlargest(right_n, "近3年").assign(channel="右侧")
    left_pool = df[(df["近1年"] < 0) & (~df.index.isin(right.index))]
    left = left_pool.nsmallest(left_n, "近1年").assign(channel="左侧")
    pool = pd.concat([right, left, ypool]).drop_duplicates("基金代码")
    print(f"[漏斗] 类型过滤后 {len(df)} | 右侧池 {len(right)} | 左侧池 {len(left)} | "
          f"🌱新星池 {len(ypool)} | 合计 {len(pool)}")
    return pool


def main(right_n=400, left_n=150, workers=6):
    t0 = time.time()
    pool = build_universe(right_n, left_n)
    chan = dict(zip(pool["基金代码"], pool["channel"]))
    rows = []
    codes = pool["基金代码"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_fund, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"code": c, "name": c, "error": str(e)[:100]}
                print(f"    !! {c}: {r['error']}")
            r["channel"] = chan.get(c)
            rows.append(r)
            if i % 50 == 0:
                print(f"    ... deep-scored {i}/{len(codes)} ({time.time()-t0:.0f}s)", flush=True)

    df = finalize(rows)
    stamp = dt.date.today().strftime("%Y%m%d")
    out_csv = f"{OUTPUT_DIR}/scan_{stamp}.csv"
    keep = ["code", "name", "ftype", "channel", "S_total", "rating",
            "F_value", "val_pct", "trend_ok", "trend_ma20", "bonus", "F_alpha",
            "ir_winrate", "down_capture", "F_momentum", "mom_4m1m", "mom_7m1m",
            "rank4", "rank7", "scale", "tenure_days", "is_passive", "penalty_str",
            "water", "weights_mode", "last_date", "error"]
    df[[k for k in keep if k in df]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    ok = df[df["error"].isna()]
    print(f"\n[完成] 深算 {len(ok)} 只 | 用时 {time.time()-t0:.0f}s")
    print(f"评分分布: ≥70={len(ok[ok.S_total>=70])}  50-70={len(ok[(ok.S_total>=50)&(ok.S_total<70)])}  "
          f"30-50={len(ok[(ok.S_total>=30)&(ok.S_total<50)])}  <30={len(ok[ok.S_total<30])}")
    print("\n========== 全市场 TOP 25 ==========")
    pd.set_option("display.width", 240)
    cols = ["code", "name", "channel", "S_total", "rating", "F_value",
            "F_alpha", "F_momentum", "val_pct", "ir_winrate", "penalty_str"]
    print(ok[cols].head(25).to_string(index=False))
    print("\n========== 左侧通道中的高分低估者 (F_value≥40) ==========")
    print(ok[ok.F_value >= 40][cols].head(20).to_string(index=False))
    print(f"\n[saved] {out_csv}")
    return df


if __name__ == "__main__":
    r, l = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (400, 150)
    main(r, l)
