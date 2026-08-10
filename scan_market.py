# -*- coding: utf-8 -*-
"""
Step 2A — 全市场扫描器
三层漏斗 + 双通道选股池 + 6线程并行深算
用法:
  python scan_market.py [右侧池大小=400] [左侧池大小=150]
  python scan_market.py --all-main        # 扫描全部主池(近3年完整+去C/E后的全量)
  python scan_market.py --all-target      # 扫描全部目标类型(类型过滤+去C/E后的全量)
"""
import argparse
import time, datetime as dt, os, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import akshare as ak

import provider
from engine import score_fund, finalize
from config import (OUTPUT_DIR, CACHE_DIR, YOUNG_TOP_N, SCAN_INCLUDE_OVERSEAS_INDEX,
                    OVERSEAS_FUND_TYPES)

# 扫描目标类型白名单（A股四类 + 可选海外指数QDII通道）
# 016452 南方纳斯达克100(QDII)A 等"指数型-海外股票"此前因此白名单被排除在全部扫描模式外，
# 但引擎本身支持境外评分(panel_mode=overseas) → 由 SCAN_INCLUDE_OVERSEAS_INDEX 控制并入。
TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活"}
if SCAN_INCLUDE_OVERSEAS_INDEX:
    TARGET_TYPES = TARGET_TYPES | {"指数型-海外股票"}
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _region(ftypes) -> pd.Series:
    """基金类型 → 市场标签（A股/海外），供榜单分市场视图。"""
    return np.where(ftypes.isin(OVERSEAS_FUND_TYPES), "海外", "A股")


def _load_rank_table() -> pd.DataFrame:
    rank_path = f"{CACHE_DIR}/rank_all.csv"
    if provider._fresh(rank_path):
        rank = pd.read_csv(rank_path, dtype={"基金代码": str})
    else:
        rank = ak.fund_open_fund_rank_em(symbol="全部")
        rank["基金代码"] = rank["基金代码"].astype(str).str.zfill(6)
        rank.to_csv(rank_path, index=False)
    return rank


def _load_target_funds() -> pd.DataFrame:
    """全市场排行 ∩ 目标基金类型，并剔除 C/E 重复份额。"""
    rank = _load_rank_table()
    meta = provider.get_fund_meta()
    df = rank.join(meta[["基金类型"]], on="基金代码")
    df = df[df["基金类型"].isin(TARGET_TYPES)].copy()
    # 剔除C/E类份额(同策略重复), 保留A或基础份额
    df = df[~df["基金简称"].str.strip().str.endswith(("C", "E"))].copy()
    if SCAN_INCLUDE_OVERSEAS_INDEX:
        n_ovs = int((df["基金类型"] == "指数型-海外股票").sum())
        print(f"[漏斗] 海外指数QDII通道已开启: 纳入 {n_ovs} 只指数型-海外股票"
              f"（含 016452 南方纳斯达克100(QDII)A 等，config.SCAN_INCLUDE_OVERSEAS_INDEX）")
    for c in ["近3年", "近1年", "近6月"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_universe(right_n=400, left_n=150, mode="default") -> pd.DataFrame:
    """构建扫描池。

    mode="default": 原三层漏斗 —— 右侧池 + 左侧池 + 新星Top100
    mode="all_main": 扫描全部主池(近3年完整 + 去C/E，当前约4530只)
    mode="all_target": 扫描全部目标类型(类型过滤 + 去C/E 后的全量, 当前约6538只)
    """
    df = _load_target_funds()
    main_df = df.dropna(subset=["近3年"]).copy()     # 主池要求≥3年完整周期数据
    main_df = main_df[main_df["近3年"] != 0]

    if mode == "all_main":
        pool = main_df.copy()
        pool["channel"] = "全量-主池"
        pool["region"] = _region(pool["基金类型"])
        pool = pool.drop_duplicates("基金代码").reset_index(drop=True)
        print(f"[漏斗] 全主池模式 | 近3年完整主池 {len(pool)}")
        return pool

    if mode == "all_target":
        main_mask = df["近3年"].notna() & (df["近3年"] != 0)
        young_mask = df["近3年"].isna() & df["近1年"].notna()
        pool = df.copy()
        pool["channel"] = np.where(main_mask, "全量-主池",
                             np.where(young_mask, "新星", "全量-次新/缺历史"))
        pool["region"] = _region(pool["基金类型"])
        pool = pool.drop_duplicates("基金代码").reset_index(drop=True)
        print(f"[漏斗] 全目标类型模式 | 类型过滤+去C/E {len(pool)} | "
              f"主池 {int(main_mask.sum())} | 新星 {int(young_mask.sum())} | "
              f"其余 {int(len(pool) - main_mask.sum() - young_mask.sum())}")
        return pool

    # V3.5 条款4: 🌱新星观察池 —— 近3年为空(成立<3年)但近1年非空(≥12个月), 取近1年TopN
    young_src = df[df["近3年"].isna() & df["近1年"].notna()]
    ypool = young_src.nlargest(YOUNG_TOP_N, "近1年").assign(channel="新星")

    right = main_df.nlargest(right_n, "近3年").assign(channel="右侧")
    left_pool = main_df[(main_df["近1年"] < 0) & (~main_df.index.isin(right.index))]
    left = left_pool.nsmallest(left_n, "近1年").assign(channel="左侧")
    pool = pd.concat([right, left, ypool]).drop_duplicates("基金代码")
    pool["region"] = _region(pool["基金类型"])
    print(f"[漏斗] 类型过滤后 {len(main_df)} | 右侧池 {len(right)} | 左侧池 {len(left)} | "
          f"🌱新星池 {len(ypool)} | 合计 {len(pool)}")
    return pool


def main(right_n=400, left_n=150, workers=6, mode="default"):
    t0 = time.time()
    pool = build_universe(right_n, left_n, mode=mode)
    chan = dict(zip(pool["基金代码"], pool["channel"]))
    reg = dict(zip(pool["基金代码"], pool["region"]))
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
            r["region"] = reg.get(c, "A股")
            rows.append(r)
            if i % 50 == 0:
                print(f"    ... deep-scored {i}/{len(codes)} ({time.time()-t0:.0f}s)", flush=True)

    df = finalize(rows)
    stamp = dt.date.today().strftime("%Y%m%d")
    out_csv = f"{OUTPUT_DIR}/scan_{stamp}.csv"
    keep = ["code", "name", "ftype", "region", "channel", "S_total", "rating",
            "F_value", "val_pct", "trend_ok", "trend_ma20", "bonus", "F_alpha",
            "ir_winrate", "down_capture", "F_momentum", "mom_4m1m", "mom_7m1m",
            "rank4", "rank7", "scale", "tenure_days", "is_passive", "penalty_str",
            "water", "weights_mode", "last_date", "rbsa", "error"]
    out = df[[k for k in keep if k in df]].copy()
    # rbsa 为 dict → JSON 串落盘（买入候选重复度过滤用；读侧 json.loads 解析）
    if "rbsa" in out:
        out["rbsa"] = out["rbsa"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else "")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ok = df[df["error"].isna()]
    print(f"\n[完成] 深算 {len(df)} 只 | 成功 {len(ok)} 只 | 用时 {time.time()-t0:.0f}s")
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
    ap = argparse.ArgumentParser(description="全市场扫描器")
    ap.add_argument("right", nargs="?", type=int, default=400, help="右侧池大小")
    ap.add_argument("left", nargs="?", type=int, default=150, help="左侧池大小")
    ap.add_argument("--all-main", action="store_true",
                    help="扫描全部主池(近3年完整+去C/E后的全量，约4530只，忽略右侧/左侧)")
    ap.add_argument("--all-target", action="store_true",
                    help="扫描全部目标类型(类型过滤+去C/E后的全量，约6538只，忽略右侧/左侧)")
    ap.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = ap.parse_args()
    mode = "all_target" if args.all_target else ("all_main" if args.all_main else "default")
    main(args.right, args.left, workers=args.workers, mode=mode)