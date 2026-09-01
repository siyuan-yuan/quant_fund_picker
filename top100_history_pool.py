#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成历史上所有进过前100名的基金池（真正历史回溯版）
自动从 factor_rows 重建 S 分数

注意：
top100_history_pool.txt 只是兼容旧流程/缓存预热/稳健性测试，不是严格 PiT 回测池。
严格回测请使用 backtest_local.py --pool-mode pit-top。
"""
import os
import sys
import pandas as pd
from glob import glob

FACTOR_ROWS_DIR = "output/factor_rows"
OUTPUT_FILE = "top100_history_pool.txt"
OUTPUT_CSV = "top100_history_pool_by_date.csv"


def score_from_raw(g):
    """和 backtest_local.py 里一样的打分逻辑"""
    g = g.copy()
    g["rank4"] = g["r4"].rank(pct=True)
    g["rank7"] = g["r7"].rank(pct=True)
    out = []
    for _, r in g.iterrows():
        fv = (0 if (r.val_cov < 0.5 or pd.isna(r.val_pct))
              else min(max((r.val_pct - 0) * 100, 0), 100))   # 简化估值分
        fa = float(pd.Series([r.wr if pd.notna(r.wr) else 50,
                              r.dc if pd.notna(r.dc) else 50]).mean())
        fm = 0.6 * r.rank4 + 0.4 * r.rank7
        pen = 1.0
        s = ((0.4 * fv + 0.35 * fa + 0.25 * fm * 100) * pen)
        out.append(min(max(s, 0), 100))
    g["S"] = out
    return g


def main(top_n=100):
    if not os.path.exists(FACTOR_ROWS_DIR):
        print(f"❌ 未找到 {FACTOR_ROWS_DIR} 目录")
        return

    files = sorted(glob(f"{FACTOR_ROWS_DIR}/*.csv"))
    if not files:
        print("❌ factor_rows 目录下没有文件")
        return

    print(f"正在扫描 {len(files)} 个历史打分文件并重建分数...")

    all_codes = set()
    audit_rows = []

    for f in files:
        try:
            df = pd.read_csv(f, dtype={"code": str})
            if df.empty or "r4" not in df.columns:
                continue
            df = score_from_raw(df)
            df_sorted = df.sort_values("S", ascending=False).reset_index(drop=True)
            top_df = df_sorted.head(top_n)
            top = top_df["code"].tolist()
            all_codes.update(top)

            date_str = os.path.splitext(os.path.basename(f))[0]
            for idx, r in top_df.iterrows():
                audit_rows.append({
                    "date": date_str,
                    "code": str(r["code"]).zfill(6),
                    "rank": idx + 1,
                    "S": r["S"]
                })
        except Exception as e:
            continue

    pool = sorted(all_codes)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        fp.write("\n".join(pool))

    audit_df = pd.DataFrame(audit_rows, columns=["date", "code", "rank", "S"])
    audit_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已生成历史上进过前{top_n}名的基金池：{OUTPUT_FILE}")
    print(f"   共收集到 {len(pool)} 只基金")
    print(f"✅ 已生成按日的前{top_n}名审计文件：{OUTPUT_CSV}")
    print(f"   共包含 {len(audit_df)} 条记录")
    print("\n" + "=" * 60)
    print("top100_history_pool.txt 只是兼容旧流程/缓存预热/稳健性测试，不是严格 PiT 回测池。")
    print("严格回测请使用 backtest_local.py --pool-mode pit-top。")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)