# -*- coding: utf-8 -*-
"""
全局一致性验证测试脚本
验证华尔街顶级量化模型重构后的全域打分一致性:
- 1) 单基透视 (_final_single / API fund/<code>)
- 2) 批量评分 (watchlist / API watchlist)
- 3) 持仓诊断 (rebalance / API rebalance)
确保无论批量传入几只或哪几只，同一时点的 S_total, F_momentum, rating 100% 一致。
"""
import webapp
import pandas as pd

def test_global_consistency():
    codes = ['000001', '000006', '000008', '000011', '000017']
    print("==========================================================================")
    print(" 华尔街量化选基引擎 V3.8 —— 全局打分一致性验证 (Global Reference Scoring)")
    print("==========================================================================")
    
    print("\n【1. 正在计算 单基透视 (_final_single) ...】")
    single_results = {}
    for c in codes:
        r_raw = webapp.score_fund(c)
        r_fin = webapp._final_single(r_raw)
        single_results[c] = {
            "name": r_fin.get("name"),
            "F_momentum": r_fin.get("F_momentum"),
            "S_total": r_fin.get("S_total"),
            "rating": r_fin.get("rating")
        }
        print(f"  单基 => [{c}] {r_fin.get('name')}: F_momentum={r_fin.get('F_momentum')}, S_total={r_fin.get('S_total')}, Rating={r_fin.get('rating')}")

    print("\n【2. 正在计算 批量自选评分 (/api/watchlist: finalize with use_global_ref=True) ...】")
    rows_batch = [webapp.score_fund(c) for c in codes]
    df_batch = webapp.finalize(rows_batch, use_global_ref=True)
    for _, row in df_batch.iterrows():
        c = str(row["code"]).zfill(6)
        s_res = single_results[c]
        print(f"  批量 => [{c}] {row['name']}: F_momentum={row['F_momentum']}, S_total={row['S_total']}, Rating={row['rating']}")
        assert row["F_momentum"] == s_res["F_momentum"], f"Mismatch F_momentum for {c}: {row['F_momentum']} vs {s_res['F_momentum']}"
        assert row["S_total"] == s_res["S_total"], f"Mismatch S_total for {c}: {row['S_total']} vs {s_res['S_total']}"
        assert row["rating"] == s_res["rating"], f"Mismatch rating for {c}: {row['rating']} vs {s_res['rating']}"

    print("\n【3. 正在计算 持仓诊断 (/api/rebalance: finalize with use_global_ref=True) ...】")
    # 模拟任意次序与子集
    subset_codes = ['000011', '000001', '000008']
    rows_reb = [webapp.score_fund(c) for c in subset_codes]
    df_reb = webapp.finalize(rows_reb, use_global_ref=True)
    for _, row in df_reb.iterrows():
        c = str(row["code"]).zfill(6)
        s_res = single_results[c]
        print(f"  诊断 => [{c}] {row['name']}: F_momentum={row['F_momentum']}, S_total={row['S_total']}, Rating={row['rating']}")
        assert row["F_momentum"] == s_res["F_momentum"], f"Mismatch F_momentum for {c}: {row['F_momentum']} vs {s_res['F_momentum']}"
        assert row["S_total"] == s_res["S_total"], f"Mismatch S_total for {c}: {row['S_total']} vs {s_res['S_total']}"
        assert row["rating"] == s_res["rating"], f"Mismatch rating for {c}: {row['rating']} vs {s_res['rating']}"

    print("\n==========================================================================")
    print(" ✅ 验证完美通过！单基透视、批量评分、持仓诊断三者各项指标 100% 绝对一致！")
    print("==========================================================================")

if __name__ == "__main__":
    test_global_consistency()
