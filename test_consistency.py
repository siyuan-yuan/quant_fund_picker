# -*- coding: utf-8 -*-
"""
全局一致性验证测试脚本（联网，需真实数据源）
验证华尔街顶级量化模型重构后的全域打分一致性:
- 1) 单基透视 (_final_single / API fund/<code>)
- 2) 批量评分 (watchlist / API watchlist)
- 3) 持仓诊断 (rebalance / API rebalance)
确保无论批量传入几只或哪几只，同一时点的 S_total, F_momentum, rating 100% 一致。

V4.1 一致性契约（两种运行态下本测试断言均须通过）:
  - V4 激活态(已装 sklearn + cache/v4_model.pkl): 三入口统一对全市场参照
    快照做 ECDF 映射(engine.get_global_ref_universe)，批次无关性严格成立；
    参照快照无 z 分布时三入口同步闸门降级 V3.7，绝不退回批内 rank。
  - V4 缺失态(无 sklearn/模型文件): 纯 V3.7 + global ref 动量尺，同样一致。
  若断言失败：检查数据源限流导致的档案降级(data_incomplete) —— 重试即可。
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
    print(f"  [环境] model_version={r_fin.get('model_version')} | ref_stamp={r_fin.get('ref_stamp')} | data_incomplete={r_fin.get('data_incomplete')}")

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
