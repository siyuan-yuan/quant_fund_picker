# -*- coding: utf-8 -*-
"""
跑批演示: 12只不同风格的代表性基金 (主动/被动/行业/宽基 混合)
用法: python run_demo.py [代码...]   不带参数则用内置演示池
"""
import sys, os, json, datetime as dt
import pandas as pd

from engine import score_fund, finalize
from config import OUTPUT_DIR

DEMO_POOL = [
    "110011",  # 易方达优质精选(消费/QDII) 张坤
    "005827",  # 易方达蓝筹精选 张坤
    "161725",  # 招商中证白酒指数A (被动-白酒)
    "000083",  # 汇添富消费行业混合
    "260108",  # 景顺长城新兴成长 刘彦春
    "163406",  # 兴全合润 谢治宇 (均衡)
    "000961",  # 天弘沪深300指数 (被动-宽基)
    "004744",  # 易方达创业板ETF联接 (被动-成长)
    "000478",  # 建信中证500指数增强 (中盘)
    "320007",  # 诺安成长混合 (半导体集中)
    "002190",  # 农银新能源主题 (新能源)
    "519736",  # 交银新成长 (均衡成长)
    "003096",  # 中欧医疗健康A 葛兰 (医药-左侧候选)
]


def main(codes):
    rows = []
    for i, c in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] scoring {c} ...", flush=True)
        try:
            rows.append(score_fund(c))
        except Exception as e:
            print(f"    !! {c} failed: {e}")
            rows.append({"code": c, "name": c, "error": str(e)})
    df = finalize(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = dt.date.today().strftime("%Y%m%d")
    out_csv = f"{OUTPUT_DIR}/scores_{stamp}.csv"
    keep = ["code", "name", "ftype", "S_total", "rating",
            "F_value", "val_pct", "trend_ok", "bonus",
            "F_alpha", "ir_winrate", "down_capture",
            "F_momentum", "rank4", "rank7",
            "scale", "tenure_days", "is_passive", "penalty_str", "last_date"]
    df[[k for k in keep if k in df]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_json(f"{OUTPUT_DIR}/scores_{stamp}.json", orient="records",
               force_ascii=False, indent=1, default_handler=str)
    print("\n========== 评分榜单 ==========")
    pd.set_option("display.width", 220)
    print(df[["code", "name", "S_total", "rating", "F_value", "F_alpha",
              "F_momentum", "val_pct", "trend_ok", "ir_winrate",
              "penalty_str"]].to_string(index=False))
    print(f"\n[saved] {out_csv}")
    return df


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.isdigit()]
    main(args if args else DEMO_POOL)
