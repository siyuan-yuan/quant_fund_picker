#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成历史上所有进过前100名的基金池（历史回溯版）

数据源自动探测(按优先级):
  1) 命令行第2个参数指定的目录
  2) output/factor_rows/        ← factor_study.py 的季度因子行(含 S_eng)
  3) output/bt_scores_cache/    ← backtest_local.py 的月度评分缓存(含 S_engine)

排名分口径: 优先用引擎分 S_engine / S_eng(当前模型, 与回测一致);
            仅当两者都缺失时才回退内置简化公式(旧口径)。

用法:
  python top100_history_pool.py            # 自动探测, Top100
  python top100_history_pool.py 50         # 自定义 TopN
  python top100_history_pool.py 100 /path/to/src

注意:
  top100_history_pool.txt 只是兼容旧流程/缓存预热/稳健性测试，不是严格 PiT 回测池。
  严格回测请使用 backtest_local.py --pool-mode pit-top。
  若数据源宇宙本身就是旧池(如默认 217 只的缓存), 重新生成不会发现池外新基金;
  需要市场级发现时, 请用更大的 --pit-universe 先生成评分缓存, 再重建本池。
"""
import os
import re
import sys

import pandas as pd

FACTOR_ROWS_DIR = "output/factor_rows"
SCORE_CACHE_DIR = "output/bt_scores_cache"
OUTPUT_FILE = "top100_history_pool.txt"
OUTPUT_CSV = "top100_history_pool_by_date.csv"
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(.*)\.csv$")


def simplified_score(g):
    """旧版简化打分(引擎分缺失时的兜底, 与历史口径一致, 仅此情况使用)"""
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
        s = (0.4 * fv + 0.35 * fa + 0.25 * fm * 100) * 1.0
        out.append(min(max(s, 0), 100))
    g["S"] = out
    return g


def _count_rows(path):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return sum(1 for _ in fp) - 1
    except Exception:
        return 0


def pick_best_suffix(src):
    """bt_scores_cache 可能并存多套后缀(不同打分宇宙), 只取覆盖行数最多的那套"""
    rows_by_suf = {}
    for f in os.listdir(src):
        m = _DATE_RE.match(f)
        if not m:
            continue
        suf = m.group(2)
        rows_by_suf[suf] = rows_by_suf.get(suf, 0) + max(_count_rows(os.path.join(src, f)), 0)
    if not rows_by_suf:
        return ""
    return max(rows_by_suf.items(), key=lambda x: x[1])[0]


def main(top_n=100, src=None):
    if src is None:
        for cand in (FACTOR_ROWS_DIR, SCORE_CACHE_DIR):
            if os.path.isdir(cand) and any(f.endswith(".csv") for f in os.listdir(cand)):
                src = cand
                break

    if not src or not os.path.isdir(src):
        print("❌ 未找到评分数据源: output/factor_rows 与 output/bt_scores_cache 均不存在")
        print("   请先运行一次:  python backtest_local.py --pool-mode pit-top --score-suffix auto")
        print("   (首次运行会自动逐季现打分, 生成 output/bt_scores_cache/ 月度评分缓存, 之后秒级)")
        print("   或运行:        python factor_study.py")
        print("   (生成 output/factor_rows/ 季度因子行)")
        return

    files = sorted(f for f in os.listdir(src) if f.endswith(".csv"))
    if not files:
        print(f"❌ {src} 目录下没有 csv 文件")
        return

    suf = pick_best_suffix(src)
    files = [f for f in files if f.endswith(suf + ".csv")]
    is_cache = os.path.abspath(src) == os.path.abspath(SCORE_CACHE_DIR)
    print(f"[源] {src}  (后缀 '{suf or '(无后缀)'}', {len(files)} 个评分文件)")

    all_codes = set()
    audit_rows = []
    score_src = None

    for f in files:
        try:
            g = pd.read_csv(os.path.join(src, f), dtype={"code": str})
            if g.empty or "code" not in g.columns:
                continue
            g = g.dropna(subset=["code"])
            g["code"] = g["code"].astype(str).str.zfill(6)
            date_str = _DATE_RE.match(f).group(1)
            # 排名分口径: 引擎分优先(与回测一致), 否则旧简化公式
            if "S_engine" in g.columns and g["S_engine"].notna().any():
                g["S"] = pd.to_numeric(g["S_engine"], errors="coerce")
                score_src = "S_engine(引擎分)"
            elif "S_eng" in g.columns and g["S_eng"].notna().any():
                g["S"] = pd.to_numeric(g["S_eng"], errors="coerce")
                score_src = "S_eng(引擎分)"
            else:
                g = simplified_score(g)
                score_src = "简化公式(旧口径)"
            g = g.dropna(subset=["S"])
            if g.empty:
                continue
            top = g.nlargest(top_n, "S")
            for rank, (_, r) in enumerate(top.iterrows(), 1):
                all_codes.add(r["code"])
                audit_rows.append({"date": date_str, "code": r["code"],
                                   "rank": rank, "S": round(float(r["S"]), 1)})
        except Exception as e:
            print(f"  [warn] {f} 解析失败: {e}")
            continue

    if not audit_rows:
        print("❌ 所有评分文件均无有效打分行, 未生成任何基金")
        return

    pool = sorted(all_codes)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        fp.write("\n".join(pool))

    audit_df = pd.DataFrame(audit_rows, columns=["date", "code", "rank", "S"])
    audit_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ 已生成历史上进过前{top_n}名的基金池: {OUTPUT_FILE}")
    print(f"   共收集到 {len(pool)} 只基金 | 排名口径: {score_src}")
    print(f"✅ 已生成按日的前{top_n}名审计文件: {OUTPUT_CSV}")
    print(f"   共包含 {len(audit_df)} 条记录")
    if is_cache:
        print("\n" + "=" * 60)
        print("注意: 数据源为 bt_scores_cache, 其宇宙来自回测打分池;")
        print("若缓存宇宙本身就是旧池(默认 217 只), 重新生成不会发现池外新基金。")
        print("=" * 60)
    print("\n" + "=" * 60)
    print("top100_history_pool.txt 只是兼容旧流程/缓存预热/稳健性测试，不是严格 PiT 回测池。")
    print("严格回测请使用 backtest_local.py --pool-mode pit-top。")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    src = sys.argv[2] if len(sys.argv) > 2 else None
    main(n, src)
