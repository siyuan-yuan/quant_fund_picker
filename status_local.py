#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本机进度查看 + 跨机一致性闸门（D0.2 复权对账）

用法:
    python status_local.py status    # 查看 S6.1 / R3.5 / 规范面板 的完成度
    python status_local.py parity    # 跨机一致性闸门：读 output/v5/d02_summary.md 与权威值逐项对账

parity 说明（先跑 python build_navadj.py --workers 4 再执行本命令）:
    权威值固化于 docs/优化计划_V5_重规范_2026-09.md §1/D0.2 与
    docs/V5_第三次工作区重置与重跑必要性分级_2026-09-02.md §4。
    **四项全同才继续下游**；任一不同即停机排查（净值缓存差异 / pandas-numpy 版本差异）。
"""
from __future__ import annotations

import os
import re
import sys

# ---- 权威值（D0.2，M11 后冻结；不得在本脚本中修改，只允许在文档中按翻案规则改判）----
AUTH = dict(
    pool=4987, built=4987, events=21454, cover=3494,
    kind_div=12090, kind_err=9248, kind_split=116,
    drag_n=4090, drag_mean=0.6215, drag_p95=3.91, drag_max=78.78,
)


def _months(d):
    if not os.path.isdir(d):
        return None
    return len([f for f in os.listdir(d) if f[:4].isdigit() and f.endswith(".csv")])


def status():
    print("=== S6.1（每组需 159 月）===")
    base = os.path.join("output", "s61_panel")
    if os.path.isdir(base):
        for tag in sorted(os.listdir(base)):
            n = _months(os.path.join(base, tag))
            if n is None:
                continue
            print(f"  {tag:<24s} {n:>4d}/159 {'OK' if n >= 159 else ''}")
    else:
        print("  (未开始)")

    print("=== R3.5 动物园缓存（需 145 月，2014-04→2026-03）===")
    d = os.path.join("output", "bt_scores_cache")
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.endswith("_2e4ec0f5.csv")])
        print(f"  {n:>4d}/145 {'OK' if n >= 145 else ''}")
    else:
        print("  (未开始)")

    print("=== 规范面板 canonical（需 159 月，s61_summarize 的前置）===")
    n = _months(os.path.join("output", "p1_panel"))
    print(f"  {0 if n is None else n:>4d}/159 {'OK' if (n or 0) >= 159 else '(缺失则 s61_summarize 无法汇总)'}")

    print("=== 净值缓存 ===")
    c = os.path.join("cache")
    if os.path.isdir(c):
        print(f"  nav_*.csv: {len([f for f in os.listdir(c) if f.startswith('nav_') and f.endswith('.csv')])} 只（目标 15,444）")
    else:
        print("  (无 cache/ 目录，需先跑 t0_fetch.py / 拷贝缓存)")


def _num(s):
    try:
        return float(s.replace(",", "").replace("%", "").replace("+", ""))
    except Exception:
        return None


def parity():
    fp = os.path.join("output", "v5", "d02_summary.md")
    if not os.path.exists(fp):
        print(f"缺 {fp} —— 先跑: python build_navadj.py --workers 4")
        return 1
    txt = open(fp, encoding="utf-8").read()

    got = {}
    m = re.search(r"池：(\d+) 只；成功构建 adj：(\d+)", txt)
    if m:
        got["pool"], got["built"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"事件日总数：([\d,]+)（覆盖 (\d+) 只）", txt)
    if m:
        got["events"], got["cover"] = int(m.group(1).replace(",", "")), int(m.group(2))
    for key, pat in (("kind_div", r"'分红/折算\(官方收益回补\)': (\d+)"),
                     ("kind_err", r"'数据异常\(待审\)': (\d+)"),
                     ("kind_split", r"'份额折算\(上拆\)': (\d+)")):
        m = re.search(pat, txt)
        if m:
            got[key] = int(m.group(1))
    m = re.search(r"n=([\d,]+), ≥3y 序列）：均值 ([+\-0-9.]+)%/yr、中位 [+\-0-9.]+%、P95 ([+\-0-9.]+)%、最大 ([+\-0-9.]+)%", txt)
    if m:
        got["drag_n"] = int(m.group(1).replace(",", ""))
        got["drag_mean"] = float(m.group(2).replace("+", ""))
        got["drag_p95"] = float(m.group(3).replace("+", ""))
        got["drag_max"] = float(m.group(4).replace("+", ""))

    rows = []
    for k, want in AUTH.items():
        v = got.get(k)
        ok = (v is not None) and (abs(v - want) < 1e-6 if isinstance(want, float) else v == want)
        rows.append((k, want, "缺" if v is None else v, "PASS" if ok else "FAIL"))

    w = max(len(str(r[1])) for r in rows)
    print(f"{'项':<12s} {'权威值':>{w}s}  {'本机值':>{w}s}  判定")
    for k, want, v, ok in rows:
        print(f"{k:<12s} {str(want):>{w}s}  {str(v):>{w}s}  {ok}")
    n_fail = sum(1 for r in rows if r[3] != "PASS")
    print()
    if n_fail:
        print(f"❌ {n_fail} 项不符 —— 停机排查（净值缓存是否 15,444 只 / pandas-numpy 是否按 requirements-v5-lock.txt 钉版），不要继续下游")
        return 1
    print("✅ D0.2 跨机对账全同（池/事件/分类/年化差），可继续下游")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "parity":
        raise SystemExit(parity())
    status()
