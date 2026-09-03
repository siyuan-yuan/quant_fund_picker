#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6.1 跑完后的放行校验（M14 一致性 + 完整性 + 汇总自检）

用法（Windows）：
    .venv_review\\Scripts\\python.exe s61_verify.py

本脚本**只读不写**（除了打印），不改任何产物、不重算任何 IC。
它回答三个问题，全部通过才允许解读 s61_summary.md：

  A. 完整性 —— 11 组是否都跑齐了？（s61_summarize.py 会静默 [skip] 掉 <100 月的组，
     汇总表照样生成，容易拿到"缺组但看起来正常"的报告）
  B. M14 一致性 —— 各变体目录各自重建的 universe.csv 是否与 canonical 逐字节相同？
     不一致 ⇒ 各组输入宇宙不同 ⇒ 组间不可比 ⇒ S6.1 结果作废、立案。
  C. 汇总自检 —— s61_robustness.csv 里是否真的有 12 个 tag（canonical + 11 组）？
     canonical 是否落在变体区间内？（**仅陈述，不平判** —— #33 自封条款）

口径提醒：S6.1 = 预登记 #33，`仅报告不平判`。无论结果紧散，**禁止据此调参**。
全部为 SURV-ADJ 口径，不构成任何采纳/否决依据。
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

BASE_DIR = "output/p1_panel"
GRID_DIR = "output/s61_panel"
SUMMARY_CSV = "output/v5/s61_robustness.csv"

EXPECTED = [f"seed{s}_maxn{m}" for s in (1, 2, 3, 4, 5) for m in (500, 1000)] + ["seed0_maxnfull"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_months(d):
    return len([f for f in os.listdir(d) if f[:4].isdigit() and f.endswith(".csv")]) \
        if os.path.isdir(d) else 0


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fail = []
    warn = []

    print("=" * 72)
    print("S6.1 放行校验")
    print("=" * 72)

    # ---------- A. 完整性 ----------
    print("\n[A] 完整性检查（11 组 × 每组月数）")
    base_n = n_months(BASE_DIR)
    print(f"  canonical (output/p1_panel): {base_n} 月")
    if base_n != 145:
        warn.append(f"canonical 月数 {base_n} ≠ 文档固化值 145")

    if not os.path.isdir(GRID_DIR):
        fail.append(f"{GRID_DIR} 不存在 —— s61_runner.py 没产出任何东西")
        print(f"  ❌ {GRID_DIR} 不存在")
    else:
        found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(GRID_DIR, "*"))
                       if os.path.isdir(p))
        for tag in EXPECTED:
            d = os.path.join(GRID_DIR, tag)
            n = n_months(d)
            if n == 0:
                print(f"  ❌ {tag:20s} 缺失/空")
                fail.append(f"{tag} 缺失或为空")
            elif n < 100:
                print(f"  ❌ {tag:20s} {n:4d} 月  ← <100，会被汇总脚本静默跳过！")
                fail.append(f"{tag} 仅 {n} 月（<100，未进汇总表）")
            else:
                flag = "" if n == base_n else f"  ← 与 canonical({base_n}) 不同"
                print(f"  ✅ {tag:20s} {n:4d} 月{flag}")
                if n != base_n:
                    warn.append(f"{tag} 月数 {n} ≠ canonical {base_n}")
        extra = [t for t in found if t not in EXPECTED]
        if extra:
            warn.append(f"存在预期外目录: {extra}")

    # ---------- B. M14 一致性 ----------
    print("\n[B] M14 校验：各变体 universe.csv vs canonical")
    base_u = os.path.join(BASE_DIR, "universe.csv")
    if not os.path.exists(base_u):
        fail.append("canonical universe.csv 不存在，无法做 M14 校验")
        print("  ❌ canonical universe.csv 不存在")
    else:
        b = sha(base_u)
        print(f"  canonical  {b[:16]}  ({os.path.getsize(base_u):,} bytes)")
        checked = 0
        for tag in EXPECTED:
            u = os.path.join(GRID_DIR, tag, "universe.csv")
            if not os.path.exists(u):
                print(f"  ⚠️  {tag:20s} 无 universe.csv（该组可能未启动）")
                continue
            checked += 1
            h = sha(u)
            if h == b:
                print(f"  ✅ {tag:20s} {h[:16]}  一致")
            else:
                print(f"  ❌ {tag:20s} {h[:16]}  **不一致**")
                fail.append(f"M14: {tag} 的 universe.csv 与 canonical 不一致")
        if checked == 0:
            fail.append("没有任何变体 universe.csv 可校验")

    # ---------- C. 汇总自检 ----------
    print("\n[C] 汇总产物自检")
    if not os.path.exists(SUMMARY_CSV):
        fail.append(f"{SUMMARY_CSV} 不存在 —— s61_summarize.py 未成功产出")
        print(f"  ❌ {SUMMARY_CSV} 不存在")
    else:
        try:
            import pandas as pd
            out = pd.read_csv(SUMMARY_CSV)
            tags = sorted(out.tag.unique())
            print(f"  汇总表含 {len(tags)} 个 tag: {tags}")
            missing = [t for t in EXPECTED if t not in tags]
            if missing:
                print(f"  ❌ 汇总表缺少: {missing}")
                fail.append(f"汇总表缺少 {len(missing)} 组: {missing}")
            else:
                print("  ✅ 12 个 tag（canonical + 11 组）齐全")

            print("\n  组间分布（仅陈述，#33 自封：不平判、不得据此调参）")
            for h in ("fwd1", "fwd3", "fwd6"):
                sub = out[(out.horizon == h) & (out.tag != "CANONICAL")]["ic_mean"].dropna()
                can = out[(out.horizon == h) & (out.tag == "CANONICAL")]["ic_mean"]
                if len(sub) and len(can):
                    c = float(can.iloc[0])
                    inside = sub.min() <= c <= sub.max()
                    # 分位：canonical 在变体分布中的位置
                    pct = float((sub < c).mean()) * 100
                    print(f"    {h}: 变体 [{sub.min():+.4f}, {sub.max():+.4f}] "
                          f"std={sub.std():.4f} | canonical {c:+.4f} "
                          f"({'区间内' if inside else '**超出区间**'}, 分位 {pct:.0f}%)")
                    if not inside:
                        warn.append(f"{h}: canonical 落在变体区间之外（仅登记，非判定）")
        except Exception as e:
            fail.append(f"读取汇总表失败: {e}")
            print(f"  ❌ 读取失败: {e}")

    # ---------- 裁决 ----------
    print("\n" + "=" * 72)
    if fail:
        print("❌ 未通过放行校验 —— S6.1 结果**不得解读**，需先处置：")
        for x in fail:
            print(f"   · {x}")
        print("\n   处置指引：")
        print("   · 若为'缺组/月数不足' → 重跑 `s61_runner.py`（断点续跑安全），跑完再 `s61_summarize.py`")
        print("   · 若为'M14 不一致'   → **停跑立案**，S6.1 结果作废；把不一致的 tag 发我定位")
    else:
        print("✅ 放行校验通过：11 组齐全、M14 一致、汇总表完整。")
        print("   可以解读 output/v5/s61_summary.md。")
        print("   但请记住 #33 自封条款：**仅报告不平判，禁止据此调整任何生产参数**；")
        print("   且为 SURV-ADJ 口径，不构成采纳/否决依据。")
    if warn:
        print("\n⚠️ 提示（不阻断放行，但需登记）：")
        for x in warn:
            print(f"   · {x}")
    print("=" * 72)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())