#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #19（M1.4）事后回补：为 output/v5 本轮第二轮复跑产物批量写 manifest。

说明：产物在其生成脚本中暂未逐个内嵌 write_manifest 钩子；本脚本对既有产物目录
按"一次复跑批次 = 一条 manifest 记录"回补，ts 为回补时刻（如实注明 backfill），
输入/输出哈希按当前文件内容计算——满足"可凭 manifest 校验产物字节级未变"的用途。
"""
from __future__ import annotations

import glob
import json
import os

from v5_manifest import _sha16, current_commit

OUT = "output/v5"

BATCHES = {
    "d02_navadj_rebuild": {
        "inputs": sorted(glob.glob("cache/nav_*.csv")),
        "outputs": ["output/v5/d02_run.log", "output/v5/d02_summary.md",
                    "output/v5/d02_audit_events.csv", "output/v5/d02_fund_summary.csv"],
        "note": "build_navadj.py 全量重建 4987 只；事件账与复跑指纹见 d02_run.log",
    },
    "d05b_pit_universe_inferred": {
        "inputs": sorted(glob.glob("cache/nav_*.csv")),
        "outputs": sorted(glob.glob("data/pit_universe/*.csv")) + ["output/v5/d05_universe_audit.md"],
        "note": "243 月末推定型 PiT 快照（推定型标签见 D0.5b 预登记豁免段）",
    },
    "p1_panel_rebuild_159m": {
        "inputs": sorted(glob.glob("cache/navadj_*.csv")) + ["cache/fund_pool.json"],
        "outputs": sorted(glob.glob("output/p1_panel/*.csv")) + ["output/v5/p1_panel_build.log"],
        "note": "2013-01→2026-03 共 159 月，幸存池口径 SURV-ADJ",
    },
    "factor_study_variants": {
        "inputs": sorted(glob.glob("cache/navadj_*.csv")),
        "outputs": ["output/factor_variant_rows.csv", "output/factor_variant_summary.csv",
                    "output/v5/factor_study_run.log"],
        "note": "R3.4 输入面板；V0 复建 vs 引擎相关 0.9765",
    },
    "r31_hac": {"inputs": sorted(glob.glob("output/p1_panel/*.csv")),
                "outputs": ["output/v5/r31_factor_ic_hac.csv", "output/v5/r31_subgroup_ic_hac.csv",
                            "output/v5/r31_summary.md"], "note": "r31_hac_redo.py"},
    "r32_hac": {"inputs": sorted(glob.glob("output/p1_panel/*.csv")),
                "outputs": [p for p in glob.glob("output/v5/r32_*.csv")] + ["output/v5/r32_summary.md"],
                "note": "r32_hac_redo.py"},
    "r33_hac": {"inputs": sorted(glob.glob("output/p1_panel/*.csv")) + ["output/p2/vol_panel.csv"],
                "outputs": [p for p in glob.glob("output/v5/r33_*.csv")] + ["output/v5/r33_summary.md"],
                "note": "r33_hac_redo.py"},
    "r34_hac": {"inputs": ["output/factor_variant_rows.csv"],
                "outputs": [p for p in glob.glob("output/v5/r34_*.csv")] + ["output/v5/r34_summary.md"],
                "note": "r34_hac_redo.py"},
    "b21_baseline": {"inputs": sorted(glob.glob("output/p1_panel/*.csv")),
                     "outputs": [p for p in glob.glob("output/v5/b21_*.csv")] + ["output/v5/b21_baseline_summary.md"],
                     "note": "b21_baseline.py；sim_core 与 legacy 位级对拍通过（max|Δ|=0）"},
    "b22_b23": {"inputs": sorted(glob.glob("output/p1_panel/*.csv")),
                "outputs": ["output/v5/b22_cost_grid.csv", "output/v5/b23_delay_diff.csv",
                            "output/v5/b23_slippage.csv", "output/v5/b22_b23_summary.md"],
                "note": "b22_b23.py 27 格 + 3 档时滞"},
    "r27_postmortem": {"inputs": [], "outputs": ["output/v5/r27_postmortem.md"],
                       "note": "第二轮复跑事实记录与第一轮 B2.x delay 语义错误作废声明"},
}


def main():
    commit = current_commit()
    recs = []
    for name, spec in BATCHES.items():
        recs.append({
            "batch": name,
            "commit": commit,
            "backfill_ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "inputs": {os.path.basename(p): _sha16(p) for p in spec["inputs"] if os.path.exists(p)},
            "outputs": {os.path.basename(p): _sha16(p) for p in spec["outputs"] if os.path.exists(p)},
            "note": spec["note"],
        })
    path = os.path.join(OUT, "manifest.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[M1.4 backfill] {len(recs)} 批 → {path}")
    for r in recs:
        print(f"  {r['batch']}: {len(r['inputs'])} inputs, {len(r['outputs'])} outputs")


if __name__ == "__main__":
    main()
