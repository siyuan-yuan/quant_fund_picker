#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6.1 批跑器（#33 子集，预登记口径如实披露）

本轮落地维度：种子 ×5 {1..5} × maxn {500,1000} + 全取 ×1 —— 共 11 组面板；
参照系维度暂用批内 rank（与 canonical 基准同通道）：S6.1 的"全市场 ECDF vs 批内 rank"
子维度依赖 **逐月全宇宙动量参照快照**（engine.get_global_ref_universe 的 as_of 目前取最新快照，
快照本身需以 U4.1 宇宙底座重造——列入依赖清单，不再宣称消解）。此披露替代任何省略。

- canonical 基准 = output/p1_panel（种子原式 YYYYMMDD×1、maxn=500、批内 rank）已有产物；
- 每组输出 output/s61_panel/seed{s}_maxn{m}/<date>.csv（目录隔离，与基准/彼此零干扰）；
- 完成后由 s61_summarize.py 对 {IC 均值, HAC t} 组间分布出报告（仅报告不平判，#33 自封）。

工程注记（2026-09-01，仅可移植性改动，不改任何实验口径/网格）：
  1. 解释器由硬编码沙箱路径改为 sys.executable（本机 venv 任意位置可跑）；
  2. 新增 --parallel N：组级并行（11 组之间目录隔离、零耦合，可安全并发）；
     单面板进程实测 ~257MB RSS，N 建议 ≤ 本机核数/2。
断点续跑：p1_panel_build 按月跳过已存在文件；本脚本对已完成 159 月的组自动 skip。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CONFIGS = [(s, m) for s in (1, 2, 3, 4, 5) for m in (500, 1000)] + [(0, 0)]  # maxn=0 ⇒ 全取（seed 维度塌缩）


def tag_of(s, m):
    return f"seed{s}_maxn{m if m else 'full'}"


def cmd_of(s, m, workers):
    outdir = os.path.join("output", "s61_panel", tag_of(s, m))
    # 全取(maxn=0): 采样永不触发 ⇒ 种子维度塌缩, 只跑 1 组；传 maxn=10**7 走原式不采样分支
    cmd = [PY, "p1_panel_build.py", "--workers", str(workers), "--panel-dir", outdir,
           "--seed-salt", str(s)] + (["--maxn", str(m)] if m else ["--maxn", "10000000"])
    return tag_of(s, m), outdir, cmd


def done_months(outdir):
    if not os.path.isdir(outdir):
        return 0
    return len([f for f in os.listdir(outdir) if f[:4].isdigit()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1, help="同时构建的面板组数（组间目录隔离，默认 1=串行）")
    ap.add_argument("--workers", type=int, default=4, help="单组面板内部线程数（默认 4，与历史一致）")
    ap.add_argument("--only", default="", help="只跑包含该子串的 tag（逗号分隔，便于分箱）")
    args = ap.parse_args()

    jobs = []
    for s, m in CONFIGS:
        tag, outdir, cmd = cmd_of(s, m, args.workers)
        if args.only and not any(o.strip() in tag for o in args.only.split(",")):
            continue
        jobs.append((tag, outdir, cmd))

    todo = []
    for tag, outdir, cmd in jobs:
        done = done_months(outdir)
        if done >= 159:
            print(f"[skip] {tag} 已完成 {done} 月", flush=True)
            continue
        todo.append((tag, outdir, cmd, done))
    print(f"[S6.1] 待构建 {len(todo)}/{len(jobs)} 组（parallel={args.parallel}, workers={args.workers}）", flush=True)
    if not todo:
        print("[S6.1] 全部 11 组面板已齐 → 跑 s61_summarize.py", flush=True)
        return

    fails = []

    def run(job):
        tag, outdir, cmd, done = job
        print(f"[S6.1] build {tag} → {outdir} (已完成 {done} 月)", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        return tag, r.returncode

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            tag, rc = fu.result()
            if rc != 0:
                print(f"[S6.1 FAIL] {tag} exit={rc}（断点续跑安全：重跑本脚本即可）", flush=True)
                fails.append(tag)
            else:
                print(f"[S6.1 OK {i}/{len(todo)}] {tag}", flush=True)

    if fails:
        print(f"[S6.1] 失败组: {fails}", flush=True)
        raise SystemExit(1)
    print("[S6.1] 全部 11 组面板构建完成 → 跑 s61_summarize.py", flush=True)


if __name__ == "__main__":
    main()
