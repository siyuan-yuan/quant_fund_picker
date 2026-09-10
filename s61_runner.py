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
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 【2026-09-02 跨平台修复 M16】原为硬编码 "/home/user/.venv_review/bin/python"（沙箱专属路径），
# 在 Windows / macOS / 其他 Linux 机器上必然 FileNotFoundError。
# 改用 sys.executable = "当前正在运行本脚本的解释器"，天然跨平台且保证子进程与父进程同环境
# （即：用哪个 venv 启动 s61_runner.py，子进程 p1_panel_build.py 就用哪个）。
# 纯执行环境修复，不改变任何计算/参数/面板内容。
PY = sys.executable

CONFIGS = [(s, m) for s in (1, 2, 3, 4, 5) for m in (500, 1000)] + [(0, 0)]  # maxn=0 ⇒ 全取（seed 维度塌缩）


def main():
    for i, (s, m) in enumerate(CONFIGS, 1):
        tag = f"seed{s}_maxn{m if m else 'full'}"
        outdir = os.path.join("output", "s61_panel", tag)
        done = len([f for f in os.listdir(outdir) if f[:4].isdigit()]) if os.path.isdir(outdir) else 0
        if done >= 159:
            print(f"[skip] {tag} 已完成 {done} 月", flush=True)
            continue
        # 全取(maxn=0): 采样永不触发 ⇒ 种子维度塌缩, 只跑 1 组；传 maxn=10**7 走原式不采样分支
        cmd = [PY, "p1_panel_build.py", "--workers", "4", "--panel-dir", outdir,
               "--seed-salt", str(s)] + (["--maxn", str(m)] if m else ["--maxn", "10000000"])
        print(f"[S6.1 {i}/{len(CONFIGS)}] build {tag} → {outdir} (已完成 {done} 月)", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"[S6.1 FAIL] {tag} exit={r.returncode}（断点续跑安全：重跑本脚本即可）", flush=True)
            raise SystemExit(r.returncode)
    print("[S6.1] 全部 11 组面板构建完成 → 跑 s61_summarize.py", flush=True)


if __name__ == "__main__":
    main()
