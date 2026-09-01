#!/usr/bin/env bash
# V5 用户本机批跑入口（S6.1 / R3.5）—— 只跑"沙箱跑不动"的小时级任务
#
# 用法:
#   ./run_local_batch.sh s61 6        # S6.1：11 组面板，6 路并行
#   ./run_local_batch.sh r35 8        # R3.5：动物园缓存 8 路并行 + ML 面板 + zoo 重跑
#   ./run_local_batch.sh all 6        # 先 r35(缓存) 后 s61
#   ./run_local_batch.sh status       # 只看进度
#
# 说明（务必读）:
#  1. 两组任务均**按月/按组断点续跑**：中断后重跑同一条命令即可接着跑，不会重算已完成部分。
#  2. 并行度 N 建议 ≤ 本机物理核数/2；单面板进程实测 ~257MB RSS，内存一般不是瓶颈，瓶颈是 CPU。
#  3. 产物（output/、cache/）按 .gitignore 不入库；跑完请把**汇总小文件**交给仓库（见 docs/V5_用户本机执行手册_2026-09-01.md §4）：
#       output/v5/s61_*            （S6.1 汇总，仅报告不平判）
#       output/v5/r35_zoo_redo/*   （R3.5 修复版动物园，IC/配对差表）
#  4. 口径标签：本批产物一律 SURV-ADJ（幸存池 + 复权 + T+1 + 阶梯成本），
#     **禁止**在任何结论中当作 FULL-PIT 证据使用（D0.6 已冻结）。
set -euo pipefail

PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

LANE="${1:-status}"
N="${2:-4}"

status() {
  echo "--- S6.1 各组已完成月数（159 = 完成）---"
  if [ -d output/s61_panel ]; then
    for d in output/s61_panel/*/; do
      printf "%-24s %s\n" "$(basename "$d")" "$(ls "$d" 2>/dev/null | grep -c '^[0-9]\{4\}-' || true)"
    done
  else
    echo "(未开始)"
  fi
  echo "--- R3.5 动物园缓存已打月数 ---"
  if [ -d output/bt_scores_cache ]; then
    echo "csv 文件数: $(ls output/bt_scores_cache/*_2e4ec0f5.csv 2>/dev/null | wc -l) / 145"
  else
    echo "(未开始)"
  fi
  echo "--- 规范面板（canonical, maxn=500）---"
  [ -d output/p1_panel ] && echo "已完成月数: $(ls output/p1_panel/*.csv 2>/dev/null | grep -c '^[0-9]\{4\}' || true) / 159（含 universe.csv/manifest.csv）"
}

run_s61() {
  echo "[S6.1] 启动 11 组面板构建（parallel=$N）"
  "$PY" s61_runner.py --parallel "$N" --workers 2
  echo "[S6.1] 汇总（仅报告不平判）"
  "$PY" s61_summarize.py
  echo "[S6.1] done → 汇总产物在 output/v5/s61_*"
}

run_r35() {
  echo "[R3.5] 动物园评分缓存（145 月, workers=$N）"
  "$PY" r35_rebuild_zoo_cache.py --workers "$N"
  echo "[R3.5] ML 面板 + 修复版动物园重跑"
  "$PY" _build_ml_panel.py
  "$PY" r35_zoo_redo.py
  echo "[R3.5] done → 产物在 output/v5/r35_zoo_redo/ + output/model_zoo_report.md 追加修订段"
}

case "$LANE" in
  s61)    run_s61 ;;
  r35)    run_r35 ;;
  all)    run_r35; run_s61 ;;
  status) status ;;
  *)      echo "用法: $0 {s61|r35|all|status} [并行度N]"; exit 1 ;;
esac
