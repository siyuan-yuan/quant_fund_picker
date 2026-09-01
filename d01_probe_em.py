#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #10（D0.1）：已消失基金数据可得性探针（用户本机执行）

为什么必须用户侧：执行环境网络出口封锁（curl 证据：fund.eastmoney.com 返回 000、
tushare TLS ERROR_SYSCALL、pypi 200——仅 pypi 通）。本脚本需在联网本机运行。

预登记判定规则（先于运行固定）：
  对"已知死亡/清盘"样本（分级基金下折清盘批次 65 只 + 清盘混合基金）逐个用 akshare
  `fund_open_fund_info_em(code, indicator="单位净值走势")` 探取：
    覆盖判据：成功取到**净值历史**（≥200 行）的样本占比 ≥ 30% → 判「可得」，
              D0.1 降为工具开发任务；< 30% → 判「不可得」，U4 冻结维持、SURV-ADJ
              口径正式接手全部裁决并显著标注。
  东财 lsjz（js 接口）与天天基金移动端两类端点都试。
  绝对纪律：不因单次失败下结论；错误分类落盘。

样例池构建（静态，2026-09-01 锚定）：分级基金清盘批次（2015 股灾上折/下折触发）
+ 少量主动/指数清盘个案。对照组 = 3 只确认在世基金。

产物：output/v5/d01_probe_results.csv（逐只：code/端点/成败/错误类型）、
      output/v5/d01_probe_summary.md（覆盖率、判定、后续动作指引）
"""
from __future__ import annotations

import os
import sys
import time

DEAD_SAMPLES = [
    # 分级A/B 与母基金清盘/转型样例（历史批次命名记忆件；真实可得性交由本探针判定）
    "150209", "150210", "150228", "150252", "150276", "150296", "150303", "150304",
    "150315", "150316", "150317", "150318", "150321", "150322", "150326", "150335",
    "150336", "150344", "150347", "150348", "150130", "150131", "150139", "150140",
    "150149", "150150", "150153", "150154", "150158", "150175", "150176", "150181",
    "150182", "150187", "150188", "150191", "150192", "150193", "150194", "150195",
    "150196", "150197", "150198", "150201", "150204", "150205", "150206", "150207",
    "150208", "150211", "150212", "150213", "150214", "150216", "150217", "150218",
    "150219", "150220", "150221", "150223", "150224", "150225", "150226", "150227",
    "150229", "150230",
]  # 65 只（已知历史里有大批清盘/转 LOF 的分级份额）
CONTROLS = ["000001", "110022", "519736"]   # 对照：确信在世的基金
ALL = DEAD_SAMPLES + CONTROLS


def probe_one(code: str, sleep: float = 0.4) -> dict:
    rec = dict(code=code, endpoint="", ok=False, rows=0, err_class="")
    try:
        import akshare as ak
    except ImportError:
        print("❌ 需要 akshare"); sys.exit(2)
    # 端点1：fund_open_fund_info_em（天天基金 单位净值走势）
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        rec.update(endpoint="em_open_info", ok=df is not None and len(df) > 0,
                   rows=0 if df is None else len(df))
        time.sleep(sleep)
        return rec
    except Exception as e:
        rec["err_class"] = type(e).__name__ + ":" + str(e)[:60]
    # 端点2：fund_etf_spot/历史（部分分级转 LOF 后在场内可查）
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="")
        rec.update(endpoint="etf_hist_em", ok=df is not None and len(df) > 0,
                   rows=0 if df is None else len(df))
    except Exception as e:
        rec["err_class"] += " | etf:" + type(e).__name__ + ":" + str(e)[:40]
        rec["endpoint"] = rec["endpoint"] or "etf_hist_em"
    time.sleep(sleep)
    return rec


def main():
    os.makedirs("output/v5", exist_ok=True)
    rows = []
    for i, c in enumerate(ALL, 1):
        r = probe_one(c)
        rows.append(r)
        mark = "✓" if r["ok"] else "✗"
        print(f"[{i}/{len(ALL)}] {mark} {c} via {r['endpoint']} rows={r['rows']} {r['err_class']}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("output/v5/d01_probe_results.csv", index=False, encoding="utf-8-sig")

    dead = df[df.code.isin(DEAD_SAMPLES)]
    ctrl = df[df.code.isin(CONTROLS)]
    cov_l = (dead.ok & (dead.rows >= 200)).mean() if len(dead) else 0.0
    ctrl_ok = ctrl.ok.mean() if len(ctrl) else 0.0
    verdict = "可得" if cov_l >= 0.30 else "不可得"
    next_step = ("D0.1 转工具开发任务：对全部命中样例批量拉取历史并核对上市/退市日。"
                 if verdict == "可得" else
                 "U4 冻结维持：SURV-ADJ 口径接手全部裁决并显著标注；"
                 "备选：Tushare fund_basic 历史名录（2000 积分）或 Wind/Choice 采购评估。")
    L = ["# D0.1 已消失基金数据可得性探针（用户本机执行）", "",
         f"- 死亡样本 {len(dead)}：覆盖（≥200 行历史）率 = **{cov_l:.1%}**",
         f"- 对照组 {len(ctrl)}：成功率 {ctrl_ok:.0%}（若对照失败则探针本身失效，结果作废）", "",
         f"## 判定（预登记门 30%）：**{verdict}**", "", next_step, "",
         "## 错误分类", df.groupby(["err_class"]).size().to_markdown()]
    open("output/v5/d01_probe_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
