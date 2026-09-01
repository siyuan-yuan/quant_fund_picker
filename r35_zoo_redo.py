#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #24（R3.5）：模型动物园修复重跑 + 报告勘误（SURV-ADJ）

修复点（已落本体）：F4（fwd12 前视→h 匹配 cutoff；全样本中位兜底→扩展窗；
60d 后向过滤已删）、M2 冷启动对齐（全部配置自「均有预测的首月」起评；三窗口在共同窗内切）、
M7 术语（报告中的"季"一律为月末，勘误段注明）。

决赛名单按原报告 §4.1 表锁定（不重新海筛，避免新增自由度）：
  lasso_F2_fwd6_rk, enet_F2_fwd6_rk, svrlin_F2_fwd6_z, lasso_F3_fwd6_rk,
  huber_F2_fwd6_z(V4), huber_F2_fwd12_z  +  对照 V3.7 规则分（面板 S_eng 列）

窗口（共同评起点起）：full / w2015(≥2015-01-31) / w2019(≥2019-01-31)。
统计：IC 均值、naive t、HAC t（L: fwd6→5, fwd12→11 月度）；V3.7 vs 每配置配对差 HAC t。
产物：output/v5/r35_zoo_redo/{ic_table,pairdiff,fwd12_ab}.csv + summary.md +
     output/model_zoo_report.md 追加修订段（原文保留）。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

from stats_hac import nw_tstat, HORIZON_LAGS

OUT = "output/v5/r35_zoo_redo"
ML_PANEL = "output/ml_panel.csv"
REPORT = "output/model_zoo_report.md"

FINALISTS = [("lasso", "F2", "fwd6_rk"), ("enet", "F2", "fwd6_rk"),
             ("svrlin", "F2", "fwd6_z"), ("lasso", "F3", "fwd6_rk"),
             ("huber", "F2", "fwd6_z"), ("huber", "F2", "fwd12_z")]
HIST = {  # 原报告 §4.1 引用值（修复前，仅对账）
    ("lasso", "F2", "fwd6_rk"): (0.0961, 5.56), ("enet", "F2", "fwd6_rk"): (0.0942, 5.43),
    ("svrlin", "F2", "fwd6_z"): (0.0936, 5.16), ("lasso", "F3", "fwd6_rk"): (0.0900, 5.25),
    ("huber", "F2", "fwd6_z"): (0.0876, 4.89), ("huber", "F2", "fwd12_z"): (0.0773, 4.20),
    "V3.7": (0.1127, 8.11)}


def monthly_ic(d, pred_col, ycol, min_n=30):
    recs = []
    for dt, g in d.groupby("date"):
        gg = g[[pred_col, ycol]].dropna()
        if len(gg) < min_n or gg[pred_col].nunique() < 5 or gg[ycol].nunique() < 5:
            continue
        v = gg[pred_col].corr(gg[ycol], method="spearman")
        if pd.notna(v):
            recs.append((dt, v))
    return pd.Series(dict(recs)).sort_index()


def lag_of(target):
    return HORIZON_LAGS["fwd12_monthly"] if target.startswith("fwd12") else HORIZON_LAGS["fwd6_monthly"]


def summarize(ic: pd.Series, L):
    st = nw_tstat(ic.values, L)
    return dict(n=st["n"], ic_mean=round(st["mean"], 4) if st["mean"] == st["mean"] else np.nan,
                t_naive=round(st["t_naive"], 2) if st["t_naive"] == st["t_naive"] else np.nan,
                t_hac=round(st["t_hac"], 2) if st["t_hac"] == st["t_hac"] else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(ML_PANEL):
        print("[R3.5] 构建修复版 ML 面板（_build_ml_panel 修复后本体）…", flush=True)
        subprocess.run([sys.executable, "_build_ml_panel.py"], check=True)
    df = pd.read_csv(ML_PANEL, dtype={"code": str}, parse_dates=["date"])
    print(f"[R3.5] 面板 {len(df)} 行 × {df.date.nunique()} 月末 × {df.code.nunique()} 只", flush=True)

    import _model_zoo as MZ

    preds, ic_ser = {}, {}
    for model, feats, target in FINALISTS:
        cfg = f"{model}__{feats}__{target}"
        d = MZ.oos_predictions(df, model, feats, target, retrain_every=1)
        preds[cfg] = d
        ycol = "fwd6" if target.startswith("fwd6") else "fwd12"
        ic_ser[cfg] = monthly_ic(d.dropna(subset=["pred"]), "pred", ycol)
        d[["date", "code", "pred"]].to_csv(f"{OUT}/pred_{cfg}.csv", index=False)
        print(f"  [OOS] {cfg}: 预测月 {ic_ser[cfg].shape[0]}", flush=True)

    v37_ic = monthly_ic(df.rename(columns={"S_eng": "pred"}), "pred", "fwd6")

    firsts = [s.dropna().index.min() for s in ic_ser.values() if len(s.dropna())]
    common_start = max(firsts) if firsts else None
    print(f"[R3.5] 冷启动对齐起点（全体配置有预测的首月）: {common_start.date() if common_start is not None else None}",
          flush=True)

    WINDOWS = {"full": None, "w2015": pd.Timestamp("2015-01-31"), "w2019": pd.Timestamp("2019-01-31")}
    rows, prows = [], []
    for cfg, s in ic_ser.items():
        s_c = s[s.index >= common_start] if common_start is not None else s
        L = lag_of(cfg.split("__")[2])
        for wname, lo in WINDOWS.items():
            sw = s_c if lo is None else s_c[s_c.index >= lo]
            met = summarize(sw.dropna(), L)
            rows.append(dict(cfg=cfg, window=wname, **met))
    for wname, lo in WINDOWS.items():
        vw = v37_ic if lo is None else v37_ic[v37_ic.index >= lo]
        met = summarize(vw.dropna(), HORIZON_LAGS["fwd6_monthly"])
        rows.append(dict(cfg="V3.7规则分(对照)", window=wname, **met))
    ic_tbl = pd.DataFrame(rows)
    ic_tbl.to_csv(f"{OUT}/ic_table.csv", index=False, encoding="utf-8-sig")

    for cfg, s in ic_ser.items():
        L = lag_of(cfg.split("__")[2])
        for wname, lo in WINDOWS.items():
            a = (v37_ic if lo is None else v37_ic[v37_ic.index >= lo])
            b = (s[s.index >= common_start] if lo is None else
                 s[(s.index >= common_start) & (s.index >= lo)])
            a2, b2 = a.align(b, join="inner")
            dser = (a2 - b2).dropna()
            st = nw_tstat(dser.values, L)
            prows.append(dict(pair=f"V3.7 − {cfg}", window=wname, n=st["n"],
                              diff_mean=round(st["mean"], 4) if st["mean"] == st["mean"] else np.nan,
                              t_naive=round(st["t_naive"], 2) if st["t_naive"] == st["t_naive"] else np.nan,
                              t_hac=round(st["t_hac"], 2) if st["t_hac"] == st["t_hac"] else np.nan))
    p_tbl = pd.DataFrame(prows)
    p_tbl.to_csv(f"{OUT}/pairdiff.csv", index=False, encoding="utf-8-sig")

    ab = []
    for model, feats, target in FINALISTS:
        cfg = f"{model}__{feats}__{target}"
        ic_new = ic_tbl[(ic_tbl.cfg == cfg) & (ic_tbl.window == "full")].iloc[0]
        ic_old, t_old = HIST[(model, feats, target)]
        ab.append(dict(cfg=cfg, ic_hist=ic_old, t_hist_naive=t_old,
                       ic_fixed=ic_new.ic_mean, t_fixed_naive=ic_new.t_naive,
                       t_fixed_hac=ic_new.t_hac))
    ic_old, t_old = HIST["V3.7"]
    v = ic_tbl[(ic_tbl.cfg == "V3.7规则分(对照)") & (ic_tbl.window == "full")].iloc[0]
    ab.append(dict(cfg="V3.7规则分(对照)", ic_hist=ic_old, t_hist_naive=t_old,
                   ic_fixed=v.ic_mean, t_fixed_naive=v.t_naive, t_fixed_hac=v.t_hac))
    ab_tbl = pd.DataFrame(ab)
    ab_tbl.to_csv(f"{OUT}/fwd12_ab.csv", index=False, encoding="utf-8-sig")

    L = ["# R3.5 模型动物园修复重跑（SURV-ADJ 半池重建）", "",
         "**口径警示**：重建池 = top100_history_pool 216 只中净值缓存覆盖者"
         "（覆盖率以 r35_cache_rebuild.log 为准）；面板为幸存池+历史并集双重口径，**本表仅作勘误对照，禁作未来裁决**。",
         f"冷启动对齐起点：{common_start.date() if common_start is not None else '—'}。"
         "修复：fwd12 cutoff q−6M→q−12M（h 匹配）；impute 兜底→扩展窗；60d 过滤删除。", "",
         "## fwd12/全体 A/B（修复前 → 修复后）", "",
         ab_tbl.to_markdown(index=False), "",
         "## 修复后 IC 台账（三窗口，naive 与 HAC 并列）", "",
         ic_tbl.to_markdown(index=False), "",
         "## V3.7 − ML 配对差 HAC t（差值为正 = V3.7 优）", "",
         p_tbl.to_markdown(index=False), ""]
    open(f"{OUT}/summary.md", "w", encoding="utf-8").write("\n".join(L))

    f12 = ab_tbl[ab_tbl.cfg.str.contains("fwd12")]
    f12_ic = f12["ic_fixed"].iloc[0] if len(f12) else np.nan
    err = ("\n\n---\n\n## 修订段（2026-09-01，R3.5 修复重跑；上文原文保留）\n\n"
           f"见 `output/v5/r35_zoo_redo/summary.md` 全套。"
           f"要点：① fwd12 训练截止已改为 q−12M（修复前存在未来 6 个月标签前视，huber fwd12_z "
           f"IC 修复前 0.0773(t=4.20) → 修复后 {f12_ic}"
           "；② “235 季/191 个 OOS 季度”应为 235/191 个**月末**（术语勘误，无数值影响）；"
           "③ 冷启动已对齐：全部配置自共同首月起评，5.1 表 full 窗口径以修订后为准；"
           "④ 本次重建为半池勘误级证据；"
           "⑤ 端到端三窗口表待执行层并轨（sim_core）后补出，届时以配对差 HAC t 表述，"
           "原文的『全部领先或并列第一』按修订段数字改写。\n")
    open(REPORT, "a", encoding="utf-8").write(err)
    print(f"\n[R3.5] 完成：{OUT}/summary.md + 报告修订段已追加")
    print(ab_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
