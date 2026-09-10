#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #24 步骤 6（R3.5-E2E）：模型动物园端到端回测层并轨 sim_core

口径标签：**SURV-ADJ 半池**（R3.5 重建池 110 只）。本产物仅作 IC 层结论的模型层补齐，
**禁作 FULL-PIT 裁决证据**；绝对收益数字与 B2.1 权威基线（全池 3,485 只、CAGR +10.84%）
**不可比**（本表宇宙仅 110 只、slots=10 相对池深占比极高），只许做**臂间差额**解读。

────────────────────────── 预登记（冻结于开跑前，跑完禁改） ──────────────────────────
【对象】R3.5 的 6 个决赛 ML 配置 + V3.7 规则分对照臂（共 7 臂）。

【分数可比化协议（关键自由度，先声明后执行）】
  sim_core 的买卖门（buy=70 / sell=45）作用在 V3.7 分数量纲上，而 ML 预测为 z/rank 量纲，
  直接喂入会让"门槛错配"冒充"模型差异"。故统一采用 **逐月分位映射（quantile mapping）**：
      对每个决策月 t，把该臂的预测值按截面分位，映射到**同月 V3.7 分数 S_eng 的经验分布**上。
  性质：① 严格保序（不改变该臂的选基排序，即不改变模型信息）；
       ② 各臂逐月分数分布与 V3.7 完全相同 ⇒ 买卖门、槽位、CPPI/危机层面对所有臂对称；
       ③ 对 V3.7 臂本身是恒等变换（自映射），故对照臂 = 原生 V3.7。
  该协议是**唯一**允许的分数变换；不得再引入任何逐臂调参。

【执行口径】与 B2.1 R3 完全一致：复权净值（navadj）+ T+1 成交 + 阶梯赎回成本
  （<7d 1.5% / <365d 0.5% / <730d 0.25% / ≥730d 0；申购 0.15%）+ V3.8 默认执行层参数。

【窗口】full（共同冷启动起点起）/ w2019（≥2019-01-31）。
  **w2015 已知塌缩**：R3.5 实测共同冷启动 2016-04-30 晚于 w2015 门槛 2015-01-31，
  两窗逐行相同 —— 本脚本如实只报 2 个独立窗口，不得宣称"三窗一致"。

【判定门 H2（模型层，双向）】某 ML 臂构成候选 ⟺ 在 **full 与 w2019 两窗均**满足：
    (a) ΔCAGR = CAGR(ML) − CAGR(V3.7) > 0；且
    (b) ΔMaxDD 劣化 ≤ 2pp（即 MaxDD(ML) ≥ MaxDD(V3.7) − 0.02）；且
    (c) Calmar 不降。
  与 H1（IC 配对差 HAC t>2，R3.5 已跑：full 1.10~1.43 / w2019 1.48~1.72 全未过）**双门并列**：
  仅当 H1 与 H2 同时通过才允许进入 A5.2 FULL-PIT 终审。
  **维持现状与改动同为合法结论；不预设方向。**

【产物】output/v5/r35_zoo_redo/e2e_{summary.csv, equity_<arm>.csv, trades_<arm>.csv} + e2e_summary.md
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import provider
provider.STALE_OK = True

import sim_core as SC

OUT = "output/v5/r35_zoo_redo"
ML_PANEL = "output/ml_panel.csv"
NAV_ADJ = "cache/navadj_%s.csv"

FINALISTS = ["lasso__F2__fwd6_rk", "enet__F2__fwd6_rk", "svrlin__F2__fwd6_z",
             "lasso__F3__fwd6_rk", "huber__F2__fwd6_z", "huber__F2__fwd12_z"]

# 与 B2.1 R3 逐字一致（禁改）
ARGS = dict(capital=100000.0, cost_in=0.0015, cost_out=0.005, slots=10,
            buy=70.0, sell=45.0, pool_mode="default", legacy=False,
            cash_yield=0.025, trail_stop=0.20, rebalance="quarterly",
            crisis=True, cppi=True, crisis_ma=200, crisis_vol_window=20,
            crisis_vol_q=0.80)

WINDOWS = {"full": None, "w2019": "2019-01-31"}


def ladder_fn(gross, hold_days):
    if hold_days < 7:
        return 0.015
    if hold_days < 365:
        return 0.005
    if hold_days < 730:
        return 0.0025
    return 0.0


class LazyNavs:
    """与 b21_baseline.LazyNavs 同式（复权净值懒加载）。"""

    def __init__(self, pattern):
        self.pattern, self._d = pattern, {}

    def _load(self, code):
        fp = self.pattern % code
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp, parse_dates=["date"])
        col = "adj_nav" if "adj_nav" in df.columns else "nav"
        s = df.set_index("date")[col].sort_index()
        return s[~s.index.duplicated(keep="last")].dropna()

    def __contains__(self, code):
        if code not in self._d:
            self._d[code] = self._load(code)
        return self._d[code] is not None and len(self._d[code]) > 60

    def __getitem__(self, code):
        if code not in self._d:
            self._d[code] = self._load(code)
        return self._d[code]

    def get(self, code):
        return self[code] if code in self else None


def bench_series():
    df = pd.read_csv("cache/idx_sh000300.csv", parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def quantile_map(pred: pd.Series, ref: pd.Series) -> pd.Series:
    """把 pred 按截面分位映射到 ref 的经验分布（严格保序；ref 自映射为恒等）。"""
    p = pred.dropna()
    r = ref.dropna()
    if len(p) == 0 or len(r) < 2:
        return pd.Series(np.nan, index=pred.index)
    q = p.rank(pct=True, method="average")
    out = pd.Series(np.quantile(r.values, np.clip(q.values, 0, 1)), index=p.index)
    return out.reindex(pred.index)


def build_arm_panel(base: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """base: ml_panel(含 date/code/S_eng/water/R_MDD)；score: 与 base 同索引的原始臂分数。"""
    d = base.copy()
    d["_raw"] = score.values
    d["S"] = np.nan
    for dt, g in d.groupby("date"):
        d.loc[g.index, "S"] = quantile_map(g["_raw"], g["S_eng"]).values
    d = d.dropna(subset=["S"])
    return d[["date", "code", "S", "water", "R_MDD"]].reset_index(drop=True)


def run_arm(tag, panel, navs, bench):
    args = types.SimpleNamespace(**ARGS)
    dates = [str(x) for x in sorted(panel.date.unique())]
    ec, tr = SC.simulate(panel, navs, bench, dates, args, exec_delay_days=1,
                         cost_out_fn=ladder_fn, label=tag)
    yrs = (ec.index[-1] - ec.index[0]).days / 365.25
    tot = ec.equity.iloc[-1] / ARGS["capital"] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1
    dd = float(ec.drawdown.min())
    return dict(arm=tag, n_months=len(dates), years=round(yrs, 2),
                cagr=cagr, maxdd=dd, calmar=cagr / abs(dd) if dd else np.nan,
                total_ret=tot, n_trades=len(tr),
                cost=float(ec.attrs.get("total_cost", np.nan))), ec, tr


def main():
    os.makedirs(OUT, exist_ok=True)
    base = pd.read_csv(ML_PANEL, dtype={"code": str})
    base["date"] = base["date"].astype(str).str[:10]
    need = ["date", "code", "S_eng", "water", "R_MDD"]
    miss = [c for c in need if c not in base.columns]
    assert not miss, f"ml_panel 缺列 {miss}（M15 修复未生效？）"

    # 共同冷启动起点 = 全体 ML 臂均有预测的首月（与 R3.5 IC 层同式，读预测产物）
    starts = []
    pred_map = {}
    for cfg in FINALISTS:
        fp = f"{OUT}/pred_{cfg}.csv"
        assert os.path.exists(fp), f"缺 {fp}；先跑 r35_zoo_redo.py"
        p = pd.read_csv(fp, dtype={"code": str})
        p["date"] = p["date"].astype(str).str[:10]
        p = p.dropna(subset=["pred"])
        pred_map[cfg] = p
        starts.append(p["date"].min())
    common_start = max(starts)
    print(f"[E2E] 共同冷启动起点 = {common_start}（与 IC 层同式）", flush=True)

    navs, bench = LazyNavs(NAV_ADJ), bench_series()
    rows, curves = [], {}

    for wname, lo in WINDOWS.items():
        lo_eff = common_start if lo is None else max(common_start, lo)
        b = base[base.date >= lo_eff].copy()

        # 对照臂 V3.7（分位映射对其为恒等）
        pv = build_arm_panel(b, b["S_eng"])
        r, ec, tr = run_arm("V3.7规则分(对照)", pv, navs, bench)
        r["window"] = wname
        rows.append(r)
        curves[(wname, "V37")] = ec
        if wname == "full":
            ec.reset_index().to_csv(f"{OUT}/e2e_equity_V37.csv", index=False, encoding="utf-8-sig")
            tr.to_csv(f"{OUT}/e2e_trades_V37.csv", index=False, encoding="utf-8-sig")
        print(f"  [{wname}] {'V3.7规则分(对照)':28s} CAGR={r['cagr']:+.2%} "
              f"MaxDD={r['maxdd']:.1%} trades={r['n_trades']}", flush=True)

        for cfg in FINALISTS:
            m = b.merge(pred_map[cfg], on=["date", "code"], how="left")
            pa = build_arm_panel(b, m["pred"])
            if pa.empty:
                print(f"  [{wname}] {cfg}: 无有效分数，跳过", flush=True)
                continue
            r, ec, tr = run_arm(cfg, pa, navs, bench)
            r["window"] = wname
            rows.append(r)
            curves[(wname, cfg)] = ec
            if wname == "full":
                ec.reset_index().to_csv(f"{OUT}/e2e_equity_{cfg}.csv", index=False, encoding="utf-8-sig")
                tr.to_csv(f"{OUT}/e2e_trades_{cfg}.csv", index=False, encoding="utf-8-sig")
            print(f"  [{wname}] {cfg:28s} CAGR={r['cagr']:+.2%} "
                  f"MaxDD={r['maxdd']:.1%} trades={r['n_trades']}", flush=True)

    res = pd.DataFrame(rows)
    # 相对对照臂的差额
    for w in WINDOWS:
        ref = res[(res.window == w) & (res.arm == "V3.7规则分(对照)")].iloc[0]
        sel = res.window == w
        res.loc[sel, "d_cagr"] = res.loc[sel, "cagr"] - ref["cagr"]
        res.loc[sel, "d_maxdd"] = res.loc[sel, "maxdd"] - ref["maxdd"]
        res.loc[sel, "d_calmar"] = res.loc[sel, "calmar"] - ref["calmar"]

    # 冻结门判定
    def gate(cfg):
        ok = []
        for w in WINDOWS:
            r = res[(res.window == w) & (res.arm == cfg)]
            if r.empty:
                return "缺样本", False
            r = r.iloc[0]
            ok.append(bool(r.d_cagr > 0 and r.d_maxdd >= -0.02 and r.d_calmar >= 0))
        return ("两窗均过" if all(ok) else "未过"), all(ok)

    res["h2_gate"] = [gate(a)[0] if a != "V3.7规则分(对照)" else "—" for a in res.arm]
    res = res[["window", "arm", "n_months", "years", "cagr", "maxdd", "calmar",
               "total_ret", "n_trades", "cost", "d_cagr", "d_maxdd", "d_calmar", "h2_gate"]]
    res.to_csv(f"{OUT}/e2e_summary.csv", index=False, encoding="utf-8-sig")

    passed = sorted({a for a in FINALISTS if gate(a)[1]})
    L = ["# R3.5 步骤 6：模型动物园端到端回测（sim_core 并轨）", "",
         "**口径：SURV-ADJ 半池（110 只）· 禁作 FULL-PIT 裁决证据。**", "",
         f"- 共同冷启动起点：{common_start}；执行口径 = B2.1 R3（复权 + T+1 + 阶梯成本 + V3.8 默认层）。",
         "- 分数可比化：逐月分位映射到同月 V3.7 分数分布（严格保序；对 V3.7 臂为恒等）。",
         "- **绝对收益与 B2.1 权威基线（全池 3,485 只，+10.84%）不可比**：本表宇宙仅 110 只，"
         "slots=10 占池深比例极高，只许做臂间差额解读。",
         "- **窗口披露**：w2015 因共同冷启动晚于其门槛而与 full 塌缩，本表如实只列 full / w2019 两个独立窗。", "",
         "## 端到端结果（差额相对同窗 V3.7 对照臂）", "",
         res.round(4).to_markdown(index=False), "",
         "## H2 冻结门判定（预登记于开跑前，跑完未改）", "",
         "门槛：两窗**均**满足 ΔCAGR>0 且 ΔMaxDD 劣化≤2pp 且 ΔCalmar≥0。", ""]
    if passed:
        L += [f"- **过 H2 的臂**：{', '.join(passed)}。",
              "- 但 H1（IC 配对差 HAC t>2）在 R3.5 中**全部未过**（full 1.10~1.43 / w2019 1.48~1.72）"
              "⇒ **双门未同时通过，维持现状（不引入 ML 腿）**；仅可申请在 A5.2 FULL-PIT 下重开评断。"]
    else:
        L += ["- **无任何 ML 臂通过 H2**。",
              "- 叠加 H1 全部未过 ⇒ **双门皆堵：ML 腿不构成候选，维持 V3.7 现状**（结论方向与 IC 层一致）。"]
    L += ["", "> 双向裁决声明：维持现状与改动同为合法结论。本表不构成对 ML 方法本身的否定——"
          "半池 110 只 / slots=10 的宇宙深度不足是已披露的结构性限制，终审在 A5.2（FULL-PIT 全池）。"]
    open(f"{OUT}/e2e_summary.md", "w", encoding="utf-8").write("\n".join(L))
    print(f"\n[E2E] 产物：{OUT}/e2e_summary.csv + e2e_summary.md")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
