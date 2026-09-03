# -*- coding: utf-8 -*-
"""
P1-0 严格 PIT 全池因子面板构建（2013-01 → 2026-03，月度决策日）

协议见 docs/优化计划_V4_2026-08.md §0：
  - 研究池 = 本地缓存全部净值文件中类型合格(偏股/股票/指数股票/灵活/指数-海外)且非C/E份额
  - 决策日 d 合格 ⇔ d 前净值行数≥800 且 最后披露≥d-7天 (引擎 min_days=800 为最终裁决)
  - |E(d)|>maxn 时种子抽样 (seed=int(YYYYMMDD))
  - 评分一律 score_fund(as_of=d, bt=True)，前瞻收益只用 d 之后数据
  - 逐月原子写 CSV，可断点续跑；每月清空净值内存缓存
  - S_total/F_momentum/rating 按 engine.finalize 同式离线重算（批内 rank），不二次打分
复现: ./.venv/bin/python p1_panel_build.py [--start 2013-01 --end 2026-03 --maxn 500 --limit N]
"""
import os, time, glob, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True          # 离线: 只用本地缓存, 绝不打源
import rbsa
import factors as F
import risk as R
from engine import score_fund, resolve_weights, market_water
from config import (CACHE_DIR, RATING_BANDS, YOUNG_MAX_DAYS)

# 【C8 研究纪律护栏, 2026-09-02】本面板为 canonical 全池因子面板(仅评估, 非训练样本)：
# fwd1/3/6 标签基期 = asof(决策日)（score_one 内 idx = searchsorted(d,'right')-1，标签口径
# exec_delay=0，base 不越过决策日 → 无 F4 类前视）。护栏把标签契约四元组写入产物 manifest 头；
# 不改任何 fwd/评分数值。执行层 T+1(D0.4) 属另一层口径，分离打标。
import research_guard as RG

PANEL_DIR = os.path.join("output", "p1_panel")

# C8 四元组：标签窗(fwd1/3/6, 交易日21/63/126) / 特征窗(决策日d截断) / 训练截止(评估标签不作训练截止)
# 执行延迟(标签口径 0 交易日)。horizon_months 仅为审计可读近似，真实窗见 HORIZONS(trading days)。
P1_CONTRACT = RG.LabelSpec(
    pipeline="p1_panel_canonical",
    horizon_months=(1, 3, 6),
    feature_snapshot_rule="决策日 d 当月截断；资格=净值≥800行 + 最近披露≥d-7天；类型来自基金类型字段(≤d 可判定)",
    train_cutoff_rule="评估标签(仅 IC/相关性)，不作训练；如被下游用于训练须另走 h 匹配截止",
    exec_delay_days=0,
    base_rule="标签以 asof(决策日) 当月 bar 为基期起算 (delay=0)；fwd1/3/6 = 21/63/126 交易日",
)

def _set_panel_dir(p):
    global PANEL_DIR
    PANEL_DIR = p
    os.makedirs(PANEL_DIR, exist_ok=True)
_set_panel_dir(PANEL_DIR)

TARGET_TYPES = {"混合型-偏股", "股票型", "指数型-股票", "混合型-灵活", "指数型-海外股票"}
HORIZONS = {"fwd1": 21, "fwd3": 63, "fwd6": 126}

# F1-3 预登记动量窗口变体: (名称, start, end)  end=0 表示不剔除(至 t)
MOM_VARIANTS = [
    ("r4", 100, 21), ("r7", 147, 21),          # 现行 4M-1M / 7M-1M
    ("r3s1", 63, 21), ("r3s0", 63, 0),
    ("r4s0", 100, 0), ("r4s2", 100, 42),
    ("r7s2", 147, 42), ("r12s1", 189, 21),
]


# ---------------- 研究池 ----------------
def build_universe():
    path = os.path.join(PANEL_DIR, "universe.csv")
    if os.path.exists(path):
        uni = pd.read_csv(path, dtype={"code": str}).set_index("code")
        uni["first"] = pd.to_datetime(uni["first"], errors="coerce")
        uni["last"] = pd.to_datetime(uni["last"], errors="coerce")
        return uni
    rows = []
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "nav_*.csv")))
    for i, f in enumerate(files):
        code = os.path.basename(f)[4:-4]
        try:
            d = pd.read_csv(f, usecols=["date"])["date"]
            if len(d) == 0:
                continue
            rows.append({"code": code, "first": d.min(), "last": d.max(), "n": len(d)})
        except Exception:
            continue
        if (i + 1) % 3000 == 0:
            print(f"[universe] 读取 {i+1}/{len(files)}", flush=True)
    uni = pd.DataFrame(rows).set_index("code")
    uni["first"] = pd.to_datetime(uni["first"], errors="coerce")
    uni["last"] = pd.to_datetime(uni["last"], errors="coerce")
    meta = provider.get_fund_meta()
    uni = uni.join(meta[["基金简称", "基金类型"]], how="left")
    uni["type_ok"] = uni["基金类型"].isin(TARGET_TYPES)
    uni["ce"] = uni["基金简称"].fillna("").str.strip().str.endswith(("C", "E"))
    uni = uni[uni["type_ok"] & ~uni["ce"]].dropna(subset=["first", "last"])
    uni.to_csv(path, encoding="utf-8-sig")
    print(f"[universe] 研究池 {len(uni)} 只 → {path}", flush=True)
    return uni


# ---------------- 单基金打分 + 原始量 + 前瞻收益 ----------------
def _win(s, start, end):
    if len(s) < start + 1:
        return np.nan
    a = float(s.iloc[-start])
    b = float(s.iloc[-1] if end == 0 else s.iloc[-end])
    if a <= 0 or a != a:
        return np.nan
    return b / a - 1


def score_one(code, d):
    try:
        r = score_fund(code, as_of=str(d), bt=True)
    except Exception:
        return None
    if r.get("error"):
        return None
    try:
        nav = provider.get_fund_nav(code).set_index("date")
        ts_d = pd.Timestamp(d)
        adj = (1 + nav["ret"].fillna(0)).cumprod()
        idx = int(adj.index.searchsorted(ts_d, side="right")) - 1
        if idx < 0 or idx >= len(adj):
            return None
        s_trunc = adj.iloc[: idx + 1]
        pdt = r.get("penalty_detail") or {}
        rec = dict(code=code, name=r.get("name"), ftype=r.get("ftype"),
                   is_passive=bool(r.get("is_passive")), n_days=r.get("n_days"),
                   val_pct=r.get("val_pct"), val_cov=r.get("val_coverage"),
                   valuation_blind=r.get("valuation_blind"),
                   trend_ok=r.get("trend_ok"), ma20_dist=r.get("ma20_dist"),
                   macd_dif=r.get("macd_dif"),
                   wr=r.get("ir_winrate"), dc=r.get("down_capture"),
                   F_value=r.get("F_value"), F_alpha=r.get("F_alpha"),
                   R_MDD=pdt.get("R_MDD"), smallcap_exp=pdt.get("smallcap_exp"),
                   top3_conc=pdt.get("top3_conc"),
                   penalties=[p for _, p in (r.get("penalties") or [])],
                   water=r.get("water"))
        w = r.get("rbsa") or {}
        wv_ = {k: v for k, v in w.items() if v == v}
        if wv_:
            top1 = max(wv_, key=lambda k: wv_[k])
            rec["top1_style"], rec["top1_w"] = top1, wv_[top1]
        else:
            rec["top1_style"], rec["top1_w"] = None, None
        for nm, st, en in MOM_VARIANTS:
            rec[nm] = _win(s_trunc, st, en)
        for nm, h in HORIZONS.items():
            j = idx + h
            rec[nm] = float(adj.iloc[j] / adj.iloc[idx] - 1) if j < len(adj) else np.nan
        return rec
    except Exception:
        return None


# ---------------- engine.finalize 同式离线重算 ----------------
def _total(r, wv, wa, wm):
    num, den = 0.0, 0.0
    fv, fa, fm = r["F_value"], r["F_alpha"], r["F_momentum"]
    if fv is not None and fv == fv:
        num += wv * min(fv, 100); den += wv
    if fa is not None and fa == fa:
        num += wa * fa; den += wa
    if fm is not None and fm == fm:
        num += wm * fm; den += wm
    base = (num / den) if den > 1e-9 else 0.0
    s = base
    for p in (r["penalties"] or []):          # 与 risk.apply_penalties 同义: 乘法惩罚链
        s *= (1 - p)
    return round(max(0.0, min(100.0, s)), 1)


def _rate(s, n_days):
    for th, lab in RATING_BANDS:
        if s >= th:
            return lab
    return RATING_BANDS[-1][1]


def aggregate(raw: pd.DataFrame, d):
    df = raw.copy()
    for c in ["r4", "r7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rank4"] = df["r4"].rank(pct=True)
    df["rank7"] = df["r7"].rank(pct=True)
    df["F_momentum"] = df.apply(
        lambda r: F.momentum_score_smooth_m1(r["rank4"], r["rank7"]), axis=1).round(1)
    water = market_water(as_of=str(d))          # 决策日级, 全部截断至 d (纯 PiT)
    water = float(water) if water == water else np.nan
    df["water"] = water
    (wv, wa, wm), mode = resolve_weights(water)
    df["w_value"], df["w_alpha"], df["w_mom"] = wv, wa, wm
    df["weights_mode"] = mode
    df["S_total"] = [
        _total({"F_value": r.F_value, "F_alpha": r.F_alpha,
                "F_momentum": r.F_momentum, "penalties": r.penalties}, wv, wa, wm)
        for r in df.itertuples()]
    df["rating"] = [
        ("🌱观察仓" if s >= 50 and isinstance(nd, (int, float)) and nd == nd and nd < YOUNG_MAX_DAYS
         else _rate(s, nd))
        for s, nd in zip(df["S_total"], df["n_days"])]
    df["date"] = str(d)
    # mergesort 稳定：S_total 并列时保留 code 升序（M11 修复配对）
    return df.sort_values("S_total", ascending=False, na_position="last",
                          kind="mergesort").reset_index(drop=True)


def build_date(d, uni, maxn, workers=2, seed_salt=0):
    dts = pd.Timestamp(d)
    firsts = uni["first"].values
    lasts = uni["last"].values
    codes_all = uni.index.tolist()
    ok = (firsts <= dts - pd.Timedelta(days=1100)) & (lasts >= dts - pd.Timedelta(days=7))
    elig = [c for c, f in zip(codes_all, ok) if f]
    n_elig = len(elig)
    if n_elig == 0:
        return None, 0, 0
    if n_elig > maxn:
        # S6.1: seed_salt 仅影响抽样种子；salt=0 保持历史原式 seed=int(YYYYMMDD)（既有面板不变）
        base_seed = int(str(d).replace("-", ""))
        seed = base_seed if seed_salt == 0 else base_seed * 10 + seed_salt
        elig = list(np.random.RandomState(seed).choice(elig, maxn, replace=False))
    # 【M13 修复，2026-09-03】原硬编码 max_workers=2，--workers 入参被静默忽略（执行台账登记 M13）。
    # 现改用调用方传入的 workers。数值零影响：M11 修复后评分结果按 code 稳定排序，产出与线程数无关
    # （只影响吞吐）；workers<=0 时回退 1，避免 ThreadPoolExecutor 抛 ValueError。
    nw = max(1, int(workers)) if workers else 1
    rows = []
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(score_one, c, d): c for c in elig}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                rows.append(rec)
    if not rows:
        return None, n_elig, len(elig)
    # 【M11 确定性修复】线程完成顺序不定 → 按 code 稳定归位，消除并列分槽位裁决的轮间漂移
    raw = pd.DataFrame(rows).sort_values("code", kind="mergesort").reset_index(drop=True)
    return aggregate(raw, d), n_elig, len(elig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2013-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--maxn", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个决策月(冒烟用)")
    ap.add_argument("--workers", type=int, default=2,
                    help="评分并行线程数（M13 修复后真正生效；<=0 回退 1）")
    ap.add_argument("--seed-salt", type=int, default=0,
                    help="S6.1: 抽样种子盐; 0=历史原式, >0 生成稳健性变体面板")
    ap.add_argument("--panel-dir", default=None, help="S6.1: 变体面板输出目录(默认 output/p1_panel)")
    args = ap.parse_args()
    if args.panel_dir:
        _set_panel_dir(args.panel_dir)

    uni = build_universe()
    rbsa.index_return_matrix(pd.date_range("2010-01-01", "2010-02-01"))   # 预热共享

    start_ts = pd.Timestamp(f"{args.start}-01")
    end_ts = pd.Timestamp(f"{args.end}-01") + pd.offsets.MonthEnd(0)
    month_ends = pd.date_range(start_ts, end_ts, freq="ME")
    # 决策日 = 当月最后一个交易日 (A股交易日历, 来自沪深300指数缓存;
    # 2020-01 疫情休市10天 → 2020-01-31 非交易日 → 决策日=2020-01-23)
    cal = pd.to_datetime(pd.read_csv(
        os.path.join(CACHE_DIR, "idx_sh000300.csv"), usecols=["date"])["date"])
    cal = set(cal.values)
    dates = []
    for me in month_ends:
        d = me
        while d not in cal:
            d -= pd.Timedelta(days=1)
        dates.append(str(d.date()))
    if args.limit:
        dates = dates[: args.limit]

    # 【C8】标签契约四元组写进产物 manifest 头（sidecar JSON + manifest.csv 记录 + stdout 日志头）
    side = RG.write_contract_sidecar(os.path.join(PANEL_DIR, "canonical_panel.csv"),
                                     P1_CONTRACT,
                                     manifest_path=os.path.join(PANEL_DIR,
                                                                "label_contract.manifest.jsonl"))
    print(f"[C8] 标签契约已写 {side} | 决策月 {len(dates)}")
    for _ln in P1_CONTRACT.header_lines():
        print("  " + _ln)

    man_rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        fp = os.path.join(PANEL_DIR, f"{d}.csv")
        if os.path.exists(fp):
            man_rows.append(dict(date=d, n_ok=-1, elapsed=0.0, note="cached"))
            continue
        t1 = time.time()
        out, n_elig, n_sampled = build_date(d, uni, args.maxn, args.workers, args.seed_salt)
        if out is None or not len(out):
            man_rows.append(dict(date=d, n_elig=n_elig, n_sampled=n_sampled,
                                 n_ok=0, elapsed=round(time.time() - t1, 1), note="empty"))
            print(f"[{i+1}/{len(dates)}] {d} 合格={n_elig} 无有效评分", flush=True)
        else:
            out.to_csv(fp + ".tmp", index=False, encoding="utf-8-sig")
            os.replace(fp + ".tmp", fp)
            man_rows.append(dict(date=d, n_elig=n_elig, n_sampled=n_sampled,
                                 n_ok=len(out), elapsed=round(time.time() - t1, 1), note="ok"))
            print(f"[{i+1}/{len(dates)}] {d} 合格={n_elig} 样本={n_sampled} 评分={len(out)} "
                  f"({time.time()-t1:.0f}s, 累计{time.time()-t0:.0f}s)", flush=True)
        for k in [k for k in list(provider._memo) if k.startswith("nav_")]:
            provider._memo.pop(k, None)
    man_path = os.path.join(PANEL_DIR, "manifest.csv")
    if os.path.exists(man_path):
        old = pd.read_csv(man_path, dtype={"date": str})
        man = pd.concat([old[~old["date"].isin([r["date"] for r in man_rows])],
                         pd.DataFrame(man_rows)], ignore_index=True)
    else:
        man = pd.DataFrame(man_rows)
    man.to_csv(man_path, index=False, encoding="utf-8-sig")
    print(f"[done] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
