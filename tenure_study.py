# -*- coding: utf-8 -*-
"""
否决研究 Part B: 经理任期<3年一票否决 是否有预测力?
近似PiT任期: tenure_T = 现任期(评估日=2026-08-03) - (今天 - T)
  >0 才可归因; 现任经理在T尚未上任 → 记0 (无法归因给现任, 即否决语义)
"""
import glob, json, time
import numpy as np
import pandas as pd

import provider
provider.STALE_OK = True
from backtest import forward_returns

TODAY = pd.Timestamp("2026-08-03")
DATES = [str(d.date()) for d in pd.date_range("2006-03-31", "2026-03-31", freq="QE")]


def tenure_now(code):
    fp = f"cache/dossier_{code}.json"
    try:
        d = json.load(open(fp, encoding="utf-8"))
        ts = [provider.parse_worktime_days(m.get("workTime", "")) for m in d.get("managers", [])]
        return max(ts) if ts else np.nan
    except Exception:
        return np.nan


def main():
    t0 = time.time()
    # 池 = scan 榜单 + veto_rows 已评分过的代码 (有净值缓存)
    codes = set()
    for fp in glob.glob("output/scan_*.csv"):
        codes |= set(pd.read_csv(fp, dtype={"code": str})["code"])
    codes = sorted(codes)
    print(f"[池] {len(codes)} 只(有档案缓存的才计入)")
    rows = []
    for code in codes:
        tn = tenure_now(code)
        if tn != tn:
            continue
        tn = float(tn)
        for d in DATES:
            t_T = tn - (TODAY - pd.Timestamp(d)).days
            t_T = max(t_T, 0.0)          # 现任经理当时未上任 → 0 (即"无法归因")
            f6, f12 = forward_returns(code, d)
            if f6 != f6:
                continue
            rows.append(dict(date=d, code=code, tenure_days=round(t_T, 0),
                             tenure_y=round(t_T / 365, 2),
                             fwd6=round(f6, 4),
                             fwd12=None if f12 != f12 else round(f12, 4)))
    df = pd.DataFrame(rows)
    df.to_csv("output/tenure_rows.csv", index=False, encoding="utf-8-sig")
    print(f"[rows] {len(df)}  ({time.time()-t0:.0f}s)")

    # 按日期截面分位 (剔除行情beta影响) + 原始收益
    df["pct6"] = df.groupby("date")["fwd6"].rank(pct=True)
    bins = [0, 365, 730, 1095, 1825, 99999]
    labs = ["<1年", "1~2年", "2~3年", "3~5年", ">5年"]
    df["bucket"] = pd.cut(df.tenure_days, bins=bins, labels=labs, right=False)
    g = df.groupby("bucket", observed=True).agg(
        n=("fwd6", "size"),
        fwd6_mean=("fwd6", "mean"), fwd6_med=("fwd6", "median"),
        win=("fwd6", lambda s: (s > 0).mean()),
        pct6_mean=("pct6", "mean"),
        fwd12_mean=("fwd12", "mean"),
        big_win=("fwd6", lambda s: (s > 0.25).mean()))
    print(g.round(4).to_string())
    g.round(4).to_csv("output/tenure_summary.csv", encoding="utf-8-sig")

    # 关键对比: <3年 vs >=3年
    a = df[df.tenure_days < 1095]; b = df[df.tenure_days >= 1095]
    print(f"\n<3年组 n={len(a)}: fwd6均值{a.fwd6.mean():+.3f} 截面分位{a.pct6.mean():.3f} "
          f"fwd12均值{a.fwd12.mean():+.3f} 大赢(>+25%)占比{(a.fwd6>0.25).mean():.1%}")
    print(f">=3年组 n={len(b)}: fwd6均值{b.fwd6.mean():+.3f} 截面分位{b.pct6.mean():.3f} "
          f"fwd12均值{b.fwd12.mean():+.3f} 大赢(>+25%)占比{(b.fwd6>0.25).mean():.1%}")
    # t检验
    from scipy import stats
    t, p = stats.ttest_ind(a.fwd6.dropna(), b.fwd6.dropna(), equal_var=False)
    print(f"fwd6差值 t={t:.2f} p={p:.3f}")
    tp, pp = stats.ttest_ind(a.pct6.dropna(), b.pct6.dropna(), equal_var=False)
    print(f"截面分位差值 t={tp:.2f} p={pp:.3f}")


if __name__ == "__main__":
    main()
