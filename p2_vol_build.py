#!/usr/bin/env python3
"""P2-2 辅助面板: 为 P1-0 面板的每个 (决策月, 基金) 计算低波因子原始量。

字段(严格 PIT, 只用决策日及以前净值):
  vol126  = 过去 126 个交易日日收益率 std × √252
  vol252  = 过去 252 个交易日日收益率 std × √252
  tuw252  = 过去 252 天中净值低于该 252 天窗口最高值的天数占比
特质波动(需基准构造) 本轮预登记排除, 不在此计算。

产物: output/p2/vol_panel.csv (date, code, vol126, vol252, tuw252, n_nav)
复现: ./.venv/bin/python p2_vol_build.py [--force]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(ROOT, "output", "p1_panel")
OUT_DIR = os.path.join(ROOT, "output", "p2")
OUT_CSV = os.path.join(OUT_DIR, "vol_panel.csv")
NAV_DIR = os.path.join(ROOT, "cache")
SQRT252 = np.sqrt(252)
MAX_LAG_DAYS = 7  # 与面板 PIT 规则一致: 最后披露日须在 d-7 天内


def load_panel_dates() -> pd.DataFrame:
    files = sorted(f for f in os.listdir(PANEL_DIR)
                   if f.endswith(".csv") and f[0:2].isdigit())
    parts = []
    for f in files:
        df = pd.read_csv(os.path.join(PANEL_DIR, f), dtype={"code": str},
                         usecols=["code", "date"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_code_series(nav_path: str):
    """返回 (dates: datetime64 数组, vol126, vol252, tuw252 同长 float 数组) 或 None"""
    try:
        nav = pd.read_csv(nav_path, parse_dates=["date"])
    except Exception:
        return None
    if len(nav) < 30 or not {"date", "nav"}.issubset(nav.columns):
        return None
    nav = nav.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    s = nav["nav"].astype(float)
    if (s <= 0).any() or s.nunique() < 2:
        return None
    dates = nav["date"].values
    rets = s.pct_change()
    v126 = (rets.rolling(126, min_periods=126).std() * SQRT252).values
    v252 = (rets.rolling(252, min_periods=252).std() * SQRT252).values
    mx = s.rolling(252, min_periods=252).max()
    tuw = ((s < mx).rolling(252, min_periods=252).mean()).values
    return dates, v126, v252, tuw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV) and not args.force:
        print(f"vol_panel.csv 已存在 ({OUT_CSV}), 跳过 (--force 重建)")
        return

    t0 = time.time()
    panel = load_panel_dates()
    print(f"面板 (date,code) 行数: {len(panel)}, 唯一代码: {panel['code'].nunique()}",
          flush=True)

    nav_index = {}
    for f in glob.glob(os.path.join(NAV_DIR, "nav_*.csv")):
        code = os.path.basename(f)[4:-4]
        nav_index[code] = f

    recs = []
    n_filled = 0
    for code, g in panel.groupby("code", sort=False):
        path = nav_index.get(code)
        if path is None:
            continue
        cs = build_code_series(path)
        if cs is None:
            continue
        dates, v126, v252, tuw = cs
        pos = np.searchsorted(dates, g["date"].values, side="right") - 1
        ok = pos >= 0
        # PIT: 最后披露日须在决策日 7 天内
        if ok.any():
            lag = (g["date"].values[ok] - dates[pos[ok]]) / np.timedelta64(1, "D")
            ok[ok] = lag <= MAX_LAG_DAYS
        vals1 = np.where(ok, v126[np.clip(pos, 0, len(v126) - 1)], np.nan)
        vals2 = np.where(ok, v252[np.clip(pos, 0, len(v252) - 1)], np.nan)
        valst = np.where(ok, tuw[np.clip(pos, 0, len(tuw) - 1)], np.nan)
        n_filled += int(np.isfinite(vals1).sum() + np.isfinite(vals2).sum())
        rec = {
            "date": g["date"].values,
            "code": g["code"].values,
            "vol126": vals1,
            "vol252": vals2,
            "tuw252": valst,
            "n_nav": np.where(ok, pos + 1, 0),
        }
        recs.append(pd.DataFrame(rec))
        if len(recs) % 800 == 0:
            print(f"  {len(recs)}/{panel['code'].nunique()} codes, "
                  f"{time.time() - t0:.0f}s", flush=True)

    out = pd.concat(recs, ignore_index=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"完成: {len(out)} 行, vol126 非空 {int(out['vol126'].notna().sum())}, "
          f"vol252 非空 {int(out['vol252'].notna().sum())}, "
          f"tuw252 非空 {int(out['tuw252'].notna().sum())}, "
          f"耗时 {time.time() - t0:.0f}s → {OUT_CSV}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
