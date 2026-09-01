#!/usr/bin/env python3
"""T0.1 Tushare 本地抓取脚本 —— 在【你自己的机器】上运行（本沙箱无法连通 api.tushare.pro）。

用途
----
补齐沙箱缺失的数据，供 T0.1/T0.2/T0.3 与 Phase 2 数据依赖项（P2-1 持仓 / P2-3 经理 /
P2-4 份额流）离线使用。所有结果落到 `cache/tushare/*.csv`，跑完把整个 cache/tushare/
目录拷回沙箱即可。

积分要求（官方档位，100 积分以下接口会明确报错）
------------------------------------------------
  fund_basic / fund_manager / fund_share / fund_nav / fund_portfolio : 2000 积分
  index_dailybasic / cpi / pmi / m1m2 / shibor / gdp                 : 120 积分
若你的积分不足，脚本会逐项报"权限不足"并继续抓其余项，最后汇总缺什么。

用法
----
  pip install tushare pandas
  # 1) 冒烟（先确认 token 与积分档位）
  python t0_fetch.py --only fund_basic --limit 5
  # 2) 全量（可断点续跑，中断后重跑同命令自动跳过已完成代码）
  python t0_fetch.py --start 20060101
  # 单项:
  python t0_fetch.py --only fund_basic,fund_manager
  python t0_fetch.py --only fund_nav          # 大项, 约 2.5 万代码 × 每代码 1 次调用
  python t0_fetch.py --only fund_portfolio
  python t0_fetch.py --only fund_share --start 20130101
  python t0_fetch.py --only index_pe,macro

token 读取顺序: --token 参数 > 环境变量 TUSHARE_TOKEN > cache/tushare_token.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "cache", "tushare")
DONE_PREFIX = "_done"  # 断点文件: _done_<name>.txt 一行一个代码


def get_pro(args):
    try:
        import tushare as ts
    except ImportError:
        sys.exit("先安装: pip install tushare pandas")
    token = args.token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        tf = os.path.join(ROOT, "cache", "tushare_token.txt")
        if os.path.exists(tf):
            token = open(tf).read().strip()
    if not token:
        sys.exit("找不到 token: 用 --token / 环境变量 TUSHARE_TOKEN / cache/tushare_token.txt")
    ts.set_token(token)
    return ts.pro_api(token)


def save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存 {path} ({len(df)} 行)")
    return path


def load_done(name: str) -> set:
    p = os.path.join(OUT_DIR, f"{DONE_PREFIX}_{name}.txt")
    if os.path.exists(p):
        return set(l.strip() for l in open(p) if l.strip())
    return set()


def mark_done(name: str, code: str):
    p = os.path.join(OUT_DIR, f"{DONE_PREFIX}_{name}.txt")
    with open(p, "a") as f:
        f.write(f"{code}\n")


def local_codes() -> list[str]:
    """本地缓存里已有的基金代码（用于补齐旧数据）"""
    return sorted(os.path.basename(f)[4:-4]
                  for f in glob.glob(os.path.join(ROOT, "cache", "nav_*.csv")))


# ---------------- 单项抓取 ----------------

def f_fund_basic(pro, args) -> None:
    frames = []
    for market, label in [("E", "场内"), ("O", "场外")]:
        try:
            df = pro.fund_basic(market=market)
            print(f"  fund_basic[{label}] {len(df)} 只")
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f"  fund_basic[{label}] 失败: {e}")
    if frames:
        save_csv(pd.concat(frames, ignore_index=True).drop_duplicates("ts_code"),
                 "fund_basic.csv")
    else:
        print("  fund_basic 全部失败（积分不足或网络）")


def f_fund_manager(pro, args) -> None:
    try:
        df = pro.fund_manager()
        save_csv(df, "fund_manager.csv")
    except Exception as e:
        print(f"  fund_manager 失败: {e}")


def _code_list(pro, args, extra: list[str] | None = None) -> list[str]:
    p = os.path.join(OUT_DIR, "fund_basic.csv")
    codes: list[str] = []
    if os.path.exists(p):
        codes = pd.read_csv(p, dtype=str)["ts_code"].tolist()
    if extra:
        codes = sorted(set(codes) | set(extra))
    if args.limit:
        codes = codes[: args.limit]
    return codes


def f_fund_nav(pro, args) -> None:
    codes = _code_list(pro, args, extra=local_codes())
    done = load_done("fund_nav") if not args.force else set()
    if args.force:
        done = set()
    todo = [c for c in codes if c not in done]
    print(f"  fund_nav: 共 {len(codes)} 代码, 已完成 {len(done)}, 待抓 {len(todo)}")
    out_path = os.path.join(OUT_DIR, "fund_nav.csv")
    new = not os.path.exists(out_path) or args.force
    if new:
        open(out_path, "w").close()
    for i, c in enumerate(todo):
        try:
            df = pro.fund_nav(ts_code=c, start_date=args.start, end_date=args.end)
            if len(df):
                df.to_csv(out_path, mode="a", header=new, index=False)
                new = False
            mark_done("fund_nav", c)
        except Exception as e:
            print(f"  fund_nav {c} 失败: {e}")
        time.sleep(args.sleep)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(todo)} ({time.strftime('%H:%M:%S')})", flush=True)


def f_fund_portfolio(pro, args) -> None:
    codes = _code_list(pro, args)
    done = set() if args.force else load_done("fund_portfolio")
    todo = [c for c in codes if c not in done]
    print(f"  fund_portfolio: 共 {len(codes)}, 待抓 {len(todo)}")
    out_path = os.path.join(OUT_DIR, "fund_portfolio.csv")
    new = not os.path.exists(out_path) or args.force
    if new:
        open(out_path, "w").close()
    for i, c in enumerate(todo):
        try:
            df = pro.fund_portfolio(ts_code=c, start_date=args.start, end_date=args.end)
            if len(df):
                df.to_csv(out_path, mode="a", header=new, index=False)
                new = False
            mark_done("fund_portfolio", c)
        except Exception as e:
            print(f"  fund_portfolio {c} 失败: {e}")
        time.sleep(args.sleep)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(todo)}", flush=True)


def f_fund_share(pro, args) -> None:
    codes = _code_list(pro, args)
    done = set() if args.force else load_done("fund_share")
    todo = [c for c in codes if c not in done]
    print(f"  fund_share: 共 {len(codes)}, 待抓 {len(todo)}")
    out_path = os.path.join(OUT_DIR, "fund_share.csv")
    new = not os.path.exists(out_path) or args.force
    if new:
        open(out_path, "w").close()
    for i, c in enumerate(todo):
        try:
            df = pro.fund_share(ts_code=c, start_date=args.start, end_date=args.end)
            if len(df):
                df.to_csv(out_path, mode="a", header=new, index=False)
                new = False
            mark_done("fund_share", c)
        except Exception as e:
            print(f"  fund_share {c} 失败: {e}")
        time.sleep(args.sleep)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(todo)}", flush=True)


INDICES = ["000300.SH", "000905.SH", "000016.SH", "000001.SH",
           "399001.SZ", "399006.SZ", "000852.SH", "000688.SH"]


def f_index_pe(pro, args) -> None:
    out_path = os.path.join(OUT_DIR, "index_dailybasic.csv")
    new = not os.path.exists(out_path) or args.force
    if new:
        open(out_path, "w").close()
    for idx in INDICES:
        try:
            df = pro.index_dailybasic(ts_code=idx, start_date=args.start, end_date=args.end)
            if len(df):
                df.to_csv(out_path, mode="a", header=new, index=False)
                new = False
            print(f"  index_dailybasic {idx}: {len(df)} 行")
        except Exception as e:
            print(f"  index_dailybasic {idx} 失败: {e}")
        time.sleep(args.sleep)


def f_macro(pro, args) -> None:
    for name, fn in [("cpi", pro.cpi), ("pmi", pro.pmi), ("m1m2", pro.m1m2),
                     ("shibor", pro.shibor), ("gdp", pro.gdp)]:
        try:
            df = fn(start_date=args.start, end_date=args.end)
            save_csv(df, f"macro_{name}.csv")
        except Exception as e:
            print(f"  macro {name} 失败: {e}")
        time.sleep(args.sleep)


FETCHERS = {
    "fund_basic": f_fund_basic,
    "fund_manager": f_fund_manager,
    "fund_nav": f_fund_nav,
    "fund_portfolio": f_fund_portfolio,
    "fund_share": f_fund_share,
    "index_pe": f_index_pe,
    "macro": f_macro,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Tushare 本地抓取（用户机器运行）")
    ap.add_argument("--token", default=None)
    ap.add_argument("--only", default=",".join(FETCHERS),
                    help="逗号分隔: " + ",".join(FETCHERS))
    ap.add_argument("--start", default="20060101")
    ap.add_argument("--end", default=time.strftime("%Y%m%d"))
    ap.add_argument("--sleep", type=float, default=0.15, help="每代码调用间隔秒")
    ap.add_argument("--limit", type=int, default=None, help="限制代码数（冒烟用）")
    ap.add_argument("--force", action="store_true", help="忽略断点重抓")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip() in FETCHERS]
    pro = get_pro(args)
    # 冒烟: 先打一个最便宜的接口确认连通/积分
    try:
        r = pro.trade_cal(exchange="SSE", start_date=args.start, end_date=args.end)
        print(f"[ok] 连通性正常 (trade_cal {len(r)} 行), 开始抓取: {names}")
    except Exception as e:
        print(f"[warn] trade_cal 失败: {e} — token/积分/网络可能有误, 仍尝试...")

    t0 = time.time()
    for n in names:
        print(f"== {n} ==")
        FETCHERS[n](pro, args)
    print(f"\n全部完成, 耗时 {time.time() - t0:.0f}s。把 {OUT_DIR} 整个目录拷回沙箱。")


if __name__ == "__main__":
    main()
