# -*- coding: utf-8 -*-
"""命令行入口：python -m pit <subcommand>

子命令：
  init     初始化目录与清单骨架
  master   抓取 Tushare 全历史主表（--import 可导入 Wind/Choice/CSMAR CSV）
  csrc     下载/解析证监会历期《公募基金产品索引》
  daily    归档今日真实截面（路线 C，从今天起）
  events   校验事件表（成立/清盘/转型/暂停/恢复/限购）
  build    物化月末不可变快照（严格策略默认；--policy lite 显式降级）
  qa       独立运行质量门禁 G1–G8
  report   打印数据覆盖/严格度报告

示例：
  TUSHARE_TOKEN=xxx python -m pit master
  python -m pit csrc --manifest data/pit_manifest/csrc_sources.csv
  python -m pit daily
  python -m pit build --enforce-qa
  python -m pit build --policy lite --type-policy fallback --purchase-policy unknown-keep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_init(args) -> int:
    from pit.common import atomic_write_bytes
    (Path("data/pit_manifest")).mkdir(parents=True, exist_ok=True)
    (Path("data/pit_events")).mkdir(parents=True, exist_ok=True)
    (Path("data/pit_universe")).mkdir(parents=True, exist_ok=True)
    manifest = Path("data/pit_manifest/csrc_sources.csv")
    if not manifest.exists():
        # published_at 留空=未知 → 归档时标记 known_at_approx=True（用下载日），
        # 绝不伪造“发布日期”。请从页面“发文日期”补填后再宣称严格。
        atomic_write_bytes(manifest, (
            "as_of,published_at,title,page_url,file_url\n"
            "2026-06-30,,公募基金产品索引（截至20260630）,"
            "https://www.csrc.gov.cn/csrc/c101900/c1029655/content.shtml,"
            "https://www.csrc.gov.cn/csrc/c101900/c1029655/1029655/files/"
            "%E5%85%AC%E5%8B%9F%E5%9F%BA%E9%87%91%E4%BA%A7%E5%93%81%E7%B4%A2%E5%BC%95"
            "%EF%BC%88%E6%88%AA%E8%87%B320260630%EF%BC%89.xlsx\n").encode("utf-8"))
    print("[init] 已创建 data/pit_manifest|csrc_sources.csv, data/pit_events/, data/pit_universe/")
    print("[init] 提示：请把证监会历史各期索引页 URL 逐行补进 csrc_sources.csv"
          "（as_of / published_at 尽量从页面填写）。")
    return 0


def cmd_master(args) -> int:
    from pit.sources_master import fetch_and_archive_tushare, import_vendor_master
    if args.import_csv:
        path, meta = import_vendor_master(Path(args.import_csv), Path(args.raw_dir),
                                          source=args.source or "vendor")
    else:
        path, meta = fetch_and_archive_tushare(Path(args.raw_dir), token=args.token or None)
    print(f"[master] 归档于 {path}，{meta['rows']} 行")
    print("[master] 注意：status/fund_type 为当前口径；purc_startdate 不等于暂停区间。")
    return 0


def cmd_csrc(args) -> int:
    from pit.sources_csrc import download_from_manifest, parse_archive
    res = download_from_manifest(Path(args.manifest), Path(args.raw_dir))
    print(f"[csrc] 归档 {len(res)} 期索引: {[x['as_of'] for x in res]}")
    parsed = parse_archive(Path(args.raw_dir))
    for k, v in sorted(parsed.items()):
        print(f"[csrc] {k}: {len(v)} 行 (known_at={v['known_at'].iloc[0]})")
    return 0


def cmd_daily(args) -> int:
    from pit.sources_daily import fetch_daily
    path, meta = fetch_daily(Path(args.raw_dir), date_str=args.date)
    print(f"[daily] {path} rows={meta['rows']} raw_sha={meta['raw']['sha256'][:12]}")
    return 0


def cmd_events(args) -> int:
    from pit.events import load_event_tables
    ev = load_event_tables(Path(args.event_dir))
    print(f"[events] {len(ev)} 条事件, 来源文件 {ev['source_file'].nunique() if len(ev) else 0} 个")
    if args.verbose and len(ev):
        print(ev.head(20).to_string())
    return 0


def cmd_build(args) -> int:
    from pit.materialize import BuildConfig, build_snapshots
    policy = args.policy
    type_policy = args.type_policy or ("strict" if policy == "strict" else "fallback")
    purchase_policy = args.purchase_policy or ("strict" if policy == "strict" else "unknown-keep")
    cfg = BuildConfig(
        raw_dir=Path(args.raw_dir), universe_dir=Path(args.universe_dir),
        event_dir=Path(args.event_dir), nav_dir=Path(args.nav_dir),
        start=args.start, end=args.end,
        type_policy=type_policy, purchase_policy=purchase_policy,
        markets=set(args.markets.split(",")) if args.markets else {"O"},
        slot_capital=float(args.slot_capital),
        min_nav_rows=int(args.min_nav_rows), require_history=bool(args.require_history),
        no_dedup=bool(args.no_dedup))
    summary = build_snapshots(cfg, enforce_qa=bool(args.enforce_qa))
    snap = summary["snapshots"]
    if not snap:
        print("[build] ⚠ 没有生成任何快照（检查 master / 索引 / 政策配置）。")
    for s in snap:
        print(f"[build] {s['as_of']} rows={s['rows']} {s['pit_level']}")
    lite = [s for s in snap if s["pit_level"] == "lite"]
    if lite:
        print(f"[build] ⚠ {len(lite)} 个月份为 PIT-lite（{lite[0]['as_of']} 起）。"
              "严格回测会被 PITUniverseStore 拒绝。")
    if summary.get("qa"):
        print(f"[build] QA fatal={len(summary['qa']['fatal'])} "
              f"warnings={len(summary['qa']['warnings'])}")
    return 0


def cmd_qa(args) -> int:
    from pit.materialize import _latest_master
    from pit.quality import run_qa
    _, master, _ = _latest_master(Path(args.raw_dir))
    qa = run_qa(Path(args.universe_dir), master)
    print(qa["report"])
    if qa["fatal"]:
        print("\n[qa] 致命问题:")
        for f in qa["fatal"]:
            print("  -", f)
    if qa["warnings"]:
        print("\n[qa] 提醒:")
        for w in qa["warnings"]:
            print("  -", w)
    return 1 if qa["fatal"] else 0


def cmd_report(args) -> int:
    from pit.sources_csrc import scan_raw_csrc
    from pit.sources_daily import scan_daily
    print("=== PIT 原始数据覆盖 ===")
    master_dir = Path(args.raw_dir) / "master"
    if master_dir.is_dir():
        print(f"主表版本: {[p.parent.name for p in sorted(master_dir.glob('*/fund_basic.csv'))][-5:]}")
    else:
        print("主表: ✗ 未归档")
    print(f"证监会索引: {[x['as_of'] for x in scan_raw_csrc(Path(args.raw_dir))] or '✗ 无'}")
    daily = scan_daily(Path(args.raw_dir))
    print(f"每日截面: {len(daily)} 天 ({min(daily) if daily else '-'} → {max(daily) if daily else '-'})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pit",
                                description="严格 Point-in-Time 基金池构建管线")
    p.add_argument("--raw-dir", default="data/pit_raw",
                   help="不可变原始数据归档根目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="初始化目录与清单").set_defaults(fn=cmd_init)

    m = sub.add_parser("master", help="全历史主表（Tushare 或 Wind/Choice 导入）")
    m.add_argument("--token", default=None, help="Tushare token（否则读 TUSHARE_TOKEN / pit_config.json）")
    m.add_argument("--import", dest="import_csv", default=None, help="导入供应商导出的 CSV 主表")
    m.add_argument("--source", default="vendor", help="导入时来源标识（如 wind/choice/csmar）")
    m.set_defaults(fn=cmd_master)

    c = sub.add_parser("csrc", help="证监会产品索引")
    c.add_argument("--manifest", default="data/pit_manifest/csrc_sources.csv")
    c.set_defaults(fn=cmd_csrc)

    d = sub.add_parser("daily", help="归档今日真实截面")
    d.add_argument("--date", default=None, help="覆盖日期（一般不用）")
    d.set_defaults(fn=cmd_daily)

    e = sub.add_parser("events", help="校验事件表")
    e.add_argument("--event-dir", default="data/pit_events")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(fn=cmd_events)

    b = sub.add_parser("build", help="物化月末快照")
    b.add_argument("--universe-dir", default="data/pit_universe")
    b.add_argument("--event-dir", default="data/pit_events")
    b.add_argument("--nav-dir", default="cache")
    b.add_argument("--start", default=None, help="起始月末（默认主表最早成立月）")
    b.add_argument("--end", default=None, help="结束月（默认今天）")
    b.add_argument("--policy", choices=["strict", "lite"], default="strict",
                   help="strict=索引截面+状态表/事件，无法证明则剔除；lite=允许当前口径假定")
    b.add_argument("--type-policy", choices=["strict", "fallback"], default=None)
    b.add_argument("--purchase-policy", choices=["strict", "unknown-keep"], default=None)
    b.add_argument("--markets", default="O", help="逗号分隔 O,E（默认仅场外开放式）")
    b.add_argument("--slot-capital", default=10000,
                   help="单槽资金(元)：低于此限额的“暂停大额申购”视为不可投")
    b.add_argument("--min-nav-rows", default=0, help="净值最低行数（0=仅标记不剔除）")
    b.add_argument("--require-history", action="store_true",
                   help="同时剔除 history_ok=False（模型可计算性过滤，默认不混入基金池过滤）")
    b.add_argument("--no-dedup", action="store_true", help="跳过 A/C/E 去重（仅排障）")
    b.add_argument("--enforce-qa", action="store_true", help="QA 有致命问题时拒绝落盘")
    b.set_defaults(fn=cmd_build)

    q = sub.add_parser("qa", help="质量门禁")
    q.add_argument("--universe-dir", default="data/pit_universe")
    q.set_defaults(fn=cmd_qa)

    sub.add_parser("report", help="数据覆盖报告").set_defaults(fn=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:
        print(f"[pit] ✗ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
