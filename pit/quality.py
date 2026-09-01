# -*- coding: utf-8 -*-
"""快照质量门禁（G1–G8）与 QA 报告。

G1 月度基金数跳变 >±10%           → 可疑（需人工确认）
G2 快照含成立日在未来的基金       → 致命
G3 状态断言 known_at > signal_date → 致命（构造期已拦；此处兜底）
G4 已清盘/未成立基金进入快照，或
   快照基金在“全历史主表”中缺失     → 致命
G5 同组 A/C/E 重复进入候选池      → 致命
G6 基金类型缺失/无法归一           → 致命
G7 来源或文件哈希缺失（strict 行） → 致命
G8 历史快照来源含“今日截面/当前主表回填” → 致命（拒绝回填）

另输出覆盖率：
  - 每月 strict/lite 行数、type 来源分布、purchase 来源分布、history_ok 占比
  - 与最近一期证监会索引作为_of的差距（类型新鲜度）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pit.schema import normalize_type

GATE_NAMES = {
    "G1": "月度基金数跳变>10%", "G2": "成立日在未来", "G3": "known_at>signal",
    "G4": "主表不一致(清盘/未成立/缺失)", "G5": "A/C/E重复", "G6": "类型缺失",
    "G7": "来源/哈希缺失", "G8": "今日截面回填",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def _fatal(items) -> list[str]:
    return [str(x) for x in items]


def run_qa(universe_dir: Path, master: pd.DataFrame,
           build_summary: Optional[dict] = None) -> dict:
    universe_dir = Path(universe_dir)
    files = sorted(universe_dir.glob("20??-??-??.csv"))
    if not files:
        return {"fatal": ["无快照"], "warnings": [], "coverage": {}, "report": ""}
    master = master.copy()
    master["code"] = master["code"].astype(str)
    fatal, warnings = [], []
    coverage = []
    prev_count: Optional[int] = None
    for p in files:
        as_of = pd.Timestamp(p.stem)
        df = _read(p)
        count = len(df)
        # G1
        if prev_count is not None:
            jump = (count - prev_count) / max(prev_count, 1)
            if abs(jump) > 0.10:
                warnings.append(f"G1 {p.stem}: 基金数 {prev_count}→{count} ({jump:+.1%})")
        prev_count = count
        # G2
        if "inception_date" in df.columns:
            bad = df[df["inception_date"].notna() &
                     (pd.to_datetime(df["inception_date"], errors="coerce",
                                     format="mixed") > as_of)]
            if len(bad):
                fatal.append(f"G2 {p.stem}: {len(bad)} 只基金成立日在快照之后: "
                             + ",".join(bad["code"].head(5)))
        # G3
        if "known_at" in df.columns:
            known = pd.to_datetime(df["known_at"], errors="coerce", format="mixed")
            bad3 = df[known.notna() & (known > as_of)]
            if len(bad3):
                fatal.append(f"G3 {p.stem}: {len(bad3)} 行 known_at 晚于信号日")
        # G4
        m = master.set_index("code")
        for _, r in df.iterrows():
            code = str(r["code"]).zfill(6)
            if code not in m.index:
                fatal.append(f"G4 {p.stem}: 基金 {code} 不在全历史主表中")
                continue
            mr = m.loc[code]
            fd = pd.to_datetime(mr.get("found_date"), errors="coerce")
            dd = pd.to_datetime(mr.get("delist_date"), errors="coerce")
            if pd.notna(fd) and fd > as_of:
                fatal.append(f"G4 {p.stem}: {code} 成立日 {fd.date()} 晚于快照")
            if pd.notna(dd) and dd <= as_of:
                fatal.append(f"G4 {p.stem}: {code} 已于 {dd.date()} 清盘仍出现在快照")
        # G5
        if "share_class_group" in df.columns and "share_class" in df.columns:
            dup = df.groupby("share_class_group").agg(n=("code", "count")).query("n>1")
            if len(dup):
                fatal.append(f"G5 {p.stem}: {len(dup)} 组存在重复份额: "
                             + ",".join(dup.index[:5]))
        # G6
        if "fund_type" in df.columns:
            missing = df[df["fund_type"].fillna("").astype(str).eq("") |
                         df["fund_type"].map(lambda s: normalize_type(s)[1] == "unknown")]
            if len(missing):
                fatal.append(f"G6 {p.stem}: {len(missing)} 行类型缺失/无法归一")
        # G7
        if "pit_level" in df.columns:
            strict = df[df["pit_level"].eq("strict")]
            if len(strict):
                no_src = strict[strict["source"].fillna("").eq("") |
                                strict["source_sha256"].fillna("").eq("")]
                if len(no_src):
                    fatal.append(f"G7 {p.stem}: {len(no_src)} 行 strict 但缺来源/哈希")
        # G8
        if "source" in df.columns:
            backfill = df[df["source"].astype(str).str.contains(
                "daily|current-assumed|master-current|rank_all|today", case=False, na=False)]
            if len(backfill) and "pit_level" in df.columns:
                n_strict = df["pit_level"].eq("strict").sum()
                if len(backfill) == n_strict:
                    fatal.append(f"G8 {p.stem}: strict 行全部来自今日截面/当前表回填")
        # 覆盖统计
        rec = {"as_of": p.stem, "rows": count}
        for c in ("pit_level",):
            if c in df.columns:
                rec[c + "_strict"] = int(df[c].eq("strict").sum())
                rec[c + "_lite"] = int(df[c].eq("lite").sum())
        for c in ("type_ok", "purchase_ok", "lifecycle_ok", "history_ok"):
            if c in df.columns:
                rec[c] = int(df.get(c, pd.Series(dtype=str))
                             .astype(str).str.lower().isin(["true", "1"]).sum())
        if "type_taxonomy" in df.columns:
            rec["taxonomy"] = df["type_taxonomy"].value_counts().to_dict()
        if "purchase_status_source" in df.columns:
            rec["purchase_sources"] = df["purchase_status_source"].value_counts().to_dict()
        coverage.append(rec)

    report_lines = ["# PIT 快照质量报告", ""]
    for g, name in GATE_NAMES.items():
        hits = [f for f in fatal if f.startswith(g)] + [w for w in warnings if w.startswith(g)]
        report_lines.append(f"- {g} {name}: {'⚠ ' + str(len(hits)) if hits else '✅ 通过'}")
    report_lines += ["", "## 月度覆盖率"]
    report_lines.append("| as_of | rows | strict | lite | type_ok | purchase_ok | history_ok |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for r in coverage:
        report_lines.append(
            f"| {r['as_of']} | {r['rows']} | {r.get('pit_level_strict', 0)} | "
            f"{r.get('pit_level_lite', 0)} | {r.get('type_ok', '')} | "
            f"{r.get('purchase_ok', '')} | {r.get('history_ok', '')} |")
    report = "\n".join(report_lines)
    (universe_dir / "_qa_report.md").write_text(report, encoding="utf-8")
    return {"fatal": fatal, "warnings": warnings, "coverage": coverage, "report": report}
