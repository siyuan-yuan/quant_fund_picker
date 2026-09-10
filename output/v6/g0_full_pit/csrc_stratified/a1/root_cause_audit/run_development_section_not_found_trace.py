"""Resumable, development-only trace for INVESTMENT_SECTION_NOT_FOUND.

This diagnostic must never read untouched-validation failures or write adjudication artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "v6" / "__init__.py").is_file()
    and (parent / "v6" / "g0_checkpoint_gate.py").is_file()
)
sys.path.insert(0, str(PROJECT_ROOT))

from v6.csrc_fund_disclosure import extract_pdf_text
from v6.g0_checkpoint_gate import _IsolatedWorker, _read_jsonl


HEADING_TERMS = ("投资范围", "投资目标", "投资策略", "资产配置", "投资限制", "投资组合")


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _source(record: dict[str, object]) -> str:
    code = _text(record.get("fundCode")).split(".")[0].zfill(6)
    upload = _text(record.get("uploadInfoId"))
    return f"{code}_{upload}.pdf" if code and upload else ""


def _title_pattern(title: str) -> str:
    rules = (
        ("TRANSFORMATION_OR_EXPIRY", ("转型", "保本期到期", "保本周期到期", "转入下一保本期")),
        ("PROSPECTUS_OR_SUMMARY", ("招募说明书", "基金产品资料概要")),
        ("CONTRACT_OR_AMENDMENT", ("基金合同", "合同修改", "修改基金合同")),
        ("PERIODIC_REPORT", ("季度报告", "半年度报告", "年度报告")),
        ("SUBSCRIPTION_OR_LAUNCH", ("发售", "认购", "募集", "成立", "生效")),
        ("SALES_OR_CHANNEL", ("代销", "销售渠道", "销售机构", "费率优惠", "定期定额")),
        ("SUBSCRIPTION_REDEMPTION_OPERATIONS", ("申购", "赎回", "转换转入", "转换转出")),
        ("DISTRIBUTION", ("分红", "收益分配")),
    )
    for label, needles in rules:
        if any(needle in title for needle in needles):
            return label
    return "OTHER_OPERATIONAL_NOTICE"


def _pdf_path(root: Path, source: str) -> Path:
    direct = root / source
    if direct.is_file():
        return direct
    nested = root / source.split("_", 1)[0] / source
    return nested


def _scan_text(text: str) -> dict[str, object]:
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines()]
    hits: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        terms = [term for term in HEADING_TERMS if term in line]
        if not terms:
            continue
        hits.append({"line": index + 1, "terms": terms, "text": line[:180]})
        if len(hits) >= 20:
            break
    compact = re.sub(r"\s+", "", text)
    return {
        "text_chars": len(text),
        "heading_term_hit_count": len(hits),
        "heading_term_hits": hits,
        "has_equity_term": "股票" in compact or "权益类" in compact,
        "has_percent": "%" in compact or "％" in compact,
    }


def run(args: argparse.Namespace) -> None:
    census = pd.read_csv(args.census, dtype=str, keep_default_na=False)
    target = census.loc[
        census["root_cause_reason"].eq("INVESTMENT_SECTION_NOT_FOUND")
        & census["split_role"].eq("development")
    ].copy()
    if target.empty or set(target["split_role"]) != {"development"}:
        raise RuntimeError("development-only target invariant failed")
    metadata = _read_jsonl(args.metadata)
    records: dict[str, dict[str, object]] = {}
    for record in metadata.to_dict("records"):
        source = _source(record)
        if source and source not in records:
            records[source] = record

    inventory_rows = []
    for row in target.to_dict("records"):
        meta = records.get(_text(row.get("source_doc")), {})
        title = _text(meta.get("reportName"))
        inventory_rows.append({
            **row,
            "report_code": _text(meta.get("reportCode")),
            "report_name": title,
            "title_pattern": _title_pattern(title),
            "report_send_date": _text(meta.get("reportSendDate")),
        })
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["trigger", "family_key", "title_pattern", "source_doc"], kind="mergesort"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output / "development_section_not_found_inventory.csv", index=False, encoding="utf-8")

    cache_path = args.output / "development_heading_scan_cache.jsonl"
    cached: dict[str, dict[str, object]] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cached[_text(item.get("source_doc"))] = item

    sources = sorted(set(inventory["source_doc"]))
    worker = _IsolatedWorker(extract_pdf_text)
    try:
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, source in enumerate(sources, start=1):
                if source in cached:
                    continue
                path = _pdf_path(args.pdf_root, source)
                result: dict[str, object] = {"source_doc": source, "pdf_path": str(path)}
                try:
                    raw = worker.call((path,), timeout_seconds=args.timeout_seconds)
                    result.update({"scan_status": "success", **_scan_text(str(raw))})
                except Exception as exc:
                    result.update({"scan_status": "failure", "scan_error": f"{type(exc).__name__}: {exc}"})
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                cached[source] = result
                print(f"[development-heading-scan] {index}/{len(sources)} {source} {result['scan_status']}", flush=True)
    finally:
        worker.close()

    scan = pd.DataFrame(cached.values())
    merged = inventory.merge(scan, on="source_doc", how="left", validate="many_to_one")
    if set(merged["split_role"]) != {"development"}:
        raise RuntimeError("untouched-validation leakage detected")
    merged.to_csv(args.output / "development_section_not_found_trace.csv", index=False, encoding="utf-8")
    summary = (
        merged.groupby(["trigger", "report_code", "title_pattern", "scan_status"], dropna=False)
        .size().rename("checkpoint_count").reset_index()
        .sort_values(["trigger", "checkpoint_count", "report_code", "title_pattern"], ascending=[True, False, True, True], kind="mergesort")
    )
    summary.to_csv(args.output / "development_section_not_found_summary.csv", index=False, encoding="utf-8")
    manifest = {
        "scope": "development-only INVESTMENT_SECTION_NOT_FOUND",
        "checkpoint_rows": len(inventory),
        "unique_sources": len(sources),
        "unique_families": inventory["family_key"].nunique(),
        "inventory_sha256": hashlib.sha256((args.output / "development_section_not_found_inventory.csv").read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256((args.output / "development_section_not_found_trace.csv").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((args.output / "development_section_not_found_summary.csv").read_bytes()).hexdigest(),
    }
    (args.output / "development_section_not_found_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
