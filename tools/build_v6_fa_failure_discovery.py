"""Rebuild phrase contexts from the *current* failed CSRC FA PDF set."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PERCENT_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*[%％])")
SIGNAL_RE = re.compile(r"股票|权益类|固定收益类|基金资产|投资组合比例")


def candidate_contexts(text: str, radius: int = 180, limit: int = 30) -> list[str]:
    normalized = re.sub(r"\s+", "", text)
    contexts: list[str] = []
    seen: set[str] = set()
    for match in PERCENT_RE.finditer(normalized):
        context = normalized[max(0, match.start() - radius) : match.end() + radius]
        if not SIGNAL_RE.search(context):
            continue
        fingerprint = context[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        contexts.append(context)
        if len(contexts) >= limit:
            break
    return contexts


def inspect_document(root: Path, values: dict) -> list[dict]:
    pdf_path = root / "pdf" / f"{values['share_code']}_{values['upload_info_id']}.pdf"
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        contexts = candidate_contexts(text) or [""]
        status = "ok" if text.strip() else "empty_text"
        error = ""
    except Exception as exc:
        text, contexts, status, error = "", [""], "pdf_error", str(exc)
    return [
        {
            **values,
            "extract_status": status,
            "text_chars": len(text),
            "context_rank": rank,
            "context": context,
            "error": error,
        }
        for rank, context in enumerate(contexts, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/v6/g0_full_pit/csrc_stratified"),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    failures = pd.read_csv(args.root / "classification_parse_failures.csv", dtype=str)
    failures = failures[
        failures["report_code"].str.startswith("FA", na=False)
        & failures["reason"].eq("No explicit equity allocation constraint found")
    ].drop_duplicates(["share_code", "upload_info_id"])
    values = failures[
        ["share_code", "upload_info_id", "report_code", "report_name"]
    ].to_dict("records")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda item: inspect_document(args.root, item), values)
        for number, document_rows in enumerate(results, start=1):
            rows.extend(document_rows)
            if number % 50 == 0:
                print(f"[progress] {number}/{len(failures)}", flush=True)
    output = args.root / "fa_failure_phrase_discovery_v8.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[ok] documents={len(failures)}, rows={len(rows)}, output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
