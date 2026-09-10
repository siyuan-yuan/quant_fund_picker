"""Print classification-related contexts from failed CSRC FA documents."""

from __future__ import annotations

import pathlib
import re

import pandas as pd
from pypdf import PdfReader


ROOT = pathlib.Path("output/v6/g0_full_pit/csrc_smoke")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")


def main() -> None:
    failures = pd.read_csv(ROOT / "classification_parse_failures.csv", dtype=str)
    failures = failures[failures["report_code"].str.startswith("FA", na=False)].copy()
    print(f"rows {len(failures)}")
    print(
        failures[
            ["share_code", "report_code", "upload_info_id", "report_name", "reason"]
        ].to_string(index=False)
    )
    print("\n--- contexts ---")
    for _, row in failures.iterrows():
        pdf_path = ROOT / "pdf" / f"{row.share_code}_{row.upload_info_id}.pdf"
        try:
            text = "".join(
                (page.extract_text() or "") + "\n"
                for page in PdfReader(str(pdf_path)).pages
            )
            text = re.sub(r"\s+", "", text)
            candidates = []
            for match in PERCENT_RE.finditer(text):
                context = text[max(0, match.start() - 180) : match.end() + 180]
                if "股票" in context and context not in candidates:
                    candidates.append(context)
        except Exception as exc:  # pragma: no cover - diagnostic helper
            print(f"ERR {pdf_path}: {exc}")
            continue
        print(
            f"### {pdf_path.name} | {row.report_code} | "
            f"chars={len(text)} | candidates={len(candidates)}"
        )
        for candidate in candidates[:5]:
            print(candidate[:320])


if __name__ == "__main__":
    main()
