"""Print a compact progress snapshot for the resumable Tushare collector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="fund_nav")
    args = parser.parse_args()
    checkpoint = ROOT / "cache" / "tushare" / "bulk" / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    completed = len(state.get("completed", {}).get(args.dataset, []))
    failures = len(state.get("failures", {}).get(args.dataset, {}))
    total_text = "unknown"
    if args.dataset in {"fund_nav", "fund_div", "fund_portfolio"}:
        basic = pd.read_csv(
            ROOT / "cache" / "tushare" / "fund_basic.csv",
            usecols=["ts_code"],
            dtype=str,
        )
        total = basic["ts_code"].nunique()
        total_text = str(total)
        remaining = max(total - completed, 0)
        progress = completed / total if total else 0.0
        print(
            f"{args.dataset}: {completed:,}/{total:,} ({progress:.2%}), "
            f"remaining={remaining:,}, current_failures={failures:,}"
        )
    else:
        print(
            f"{args.dataset}: completed_jobs={completed:,}, "
            f"current_failures={failures:,}, total={total_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
