"""Convert aggregate Tushare fund NAV data into auditable per-fund caches."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NavConversionReport:
    input_rows: int
    input_codes: int
    output_rows: int
    output_codes: int
    duplicate_rows: int
    conflicting_duplicate_dates: int


def convert_fund_nav(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert one fund's rows, preserving both economic and availability dates."""
    required = {"ann_date", "nav_date", "unit_nav", "adj_nav"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"fund_nav missing columns: {missing}")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["nav_date"], format="%Y%m%d", errors="coerce")
    frame["known_at"] = pd.to_datetime(frame["ann_date"], format="%Y%m%d", errors="coerce")
    frame["nav"] = pd.to_numeric(frame["unit_nav"], errors="coerce")
    frame["adj_nav"] = pd.to_numeric(frame["adj_nav"], errors="coerce")
    frame = frame.dropna(subset=["date", "known_at", "nav", "adj_nav"])
    frame = frame.loc[frame["known_at"].ge(frame["date"]) & frame["adj_nav"].gt(0)]
    frame = frame.sort_values(["date", "known_at"]).drop_duplicates("date", keep="first")
    frame["ret"] = frame["adj_nav"].pct_change()
    return frame[["date", "known_at", "nav", "adj_nav", "ret"]].reset_index(drop=True)


def convert_nav_file(source: str | Path, output_dir: str | Path) -> NavConversionReport:
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, dtype={"ts_code": str, "ann_date": str, "nav_date": str})
    if "ts_code" not in raw:
        raise ValueError("fund_nav missing columns: ['ts_code']")
    duplicate_mask = raw.duplicated(["ts_code", "nav_date"], keep=False)
    conflicts = 0
    if duplicate_mask.any():
        duplicate_groups = raw.loc[duplicate_mask].groupby(["ts_code", "nav_date"])
        conflicts = int(
            sum(
                group[["unit_nav", "adj_nav"]].astype(str).drop_duplicates().shape[0] > 1
                for _, group in duplicate_groups
            )
        )
    output_rows = 0
    output_codes = 0
    for ts_code, group in raw.groupby("ts_code", sort=True):
        converted = convert_fund_nav(group)
        if converted.empty:
            continue
        code = str(ts_code).split(".", 1)[0]
        temp = output_dir / f"nav_{code}.csv.tmp"
        target = output_dir / f"nav_{code}.csv"
        converted.to_csv(temp, index=False, encoding="utf-8-sig")
        temp.replace(target)
        output_rows += len(converted)
        output_codes += 1
    return NavConversionReport(
        input_rows=len(raw),
        input_codes=int(raw["ts_code"].nunique()),
        output_rows=output_rows,
        output_codes=output_codes,
        duplicate_rows=int(raw.duplicated(["ts_code", "nav_date"]).sum()),
        conflicting_duplicate_dates=conflicts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Tushare fund_nav to per-fund caches")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = convert_nav_file(args.source, args.out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(report)]).to_csv(report_path, index=False)
    print(asdict(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
