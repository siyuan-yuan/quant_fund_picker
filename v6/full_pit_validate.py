"""G0 validator for FULL-PIT metadata, snapshots, and delisted NAV coverage."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    delisted_count: int
    delisted_nav_count: int
    delisted_nav_coverage: float
    causality_violations: int
    missing_months: tuple[str, ...]


def _snapshot_date(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(path.stem).normalize()
    except (TypeError, ValueError):
        return None


def validate_full_pit(
    master: pd.DataFrame,
    nav_root: str | Path,
    snapshot_root: str | Path,
    expected_dates: pd.DatetimeIndex,
) -> ValidationReport:
    nav_root = Path(nav_root)
    snapshot_root = Path(snapshot_root)
    if master.empty:
        delisted = master
    else:
        status = master.get("status", pd.Series("", index=master.index)).fillna("").astype(str)
        delisted = master.loc[status.isin({"D", "清盘", "终止", "摘牌"})]
    codes = (
        delisted.get("share_code", pd.Series(dtype=object))
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    nav_count = sum((nav_root / f"nav_{code}.csv").is_file() for code in codes)
    delisted_count = len(delisted)
    coverage = nav_count / delisted_count if delisted_count else 0.0

    paths = {date: path for path in snapshot_root.glob("*.csv") if (date := _snapshot_date(path)) is not None}
    expected = tuple(pd.Timestamp(date).normalize() for date in expected_dates)
    missing = tuple(str(date.date()) for date in expected if date not in paths)
    violations = 0
    for as_of, path in paths.items():
        frame = pd.read_csv(path)
        for column in ("known_at", "inception_date"):
            if column in frame:
                dates = pd.to_datetime(frame[column], errors="coerce")
                violations += int((dates.notna() & dates.gt(as_of)).sum())
        if "canonical_fund_id" in frame:
            violations += int(frame["canonical_fund_id"].dropna().duplicated().sum())

    passed = coverage >= 0.90 and violations == 0 and not missing
    return ValidationReport(
        passed=passed,
        delisted_count=delisted_count,
        delisted_nav_count=nav_count,
        delisted_nav_coverage=float(coverage),
        causality_violations=violations,
        missing_months=missing,
    )


def _write_report(report: ValidationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(report)]).to_csv(output_dir / "coverage.csv", index=False)
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# G0 FULL-PIT Validation",
        "",
        f"- Status: **{status}**",
        f"- Delisted NAV coverage: {report.delisted_nav_count}/{report.delisted_count} ({report.delisted_nav_coverage:.2%})",
        f"- Causality violations: {report.causality_violations}",
        f"- Missing months: {', '.join(report.missing_months) if report.missing_months else 'none'}",
    ]
    (output_dir / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the V6 FULL-PIT data gate")
    parser.add_argument("--master", required=True)
    parser.add_argument("--nav-root", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-03-31")
    args = parser.parse_args()
    master_path = Path(args.master)
    master = (
        pd.read_csv(
            master_path,
            dtype={"share_code": str, "canonical_fund_id": str, "source_ts_code": str},
        )
        if master_path.is_file()
        else pd.DataFrame()
    )
    if not master.empty:
        for column in ("inception_date", "end_date", "known_at"):
            if column in master:
                master[column] = pd.to_datetime(master[column], errors="coerce")
    expected = pd.date_range(args.start, args.end, freq="ME")
    report = validate_full_pit(master, args.nav_root, args.snapshot_root, expected)
    _write_report(report, Path(args.out))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
