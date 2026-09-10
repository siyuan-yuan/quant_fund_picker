"""Normalize historical fund metadata and build causal monthly snapshots."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from pit_universe import TARGET_TYPES


REQUIRED_COLUMNS = (
    "share_code",
    "name",
    "fund_type",
    "status",
    "inception_date",
    "end_date",
    "known_at",
    "source",
)

_OVERSEAS_WORDS = re.compile(
    r"海外|全球|港股|香港|恒生|纳斯达克|标普|日经|德国|法国|印度|越南|美国|日本|MSCI",
    re.IGNORECASE,
)
_EXCLUDED_MIXED_WORDS = re.compile(r"保本|平衡|稳健|稳定|债券")


def _base_name(name: object) -> str:
    value = re.sub(r"\s+", "", str(name or "")).strip()
    return re.sub(r"(?:A|C|E)(?:类|份额)?$", "", value, flags=re.IGNORECASE)


def _parse_yyyymmdd(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame else pd.Series(pd.NA, index=frame.index)
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce").dt.normalize()


def _dedupe_tushare_markets(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["_numeric_code"] = frame["ts_code"].astype(str).str.extract(r"(\d+)", expand=False)
    frame["_market_rank"] = frame.get("market", pd.Series("", index=frame.index)).eq("O").map({True: 0, False: 1})
    frame["_complete"] = frame.notna().sum(axis=1)
    rows = []
    for _, group in frame.sort_values(["_numeric_code", "_market_rank", "_complete"], ascending=[True, True, False]).groupby("_numeric_code", sort=False):
        chosen = group.iloc[0].copy()
        for column in raw.columns:
            if pd.isna(chosen[column]) or chosen[column] == "":
                values = group[column].dropna()
                if len(values):
                    chosen[column] = values.iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows).drop(columns=["_numeric_code", "_market_rank", "_complete"])


def from_tushare_fund_basic(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare ``fund_basic`` rows to the conservative V6 master schema.

    Generic mixed funds cannot be separated into equity-biased and bond-biased
    funds from ``fund_basic`` alone, so only explicit flexible-allocation rows are
    accepted here. The exclusion is measurable and must remain in the G0 report.
    """
    required = {"ts_code", "name", "fund_type", "status"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Tushare fund_basic missing columns: {missing}")
    frame = _dedupe_tushare_markets(raw)
    frame["inception_date"] = _parse_yyyymmdd(frame, "found_date")
    for fallback in ("issue_date", "list_date"):
        frame["inception_date"] = frame["inception_date"].fillna(_parse_yyyymmdd(frame, fallback))
    due = _parse_yyyymmdd(frame, "due_date")
    delisted = _parse_yyyymmdd(frame, "delist_date")
    frame["end_date"] = delisted.fillna(due)
    frame["end_known_at"] = frame["end_date"]

    invest = frame.get("invest_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    broad = frame["fund_type"].fillna("").astype(str)
    text = (
        frame["name"].fillna("").astype(str)
        + " "
        + frame.get("benchmark", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    is_index = broad.eq("股票型") & invest.str.contains("指数型", regex=False)
    is_overseas = text.str.contains(_OVERSEAS_WORDS)
    mapped = pd.Series(pd.NA, index=frame.index, dtype="object")
    mapped.loc[broad.eq("股票型") & ~is_index] = "股票型"
    mapped.loc[is_index & ~is_overseas] = "指数型-股票"
    mapped.loc[is_index & is_overseas] = "指数型-海外股票"
    flexible = broad.eq("混合型") & invest.eq("灵活配置型")
    mapped.loc[flexible & ~invest.str.contains(_EXCLUDED_MIXED_WORDS)] = "混合型-灵活"

    keep = mapped.notna() & frame["inception_date"].notna()
    frame = frame.loc[keep].copy()
    mapped = mapped.loc[keep]
    frame["share_code"] = frame["ts_code"].astype(str).str.extract(r"(\d+)", expand=False)
    frame["source_ts_code"] = frame["ts_code"].astype(str)
    frame["fund_type"] = mapped
    frame["status"] = frame["status"].map({"D": "清盘", "I": "发行", "L": "存续"}).fillna("未知")
    frame["known_at"] = frame["inception_date"]
    frame["source"] = "tushare_fund_basic"
    frame["family_id"] = frame.get("management", pd.Series("", index=frame.index)).fillna("").astype(str)
    columns = [*REQUIRED_COLUMNS, "family_id", "end_known_at", "source_ts_code"]
    return normalize_fund_master(frame.reindex(columns=columns))


def normalize_fund_master(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable row per share class with a deterministic representative."""
    if raw.empty:
        return pd.DataFrame(
            columns=[*REQUIRED_COLUMNS, "canonical_fund_id", "is_representative"]
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"fund master missing columns: {missing}")

    out = raw.copy()
    out["share_code"] = out["share_code"].astype(str).str.strip()
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["family_id"] = out.get("family_id", pd.Series("", index=out.index)).fillna("").astype(str)
    for column in ("inception_date", "end_date", "known_at", "end_known_at"):
        if column not in out:
            continue
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    if out["inception_date"].isna().any() or out["known_at"].isna().any():
        raise ValueError("inception_date and known_at must be valid dates")
    if out["share_code"].duplicated().any():
        raise ValueError("share_code must be unique")

    out["_base_name"] = out["name"].map(_base_name)
    out["_group_key"] = out["family_id"] + "|" + out["_base_name"]
    order = out.sort_values(["_group_key", "inception_date", "share_code"])
    representative = order.groupby("_group_key", sort=False)["share_code"].first()
    out["canonical_fund_id"] = out["_group_key"].map(representative)
    out["is_representative"] = out["share_code"].eq(out["canonical_fund_id"])
    return out.drop(columns=["_base_name", "_group_key"]).sort_values("share_code").reset_index(drop=True)


def build_monthly_snapshots(
    master: pd.DataFrame,
    end_dates: Iterable[pd.Timestamp],
    output_dir: str | Path | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build representative-share universes using information known at each date."""
    root = Path(output_dir) if output_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for value in end_dates:
        as_of = pd.Timestamp(value).normalize()
        if master.empty:
            eligible = master.copy()
        else:
            eligible = master.loc[
                master["is_representative"].astype(bool)
                & master["fund_type"].isin(TARGET_TYPES)
                & master["inception_date"].le(as_of)
                & master["known_at"].le(as_of)
                & (master["end_date"].isna() | master["end_date"].ge(as_of))
            ].copy()
        eligible["code"] = eligible.get("share_code", pd.Series(dtype=str))
        eligible["as_of"] = as_of
        columns = [
            "code", "share_code", "canonical_fund_id", "name", "fund_type",
            "status", "inception_date", "end_date", "end_known_at", "known_at", "source",
            "source_ts_code", "as_of",
        ]
        snapshot = eligible.reindex(columns=columns).sort_values("code").reset_index(drop=True)
        result[as_of] = snapshot
        if root is not None:
            snapshot.to_csv(root / f"{as_of.date()}.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V6 FULL-PIT fund master")
    parser.add_argument("--fund-basic", required=True)
    parser.add_argument("--master-out", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-03-31")
    args = parser.parse_args()

    raw = pd.read_csv(args.fund_basic, dtype=str)
    master = from_tushare_fund_basic(raw)
    end = pd.Timestamp(args.end).normalize()
    master = master.loc[master["inception_date"].le(end)].copy()
    master_path = Path(args.master_out)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    dates = pd.date_range(args.start, args.end, freq="ME")
    build_monthly_snapshots(master, dates, args.snapshot_root)
    print(
        f"raw={len(raw)} master={len(master)} canonical={master.canonical_fund_id.nunique()} "
        f"delisted={master.status.eq('清盘').sum()} snapshots={len(dates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
