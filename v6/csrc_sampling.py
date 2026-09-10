"""Deterministic stratified sample for validating CSRC mixed-fund disclosures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ERA_LABELS = ("pre2013", "2013_2019", "2020_2026")
STATUS_LABELS = ("存续", "清盘")


def _base_name(value: object) -> str:
    name = re.sub(r"\s+", "", str(value or "")).strip()
    return re.sub(
        r"[-—_－]?(?:(?:A|B|C|D|E|H|I|Y)(?:类|份额)?|优先|进取)$",
        "",
        name,
        flags=re.IGNORECASE,
    )


def expand_sample_family_aliases(raw: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Return every Tushare share code belonging to each frozen sample family."""
    required_raw = {"ts_code", "name", "management"}
    required_sample = {"share_code", "name", "management", "status", "inception_era"}
    missing_raw = sorted(required_raw.difference(raw.columns))
    missing_sample = sorted(required_sample.difference(sample.columns))
    if missing_raw or missing_sample:
        raise ValueError(
            f"missing alias columns: raw={missing_raw}, sample={missing_sample}"
        )

    frozen = sample.copy()
    frozen["normalized_base_name"] = frozen["name"].map(_base_name)
    frozen["alias_family_key"] = (
        frozen["management"].fillna("").astype(str)
        + "|"
        + frozen["normalized_base_name"]
    )
    if frozen["alias_family_key"].duplicated().any():
        raise ValueError("frozen sample contains duplicate normalized families")

    candidates = raw.copy()
    candidates["query_code"] = candidates["ts_code"].astype(str).str.extract(
        r"(\d{6})", expand=False
    )
    candidates["normalized_base_name"] = candidates["name"].map(_base_name)
    candidates["alias_family_key"] = (
        candidates["management"].fillna("").astype(str)
        + "|"
        + candidates["normalized_base_name"]
    )
    candidates["_market_rank"] = candidates.get(
        "market", pd.Series("", index=candidates.index)
    ).eq("O").map({True: 0, False: 1})
    candidates = candidates.loc[
        candidates["alias_family_key"].isin(frozen["alias_family_key"])
    ]
    candidates = (
        candidates.loc[candidates["query_code"].notna()]
        .sort_values(["query_code", "_market_rank", "ts_code"])
        .drop_duplicates("query_code")
    )

    aliases = frozen[
        ["share_code", "status", "inception_era", "alias_family_key"]
    ].merge(
        candidates[["query_code", "ts_code", "name", "alias_family_key"]],
        on="alias_family_key",
        how="left",
        validate="one_to_many",
    )
    aliases = aliases.rename(
        columns={
            "share_code": "sample_share_code",
            "ts_code": "query_ts_code",
            "name": "query_name",
            "alias_family_key": "family_key",
        }
    )
    if aliases["query_code"].isna().any():
        raise ValueError("at least one frozen family has no query-code alias")
    return aliases.sort_values(
        ["sample_share_code", "query_code"]
    ).reset_index(drop=True)


def build_stratified_mixed_sample(
    raw: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp = "2026-03-31",
    per_stratum: int = 10,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Sample one share per family across status and inception-era strata."""
    required = {"ts_code", "name", "management", "fund_type", "found_date", "status"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"fund_basic missing columns: {missing}")
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")

    frame = raw.loc[raw["fund_type"].eq("混合型")].copy()
    frame["share_code"] = frame["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["inception_date"] = pd.to_datetime(
        frame["found_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    frame["status"] = frame["status"].map({"L": "存续", "D": "清盘"})
    frame = frame.loc[
        frame["share_code"].notna()
        & frame["inception_date"].notna()
        & frame["inception_date"].le(pd.Timestamp(cutoff))
        & frame["status"].isin(STATUS_LABELS)
    ].copy()
    frame["inception_era"] = pd.cut(
        frame["inception_date"],
        bins=[pd.Timestamp.min, pd.Timestamp("2012-12-31"), pd.Timestamp("2019-12-31"), pd.Timestamp(cutoff)],
        labels=ERA_LABELS,
    ).astype("object")
    frame["base_name"] = frame["name"].map(_base_name)
    frame["family_key"] = frame["management"].fillna("").astype(str) + "|" + frame["base_name"]
    frame = frame.sort_values(["family_key", "inception_date", "share_code"])
    frame = frame.drop_duplicates("family_key", keep="first")

    rng = np.random.default_rng(seed)
    sampled = []
    for status in STATUS_LABELS:
        for era in ERA_LABELS:
            group = frame.loc[
                frame["status"].eq(status) & frame["inception_era"].eq(era)
            ].sort_values("share_code")
            take = min(per_stratum, len(group))
            if take:
                positions = np.sort(rng.choice(len(group), size=take, replace=False))
                chosen = group.iloc[positions].copy()
                chosen["stratum_pool_size"] = len(group)
                chosen["stratum_quota"] = per_stratum
                sampled.append(chosen)
    columns = [
        "share_code", "ts_code", "name", "management", "fund_type", "status",
        "inception_date", "inception_era", "family_key", "stratum_pool_size",
        "stratum_quota",
    ]
    if not sampled:
        return pd.DataFrame(columns=columns)
    return pd.concat(sampled, ignore_index=True).reindex(columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic CSRC G0 sample")
    parser.add_argument("--fund-basic", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--codes-out", required=True)
    parser.add_argument("--cutoff", default="2026-03-31")
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--aliases-out")
    parser.add_argument("--alias-codes-out")
    args = parser.parse_args()

    raw = pd.read_csv(args.fund_basic, dtype=str)
    sample = build_stratified_mixed_sample(
        raw, cutoff=args.cutoff, per_stratum=args.per_stratum, seed=args.seed
    )
    audit_path = Path(args.audit_out)
    codes_path = Path(args.codes_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    codes_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(audit_path, index=False, encoding="utf-8-sig")
    codes_path.write_text("\n".join(sample["share_code"]) + "\n", encoding="utf-8")
    if args.aliases_out or args.alias_codes_out:
        if not (args.aliases_out and args.alias_codes_out):
            parser.error("--aliases-out and --alias-codes-out must be provided together")
        aliases = expand_sample_family_aliases(raw, sample)
        aliases_path = Path(args.aliases_out)
        alias_codes_path = Path(args.alias_codes_out)
        aliases_path.parent.mkdir(parents=True, exist_ok=True)
        alias_codes_path.parent.mkdir(parents=True, exist_ok=True)
        aliases.to_csv(aliases_path, index=False, encoding="utf-8-sig")
        alias_codes_path.write_text(
            "\n".join(aliases["query_code"].drop_duplicates()) + "\n",
            encoding="utf-8",
        )
        print(
            f"alias_rows={len(aliases)} alias_codes={aliases.query_code.nunique()} "
            f"families={aliases.family_key.nunique()}"
        )
    counts = sample.groupby(["status", "inception_era"], observed=True).size()
    print(f"sample={len(sample)} seed={args.seed} cutoff={args.cutoff}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
