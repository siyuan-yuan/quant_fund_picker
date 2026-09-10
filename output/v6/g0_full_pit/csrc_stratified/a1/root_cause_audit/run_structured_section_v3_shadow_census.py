"""Read-only A1 shadow census: never adjudicates or writes Gate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v6.g0_checkpoint_gate import (
    _authoritative_alias_family_map,
    _extract_denominator_sections,
    _read_jsonl,
)


def _date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _source(record: dict[str, object]) -> str:
    code = _text(record.get("fundCode") or record.get("share_code")).split(".")[0].zfill(6)
    upload = _text(record.get("uploadInfoId") or record.get("upload_info_id"))
    return f"{code}_{upload}.pdf" if code and upload else ""


def _pivot(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(columns, dropna=False)
        .size()
        .rename("checkpoint_count")
        .reset_index()
        .sort_values([*columns, "checkpoint_count"], kind="mergesort")
    )


def run(args: argparse.Namespace) -> None:
    checkpoints = pd.read_csv(args.checkpoints, dtype=str, keep_default_na=False)
    sample = pd.read_csv(args.sample, dtype=str, keep_default_na=False)
    aliases = pd.read_csv(args.aliases, dtype=str, keep_default_na=False)
    split = pd.read_csv(args.split, dtype=str, keep_default_na=False)
    metadata = _read_jsonl(args.metadata)
    family_by_code = _authoritative_alias_family_map(aliases, sample)

    meta = metadata.copy()
    meta["_source_document"] = [_source(row) for row in meta.to_dict("records")]
    meta["_share_code"] = [item.split("_", 1)[0] for item in meta["_source_document"]]
    meta["_family_key"] = meta["_share_code"].map(family_by_code).fillna("")
    sampled_families = set(sample["family_key"])
    ordinary_sources = set(
        checkpoints.loc[
            ~checkpoints["trigger_type"].isin(["INCEPTION", "TERMINAL_STATE"]),
            "source_document",
        ]
    )
    relevant = meta.loc[
        meta["_source_document"].isin(ordinary_sources)
        | meta["_family_key"].isin(sampled_families)
    ].drop(columns=["_source_document", "_share_code", "_family_key"])

    sections = _extract_denominator_sections(
        relevant,
        args.pdf_root,
        cache_path=args.cache,
        timeout_seconds=args.timeout_seconds,
    )
    section_records = sections.to_dict("records")
    by_upload: dict[str, list[dict[str, object]]] = {}
    by_family: dict[str, list[dict[str, object]]] = {}
    seen_upload_source: set[tuple[str, str]] = set()
    seen_family_source: set[tuple[str, str]] = set()
    for row in section_records:
        upload = _text(row.get("upload_info_id"))
        code = _text(row.get("share_code")).split(".")[0].zfill(6)
        source_document = _text(row.get("source_document"))
        upload_source = (upload, source_document)
        if upload and source_document and upload_source not in seen_upload_source:
            by_upload.setdefault(upload, []).append(row)
            seen_upload_source.add(upload_source)
        family = family_by_code.get(code, "")
        family_source = (family, source_document)
        if family and source_document and family_source not in seen_family_source:
            enriched = dict(row)
            enriched["_family_key"] = family
            by_family.setdefault(family, []).append(enriched)
            seen_family_source.add(family_source)

    split_role = split.set_index("family_key")["split_role"].to_dict()
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints.to_dict("records"):
        trigger = _text(checkpoint.get("trigger_type"))
        family = _text(checkpoint.get("family_key"))
        chosen: dict[str, object] | None = None
        diagnostic = ""
        candidate_count = 0
        causal_candidate_count = 0
        if trigger in {"INCEPTION", "TERMINAL_STATE"}:
            candidates = by_family.get(family, [])
            candidate_count = len(candidates)
            checkpoint_date = _date(checkpoint.get("known_at"))
            if checkpoint_date is None:
                diagnostic = "DATE_OR_STAGE_UNDETERMINED"
            else:
                causal = [
                    item for item in candidates
                    if _date(item.get("known_at")) is not None
                    and _date(item.get("known_at")) <= checkpoint_date
                ]
                causal_candidate_count = len(causal)
                if not causal:
                    diagnostic = (
                        "OFFICIAL_DOCUMENT_NOT_FOUND"
                        if not candidates else "NO_CAUSAL_SOURCE_DOCUMENT"
                    )
                else:
                    successful = [item for item in causal if item.get("extraction_status") == "success"]
                    pool = successful or causal
                    chosen = max(
                        pool,
                        key=lambda item: (
                            _date(item.get("known_at")) or pd.Timestamp.min,
                            _text(item.get("source_document")),
                        ),
                    )
        else:
            exact = [
                item for item in by_upload.get(_text(checkpoint.get("upload_info_id")), [])
                if _text(item.get("source_document")) == _text(checkpoint.get("source_document"))
            ]
            candidate_count = len(exact)
            if len(exact) == 1:
                chosen = exact[0]
                causal_candidate_count = 1
            elif not exact:
                diagnostic = "OFFICIAL_DOCUMENT_NOT_FOUND"
            else:
                diagnostic = "SOURCE_DOCUMENT_AMBIGUOUS"

        if chosen is not None:
            diagnostic = _text(chosen.get("root_cause_reason"))
            clause_status = (
                "PARSED" if chosen.get("extraction_status") == "success" else "NOT_PARSED"
            )
        else:
            clause_status = "NOT_ATTEMPTED"
        root_cause = diagnostic or "PARSED_CLAUSE"
        rows.append(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "family_key": family,
                "stratum_status": split.loc[split["family_key"].eq(family), "status"].iloc[0],
                "stratum_era": split.loc[split["family_key"].eq(family), "inception_era"].iloc[0],
                "split_role": split_role.get(family, ""),
                "trigger": trigger,
                "source_doc": _text(chosen.get("source_document")) if chosen else _text(checkpoint.get("source_document")),
                "document_applicability": _text(chosen.get("document_applicability")) if chosen else "",
                "section_heading": _text(chosen.get("section_heading")) if chosen else "",
                "section_page_start": _text(chosen.get("section_page_start")) if chosen else "",
                "section_locator": _text(chosen.get("section_locator")) if chosen else "",
                "root_cause_reason": root_cause,
                "clause_parse_status": clause_status,
                "candidate_document_count": candidate_count,
                "causal_candidate_count": causal_candidate_count,
            }
        )

    census = pd.DataFrame(rows)
    if len(census) != 1371 or census["checkpoint_id"].nunique() != 1371:
        raise RuntimeError("shadow census must preserve exactly 1,371 unique checkpoint IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    census.to_csv(args.output / "shadow_checkpoint_census.csv", index=False, encoding="utf-8")
    _pivot(census, ["trigger", "root_cause_reason"]).to_csv(
        args.output / "root_cause_by_trigger.csv", index=False, encoding="utf-8"
    )
    _pivot(census, ["stratum_status", "stratum_era", "root_cause_reason"]).to_csv(
        args.output / "root_cause_by_stratum.csv", index=False, encoding="utf-8"
    )
    _pivot(census, ["family_key", "root_cause_reason"]).to_csv(
        args.output / "root_cause_by_family.csv", index=False, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
