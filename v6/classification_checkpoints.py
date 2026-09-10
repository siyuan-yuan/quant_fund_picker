"""Stable domain primitives for investment-clause classification checkpoints."""

from __future__ import annotations

import hashlib
import os
import re
import time
import unicodedata
from enum import Enum
from pathlib import Path

import pandas as pd

from v6.csrc_fund_disclosure import extract_equity_constraint


NORMALIZATION_VERSION: int = 1
GENERATION_RULE_VERSION: int = 2

_CHECKPOINT_COLUMNS = (
    "checkpoint_id",
    "family_key",
    "share_code",
    "upload_info_id",
    "known_at",
    "effective_date",
    "trigger_type",
    "is_critical",
    "source_document",
    "generation_rule_version",
    "normalization_version",
    "manifest_revision",
    "supersedes_checkpoint_id",
    "revision_reason",
)
_TERMINAL_STATES = frozenset({"清盘", "终止", "终止上市", "terminated", "terminal"})


class Disposition(str, Enum):
    """Disposition assigned to a clause at a classification checkpoint."""

    PARSED_STATE = "PARSED_STATE"
    VERIFIED_NOOP = "VERIFIED_NOOP"
    UNRESOLVED = "UNRESOLVED"


_PAGE_MARKER_RE = re.compile(r"第\s*\d+\s*页")
_DASH_VARIANTS = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "﹘": "-",
        "﹣": "-",
        "－": "-",
        "−": "-",
    }
)


def normalize_investment_clause(text: str) -> str:
    """Return a stable, whitespace-free representation of an extracted clause.

    Unicode compatibility characters are folded first, then page markers and
    presentation-only separators are removed. Semantic content, including
    numbers, negation, comparators, denominators, and scope, is retained.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _PAGE_MARKER_RE.sub("", normalized)
    normalized = normalized.translate(_DASH_VARIANTS)
    normalized = normalized.replace("％", "%")
    return re.sub(r"\s+", "", normalized)


def fingerprint_clause(normalized_text: str) -> str:
    """Return the lowercase SHA-256 fingerprint of normalized clause text."""

    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _value(row: pd.Series, *names: str) -> object:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return ""


def _code(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    match = re.match(r"(\d+)(?:\.0)?(?:\D.*)?$", text)
    if match is not None:
        text = match.group(1)
    return text.zfill(6)


def _date(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.normalize().strftime("%Y-%m-%d")


def _text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "是"}


def _checkpoint_id(
    family_key: str, share_code: str, upload_info_id: str, trigger_type: str
) -> str:
    identity = "|".join(
        (family_key, share_code, upload_info_id, trigger_type, str(GENERATION_RULE_VERSION))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _source_document(row: pd.Series, share_code: str, upload_info_id: str) -> str:
    value = _text(_value(row, "source_document", "document_name", "filename"))
    return value or f"{share_code}_{upload_info_id}.pdf"


def _is_critical_section(row: pd.Series) -> bool:
    if "is_critical" in row.index:
        return _bool(row["is_critical"])
    section = _text(_value(row, "section_name", "section_type", "section"))
    if not section:
        return True
    folded = section.lower()
    return any(token in folded for token in ("investment", "allocation", "投资范围", "投资比例"))


def _section_is_uncomparable(row: pd.Series) -> bool:
    for name in ("section_comparable", "is_comparable", "critical_section_extracted"):
        if name in row.index and not _bool(row[name]):
            return True
    return not _text(_value(row, "section_text", "clause", "text", "content", "evidence"))


def _event_trigger(row: pd.Series) -> str | None:
    explicit = _text(_value(row, "event_type", "event", "transition_type")).lower()
    title = " ".join(
        _text(_value(row, name))
        for name in ("report_name", "reportName", "title", "source_document")
    ).lower()
    combined = f"{explicit} {title}"
    if any(token in combined for token in ("合并", "merge", "merger")):
        return "MERGER"
    if any(token in combined for token in ("转型", "转换", "transform", "conversion")):
        return "TRANSFORMATION"
    if any(token in combined for token in ("基金类型", "类型变更", "type_change", "type change")):
        return "TYPE_CHANGE"
    if any(token in combined for token in ("投资范围", "投资比例", "investment_range", "investment range")):
        return "INVESTMENT_RANGE_CHANGE"
    return None


def _candidate(
    *,
    family_key: str,
    share_code: str,
    upload_info_id: str,
    known_at: str,
    effective_date: str,
    trigger_type: str,
    source_document: str,
) -> dict[str, object]:
    return {
        "checkpoint_id": _checkpoint_id(family_key, share_code, upload_info_id, trigger_type),
        "family_key": family_key,
        "share_code": share_code,
        "upload_info_id": upload_info_id,
        "known_at": known_at,
        "effective_date": effective_date,
        "trigger_type": trigger_type,
        "is_critical": True,
        "source_document": source_document,
        "generation_rule_version": GENERATION_RULE_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "manifest_revision": 1,
        "supersedes_checkpoint_id": "",
        "revision_reason": "",
    }


def _with_codes(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    column = next((name for name in candidates if name in out.columns), None)
    if column is None:
        out["_share_code"] = ""
    else:
        out["_share_code"] = out[column].map(_code)
    return out


def _alias_join_key(value: object) -> str:
    """Normalize only known share-class suffix drift for authoritative sample joins."""

    text = _text(value)
    text = re.sub(r"-(?:进取|优先|[A-Z])$", "", text)
    return text.rstrip("-")


def _sample_authority(
    fund_aliases: pd.DataFrame, family_sample: pd.DataFrame
) -> tuple[dict[str, str], pd.DataFrame]:
    required = {"share_code", "family_key", "status"}
    missing = required - set(family_sample.columns)
    if missing:
        raise ValueError(f"family_sample must contain columns: {sorted(missing)}")

    sample = _with_codes(family_sample, ("share_code",))
    sample["family_key"] = sample["family_key"].map(_text)
    sample["status"] = sample["status"].map(_text)
    if "inception_date" in sample:
        sample["inception_date"] = sample["inception_date"].map(_date)
    else:
        sample["inception_date"] = ""
    sample = sample.loc[sample["_share_code"].ne("") & sample["family_key"].ne("")]
    sample = sample.sort_values(["family_key", "_share_code"], kind="stable")

    code_conflicts = sample.groupby("_share_code")["family_key"].nunique()
    if (code_conflicts > 1).any():
        raise ValueError("family sample assigns one share_code to multiple family keys")
    family_conflicts = sample.groupby("family_key")["_share_code"].nunique()
    if (family_conflicts > 1).any():
        raise ValueError("family sample assigns one family_key to multiple share codes")
    sample = sample.drop_duplicates("family_key", keep="first").reset_index(drop=True)

    exact_sample_keys = set(sample["family_key"])
    sample_key_by_code = sample.set_index("_share_code")["family_key"].to_dict()
    normalized_candidates: dict[str, set[str]] = {}
    for family_key in exact_sample_keys:
        normalized_candidates.setdefault(_alias_join_key(family_key), set()).add(family_key)

    aliases = _with_codes(fund_aliases, ("query_code", "share_code", "fund_code"))
    if "family_key" not in aliases.columns:
        raise ValueError("fund_aliases must contain family_key")
    aliases["family_key"] = aliases["family_key"].map(_text)

    family_by_code = dict(sample_key_by_code)
    for record in aliases.to_dict("records"):
        query_code = _code(record.get("_share_code"))
        alias_key = _text(record.get("family_key"))
        if not query_code or not alias_key:
            continue

        sample_code = _code(record.get("sample_share_code"))
        if sample_code and sample_code in sample_key_by_code:
            authoritative = sample_key_by_code[sample_code]
        elif alias_key in exact_sample_keys:
            authoritative = alias_key
        else:
            candidates = normalized_candidates.get(_alias_join_key(alias_key), set())
            if len(candidates) > 1:
                raise ValueError(
                    f"ambiguous normalized alias match for {alias_key!r}: {sorted(candidates)}"
                )
            if not candidates:
                continue
            authoritative = next(iter(candidates))

        existing = family_by_code.get(query_code)
        if existing is not None and existing != authoritative:
            raise ValueError(
                f"alias code {query_code} maps to multiple sample families: "
                f"{sorted({existing, authoritative})}"
            )
        family_by_code[query_code] = authoritative
    return family_by_code, sample


def generate_required_checkpoints(
    metadata: pd.DataFrame,
    fund_aliases: pd.DataFrame,
    document_sections: pd.DataFrame | None = None,
    *,
    family_sample: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a deterministic, parser-outcome-independent checkpoint denominator."""

    sample: pd.DataFrame | None = None
    if family_sample is not None:
        family_by_code, sample = _sample_authority(fund_aliases, family_sample)
    else:
        aliases = _with_codes(fund_aliases, ("query_code", "share_code", "fund_code"))
        if "family_key" not in aliases.columns:
            raise ValueError("fund_aliases must contain family_key")
        aliases["family_key"] = aliases["family_key"].map(_text)
        aliases = aliases.loc[aliases["_share_code"].ne("") & aliases["family_key"].ne("")]
        conflicts = aliases.groupby("_share_code")["family_key"].nunique()
        if (conflicts > 1).any():
            raise ValueError("alias code maps to multiple family keys")
        aliases = aliases.sort_values(["family_key", "_share_code"], kind="stable").drop_duplicates(
            "_share_code", keep="first"
        )
        family_by_code = aliases.set_index("_share_code")["family_key"].to_dict()

    records = _with_codes(metadata, ("fundCode", "share_code", "fund_code", "query_code"))
    records = records.loc[records["_share_code"].isin(family_by_code)].copy()
    result: list[dict[str, object]] = []

    if sample is not None:
        family_for_record = records["_share_code"].map(family_by_code)
        for _, sample_row in sample.iterrows():
            family_key = sample_row["family_key"]
            share_code = sample_row["_share_code"]
            inception = sample_row["inception_date"]
            if inception:
                result.append(
                    _candidate(
                        family_key=family_key,
                        share_code=share_code,
                        upload_info_id=f"INCEPTION:{share_code}:{inception}",
                        known_at=inception,
                        effective_date=inception,
                        trigger_type="INCEPTION",
                        source_document=f"fund_master:{share_code}",
                    )
                )
            if sample_row["status"] != "清盘":
                continue
            family_records = records.loc[family_for_record.eq(family_key)]
            terminal_evidence: list[tuple[str, str, str]] = []
            for _, row in family_records.iterrows():
                terminal_date = _date(
                    _value(row, "end_date", "termination_date", "delist_date")
                )
                if not terminal_date:
                    continue
                known_at = _date(
                    _value(row, "end_known_at", "known_at", "reportSendDate", "report_send_date")
                )
                terminal_evidence.append((terminal_date, known_at, row["_share_code"]))
            if terminal_evidence:
                terminal_date, known_at, _ = sorted(terminal_evidence)[-1]
            else:
                terminal_date, known_at = "", ""
            result.append(
                _candidate(
                    family_key=family_key,
                    share_code=share_code,
                    upload_info_id=f"TERMINAL_STATE:{share_code}:{terminal_date or 'UNKNOWN'}",
                    known_at=known_at,
                    effective_date=terminal_date,
                    trigger_type="TERMINAL_STATE",
                    source_document=f"fund_master:{share_code}",
                )
            )

    for share_code, group in records.groupby("_share_code", sort=True):
        family_key = family_by_code[share_code]
        ordered = group.assign(
            _known=group.apply(lambda row: _date(_value(row, "known_at", "reportSendDate", "report_send_date")), axis=1),
            _upload=group.apply(lambda row: _text(_value(row, "upload_info_id", "uploadInfoId", "upload_id")), axis=1),
        ).sort_values(["_known", "_upload"], kind="stable")
        first = ordered.iloc[0]
        inception = _date(_value(first, "inception_date", "found_date", "issue_date", "list_date"))
        if sample is None and inception:
            result.append(
                _candidate(
                    family_key=family_key,
                    share_code=share_code,
                    upload_info_id=f"INCEPTION:{share_code}:{inception}",
                    known_at=inception,
                    effective_date=inception,
                    trigger_type="INCEPTION",
                    source_document=f"fund_master:{share_code}",
                )
            )
        for _, row in ordered.iterrows():
            upload_info_id = _text(_value(row, "upload_info_id", "uploadInfoId", "upload_id"))
            if not upload_info_id:
                continue
            known_at = _date(_value(row, "known_at", "reportSendDate", "report_send_date"))
            effective_date = _date(_value(row, "effective_date", "effectiveDate"))
            trigger = _event_trigger(row)
            if trigger:
                result.append(
                    _candidate(
                        family_key=family_key,
                        share_code=share_code,
                        upload_info_id=upload_info_id,
                        known_at=known_at,
                        effective_date=effective_date,
                        trigger_type=trigger,
                        source_document=_source_document(row, share_code, upload_info_id),
                    )
                )
        if sample is None:
            terminal_evidence: list[tuple[str, str, str]] = []
            for _, row in ordered.iterrows():
                terminal_date = _date(
                    _value(row, "end_date", "termination_date", "delist_date", "due_date")
                )
                state = _text(_value(row, "status", "fund_status")).lower()
                known_at = _date(_value(row, "end_known_at", "known_at", "reportSendDate"))
                if terminal_date or state in _TERMINAL_STATES:
                    terminal_evidence.append(
                        (terminal_date or known_at, known_at or terminal_date, row["_upload"])
                    )
            terminal_evidence = [evidence for evidence in terminal_evidence if evidence[0]]
            if terminal_evidence:
                terminal_date, known_at, _ = sorted(terminal_evidence)[-1]
                result.append(
                    _candidate(
                        family_key=family_key,
                        share_code=share_code,
                        upload_info_id=f"TERMINAL_STATE:{share_code}:{terminal_date}",
                        known_at=known_at,
                        effective_date=terminal_date,
                        trigger_type="TERMINAL_STATE",
                        source_document=f"fund_master:{share_code}",
                    )
                )

    if document_sections is not None:
        sections = _with_codes(document_sections, ("share_code", "fundCode", "fund_code", "query_code"))
        sections = sections.loc[sections["_share_code"].isin(family_by_code)].copy()
        sections["_known"] = sections.apply(
            lambda row: _date(_value(row, "known_at", "reportSendDate", "report_send_date")), axis=1
        )
        sections["_upload"] = sections.apply(
            lambda row: _text(_value(row, "upload_info_id", "uploadInfoId", "upload_id")), axis=1
        )
        sections = sections.sort_values(["_share_code", "_known", "_upload"], kind="stable")
        previous_clause: dict[tuple[str, str], str] = {}
        for _, row in sections.iterrows():
            if not _is_critical_section(row):
                continue
            share_code = row["_share_code"]
            family_key = family_by_code[share_code]
            upload_info_id = row["_upload"]
            if not upload_info_id:
                continue
            known_at = row["_known"]
            effective_date = _date(_value(row, "effective_date", "effectiveDate"))
            if _section_is_uncomparable(row):
                result.append(
                    _candidate(
                        family_key=family_key,
                        share_code=share_code,
                        upload_info_id=upload_info_id,
                        known_at=known_at,
                        effective_date=effective_date,
                        trigger_type="SECTION_UNCOMPARABLE",
                        source_document=_source_document(row, share_code, upload_info_id),
                    )
                )
                continue
            normalized = normalize_investment_clause(
                _text(_value(row, "section_text", "clause", "text", "content", "evidence"))
            )
            key = (family_key, share_code)
            if previous_clause.get(key) != normalized:
                result.append(
                    _candidate(
                        family_key=family_key,
                        share_code=share_code,
                        upload_info_id=upload_info_id,
                        known_at=known_at,
                        effective_date=effective_date,
                        trigger_type="MATERIAL_SECTION_CHANGE",
                        source_document=_source_document(row, share_code, upload_info_id),
                    )
                )
            previous_clause[key] = normalized

    if not result:
        return pd.DataFrame(columns=_CHECKPOINT_COLUMNS)
    out = pd.DataFrame(result).sort_values(list(_CHECKPOINT_COLUMNS), kind="stable")
    out = out.drop_duplicates("checkpoint_id", keep="first")
    return out.sort_values(
        ["family_key", "known_at", "upload_info_id", "trigger_type", "checkpoint_id"], kind="stable"
    ).reset_index(drop=True)[list(_CHECKPOINT_COLUMNS)]


def _canonical_manifest(checkpoints: pd.DataFrame) -> pd.DataFrame:
    missing = set(_CHECKPOINT_COLUMNS) - set(checkpoints.columns)
    if missing:
        raise ValueError(f"Missing checkpoint columns: {sorted(missing)}")
    out = checkpoints.loc[:, _CHECKPOINT_COLUMNS].copy()
    out["checkpoint_id"] = out["checkpoint_id"].map(_text)
    if out["checkpoint_id"].eq("").any():
        raise ValueError("checkpoint_id must be nonempty")
    if out["checkpoint_id"].duplicated().any():
        raise ValueError("duplicate checkpoint_id")
    out["family_key"] = out["family_key"].map(_text)
    out["share_code"] = out["share_code"].map(_code)
    out["upload_info_id"] = out["upload_info_id"].map(_text)
    for column in ("known_at", "effective_date"):
        out[column] = out[column].map(_date)
    out["trigger_type"] = out["trigger_type"].map(_text)
    out["is_critical"] = out["is_critical"].map(_bool)
    out["source_document"] = out["source_document"].map(_text)
    for column in ("generation_rule_version", "normalization_version", "manifest_revision"):
        numeric = pd.to_numeric(out[column], errors="raise")
        if (numeric % 1 != 0).any():
            raise ValueError(f"{column} must be an integer")
        out[column] = numeric.astype(int)
    out["supersedes_checkpoint_id"] = out["supersedes_checkpoint_id"].map(_text)
    out["revision_reason"] = out["revision_reason"].map(_text)
    return out


def freeze_checkpoint_manifest(checkpoints: pd.DataFrame, path: str | Path) -> Path:
    """Atomically freeze a checkpoint manifest while enforcing append-only revisions."""

    destination = Path(path)
    candidate = _canonical_manifest(checkpoints)
    if destination.exists():
        existing = _canonical_manifest(pd.read_csv(destination, dtype=str, keep_default_na=False))
        frozen_ids = set(existing["checkpoint_id"])
        submitted_ids = set(candidate["checkpoint_id"])
        if not frozen_ids.issubset(submitted_ids):
            raise ValueError("manifest deletion is not allowed")
        existing_by_id = existing.set_index("checkpoint_id")
        submitted_existing = candidate.loc[candidate["checkpoint_id"].isin(frozen_ids)].set_index("checkpoint_id")
        if not existing_by_id.equals(submitted_existing.loc[existing_by_id.index]):
            raise ValueError("manifest mutation is not allowed")
        additions = candidate.loc[~candidate["checkpoint_id"].isin(frozen_ids)].copy()
        if not additions.empty:
            if (additions["manifest_revision"] <= existing["manifest_revision"].max()).any():
                raise ValueError("appended revision must be higher than frozen revisions")
            if additions["supersedes_checkpoint_id"].eq("").any() or additions["revision_reason"].eq("").any():
                raise ValueError("appended revision requires supersedes_checkpoint_id and revision_reason")
            if not additions["supersedes_checkpoint_id"].isin(frozen_ids).all():
                raise ValueError("appended revision must supersede an existing checkpoint")
        frozen = pd.concat([existing, additions], ignore_index=True)
    else:
        if (candidate["manifest_revision"] != 1).any():
            raise ValueError("initial manifest rows must have manifest_revision 1")
        if candidate["supersedes_checkpoint_id"].ne("").any() or candidate["revision_reason"].ne("").any():
            raise ValueError("initial manifest cannot contain corrections")
        frozen = candidate
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        frozen.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            temporary.replace(destination)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (2**attempt))
    return destination


_ADJUDICATION_COLUMNS = (
    "checkpoint_id",
    "disposition",
    "decision_method",
    "evidence_raw",
    "evidence_normalized",
    "evidence_location",
    "evidence_source_document",
    "evidence_upload_info_id",
    "evidence_known_at",
    "evidence_effective_date",
    "clause_sha256",
    "normalization_version",
    "predecessor_checkpoint_id",
    "fund_type",
    "equity_min_pct",
    "equity_max_pct",
    "parser_version",
    "reviewer_version",
    "failure_reason",
)
_FROZEN_FAILURE_REASONS = frozenset(
    {
        "PDF_MISSING",
        "SECTION_EXTRACTION_FAILED",
        "SECTION_COMPARISON_FAILED",
        "AMBIGUOUS_ASSET_DENOMINATOR",
        "POSSIBLE_STATE_CHANGE",
        "SOURCE_CONFLICT",
        "MISSING_PREDECESSOR",
    }
)


def _record_text(record: dict[str, object], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
    return ""


def _section_raw(record: dict[str, object]) -> str:
    return _record_text(record, "evidence_raw", "section_text", "clause", "text", "content", "evidence")


def _section_location(record: dict[str, object]) -> str:
    return _record_text(
        record,
        "evidence_location",
        "source_location",
        "page_location",
        "location",
        "source_document",
    )


def _section_normalized(record: dict[str, object]) -> str:
    raw = _section_raw(record)
    if raw:
        return normalize_investment_clause(raw)
    return normalize_investment_clause(_record_text(record, "normalized_clause"))


def _critical_records(sections: pd.DataFrame) -> list[dict[str, object]]:
    if sections.empty:
        return []
    records: list[dict[str, object]] = []
    for record in sections.to_dict("records"):
        if _is_critical_section(pd.Series(record)):
            record = dict(record)
            record["normalized_clause"] = _section_normalized(record)
            records.append(record)
    return records


def build_clause_clusters(sections: pd.DataFrame) -> pd.DataFrame:
    """Cluster non-empty normalized critical sections without dropping sources."""

    records = [record for record in _critical_records(sections) if record["normalized_clause"]]
    if not records:
        columns = list(sections.columns)
        columns.extend(
            column
            for column in ("normalized_clause", "normalization_version", "clause_sha256", "cluster_id", "cluster_member_count")
            if column not in columns
        )
        return pd.DataFrame(columns=columns)
    clustered = pd.DataFrame(records)
    clustered["normalization_version"] = NORMALIZATION_VERSION
    clustered["clause_sha256"] = clustered["normalized_clause"].map(fingerprint_clause)
    clustered["cluster_id"] = (
        "n" + clustered["normalization_version"].astype(str) + "-" + clustered["clause_sha256"]
    )
    clustered["cluster_member_count"] = clustered.groupby("cluster_id", dropna=False)["cluster_id"].transform("size")
    for column in ("known_at", "effective_date", "upload_info_id"):
        if column not in clustered:
            clustered[column] = ""
        clustered[column] = clustered[column].map(_text)
    if "source_document" not in clustered:
        clustered["source_document"] = ""
    clustered["source_document"] = clustered["source_document"].map(_text)
    tie_columns = sorted(
        column
        for column in clustered.columns
        if column
        not in {
            "cluster_id",
            "known_at",
            "effective_date",
            "upload_info_id",
            "source_document",
        }
    )
    clustered["_cluster_tie_breaker"] = clustered.apply(
        lambda row: "\x1f".join(f"{column}={_text(row[column])}" for column in tie_columns),
        axis=1,
    )
    return clustered.sort_values(
        [
            "cluster_id",
            "known_at",
            "effective_date",
            "upload_info_id",
            "source_document",
            "_cluster_tie_breaker",
        ],
        kind="stable",
    ).drop(columns="_cluster_tie_breaker").reset_index(drop=True)


def _flatten_parsed_documents(parsed_documents: pd.DataFrame) -> list[dict[str, object]]:
    """Accept both cache rows (with a document object) and already-flattened rows."""

    if parsed_documents.empty:
        return []
    flattened: list[dict[str, object]] = []
    for record in parsed_documents.to_dict("records"):
        document = record.get("document")
        failure = record.get("failure")
        merged = dict(record)
        if isinstance(document, dict):
            merged.update(document)
        if isinstance(failure, dict):
            merged.update(failure)
        flattened.append(merged)
    return flattened


def _numeric(value: object) -> float | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _explicit_states(
    records: list[dict[str, object]],
    *,
    current_normalized: str,
    current_source_document: str,
    current_location: str,
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    for record in records:
        status = _record_text(record, "status").lower()
        low = _numeric(record.get("equity_min_pct"))
        high = _numeric(record.get("equity_max_pct"))
        quote = _record_text(record, "evidence", "evidence_raw")
        parser_version = _record_text(record, "parser_version")
        source_document = _record_text(record, "source_document")
        quote_normalized = normalize_investment_clause(quote) if quote else ""
        try:
            quote_constraint = extract_equity_constraint(quote)
        except ValueError:
            quote_constraint = None
        if (
            status not in ("", "success")
            or low is None
            or high is None
            or not quote_normalized
            or not parser_version
            or not source_document
            or not current_source_document
            or not current_location
            or source_document != current_source_document
            or quote_normalized not in current_normalized
            or quote_constraint is None
            or quote_constraint.equity_min_pct != low
            or quote_constraint.equity_max_pct != high
        ):
            continue
        states.append(
            {
                "fund_type": _record_text(record, "fund_type", "fund_type_override"),
                "equity_min_pct": low,
                "equity_max_pct": high,
                "parser_version": parser_version,
                "source_document": source_document,
            }
        )
    return states


def _has_quote_state_conflict(
    records: list[dict[str, object]],
    *,
    current_normalized: str,
    current_source_document: str,
    current_location: str,
) -> bool:
    """Detect an explicit parser state contradicted by its same-source quote."""

    for record in records:
        status = _record_text(record, "status").lower()
        low = _numeric(record.get("equity_min_pct"))
        high = _numeric(record.get("equity_max_pct"))
        quote = _record_text(record, "evidence", "evidence_raw")
        parser_version = _record_text(record, "parser_version")
        source_document = _record_text(record, "source_document")
        quote_normalized = normalize_investment_clause(quote) if quote else ""
        if (
            status not in ("", "success")
            or low is None
            or high is None
            or not quote_normalized
            or not parser_version
            or not source_document
            or not current_source_document
            or not current_location
            or source_document != current_source_document
            or quote_normalized not in current_normalized
        ):
            continue
        try:
            quote_constraint = extract_equity_constraint(quote)
        except ValueError:
            continue
        if (
            quote_constraint.equity_min_pct != low
            or quote_constraint.equity_max_pct != high
        ):
            return True
    return False


def _failure_from_records(records: list[dict[str, object]]) -> str:
    for record in records:
        if any(
            _record_text(record, name).upper() == "SOURCE_CONFLICT"
            for name in ("failure_reason", "reason")
        ):
            return "SOURCE_CONFLICT"
        if _record_text(record, "pdf_status").lower() in {"missing", "pdf_missing", "not_found"}:
            return "PDF_MISSING"
        if _record_text(record, "asset_denominator_status").lower() in {"ambiguous", "ambiguous_asset_denominator"}:
            return "AMBIGUOUS_ASSET_DENOMINATOR"
    values: list[str] = []
    for record in records:
        values.extend(
            _record_text(record, name).lower()
            for name in ("failure_reason", "reason", "status", "pdf_status", "extraction_status", "asset_denominator_status")
        )
    joined = " ".join(values)
    if "pdf_missing" in joined or "pdf missing" in joined or "missing pdf" in joined or "pdf_status" in joined or "pdf" in joined and "missing" in joined:
        return "PDF_MISSING"
    if "ambiguous_asset_denominator" in joined or "ambiguous denominator" in joined or "asset_denominator_status ambiguous" in joined or "denominator" in joined and "ambiguous" in joined:
        return "AMBIGUOUS_ASSET_DENOMINATOR"
    if "comparison_failed" in joined or "comparison failed" in joined or "uncomparable" in joined:
        return "SECTION_COMPARISON_FAILED"
    if "extraction_failed" in joined or "extraction failed" in joined or "failure" in joined:
        return "SECTION_EXTRACTION_FAILED"
    return ""


def _empty_disposition(record: dict[str, object]) -> dict[str, object]:
    decision = dict(record)
    decision.update(
        {
            "disposition": Disposition.UNRESOLVED.value,
            "decision_method": "",
            "evidence_raw": "",
            "evidence_normalized": "",
            "evidence_location": "",
            "evidence_source_document": "",
            "evidence_upload_info_id": "",
            "evidence_known_at": "",
            "evidence_effective_date": "",
            "clause_sha256": "",
            "normalization_version": NORMALIZATION_VERSION,
            "predecessor_checkpoint_id": "",
            "fund_type": "",
            "equity_min_pct": None,
            "equity_max_pct": None,
            "parser_version": "",
            "reviewer_version": "",
            "failure_reason": "",
        }
    )
    return decision


def resolve_lifecycle_evidence(
    checkpoint: dict[str, object],
    candidate_sections: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """Bind a dated lifecycle checkpoint to same-family causal section evidence.

    Candidates are accepted only when the caller supplies complete provenance.
    The closest known date at or before the checkpoint wins. Equal-date records
    are retained in deterministic order so the adjudicator can apply its normal
    explicit quote/state checks.
    """

    trigger_type = _record_text(checkpoint, "trigger_type").upper()
    if trigger_type not in {"INCEPTION", "TERMINAL_STATE"}:
        return [], "SECTION_EXTRACTION_FAILED"
    checkpoint_family = _record_text(checkpoint, "family_key")
    checkpoint_known_at = _date(checkpoint.get("known_at"))
    if not checkpoint_family or not checkpoint_known_at:
        return [], "SECTION_EXTRACTION_FAILED"

    eligible: list[dict[str, object]] = []
    for candidate in candidate_sections:
        candidate_known_at = _date(candidate.get("known_at"))
        raw = _record_text(candidate, "section_text")
        normalized = _section_normalized(candidate)
        if (
            _record_text(candidate, "family_key") != checkpoint_family
            or not candidate_known_at
            or candidate_known_at > checkpoint_known_at
            or not _record_text(candidate, "upload_info_id")
            or not _record_text(candidate, "source_document")
            or not _record_text(candidate, "evidence_location")
            or not raw
            or _record_text(candidate, "parsed_upload_info_id")
            != _record_text(candidate, "upload_info_id")
            or _date(candidate.get("parsed_known_at")) != candidate_known_at
            or _record_text(candidate, "parsed_source_document")
            != _record_text(candidate, "source_document")
            or _record_text(candidate, "parsed_evidence_location")
            != _record_text(candidate, "evidence_location")
            or not _record_text(candidate, "parsed_evidence_raw")
            or _record_text(candidate, "parsed_normalized_clause") != normalized
            or not _record_text(candidate, "evidence", "evidence_raw")
            or normalize_investment_clause(
                _record_text(candidate, "evidence", "evidence_raw")
            )
            != normalized
        ):
            continue
        record = dict(candidate)
        record["normalized_clause"] = normalized
        eligible.append(record)

    if not eligible:
        return [], "SECTION_EXTRACTION_FAILED"
    latest_known_at = max(_date(record.get("known_at")) for record in eligible)
    selected = [
        record for record in eligible if _date(record.get("known_at")) == latest_known_at
    ]
    if _failure_from_records(selected) == "SOURCE_CONFLICT" or len(
        {record["normalized_clause"] for record in selected}
    ) != 1:
        return [], "SOURCE_CONFLICT"
    selected.sort(
        key=lambda record: (
            _record_text(record, "source_document"),
            _record_text(record, "upload_info_id"),
            _section_location(record),
            record["normalized_clause"],
        )
    )
    return selected, ""


def adjudicate_checkpoints(
    checkpoints: pd.DataFrame,
    sections: pd.DataFrame,
    parsed_documents: pd.DataFrame,
    *,
    lifecycle_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adjudicate checkpoint evidence conservatively and only from causal sources."""

    if checkpoints.empty:
        return pd.DataFrame(columns=list(_ADJUDICATION_COLUMNS))
    section_by_upload: dict[str, list[dict[str, object]]] = {}
    for record in _critical_records(sections):
        section_by_upload.setdefault(_record_text(record, "upload_info_id"), []).append(record)
    parsed_by_upload: dict[str, list[dict[str, object]]] = {}
    for record in _flatten_parsed_documents(parsed_documents):
        parsed_by_upload.setdefault(_record_text(record, "upload_info_id"), []).append(record)
    lifecycle_records = _critical_records(
        lifecycle_evidence if lifecycle_evidence is not None else pd.DataFrame()
    )

    ordered = checkpoints.copy()
    ordered["_adjudication_known_at"] = ordered.get("known_at", "").map(_date)
    ordered["_adjudication_effective_date"] = ordered.get("effective_date", "").map(_date)
    ordered["_adjudication_checkpoint_id"] = ordered.get("checkpoint_id", "").map(_text)
    if "family_key" not in ordered:
        ordered["family_key"] = ""
    ordered = ordered.sort_values(
        ["family_key", "_adjudication_known_at", "_adjudication_effective_date", "_adjudication_checkpoint_id"],
        kind="stable",
    )

    previous_by_family: dict[str, dict[str, object] | None] = {}
    decisions: list[dict[str, object]] = []
    for checkpoint in ordered.drop(columns=["_adjudication_known_at", "_adjudication_effective_date", "_adjudication_checkpoint_id"]).to_dict("records"):
        decision = _empty_disposition(checkpoint)
        family_key = _record_text(checkpoint, "family_key")
        upload_info_id = _record_text(checkpoint, "upload_info_id")
        trigger_type = _record_text(checkpoint, "trigger_type").upper()
        is_lifecycle = trigger_type in {"INCEPTION", "TERMINAL_STATE"}
        resolver_failure = ""
        if is_lifecycle:
            current_sections, resolver_failure = resolve_lifecycle_evidence(
                checkpoint, lifecycle_records
            )
        else:
            current_sections = section_by_upload.get(upload_info_id, [])
            current_parsed = parsed_by_upload.get(upload_info_id, [])
        if resolver_failure:
            decision["failure_reason"] = resolver_failure
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        if is_lifecycle:
            supported: list[tuple[dict[str, object], dict[str, object]]] = []
            for section in current_sections:
                parser_record = {
                    **section,
                    "upload_info_id": _record_text(section, "parsed_upload_info_id"),
                    "known_at": _record_text(section, "parsed_known_at"),
                    "source_document": _record_text(section, "parsed_source_document"),
                    "evidence_location": _record_text(section, "parsed_evidence_location"),
                    "evidence_raw": _record_text(section, "parsed_evidence_raw"),
                }
                states = _explicit_states(
                    [parser_record],
                    current_normalized=section["normalized_clause"],
                    current_source_document=_record_text(section, "source_document"),
                    current_location=_record_text(section, "evidence_location"),
                )
                supported.extend((section, state) for state in states)
            unique_states = {
                (state["fund_type"], state["equity_min_pct"], state["equity_max_pct"])
                for _, state in supported
            }
            if len(unique_states) > 1:
                decision["failure_reason"] = "SOURCE_CONFLICT"
                previous_by_family[family_key] = None
                decisions.append(decision)
                continue
            if not supported:
                decision["failure_reason"] = "POSSIBLE_STATE_CHANGE"
                previous_by_family[family_key] = None
                decisions.append(decision)
                continue
            section, state = sorted(
                supported,
                key=lambda item: (
                    _record_text(item[0], "source_document"),
                    _record_text(item[0], "upload_info_id"),
                    _record_text(item[0], "evidence_location"),
                ),
            )[0]
            normalized = section["normalized_clause"]
            decision.update(
                {
                    "disposition": Disposition.PARSED_STATE.value,
                    "decision_method": "EXPLICIT_STATE",
                    "evidence_raw": _section_raw(section),
                    "evidence_normalized": normalized,
                    "evidence_location": _record_text(section, "evidence_location"),
                    "evidence_source_document": _record_text(section, "source_document"),
                    "evidence_upload_info_id": _record_text(section, "upload_info_id"),
                    "evidence_known_at": _date(section.get("known_at")),
                    "evidence_effective_date": _date(section.get("effective_date")),
                    "clause_sha256": fingerprint_clause(normalized),
                    "normalization_version": NORMALIZATION_VERSION,
                    "fund_type": state["fund_type"],
                    "equity_min_pct": state["equity_min_pct"],
                    "equity_max_pct": state["equity_max_pct"],
                    "parser_version": state["parser_version"],
                    "failure_reason": "",
                }
            )
            previous_by_family[family_key] = decision
            decisions.append(decision)
            continue
        failure = _failure_from_records([*current_sections, *current_parsed])
        raw_sections = [record for record in current_sections if _section_raw(record)]
        if failure:
            decision["failure_reason"] = failure
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        if not raw_sections:
            decision["failure_reason"] = "SECTION_EXTRACTION_FAILED"
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        clauses = {(record["normalized_clause"], _section_raw(record), _section_location(record)) for record in raw_sections}
        if len({clause[0] for clause in clauses}) != 1:
            decision["failure_reason"] = "SOURCE_CONFLICT"
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        normalized, raw, location = sorted(clauses)[0]
        if not location:
            decision["failure_reason"] = "SECTION_COMPARISON_FAILED"
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        decision.update(
            {
                "evidence_raw": raw,
                "evidence_normalized": normalized,
                "evidence_location": location,
                "clause_sha256": fingerprint_clause(normalized),
                "normalization_version": NORMALIZATION_VERSION,
            }
        )
        current_source_document = _record_text(raw_sections[0], "source_document") or _record_text(
            checkpoint, "source_document"
        )
        if _has_quote_state_conflict(
            current_parsed,
            current_normalized=normalized,
            current_source_document=current_source_document,
            current_location=location,
        ):
            decision["failure_reason"] = "SOURCE_CONFLICT"
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        states = _explicit_states(
            current_parsed,
            current_normalized=normalized,
            current_source_document=current_source_document,
            current_location=location,
        )
        unique_states = {
            (state["fund_type"], state["equity_min_pct"], state["equity_max_pct"])
            for state in states
        }
        if len(unique_states) > 1:
            decision["failure_reason"] = "SOURCE_CONFLICT"
            previous_by_family[family_key] = None
            decisions.append(decision)
            continue
        if states:
            state = sorted(
                states,
                key=lambda item: (
                    _text(item["source_document"]),
                    _text(item["parser_version"]),
                ),
            )[0]
            decision.update(
                {
                    "disposition": Disposition.PARSED_STATE.value,
                    "decision_method": "EXPLICIT_STATE",
                    "fund_type": state["fund_type"],
                    "equity_min_pct": state["equity_min_pct"],
                    "equity_max_pct": state["equity_max_pct"],
                    "parser_version": state["parser_version"],
                    "failure_reason": "",
                }
            )
            previous_by_family[family_key] = decision
            decisions.append(decision)
            continue
        previous = previous_by_family.get(family_key)
        exact_match = bool(previous) and decision["clause_sha256"] == previous.get("clause_sha256")
        if previous and exact_match:
            decision.update(
                {
                    "disposition": Disposition.VERIFIED_NOOP.value,
                    "decision_method": "EXACT_CRITICAL_SECTION_MATCH",
                    "predecessor_checkpoint_id": previous["checkpoint_id"],
                    "fund_type": previous["fund_type"],
                    "equity_min_pct": previous["equity_min_pct"],
                    "equity_max_pct": previous["equity_max_pct"],
                    "parser_version": previous["parser_version"],
                    "failure_reason": "",
                }
            )
            previous_by_family[family_key] = decision
        else:
            decision["failure_reason"] = "MISSING_PREDECESSOR" if previous is None else "POSSIBLE_STATE_CHANGE"
            previous_by_family[family_key] = None
        decisions.append(decision)
    result = pd.DataFrame(decisions)
    ordered_columns = list(checkpoints.columns) + [
        column for column in _ADJUDICATION_COLUMNS if column not in checkpoints.columns
    ]
    return result.reindex(columns=ordered_columns)


_TIMELINE_STATE_COLUMNS = (
    "fund_type",
    "equity_min_pct",
    "equity_max_pct",
    "evidence_raw",
    "evidence_normalized",
    "clause_sha256",
    "parser_version",
    "reviewer_version",
)
_TIMELINE_EXTRA_COLUMNS = (
    "state_source_checkpoint_id",
    "effective_from",
    "effective_to",
    "retrospective_effective_date",
    "causal_violation",
)


def _strict_timeline_date(value: object, field: str) -> pd.Timestamp:
    """Parse a timeline date, rejecting malformed non-empty values."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, str) and not value.strip():
        return pd.NaT
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"invalid {field}: {value!r}")
    try:
        return pd.Timestamp(parsed).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def _timeline_sort_date(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp.max if pd.isna(value) else value


def _timeline_causal_key(record: dict[str, object]) -> tuple[object, object, str]:
    return (
        _timeline_sort_date(record["_known_at"]),
        _timeline_sort_date(record["_effective_date"]),
        str(record["checkpoint_id"]),
    )


def _timeline_number(value: object) -> float | None:
    numeric = _numeric(value)
    return None if numeric is None or pd.isna(numeric) else numeric


def _timeline_usable_state(record: dict[str, object]) -> bool:
    return bool(record.get("_usable_state", False)) and not pd.isna(record.get("effective_from", pd.NaT))


def _timeline_copy_state(target: dict[str, object], source: dict[str, object]) -> None:
    for column in _TIMELINE_STATE_COLUMNS:
        if column in source:
            target[column] = source[column]


def _timeline_clear_state(target: dict[str, object]) -> None:
    for column in _TIMELINE_STATE_COLUMNS:
        if column in target:
            target[column] = None if column.startswith("equity_") else ""


def build_checkpoint_timeline(dispositions: pd.DataFrame) -> pd.DataFrame:
    """Build a causal, family-scoped state timeline from adjudications.

    A parsed state is anchored at the later of its discovery and quoted
    effective dates.  A NO-OP can only inherit from the predecessor named in
    its own disposition, and unresolved rows never participate in intervals.
    """

    if dispositions.empty:
        columns = list(dispositions.columns)
        for column in _TIMELINE_EXTRA_COLUMNS:
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)
    required = {"checkpoint_id", "family_key", "known_at", "disposition"}
    missing = required - set(dispositions.columns)
    if missing:
        raise ValueError(f"Missing disposition columns: {sorted(missing)}")

    original_columns = list(dispositions.columns)
    records: list[dict[str, object]] = []
    for raw in dispositions.to_dict("records"):
        record = dict(raw)
        record["checkpoint_id"] = _text(record.get("checkpoint_id"))
        if not record["checkpoint_id"]:
            raise ValueError("checkpoint_id must be nonempty")
        record["family_key"] = _text(record.get("family_key"))
        record["disposition"] = _text(record.get("disposition")).upper()
        record["predecessor_checkpoint_id"] = _text(record.get("predecessor_checkpoint_id"))
        record["_known_at"] = _strict_timeline_date(record.get("known_at"), "known_at")
        record["_effective_date"] = _strict_timeline_date(
            record.get("effective_date"), "effective_date"
        )
        record["effective_from"] = pd.NaT
        record["effective_to"] = pd.NaT
        record["state_source_checkpoint_id"] = ""
        record["retrospective_effective_date"] = bool(
            not pd.isna(record["_known_at"])
            and not pd.isna(record["_effective_date"])
            and record["_effective_date"] < record["_known_at"]
        )
        record["causal_violation"] = False
        record["_usable_state"] = False
        records.append(record)
    if len({record["checkpoint_id"] for record in records}) != len(records):
        raise ValueError("duplicate checkpoint_id")
    records.sort(key=_timeline_causal_key)
    by_id = {record["checkpoint_id"]: record for record in records}

    for record in records:
        disposition = record["disposition"]
        known_at = record["_known_at"]
        effective_date = record["_effective_date"]
        if disposition == Disposition.PARSED_STATE.value:
            if pd.isna(known_at):
                effective_from = pd.NaT
            elif pd.isna(effective_date):
                effective_from = known_at
            else:
                effective_from = max(known_at, effective_date)
            record["effective_from"] = effective_from
            record["state_source_checkpoint_id"] = record["checkpoint_id"]
            record["_usable_state"] = not pd.isna(effective_from)
            record["causal_violation"] = bool(
                pd.isna(known_at)
                or (
                    not pd.isna(effective_from)
                    and effective_from < known_at
                )
            )
            continue
        if disposition != Disposition.VERIFIED_NOOP.value:
            if disposition == Disposition.UNRESOLVED.value:
                _timeline_clear_state(record)
            continue

        predecessor_id = record["predecessor_checkpoint_id"]
        predecessor = by_id.get(predecessor_id)
        valid_predecessor = bool(predecessor)
        if valid_predecessor:
            valid_predecessor = predecessor["family_key"] == record["family_key"]
        if valid_predecessor:
            valid_predecessor = _timeline_causal_key(predecessor) < _timeline_causal_key(record)
        if valid_predecessor:
            valid_predecessor = _timeline_usable_state(predecessor)
        if not valid_predecessor:
            _timeline_clear_state(record)
            if not _text(record.get("failure_reason")):
                record["failure_reason"] = "MISSING_PREDECESSOR"
            if predecessor is not None and (
                predecessor["family_key"] != record["family_key"]
                or _timeline_causal_key(predecessor) >= _timeline_causal_key(record)
            ):
                record["causal_violation"] = True
            continue

        _timeline_copy_state(record, predecessor)
        predecessor_effective = predecessor["effective_from"]
        if pd.isna(known_at):
            effective_from = pd.NaT
        elif pd.isna(predecessor_effective):
            effective_from = pd.NaT
        else:
            effective_from = max(known_at, predecessor_effective)
        record["effective_from"] = effective_from
        record["state_source_checkpoint_id"] = predecessor["state_source_checkpoint_id"]
        record["_usable_state"] = not pd.isna(effective_from)
        source = by_id.get(record["state_source_checkpoint_id"])
        source_known_at = pd.NaT if source is None else source["_known_at"]
        record["causal_violation"] = bool(
            pd.isna(known_at)
            or (
                not pd.isna(effective_from)
                and effective_from < known_at
            )
        ) or bool(
            source is not None
            and not pd.isna(source_known_at)
            and not pd.isna(effective_from)
            and source_known_at > effective_from
        )

    # A usable row must resolve to an already-known, same-family parsed-state
    # source.  NO-OP rows point through their predecessor to that final source.
    for record in records:
        if not _timeline_usable_state(record):
            continue
        source = by_id.get(_text(record.get("state_source_checkpoint_id")))
        source_known_at = pd.NaT if source is None else source["_known_at"]
        valid_source = bool(
            source is not None
            and source["family_key"] == record["family_key"]
            and source["disposition"] == Disposition.PARSED_STATE.value
            and not pd.isna(source_known_at)
            and source_known_at <= record["effective_from"]
        )
        if not valid_source:
            record["causal_violation"] = True
            record["effective_from"] = pd.NaT
            record["effective_to"] = pd.NaT
            record["_usable_state"] = False

    # Preserve the legacy collector's same-time tightening for compatible
    # parsed ranges, while leaving explicit NO-OP inheritance untouched and
    # turning disjoint/categorically inconsistent ranges into source conflicts.
    groups: dict[tuple[str, pd.Timestamp], list[dict[str, object]]] = {}
    for record in records:
        if record["disposition"] == Disposition.PARSED_STATE.value and _timeline_usable_state(record):
            groups.setdefault((record["family_key"], record["effective_from"]), []).append(record)
    for group in groups.values():
        fund_types = {_text(record.get("fund_type")) for record in group}
        if len(fund_types) > 1:
            for record in group:
                record["disposition"] = Disposition.UNRESOLVED.value
                record["failure_reason"] = "SOURCE_CONFLICT"
                _timeline_clear_state(record)
                record["effective_from"] = pd.NaT
                record["effective_to"] = pd.NaT
                record["state_source_checkpoint_id"] = ""
                record["_usable_state"] = False
            continue
        ranges = [
            (_timeline_number(record.get("equity_min_pct")), _timeline_number(record.get("equity_max_pct")))
            for record in group
        ]
        if len(group) < 2 or any(low is None or high is None for low, high in ranges):
            continue
        tight_min = max(low for low, _ in ranges if low is not None)
        tight_max = min(high for _, high in ranges if high is not None)
        if tight_min > tight_max:
            for record in group:
                record["disposition"] = Disposition.UNRESOLVED.value
                record["failure_reason"] = "SOURCE_CONFLICT"
                _timeline_clear_state(record)
                record["effective_from"] = pd.NaT
                record["effective_to"] = pd.NaT
                record["state_source_checkpoint_id"] = ""
                record["_usable_state"] = False
            continue
        for record in group:
            record["equity_min_pct"] = tight_min
            record["equity_max_pct"] = tight_max

    # Every usable state at one time shares the next strictly later boundary,
    # independent of the causal/input ordering of the records themselves.
    effective_times_by_family: dict[str, set[pd.Timestamp]] = {}
    for record in records:
        if _timeline_usable_state(record):
            effective_times_by_family.setdefault(record["family_key"], set()).add(
                record["effective_from"]
            )
    next_time_by_family: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
    for family_key, effective_times in effective_times_by_family.items():
        ordered_times = sorted(effective_times)
        next_time_by_family[family_key] = dict(zip(ordered_times, ordered_times[1:]))
    for record in records:
        if _timeline_usable_state(record):
            record["effective_to"] = next_time_by_family[record["family_key"]].get(
                record["effective_from"], pd.NaT
            )

    result_records: list[dict[str, object]] = []
    for record in records:
        result = {column: record.get(column, "") for column in original_columns}
        result["effective_from"] = record["effective_from"]
        result["effective_to"] = record["effective_to"]
        result["state_source_checkpoint_id"] = record["state_source_checkpoint_id"]
        result["retrospective_effective_date"] = bool(record["retrospective_effective_date"])
        result["causal_violation"] = bool(record["causal_violation"])
        for column in _TIMELINE_STATE_COLUMNS:
            if column in record and column not in result:
                result[column] = record[column]
        result_records.append(result)
    output_columns = list(original_columns)
    output_columns.extend(column for column in _TIMELINE_EXTRA_COLUMNS if column not in output_columns)
    return pd.DataFrame(result_records).reindex(columns=output_columns).reset_index(drop=True)


def validate_checkpoint_invariants(
    checkpoints: pd.DataFrame,
    dispositions: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict[str, int]:
    """Count causal and integrity violations without repairing the inputs."""

    counts = {
        "causal_violations": 0,
        "unresolved_conflicts": 0,
        "nonpositive_intervals": 0,
        "orphan_noops": 0,
        "critical_unresolved": 0,
    }
    if dispositions.empty:
        return counts
    disposition_records: list[dict[str, object]] = []
    for raw in dispositions.to_dict("records"):
        record = dict(raw)
        record["checkpoint_id"] = _text(record.get("checkpoint_id"))
        record["family_key"] = _text(record.get("family_key"))
        record["disposition"] = _text(record.get("disposition")).upper()
        record["predecessor_checkpoint_id"] = _text(record.get("predecessor_checkpoint_id"))
        record["_known_at"] = _strict_timeline_date(record.get("known_at"), "known_at")
        record["_effective_date"] = _strict_timeline_date(
            record.get("effective_date"), "effective_date"
        )
        disposition_records.append(record)
    by_id = {record["checkpoint_id"]: record for record in disposition_records}
    timeline_by_id = {
        _text(record.get("checkpoint_id")): record
        for record in timeline.to_dict("records")
        if _text(record.get("checkpoint_id"))
    }
    causal_bad: set[str] = set()
    for record in disposition_records:
        checkpoint_id = record["checkpoint_id"]
        timeline_record = timeline_by_id.get(checkpoint_id, {})
        if _bool(timeline_record.get("causal_violation", False)):
            causal_bad.add(checkpoint_id)
        effective_from = _strict_timeline_date(
            timeline_record.get("effective_from"), "effective_from"
        )
        if (
            not pd.isna(record["_known_at"])
            and not pd.isna(effective_from)
            and effective_from < record["_known_at"]
        ):
            causal_bad.add(checkpoint_id)
        source_id = _text(timeline_record.get("state_source_checkpoint_id"))
        source = by_id.get(source_id)
        if record["disposition"] in {
            Disposition.PARSED_STATE.value,
            Disposition.VERIFIED_NOOP.value,
        } and pd.isna(record["_known_at"]):
            causal_bad.add(checkpoint_id)
        if not pd.isna(effective_from):
            valid_source = bool(
                source is not None
                and source["family_key"] == record["family_key"]
                and source["disposition"] == Disposition.PARSED_STATE.value
                and not pd.isna(source["_known_at"])
                and source["_known_at"] <= effective_from
            )
            if not valid_source:
                causal_bad.add(checkpoint_id)
        if record["disposition"] != Disposition.VERIFIED_NOOP.value:
            continue
        predecessor = by_id.get(record["predecessor_checkpoint_id"])
        if predecessor is None:
            counts["orphan_noops"] += 1
            continue
        predecessor_key = _timeline_causal_key(predecessor)
        current_key = _timeline_causal_key(record)
        cross_family = predecessor["family_key"] != record["family_key"]
        later = predecessor_key >= current_key
        if cross_family or later:
            causal_bad.add(checkpoint_id)
        predecessor_timeline = timeline_by_id.get(predecessor["checkpoint_id"], {})
        predecessor_usable = (
            predecessor["disposition"] in {Disposition.PARSED_STATE.value, Disposition.VERIFIED_NOOP.value}
            and not pd.isna(predecessor_timeline.get("effective_from", pd.NaT))
        )
        if cross_family or later or not predecessor_usable:
            counts["orphan_noops"] += 1
    counts["causal_violations"] = len(causal_bad)

    conflict_ids: set[str] = set()
    for record in disposition_records:
        reasons = {
            _text(record.get("failure_reason", "")).upper(),
            _text(record.get("reason", "")).upper(),
        }
        if "SOURCE_CONFLICT" in reasons:
            conflict_ids.add(record["checkpoint_id"])
    for record in timeline.to_dict("records"):
        reasons = {
            _text(record.get("failure_reason", "")).upper(),
            _text(record.get("reason", "")).upper(),
        }
        if "SOURCE_CONFLICT" in reasons:
            conflict_ids.add(_text(record.get("checkpoint_id")))
    counts["unresolved_conflicts"] = len({checkpoint_id for checkpoint_id in conflict_ids if checkpoint_id})

    for raw in timeline.to_dict("records"):
        start = _strict_timeline_date(raw.get("effective_from"), "effective_from")
        end = _strict_timeline_date(raw.get("effective_to"), "effective_to")
        if not pd.isna(start) and not pd.isna(end) and end <= start:
            counts["nonpositive_intervals"] += 1

    critical_by_id = {
        _text(record.get("checkpoint_id")): _bool(record.get("is_critical", False))
        for record in checkpoints.to_dict("records")
    }
    unresolved_ids = {
        record["checkpoint_id"]
        for record in disposition_records
        if record["disposition"] == Disposition.UNRESOLVED.value
    }
    unresolved_ids.update(
        _text(record.get("checkpoint_id"))
        for record in timeline.to_dict("records")
        if _text(record.get("disposition")).upper() == Disposition.UNRESOLVED.value
    )
    for checkpoint_id in unresolved_ids:
        if critical_by_id.get(checkpoint_id, False):
            counts["critical_unresolved"] += 1
    return counts
