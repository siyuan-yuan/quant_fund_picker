"""Task 8 validation cohorts and reviewed A2 denominator identities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import pandas as pd

from .a2_events import classify_event_title


EXTERNAL_COHORT_SEED = "A2-EVENT-OOS-20260909"
EXTERNAL_TITLE_TOKENS = ("转型", "转换", "合并", "变更", "到期", "终止")
CLOSED_EXTERNAL_LABELS = {
    "INCEPTION",
    "LEGAL_TRANSFORMATION",
    "MERGER_OR_SUCCESSION",
    "FUND_TYPE_CHANGE",
    "INVESTMENT_MANDATE_CHANGE",
    "TERMINATION",
    "NOT_EVENT",
    "REVIEW_REQUIRED",
}
EXPECTED_MONTHS = tuple(
    pd.period_range("2006-01", "2026-03", freq="M").astype(str).tolist()
)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _code(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.split(".")[0].zfill(6)


def _date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _source_document(row: dict[str, object]) -> str:
    code = _code(row.get("fundCode") or row.get("share_code"))
    upload = _text(row.get("uploadInfoId") or row.get("upload_info_id"))
    if code and upload:
        return f"{code}_{upload}.pdf"
    return _text(row.get("source_document"))


def _family_by_code(sample: pd.DataFrame, aliases: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in sample.to_dict("records"):
        family = _text(row.get("family_key"))
        code = _code(row.get("share_code"))
        if code and family:
            mapping[code] = family
    for row in aliases.to_dict("records"):
        family = _text(row.get("family_key"))
        code = _code(row.get("query_code") or row.get("sample_share_code"))
        if code and family:
            mapping[code] = family
    return mapping


def _split_role_column(split: pd.DataFrame) -> str:
    for column in ("split_role", "role", "split"):
        if column in split.columns:
            return column
    raise ValueError("split missing split_role")


def _event_decision(row: dict[str, object]) -> tuple[str, str]:
    title = _text(row.get("reportName") or row.get("report_name"))
    decision = classify_event_title(title)
    return decision.status, decision.event_type


def _audit_rows(
    metadata: pd.DataFrame,
    family_map: dict[str, str],
    allowed_families: set[str],
    split_role: str,
) -> list[dict[str, object]]:
    rows = []
    for record in metadata.to_dict("records"):
        code = _code(record.get("fundCode") or record.get("share_code"))
        family = family_map.get(code, "")
        if family not in allowed_families:
            continue
        status, event_type = _event_decision(record)
        source = _source_document(record)
        rows.append(
            {
                "source_document": source,
                "share_code": code,
                "family_key": family,
                "split_role": split_role,
                "title_evidence": _text(record.get("reportName") or record.get("report_name")),
                "decision_status": status,
                "event_type": event_type,
                "event_anchor_date": _date(
                    record.get("event_anchor_date") or record.get("reportSendDate")
                ),
                "rule_revision": "a2-legal-event-v1",
            }
        )
    return rows


def build_development_event_audit(
    sample: pd.DataFrame,
    metadata: pd.DataFrame,
    aliases: pd.DataFrame,
    split: pd.DataFrame,
    sections: pd.DataFrame,
    *,
    allow_validation: bool = True,
) -> pd.DataFrame:
    """Audit event semantics using development families only."""

    role_column = _split_role_column(split)
    roles = split[role_column].map(lambda value: _text(value).lower().replace("_", "-"))
    if not allow_validation and roles.ne("development").any():
        raise ValueError("validation leakage in development event audit")
    development = split.loc[roles.isin({"development", "dev"})]
    families = set(development["family_key"].map(_text))
    rows = _audit_rows(metadata, _family_by_code(sample, aliases), families, "development")
    columns = [
        "source_document",
        "share_code",
        "family_key",
        "split_role",
        "title_evidence",
        "decision_status",
        "event_type",
        "event_anchor_date",
        "rule_revision",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["family_key", "source_document"], kind="mergesort"
    ).reset_index(drop=True)


def freeze_external_event_validation_cohort(
    metadata: pd.DataFrame,
    sample: pd.DataFrame,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    """Mechanically freeze the 120-row out-of-sample title cohort without labels."""

    family_map = _family_by_code(sample, aliases)
    sample_families = set(sample["family_key"].map(_text))
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict("records"):
        code = _code(record.get("fundCode") or record.get("share_code"))
        family = family_map.get(code, "")
        title = _text(record.get("reportName") or record.get("report_name"))
        published = _date(record.get("reportSendDate") or record.get("uploadDate"))
        source = _source_document(record)
        if not source or not title or not published:
            continue
        if published > pd.Timestamp("2026-03-31"):
            continue
        if family in sample_families or code in family_map:
            continue
        if not any(token in title for token in EXTERNAL_TITLE_TOKENS):
            continue
        sort_hash = hashlib.sha256(
            f"{EXTERNAL_COHORT_SEED}|{source}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "external_row_id": sort_hash,
                "canonical_sort_sha256": sort_hash,
                "source_document": source,
                "share_code": code,
                "family_key": family,
                "published_at": str(published.date()),
                "title": title,
                "event_scope": "EXTERNAL",
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("source_document", keep="first")
    if frame.empty:
        raise ValueError("no eligible external event-title rows")
    frame = frame.sort_values(
        ["canonical_sort_sha256", "source_document"], kind="mergesort"
    )
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for row in frame.to_dict("records"):
        code = str(row["share_code"])
        if counts.get(code, 0) >= 3:
            continue
        selected.append(row)
        counts[code] = counts.get(code, 0) + 1
        if len(selected) == 120:
            break
    if len(selected) != 120:
        raise ValueError(f"external event cohort has {len(selected)} eligible rows; expected 120")
    return pd.DataFrame(selected, columns=[
        "external_row_id",
        "canonical_sort_sha256",
        "source_document",
        "share_code",
        "family_key",
        "published_at",
        "title",
        "event_scope",
    ]).reset_index(drop=True)


def validate_external_event_labels(
    frozen_cohort: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Validate one closed-taxonomy label pass without changing event rules."""

    required = {"external_row_id", "label"}
    if not required.issubset(labels.columns):
        raise ValueError("external labels missing external_row_id or label")
    expected = set(frozen_cohort["external_row_id"].astype(str))
    actual = set(labels["external_row_id"].astype(str))
    if actual != expected or len(labels) != len(frozen_cohort):
        raise ValueError("external labels must exactly cover the frozen cohort")
    if not bool(labels["label"].astype(str).isin(CLOSED_EXTERNAL_LABELS).all()):
        raise ValueError("external labels must use the closed taxonomy")
    merged = frozen_cohort.merge(
        labels[["external_row_id", "label"]], on="external_row_id", how="left", validate="one_to_one"
    )
    decisions = merged["title"].map(classify_event_title)
    merged["predicted_status"] = [decision.status for decision in decisions]
    merged["predicted_event_type"] = [decision.event_type for decision in decisions]
    merged["is_predicted_event"] = merged["predicted_status"].eq("EVENT")
    merged["is_labelled_event"] = merged["label"].isin(CLOSED_EXTERNAL_LABELS - {"NOT_EVENT", "REVIEW_REQUIRED"})
    true_positive = int((merged["is_predicted_event"] & merged["is_labelled_event"]).sum())
    predicted_positive = int(merged["is_predicted_event"].sum())
    labelled_positive = int(merged["is_labelled_event"].sum())
    precision = Fraction(true_positive, predicted_positive) if predicted_positive else Fraction(0, 1)
    recall = Fraction(true_positive, labelled_positive) if labelled_positive else Fraction(0, 1)
    merged["precision"] = f"{precision.numerator}/{precision.denominator}"
    merged["recall"] = f"{recall.numerator}/{recall.denominator}"
    return merged


def build_family_validation_audit(
    sample: pd.DataFrame,
    metadata: pd.DataFrame,
    aliases: pd.DataFrame,
    split: pd.DataFrame,
    sections: pd.DataFrame,
) -> pd.DataFrame:
    """Audit the 36-family side without deriving rules from its titles."""

    role_column = _split_role_column(split)
    roles = split[role_column].map(lambda value: _text(value).lower().replace("_", "-"))
    validation = split.loc[roles.str.contains("validation", na=False)].copy()
    if validation.empty:
        raise ValueError("no validation families in split")
    family_map = _family_by_code(sample, aliases)
    metadata_rows = _audit_rows(
        metadata,
        family_map,
        set(validation["family_key"].map(_text)),
        "untouched_validation",
    )
    by_family = pd.DataFrame(metadata_rows).groupby("family_key") if metadata_rows else None
    rows = []
    for family in validation["family_key"].drop_duplicates().map(_text):
        count = 0 if by_family is None or family not in by_family.groups else len(by_family.get_group(family))
        rows.append(
            {
                "family_key": family,
                "split_role": "untouched_validation",
                "metadata_rows": count,
                "event_count": count,
                "title_semantics_exposed_during_a1": True,
                "new_rule_from_validation": False,
                "validation_note": "family/section/timeline validation only; title metadata was previously exposed",
            }
        )
    return pd.DataFrame(rows).sort_values("family_key", kind="mergesort").reset_index(drop=True)


def _stable_bytes(frame: pd.DataFrame) -> bytes:
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: "<NA>" if pd.isna(value) else str(value)
        )
    if not normalized.empty:
        normalized = normalized.sort_values(list(normalized.columns), kind="mergesort").reset_index(drop=True)
    return normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _hash_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_stable_bytes(frame)).hexdigest()


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_reviewed_denominators(
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    observations: pd.DataFrame,
    split: pd.DataFrame,
    schedule: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Atomically freeze only reviewed event and observation identities."""

    if len(schedule) != 243 or "decision_date" not in schedule.columns:
        raise ValueError("denominator freeze requires 243 monthly decision dates")
    dates = pd.to_datetime(schedule["decision_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.dt.to_period("M").astype(str).tolist() != list(EXPECTED_MONTHS):
        raise ValueError("denominator freeze requires ordered 243 monthly decision dates")
    if "family_key" not in split.columns:
        raise ValueError("split missing family_key")
    strata = (
        split["stratum"] if "stratum" in split.columns
        else split["status"].astype(str) + "|" + split["inception_era"].astype(str)
        if {"status", "inception_era"}.issubset(split.columns)
        else None
    )
    if strata is None or split["family_key"].nunique() != 60 or strata.astype(str).nunique() != 6:
        raise ValueError("denominator freeze requires exactly 60 families and six strata")
    required_event_columns = {
        "event_id", "family_key", "event_type", "legal_subject", "event_anchor_date", "generation_version"
    }
    if not required_event_columns.issubset(events.columns):
        raise ValueError("event identity columns are incomplete")
    if events["event_id"].duplicated().any() or events["event_id"].isna().any():
        raise ValueError("event identities must be unique")
    if "source_document" in events.columns:
        raise ValueError("source_document is forbidden in event identity")
    if not {"event_id", "source_document"}.issubset(evidence.columns):
        raise ValueError("evidence candidates require event_id and source_document")
    if evidence[["event_id", "source_document"]].duplicated().any():
        raise ValueError("duplicate event evidence candidate")
    if not set(evidence["event_id"]).issubset(set(events["event_id"])):
        raise ValueError("evidence candidate references unknown event")
    if not {"family_key", "decision_date", "active_eligibility"}.issubset(observations.columns):
        raise ValueError("decision observation identity columns are incomplete")
    identity = observations[["family_key", "decision_date", "active_eligibility"]].copy()
    identity["decision_date"] = pd.to_datetime(identity["decision_date"], errors="coerce").dt.normalize()
    if identity[["family_key", "decision_date"]].duplicated().any():
        raise ValueError("decision observation identities must be unique")
    expected_pairs = {
        (str(family), pd.Timestamp(date))
        for family in split["family_key"].astype(str).unique()
        for date in dates
    }
    actual_pairs = set(zip(identity["family_key"].astype(str), identity["decision_date"]))
    if actual_pairs != expected_pairs:
        raise ValueError("decision observations must enumerate every family and decision date")

    event_identity = events[[
        "event_id", "family_key", "event_type", "legal_subject", "event_anchor_date", "generation_version"
    ]].copy()
    evidence_identity = evidence.copy()
    observation_identity = identity.sort_values(["family_key", "decision_date"], kind="mergesort").reset_index(drop=True)
    frames = {
        "legal_state_events.csv": event_identity,
        "event_evidence_candidates.csv": evidence_identity,
        "decision_observation_ids.csv": observation_identity,
    }
    manifest_payload = {
        "schema_version": "a2-denominator-manifest-v1",
        "schedule_version": "monthly-sse-last-open-v1",
        "decision_start": "2006-01",
        "decision_end": "2026-03",
        "decision_date_count": 243,
        "family_count": 60,
        "stratum_count": 6,
        "event_count": len(event_identity),
        "evidence_candidate_count": len(evidence_identity),
        "observation_count": len(observation_identity),
        "sha256": {name: _hash_frame(frame) for name, frame in frames.items()},
    }
    manifest = (json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    files = {name: frame.to_csv(index=False, lineterminator="\n").encode("utf-8") for name, frame in frames.items()}
    files["a2_denominator_manifest.sha256"] = manifest
    target = Path(output_dir)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"denominator target is not a directory: {target}")
        for name, payload in files.items():
            path = target / name
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError("denominator freeze hash mismatch")
        if {path.name for path in target.iterdir() if path.is_file()} != set(files):
            raise ValueError("denominator freeze artifact set mismatch")
        return {name: target / name for name in files}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        for name, payload in files.items():
            _write_fsync(temporary / name, payload)
        _fsync_dir(temporary)
        os.replace(temporary, target)
        _fsync_dir(target.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: target / name for name in files}
