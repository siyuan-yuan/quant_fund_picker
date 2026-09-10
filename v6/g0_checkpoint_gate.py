"""Reproducible V6-G0-A1 classification-checkpoint gate and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from v6.classification_checkpoints import (
    Disposition,
    adjudicate_checkpoints,
    build_clause_clusters,
    build_checkpoint_timeline,
    freeze_checkpoint_manifest,
    generate_required_checkpoints,
    normalize_investment_clause,
    validate_checkpoint_invariants,
)
from v6.csrc_fund_disclosure import (
    assess_document_applicability,
    extract_effective_date,
    extract_equity_constraint,
    extract_equity_constraint_candidates,
    extract_investment_sections,
    extract_pdf_text,
    locate_evidence_page,
    proposed_state_is_pending,
    text_extraction_is_usable,
)


PROTOCOL_VERSION = "V6-G0-A1"
_STRATA = (
    ("存续", "pre2013"),
    ("存续", "2013_2019"),
    ("存续", "2020_2026"),
    ("清盘", "pre2013"),
    ("清盘", "2013_2019"),
    ("清盘", "2020_2026"),
)
_ARTIFACT_FILES = {
    "required_classification_checkpoints": "required_classification_checkpoints.csv",
    "document_dispositions": "document_dispositions.csv",
    "classification_clause_clusters": "classification_clause_clusters.csv",
    "classification_state_timeline": "classification_state_timeline.csv",
    "g0_amendment_a1_gate_metrics": "g0_amendment_a1_gate_metrics.csv",
    "g0_amendment_a1_gate_review": "g0_amendment_a1_gate_review.md",
}
_SECTION_CACHE_SCHEMA_VERSION = 1
_SECTION_EXTRACTOR_VERSION = "structured-section-v3"
_SAFE_SHARE_CODE = re.compile(r"\d+(?:\.0)?\Z")
_SAFE_UPLOAD_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


def _stop_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _isolated_worker(connection: Any, function: Any, args: tuple[object, ...]) -> None:
    try:
        connection.send((True, function(*args)))
    except BaseException as exc:  # transported to the parent as data
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _run_isolated(
    function: Any, args: tuple[object, ...], *, timeout_seconds: float
) -> object:
    """Run one potentially hostile PDF operation in a killable Windows process."""

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_worker, args=(child, function, args))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            _stop_process(process)
            raise TimeoutError(f"isolated operation timed out after {timeout_seconds:g}s")
        succeeded, value = parent.recv()
        process.join(timeout=1.0)
        if not succeeded:
            raise RuntimeError(str(value))
        return value
    finally:
        parent.close()
        _stop_process(process)


def _isolated_worker_loop(connection: Any, function: Any) -> None:
    try:
        while True:
            args = connection.recv()
            if args is None:
                return
            try:
                connection.send((True, function(*args)))
            except BaseException as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class _IsolatedWorker:
    """Reuse one spawn process, while retaining per-call killable timeouts."""

    def __init__(self, function: Any):
        self._context = multiprocessing.get_context("spawn")
        self._function = function
        self._start()

    def _start(self) -> None:
        self._parent, child = self._context.Pipe(duplex=True)
        self._process = self._context.Process(
            target=_isolated_worker_loop, args=(child, self._function)
        )
        self._process.start()
        child.close()

    def call(self, args: tuple[object, ...], *, timeout_seconds: float) -> object:
        try:
            self._parent.send(args)
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._restart()
            raise RuntimeError("isolated worker terminated unexpectedly") from exc
        if not self._parent.poll(timeout_seconds):
            self._restart()
            raise TimeoutError(f"isolated operation timed out after {timeout_seconds:g}s")
        try:
            succeeded, value = self._parent.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._restart()
            raise RuntimeError("isolated worker terminated unexpectedly") from exc
        if not succeeded:
            raise RuntimeError(str(value))
        return value

    def _terminate(self) -> None:
        _stop_process(self._process)

    def _restart(self) -> None:
        self._terminate()
        self._parent.close()
        self._start()

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._parent.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=1.0)
        self._terminate()
        self._parent.close()

    def __enter__(self) -> "_IsolatedWorker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class GateInputs:
    metadata_jsonl: Path
    parsed_documents_jsonl: Path | None
    pdf_root: Path
    family_sample_csv: Path
    aliases_csv: Path
    legacy_parse_failures_csv: Path | None


@dataclass
class GateResult:
    passed: bool
    metrics: pd.DataFrame
    checkpoints: pd.DataFrame
    dispositions: pd.DataFrame
    clusters: pd.DataFrame
    timeline: pd.DataFrame
    review_markdown: str

    def metric(self, name: str) -> pd.Series:
        """Return one named metric row, raising for absent or duplicate names."""

        matches = self.metrics.loc[self.metrics["metric"].eq(name)]
        if len(matches) != 1:
            raise KeyError(f"expected one metric named {name!r}, found {len(matches)}")
        return matches.iloc[0]


def _clean_text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _first_nonblank(*values: object) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def _stable_json_scalar(value: object) -> str | int | float | bool | None:
    """Convert supported metadata scalars into deterministic JSON values.

    Date-like values use ISO calendar dates because gate metadata is date-granular.
    Unknown values deliberately raise rather than relying on ``json.dumps(default=str)``.
    """

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return str(value.astype("datetime64[D]"))
    if isinstance(value, np.generic):
        unwrapped = value.item()
        if isinstance(unwrapped, np.generic):
            raise TypeError(f"unsupported JSON scalar: {type(value).__name__}")
        return _stable_json_scalar(unwrapped)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise TypeError("unsupported JSON scalar: non-finite float")
        return value
    raise TypeError(f"unsupported JSON scalar: {type(value).__name__}")


def _stable_json_value(value: object) -> object:
    """Normalize cache/fingerprint JSON values without silently stringifying types."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported JSON object key: {type(key).__name__}")
            normalized[key] = _stable_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    return _stable_json_scalar(value)


def _first_stable_json_scalar(*values: object) -> str | int | float | bool | None:
    """Select the first nonblank value after applying JSON-scalar normalization."""

    for value in values:
        normalized = _stable_json_scalar(value)
        if normalized is None:
            continue
        if isinstance(normalized, str):
            cleaned = normalized.strip()
            if cleaned and cleaned.lower() != "nan":
                return cleaned
            continue
        return normalized
    return ""


def _code(value: object) -> str:
    text = _clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text else ""


def _document_identity(record: Mapping[str, object]) -> tuple[str, str, str]:
    raw_code = _first_nonblank(record.get("fundCode"), record.get("share_code"))
    upload_id = _first_nonblank(record.get("uploadInfoId"), record.get("upload_info_id"))
    if not raw_code or not upload_id:
        return "", "", ""
    if _SAFE_SHARE_CODE.fullmatch(raw_code) is None or _SAFE_UPLOAD_ID.fullmatch(upload_id) is None:
        raise ValueError("unsafe metadata document token")
    share_code = _code(raw_code)
    if re.fullmatch(r"\d{6}", share_code) is None:
        raise ValueError("unsafe metadata document token")
    return share_code, upload_id, f"{share_code}_{upload_id}.pdf"


def _pdf_path(pdf_root: Path, source_document: str) -> Path:
    root = pdf_root.resolve()
    candidate = (root / source_document).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("PDF path escapes pdf_root") from exc
    return candidate


def _metric_row(
    metric: str,
    *,
    role: str,
    threshold: str,
    numerator: int | float,
    denominator: int | float,
    actual: float,
    passed: bool,
    failures: int | float = 0,
) -> dict[str, object]:
    return {
        "gate": "G0(c)",
        "metric": metric,
        "role": role,
        "threshold": threshold,
        "numerator": numerator,
        "denominator": denominator,
        "actual": actual,
        "result": "PASS" if passed else "FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "failures": failures,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _resolved_mask(frame: pd.DataFrame) -> pd.Series:
    if "disposition" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["disposition"].astype(str).isin(
        {Disposition.PARSED_STATE.value, Disposition.VERIFIED_NOOP.value}
    )


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(
        lambda value: value is True or _clean_text(value).lower() in {"1", "true", "yes", "y", "是"}
    )


def _covered_family_keys(timeline: pd.DataFrame, aliases: pd.DataFrame) -> set[str]:
    if timeline.empty or "effective_from" not in timeline:
        return set()
    usable = timeline.loc[timeline["effective_from"].notna()].copy()
    usable = usable.loc[usable["effective_from"].map(_clean_text).ne("")]
    if "disposition" in usable:
        usable = usable.loc[_resolved_mask(usable)]
    covered = (
        set(usable["family_key"].map(_clean_text))
        if "family_key" in usable
        else set()
    )
    covered.discard("")
    if "share_code" not in usable or "family_key" not in aliases:
        return covered
    alias_code_column = next(
        (name for name in ("query_code", "share_code", "fund_code") if name in aliases), None
    )
    if alias_code_column is None:
        return covered
    family_by_code = {
        _code(record[alias_code_column]): _clean_text(record["family_key"])
        for record in aliases.to_dict("records")
        if _code(record[alias_code_column]) and _clean_text(record["family_key"])
    }
    covered.update(
        family_by_code.get(_code(value), "") for value in usable["share_code"].tolist()
    )
    covered.discard("")
    return covered


def _sample_families(family_sample: pd.DataFrame) -> pd.DataFrame:
    required = {"family_key", "status", "inception_era"}
    missing = required - set(family_sample.columns)
    if missing:
        raise ValueError(f"family sample missing columns: {sorted(missing)}")
    sample = family_sample.copy()
    for column in required:
        sample[column] = sample[column].map(_clean_text)
    sample = sample.loc[sample["family_key"].ne("")]
    conflicts = sample.groupby("family_key")[["status", "inception_era"]].nunique()
    if (conflicts > 1).any(axis=None):
        raise ValueError("family sample assigns one family_key to multiple strata")
    return sample.drop_duplicates("family_key", keep="first").reset_index(drop=True)


def evaluate_gate(
    *,
    checkpoints: pd.DataFrame,
    dispositions: pd.DataFrame,
    timeline: pd.DataFrame,
    family_sample: pd.DataFrame,
    aliases: pd.DataFrame,
    invariant_counts: Mapping[str, int],
    legacy_document_counts: Mapping[str, int],
    clusters: pd.DataFrame | None = None,
    pdf_counts: Mapping[str, int] | None = None,
) -> GateResult:
    """Calculate every binding and diagnostic V6-G0-A1 metric."""

    required_invariants = (
        "causal_violations",
        "unresolved_conflicts",
        "nonpositive_intervals",
        "orphan_noops",
        "critical_unresolved",
    )
    absent = set(required_invariants) - set(invariant_counts)
    if absent:
        raise ValueError(f"missing invariant counts: {sorted(absent)}")
    if "checkpoint_id" not in checkpoints or "checkpoint_id" not in dispositions:
        raise ValueError("checkpoints and dispositions require checkpoint_id")
    if checkpoints["checkpoint_id"].astype(str).duplicated().any():
        raise ValueError("duplicate checkpoint_id in frozen checkpoint denominator")

    disposition_by_id = dispositions.drop_duplicates("checkpoint_id", keep=False).set_index(
        "checkpoint_id"
    )
    joined = checkpoints[["checkpoint_id"]].join(
        disposition_by_id[["disposition"]], on="checkpoint_id"
    )
    resolved = int(_resolved_mask(joined).sum())
    checkpoint_total = len(checkpoints)
    checkpoint_actual = _ratio(resolved, checkpoint_total)
    rows = [
        _metric_row(
            "checkpoint_coverage",
            role="PRIMARY",
            threshold=">= 0.95",
            numerator=resolved,
            denominator=checkpoint_total,
            actual=checkpoint_actual,
            passed=checkpoint_total > 0 and checkpoint_actual >= 0.95,
            failures=checkpoint_total - resolved,
        )
    ]

    critical_mask = _bool_series(checkpoints, "is_critical")
    critical_ids = set(checkpoints.loc[critical_mask, "checkpoint_id"].astype(str))
    resolved_ids = set(joined.loc[_resolved_mask(joined), "checkpoint_id"].astype(str))
    critical_total = len(critical_ids)
    critical_resolved = len(critical_ids & resolved_ids)
    critical_actual = _ratio(critical_resolved, critical_total) if critical_total else 1.0
    rows.append(
        _metric_row(
            "critical_checkpoint_resolution",
            role="PRIMARY",
            threshold="== 1.0",
            numerator=critical_resolved,
            denominator=critical_total,
            actual=critical_actual,
            passed=critical_actual == 1.0,
            failures=critical_total - critical_resolved,
        )
    )

    sample = _sample_families(family_sample)
    covered = _covered_family_keys(timeline, aliases)
    overall_numerator = int(sample["family_key"].isin(covered).sum())
    overall_denominator = len(sample)
    overall_actual = _ratio(overall_numerator, overall_denominator)
    rows.append(
        _metric_row(
            "fund_coverage_overall",
            role="PRIMARY",
            threshold=">= 0.90",
            numerator=overall_numerator,
            denominator=overall_denominator,
            actual=overall_actual,
            passed=overall_denominator > 0 and overall_actual >= 0.90,
            failures=overall_denominator - overall_numerator,
        )
    )
    for status, era in _STRATA:
        stratum = sample.loc[sample["status"].eq(status) & sample["inception_era"].eq(era)]
        denominator = len(stratum)
        numerator = int(stratum["family_key"].isin(covered).sum())
        actual = _ratio(numerator, denominator)
        rows.append(
            _metric_row(
                f"coverage_{status}_{era}",
                role="PRIMARY",
                threshold=">= 0.80",
                numerator=numerator,
                denominator=denominator,
                actual=actual,
                passed=denominator > 0 and actual >= 0.80,
                failures=denominator - numerator,
            )
        )

    for name in required_invariants:
        count = int(invariant_counts[name])
        rows.append(
            _metric_row(
                name,
                role="PRIMARY",
                threshold="== 0",
                numerator=count,
                denominator=1,
                actual=float(count),
                passed=count == 0,
                failures=count,
            )
        )

    document_success = int(legacy_document_counts.get("success", 0))
    document_valid = int(legacy_document_counts.get("valid", 0))
    document_failure = int(
        legacy_document_counts.get("failure", max(document_valid - document_success, 0))
    )
    rows.append(
        _metric_row(
            "document_parser_coverage",
            role="DIAGNOSTIC",
            threshold="diagnostic only",
            numerator=document_success,
            denominator=document_valid,
            actual=_ratio(document_success, document_valid),
            passed=True,
            failures=document_failure,
        )
    )
    pdf = dict(pdf_counts or {})
    available = int(pdf.get("available", 0))
    required = int(pdf.get("required", 0))
    missing = int(pdf.get("missing", max(required - available, 0)))
    rows.append(
        _metric_row(
            "pdf_acquisition_coverage",
            role="DIAGNOSTIC",
            threshold="diagnostic only",
            numerator=available,
            denominator=required,
            actual=_ratio(available, required),
            passed=True,
            failures=missing,
        )
    )
    rows.append(
        _metric_row(
            "historical_missing_pdf_count",
            role="DIAGNOSTIC",
            threshold="diagnostic only",
            numerator=1,
            denominator=1,
            actual=1.0,
            passed=True,
            failures=1,
        )
    )

    metrics = pd.DataFrame(rows)
    passed = bool(
        metrics.loc[metrics["role"].eq("PRIMARY"), "result"].eq("PASS").all()
    )
    review_lines = [
        f"# G0 Amendment A1 Gate Review ({PROTOCOL_VERSION})",
        "",
        "| Role | Metric | Threshold | Numerator | Denominator | Actual | Failures | Result |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.to_dict("records"):
        review_lines.append(
            f"| {row['role']} | {row['metric']} | {row['threshold']} | "
            f"{row['numerator']} | {row['denominator']} | {row['actual']:.10g} | "
            f"{row['failures']} | {row['result']} |"
        )
    review_lines.extend(
        [
            "",
            "Historical missing PDF remains disclosed: `162703/420632`.",
            "",
            f"FINAL: G0 {'PASS' if passed else 'FAIL'} under {PROTOCOL_VERSION}",
            "",
        ]
    )
    return GateResult(
        passed=passed,
        metrics=metrics,
        checkpoints=checkpoints.copy(),
        dispositions=dispositions.copy(),
        clusters=(
            pd.DataFrame(
                columns=[
                    "normalized_clause",
                    "normalization_version",
                    "clause_sha256",
                    "cluster_id",
                    "cluster_member_count",
                ]
            )
            if clusters is None
            else clusters.copy()
        ),
        timeline=timeline.copy(),
        review_markdown="\n".join(review_lines),
    )


def _read_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            records.append(value)
    return pd.DataFrame(records)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _latest_parsed_documents(parsed: pd.DataFrame) -> pd.DataFrame:
    """Collapse an append-only parser cache by last source occurrence."""

    latest: dict[str, tuple[int, dict[str, object]]] = {}
    unkeyed: list[tuple[int, dict[str, object]]] = []
    for index, record in enumerate(parsed.to_dict("records")):
        source_document = _clean_text(record.get("source_document"))
        if not source_document:
            nested = record.get("document")
            if not isinstance(nested, dict):
                nested = record.get("failure")
            if isinstance(nested, dict):
                source_document = _clean_text(nested.get("source_document"))
                if not source_document:
                    code = _code(_first_nonblank(nested.get("share_code"), nested.get("fundCode")))
                    upload = _first_nonblank(
                        nested.get("upload_info_id"), nested.get("uploadInfoId")
                    )
                    if code and upload:
                        source_document = f"{code}_{upload}.pdf"
        if source_document:
            latest[source_document] = (index, record)
        else:
            unkeyed.append((index, record))
    ordered = sorted([*latest.values(), *unkeyed], key=lambda item: item[0])
    return pd.DataFrame([record for _, record in ordered], columns=parsed.columns)


def _sections_from_parsed(parsed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for outer in parsed.to_dict("records"):
        merged = dict(outer)
        document = outer.get("document")
        failure = outer.get("failure")
        if isinstance(document, dict):
            merged.update(document)
        if isinstance(failure, dict):
            merged.update(failure)
        source_document = _clean_text(merged.get("source_document"))
        evidence = _clean_text(merged.get("evidence"))
        status = _clean_text(outer.get("status")).lower()
        records.append(
            {
                **merged,
                "section_name": "investment_range",
                "section_text": evidence,
                "evidence_location": source_document,
                "source_document": source_document,
                "extraction_status": "success" if status == "success" and evidence else "failure",
            }
        )
    return pd.DataFrame(records)


def _authoritative_alias_family_map(
    aliases: pd.DataFrame, family_sample: pd.DataFrame
) -> dict[str, str]:
    """Map explicit alias codes to exact sampled families, rejecting ambiguity."""

    required_sample = {"share_code", "family_key"}
    missing_sample = required_sample - set(family_sample.columns)
    if missing_sample:
        raise ValueError(f"family sample missing columns: {sorted(missing_sample)}")
    required_aliases = {"query_code", "sample_share_code"}
    missing_aliases = required_aliases - set(aliases.columns)
    if missing_aliases:
        raise ValueError(f"aliases missing columns: {sorted(missing_aliases)}")
    sample_family_by_code: dict[str, set[str]] = {}
    for record in family_sample.to_dict("records"):
        sample_code = _code(record.get("share_code"))
        family_key = _clean_text(record.get("family_key"))
        if sample_code and family_key:
            sample_family_by_code.setdefault(sample_code, set()).add(family_key)
    duplicate_sample = {
        code: sorted(families)
        for code, families in sample_family_by_code.items()
        if len(families) > 1
    }
    if duplicate_sample:
        code = sorted(duplicate_sample)[0]
        raise ValueError(f"ambiguous sample family for {code}: {duplicate_sample[code]}")
    families_by_code: dict[str, set[str]] = {}
    for record in aliases.to_dict("records"):
        code = _code(record.get("query_code"))
        sample_code = _code(record.get("sample_share_code"))
        if not sample_code or sample_code not in sample_family_by_code:
            raise ValueError(
                f"aliases sample_share_code is missing or unknown for query {code or '<blank>'}"
            )
        family_key = next(iter(sample_family_by_code[sample_code]))
        if code:
            families_by_code.setdefault(code, set()).add(family_key)
    ambiguous = {
        code: sorted(families) for code, families in families_by_code.items() if len(families) > 1
    }
    if ambiguous:
        code = sorted(ambiguous)[0]
        raise ValueError(f"ambiguous authoritative alias for {code}: {ambiguous[code]}")
    return {code: next(iter(families)) for code, families in families_by_code.items()}


def _build_lifecycle_evidence(
    sections: pd.DataFrame,
    parsed: pd.DataFrame,
    aliases: pd.DataFrame,
    family_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Build explicit lifecycle evidence only from exactly matching provenance."""

    family_by_code = _authoritative_alias_family_map(aliases, family_sample)
    if sections.empty or parsed.empty:
        return pd.DataFrame()
    parsed_by_provenance: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = {}
    for outer in parsed.to_dict("records"):
        merged = dict(outer)
        nested = outer.get("document")
        if isinstance(nested, dict):
            merged.update(nested)
        raw = _clean_text(merged.get("evidence"))
        source_document = _clean_text(merged.get("source_document"))
        evidence_location = _clean_text(merged.get("evidence_location")) or source_document
        provenance = (
            _code(_first_nonblank(merged.get("share_code"), merged.get("fundCode"))),
            _clean_text(_first_nonblank(merged.get("upload_info_id"), merged.get("uploadInfoId"))),
            _clean_text(merged.get("known_at")),
            source_document,
            evidence_location,
            normalize_investment_clause(raw),
        )
        if all(provenance):
            parsed_by_provenance.setdefault(provenance, []).append(merged)

    candidates: list[dict[str, object]] = []
    for section in sections.to_dict("records"):
        code = _code(_first_nonblank(section.get("share_code"), section.get("fundCode")))
        family_key = family_by_code.get(code, "")
        raw = _clean_text(
            _first_nonblank(section.get("section_text"), section.get("evidence_raw"))
        )
        provenance = (
            code,
            _clean_text(section.get("upload_info_id")),
            _clean_text(section.get("known_at")),
            _clean_text(section.get("source_document")),
            _clean_text(section.get("evidence_location")),
            normalize_investment_clause(raw),
        )
        if not family_key or not all(provenance):
            continue
        for parser_record in parsed_by_provenance.get(provenance, []):
            candidates.append(
                {
                    **section,
                    **parser_record,
                    "family_key": family_key,
                    "upload_info_id": provenance[1],
                    "known_at": provenance[2],
                    "source_document": provenance[3],
                    "evidence_location": provenance[4],
                    "section_text": raw,
                    "normalized_clause": normalize_investment_clause(raw),
                    "parsed_upload_info_id": provenance[1],
                    "parsed_known_at": provenance[2],
                    "parsed_source_document": provenance[3],
                    "parsed_evidence_location": provenance[4],
                    "parsed_evidence_raw": _clean_text(parser_record.get("evidence")),
                    "parsed_normalized_clause": provenance[5],
                }
            )
    return pd.DataFrame(candidates)


def _atomic_jsonl(records: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    _stable_json_value(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, destination)


def _fingerprint_metadata(
    record: Mapping[str, object], share_code: str, upload_id: str
) -> str:
    relevant = {
        "share_code": share_code,
        "upload_info_id": upload_id,
        "known_at": _first_stable_json_scalar(
            record.get("reportSendDate"), record.get("known_at")
        ),
        "effective_date": _first_stable_json_scalar(
            record.get("effective_date"), record.get("effectiveDate")
        ),
        "report_code": _first_stable_json_scalar(
            record.get("reportCode"), record.get("report_code")
        ),
        "report_name": _first_stable_json_scalar(
            record.get("reportName"), record.get("report_name")
        ),
    }
    canonical = json.dumps(
        _stable_json_value(relevant),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_pdf(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "missing", "size": 0, "mtime_ns": 0, "sha256": ""}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
    except OSError as exc:
        return {
            "status": "unreadable",
            "size": 0,
            "mtime_ns": 0,
            "sha256": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _cache_matches(
    cached: Mapping[str, object], metadata_fingerprint: str, pdf_fingerprint: Mapping[str, object]
) -> bool:
    return (
        cached.get("cache_schema_version") == _SECTION_CACHE_SCHEMA_VERSION
        and cached.get("extractor_version") == _SECTION_EXTRACTOR_VERSION
        and cached.get("metadata_fingerprint") == metadata_fingerprint
        and cached.get("pdf_fingerprint") == dict(pdf_fingerprint)
    )


def _extract_denominator_sections(
    metadata: pd.DataFrame,
    pdf_root: Path,
    *,
    cache_path: Path | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Extract comparison evidence without consulting parser-result artifacts."""

    cached_by_key: dict[str, dict[str, object]] = {}
    if cache_path is not None and cache_path.is_file():
        cached = _read_jsonl(cache_path)
        for cached_record in cached.to_dict("records"):
            key = _clean_text(cached_record.get("document_key"))
            if key:
                cached_by_key[key] = cached_record
    sections: list[dict[str, object]] = []
    records = metadata.to_dict("records")
    total = len(records)
    resumed = 0
    worker: _IsolatedWorker | None = None
    try:
        for index, record in enumerate(records, start=1):
            share_code, upload_id, source_document = _document_identity(record)
            if not share_code or not upload_id:
                continue
            document_key = source_document
            path = _pdf_path(pdf_root, source_document)
            metadata_fingerprint = _fingerprint_metadata(record, share_code, upload_id)
            pdf_fingerprint = _fingerprint_pdf(path)
            cached_record = cached_by_key.get(document_key)
            if cached_record is not None and _cache_matches(
                cached_record, metadata_fingerprint, pdf_fingerprint
            ):
                sections.append(cached_by_key[document_key])
                resumed += 1
                if index == 1 or index % 25 == 0 or index == total:
                    print(
                        f"[denominator-sections] {index}/{total} cached={resumed} "
                        f"latest={source_document} status=cached",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            base = {
                "cache_schema_version": _SECTION_CACHE_SCHEMA_VERSION,
                "extractor_version": _SECTION_EXTRACTOR_VERSION,
                "metadata_fingerprint": metadata_fingerprint,
                "pdf_fingerprint": pdf_fingerprint,
                "document_key": document_key,
                "share_code": share_code,
                "upload_info_id": upload_id,
                "known_at": _first_stable_json_scalar(
                    record.get("reportSendDate"), record.get("known_at")
                ),
                "effective_date": _first_stable_json_scalar(
                    record.get("effective_date"), record.get("effectiveDate")
                ),
                "section_name": "investment_range",
                "source_document": source_document,
                "evidence_location": source_document,
                "document_applicability": "UNDETERMINED",
                "section_heading": "",
                "section_page_start": "",
                "section_raw": "",
                "section_normalized": "",
                "section_locator": "",
                "root_cause_reason": "",
            }
            try:
                if worker is None and timeout_seconds and timeout_seconds > 0:
                    worker = _IsolatedWorker(extract_pdf_text)
                text = (
                    worker.call((path,), timeout_seconds=float(timeout_seconds))
                    if worker is not None
                    else extract_pdf_text(path)
                )
                raw_text = str(text)
                applicability = assess_document_applicability(
                    report_code=_clean_text(_first_stable_json_scalar(
                        record.get("reportCode"), record.get("report_code")
                    )),
                    report_name=_clean_text(_first_stable_json_scalar(
                        record.get("reportName"), record.get("report_name")
                    )),
                    text=raw_text,
                )
                if applicability == "NOT_APPLICABLE":
                    raise LookupError("DOCUMENT_NOT_APPLICABLE")
                if not text_extraction_is_usable(raw_text):
                    raise LookupError("TEXT_EXTRACTION_DAMAGED")
                if proposed_state_is_pending(
                    raw_text,
                    report_name=_clean_text(_first_stable_json_scalar(
                        record.get("reportName"), record.get("report_name")
                    )),
                ):
                    raise LookupError("PROPOSED_STATE_NOT_EFFECTIVE")
                candidates = extract_investment_sections(raw_text)
                parsed_candidates = []
                for candidate in candidates:
                    parsed_candidates.extend(
                        (candidate, constraint)
                        for constraint in extract_equity_constraint_candidates(
                            candidate.normalized_text
                        )
                    )
                if not parsed_candidates:
                    if not candidates:
                        fallback_constraints = extract_equity_constraint_candidates(raw_text)
                        if not fallback_constraints:
                            raise LookupError("INVESTMENT_SECTION_NOT_FOUND")
                        fallback_states = {
                            (item.equity_min_pct, item.equity_max_pct)
                            for item in fallback_constraints
                        }
                        if len(fallback_states) != 1:
                            raise LookupError("EQUITY_CLAUSE_AMBIGUOUS")
                        constraint = fallback_constraints[0]
                        selected = None
                        fallback_page = locate_evidence_page(raw_text, constraint.evidence)
                        if fallback_page is None:
                            raise LookupError("EVIDENCE_LOCATION_UNRESOLVED")
                    else:
                        raise LookupError("EQUITY_CLAUSE_NOT_FOUND")
                else:
                    states = {
                        (item[1].equity_min_pct, item[1].equity_max_pct)
                        for item in parsed_candidates
                    }
                    if len(states) != 1:
                        raise LookupError("EQUITY_CLAUSE_AMBIGUOUS")
                    selected, constraint = parsed_candidates[0]
                effective_date = extract_effective_date(str(text))
            except LookupError as exc:
                root_reason = str(exc)
                section = {
                    **base,
                    "document_applicability": applicability,
                    "section_text": "",
                    "section_comparable": False,
                    "extraction_status": "failure",
                    "failure_reason": "SECTION_EXTRACTION_FAILED",
                    "root_cause_reason": root_reason,
                    "extraction_error": root_reason,
                }
            except Exception as exc:
                section = {
                    **base,
                    "section_text": "",
                    "section_comparable": False,
                    "extraction_status": "failure",
                    "failure_reason": "PDF_MISSING" if not path.is_file() else "SECTION_EXTRACTION_FAILED",
                    "root_cause_reason": "PDF_MISSING" if not path.is_file() else "TEXT_EXTRACTION_FAILED",
                    "extraction_error": str(exc),
                }
            else:
                section_heading = selected.heading if selected is not None else "DOCUMENT_FALLBACK"
                section_page = (
                    selected.page_start
                    if selected is not None
                    else fallback_page
                )
                section_raw = selected.text if selected is not None else constraint.evidence
                section_normalized = (
                    selected.normalized_text if selected is not None else constraint.evidence
                )
                section = {
                    **base,
                    "document_applicability": applicability,
                    "effective_date": _stable_json_scalar(effective_date)
                    if effective_date is not None
                    else base["effective_date"],
                    "section_heading": section_heading,
                    "section_page_start": str(section_page),
                    "section_raw": section_raw,
                    "section_normalized": section_normalized,
                    "section_text": constraint.evidence,
                    "section_locator": (
                        f"{source_document}#page={section_page};heading={section_heading}"
                    ),
                    "section_comparable": True,
                    "extraction_status": "success",
                    "failure_reason": "",
                    "extraction_error": "",
                }
            sections.append(section)
            cached_by_key[document_key] = section
            if cache_path is not None:
                _atomic_jsonl(list(cached_by_key.values()), cache_path)
            if index == 1 or index % 25 == 0 or index == total:
                print(
                    f"[denominator-sections] {index}/{total} cached={resumed} "
                    f"latest={source_document} status={section['extraction_status']}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if worker is not None:
            worker.close()
    return pd.DataFrame(sections)


def _generate_checkpoints(
    metadata: pd.DataFrame,
    aliases: pd.DataFrame,
    family_sample: pd.DataFrame,
    pdf_root: Path,
    *,
    cache_path: Path | None = None,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    sections = _extract_denominator_sections(
        metadata, pdf_root, cache_path=cache_path, timeout_seconds=timeout_seconds
    )
    return generate_required_checkpoints(
        metadata, aliases, sections, family_sample=family_sample
    )


def _manifest_comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
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
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"frozen checkpoint manifest missing columns: {sorted(missing)}")
    out = frame.loc[:, columns].copy()
    for column in columns:
        if column == "share_code":
            out[column] = out[column].map(_code)
        elif column == "is_critical":
            out[column] = _bool_series(out, column).map(lambda value: "true" if value else "false")
        else:
            out[column] = out[column].map(_clean_text)
    return out.sort_values(list(columns), kind="stable").reset_index(drop=True)


def _load_or_validate_frozen(
    generated: pd.DataFrame, frozen_checkpoints: Path | None
) -> pd.DataFrame:
    if frozen_checkpoints is None:
        return generated
    if not frozen_checkpoints.is_file():
        raise FileNotFoundError(f"frozen checkpoint manifest not found: {frozen_checkpoints}")
    frozen = _read_csv(frozen_checkpoints)
    generated_records = _manifest_comparison_frame(generated).to_dict("records")
    frozen_records = _manifest_comparison_frame(frozen).to_dict("records")
    if generated_records != frozen_records:
        raise ValueError("frozen checkpoint manifest mismatch")
    return frozen


def _legacy_counts(
    parsed: pd.DataFrame, metadata: pd.DataFrame, pdf_root: Path
) -> dict[str, int]:
    valid_fa_sources: set[str] = set()
    for record in metadata.to_dict("records"):
        report_code = _first_nonblank(record.get("reportCode"), record.get("report_code"))
        if not report_code.startswith("FA"):
            continue
        code, upload, source_document = _document_identity(record)
        if not code or not upload:
            continue
        if _pdf_path(pdf_root, source_document).is_file():
            valid_fa_sources.add(source_document)
    latest_by_source = {
        _clean_text(record.get("source_document")): record
        for record in parsed.to_dict("records")
        if _clean_text(record.get("source_document"))
    }
    success = sum(
        _clean_text(latest_by_source.get(source, {}).get("status")).lower() == "success"
        for source in valid_fa_sources
    )
    valid = len(valid_fa_sources)
    return {"success": success, "failure": valid - success, "valid": valid}


def _pdf_counts(metadata: pd.DataFrame, root: Path) -> dict[str, int]:
    expected: set[str] = set()
    for record in metadata.to_dict("records"):
        code, upload, source_document = _document_identity(record)
        if code and upload:
            expected.add(source_document)
    available = sum(_pdf_path(root, name).is_file() for name in expected)
    return {
        "available": available,
        "required": len(expected),
        "missing": len(expected) - available,
    }


def run_checkpoint_gate(
    inputs: GateInputs, *, frozen_checkpoints: Path | None = None
) -> GateResult:
    """Load inputs, validate any frozen denominator, and evaluate the amended gate."""

    required_files = (inputs.metadata_jsonl, inputs.family_sample_csv, inputs.aliases_csv)
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not Path(inputs.pdf_root).is_dir():
        raise FileNotFoundError(inputs.pdf_root)
    if inputs.parsed_documents_jsonl is None:
        raise ValueError("parsed_documents_jsonl is required for a normal gate run")
    if not Path(inputs.parsed_documents_jsonl).is_file():
        raise FileNotFoundError(inputs.parsed_documents_jsonl)
    if inputs.legacy_parse_failures_csv is not None and not Path(
        inputs.legacy_parse_failures_csv
    ).is_file():
        raise FileNotFoundError(inputs.legacy_parse_failures_csv)

    metadata = _read_jsonl(Path(inputs.metadata_jsonl))
    sample = _read_csv(Path(inputs.family_sample_csv))
    aliases = _read_csv(Path(inputs.aliases_csv))
    generated = _generate_checkpoints(metadata, aliases, sample, Path(inputs.pdf_root))
    checkpoints = _load_or_validate_frozen(generated, frozen_checkpoints)

    parsed = _latest_parsed_documents(_read_jsonl(Path(inputs.parsed_documents_jsonl)))
    sections = _sections_from_parsed(parsed)
    lifecycle_evidence = _build_lifecycle_evidence(sections, parsed, aliases, sample)
    clusters = build_clause_clusters(sections)
    dispositions = adjudicate_checkpoints(
        checkpoints, sections, parsed, lifecycle_evidence=lifecycle_evidence
    )
    timeline = build_checkpoint_timeline(dispositions)
    invariants = validate_checkpoint_invariants(checkpoints, dispositions, timeline)
    return evaluate_gate(
        checkpoints=checkpoints,
        dispositions=dispositions,
        timeline=timeline,
        family_sample=sample,
        aliases=aliases,
        invariant_counts=invariants,
        legacy_document_counts=_legacy_counts(parsed, metadata, Path(inputs.pdf_root)),
        clusters=clusters,
        pdf_counts=_pdf_counts(metadata, Path(inputs.pdf_root)),
    )


def _replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2**attempt))


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, destination)


def _atomic_text(text: str, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, destination)


def write_gate_artifacts(
    result: GateResult,
    out_dir: str | Path,
    *,
    frozen_checkpoints: str | Path | None = None,
) -> dict[str, Path]:
    """Atomically write the six V6-G0-A1 artifacts below ``out_dir`` only."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: root / filename for key, filename in _ARTIFACT_FILES.items()}
    checkpoint_path = paths["required_classification_checkpoints"]
    authoritative = Path(frozen_checkpoints) if frozen_checkpoints is not None else None
    same_authoritative_path = bool(
        authoritative is not None
        and authoritative.resolve() == checkpoint_path.resolve()
    )
    if same_authoritative_path:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        frozen = _read_csv(checkpoint_path)
        if (
            _manifest_comparison_frame(result.checkpoints).to_dict("records")
            != _manifest_comparison_frame(frozen).to_dict("records")
        ):
            raise ValueError("authoritative frozen checkpoint manifest mismatch")
    else:
        _atomic_csv(result.checkpoints, checkpoint_path)
    _atomic_csv(result.dispositions, paths["document_dispositions"])
    _atomic_csv(result.clusters, paths["classification_clause_clusters"])
    _atomic_csv(result.timeline, paths["classification_state_timeline"])
    _atomic_csv(result.metrics, paths["g0_amendment_a1_gate_metrics"])
    _atomic_text(result.review_markdown, paths["g0_amendment_a1_gate_review"])
    return paths


def _freeze_only(
    inputs: GateInputs,
    out_dir: Path,
    *,
    section_cache: Path | None = None,
    pdf_timeout_seconds: float | None = None,
) -> Path:
    for path in (inputs.metadata_jsonl, inputs.family_sample_csv, inputs.aliases_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not Path(inputs.pdf_root).is_dir():
        raise FileNotFoundError(inputs.pdf_root)
    metadata = _read_jsonl(Path(inputs.metadata_jsonl))
    sample = _read_csv(Path(inputs.family_sample_csv))
    aliases = _read_csv(Path(inputs.aliases_csv))
    print("[freeze] validating sample/alias/lifecycle inputs", file=sys.stderr, flush=True)
    generate_required_checkpoints(metadata, aliases, pd.DataFrame(), family_sample=sample)
    cache = section_cache or out_dir.with_name(f".{out_dir.name}_denominator_sections.jsonl")
    print(f"[freeze] extracting denominator sections cache={cache}", file=sys.stderr, flush=True)
    checkpoints = _generate_checkpoints(
        metadata,
        aliases,
        sample,
        Path(inputs.pdf_root),
        cache_path=cache,
        timeout_seconds=pdf_timeout_seconds,
    )
    destination = out_dir / _ARTIFACT_FILES["required_classification_checkpoints"]
    freeze_checkpoint_manifest(checkpoints, destination)
    restored = _read_csv(destination)
    if (
        _manifest_comparison_frame(checkpoints).to_dict("records")
        != _manifest_comparison_frame(restored).to_dict("records")
    ):
        raise OSError("checkpoint manifest failed durable reload verification")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--parsed-documents", type=Path)
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--family-sample", required=True, type=Path)
    parser.add_argument("--aliases", required=True, type=Path)
    parser.add_argument("--legacy-failures", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frozen-checkpoints", type=Path)
    parser.add_argument("--freeze-checkpoints-only", action="store_true")
    parser.add_argument("--section-cache", type=Path)
    parser.add_argument("--pdf-timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = GateInputs(
        metadata_jsonl=args.metadata,
        parsed_documents_jsonl=args.parsed_documents,
        pdf_root=args.pdf_root,
        family_sample_csv=args.family_sample,
        aliases_csv=args.aliases,
        legacy_parse_failures_csv=args.legacy_failures,
    )
    try:
        if args.freeze_checkpoints_only:
            _freeze_only(
                inputs,
                args.out,
                section_cache=args.section_cache,
                pdf_timeout_seconds=args.pdf_timeout_seconds,
            )
            return 0
        result = run_checkpoint_gate(inputs, frozen_checkpoints=args.frozen_checkpoints)
        write_gate_artifacts(
            result,
            args.out,
            frozen_checkpoints=args.frozen_checkpoints,
        )
        return 0 if result.passed else 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
