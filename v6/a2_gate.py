"""Deterministic V6-G0-A2 Gate metrics and append-only artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import pandas as pd


CRITICAL_EVENT_TYPES = frozenset(
    {
        "INCEPTION",
        "LEGAL_TRANSFORMATION",
        "MERGER_OR_SUCCESSION",
        "FUND_TYPE_CHANGE",
        "INVESTMENT_MANDATE_CHANGE",
        "TERMINATION",
    }
)
RESOLVED_RESULTS = frozenset({"NEW_STATE", "VERIFIED_CONTINUITY", "TERMINATED"})


@dataclass(frozen=True)
class GateMetric:
    """One exact integer ratio evaluated against an exact integer threshold."""

    name: str
    numerator: int
    denominator: int
    threshold: Fraction
    result: str
    role: str = "PRIMARY"

    @property
    def fraction(self) -> Fraction:
        if self.denominator == 0:
            return Fraction(0, 1)
        return Fraction(self.numerator, self.denominator)

    @property
    def passed(self) -> bool:
        return self.result == "PASS"


@dataclass
class A2GateResult:
    events: pd.DataFrame
    dispositions: pd.DataFrame
    timeline: pd.DataFrame
    observations: pd.DataFrame
    split: pd.DataFrame
    passed: bool
    event_coverage: GateMetric
    critical_events: GateMetric
    decision_coverage: GateMetric
    family_coverage: GateMetric
    stratum_coverages: tuple[GateMetric, ...]
    invariant_metrics: dict[str, GateMetric]
    invariants: dict[str, int]
    metrics: pd.DataFrame
    review_markdown: str
    input_sha256: str
    p0_invariants_passed: bool
    temporal_invariants_passed: bool
    frozen_manifest_path: Path | None = None
    frozen_manifest_bytes: bytes | None = None
    frozen_denominator_sha256: str | None = None

    @property
    def event_coverage_passed(self) -> bool:
        return self.event_coverage.passed

    @property
    def critical_events_passed(self) -> bool:
        return self.critical_events.passed

    @property
    def decision_coverage_passed(self) -> bool:
        return self.decision_coverage.passed

    @property
    def family_coverage_passed(self) -> bool:
        return self.family_coverage.passed

    @property
    def strata_passed(self) -> bool:
        return all(metric.passed for metric in self.stratum_coverages)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date_present(value: object) -> bool:
    return pd.notna(pd.to_datetime(value, errors="coerce"))


def _stable_frame_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame canonically, including columns and row order."""

    if frame is None:
        return b"<none>\n"
    if frame.empty:
        return ("columns=" + json.dumps(sorted(map(str, frame.columns))) + "\n").encode()
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: "<NA>" if pd.isna(value) else str(value)
        )
    normalized = normalized.sort_values(
        list(normalized.columns), kind="mergesort", na_position="first"
    ).reset_index(drop=True)
    return normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _input_hash(
    events: pd.DataFrame,
    dispositions: pd.DataFrame,
    timeline: pd.DataFrame,
    observations: pd.DataFrame,
    split: pd.DataFrame,
) -> str:
    parts = (
        _stable_frame_bytes(events),
        _stable_frame_bytes(dispositions),
        _stable_frame_bytes(timeline),
        _stable_frame_bytes(observations),
        _stable_frame_bytes(split),
    )
    return _sha256(b"".join(len(part).to_bytes(8, "big") + part for part in parts))


def _event_keys(events: pd.DataFrame) -> list[str]:
    if "event_id" in events.columns:
        keys = events["event_id"].map(_text).tolist()
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("events must contain unique non-empty event_id values")
        return keys
    return [f"row-{index}" for index in range(len(events))]


def _disposition_is_resolved(group: pd.DataFrame) -> bool:
    return (
        not group.empty
        and group.get("result", pd.Series(dtype=str))
        .astype(str)
        .isin(RESOLVED_RESULTS)
        .any()
    )


def _metric(
    name: str,
    numerator: int,
    denominator: int,
    threshold: Fraction,
    *,
    role: str = "PRIMARY",
) -> GateMetric:
    result = (
        "PASS"
        if denominator > 0 and Fraction(numerator, denominator) >= threshold
        else "FAIL"
    )
    return GateMetric(name, int(numerator), int(denominator), threshold, result, role)


def _numeric_or_boolean_count(frame: pd.DataFrame, columns: Iterable[str]) -> int:
    total = 0
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            total += int(numeric.fillna(0).clip(lower=0).sum())
        else:
            total += int(
                series.map(
                    lambda value: _text(value).lower()
                    in {"1", "true", "yes", "y"}
                ).sum()
            )
    return total


def _invariant_counts(
    dispositions: pd.DataFrame,
    timeline: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[dict[str, int], int]:
    causal = _numeric_or_boolean_count(
        dispositions, ("causal_violation", "causal_violations", "causal_violation_count")
    ) + _numeric_or_boolean_count(
        timeline, ("causal_violation", "causal_violations", "causal_violation_count")
    )
    nonpositive = _numeric_or_boolean_count(
        dispositions,
        ("nonpositive_interval", "nonpositive_intervals", "nonpositive_interval_count"),
    ) + _numeric_or_boolean_count(
        timeline,
        ("nonpositive_interval", "nonpositive_intervals", "nonpositive_interval_count"),
    )
    if {"usable_from", "valid_to"}.issubset(timeline.columns):
        starts = pd.to_datetime(timeline["usable_from"], errors="coerce")
        ends = pd.to_datetime(timeline["valid_to"], errors="coerce")
        nonpositive += int((starts.notna() & ends.notna() & ends.le(starts)).sum())

    conflict_disposition = 0
    if "failure_reason" in dispositions.columns:
        conflict_disposition = int(
            dispositions["failure_reason"].astype(str)
            .eq("UNRESOLVED_EVENT_CONFLICT")
            .sum()
        )
    conflict_timeline = 0
    if "timeline_status" in timeline.columns:
        conflict_timeline = int(
            timeline["timeline_status"].astype(str)
            .eq("UNRESOLVED_EVENT_CONFLICT")
            .sum()
        )
    conflicts = max(conflict_disposition, conflict_timeline)
    conflicts = max(
        conflicts,
        _numeric_or_boolean_count(
            dispositions,
            ("unresolved_conflict", "unresolved_conflicts", "unresolved_conflict_count"),
        ),
        _numeric_or_boolean_count(
            timeline,
            ("unresolved_conflict", "unresolved_conflicts", "unresolved_conflict_count"),
        ),
    )

    future = _numeric_or_boolean_count(
        observations,
        (
            "future_evidence_violation",
            "future_evidence_violations",
            "future_evidence_violation_count",
        ),
    )
    if {"decision_date", "known_at"}.issubset(observations.columns):
        decision = pd.to_datetime(observations["decision_date"], errors="coerce")
        known = pd.to_datetime(observations["known_at"], errors="coerce")
        future += int((decision.notna() & known.notna() & known.gt(decision)).sum())
    if {"decision_date", "legal_effective_from"}.issubset(observations.columns):
        decision = pd.to_datetime(observations["decision_date"], errors="coerce")
        effective = pd.to_datetime(
            observations["legal_effective_from"], errors="coerce"
        )
        future += int(
            (decision.notna() & effective.notna() & effective.gt(decision)).sum()
        )
    return {
        "causal_violations": causal,
        "nonpositive_intervals": nonpositive,
        "unresolved_conflicts": conflicts,
    }, future


def _split_stratum_column(split: pd.DataFrame) -> pd.Series:
    if "stratum" in split.columns:
        return split["stratum"].map(_text)
    if "stratum_key" in split.columns:
        return split["stratum_key"].map(_text)
    if {"status", "inception_era"}.issubset(split.columns):
        return split["status"].map(_text) + "|" + split["inception_era"].map(_text)
    raise ValueError("split must contain stratum or status/inception_era")


def _metrics_frame(metrics: Iterable[GateMetric]) -> pd.DataFrame:
    rows = []
    for item in metrics:
        rows.append(
            {
                "metric": item.name,
                "numerator": item.numerator,
                "denominator": item.denominator,
                "threshold_numerator": item.threshold.numerator,
                "threshold_denominator": item.threshold.denominator,
                "fraction": f"{item.fraction.numerator}/{item.fraction.denominator}",
                "result": item.result,
                "role": item.role,
            }
        )
    return pd.DataFrame(rows)


def compute_a2_gate(
    events: pd.DataFrame,
    dispositions: pd.DataFrame,
    timeline: pd.DataFrame,
    observations: pd.DataFrame,
    split: pd.DataFrame,
) -> A2GateResult:
    """Compute all A2 gates with integer counters and fail-closed ratios."""

    for name, frame in {
        "events": events,
        "dispositions": dispositions,
        "timeline": timeline,
        "observations": observations,
        "split": split,
    }.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
    event_frame = events.copy(deep=True)
    disposition_frame = dispositions.copy(deep=True)
    timeline_frame = timeline.copy(deep=True)
    observation_frame = observations.copy(deep=True)
    split_frame = split.copy(deep=True)
    keys = _event_keys(event_frame)

    if "event_id" in disposition_frame.columns:
        by_event = {
            _text(event_id): group
            for event_id, group in disposition_frame.groupby("event_id", sort=False)
        }
    else:
        by_event = {}
    resolved_ids = {
        event_id
        for event_id, group in by_event.items()
        if _disposition_is_resolved(group)
    }
    event_coverage = _metric(
        "event_coverage",
        sum(key in resolved_ids for key in keys),
        len(keys),
        Fraction(95, 100),
    )

    critical_positions = []
    for index, event in event_frame.iterrows():
        event_type = _text(event.get("event_type"))
        if event_type not in CRITICAL_EVENT_TYPES:
            continue
        if event_type == "TERMINATION" and not (
            _date_present(event.get("event_anchor_date"))
            or _date_present(event.get("legal_effective_from"))
        ):
            continue
        critical_positions.append(index)
    critical_keys = [keys[index] for index in critical_positions]
    critical_resolved = sum(key in resolved_ids for key in critical_keys)
    critical_events = _metric(
        "critical_events", critical_resolved, len(critical_keys), Fraction(1, 1)
    )

    active_mask = (
        observation_frame["active_eligibility"].astype(bool)
        if "active_eligibility" in observation_frame.columns
        else pd.Series(True, index=observation_frame.index)
    )
    active = observation_frame.loc[active_mask]
    observed = int(
        active.get(
            "observation_status", pd.Series(index=active.index, dtype=str)
        )
        .astype(str)
        .eq("OBSERVED_STATE")
        .sum()
    )
    decision_coverage = _metric(
        "decision_observation_coverage", observed, len(active), Fraction(95, 100)
    )

    if "family_key" not in split_frame.columns:
        raise ValueError("split must contain family_key")
    split_frame["_stratum"] = _split_stratum_column(split_frame)
    family_rows = split_frame[["family_key", "_stratum"]].drop_duplicates()
    family_keys = set(family_rows["family_key"].map(_text)) - {""}
    covered_families: set[str] = set()
    if "family_key" in timeline_frame.columns:
        state_mask = timeline_frame.get(
            "timeline_status", pd.Series("OBSERVED_STATE", index=timeline_frame.index)
        ).astype(str).eq("OBSERVED_STATE")
        covered_families = set(
            timeline_frame.loc[state_mask, "family_key"].map(_text)
        ) - {""}
    family_coverage = _metric(
        "family_coverage",
        len(family_keys & covered_families),
        len(family_keys),
        Fraction(90, 100),
    )

    stratum_coverages: list[GateMetric] = []
    for stratum, group in family_rows.groupby("_stratum", sort=True):
        families = set(group["family_key"].map(_text)) - {""}
        stratum_coverages.append(
            _metric(
                f"stratum_coverage_{stratum}",
                len(families & covered_families),
                len(families),
                Fraction(80, 100),
            )
        )
    if len(stratum_coverages) != 6:
        raise ValueError("A2 split must contain exactly six strata")

    invariants, future_count = _invariant_counts(
        disposition_frame, timeline_frame, observation_frame
    )
    invariant_metrics = {
        name: GateMetric(
            name,
            0 if count == 0 else 1,
            1,
            Fraction(1, 1),
            "PASS" if count == 0 else "FAIL",
            "P0",
        )
        for name, count in invariants.items()
    }
    temporal_invariants_passed = future_count == 0
    all_metrics = [
        event_coverage,
        critical_events,
        family_coverage,
        *stratum_coverages,
        decision_coverage,
        *invariant_metrics.values(),
        GateMetric(
            "future_evidence_violations",
            0 if temporal_invariants_passed else 1,
            1,
            Fraction(1, 1),
            "PASS" if temporal_invariants_passed else "FAIL",
            "P0",
        ),
    ]
    metrics = _metrics_frame(all_metrics)
    p0_passed = all(metric.passed for metric in invariant_metrics.values())
    passed = all(metric.passed for metric in all_metrics)
    verdict = "PASS" if passed else "FAIL"
    review_lines = [
        "# V6-G0-A2 Observable Legal-State Gate",
        "",
        f"## Verdict: A2 {verdict}",
        "",
        "A1/v2 remains immutable: the first formal A1 verdict is `G0 FAIL` "
        "with 1,371 frozen checkpoints.",
        "",
        "| metric | numerator | denominator | threshold | result |",
        "|---|---:|---:|---:|---|",
    ]
    for item in all_metrics:
        review_lines.append(
            f"| {item.name} | {item.numerator} | {item.denominator} | "
            f"{item.threshold.numerator}/{item.threshold.denominator} | {item.result} |"
        )
    review_lines.extend(
        [
            "",
            "No A1 artifact or downstream G1-G4 result is modified by A2.",
        ]
    )
    split_without_helper = split_frame.drop(columns=["_stratum"])
    result = A2GateResult(
        events=event_frame,
        dispositions=disposition_frame,
        timeline=timeline_frame,
        observations=observation_frame,
        split=split_without_helper,
        passed=passed,
        event_coverage=event_coverage,
        critical_events=critical_events,
        decision_coverage=decision_coverage,
        family_coverage=family_coverage,
        stratum_coverages=tuple(stratum_coverages),
        invariant_metrics=invariant_metrics,
        invariants=invariants,
        metrics=metrics,
        review_markdown="\n".join(review_lines) + "\n",
        input_sha256=_input_hash(
            event_frame,
            disposition_frame,
            timeline_frame,
            observation_frame,
            split_without_helper,
        ),
        p0_invariants_passed=p0_passed,
        temporal_invariants_passed=temporal_invariants_passed,
    )
    return result


def _identity_frame(frame: pd.DataFrame, preferred: tuple[str, ...]) -> pd.DataFrame:
    columns = [column for column in preferred if column in frame.columns]
    if not columns:
        columns = list(frame.columns)
    return frame.loc[:, columns].copy()


def _denominator_frames(result: A2GateResult) -> dict[str, pd.DataFrame]:
    event_identity = _identity_frame(
        result.events,
        (
            "event_id",
            "family_key",
            "event_type",
            "legal_subject",
            "event_anchor_date",
            "generation_version",
        ),
    )
    if "event_id" not in event_identity.columns:
        raise ValueError("events must expose event_id before denominator freeze")
    evidence_columns = [
        column
        for column in (
            "event_id",
            "evidence_candidate_id",
            "source_document",
            "candidate_id",
            "known_at",
            "document_stage",
        )
        if column in result.dispositions.columns
    ]
    if evidence_columns:
        evidence = result.dispositions.loc[:, evidence_columns].copy()
    else:
        evidence = pd.DataFrame(
            {"event_id": result.dispositions.get("event_id", pd.Series(dtype=str))}
        )
    observation_identity = _identity_frame(
        result.observations, ("family_key", "decision_date", "active_eligibility")
    )
    if not {"family_key", "decision_date"}.issubset(observation_identity.columns):
        raise ValueError("observations must expose family_key and decision_date")
    return {
        "legal_state_events.csv": event_identity,
        "event_evidence_candidates.csv": evidence,
        "decision_observation_coverage.csv": observation_identity,
    }


def _denominator_hash(frames: dict[str, pd.DataFrame]) -> str:
    chunks = []
    for name in sorted(frames):
        payload = _stable_frame_bytes(frames[name])
        chunks.append(name.encode() + len(payload).to_bytes(8, "big") + payload)
    return _sha256(b"".join(chunks))


def _manifest_bytes(
    result: A2GateResult, mode: str, input_hash: str, denominator_hash: str
) -> bytes:
    payload = {
        "schema_version": "a2-manifest-v1",
        "mode": mode,
        "input_sha256": input_hash,
        "denominator_sha256": denominator_hash,
        "event_count": len(result.events),
        "observation_count": len(result.observations),
        "family_count": int(result.split["family_key"].nunique())
        if "family_key" in result.split
        else 0,
        "decision_threshold": "95/100",
        "event_threshold": "95/100",
        "family_threshold": "90/100",
        "stratum_threshold": "80/100",
        "identity_files": [
            "legal_state_events.csv",
            "event_evidence_candidates.csv",
            "decision_observation_coverage.csv",
        ],
    }
    if mode == "adjudicate" and result.frozen_denominator_sha256:
        payload["frozen_denominator_sha256"] = result.frozen_denominator_sha256
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_file_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _formal_frames(result: A2GateResult) -> dict[str, bytes]:
    denominator = _denominator_frames(result)
    evidence = denominator["event_evidence_candidates.csv"]
    return {
        "legal_state_events.csv": _csv_bytes(result.events),
        "event_evidence_candidates.csv": _csv_bytes(evidence),
        "event_dispositions.csv": _csv_bytes(result.dispositions),
        "legal_state_timeline.csv": _csv_bytes(result.timeline),
        "decision_observation_coverage.csv": _csv_bytes(result.observations),
        "a2_gate_metrics.csv": _csv_bytes(result.metrics),
        "a2_gate_review.md": result.review_markdown.encode("utf-8"),
    }


def _published_paths(target: Path, names: Iterable[str]) -> dict[str, Path]:
    return {name: target / name for name in names}


def write_a2_artifacts(
    result: A2GateResult,
    output_dir: str | Path,
    mode: str,
) -> dict[str, Path]:
    """Write A2 artifacts using a directory-level atomic publish.

    ``shadow`` deliberately performs no filesystem mutation.  A denominator
    freeze is idempotent only when the existing manifest and all identity
    bytes match.  Formal adjudication is append-only and requires a prior
    freeze made from the same result object.
    """

    if mode not in {"shadow", "freeze_denominator", "adjudicate"}:
        raise ValueError("mode must be shadow, freeze_denominator, or adjudicate")
    if mode == "shadow":
        return {}
    target = Path(output_dir)
    current_input_hash = _input_hash(
        result.events,
        result.dispositions,
        result.timeline,
        result.observations,
        result.split,
    )
    if current_input_hash != result.input_sha256:
        raise ValueError("input hash mismatch before A2 artifact mutation")
    denominator = _denominator_frames(result)
    current_denominator_hash = _denominator_hash(denominator)

    if mode == "adjudicate":
        if target.exists():
            raise FileExistsError(f"formal A2 adjudication directory already exists: {target}")
        if result.frozen_manifest_bytes is None or result.frozen_denominator_sha256 is None:
            raise ValueError("adjudicate requires a frozen denominator hash")
        if result.frozen_manifest_path is not None:
            if not result.frozen_manifest_path.is_file():
                raise ValueError("frozen manifest hash mismatch before adjudication")
            if result.frozen_manifest_path.read_bytes() != result.frozen_manifest_bytes:
                raise ValueError("frozen manifest hash mismatch before adjudication")
        if current_denominator_hash != result.frozen_denominator_sha256:
            raise ValueError("frozen denominator hash mismatch before adjudication")
        files = _formal_frames(result)
        manifest = _manifest_bytes(
            result, mode, current_input_hash, current_denominator_hash
        )
        files["a2_manifest.sha256"] = manifest
    else:
        manifest = _manifest_bytes(
            result, mode, current_input_hash, current_denominator_hash
        )
        identity_files = {
            name: _csv_bytes(frame) for name, frame in denominator.items()
        }
        identity_files["a2_manifest.sha256"] = manifest
        files = identity_files
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"A2 denominator target is not a directory: {target}")
            expected_names = set(files)
            existing_names = {path.name for path in target.iterdir() if path.is_file()}
            if existing_names != expected_names:
                raise ValueError("frozen denominator hash mismatch: artifact set differs")
            for name, payload in files.items():
                if (target / name).read_bytes() != payload:
                    raise ValueError("frozen denominator hash mismatch: bytes differ")
            result.frozen_manifest_path = target / "a2_manifest.sha256"
            result.frozen_manifest_bytes = manifest
            result.frozen_denominator_sha256 = current_denominator_hash
            return _published_paths(target, files)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        for name, payload in files.items():
            _write_file_fsync(temporary / name, payload)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if mode == "freeze_denominator":
        result.frozen_manifest_path = target / "a2_manifest.sha256"
        result.frozen_manifest_bytes = manifest
        result.frozen_denominator_sha256 = current_denominator_hash
    return _published_paths(target, files)
