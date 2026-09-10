from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from v6.a2_gate import A2GateResult, compute_a2_gate, write_a2_artifacts


EVENT_TYPES = (
    "INCEPTION",
    "LEGAL_TRANSFORMATION",
    "MERGER_OR_SUCCESSION",
    "FUND_TYPE_CHANGE",
    "INVESTMENT_MANDATE_CHANGE",
    "TERMINATION",
)


def _gate_inputs(
    *,
    event_resolved: int = 1000,
    active_observed: int = 1000,
    covered_families: int = 54,
    stratum_counts: tuple[int, ...] = (9, 9, 9, 9, 9, 9),
    critical_unresolved: int = 0,
    invariant: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build deterministic frames with integer numerators and denominators."""

    events = pd.DataFrame(
        {
            "event_id": [f"event-{index:04d}" for index in range(1000)],
            "family_key": [f"family-{index % 60:02d}" for index in range(1000)],
            "event_type": [EVENT_TYPES[index % len(EVENT_TYPES)] for index in range(1000)],
            "legal_subject": ["fund-state"] * 1000,
            "event_anchor_date": ["2020-01-01"] * 1000,
            "generation_version": ["a2-legal-event-v1"] * 1000,
        }
    )
    dispositions = events.copy()
    dispositions["result"] = [
        "NEW_STATE" if index < event_resolved else "UNRESOLVED"
        for index in range(1000)
    ]
    dispositions["failure_reason"] = ""
    if critical_unresolved:
        critical_positions = [
            index
            for index, event_type in enumerate(dispositions["event_type"])
            if event_type in EVENT_TYPES
        ][:critical_unresolved]
        dispositions.loc[critical_positions, "result"] = "UNRESOLVED"
        dispositions.loc[critical_positions, "failure_reason"] = "MISSING_EVIDENCE"

    split_rows = []
    for stratum_index in range(6):
        for family_index in range(10):
            split_rows.append(
                {
                    "family_key": f"family-{stratum_index * 10 + family_index:02d}",
                    "stratum": f"stratum-{stratum_index + 1}",
                }
            )
    split = pd.DataFrame(split_rows)

    counts = list(stratum_counts)
    while sum(counts) > covered_families:
        for index, count in enumerate(counts):
            if sum(counts) <= covered_families:
                break
            if count > 0:
                counts[index] -= 1
    while sum(counts) < covered_families:
        for index, count in enumerate(counts):
            if sum(counts) >= covered_families:
                break
            if count < 10:
                counts[index] += 1
    covered = {
        f"family-{stratum_index * 10 + family_index:02d}"
        for stratum_index, count in enumerate(counts)
        for family_index in range(count)
    }
    timeline = pd.DataFrame(
        [
            {
                "family_key": family_key,
                "usable_from": "2020-01-01",
                "valid_to": pd.NaT,
                "timeline_status": "OBSERVED_STATE",
                "event_ids": "event-0000",
            }
            for family_key in sorted(covered)
        ]
    )

    observations = pd.DataFrame(
        {
            "family_key": [f"family-{index % 60:02d}" for index in range(1000)],
            "decision_date": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "active_eligibility": [True] * 1000,
            "observation_status": [
                "OBSERVED_STATE" if index < active_observed else "PRE_OBSERVABLE"
                for index in range(1000)
            ],
        }
    )

    if invariant == "causal_violations":
        timeline["causal_violation"] = [True] + [False] * (len(timeline) - 1)
    elif invariant == "nonpositive_intervals":
        timeline["nonpositive_interval"] = [True] + [False] * (len(timeline) - 1)
    elif invariant == "unresolved_conflicts":
        dispositions.loc[0, "result"] = "UNRESOLVED"
        dispositions.loc[0, "failure_reason"] = "UNRESOLVED_EVENT_CONFLICT"

    return {
        "events": events,
        "dispositions": dispositions,
        "timeline": timeline,
        "observations": observations,
        "split": split,
    }


def _compute(**kwargs: object) -> A2GateResult:
    return compute_a2_gate(**_gate_inputs(**kwargs))


def test_event_coverage_uses_exact_95_percent_boundary() -> None:
    below = _compute(event_resolved=949)
    at = _compute(event_resolved=950)

    assert below.event_coverage_passed is False
    assert below.event_coverage.numerator == 949
    assert below.event_coverage.denominator == 1000
    assert below.event_coverage.fraction == Fraction(949, 1000)
    assert at.event_coverage_passed is True
    assert at.event_coverage.fraction == Fraction(19, 20)


@pytest.mark.parametrize(
    "event_type",
    [
        "INCEPTION",
        "LEGAL_TRANSFORMATION",
        "MERGER_OR_SUCCESSION",
        "FUND_TYPE_CHANGE",
        "INVESTMENT_MANDATE_CHANGE",
        "TERMINATION",
    ],
)
def test_any_unresolved_critical_event_fails_critical_gate(event_type: str) -> None:
    inputs = _gate_inputs()
    position = inputs["events"].index[inputs["events"]["event_type"].eq(event_type)][0]
    inputs["dispositions"].loc[position, "result"] = "UNRESOLVED"
    inputs["dispositions"].loc[position, "failure_reason"] = "MISSING_EVIDENCE"

    result = compute_a2_gate(**inputs)

    assert result.critical_events_passed is False
    assert result.passed is False


def test_decision_observation_coverage_uses_exact_95_percent_boundary() -> None:
    below = _compute(active_observed=949)
    at = _compute(active_observed=950)

    assert below.decision_coverage_passed is False
    assert below.decision_coverage.numerator == 949
    assert below.decision_coverage.denominator == 1000
    assert below.decision_coverage.fraction == Fraction(949, 1000)
    assert at.decision_coverage_passed is True
    assert at.decision_coverage.fraction == Fraction(19, 20)


def test_family_coverage_uses_exact_90_percent_boundary() -> None:
    below = _compute(covered_families=53)
    at = _compute(covered_families=54)

    assert below.family_coverage_passed is False
    assert below.family_coverage.fraction == Fraction(53, 60)
    assert at.family_coverage_passed is True
    assert at.family_coverage.fraction == Fraction(9, 10)


def test_each_stratum_must_reach_exact_80_percent() -> None:
    below = _compute(covered_families=47, stratum_counts=(7, 8, 8, 8, 8, 8))
    at = _compute(covered_families=48, stratum_counts=(8, 8, 8, 8, 8, 8))

    assert below.strata_passed is False
    assert at.strata_passed is True
    assert at.stratum_coverages[0].fraction == Fraction(4, 5)


@pytest.mark.parametrize(
    "invariant",
    ["causal_violations", "nonpositive_intervals", "unresolved_conflicts"],
)
def test_each_p0_invariant_independently_blocks_gate(invariant: str) -> None:
    result = _compute(invariant=invariant)

    assert result.passed is False
    assert result.invariants[invariant] == 1
    assert result.invariant_metrics[invariant].result == "FAIL"


def test_passing_defaults_require_all_gate_conditions() -> None:
    result = _compute()

    assert result.passed is True
    assert result.event_coverage_passed is True
    assert result.critical_events_passed is True
    assert result.family_coverage_passed is True
    assert result.strata_passed is True
    assert result.decision_coverage_passed is True
    assert result.p0_invariants_passed is True


def test_shadow_mode_does_not_write_verdict_or_dispositions(tmp_path: Path) -> None:
    result = _compute()
    output = tmp_path / "shadow"

    paths = write_a2_artifacts(result, output, "shadow")

    assert paths == {}
    assert not output.exists()
    assert not list(tmp_path.rglob("*verdict*"))
    assert not list(tmp_path.rglob("*disposition*"))


def test_freeze_mode_writes_only_denominator_identities_and_manifest(tmp_path: Path) -> None:
    result = _compute()

    paths = write_a2_artifacts(result, tmp_path / "frozen", "freeze_denominator")
    names = {path.name for path in paths.values()}

    assert names == {
        "legal_state_events.csv",
        "event_evidence_candidates.csv",
        "decision_observation_coverage.csv",
        "a2_manifest.sha256",
    }
    assert not (tmp_path / "frozen" / "event_dispositions.csv").exists()
    assert not (tmp_path / "frozen" / "a2_gate_metrics.csv").exists()
    assert not (tmp_path / "frozen" / "a2_gate_review.md").exists()


def test_repeated_freeze_is_byte_identical(tmp_path: Path) -> None:
    result = _compute()
    output = tmp_path / "frozen"
    first = write_a2_artifacts(result, output, "freeze_denominator")
    before = {name: path.read_bytes() for name, path in first.items()}

    second = write_a2_artifacts(result, output, "freeze_denominator")
    after = {name: path.read_bytes() for name, path in second.items()}

    assert before == after


def test_existing_adjudication_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    result = _compute()
    output = tmp_path / "adjudication"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_a2_artifacts(result, output, "adjudicate")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output.iterdir()) == [sentinel]


def test_hash_mismatch_fails_before_any_output_mutation(tmp_path: Path) -> None:
    result = _compute()
    output = tmp_path / "frozen"
    write_a2_artifacts(result, output, "freeze_denominator")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    result.observations.loc[0, "observation_status"] = "UNRESOLVED_EVENT_CONFLICT"
    with pytest.raises(ValueError, match="hash"):
        write_a2_artifacts(result, output, "freeze_denominator")

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before


def test_adjudicate_requires_and_verifies_frozen_denominator(tmp_path: Path) -> None:
    result = _compute()
    frozen = tmp_path / "frozen"
    write_a2_artifacts(result, frozen, "freeze_denominator")

    adjudication = tmp_path / "adjudication"
    paths = write_a2_artifacts(result, adjudication, "adjudicate")

    assert {
        "legal_state_events.csv",
        "event_evidence_candidates.csv",
        "event_dispositions.csv",
        "legal_state_timeline.csv",
        "decision_observation_coverage.csv",
        "a2_gate_metrics.csv",
        "a2_gate_review.md",
        "a2_manifest.sha256",
    } == {path.name for path in paths.values()}


def test_adjudicate_rejects_tampered_frozen_manifest_before_publish(tmp_path: Path) -> None:
    result = _compute()
    frozen = tmp_path / "frozen"
    write_a2_artifacts(result, frozen, "freeze_denominator")
    (frozen / "a2_manifest.sha256").write_text("tampered\\n", encoding="utf-8")

    adjudication = tmp_path / "adjudication"
    with pytest.raises(ValueError, match="frozen manifest hash"):
        write_a2_artifacts(result, adjudication, "adjudicate")

    assert not adjudication.exists()


def test_atomic_writer_flushes_and_fsyncs_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _compute()
    calls: list[str] = []

    import v6.a2_gate as gate_module

    real_fsync = gate_module.os.fsync
    real_replace = gate_module.os.replace

    def fsync(fd: int) -> None:
        calls.append("fsync")
        real_fsync(fd)

    def replace(source: str | Path, destination: str | Path) -> None:
        calls.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(gate_module.os, "fsync", fsync)
    monkeypatch.setattr(gate_module.os, "replace", replace)

    write_a2_artifacts(result, tmp_path / "frozen", "freeze_denominator")

    assert "replace" in calls
    assert calls.index("fsync") < calls.index("replace")

