from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from v6.a2_events import A2_EVENT_GENERATION_VERSION, _event_id
from v6.a2_validation import (
    build_development_event_audit,
    build_family_validation_audit,
    freeze_external_event_validation_cohort,
    freeze_reviewed_denominators,
    validate_external_event_labels,
)


CLOSED_LABELS = {
    "INCEPTION",
    "LEGAL_TRANSFORMATION",
    "MERGER_OR_SUCCESSION",
    "FUND_TYPE_CHANGE",
    "INVESTMENT_MANDATE_CHANGE",
    "TERMINATION",
    "NOT_EVENT",
    "REVIEW_REQUIRED",
}


def _metadata() -> pd.DataFrame:
    rows = [
        {
            "fundCode": "000001",
            "uploadInfoId": "1",
            "reportName": "保本周期到期后转型为灵活配置混合型基金公告",
            "reportSendDate": "2019-01-02",
            "reportCode": "FA010010",
        },
        {
            "fundCode": "000001",
            "uploadInfoId": "2",
            "reportName": "开通基金转换业务公告",
            "reportSendDate": "2019-02-02",
            "reportCode": "FA010010",
        },
    ]
    for index in range(150):
        rows.append(
            {
                "fundCode": f"{100000 + index // 3:06d}",
                "uploadInfoId": f"external-{index}",
                "reportName": (
                    "基金转型公告" if index % 2 == 0 else "基金转换业务公告"
                ),
                "reportSendDate": "2020-01-02",
                "reportCode": "FA010010",
            }
        )
    return pd.DataFrame(rows)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "share_code": "000001",
                "family_key": "manager|sample",
                "name": "样本基金",
                "inception_date": "2018-01-01",
            }
        ]
    )


def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_share_code": "000001",
                "query_code": "000001",
                "family_key": "manager|sample",
            }
        ]
    )


def _split() -> pd.DataFrame:
    rows = []
    for index in range(60):
        rows.append(
            {
                "family_key": "manager|sample" if index == 0 else f"manager|fund-{index:02d}",
                "share_code": f"{index + 1:06d}",
                "split_role": "development" if index < 24 else "untouched_validation",
                "stratum": f"stratum-{index // 10 + 1}",
            }
        )
    return pd.DataFrame(rows)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": pd.date_range("2006-01-31", periods=243, freq="ME")
        }
    )


def _events_and_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_id = _event_id(
        "manager|sample",
        "INCEPTION",
        "样本基金",
        "2018-01-01",
    )
    events = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "family_key": "manager|sample",
                "event_type": "INCEPTION",
                "legal_subject": "样本基金",
                "event_anchor_date": "2018-01-01",
                "generation_version": A2_EVENT_GENERATION_VERSION,
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "source_document": "000001_1.pdf",
                "known_at": "2018-01-02",
            },
            {
                "event_id": event_id,
                "source_document": "000001_2.pdf",
                "known_at": "2018-02-02",
            },
        ]
    )
    return events, evidence


def test_development_audit_isolated_to_development_rows() -> None:
    split = _split()
    audit = build_development_event_audit(
        _sample(), _metadata(), _aliases(), split, pd.DataFrame()
    )

    assert set(audit["split_role"]) == {"development"}
    assert not audit.empty
    assert {"event_type", "decision_status", "source_document", "title_evidence"}.issubset(
        audit.columns
    )


def test_development_audit_rejects_validation_leakage() -> None:
    split = _split()
    with pytest.raises(ValueError, match="validation leakage"):
        build_development_event_audit(
            _sample(), _metadata(), _aliases(), split, pd.DataFrame(),
            allow_validation=False,
        )


def test_external_cohort_is_deterministic_bounded_and_unlabelled() -> None:
    cohort = freeze_external_event_validation_cohort(
        _metadata(), _sample(), _aliases()
    )
    repeated = freeze_external_event_validation_cohort(
        _metadata(), _sample(), _aliases()
    )

    assert len(cohort) == 120
    assert cohort.to_csv(index=False) == repeated.to_csv(index=False)
    assert "label" not in cohort.columns
    assert cohort["source_document"].is_unique
    assert cohort.groupby("share_code").size().max() <= 3
    assert set(cohort["event_scope"]) == {"EXTERNAL"}
    assert cohort["canonical_sort_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_external_labels_are_closed_and_cover_frozen_ids() -> None:
    cohort = freeze_external_event_validation_cohort(
        _metadata(), _sample(), _aliases()
    )
    labels = cohort[["external_row_id"]].copy()
    labels["label"] = "NOT_EVENT"
    labels.loc[0, "label"] = "LEGAL_TRANSFORMATION"

    result = validate_external_event_labels(cohort, labels)

    assert set(result["label"]) <= CLOSED_LABELS
    assert len(result) == 120
    assert {"precision", "recall", "predicted_event_type"}.issubset(result.columns)


def test_external_labels_reject_unknown_or_missing_rows() -> None:
    cohort = freeze_external_event_validation_cohort(
        _metadata(), _sample(), _aliases()
    )
    labels = cohort[["external_row_id"]].iloc[:-1].copy()
    labels["label"] = "NOT_EVENT"

    with pytest.raises(ValueError, match="exactly cover"):
        validate_external_event_labels(cohort, labels)

    labels = cohort[["external_row_id"]].copy()
    labels["label"] = "NEW_RULE"
    with pytest.raises(ValueError, match="closed taxonomy"):
        validate_external_event_labels(cohort, labels)


def test_family_audit_reports_validation_exposure_without_creating_rules() -> None:
    audit = build_family_validation_audit(
        _sample(), _metadata(), _aliases(), _split(), pd.DataFrame()
    )

    assert set(audit["split_role"]) == {"untouched_validation"}
    assert audit["title_semantics_exposed_during_a1"].eq(True).all()
    assert "new_rule_from_validation" in audit.columns
    assert audit["new_rule_from_validation"].eq(False).all()


def test_denominator_freeze_writes_only_reviewed_identities_atomically(
    tmp_path: Path,
) -> None:
    events, evidence = _events_and_evidence()
    observations = pd.DataFrame(
        [
            {
                "family_key": family,
                "decision_date": date,
                "active_eligibility": True,
            }
            for family in _split()["family_key"]
            for date in _schedule()["decision_date"]
        ]
    )

    paths = freeze_reviewed_denominators(
        events,
        evidence,
        observations,
        _split(),
        _schedule(),
        tmp_path / "frozen",
    )
    assert {path.name for path in paths.values()} == {
        "legal_state_events.csv",
        "event_evidence_candidates.csv",
        "decision_observation_ids.csv",
        "a2_denominator_manifest.sha256",
    }
    assert not (tmp_path / "frozen" / "event_dispositions.csv").exists()
    assert not (tmp_path / "frozen" / "a2_gate_metrics.csv").exists()

    event_columns = pd.read_csv(paths["legal_state_events.csv"]).columns
    assert "source_document" not in event_columns
    assert len(pd.read_csv(paths["event_evidence_candidates.csv"])) == 2
    assert len(pd.read_csv(paths["decision_observation_ids.csv"])) == 60 * 243

    before = {name: path.read_bytes() for name, path in paths.items()}
    repeated = freeze_reviewed_denominators(
        events,
        evidence,
        observations,
        _split(),
        _schedule(),
        tmp_path / "frozen",
    )
    after = {name: path.read_bytes() for name, path in repeated.items()}
    assert before == after


def test_denominator_freeze_requires_exact_60_family_six_stratum_split(
    tmp_path: Path,
) -> None:
    events, evidence = _events_and_evidence()
    observations = pd.DataFrame(
        {
            "family_key": ["manager|sample"],
            "decision_date": [pd.Timestamp("2006-01-31")],
            "active_eligibility": [True],
        }
    )
    with pytest.raises(ValueError, match="60 families and six strata"):
        freeze_reviewed_denominators(
            events,
            evidence,
            observations,
            _split().iloc[:59],
            _schedule(),
            tmp_path / "invalid",
        )
