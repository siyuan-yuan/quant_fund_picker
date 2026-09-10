from __future__ import annotations

import pandas as pd
import pytest

from v6.a2_evidence import resolve_event_evidence


def _event(event_type: str = "LEGAL_TRANSFORMATION") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "family_key": "manager|fund",
                "event_type": event_type,
                "legal_subject": "fund",
                "event_anchor_date": "2020-01-10",
            }
        ]
    )


def _candidates(*sources: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "source_document": source,
                "known_at": "2020-01-01",
                "document_stage": stage,
                "report_code": "FA010010",
                "report_name": source,
                "source_sha256": "a" * 64,
                "evidence_location": f"{source}#document",
            }
            for source, stage in sources
        ]
    )


def _sections(*states: tuple[str, str, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_document": source,
                "extraction_status": "success",
                "fund_type": fund_type,
                "equity_min_pct": minimum,
                "equity_max_pct": maximum,
                "legal_effective_from": "2020-01-10",
                "section_text": f"股票资产占基金资产的{minimum}%-{maximum}%",
                "section_locator": f"{source}#page=10;heading=投资范围",
                "source_sha256": "a" * 64,
            }
            for source, fund_type, minimum, maximum in states
        ]
    )


def test_source_priority_cannot_override_disjoint_same_event_states() -> None:
    result = resolve_event_evidence(
        _event(),
        _candidates(("contract.pdf", "FUND_CONTRACT"), ("prospectus.pdf", "PROSPECTUS")),
        _sections(
            ("contract.pdf", "混合型", 60.0, 95.0),
            ("prospectus.pdf", "混合型", 0.0, 30.0),
        ),
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "UNRESOLVED_EVENT_CONFLICT",
    )
    assert result["contributing_sources"] == "contract.pdf|prospectus.pdf"


def test_compatible_constraints_intersect_and_retain_all_sources() -> None:
    result = resolve_event_evidence(
        _event(),
        _candidates(("contract.pdf", "FUND_CONTRACT"), ("prospectus.pdf", "PROSPECTUS")),
        _sections(
            ("contract.pdf", "混合型", 40.0, 95.0),
            ("prospectus.pdf", "混合型", 40.0, 100.0),
        ),
    ).iloc[0]

    assert (result["result"], result["equity_min_pct"], result["equity_max_pct"]) == (
        "NEW_STATE",
        40.0,
        95.0,
    )
    assert result["operative_source"] == "contract.pdf"
    assert result["contributing_sources"] == "contract.pdf|prospectus.pdf"


def test_document_pointer_is_not_treated_as_state_evidence() -> None:
    sections = pd.DataFrame(
        [
            {
                "source_document": "notice.pdf",
                "extraction_status": "failure",
                "root_cause_reason": "INVESTMENT_SECTION_NOT_FOUND",
                "section_text": "投资范围及投资比例详见另行刊登的基金合同及招募说明书。",
                "section_locator": "notice.pdf#page=5",
            }
        ]
    )

    result = resolve_event_evidence(
        _event(), _candidates(("notice.pdf", "ANNOUNCEMENT")), sections
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "GOVERNING_DOCUMENT_REQUIRED",
    )


def test_affirmative_continuity_requires_causal_same_family_predecessor() -> None:
    candidates = _candidates(("notice.pdf", "ANNOUNCEMENT"))
    candidates["affirmative_continuity"] = True
    candidates["evidence_raw"] = "投资目标、投资范围和投资策略不变。"
    predecessors = pd.DataFrame(
        [
            {
                "event_id": "prior-1",
                "family_key": "manager|fund",
                "usable_from": "2019-01-01",
                "fund_type": "混合型",
                "equity_min_pct": 60.0,
                "equity_max_pct": 95.0,
            }
        ]
    )

    result = resolve_event_evidence(
        _event(), candidates, pd.DataFrame(), predecessor_states=predecessors
    ).iloc[0]

    assert (result["result"], result["predecessor_event_id"]) == (
        "VERIFIED_CONTINUITY",
        "prior-1",
    )
    assert (result["fund_type"], result["equity_min_pct"], result["equity_max_pct"]) == (
        "混合型",
        60.0,
        95.0,
    )


def test_affirmative_continuity_rejects_future_predecessor() -> None:
    candidates = _candidates(("notice.pdf", "ANNOUNCEMENT"))
    candidates["affirmative_continuity"] = True
    predecessors = pd.DataFrame(
        [
            {
                "event_id": "future",
                "family_key": "manager|fund",
                "usable_from": "2020-02-01",
                "fund_type": "混合型",
                "equity_min_pct": 60.0,
                "equity_max_pct": 95.0,
            }
        ]
    )

    result = resolve_event_evidence(
        _event(), candidates, pd.DataFrame(), predecessor_states=predecessors
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "MISSING_CAUSAL_PREDECESSOR",
    )


def test_incompatible_legal_types_are_an_event_conflict() -> None:
    result = resolve_event_evidence(
        _event(),
        _candidates(("contract.pdf", "FUND_CONTRACT"), ("prospectus.pdf", "PROSPECTUS")),
        _sections(
            ("contract.pdf", "混合型", 60.0, 95.0),
            ("prospectus.pdf", "股票型", 60.0, 95.0),
        ),
    ).iloc[0]

    assert result["failure_reason"] == "UNRESOLVED_EVENT_CONFLICT"


def test_confirmed_termination_closes_state_without_allocation_clause() -> None:
    candidates = _candidates(("termination.pdf", "EFFECTIVENESS_RESOLUTION"))
    candidates["termination_confirmed"] = True

    result = resolve_event_evidence(
        _event("TERMINATION"), candidates, pd.DataFrame()
    ).iloc[0]

    assert (result["result"], result["legal_effective_from"]) == (
        "TERMINATED",
        "2020-01-10",
    )


@pytest.mark.parametrize("event_kind", ["termination", "continuity"])
def test_non_state_dispositions_also_require_complete_document_provenance(
    event_kind: str,
) -> None:
    candidates = _candidates(("notice.pdf", "ANNOUNCEMENT"))
    candidates["source_sha256"] = ""
    if event_kind == "termination":
        candidates["termination_confirmed"] = True
        events = _event("TERMINATION")
        predecessors = None
    else:
        candidates["affirmative_continuity"] = True
        candidates["evidence_raw"] = "投资范围不变。"
        events = _event()
        predecessors = pd.DataFrame(
            [
                {
                    "event_id": "prior-1",
                    "family_key": "manager|fund",
                    "usable_from": "2019-01-01",
                    "fund_type": "混合型",
                    "equity_min_pct": 60.0,
                    "equity_max_pct": 95.0,
                }
            ]
        )

    result = resolve_event_evidence(
        events, candidates, pd.DataFrame(), predecessor_states=predecessors
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "INCOMPLETE_EVIDENCE_PROVENANCE",
    )


def test_state_evidence_without_source_checksum_is_unresolved() -> None:
    sections = _sections(("contract.pdf", "混合型", 60.0, 95.0)).drop(
        columns="source_sha256"
    )

    result = resolve_event_evidence(
        _event(), _candidates(("contract.pdf", "FUND_CONTRACT")), sections
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "INCOMPLETE_EVIDENCE_PROVENANCE",
    )


def test_explicit_candidate_family_mismatch_is_unresolved() -> None:
    candidates = _candidates(("contract.pdf", "FUND_CONTRACT"))
    candidates["family_key"] = "other-manager|other-fund"

    result = resolve_event_evidence(
        _event(), candidates, _sections(("contract.pdf", "混合型", 60.0, 95.0))
    ).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "EVIDENCE_FAMILY_MISMATCH",
    )


@pytest.mark.parametrize(
    ("candidate_change", "section_change"),
    [
        ({"known_at": ""}, {}),
        ({"document_stage": "UNDETERMINED"}, {}),
        ({}, {"section_locator": ""}),
    ],
)
def test_state_requires_complete_temporal_stage_and_location_provenance(
    candidate_change: dict[str, object], section_change: dict[str, object]
) -> None:
    candidates = _candidates(("contract.pdf", "FUND_CONTRACT"))
    sections = _sections(("contract.pdf", "混合型", 60.0, 95.0))
    for key, value in candidate_change.items():
        candidates[key] = value
    for key, value in section_change.items():
        sections[key] = value

    result = resolve_event_evidence(_event(), candidates, sections).iloc[0]

    assert (result["result"], result["failure_reason"]) == (
        "UNRESOLVED",
        "INCOMPLETE_EVIDENCE_PROVENANCE",
    )
