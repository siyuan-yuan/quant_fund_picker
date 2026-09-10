from __future__ import annotations

import pandas as pd

from v6.a2_evidence import resolve_event_evidence
from v6.a2_timeline import (
    build_decision_observations,
    build_legal_state_timeline,
)


def _disposition(
    *,
    event_id: str = "event-1",
    family_key: str = "manager|fund",
    result: str = "NEW_STATE",
    known_at: str = "2019-02-01",
    legal_effective_from: str = "2018-01-01",
    event_anchor_date: str = "2018-01-01",
    fund_type: str = "混合型",
    minimum: float = 60.0,
    maximum: float = 95.0,
    failure_reason: str = "",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "family_key": family_key,
        "event_type": "LEGAL_TRANSFORMATION",
        "event_anchor_date": event_anchor_date,
        "result": result,
        "failure_reason": failure_reason,
        "fund_type": fund_type,
        "equity_min_pct": minimum,
        "equity_max_pct": maximum,
        "known_at": known_at,
        "legal_effective_from": legal_effective_from,
    }


def _families(
    *, inception: str = "2019-01-01", termination: str = ""
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family_key": "manager|fund",
                "inception_date": inception,
                "termination_date": termination,
            }
        ]
    )


def _schedule(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"decision_date": pd.to_datetime(list(dates))})


def test_state_starts_at_later_of_known_and_legal_dates() -> None:
    timeline = build_legal_state_timeline(pd.DataFrame([_disposition()]))

    assert timeline.loc[0, "usable_from"] == pd.Timestamp("2019-02-01")
    assert timeline.loc[0, "legal_effective_from"] == pd.Timestamp("2018-01-01")


def test_same_time_compatible_states_intersect_before_interval_creation() -> None:
    dispositions = pd.DataFrame(
        [
            _disposition(event_id="event-a", minimum=40.0, maximum=95.0),
            _disposition(event_id="event-b", minimum=60.0, maximum=100.0),
            _disposition(
                event_id="event-c",
                known_at="2020-01-01",
                legal_effective_from="2020-01-01",
                event_anchor_date="2020-01-01",
                minimum=70.0,
                maximum=90.0,
            ),
        ]
    )

    timeline = build_legal_state_timeline(dispositions)

    assert len(timeline) == 2
    assert timeline.loc[0, ["equity_min_pct", "equity_max_pct"]].tolist() == [60.0, 95.0]
    assert timeline.loc[0, "event_ids"] == "event-a|event-b"
    assert timeline.loc[0, "valid_to"] == pd.Timestamp("2020-01-01")


def test_preobservable_month_is_counted_but_not_backfilled() -> None:
    timeline = build_legal_state_timeline(pd.DataFrame([_disposition()]))

    observations = build_decision_observations(
        _families(), _schedule("2019-01-31", "2019-02-28"), timeline
    )

    assert observations["observation_status"].tolist() == [
        "PRE_OBSERVABLE",
        "OBSERVED_STATE",
    ]
    assert observations["active_eligibility"].tolist() == [True, True]


def test_unresolved_event_conflict_closes_old_state_until_later_state() -> None:
    dispositions = pd.DataFrame(
        [
            _disposition(
                event_id="initial",
                known_at="2019-01-01",
                legal_effective_from="2019-01-01",
                event_anchor_date="2019-01-01",
            ),
            _disposition(
                event_id="conflict",
                result="UNRESOLVED",
                failure_reason="UNRESOLVED_EVENT_CONFLICT",
                known_at="2019-06-01",
                legal_effective_from="2019-06-01",
                event_anchor_date="2019-06-01",
            ),
            _disposition(
                event_id="recovered",
                known_at="2020-01-01",
                legal_effective_from="2020-01-01",
                event_anchor_date="2020-01-01",
                minimum=70.0,
                maximum=90.0,
            ),
        ]
    )
    timeline = build_legal_state_timeline(dispositions)

    observations = build_decision_observations(
        _families(),
        _schedule("2019-05-31", "2019-06-28", "2020-01-31"),
        timeline,
    )

    assert observations["observation_status"].tolist() == [
        "OBSERVED_STATE",
        "UNRESOLVED_EVENT_CONFLICT",
        "OBSERVED_STATE",
    ]


def test_post_termination_month_is_not_active_denominator() -> None:
    timeline = build_legal_state_timeline(
        pd.DataFrame(
            [
                _disposition(
                    event_id="initial",
                    known_at="2019-01-01",
                    legal_effective_from="2019-01-01",
                    event_anchor_date="2019-01-01",
                ),
                _disposition(
                    event_id="terminated",
                    result="TERMINATED",
                    known_at="2020-01-10",
                    legal_effective_from="2020-01-15",
                    event_anchor_date="2020-01-15",
                ),
            ]
        )
    )

    observations = build_decision_observations(
        _families(termination=""),
        _schedule("2019-12-31", "2020-01-23"),
        timeline,
    )

    assert observations["observation_status"].tolist() == ["OBSERVED_STATE", "TERMINATED"]
    assert observations["active_eligibility"].tolist() == [True, False]


def test_query_applies_frozen_eligibility_mapping_and_never_crosses_family() -> None:
    timeline = build_legal_state_timeline(
        pd.DataFrame(
            [
                _disposition(
                    family_key="other|family",
                    known_at="2018-01-01",
                    legal_effective_from="2018-01-01",
                ),
                _disposition(
                    known_at="2019-01-01",
                    legal_effective_from="2019-01-01",
                ),
            ]
        )
    )

    observations = build_decision_observations(
        _families(), _schedule("2019-01-31"), timeline
    )
    row = observations.iloc[0]

    assert (row["observation_status"], row["eligibility_status"]) == (
        "OBSERVED_STATE",
        "ELIGIBLE",
    )
    assert row["target_type"] == "混合型-偏股"
    assert row["eligibility_mapping_version"] == "state_to_universe_eligibility_v1"
    assert row["event_ids"] != ""


def test_resolved_evidence_conflict_retains_causal_closure_boundary() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "conflict",
                "family_key": "manager|fund",
                "event_type": "LEGAL_TRANSFORMATION",
                "event_anchor_date": "2020-01-10",
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "event_id": "conflict",
                "source_document": source,
                "known_at": "2020-01-01",
                "document_stage": stage,
                "source_sha256": "a" * 64,
                "evidence_location": f"{source}#document",
            }
            for source, stage in [
                ("contract.pdf", "FUND_CONTRACT"),
                ("prospectus.pdf", "PROSPECTUS"),
            ]
        ]
    )
    sections = pd.DataFrame(
        [
            {
                "source_document": source,
                "extraction_status": "success",
                "fund_type": "混合型",
                "equity_min_pct": minimum,
                "equity_max_pct": maximum,
                "legal_effective_from": "2020-01-10",
                "section_text": "股票仓位约束",
                "section_locator": f"{source}#page=10",
                "source_sha256": "a" * 64,
            }
            for source, minimum, maximum in [
                ("contract.pdf", 60.0, 95.0),
                ("prospectus.pdf", 0.0, 30.0),
            ]
        ]
    )

    dispositions = resolve_event_evidence(events, candidates, sections)
    timeline = build_legal_state_timeline(dispositions)

    assert dispositions.loc[0, "failure_reason"] == "UNRESOLVED_EVENT_CONFLICT"
    assert dispositions.loc[0, "known_at"] == "2020-01-01"
    assert timeline.loc[0, "usable_from"] == pd.Timestamp("2020-01-10")
    assert timeline.loc[0, "timeline_status"] == "UNRESOLVED_EVENT_CONFLICT"
