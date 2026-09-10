"""Causal V6-G0-A2 legal-state intervals and monthly observations."""

from __future__ import annotations

import pandas as pd

from .a2_eligibility import state_to_universe_eligibility_v1
from .contracts import A2_ELIGIBILITY_VERSION


_TIMELINE_COLUMNS = (
    "family_key",
    "usable_from",
    "valid_to",
    "timeline_status",
    "event_ids",
    "event_type",
    "legal_effective_from",
    "known_at",
    "fund_type",
    "equity_min_pct",
    "equity_max_pct",
    "failure_reason",
)

_OBSERVATION_COLUMNS = (
    "family_key",
    "decision_date",
    "active_eligibility",
    "observation_status",
    "event_ids",
    "fund_type",
    "equity_min_pct",
    "equity_max_pct",
    "eligibility_status",
    "target_type",
    "eligibility_exclusion_reason",
    "eligibility_mapping_version",
)


def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def build_legal_state_timeline(event_dispositions: pd.DataFrame) -> pd.DataFrame:
    """Build one deterministic boundary row per family and causal usable date."""

    required = {"event_id", "family_key", "result", "event_anchor_date"}
    missing = sorted(required.difference(event_dispositions.columns))
    if missing:
        raise ValueError(f"event_dispositions missing required columns: {','.join(missing)}")
    if event_dispositions.empty:
        return pd.DataFrame(columns=_TIMELINE_COLUMNS)

    frame = event_dispositions.copy()
    frame["_known"] = _date_series(frame, "known_at")
    frame["_legal"] = _date_series(frame, "legal_effective_from")
    frame["_anchor"] = _date_series(frame, "event_anchor_date")
    frame["_is_conflict"] = frame.get(
        "failure_reason", pd.Series("", index=frame.index)
    ).astype(str).eq("UNRESOLVED_EVENT_CONFLICT")
    frame["_is_termination"] = frame["result"].astype(str).eq("TERMINATED")
    frame["_is_state"] = frame["result"].astype(str).isin(
        {"NEW_STATE", "VERIFIED_CONTINUITY"}
    )

    relevant = frame["_is_state"] | frame["_is_conflict"] | frame["_is_termination"]
    frame = frame.loc[relevant].copy()
    if frame.empty:
        return pd.DataFrame(columns=_TIMELINE_COLUMNS)

    ordinary = ~frame["_is_termination"]
    frame.loc[ordinary, "_boundary"] = frame.loc[ordinary, ["_known", "_legal"]].max(axis=1)
    frame.loc[frame["_is_termination"], "_boundary"] = frame.loc[
        frame["_is_termination"], "_legal"
    ].fillna(frame.loc[frame["_is_termination"], "_anchor"])
    if frame["_boundary"].isna().any():
        bad = "|".join(sorted(frame.loc[frame["_boundary"].isna(), "event_id"].astype(str)))
        raise ValueError(f"causal boundary date missing for events: {bad}")

    rows: list[dict[str, object]] = []
    for (family_key, boundary), group in frame.groupby(
        ["family_key", "_boundary"], sort=True, dropna=False
    ):
        event_ids = "|".join(sorted(group["event_id"].astype(str).unique()))
        base: dict[str, object] = {
            "family_key": str(family_key),
            "usable_from": pd.Timestamp(boundary),
            "valid_to": pd.NaT,
            "timeline_status": "",
            "event_ids": event_ids,
            "event_type": "|".join(sorted(group.get("event_type", pd.Series(dtype=str)).astype(str).unique())),
            "legal_effective_from": group["_legal"].dropna().max() if group["_legal"].notna().any() else pd.NaT,
            "known_at": group["_known"].dropna().max() if group["_known"].notna().any() else pd.NaT,
            "fund_type": "",
            "equity_min_pct": pd.NA,
            "equity_max_pct": pd.NA,
            "failure_reason": "",
        }
        if group["_is_termination"].any():
            base["timeline_status"] = "TERMINATED"
        elif group["_is_conflict"].any():
            base["timeline_status"] = "UNRESOLVED_EVENT_CONFLICT"
            base["failure_reason"] = "UNRESOLVED_EVENT_CONFLICT"
        else:
            types = {_text(value) for value in group["fund_type"] if _text(value)}
            minimums = pd.to_numeric(group["equity_min_pct"], errors="coerce")
            maximums = pd.to_numeric(group["equity_max_pct"], errors="coerce")
            tight_min = float(minimums.max())
            tight_max = float(maximums.min())
            if len(types) != 1 or pd.isna(tight_min) or pd.isna(tight_max) or tight_min > tight_max:
                base["timeline_status"] = "UNRESOLVED_EVENT_CONFLICT"
                base["failure_reason"] = "UNRESOLVED_EVENT_CONFLICT"
            else:
                base.update(
                    {
                        "timeline_status": "OBSERVED_STATE",
                        "fund_type": next(iter(types)),
                        "equity_min_pct": tight_min,
                        "equity_max_pct": tight_max,
                    }
                )
        rows.append(base)

    timeline = pd.DataFrame(rows, columns=_TIMELINE_COLUMNS).sort_values(
        ["family_key", "usable_from", "event_ids"], kind="mergesort"
    ).reset_index(drop=True)
    timeline["valid_to"] = timeline.groupby("family_key", sort=False)["usable_from"].shift(-1)
    return timeline


def build_decision_observations(
    families: pd.DataFrame,
    schedule: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Query every requested family/date without causal backfilling."""

    if "family_key" not in families.columns:
        raise ValueError("families missing required column: family_key")
    if "decision_date" not in schedule.columns:
        raise ValueError("schedule missing required column: decision_date")

    family_rows = families.copy()
    family_rows["_inception"] = _date_series(family_rows, "inception_date")
    family_rows["_termination"] = _date_series(family_rows, "termination_date")
    decisions = pd.to_datetime(schedule["decision_date"], errors="coerce").dt.normalize()
    if decisions.isna().any():
        raise ValueError("schedule contains invalid decision_date")

    rows: list[dict[str, object]] = []
    for family in family_rows.sort_values("family_key", kind="mergesort").to_dict("records"):
        family_key = str(family["family_key"])
        inception = family.get("_inception")
        termination = family.get("_termination")
        history = (
            timeline.loc[timeline["family_key"].astype(str).eq(family_key)].sort_values(
                "usable_from", kind="mergesort"
            )
            if not timeline.empty and "family_key" in timeline.columns
            else pd.DataFrame()
        )
        if not history.empty:
            termination_boundaries = history.loc[
                history["timeline_status"].eq("TERMINATED"), "usable_from"
            ]
            if not termination_boundaries.empty:
                timeline_termination = termination_boundaries.min()
                termination = (
                    timeline_termination
                    if pd.isna(termination)
                    else min(termination, timeline_termination)
                )
        for decision in decisions.sort_values():
            active = bool(
                pd.notna(inception)
                and decision >= inception
                and (pd.isna(termination) or decision < termination)
            )
            row: dict[str, object] = {
                "family_key": family_key,
                "decision_date": decision,
                "active_eligibility": active,
                "observation_status": "PRE_OBSERVABLE",
                "event_ids": "",
                "fund_type": "",
                "equity_min_pct": pd.NA,
                "equity_max_pct": pd.NA,
                "eligibility_status": "",
                "target_type": "",
                "eligibility_exclusion_reason": "",
                "eligibility_mapping_version": A2_ELIGIBILITY_VERSION,
            }
            if pd.notna(inception) and decision < inception:
                row["active_eligibility"] = False
                row["observation_status"] = "NOT_YET_ACTIVE"
            elif pd.notna(termination) and decision >= termination:
                row["active_eligibility"] = False
                row["observation_status"] = "TERMINATED"
            elif not history.empty:
                matches = history.loc[
                    history["usable_from"].le(decision)
                    & (history["valid_to"].isna() | history["valid_to"].gt(decision))
                ]
                if not matches.empty:
                    state = matches.iloc[-1]
                    row["event_ids"] = state["event_ids"]
                    row["observation_status"] = state["timeline_status"]
                    if state["timeline_status"] == "OBSERVED_STATE":
                        result = state_to_universe_eligibility_v1(
                            state["fund_type"],
                            state["equity_min_pct"],
                            state["equity_max_pct"],
                        )
                        row.update(
                            {
                                "fund_type": state["fund_type"],
                                "equity_min_pct": state["equity_min_pct"],
                                "equity_max_pct": state["equity_max_pct"],
                                "eligibility_status": result.status,
                                "target_type": result.target_type,
                                "eligibility_exclusion_reason": result.exclusion_reason,
                                "eligibility_mapping_version": result.mapping_version,
                            }
                        )
            rows.append(row)

    return pd.DataFrame(rows, columns=_OBSERVATION_COLUMNS)
