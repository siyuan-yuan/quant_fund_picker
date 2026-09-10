"""Resolve V6-G0-A2 event evidence without allowing priority to hide conflicts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


_STAGE_PRIORITY = {
    "FUND_CONTRACT": 0,
    "EFFECTIVENESS_RESOLUTION": 1,
    "PROSPECTUS": 2,
    "ANNOUNCEMENT": 3,
    "UNDETERMINED": 4,
}

_OUTPUT_COLUMNS = (
    "event_id",
    "family_key",
    "event_type",
    "event_anchor_date",
    "result",
    "failure_reason",
    "fund_type",
    "equity_min_pct",
    "equity_max_pct",
    "known_at",
    "legal_effective_from",
    "operative_source",
    "contributing_sources",
    "evidence_raw",
    "evidence_location",
    "source_sha256",
    "predecessor_event_id",
)


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _date_text(value: object) -> str:
    parsed = _date(value)
    return "" if parsed is None else str(parsed.date())


def _candidate_provenance_complete(candidate: Mapping[str, object]) -> bool:
    return bool(
        _date(candidate.get("known_at")) is not None
        and _text(candidate.get("document_stage")) not in {"", "UNDETERMINED"}
        and _text(candidate.get("evidence_location"))
        and len(_text(candidate.get("source_sha256"))) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in _text(candidate.get("source_sha256"))
        )
    )


def _empty(event: Mapping[str, object]) -> dict[str, object]:
    return {
        "event_id": _text(event.get("event_id")),
        "family_key": _text(event.get("family_key")),
        "event_type": _text(event.get("event_type")),
        "event_anchor_date": _date_text(event.get("event_anchor_date")),
        "result": "UNRESOLVED",
        "failure_reason": "NO_USABLE_STATE_EVIDENCE",
        "fund_type": "",
        "equity_min_pct": pd.NA,
        "equity_max_pct": pd.NA,
        "known_at": "",
        "legal_effective_from": "",
        "operative_source": "",
        "contributing_sources": "",
        "evidence_raw": "",
        "evidence_location": "",
        "source_sha256": "",
        "predecessor_event_id": "",
    }


def _candidate_group(candidates: pd.DataFrame, event_id: str) -> pd.DataFrame:
    if candidates.empty or "event_id" not in candidates:
        return pd.DataFrame()
    group = candidates.loc[candidates["event_id"].astype(str).eq(event_id)].copy()
    if group.empty:
        return group
    group["_priority"] = group.get(
        "document_stage", pd.Series("UNDETERMINED", index=group.index)
    ).map(lambda value: _STAGE_PRIORITY.get(_text(value), 4))
    group["_source"] = group.get(
        "source_document", pd.Series("", index=group.index)
    ).map(_text)
    return group.sort_values(["_priority", "_source"], kind="mergesort")


def _joined_sections(group: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    if group.empty or sections.empty or "source_document" not in sections:
        return pd.DataFrame()
    right = sections.copy()
    right["source_document"] = right["source_document"].map(_text)
    if "source_sha256" in right.columns:
        right = right.rename(columns={"source_sha256": "section_source_sha256"})
    return group.merge(right, on="source_document", how="left", suffixes=("_candidate", "_section"))


def _has_pointer(joined: pd.DataFrame) -> bool:
    if joined.empty:
        return False
    for column in ("section_text", "evidence_raw", "report_name"):
        if column not in joined:
            continue
        for value in joined[column]:
            text = _text(value).replace(" ", "")
            if ("详见" in text or "按照" in text) and (
                "基金合同" in text or "招募说明书" in text
            ):
                return True
    return False


def _causal_predecessor(
    predecessor_states: pd.DataFrame | None,
    family_key: str,
    anchor: pd.Timestamp | None,
) -> Mapping[str, object] | None:
    if predecessor_states is None or predecessor_states.empty or anchor is None:
        return None
    required = {"family_key", "usable_from"}
    if not required.issubset(predecessor_states.columns):
        return None
    frame = predecessor_states.loc[
        predecessor_states["family_key"].astype(str).eq(family_key)
    ].copy()
    frame["_usable"] = pd.to_datetime(frame["usable_from"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["_usable"].notna() & frame["_usable"].le(anchor)]
    if frame.empty:
        return None
    return frame.sort_values(["_usable", "event_id"], kind="mergesort").iloc[-1].to_dict()


def resolve_event_evidence(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    sections: pd.DataFrame,
    *,
    predecessor_states: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve each legal event while retaining every contributing source."""

    required = {"event_id", "family_key", "event_type", "event_anchor_date"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {','.join(missing)}")

    resolved: list[dict[str, object]] = []
    for event in events.sort_values(
        ["event_anchor_date", "family_key", "event_id"], kind="mergesort"
    ).to_dict("records"):
        output = _empty(event)
        event_id = output["event_id"]
        group = _candidate_group(candidates, event_id)
        sources = sorted(set(group.get("_source", pd.Series(dtype=str))))
        output["contributing_sources"] = "|".join(source for source in sources if source)
        anchor = _date(event.get("event_anchor_date"))

        if "family_key" in group.columns:
            explicit_families = {
                value for value in group["family_key"].map(_text) if value
            }
            if explicit_families and explicit_families != {output["family_key"]}:
                output["failure_reason"] = "EVIDENCE_FAMILY_MISMATCH"
                resolved.append(output)
                continue

        if output["event_type"] == "TERMINATION" and not group.empty:
            confirmed = group.get(
                "termination_confirmed", pd.Series(False, index=group.index)
            ).map(_bool)
            if confirmed.any():
                chosen = group.loc[confirmed].iloc[0]
                if not _candidate_provenance_complete(chosen):
                    output["failure_reason"] = "INCOMPLETE_EVIDENCE_PROVENANCE"
                    resolved.append(output)
                    continue
                output.update(
                    {
                        "result": "TERMINATED",
                        "failure_reason": "",
                        "known_at": _date_text(chosen.get("known_at")),
                        "legal_effective_from": output["event_anchor_date"],
                        "operative_source": _text(chosen.get("source_document")),
                        "evidence_raw": _text(chosen.get("evidence_raw")),
                        "evidence_location": _text(chosen.get("evidence_location")),
                        "source_sha256": _text(chosen.get("source_sha256")),
                    }
                )
                resolved.append(output)
                continue

        joined = _joined_sections(group, sections)
        if not joined.empty:
            success = joined.get(
                "extraction_status", pd.Series("", index=joined.index)
            ).astype(str).eq("success")
            has_state_semantics = (
                joined.get("fund_type", pd.Series("", index=joined.index)).map(_text).ne("")
                & pd.to_numeric(
                    joined.get("equity_min_pct", pd.Series(index=joined.index, dtype=float)),
                    errors="coerce",
                ).notna()
                & pd.to_numeric(
                    joined.get("equity_max_pct", pd.Series(index=joined.index, dtype=float)),
                    errors="coerce",
                ).notna()
                & joined.get("section_text", pd.Series("", index=joined.index)).map(_text).ne("")
            )
            has_section_checksum = joined.get(
                "section_source_sha256", pd.Series("", index=joined.index)
            ).map(_text).str.fullmatch(r"[0-9a-fA-F]{64}")
            has_location = joined.get(
                "section_locator", pd.Series("", index=joined.index)
            ).map(_text).ne("")
            has_known_at = joined.get(
                "known_at", pd.Series("", index=joined.index)
            ).map(_date).notna()
            has_legal_stage = joined.get(
                "document_stage", pd.Series("UNDETERMINED", index=joined.index)
            ).map(_text).ne("UNDETERMINED")
            complete_provenance = (
                has_section_checksum & has_location & has_known_at & has_legal_stage
            )
            incomplete_provenance = success & has_state_semantics & ~complete_provenance
            usable = joined.loc[
                success & has_state_semantics & complete_provenance
            ].copy()
        else:
            incomplete_provenance = pd.Series(dtype=bool)
            usable = pd.DataFrame()

        if not incomplete_provenance.empty and incomplete_provenance.any():
            output["failure_reason"] = "INCOMPLETE_EVIDENCE_PROVENANCE"
            resolved.append(output)
            continue

        if not usable.empty:
            usable["_minimum"] = pd.to_numeric(usable["equity_min_pct"], errors="raise")
            usable["_maximum"] = pd.to_numeric(usable["equity_max_pct"], errors="raise")
            types = set(usable["fund_type"].map(_text))
            effective_dates = {
                value
                for value in usable.get(
                    "legal_effective_from", pd.Series("", index=usable.index)
                ).map(_date_text)
                if value
            }
            tight_min = float(usable["_minimum"].max())
            tight_max = float(usable["_maximum"].min())
            incompatible = (
                len(types) != 1
                or len(effective_dates) > 1
                or tight_min > tight_max
                or bool((usable["_minimum"] < 0).any())
                or bool((usable["_maximum"] > 100).any())
            )
            if incompatible:
                output["failure_reason"] = "UNRESOLVED_EVENT_CONFLICT"
                conflict_known_dates = [
                    parsed
                    for parsed in usable.get(
                        "known_at", pd.Series("", index=usable.index)
                    ).map(_date)
                    if parsed is not None
                ]
                output["known_at"] = (
                    str(max(conflict_known_dates).date()) if conflict_known_dates else ""
                )
                output["legal_effective_from"] = (
                    max(effective_dates) if effective_dates else output["event_anchor_date"]
                )
                resolved.append(output)
                continue

            chosen = usable.sort_values(["_priority", "_source"], kind="mergesort").iloc[0]
            known_dates = [
                parsed
                for parsed in usable.get(
                    "known_at", pd.Series("", index=usable.index)
                ).map(_date)
                if parsed is not None
            ]
            output.update(
                {
                    "result": "NEW_STATE",
                    "failure_reason": "",
                    "fund_type": next(iter(types)),
                    "equity_min_pct": tight_min,
                    "equity_max_pct": tight_max,
                    "known_at": str(max(known_dates).date()) if known_dates else "",
                    "legal_effective_from": (
                        next(iter(effective_dates))
                        if effective_dates
                        else output["event_anchor_date"]
                    ),
                    "operative_source": _text(chosen.get("source_document")),
                    "evidence_raw": _text(chosen.get("section_text")),
                    "evidence_location": _text(chosen.get("section_locator")),
                    "source_sha256": _text(chosen.get("section_source_sha256")),
                }
            )
            resolved.append(output)
            continue

        continuity = (
            group.get("affirmative_continuity", pd.Series(False, index=group.index)).map(_bool)
            if not group.empty
            else pd.Series(dtype=bool)
        )
        if not continuity.empty and continuity.any():
            predecessor = _causal_predecessor(
                predecessor_states, output["family_key"], anchor
            )
            if predecessor is None:
                output["failure_reason"] = "MISSING_CAUSAL_PREDECESSOR"
            else:
                chosen = group.loc[continuity].iloc[0]
                if not _candidate_provenance_complete(chosen):
                    output["failure_reason"] = "INCOMPLETE_EVIDENCE_PROVENANCE"
                else:
                    output.update(
                        {
                            "result": "VERIFIED_CONTINUITY",
                            "failure_reason": "",
                            "fund_type": _text(predecessor.get("fund_type")),
                            "equity_min_pct": predecessor.get("equity_min_pct", pd.NA),
                            "equity_max_pct": predecessor.get("equity_max_pct", pd.NA),
                            "known_at": _date_text(chosen.get("known_at")),
                            "legal_effective_from": output["event_anchor_date"],
                            "operative_source": _text(chosen.get("source_document")),
                            "evidence_raw": _text(chosen.get("evidence_raw")),
                            "evidence_location": _text(chosen.get("evidence_location")),
                            "source_sha256": _text(chosen.get("source_sha256")),
                            "predecessor_event_id": _text(predecessor.get("event_id")),
                        }
                    )
            resolved.append(output)
            continue

        if _has_pointer(joined):
            output["failure_reason"] = "GOVERNING_DOCUMENT_REQUIRED"
        resolved.append(output)

    return pd.DataFrame(resolved, columns=_OUTPUT_COLUMNS)
