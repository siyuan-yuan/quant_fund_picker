"""Frozen monthly decision schedule for V6-G0-A2."""

from __future__ import annotations

import pandas as pd

from .contracts import A2_SCHEDULE_VERSION, DECISION_END, DECISION_START


_REQUIRED_COLUMNS = ("exchange", "cal_date", "is_open")


def build_a2_monthly_schedule(
    trade_calendar: pd.DataFrame,
    *,
    start: str = DECISION_START,
    end: str = DECISION_END,
) -> pd.DataFrame:
    """Return the last open SSE trading day in every calendar month."""

    missing = [column for column in _REQUIRED_COLUMNS if column not in trade_calendar]
    if missing:
        raise ValueError(f"missing required columns: {','.join(missing)}")

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        raise ValueError("schedule start must not be after end")

    frame = trade_calendar.loc[
        trade_calendar["exchange"].astype(str).str.strip().eq("SSE")
        & trade_calendar["is_open"].astype(str).str.strip().eq("1")
    ].copy()
    frame["decision_date"] = pd.to_datetime(
        frame["cal_date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    frame = frame.loc[
        frame["decision_date"].notna()
        & frame["decision_date"].between(start_date, end_date, inclusive="both")
    ].copy()
    frame["calendar_month"] = frame["decision_date"].dt.to_period("M").astype(str)

    expected_months = pd.period_range(start_date, end_date, freq="M").astype(str).tolist()
    available = set(frame["calendar_month"])
    for month in expected_months:
        if month not in available:
            raise ValueError(f"missing open SSE month: {month}")

    result = (
        frame.groupby("calendar_month", sort=True, as_index=False)["decision_date"]
        .max()
        .sort_values("decision_date", kind="mergesort")
        .reset_index(drop=True)
    )
    result["schedule_version"] = A2_SCHEDULE_VERSION
    return result[["decision_date", "calendar_month", "schedule_version"]]
