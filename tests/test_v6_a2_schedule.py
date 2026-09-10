from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v6.a2_schedule import build_a2_monthly_schedule


def test_schedule_uses_last_open_sse_day_per_month() -> None:
    calendar = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE", "SSE", "SSE", "SZSE"],
            "cal_date": ["20200123", "20200124", "20200131", "20200228", "20200229"],
            "is_open": [1, 0, 0, 1, 1],
        }
    )

    result = build_a2_monthly_schedule(
        calendar, start="2020-01-01", end="2020-02-29"
    )

    assert result["decision_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-23",
        "2020-02-28",
    ]
    assert result["calendar_month"].tolist() == ["2020-01", "2020-02"]


def test_full_schedule_has_frozen_range_count_and_pandemic_month() -> None:
    calendar = pd.read_csv(Path("cache/tushare/bulk/trade_cal.csv"), dtype=str)

    result = build_a2_monthly_schedule(calendar)

    assert len(result) == 243
    assert result["calendar_month"].iloc[[0, -1]].tolist() == ["2006-01", "2026-03"]
    assert result["decision_date"].is_monotonic_increasing
    assert result.loc[result["calendar_month"].eq("2020-01"), "decision_date"].item() == pd.Timestamp("2020-01-23")


def test_schedule_rejects_a_missing_calendar_month() -> None:
    calendar = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE"],
            "cal_date": ["20200123", "20200331"],
            "is_open": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="missing open SSE month: 2020-02"):
        build_a2_monthly_schedule(calendar, start="2020-01-01", end="2020-03-31")


@pytest.mark.parametrize("missing", ["exchange", "cal_date", "is_open"])
def test_schedule_requires_complete_calendar_schema(missing: str) -> None:
    calendar = pd.DataFrame(
        {"exchange": ["SSE"], "cal_date": ["20200123"], "is_open": [1]}
    ).drop(columns=missing)

    with pytest.raises(ValueError, match=f"missing required columns: {missing}"):
        build_a2_monthly_schedule(calendar, start="2020-01-01", end="2020-01-31")
