from __future__ import annotations

import pytest

from v6.a2_eligibility import state_to_universe_eligibility_v1


@pytest.mark.parametrize(
    ("minimum", "status", "target_type"),
    [
        (59.999, "UNRESOLVED_ELIGIBILITY", ""),
        (60.0, "ELIGIBLE", "混合型-偏股"),
    ],
)
def test_mixed_equity_minimum_boundary_is_frozen(
    minimum: float, status: str, target_type: str
) -> None:
    result = state_to_universe_eligibility_v1("混合型", minimum, 95.0)

    assert (result.status, result.target_type) == (status, target_type)


@pytest.mark.parametrize(
    ("maximum", "status", "target_type"),
    [
        (30.0, "EXCLUDED", ""),
        (30.001, "ELIGIBLE", "混合型-灵活"),
    ],
)
def test_flexible_upper_boundary_is_frozen(
    maximum: float, status: str, target_type: str
) -> None:
    result = state_to_universe_eligibility_v1("灵活配置混合型", 0.0, maximum)

    assert (result.status, result.target_type) == (status, target_type)


@pytest.mark.parametrize(
    ("legal_type", "target_type"),
    [
        ("股票型", "股票型"),
        ("指数型-股票", "指数型-股票"),
        ("指数型-海外股票", "指数型-海外股票"),
    ],
)
def test_explicit_equity_types_require_the_same_sixty_percent_minimum(
    legal_type: str, target_type: str
) -> None:
    assert state_to_universe_eligibility_v1(legal_type, 60.0, 100.0).target_type == target_type
    assert state_to_universe_eligibility_v1(legal_type, 59.999, 100.0).status == "UNRESOLVED_ELIGIBILITY"


@pytest.mark.parametrize(
    "legal_type",
    ["债券型", "货币市场型", "FOF", "商品型", "REIT", "保本混合型", "平衡混合型", "稳健收益混合型", "其他型"],
)
def test_explicit_non_target_mandate_precedes_numeric_bounds(legal_type: str) -> None:
    result = state_to_universe_eligibility_v1(legal_type, 60.0, 95.0)

    assert (result.status, result.target_type, result.exclusion_reason) == (
        "EXCLUDED",
        "",
        "EXPLICIT_NON_TARGET_MANDATE",
    )


def test_mixed_upper_bound_at_thirty_is_bond_biased() -> None:
    result = state_to_universe_eligibility_v1("混合型", 0.0, 30.0)

    assert (result.status, result.exclusion_reason) == ("EXCLUDED", "MIXED_BOND_BIASED")


def test_explicit_equity_biased_mixed_type_preserves_target_type() -> None:
    result = state_to_universe_eligibility_v1("混合型-偏股", 60.0, 95.0)

    assert (result.status, result.target_type) == ("ELIGIBLE", "混合型-偏股")


@pytest.mark.parametrize(
    ("legal_type", "minimum", "maximum", "reason"),
    [
        ("", 60.0, 95.0, "MISSING_OR_AMBIGUOUS_LEGAL_TYPE"),
        ("混合型", None, 95.0, "MISSING_EQUITY_BOUND"),
        ("股票型", 60.0, None, "MISSING_EQUITY_BOUND"),
        ("混合型", -1.0, 95.0, "INVALID_EQUITY_INTERVAL"),
        ("混合型", 96.0, 95.0, "INVALID_EQUITY_INTERVAL"),
        ("混合型", 0.0, 101.0, "INVALID_EQUITY_INTERVAL"),
        ("未知混合类别", 60.0, 95.0, "NON_TARGET_OR_AMBIGUOUS_MANDATE"),
    ],
)
def test_missing_invalid_or_ambiguous_state_never_enters_universe(
    legal_type: str, minimum: float | None, maximum: float | None, reason: str
) -> None:
    result = state_to_universe_eligibility_v1(legal_type, minimum, maximum)

    assert (result.status, result.target_type, result.exclusion_reason) == (
        "UNRESOLVED_ELIGIBILITY",
        "",
        reason,
    )
