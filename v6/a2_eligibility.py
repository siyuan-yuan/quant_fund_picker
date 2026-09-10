"""Frozen V6-G0-A2 legal-state to universe eligibility mapping."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .contracts import A2_ELIGIBILITY_VERSION, A2_TARGET_TYPES


@dataclass(frozen=True)
class EligibilityResult:
    status: str
    target_type: str = ""
    exclusion_reason: str = ""
    mapping_version: str = A2_ELIGIBILITY_VERSION


_EXPLICIT_NON_TARGET = re.compile(
    r"债券|货币|FOF|基金中基金|商品|REIT|不动产投资信托|保本|平衡|稳健|稳定|其他",
    re.IGNORECASE,
)
_FLEXIBLE_TYPES = {"混合型-灵活", "灵活配置混合型"}
_DIRECT_TARGET_TYPES = {"股票型", "指数型-股票", "指数型-海外股票"}


def _number(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def state_to_universe_eligibility_v1(
    legal_fund_type: str,
    equity_min_pct: float | None,
    equity_max_pct: float | None,
) -> EligibilityResult:
    """Map an evidenced legal state using the frozen A2 precedence rules."""

    legal_type = re.sub(r"\s+", "", str(legal_fund_type or ""))
    if legal_type and _EXPLICIT_NON_TARGET.search(legal_type):
        return EligibilityResult("EXCLUDED", exclusion_reason="EXPLICIT_NON_TARGET_MANDATE")
    if not legal_type:
        return EligibilityResult(
            "UNRESOLVED_ELIGIBILITY",
            exclusion_reason="MISSING_OR_AMBIGUOUS_LEGAL_TYPE",
        )

    minimum = _number(equity_min_pct)
    maximum = _number(equity_max_pct)
    if minimum is None or maximum is None:
        return EligibilityResult("UNRESOLVED_ELIGIBILITY", exclusion_reason="MISSING_EQUITY_BOUND")
    if not 0.0 <= minimum <= maximum <= 100.0:
        return EligibilityResult("UNRESOLVED_ELIGIBILITY", exclusion_reason="INVALID_EQUITY_INTERVAL")

    if "混合型" in legal_type and maximum <= 30.0:
        return EligibilityResult("EXCLUDED", exclusion_reason="MIXED_BOND_BIASED")
    if legal_type in _DIRECT_TARGET_TYPES:
        if minimum >= 60.0:
            return EligibilityResult("ELIGIBLE", target_type=legal_type)
        return EligibilityResult(
            "UNRESOLVED_ELIGIBILITY",
            exclusion_reason="TARGET_MINIMUM_NOT_ESTABLISHED",
        )
    if legal_type in {"混合型", "混合型-偏股"} and minimum >= 60.0:
        return EligibilityResult("ELIGIBLE", target_type="混合型-偏股")
    if legal_type in _FLEXIBLE_TYPES and maximum > 30.0:
        return EligibilityResult("ELIGIBLE", target_type="混合型-灵活")
    return EligibilityResult(
        "UNRESOLVED_ELIGIBILITY",
        exclusion_reason="NON_TARGET_OR_AMBIGUOUS_MANDATE",
    )


assert set(A2_TARGET_TYPES) == {
    "股票型",
    "指数型-股票",
    "指数型-海外股票",
    "混合型-偏股",
    "混合型-灵活",
}
