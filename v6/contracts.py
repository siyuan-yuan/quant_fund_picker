"""Frozen names and thresholds shared by every V6 research stage."""
from enum import Enum


class UniverseLabel(str, Enum):
    FULL_PIT = "FULL-PIT"
    SURV_ADJ = "SURV-ADJ"


class AlphaMode(str, Enum):
    STATIC_CURRENT_WEIGHTS = "static_current_weights"
    DYNAMIC_PATH = "dynamic_path"


class TargetKind(str, Enum):
    RAW = "raw"
    STYLE = "style"
    SKILL = "skill"


HORIZON_MONTHS = (1, 3, 6, 12)
TARGET_LAGS = {1: 1, 3: 3, 6: 5, 12: 11}
MIN_CROSS_SECTION = 30
THIN_CROSS_SECTION = 100
DECISION_START = "2006-01-01"
DECISION_END = "2026-03-31"

A2_ELIGIBILITY_VERSION = "state_to_universe_eligibility_v1"
A2_SCHEDULE_VERSION = "monthly-sse-last-open-v1"
A2_TARGET_TYPES = frozenset(
    {
        "股票型",
        "指数型-股票",
        "指数型-海外股票",
        "混合型-偏股",
        "混合型-灵活",
    }
)
