from v6.contracts import (
    AlphaMode,
    HORIZON_MONTHS,
    MIN_CROSS_SECTION,
    TARGET_LAGS,
    THIN_CROSS_SECTION,
    TargetKind,
    UniverseLabel,
)


def test_v6_contract_values_are_frozen():
    assert UniverseLabel.FULL_PIT.value == "FULL-PIT"
    assert UniverseLabel.SURV_ADJ.value == "SURV-ADJ"
    assert AlphaMode.STATIC_CURRENT_WEIGHTS.value == "static_current_weights"
    assert AlphaMode.DYNAMIC_PATH.value == "dynamic_path"
    assert TargetKind.RAW.value == "raw"
    assert TargetKind.STYLE.value == "style"
    assert TargetKind.SKILL.value == "skill"
    assert HORIZON_MONTHS == (1, 3, 6, 12)
    assert TARGET_LAGS == {1: 1, 3: 3, 6: 5, 12: 11}
    assert MIN_CROSS_SECTION == 30
    assert THIN_CROSS_SECTION == 100
