from __future__ import annotations

import pandas as pd
import pytest

from v6.csrc_fund_disclosure import (
    assess_document_applicability,
    classify_mixed_fund,
    extract_effective_date,
    extract_equity_constraint,
    extract_equity_constraint_candidates,
    extract_investment_sections,
    extract_pdf_text,
    proposed_state_is_pending,
)


def test_section_pipeline_normalizes_headers_page_breaks_and_preserves_heading_provenance() -> None:
    text = (
        "某基金招募说明书更新 第1页共20页\n第九部分 投资目标\n追求长期回报。\n"
        "\f某基金招募说明书更新 第2页共20页\n第十部分 投资范围\n"
        "股票资产占基金资产的比例为60%-95%。\n第十一部分 投资策略\n动态配置。"
    )

    sections = extract_investment_sections(text)

    investment_range = next(item for item in sections if item.heading == "投资范围")
    assert investment_range.page_start == 2
    assert "第2页共20页" not in investment_range.text
    assert investment_range.text == "股票资产占基金资产的比例为60%-95%。"
    assert investment_range.normalized_text == "股票资产占基金资产的比例为60%-95%。"


def test_document_applicability_separates_obvious_operational_notice_from_legal_material() -> None:
    assert assess_document_applicability(
        report_code="FC900090", report_name="关于基金经理变更的公告", text="基金经理发生变更。"
    ) == "NOT_APPLICABLE"
    assert assess_document_applicability(
        report_code="FC900090",
        report_name="基金转型并修改基金合同的公告",
        text="第八部分 投资范围 股票资产占基金资产的比例为60%-95%。",
    ) == "APPLICABLE"
    assert assess_document_applicability(
        report_code="FC900090",
        report_name="关于基金经理变更的公告",
        text="第八部分 投资范围\n股票资产占基金资产的比例为60%-95%。",
    ) == "NOT_APPLICABLE"


def test_pending_proposal_detection_is_local_and_does_not_reject_current_contract_governance() -> None:
    current_contract = "基金合同现行有效。修改基金合同需经持有人大会表决通过后方可生效。"
    proposal = "本次拟修改投资范围，尚需持有人大会表决通过后方可生效。"

    assert proposed_state_is_pending(current_contract, report_name="基金合同") is False
    assert proposed_state_is_pending(
        proposal, report_name="持有人大会拟修改基金合同的公告"
    ) is True
    assert proposed_state_is_pending(
        "本议案拟变更基金合同中的投资范围。"
        "上述事项尚需基金份额持有人大会表决通过后方可生效。",
        report_name="关于召开基金份额持有人大会的公告",
    ) is True
    assert proposed_state_is_pending(
        "本次大会审议通过的事项，自表决通过之日起生效。",
        report_name="关于召开基金份额持有人大会的公告",
    ) is True
    assert proposed_state_is_pending(
        "本次更新招募说明书中的基金经理信息。"
        "修改基金合同需经基金份额持有人大会表决通过后方可生效。",
        report_name="招募说明书更新",
    ) is False
    assert proposed_state_is_pending(
        "八、投资范围 股票资产占基金资产的比例为60%-95%。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "八、投资范围 股票资产占基金资产的比例为60%-95%。",
        report_name="关于拟修改基金合同的公告",
    ) is True
    assert proposed_state_is_pending(
        "旧议案已经持有人大会表决通过，旧合同自2020年1月1日起生效。现附新投资范围。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "本议案已经持有人大会表决通过。上述事项自2024年6月1日起生效。",
        report_name="关于拟修改基金合同的表决结果公告",
    ) is False
    assert proposed_state_is_pending(
        "若本议案已经持有人大会表决通过，自2025年1月1日起生效。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "本议案如已经持有人大会表决通过，自2025年1月1日起生效。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "八、投资范围 股票资产占基金资产的比例为60%-95%。",
        report_name="关于变更投资范围的议案",
    ) is True
    assert proposed_state_is_pending(
        "八、投资范围 股票资产占基金资产的比例为60%-95%。",
        report_name="基金合同（征求意见稿）",
    ) is True
    assert proposed_state_is_pending(
        "本草案未经持有人大会表决通过。若获通过，自2025年1月1日起生效。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "本次更新不涉及变更。任何主体拟修改基金合同，尚需持有人大会表决通过后方可生效。",
        report_name="招募说明书更新",
    ) is False
    assert proposed_state_is_pending(
        "本议案拟经持有人大会表决通过，自2025年1月1日起生效。",
        report_name="基金合同（草案）",
    ) is True
    assert proposed_state_is_pending(
        "本议案尚未经持有人大会审议通过。旧议案已经表决通过，自2020年1月1日起生效。",
        report_name="关于变更投资范围的议案",
    ) is True
    assert proposed_state_is_pending(
        "本议案已经持有人大会表决通过。上述事项自2024年6月1日起生效。"
        "旧制度规定修改其他事项尚需持有人大会表决通过后方可生效。",
        report_name="关于拟修改基金合同的表决结果公告",
    ) is False
    for unresolved in (
        "本议案拟修改投资范围，旧议案已经表决通过，自2020年1月1日起生效。",
        "本议案拟提交持有人大会审议，经审议通过，自2025年1月1日起生效。",
        "本议案没有经持有人大会表决通过，自2025年1月1日起生效。",
        "本议案已经表决通过，新合同生效日期另行通知，旧合同自2020年1月1日起生效。",
    ):
        assert proposed_state_is_pending(
            unresolved, report_name="关于拟修改基金合同的表决结果公告"
        ) is True
    assert proposed_state_is_pending(
        "本次更新不涉及变更，任何主体拟修改基金合同，尚需持有人大会表决通过后方可生效。",
        report_name="招募说明书更新",
    ) is False
    assert proposed_state_is_pending(
        "本议案未经持有人大会表决通过。",
        report_name="关于基金份额持有人大会表决结果公告",
    ) is True
    assert proposed_state_is_pending(
        "若本议案经持有人大会表决通过，自2025年1月1日起生效。",
        report_name="关于基金份额持有人大会表决结果公告",
    ) is True


def test_clause_pipeline_enumerates_distinct_states_inside_one_section() -> None:
    candidates = extract_equity_constraint_candidates(
        "股票资产占基金资产的比例为60%-95%；股票资产占基金资产的比例为0%-30%。"
    )

    assert {(item.equity_min_pct, item.equity_max_pct) for item in candidates} == {
        (60.0, 95.0), (0.0, 30.0)
    }


def test_clause_pipeline_selects_post_conversion_phase_before_ambiguity_check() -> None:
    candidates = extract_equity_constraint_candidates(
        "封闭期内股票资产占基金资产的比例为60%-100%；"
        "封闭期届满，转为上市开放式基金（LOF）后，"
        "股票资产占基金资产的比例为60%-95%。"
    )

    assert [(item.equity_min_pct, item.equity_max_pct) for item in candidates] == [(60.0, 95.0)]


def test_clause_phase_binding_ignores_later_non_equity_open_period_marker() -> None:
    candidates = extract_equity_constraint_candidates(
        "封闭期内股票投资占基金资产的60%-100%；"
        "封闭期届满转为上市开放式基金（LOF）后股票投资占基金资产的60%-95%；"
        "开放期内现金资产不低于基金资产净值的5%。"
    )

    assert [(item.equity_min_pct, item.equity_max_pct) for item in candidates] == [(60.0, 95.0)]


@pytest.mark.parametrize("heading", ["八、投资范围", "（八）投资范围", "8. 投资范围"])
def test_section_pipeline_recognizes_common_legal_heading_formats(heading: str) -> None:
    sections = extract_investment_sections(
        f"{heading}\n股票资产占基金资产的比例为60%-95%。\n九、基金的费用"
    )

    assert len(sections) == 1
    assert sections[0].heading == "投资范围"
    assert "基金的费用" not in sections[0].text


def test_section_pipeline_does_not_turn_combined_asset_constraint_into_equity_clause() -> None:
    section = extract_investment_sections(
        "第八部分 投资范围\n股票、债券等资产合计占基金资产的80%-95%。"
    )[0]

    with pytest.raises(ValueError, match="No explicit equity allocation"):
        extract_equity_constraint(section.text)


def test_extracts_minimum_equity_constraint_from_official_prospectus_text() -> None:
    text = "本基金股票投资占基金资产的比例不低于60%；"

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 60.0
    assert constraint.equity_max_pct == 100.0
    assert constraint.evidence == "本基金股票投资占基金资产的比例不低于60%"
    assert classify_mixed_fund(constraint) == "混合型-偏股"


def test_extracts_maximum_equity_constraint_and_excludes_mixed_bond_fund() -> None:
    text = "（1）本基金投资于股票资产的比例不高于基金资产的 30%；"

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 30.0
    assert classify_mixed_fund(constraint) == "混合型-偏债"


def test_extracts_maximum_when_equity_term_includes_depositary_receipts() -> None:
    text = "本基金投资于股票资产及存托凭证的比例不高于基金资产的30%。"

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 30.0
    assert constraint.evidence == text.rstrip("。")
    assert classify_mixed_fund(constraint) == "混合型-偏债"


@pytest.mark.parametrize(
    "text",
    [
        "股票等权益类资产占基金资产的比例不超过30%，债券资产比例不低于70%。",
        "投资于权益类资产的比例不超过基金资产的30%。",
    ],
)
def test_extracts_maximum_from_observed_equity_asset_phrases(text: str) -> None:
    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 30.0
    assert classify_mixed_fund(constraint) == "混合型-偏债"


def test_extracts_range_when_wording_says_ratio_range() -> None:
    constraint = extract_equity_constraint(
        "本基金股票资产占基金资产的比例范围为0%—95%。"
    )

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 95.0
    assert classify_mixed_fund(constraint) is None


@pytest.mark.parametrize(
    ("text", "expected_low", "expected_high"),
    [
        ("本基金股票投资占基金资产的比例范围为0-30%。", 0.0, 30.0),
        (
            "本基金股票（含存托凭证）投资占基金资产的比例范围为0-30%。",
            0.0,
            30.0,
        ),
        ("本基金的股票投资比例为基金资产的60%-95%。", 60.0, 95.0),
        ("本基金股票及存托凭证资产占基金资产的比例为60%-95%。", 60.0, 95.0),
        ("本基金投资于股票的比例为基金资产的60%-95%。", 60.0, 95.0),
        (
            "本基金的股票（含存托凭证）资产投资比例为基金资产的60%-95%。",
            60.0,
            95.0,
        ),
        ("股票及存托凭证投资占基金资产的比例为0%–30%。", 0.0, 30.0),
    ],
)
def test_extracts_ranges_from_observed_modern_prospectus_phrases(
    text: str, expected_low: float, expected_high: float
) -> None:
    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == expected_low
    assert constraint.equity_max_pct == expected_high


@pytest.mark.parametrize(
    ("text", "expected_low", "expected_high"),
    [
        ("本基金股票投资占基金资产的0-30%。", 0.0, 30.0),
        ("股票资产（含存托凭证）的投资比例为基金资产的60%-95%。", 60.0, 95.0),
        (
            "本基金投资于权益类资产（股票、存托凭证、股票型基金、混合型基金）"
            "的投资比例范围为基金资产的60%-85%。",
            60.0,
            85.0,
        ),
        ("本基金的股票（含存托凭证）投资比例范围为：60%-95%。", 60.0, 95.0),
        ("股票资产（含存托凭证）占基金净值的比例范围为60%-95%。", 60.0, 95.0),
        ("股票资产占基金总资产60%－95%。", 60.0, 95.0),
        ("封闭期内股票投资占基金资产的60%-100%。", 60.0, 100.0),
    ],
)
def test_extracts_ranges_from_second_observed_phrase_family(
    text: str, expected_low: float, expected_high: float
) -> None:
    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == expected_low
    assert constraint.equity_max_pct == expected_high


@pytest.mark.parametrize(
    ("text", "expected_low", "expected_high"),
    [
        ("股票资产配置比例为60%-95%，债券资产配置比例为0-15%。", 60.0, 95.0),
        (
            "股票投资比例的变动范围为：基金资产净值的30%-95%；"
            "债券投资比例的变动范围为基金资产净值的0%-65%。",
            30.0,
            95.0,
        ),
    ],
)
def test_extracts_ranges_from_legacy_allocation_wording(
    text: str, expected_low: float, expected_high: float
) -> None:
    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == expected_low
    assert constraint.equity_max_pct == expected_high


def test_extracts_bare_stock_range_after_explicit_asset_allocation_heading() -> None:
    constraint = extract_equity_constraint(
        "本基金大类资产的投资比例范围是：股票50％－70％，"
        "债券25％－45％，现金保持在5％以上。"
    )

    assert constraint.equity_min_pct == 50.0
    assert constraint.equity_max_pct == 70.0
    assert constraint.evidence == "股票50％－70％"


def test_extracts_stock_and_depositary_receipt_range_joined_by_dunhao() -> None:
    constraint = extract_equity_constraint(
        "转为上市开放式基金（LOF）后，"
        "股票、存托凭证占基金资产的比例为60%-95%。"
    )

    assert constraint.equity_min_pct == 60.0
    assert constraint.equity_max_pct == 95.0
    assert constraint.evidence == "股票、存托凭证占基金资产的比例为60%-95%"


def test_extracts_minimum_with_fund_net_asset_denominator_after_comparator() -> None:
    constraint = extract_equity_constraint(
        "本基金投资股票资产及存托凭证的比例不低于基金资产净值的40%。"
    )

    assert constraint.equity_min_pct == 40.0
    assert constraint.equity_max_pct == 100.0
    assert constraint.evidence == "本基金投资股票资产及存托凭证的比例不低于基金资产净值的40%"


@pytest.mark.parametrize(
    ("text", "expected_low", "expected_high"),
    [
        (
            "封闭期内股票投资占基金资产的60%-100%；"
            "封闭期届满，本基金转换为上市开放式基金（LOF）后，"
            "股票投资占基金资产的60%-95%。",
            60.0,
            95.0,
        ),
        (
            "封闭期内股票资产占基金资产的比例范围为0%-100%；"
            "开放期内或本基金转型为上市开放式基金（LOF）后"
            "股票资产占基金资产的比例范围为0%-95%。",
            0.0,
            95.0,
        ),
        (
            "在封闭运作期，本基金投资组合中股票投资比例为基金资产的0%-100%；"
            "封闭运作期届满，转为上市开放式基金（LOF）后："
            "本基金投资组合中股票投资比例为基金资产的50%-95%。",
            50.0,
            95.0,
        ),
    ],
)
def test_prefers_post_conversion_constraint_over_expired_closed_period(
    text: str, expected_low: float, expected_high: float
) -> None:
    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == expected_low
    assert constraint.equity_max_pct == expected_high


def test_selects_closed_period_constraint_when_document_is_still_in_closed_phase() -> None:
    text = (
        "封闭期内股票资产占基金资产的比例范围为0%-100%；"
        "转为上市开放式基金（LOF）后股票资产占基金资产的比例范围为0%-95%。"
    )

    constraint = extract_equity_constraint(text, phase_hint="closed")

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 100.0


def test_uses_nearest_phase_marker_for_each_constraint_candidate() -> None:
    text = (
        "本基金转型为上市开放式基金（LOF）后采用灵活投资策略。"
        + "风险说明" * 80
        + "封闭期内股票资产占基金资产的比例范围为0%-100%；"
        + "开放期内或本基金转型为上市开放式基金（LOF）后"
        + "股票资产占基金资产的比例范围为0%-95%。"
    )

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 95.0


def test_ignores_product_summary_page_header_inside_equity_clause() -> None:
    text = (
        "封闭期内股票资产占基金资产的比例范围为0%-100%；"
        "本基金转型为上市开放式基金（LOF）后股票资产占基金资"
        "平安鼎越灵活配置混合型证券投资基金基金产品资料概要更新第2页共5页"
        "产的比例范围为0%-95%。"
    )

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 95.0


def test_extracts_explicit_equity_range_but_does_not_invent_subtype() -> None:
    text = "股票资产占基金资产的比例为 0%-95%。"

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 0.0
    assert constraint.equity_max_pct == 95.0
    assert classify_mixed_fund(constraint) is None


def test_extracts_explicit_contract_effective_date() -> None:
    text = (
        "自 2019 年 1 月 11 日起，原基金合同失效，"
        "《华安量化多因子混合型证券投资基金（LOF）基金合同》生效。"
    )

    assert extract_effective_date(text) == pd.Timestamp("2019-01-11")


def test_does_not_bind_securities_fund_law_implementation_date_to_contract() -> None:
    text = extract_pdf_text(
        "output/v6/g0_full_pit/csrc_stratified/pdf/400013_321232.pdf"
    )

    assert extract_effective_date(text) is None


def test_does_not_bind_short_law_effective_sentence_to_fund_contract() -> None:
    text = (
        "《中华人民共和国证券投资基金法》自2004年6月1日起施行，"
        "根据本基金基金合同约定，本基金合同生效。"
    )

    assert extract_effective_date(text) is None


def test_proposed_contract_with_placeholder_date_does_not_reuse_historical_date() -> None:
    text = (
        "本基金于2017年5月11日根据《东方保本混合型开放式证券投资基金基金合同》"
        "的约定转型而来。基金管理人提议修改本基金的投资范围。"
        "根据该持有人大会决议，自2018年XX月XX日起，"
        "《东方成长收益灵活配置混合型证券投资基金基金合同》生效。"
        "历史沿革：东方保本自2017年5月11日起转型为东方成长收益平衡混合型证券投资基金，"
        "《东方成长收益平衡混合型证券投资基金基金合同》于2017年5月11日起生效。"
    )

    assert extract_effective_date(text) is None


@pytest.mark.parametrize(
    "placeholder_clause",
    [
        "《新基金基金合同》自2018年XX月XX日起生效。",
        "《基金合同》于XXXX年XX月XX日生效。",
    ],
)
def test_contract_bound_placeholder_date_fails_closed_before_historical_date(
    placeholder_clause: str,
) -> None:
    text = (
        placeholder_clause
        + "历史沿革：《旧基金基金合同》自2017年5月11日起生效。"
    )

    assert extract_effective_date(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("自2017年5月11日起转型生效。", "2017-05-11"),
        ("《某基金基金合同》自2007年2月12日起生效。", "2007-02-12"),
        ("《基金合同》自2007年2月12日起生效。", "2007-02-12"),
        ("《基金合同》于2007年2月12日生效。", "2007-02-12"),
    ],
)
def test_extracts_locally_bound_fund_effective_dates(
    text: str, expected: str
) -> None:
    assert extract_effective_date(text) == pd.Timestamp(expected)


def test_returns_none_when_distinct_locally_bound_effective_dates_are_ambiguous() -> None:
    text = (
        "《甲基金基金合同》自2017年5月11日起生效。"
        "《乙基金基金合同》自2018年1月17日起生效。"
    )

    assert extract_effective_date(text) is None


def test_extracts_legacy_index_constituent_equity_floor() -> None:
    text = (
        "标的指数成份股及其备选成份股的投资比例不低于基金资产的90%；"
        "其他金融工具的投资比例为基金资产的5%-10%。"
    )

    constraint = extract_equity_constraint(text)

    assert constraint.equity_min_pct == 90.0
    assert constraint.equity_max_pct == 100.0


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("501039_initial_prospectus_20170810.pdf", "混合型-偏债"),
        ("160415_initial_prospectus_20190110.pdf", "混合型-偏股"),
    ],
)
def test_extracts_constraint_from_downloaded_official_pdf(
    filename: str, expected_type: str
) -> None:
    path = "data/v6/official/smoke/" + filename

    constraint = extract_equity_constraint(extract_pdf_text(path))

    assert classify_mixed_fund(constraint) == expected_type
