from __future__ import annotations

import pandas as pd
import pytest

from v6.a2_events import classify_event_title, generate_legal_state_events


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "share_code": "400013",
                "family_key": "东方基金|东方保本混合",
                "name": "东方保本混合",
                "inception_date": "2011-04-14",
                "status": "存续",
            }
        ]
    )


def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_share_code": "400013",
                "family_key": "东方基金|东方保本混合",
                "query_code": "400013",
            }
        ]
    )


def test_multiple_documents_share_one_document_independent_event_id() -> None:
    metadata = pd.DataFrame(
        [
            {
                "fundCode": "400013",
                "uploadInfoId": str(index),
                "reportSendDate": "2017-04-27",
                "reportName": title,
                "legal_event_type": "LEGAL_TRANSFORMATION",
                "legal_subject": "东方成长收益平衡混合",
                "event_anchor_date": "2017-05-11",
            }
            for index, title in enumerate(
                ["转型公告", "转型后基金合同", "转型后招募说明书"], start=1
            )
        ]
    )

    events, evidence = generate_legal_state_events(_sample(), metadata, _aliases())
    transformed = events.loc[events["event_type"].eq("LEGAL_TRANSFORMATION")]

    assert len(transformed) == 1
    assert evidence.loc[evidence["event_id"].eq(transformed.iloc[0]["event_id"]), "source_document"].tolist() == [
        "400013_1.pdf",
        "400013_2.pdf",
        "400013_3.pdf",
    ]
    assert "source_document" not in transformed.columns


@pytest.mark.parametrize(
    "title",
    [
        "暂停大额申购和转换转入业务的公告",
        "新增代销机构并开通基金转换业务的公告",
        "调整基金转换费率的公告",
        "开放日常申购、赎回、转换及定投业务公告",
    ],
)
def test_transaction_conversion_is_not_legal_transformation(title: str) -> None:
    decision = classify_event_title(title)

    assert (decision.status, decision.event_type) == ("NOT_EVENT", "")


@pytest.mark.parametrize(
    "title",
    [
        "保本周期到期并转型为灵活配置混合型基金的公告",
        "基金变更为上市开放式基金并于2020年1月1日生效的公告",
        "封闭式基金终止上市并转型为开放式基金的公告",
    ],
)
def test_explicit_legal_transformation_title_is_positive(title: str) -> None:
    decision = classify_event_title(title)

    assert (decision.status, decision.event_type) == ("EVENT", "LEGAL_TRANSFORMATION")


def test_bare_share_conversion_requires_legal_context_review() -> None:
    decision = classify_event_title("关于原久富证券投资基金基金份额转换结果的公告")

    assert (decision.status, decision.event_type) == ("REVIEW_REQUIRED", "")


def test_inception_identity_is_stable_and_has_no_evidence_source_component() -> None:
    events, evidence = generate_legal_state_events(_sample(), pd.DataFrame(), _aliases())

    assert events["event_type"].tolist() == ["INCEPTION"]
    assert events["event_anchor_date"].tolist() == ["2011-04-14"]
    assert evidence.empty
    mutated = _sample().assign(unused_source="different.pdf")
    repeated, _ = generate_legal_state_events(mutated, pd.DataFrame(), _aliases())
    assert repeated["event_id"].tolist() == events["event_id"].tolist()

