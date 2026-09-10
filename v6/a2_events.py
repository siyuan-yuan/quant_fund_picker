"""Document-independent legal-state events for V6-G0-A2."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

import pandas as pd


A2_EVENT_GENERATION_VERSION = "a2-legal-event-v1"
_EVENT_TYPES = {
    "INCEPTION",
    "LEGAL_TRANSFORMATION",
    "MERGER_OR_SUCCESSION",
    "FUND_TYPE_CHANGE",
    "INVESTMENT_MANDATE_CHANGE",
    "TERMINATION",
}


@dataclass(frozen=True)
class EventTitleDecision:
    status: str
    event_type: str = ""


@dataclass(frozen=True)
class LegalStateEvent:
    event_id: str
    family_key: str
    event_type: str
    legal_subject: str
    event_anchor_date: str
    generation_version: str = A2_EVENT_GENERATION_VERSION


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def _date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else str(pd.Timestamp(parsed).normalize().date())


def _event_id(
    family_key: str, event_type: str, legal_subject: str, event_anchor_date: str
) -> str:
    identity = "|".join(
        (
            family_key,
            event_type,
            legal_subject,
            event_anchor_date,
            A2_EVENT_GENERATION_VERSION,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


_TRANSACTION_CONVERSION = re.compile(
    r"转换转入|转换转出|基金转换业务|开通.*转换|暂停.*转换|恢复.*转换|转换费率|"
    r"日常申购.*转换|申购.*赎回.*转换|代销.*转换"
)
_LEGAL_TRANSFORMATION = re.compile(
    r"转型为|转型运作|由.{0,80}转型|变更为.{0,40}(?:基金|LOF)|"
    r"(?:保本期|保本周期)到期.{0,80}转|封闭(?:期|式基金).{0,80}(?:届满|终止上市).{0,80}(?:转型|转换)"
)


def classify_event_title(title: object) -> EventTitleDecision:
    """Classify title semantics without consulting parsed state values."""

    normalized = _text(title)
    if not normalized:
        return EventTitleDecision("NOT_EVENT")
    if _TRANSACTION_CONVERSION.search(normalized):
        return EventTitleDecision("NOT_EVENT")
    if _LEGAL_TRANSFORMATION.search(normalized):
        return EventTitleDecision("EVENT", "LEGAL_TRANSFORMATION")
    if re.search(r"合并|吸收合并|基金继承", normalized):
        return EventTitleDecision("EVENT", "MERGER_OR_SUCCESSION")
    if re.search(r"变更基金类型|基金类型变更", normalized):
        return EventTitleDecision("EVENT", "FUND_TYPE_CHANGE")
    if re.search(r"变更投资范围|调整投资比例|投资范围变更", normalized):
        return EventTitleDecision("EVENT", "INVESTMENT_MANDATE_CHANGE")
    if re.search(r"终止基金合同|基金合同终止|进入清算|基金清算", normalized):
        return EventTitleDecision("EVENT", "TERMINATION")
    if "转换" in normalized:
        return EventTitleDecision("REVIEW_REQUIRED")
    return EventTitleDecision("NOT_EVENT")


def _family_by_code(sample: pd.DataFrame, aliases: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in sample.to_dict("records"):
        code = _text(row.get("share_code")).split(".")[0].zfill(6)
        family = str(row.get("family_key") or "").strip()
        if code and family:
            mapping[code] = family
    for row in aliases.to_dict("records"):
        code = _text(row.get("query_code")).split(".")[0].zfill(6)
        family = str(row.get("family_key") or "").strip()
        if code and family:
            mapping[code] = family
    return mapping


def _document_stage(title: str, report_code: str) -> str:
    if "基金合同" in title:
        return "FUND_CONTRACT"
    if "招募说明书" in title:
        return "PROSPECTUS"
    if "决议" in title and ("生效" in title or "表决结果" in title):
        return "EFFECTIVENESS_RESOLUTION"
    if report_code.startswith("FC") or "公告" in title:
        return "ANNOUNCEMENT"
    return "UNDETERMINED"


def generate_legal_state_events(
    sample: pd.DataFrame,
    metadata: pd.DataFrame,
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate legal events and attach official documents as 1:N evidence candidates."""

    event_by_id: dict[str, dict[str, object]] = {}
    evidence_rows: list[dict[str, object]] = []

    for row in sample.to_dict("records"):
        family = str(row.get("family_key") or "").strip()
        subject = _text(row.get("name")) or family
        anchor = _date(row.get("inception_date"))
        if not family or not anchor:
            raise ValueError("every sampled family requires family_key and inception_date")
        event = LegalStateEvent(
            _event_id(family, "INCEPTION", subject, anchor),
            family,
            "INCEPTION",
            subject,
            anchor,
        )
        event_by_id[event.event_id] = asdict(event)
        end_date = _date(row.get("end_date"))
        if end_date:
            terminal = LegalStateEvent(
                _event_id(family, "TERMINATION", subject, end_date),
                family,
                "TERMINATION",
                subject,
                end_date,
            )
            event_by_id[terminal.event_id] = asdict(terminal)

    family_by_code = _family_by_code(sample, aliases)
    if not metadata.empty:
        for row in metadata.to_dict("records"):
            code = _text(row.get("fundCode") or row.get("share_code")).split(".")[0].zfill(6)
            family = family_by_code.get(code, "")
            if not family:
                continue
            explicit_type = _text(row.get("legal_event_type"))
            decision = (
                EventTitleDecision("EVENT", explicit_type)
                if explicit_type in _EVENT_TYPES
                else classify_event_title(row.get("reportName") or row.get("report_name"))
            )
            if decision.status != "EVENT":
                continue
            event_type = decision.event_type
            subject = _text(row.get("legal_subject")) or _text(row.get("fundShortName"))
            if not subject:
                matching = sample.loc[sample["family_key"].astype(str).eq(family)]
                subject = _text(matching.iloc[0].get("name")) if not matching.empty else family
            anchor = _date(row.get("event_anchor_date")) or _date(
                row.get("reportSendDate") or row.get("known_at")
            )
            if not anchor:
                continue
            event_id = _event_id(family, event_type, subject, anchor)
            event = LegalStateEvent(event_id, family, event_type, subject, anchor)
            event_by_id[event_id] = asdict(event)
            upload = _text(row.get("uploadInfoId") or row.get("upload_info_id"))
            source = f"{code}_{upload}.pdf" if code and upload else ""
            if not source:
                continue
            title = str(row.get("reportName") or row.get("report_name") or "").strip()
            evidence_rows.append(
                {
                    "event_id": event_id,
                    "source_document": source,
                    "known_at": _date(row.get("reportSendDate") or row.get("known_at")),
                    "document_stage": _document_stage(title, _text(row.get("reportCode"))),
                    "report_code": _text(row.get("reportCode") or row.get("report_code")),
                    "report_name": title,
                }
            )

    event_columns = [field.name for field in LegalStateEvent.__dataclass_fields__.values()]
    events = pd.DataFrame(event_by_id.values(), columns=event_columns).sort_values(
        ["event_anchor_date", "family_key", "event_type", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    evidence_columns = [
        "event_id", "source_document", "known_at", "document_stage", "report_code", "report_name"
    ]
    evidence = pd.DataFrame(evidence_rows, columns=evidence_columns)
    if not evidence.empty:
        evidence = evidence.drop_duplicates(
            ["event_id", "source_document"], keep="first"
        ).sort_values(["event_id", "source_document"], kind="mergesort").reset_index(drop=True)
    return events, evidence
