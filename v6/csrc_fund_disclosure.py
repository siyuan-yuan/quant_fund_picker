from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike

import pandas as pd
from pypdf import PdfReader


_EQUITY_TERM = (
    r"(?:本基金的?)?(?:"
    r"(?:投资于?)?股票(?:(?:（含存托凭证）|\(含存托凭证\)|及存托凭证|、存托凭证)"
    r"(?:资产)?|资产(?:及存托凭证)?)?(?:投资)?|"
    r"股票等权益类资产|投资于权益类资产|权益类资产|"
    r"标的指数成份股及其备选成份股)"
    r"(?:（[^（）]{1,80}）|\([^()]{1,80}\))?"
)
_ASSET_TERM = (
    r"(?:占基金(?:资产|净值|总资产)(?:的比例|的)?|"
    r"的?投资比例(?:的变动范围)?|的?比例(?:的变动范围)?|"
    r"配置比例|比例)"
)
_NUMBER = r"(?P<value>\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class EquityConstraint:
    equity_min_pct: float
    equity_max_pct: float
    evidence: str


@dataclass(frozen=True)
class InvestmentSection:
    heading: str
    text: str
    normalized_text: str
    page_start: int


_INVESTMENT_HEADINGS = ("投资目标", "投资范围", "投资策略", "资产配置", "投资限制")
_PAGE_HEADER = re.compile(r"^.*?第\s*\d+\s*页\s*共\s*\d+\s*页\s*$")
_HEADING_PREFIX = (
    r"(?:(?:第[一二三四五六七八九十百零〇0-9]+(?:部分|章|节)|"
    r"[（(]?[一二三四五六七八九十百零〇0-9]+[）)]?)[：:、.．\s]*)?"
)
_STRUCTURED_HEADING = re.compile(
    rf"^{_HEADING_PREFIX}"
    r"(?P<heading>投资目标|投资范围|投资策略|资产配置|投资限制)"
    r"[：:\s]*(?P<rest>.*)$"
)
_ANY_NUMBERED_HEADING = re.compile(
    r"^(?:(?:第[一二三四五六七八九十百零〇0-9]+(?:部分|章|节)|"
    r"[（(]?[一二三四五六七八九十百零〇0-9]+[）)]?)[：:、.．\s]+)\S+"
)


def assess_document_applicability(
    *, report_code: str = "", report_name: str = "", text: str = ""
) -> str:
    """Classify whether a document can legally define the fund's investment state."""

    title = re.sub(r"\s+", "", str(report_name or ""))
    compact = re.sub(r"\s+", "", str(text or ""))[:4000]
    legal_title_markers = (
        "招募说明书", "基金合同", "投资范围", "资产配置", "转型", "变更注册", "持有人大会"
    )
    if any(marker in title for marker in legal_title_markers):
        return "APPLICABLE"
    operational_markers = (
        "基金经理变更", "分红公告", "收益分配", "净值公告", "暂停申购", "恢复申购",
        "开放日常申购", "销售机构", "交易状态", "停牌公告",
    )
    if any(marker in title for marker in operational_markers):
        strong_change_markers = (
            "本公告同时修改基金合同", "投资范围变更自", "转型生效", "变更注册生效"
        )
        if any(marker in compact for marker in strong_change_markers):
            return "APPLICABLE"
        return "NOT_APPLICABLE"
    substantive_text_markers = ("投资范围", "资产配置", "股票资产占基金", "转型生效")
    if any(marker in compact for marker in substantive_text_markers):
        return "APPLICABLE"
    return "UNDETERMINED"


def extract_investment_sections(text: str) -> list[InvestmentSection]:
    """Normalize PDF layout and return bounded legal investment sections."""

    sections: list[InvestmentSection] = []
    current_heading: str | None = None
    current_page = 1
    start_page = 1
    body: list[str] = []

    def finish() -> None:
        nonlocal body
        if current_heading is None:
            return
        bounded_raw = "\n".join(body).strip()
        normalized = re.sub(r"\s+", "", bounded_raw).strip()
        if normalized:
            sections.append(
                InvestmentSection(current_heading, bounded_raw, normalized, start_page)
            )
        body = []

    pages = str(text or "").split("\f")
    for page_number, page in enumerate(pages, start=1):
        current_page = page_number
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if not line or _PAGE_HEADER.fullmatch(line):
                continue
            heading_match = _STRUCTURED_HEADING.match(line)
            if heading_match:
                finish()
                current_heading = heading_match.group("heading")
                start_page = current_page
                rest = heading_match.group("rest").strip()
                body = [rest] if rest else []
                continue
            if current_heading is not None and _ANY_NUMBERED_HEADING.match(line):
                finish()
                current_heading = None
                body = []
                continue
            if current_heading is not None:
                body.append(line)
    finish()
    return sections


def extract_equity_constraint_candidates(text: str) -> list[EquityConstraint]:
    """Enumerate every distinct explicit state that the existing parser can expose."""

    original = re.sub(r"\s+", "", str(text or ""))
    remaining = original
    post_marker = re.compile(
        r"(?:开放期内|(?:封闭期|封闭运作期)届满|"
        r"(?:转型|转换|转)为上市开放式基金（?LOF）?后)"
    )
    closed_marker = re.compile(r"(?:封闭期|封闭运作期)(?:内|：|,|，)")
    found: list[EquityConstraint] = []
    found_with_position: list[tuple[EquityConstraint, int]] = []
    seen: set[tuple[float, float, str]] = set()
    for _ in range(32):
        try:
            constraint = extract_equity_constraint(remaining)
        except ValueError:
            break
        key = (
            constraint.equity_min_pct,
            constraint.equity_max_pct,
            constraint.evidence,
        )
        if key in seen:
            break
        seen.add(key)
        found.append(constraint)
        evidence = re.sub(r"\s+", "", constraint.evidence)
        original_position = original.find(evidence)
        found_with_position.append((constraint, original_position))
        position = remaining.find(evidence)
        if position < 0:
            break
        remaining = remaining[:position] + ("。" * len(evidence)) + remaining[position + len(evidence):]
    markers = sorted(
        [(match.start(), "closed") for match in closed_marker.finditer(original)]
        + [(match.start(), "post") for match in post_marker.finditer(original)]
    )
    phased: list[tuple[EquityConstraint, str | None]] = []
    for constraint, position in found_with_position:
        nearest = None
        for marker_position, marker_kind in markers:
            if marker_position > position:
                break
            nearest = marker_kind
        phased.append((constraint, nearest))
    post_constraints = [constraint for constraint, phase in phased if phase == "post"]
    return post_constraints or found


def text_extraction_is_usable(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) < 4:
        return False
    replacement_count = compact.count("\ufffd")
    return replacement_count < 5 and replacement_count / len(compact) <= 0.05


def proposed_state_is_pending(text: str, *, report_name: str = "") -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    title = re.sub(r"\s+", "", str(report_name or ""))
    explicit_proposal_title = any(
        marker in title
        for marker in ("草案", "拟修改", "拟变更", "拟修订", "议案", "征求意见稿")
    )
    result_title = any(marker in title for marker in ("表决结果", "决议生效", "生效公告"))
    sentences = [item for item in re.split(r"[。；;]", compact) if item]
    current_resolution = False
    for index, sentence in enumerate(sentences):
        if not re.search(r"(?:本议案|本次(?:修改|变更|修订)|上述事项)", sentence):
            continue
        chain = sentence
        if index + 1 < len(sentences) and sentences[index + 1].startswith("上述事项"):
            chain += "。" + sentences[index + 1]
        conditional = bool(re.search(
            r"(?:若|如|尚未|未经|没有经|拟经|拟获|拟于|拟提交|旧议案|旧合同|另行通知)",
            chain,
        ))
        affirmative = re.search(
            r"(?:(?:已经|已)|(?<!未)经).{0,50}(?:表决|审议)通过", chain
        )
        dated = re.search(r"自\d{4}年\d{1,2}月\d{1,2}日起.{0,40}生效", chain)
        if result_title and affirmative and dated and not conditional:
            current_resolution = True
            break
    if explicit_proposal_title and not current_resolution:
        return True
    current_item_present = bool(
        re.search(r"(?:本议案|上述事项|本次(?:修改|变更|修订))", compact)
    )
    if result_title and current_item_present and not current_resolution:
        return True
    for index, sentence in enumerate(sentences):
        proposal_action = re.search(
            r"(?:本议案|本次)[^。；;]{0,120}(?:拟修改|拟变更|拟修订|大会审议|审议)",
            sentence,
        )
        if not proposal_action:
            continue
        if re.search(r"(?:不涉及变更|不涉及基金合同|不涉及投资范围)", sentence):
            continue
        window = sentence
        if index + 1 < len(sentences) and sentences[index + 1].startswith("上述事项"):
            window += "。" + sentences[index + 1]
        if re.search(
            r"(?:尚需.{0,100}(?:表决|审议).{0,60}通过后方可生效|"
            r"自表决通过之日起生效|表决通过后方可生效)",
            window,
        ):
            return True
    title_calls_meeting = "召开" in title and "持有人大会" in title
    return bool(title_calls_meeting and not current_resolution)


def locate_evidence_page(text: str, evidence: str) -> int | None:
    needle = re.sub(r"\s+", "", str(evidence or ""))
    if not needle:
        return None
    for page_number, page in enumerate(str(text or "").split("\f"), start=1):
        if needle in re.sub(r"\s+", "", page):
            return page_number
    return None


def extract_pdf_text(path: str | PathLike[str]) -> str:
    reader = PdfReader(path)
    return "\n\f\n".join(page.extract_text() or "" for page in reader.pages)


def _clean_evidence(value: str) -> str:
    return re.sub(r"^[（(]?\d+[）)]?\s*", "", value).strip().rstrip("；;。")


def _prefer_current_phase_match(
    normalized: str,
    matches: list[re.Match[str]],
    phase_hint: str | None = None,
) -> re.Match[str] | None:
    """Prefer an explicitly post-conversion range over an expired closed-period range."""
    if not matches:
        return None
    if phase_hint == "closed":
        for match in matches:
            prefix = normalized[max(0, match.start() - 40):match.start()]
            if re.search(r"(?:封闭期|封闭运作期)(?:内|：|,|，)", prefix):
                return match
        return matches[0]
    closed_marker = re.compile(r"(?:封闭期|封闭运作期)(?:内|：|,|，)")
    post_marker = re.compile(
        r"(?:开放期内|(?:封闭期|封闭运作期)届满|"
        r"(?:转型|转换|转)为上市开放式基金（?LOF）?后)"
    )
    for match in matches:
        prefix = normalized[max(0, match.start() - 800):match.start()]
        closed_positions = [item.start() for item in closed_marker.finditer(prefix)]
        post_positions = [item.start() for item in post_marker.finditer(prefix)]
        if post_positions and (
            not closed_positions or post_positions[-1] > closed_positions[-1]
        ):
            return match
    return matches[0]


def extract_equity_constraint(
    text: str, *, phase_hint: str | None = None
) -> EquityConstraint:
    normalized = re.sub(r"\s+", "", text)
    normalized = re.sub(
        r"基金资[^%％；;。]{4,120}第\d+页共\d+页产(?=的?(?:比例|投资))",
        "基金资产",
        normalized,
    )

    bare_allocation_pattern = re.compile(
        r"(?:本基金)?大类资产的?投资比例范围(?:为|是)[：:]?"
        r"(?P<clause>股票(?P<low>\d+(?:\.\d+)?)(?:%|％)?"
        r"[-—~至到－–](?P<high>\d+(?:\.\d+)?)(?:%|％))"
    )
    match = bare_allocation_pattern.search(normalized)
    if match:
        return EquityConstraint(
            float(match.group("low")),
            float(match.group("high")),
            _clean_evidence(match.group("clause")),
        )

    range_pattern = re.compile(
        rf"{_EQUITY_TERM}{_ASSET_TERM}(?:范围)?(?:为|是)?[：:]?"
        r"(?:基金(?:资产净值|资产|净值|总资产)的)?"
        r"(?P<low>\d+(?:\.\d+)?)(?:%|％)?[-—~至到－–]"
        r"(?P<high>\d+(?:\.\d+)?)(?:%|％)"
    )
    match = _prefer_current_phase_match(
        normalized, list(range_pattern.finditer(normalized)), phase_hint
    )
    if match:
        return EquityConstraint(
            float(match.group("low")),
            float(match.group("high")),
            _clean_evidence(match.group(0)),
        )

    minimum_pattern = re.compile(
        rf"{_EQUITY_TERM}{_ASSET_TERM}(?:范围)?(?:为|是)?不低于"
        rf"(?:基金(?:资产净值|资产|净值|总资产)的)?{_NUMBER}%"
    )
    match = minimum_pattern.search(normalized)
    if match:
        return EquityConstraint(
            float(match.group("value")),
            100.0,
            _clean_evidence(match.group(0)),
        )

    maximum_pattern = re.compile(
        rf"{_EQUITY_TERM}{_ASSET_TERM}(?:范围)?(?:为|是)?"
        rf"(?:不高于|不超过)(?:基金资产的)?{_NUMBER}%"
    )
    match = maximum_pattern.search(normalized)
    if match:
        return EquityConstraint(
            0.0,
            float(match.group("value")),
            _clean_evidence(match.group(0)),
        )

    raise ValueError("No explicit equity allocation constraint found")


def classify_mixed_fund(constraint: EquityConstraint) -> str | None:
    if constraint.equity_min_pct >= 60.0:
        return "混合型-偏股"
    if constraint.equity_max_pct <= 30.0:
        return "混合型-偏债"
    return None


def extract_effective_date(text: str) -> pd.Timestamp | None:
    normalized = re.sub(r"\s+", "", text)
    contract_subject = (
        r"(?:《[^》]{0,120}基金合同》|本基金的?基金合同|新基金合同|基金合同)"
    )
    placeholder_date = (
        r"(?=[^日]{1,20}[X_])(?:\d{4}|X{4}|_+)年"
        r"(?:\d{1,2}|X{1,4}|_+)月(?:\d{1,2}|X{1,4}|_+)日"
    )
    placeholder_patterns = (
        rf"{contract_subject}(?:自|于){placeholder_date}(?:起)?[^。；;]{{0,30}}生效",
        rf"(?:自|于){placeholder_date}(?:起)?[^。；;]{{0,30}}{contract_subject}"
        rf"[^。；;]{{0,30}}生效",
    )
    if any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in placeholder_patterns
    ):
        return None

    date = (
        r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
    )
    patterns = (
        re.compile(
            rf"{contract_subject}(?:自|于){date}(?:起)?[^。；;]{{0,40}}?生效"
        ),
        re.compile(
            rf"自{date}起[^。；;]{{0,40}}?转型[^。；;]{{0,20}}?生效"
        ),
        re.compile(
            rf"自{date}起[，,]?(?:原基金合同[^。；;]{{0,10}}失效[，,]?)?"
            rf"{contract_subject}"
            rf"[^。；;]{{0,30}}?生效"
        ),
    )
    candidates: set[pd.Timestamp] = set()
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            try:
                candidates.add(
                    pd.Timestamp(
                        year=int(match.group("year")),
                        month=int(match.group("month")),
                        day=int(match.group("day")),
                    )
                )
            except ValueError:
                continue
    if len(candidates) != 1:
        return None
    return next(iter(candidates))
