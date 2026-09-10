from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pypdf import PdfWriter

import v6.csrc_fund_collector as collector
import v6.g0_checkpoint_gate as checkpoint_gate_module

from v6.csrc_fund_collector import (
    CsrcDisclosureClient,
    CheckpointStore,
    build_classification_timeline,
    build_timeline_from_downloads,
    collect_fund_disclosures,
    collect_report_metadata,
    validate_pdf,
)


class FakeMetadataClient:
    def __init__(self, pages: dict[int, dict], fail_once_at: int | None = None):
        self.pages = pages
        self.fail_once_at = fail_once_at
        self.calls: list[int] = []

    def fetch_page(
        self, fund_code: str, report_type: str, start: int, page_size: int
    ) -> dict:
        self.calls.append(start)
        if self.fail_once_at == start:
            self.fail_once_at = None
            raise ConnectionError("simulated interruption")
        return self.pages[start]


def _row(upload_id: str, date: str, title: str = "招募说明书") -> dict:
    return {
        "fundCode": "000001",
        "fundShortName": "测试基金",
        "reportCode": "FA010030",
        "reportName": title,
        "reportSendDate": date,
        "uploadInfoId": upload_id,
        "uploadInfoDetailId": None,
    }


def test_metadata_collection_resumes_from_last_completed_page(tmp_path: Path) -> None:
    pages = {
        0: {"iTotalRecords": 3, "aaData": [_row("a", "2020-01-01"), _row("b", "2021-01-01")]},
        2: {"iTotalRecords": 3, "aaData": [_row("c", "2022-01-01")]},
    }
    checkpoint = CheckpointStore(tmp_path / "checkpoint.json")
    output = tmp_path / "metadata.jsonl"
    interrupted = FakeMetadataClient(pages, fail_once_at=2)

    with pytest.raises(ConnectionError):
        collect_report_metadata(
            interrupted, "000001", "FA010030", output, checkpoint, page_size=2
        )

    assert interrupted.calls == [0, 2]
    assert checkpoint.next_start("000001", "FA010030") == 2

    resumed = FakeMetadataClient(pages)
    result = collect_report_metadata(
        resumed, "000001", "FA010030", output, checkpoint, page_size=2
    )

    assert resumed.calls == [2]
    assert [row["uploadInfoId"] for row in result] == ["a", "b", "c"]
    assert checkpoint.is_complete("000001", "FA010030")


def test_resume_is_idempotent_when_last_page_is_replayed(tmp_path: Path) -> None:
    pages = {0: {"iTotalRecords": 1, "aaData": [_row("a", "2020-01-01")]}}
    checkpoint = CheckpointStore(tmp_path / "checkpoint.json")
    output = tmp_path / "metadata.jsonl"

    collect_report_metadata(
        FakeMetadataClient(pages), "000001", "FA010030", output, checkpoint
    )
    collect_report_metadata(
        FakeMetadataClient(pages), "000001", "FA010030", output, checkpoint
    )

    assert output.read_text(encoding="utf-8").count("\n") == 1


def test_checkpoint_retries_transient_windows_replace_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = CheckpointStore(checkpoint_path)
    original_replace = Path.replace
    attempts = 0

    def fail_once(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "simulated transient Windows lock", str(target))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once)

    checkpoint.save_progress("000001", "FA010030", 2, complete=False)

    reloaded = CheckpointStore(checkpoint_path)
    assert attempts == 2
    assert reloaded.next_start("000001", "FA010030") == 2


def test_pdf_validation_rejects_html_and_accepts_readable_pdf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"<html>rate limited</html>")
    with pytest.raises(ValueError, match="PDF signature"):
        validate_pdf(bad)

    good = tmp_path / "good.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with good.open("wb") as stream:
        writer.write(stream)

    result = validate_pdf(good)
    assert result.page_count == 1
    assert result.size_bytes == good.stat().st_size


def test_timeline_preserves_known_at_and_explicit_effective_date() -> None:
    docs = pd.DataFrame(
        [
            {
                "share_code": "160415",
                "known_at": "2019-01-10",
                "effective_date": "2019-01-11",
                "equity_min_pct": 60.0,
                "equity_max_pct": 100.0,
                "evidence": "股票投资占基金资产的比例不低于60%",
                "source_document": "official-a.pdf",
            },
            {
                "share_code": "160415",
                "known_at": "2021-12-20",
                "effective_date": None,
                "equity_min_pct": 0.0,
                "equity_max_pct": 95.0,
                "evidence": "股票资产占基金资产的比例为0%-95%",
                "source_document": "official-b.pdf",
            },
        ]
    )

    timeline = build_classification_timeline(docs)

    assert timeline.loc[0, "known_at"] == pd.Timestamp("2019-01-10")
    assert timeline.loc[0, "effective_from"] == pd.Timestamp("2019-01-11")
    assert timeline.loc[0, "fund_type"] == "混合型-偏股"
    assert timeline.loc[1, "effective_from"] == pd.Timestamp("2021-12-20")
    assert pd.isna(timeline.loc[1, "fund_type"])
    assert timeline.loc[0, "effective_to"] == pd.Timestamp("2021-12-20")


def test_historical_effective_date_quoted_by_update_does_not_backdate_known_state() -> None:
    docs = pd.DataFrame(
        [
            {
                "share_code": "160415",
                "known_at": "2021-12-20",
                "effective_date": "2019-01-11",
                "equity_min_pct": 60,
                "equity_max_pct": 100,
                "evidence": "股票投资占基金资产的比例不低于60%",
                "source_document": "2021-update.pdf",
            }
        ]
    )

    timeline = build_classification_timeline(docs)

    assert timeline.loc[0, "effective_from"] == pd.Timestamp("2021-12-20")


def test_duplicate_same_day_state_is_collapsed_but_conflict_is_rejected() -> None:
    base = {
        "share_code": "000001",
        "known_at": "2020-01-01",
        "effective_date": None,
        "equity_min_pct": 60,
        "equity_max_pct": 100,
        "evidence": "same",
    }
    duplicates = pd.DataFrame(
        [
            {**base, "source_document": "a.pdf"},
            {**base, "source_document": "b.pdf"},
        ]
    )
    assert len(build_classification_timeline(duplicates)) == 1

    conflict = duplicates.copy()
    conflict.loc[1, ["equity_min_pct", "equity_max_pct"]] = [0, 30]
    with pytest.raises(ValueError, match="Conflicting classifications"):
        build_classification_timeline(conflict)


def test_same_day_compatible_constraints_use_the_tighter_intersection() -> None:
    base = {
        "share_code": "163801",
        "known_at": "2020-11-18",
        "effective_date": None,
    }
    documents = pd.DataFrame(
        [
            {
                **base,
                "equity_min_pct": 40,
                "equity_max_pct": 95,
                "evidence": "股票资产及存托凭证投资比例为40%-95%",
                "source_document": "prospectus.pdf",
            },
            {
                **base,
                "equity_min_pct": 40,
                "equity_max_pct": 100,
                "evidence": "股票资产及存托凭证比例不低于基金资产净值40%",
                "source_document": "summary.pdf",
            },
        ]
    )

    timeline = build_classification_timeline(documents)

    assert len(timeline) == 1
    assert timeline.loc[0, "equity_min_pct"] == 40
    assert timeline.loc[0, "equity_max_pct"] == 95
    assert timeline.loc[0, "source_document"] == "prospectus.pdf"


def test_timeline_rejects_effective_date_before_inception() -> None:
    docs = pd.DataFrame(
        [
            {
                "share_code": "000001",
                "known_at": "2010-01-01",
                "effective_date": "2009-12-31",
                "inception_date": "2010-01-01",
                "equity_min_pct": 60,
                "equity_max_pct": 100,
                "evidence": "x",
                "source_document": "x.pdf",
            }
        ]
    )

    with pytest.raises(ValueError, match="before inception"):
        build_classification_timeline(docs)


class FakeResponse:
    def __init__(self, *, payload: dict | None = None, body: bytes = b""):
        self._payload = payload
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload

    def iter_content(self, chunk_size: int):
        yield self.body


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_http_client_sends_datatables_pagination_contract() -> None:
    session = FakeSession([FakeResponse(payload={"iTotalRecords": 0, "aaData": []})])
    client = CsrcDisclosureClient(session=session, base_url="http://official.test")

    client.fetch_page("000001", "FA010030", start=40, page_size=20)

    url, kwargs = session.calls[0]
    assert url == "http://official.test/fund/disclose/advanced_search_report.do"
    ao_data = __import__("json").loads(kwargs["params"]["aoData"])
    values = {item["name"]: item["value"] for item in ao_data}
    assert values["iDisplayStart"] == 40
    assert values["iDisplayLength"] == 20
    assert values["fundCode"] == "000001"
    assert values["reportType"] == "FA010030"


def _pdf_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)
    return path.read_bytes()


def test_pdf_download_is_atomic_and_skips_existing_valid_file(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(body=_pdf_bytes(tmp_path))])
    client = CsrcDisclosureClient(session=session, base_url="http://official.test")
    destination = tmp_path / "download.pdf"

    first = client.download_pdf("123", destination)
    second = client.download_pdf("123", destination)

    assert first.page_count == second.page_count == 1
    assert len(session.calls) == 1
    assert not destination.with_suffix(".pdf.part").exists()


def test_invalid_download_does_not_leave_destination_or_partial(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(body=b"<html>not a pdf</html>")])
    client = CsrcDisclosureClient(session=session, base_url="http://official.test")
    destination = tmp_path / "download.pdf"

    with pytest.raises(ValueError, match="PDF signature"):
        client.download_pdf("123", destination)

    assert not destination.exists()
    assert not destination.with_suffix(".pdf.part").exists()


def test_build_timeline_from_downloads_keeps_parse_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        _row("good", "2019-01-10"),
        _row("unknown", "2020-01-10"),
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_good.pdf").write_bytes(b"%PDF-placeholder")
    (pdf_root / "000001_unknown.pdf").write_bytes(b"%PDF-placeholder")

    texts = {
        "000001_good.pdf": "股票投资占基金资产的比例不低于60%",
        "000001_unknown.pdf": "没有明确的股票仓位约束",
    }
    monkeypatch.setattr(
        "v6.csrc_fund_collector.extract_pdf_text",
        lambda path: texts[Path(path).name],
    )

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert timeline.loc[0, "fund_type"] == "混合型-偏股"
    assert timeline.loc[0, "known_at"] == pd.Timestamp("2019-01-10")
    assert failures.to_dict("records") == [
        {
            "share_code": "000001",
            "upload_info_id": "unknown",
            "report_code": "FA010030",
            "report_name": "招募说明书",
            "reason": "No explicit equity allocation constraint found",
        }
    ]


def test_official_index_title_preserves_pre_transformation_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        {
            **_row("old", "2011-07-27", "华安深证300指数基金LOF基金合同"),
            "reportCode": "FA010010",
        },
        {
            **_row("new", "2019-01-10", "华安量化多因子混合型证券投资基金招募说明书"),
            "reportCode": "FA010010",
        },
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    for upload_id in ("old", "new"):
        (pdf_root / f"000001_{upload_id}.pdf").write_bytes(b"%PDF-placeholder")
    texts = {
        "000001_old.pdf": "标的指数成份股及其备选成份股的投资比例不低于基金资产的90%",
        "000001_new.pdf": (
            "自2019年1月11日起，新基金合同生效。"
            "本基金股票投资占基金资产的比例不低于60%"
        ),
    }
    monkeypatch.setattr(
        "v6.csrc_fund_collector.extract_pdf_text",
        lambda path: texts[Path(path).name],
    )

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert failures.empty
    assert timeline["fund_type"].tolist() == ["指数型-股票", "混合型-偏股"]
    assert timeline["effective_from"].tolist() == [
        pd.Timestamp("2011-07-27"),
        pd.Timestamp("2019-01-11"),
    ]


def test_document_parse_cache_skips_completed_pdf_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [_row("cached", "2019-01-10")]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_cached.pdf").write_bytes(b"%PDF-placeholder")
    calls = 0

    def extract(_path: Path) -> str:
        nonlocal calls
        calls += 1
        return "股票投资占基金资产的比例不低于60%"

    monkeypatch.setattr("v6.csrc_fund_collector.extract_pdf_text", extract)
    cache = tmp_path / "parsed_documents.jsonl"

    first, _ = build_timeline_from_downloads(
        metadata, pdf_root, parse_cache_path=cache
    )
    second, _ = build_timeline_from_downloads(
        metadata, pdf_root, parse_cache_path=cache
    )

    assert calls == 1
    pd.testing.assert_frame_equal(first, second)
    assert cache.read_text(encoding="utf-8").count("\n") == 1


def test_document_parse_cache_reuses_success_from_older_parser_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [_row("cached", "2019-01-10")]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_cached.pdf").write_bytes(b"%PDF-placeholder")
    cache = tmp_path / "parsed_documents.jsonl"
    document = {
        "share_code": "000001",
        "known_at": "2019-01-10",
        "effective_date": None,
        "equity_min_pct": 60.0,
        "equity_max_pct": 95.0,
        "evidence": "股票占基金资产的比例为60%-95%",
        "source_document": "000001_cached.pdf",
        "report_code": "FA010030",
        "report_name": "招募说明书",
        "upload_info_id": "cached",
    }
    cache.write_text(
        json.dumps(
            {
                "document": document,
                "parser_version": 15,
                "source_document": "000001_cached.pdf",
                "status": "success",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def extract(_path: Path) -> str:
        nonlocal calls
        calls += 1
        return ""

    monkeypatch.setattr("v6.csrc_fund_collector.extract_pdf_text", extract)

    timeline, failures = build_timeline_from_downloads(
        metadata, pdf_root, parse_cache_path=cache
    )

    assert calls == 0
    assert failures.empty
    assert timeline["equity_min_pct"].tolist() == [60.0]


def test_parse_only_rebuild_skips_metadata_and_pdf_download_stages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collector"
    pdf_root = root / "pdf"
    pdf_root.mkdir(parents=True)
    metadata = _row("cached", "2019-01-10")
    (root / "metadata.jsonl").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (pdf_root / "000001_cached.pdf").write_bytes(b"%PDF-placeholder")
    document = {
        "share_code": "000001",
        "known_at": "2019-01-10",
        "effective_date": None,
        "equity_min_pct": 60.0,
        "equity_max_pct": 95.0,
        "evidence": "股票占基金资产的比例为60%-95%",
        "source_document": "000001_cached.pdf",
        "report_code": "FA010030",
        "report_name": "招募说明书",
        "upload_info_id": "cached",
    }
    (root / "parsed_documents.jsonl").write_text(
        json.dumps(
            {
                "document": document,
                "parser_version": 16,
                "source_document": "000001_cached.pdf",
                "status": "success",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class NoNetworkClient:
        def fetch_page(self, *_args, **_kwargs):
            raise AssertionError("parse-only must not fetch metadata")

        def download_pdf(self, *_args, **_kwargs):
            raise AssertionError("parse-only must not validate or download PDFs")

    artifacts = collect_fund_disclosures(
        ["000001"],
        ["FA010030"],
        root,
        client=NoNetworkClient(),
        parse_only=True,
    )

    timeline = pd.read_csv(artifacts["classification_timeline"], dtype=str)
    assert timeline["share_code"].tolist() == ["000001"]


def test_reparses_version_fifteen_fa_after_net_asset_denominator_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [_row("cached", "2019-01-10")]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_cached.pdf").write_bytes(b"%PDF-placeholder")
    cache = tmp_path / "parsed_documents.jsonl"
    cache.write_text(
        json.dumps(
            {
                "failure": {
                    "reason": "No explicit equity allocation constraint found",
                    "report_code": "FA010030",
                    "report_name": "招募说明书",
                    "share_code": "000001",
                    "upload_info_id": "cached",
                },
                "parser_version": 15,
                "source_document": "000001_cached.pdf",
                "status": "failure",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def extract(_path: Path) -> str:
        nonlocal calls
        calls += 1
        return "本基金股票投资占基金资产的0-30%"

    monkeypatch.setattr("v6.csrc_fund_collector.extract_pdf_text", extract)

    timeline, failures = build_timeline_from_downloads(
        metadata, pdf_root, parse_cache_path=cache
    )

    assert calls == 1
    assert failures.empty
    assert timeline["fund_type"].tolist() == ["混合型-偏债"]


def test_timeline_quarantines_one_conflicting_fund_without_aborting_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        _row("conflict_a", "2019-12-14"),
        _row("conflict_b", "2019-12-14", title="基金合同更新"),
        {**_row("valid", "2020-01-02"), "fundCode": "000002"},
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    for code, upload_id in (("000001", "conflict_a"), ("000001", "conflict_b"), ("000002", "valid")):
        (pdf_root / f"{code}_{upload_id}.pdf").write_bytes(b"%PDF-placeholder")
    texts = {
        "000001_conflict_a.pdf": "股票资产占基金资产的比例范围为60%-95%",
        "000001_conflict_b.pdf": "股票资产占基金资产的比例范围为0%-30%",
        "000002_valid.pdf": "股票等权益类资产占基金资产的比例不超过30%",
    }
    monkeypatch.setattr(
        "v6.csrc_fund_collector.extract_pdf_text",
        lambda path: texts[Path(path).name],
    )

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert timeline["share_code"].tolist() == ["000002"]
    assert failures.loc[failures["share_code"].eq("000001"), "reason"].tolist() == [
        "Conflicting classifications at ('000001', Timestamp('2019-12-14 00:00:00'))"
    ]


def test_transition_notice_naming_old_index_fund_does_not_override_new_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        {
            **_row(
                "transition",
                "2018-11-26",
                "关于华安深证300指数基金转型有关事项的公告",
            ),
            "reportCode": "FC900090",
        }
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_transition.pdf").write_bytes(b"%PDF-placeholder")
    monkeypatch.setattr(
        "v6.csrc_fund_collector.extract_pdf_text",
        lambda _path: (
            "自2019年1月11日起，新基金合同生效。"
            "本基金股票投资占基金资产的比例不低于60%"
        ),
    )

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert failures.empty
    assert timeline.loc[0, "fund_type"] == "混合型-偏股"


def test_document_title_selects_closed_then_post_conversion_constraints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        _row("initial", "2016-08-11", title="鼎越定期开放混合基金招募说明书"),
        _row("updated", "2019-12-14", title="鼎越混合基金（LOF）更新招募说明书"),
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    for upload_id in ("initial", "updated"):
        (pdf_root / f"000001_{upload_id}.pdf").write_bytes(b"%PDF-placeholder")
    text = (
        "封闭期内股票资产占基金资产的比例范围为0%-100%；"
        "转为上市开放式基金（LOF）后股票资产占基金资产的比例范围为0%-95%。"
    )
    monkeypatch.setattr("v6.csrc_fund_collector.extract_pdf_text", lambda _path: text)

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert failures.empty
    assert timeline[["equity_min_pct", "equity_max_pct"]].values.tolist() == [
        [0.0, 100.0],
        [0.0, 95.0],
    ]


def test_event_notice_without_explicit_effective_date_cannot_start_type_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = [
        {
            **_row("notice", "2018-11-26", "关于基金转型有关事项的公告"),
            "reportCode": "FC900090",
        }
    ]
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_notice.pdf").write_bytes(b"%PDF-placeholder")
    monkeypatch.setattr(
        "v6.csrc_fund_collector.extract_pdf_text",
        lambda _path: "本基金股票投资占基金资产的比例不低于60%",
    )

    timeline, failures = build_timeline_from_downloads(metadata, pdf_root)

    assert timeline.empty
    assert failures.loc[0, "reason"] == "Event notice has no explicit effective date"


def _seed_parse_only_collector(root: Path) -> None:
    pdf_root = root / "pdf"
    pdf_root.mkdir(parents=True)
    metadata = _row("cached", "2019-01-10")
    (root / "metadata.jsonl").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (pdf_root / "000001_cached.pdf").write_bytes(b"%PDF-placeholder")
    document = {
        "share_code": "000001",
        "known_at": "2019-01-10",
        "effective_date": None,
        "equity_min_pct": 60.0,
        "equity_max_pct": 95.0,
        "evidence": "股票占基金资产的比例为60%-95%",
        "source_document": "000001_cached.pdf",
        "report_code": "FA010030",
        "report_name": "招募说明书",
        "upload_info_id": "cached",
    }
    (root / "parsed_documents.jsonl").write_text(
        json.dumps(
            {
                "document": document,
                "parser_version": 16,
                "source_document": "000001_cached.pdf",
                "status": "success",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sample_and_aliases(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "family_sample.csv"
    aliases = tmp_path / "aliases.csv"
    sample.write_text(
        "family_key,status,inception_era\nfamily-1,存续,2013_2019\n",
        encoding="utf-8",
    )
    aliases.write_text("query_code,family_key\n000001,family-1\n", encoding="utf-8")
    return sample, aliases


def _fake_gate_artifacts(_result: object, out_dir: Path, **_kwargs: object) -> dict[str, Path]:
    filenames = {
        "required_classification_checkpoints": "required_classification_checkpoints.csv",
        "document_dispositions": "document_dispositions.csv",
        "classification_clause_clusters": "classification_clause_clusters.csv",
        "classification_state_timeline": "classification_state_timeline.csv",
        "g0_amendment_a1_gate_metrics": "g0_amendment_a1_gate_metrics.csv",
        "g0_amendment_a1_gate_review": "g0_amendment_a1_gate_review.md",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / filename for name, filename in filenames.items()}
    for path in paths.values():
        if not path.exists():
            path.write_text("artifact\n", encoding="utf-8")
    return paths


def test_parse_only_default_skips_a1_gate_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "collector"
    _seed_parse_only_collector(root)
    monkeypatch.setattr(
        checkpoint_gate_module,
        "run_checkpoint_gate",
        lambda *_args, **_kwargs: pytest.fail("default collector must not run A1"),
    )

    artifacts = collect_fund_disclosures(
        ["000001"], ["FA010030"], root, parse_only=True
    )

    assert not (root / "a1").exists()
    assert not any(name.startswith("a1_") for name in artifacts)


def test_opt_in_gate_adds_six_a1_artifacts_without_moving_legacy_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "collector"
    _seed_parse_only_collector(root)
    sample, aliases = _sample_and_aliases(tmp_path)
    seen: dict[str, object] = {}

    def run(inputs: object, *, frozen_checkpoints: Path | None = None) -> object:
        seen["inputs"] = inputs
        seen["frozen"] = frozen_checkpoints
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(checkpoint_gate_module, "run_checkpoint_gate", run)
    monkeypatch.setattr(checkpoint_gate_module, "write_gate_artifacts", _fake_gate_artifacts)

    artifacts = collect_fund_disclosures(
        ["000001"],
        ["FA010030"],
        root,
        parse_only=True,
        checkpoint_gate=True,
        family_sample_path=sample,
        aliases_path=aliases,
    )

    assert artifacts["classification_timeline"] == root / "classification_timeline.csv"
    assert artifacts["a1_classification_state_timeline"] == (
        root / "a1" / "classification_state_timeline.csv"
    )
    assert artifacts["a1_gate_metrics"] == (
        root / "a1" / "g0_amendment_a1_gate_metrics.csv"
    )
    assert artifacts["a1_required_classification_checkpoints"].is_file()
    assert {name for name in artifacts if name.startswith("a1_")} == {
        "a1_required_classification_checkpoints",
        "a1_document_dispositions",
        "a1_classification_clause_clusters",
        "a1_classification_state_timeline",
        "a1_gate_metrics",
        "a1_gate_review",
    }
    assert seen["inputs"].metadata_jsonl == root / "metadata.jsonl"
    assert seen["inputs"].parsed_documents_jsonl == root / "parsed_documents.jsonl"
    assert seen["inputs"].pdf_root == root / "pdf"
    assert seen["inputs"].legacy_parse_failures_csv == root / "classification_parse_failures.csv"
    assert seen["frozen"] is None


@pytest.mark.parametrize("missing", ["sample", "aliases"])
def test_opt_in_gate_rejects_missing_prerequisite_before_a1_output(
    tmp_path: Path, missing: str
) -> None:
    root = tmp_path / "collector"
    _seed_parse_only_collector(root)
    sample, aliases = _sample_and_aliases(tmp_path)
    kwargs = {
        "checkpoint_gate": True,
        "family_sample_path": None if missing == "sample" else sample,
        "aliases_path": None if missing == "aliases" else aliases,
    }

    with pytest.raises(ValueError, match=missing):
        collect_fund_disclosures(["000001"], ["FA010030"], root, parse_only=True, **kwargs)

    assert not (root / "a1").exists()


def test_opt_in_gate_preserves_existing_frozen_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "collector"
    _seed_parse_only_collector(root)
    sample, aliases = _sample_and_aliases(tmp_path)
    frozen = root / "a1" / "required_classification_checkpoints.csv"
    frozen.parent.mkdir()
    frozen.write_bytes(b"frozen denominator\n")
    before = (frozen.read_bytes(), frozen.stat().st_mtime_ns)
    seen: dict[str, object] = {}

    def run(_inputs: object, *, frozen_checkpoints: Path | None = None) -> object:
        seen["frozen"] = frozen_checkpoints
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(checkpoint_gate_module, "run_checkpoint_gate", run)
    monkeypatch.setattr(checkpoint_gate_module, "write_gate_artifacts", _fake_gate_artifacts)

    collect_fund_disclosures(
        ["000001"], ["FA010030"], root, parse_only=True, checkpoint_gate=True,
        family_sample_path=sample, aliases_path=aliases,
    )

    assert seen["frozen"] == frozen
    assert (frozen.read_bytes(), frozen.stat().st_mtime_ns) == before


def test_cli_forwards_checkpoint_gate_options_and_keeps_mode_exclusion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codes = tmp_path / "codes.txt"
    codes.write_text("000001\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(collector, "load_fund_codes", lambda _path: ["000001"])
    monkeypatch.setattr(
        collector,
        "collect_fund_disclosures",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or {},
    )

    assert collector.main([
        "--codes-file", str(codes), "--out", str(tmp_path / "out"), "--parse-only",
        "--checkpoint-gate", "--family-sample", "sample.csv", "--aliases", "aliases.csv",
    ]) == 0
    assert captured["kwargs"]["checkpoint_gate"] is True
    assert captured["kwargs"]["family_sample_path"] == "sample.csv"
    assert captured["kwargs"]["aliases_path"] == "aliases.csv"
    with pytest.raises(SystemExit):
        collector._parse_args([
            "--codes-file", str(codes), "--out", str(tmp_path / "out"),
            "--metadata-only", "--parse-only",
        ])


def test_a1_gate_failure_returns_written_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "collector"
    _seed_parse_only_collector(root)
    sample, aliases = _sample_and_aliases(tmp_path)
    monkeypatch.setattr(
        checkpoint_gate_module, "run_checkpoint_gate", lambda *_args, **_kwargs: SimpleNamespace(passed=False)
    )
    monkeypatch.setattr(checkpoint_gate_module, "write_gate_artifacts", _fake_gate_artifacts)

    artifacts = collect_fund_disclosures(
        ["000001"], ["FA010030"], root, parse_only=True, checkpoint_gate=True,
        family_sample_path=sample, aliases_path=aliases,
    )

    assert artifacts["a1_gate_metrics"].is_file()
