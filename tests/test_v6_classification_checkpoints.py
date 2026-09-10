import os
from pathlib import Path

import pandas as pd
import pytest
import v6.classification_checkpoints as classification_checkpoints

from v6.classification_checkpoints import (
    GENERATION_RULE_VERSION,
    NORMALIZATION_VERSION,
    Disposition,
    adjudicate_checkpoints,
    build_clause_clusters,
    build_checkpoint_timeline,
    fingerprint_clause,
    freeze_checkpoint_manifest,
    generate_required_checkpoints,
    normalize_investment_clause,
    validate_checkpoint_invariants,
)


def _timeline_dispositions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checkpoint_id": "a-1",
                "family_key": "family-a",
                "known_at": "2020-01-10",
                "effective_date": "2020-01-09",
                "disposition": "PARSED_STATE",
                "fund_type": "混合型-偏股",
                "equity_min_pct": 60.0,
                "equity_max_pct": 95.0,
                "failure_reason": "",
            },
            {
                "checkpoint_id": "a-2",
                "family_key": "family-a",
                "known_at": "2020-02-10",
                "effective_date": "2020-02-10",
                "disposition": "VERIFIED_NOOP",
                "predecessor_checkpoint_id": "a-1",
                "fund_type": "",
                "equity_min_pct": None,
                "equity_max_pct": None,
                "failure_reason": "",
            },
        ]
    )


def _manifest_for(dispositions: pd.DataFrame) -> pd.DataFrame:
    return dispositions.assign(
        is_critical=True,
        share_code="000049",
        upload_info_id=dispositions["checkpoint_id"],
    )[["checkpoint_id", "family_key", "share_code", "upload_info_id", "known_at", "effective_date", "is_critical"]]


def _adjudication_checkpoints(count: int = 1) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "checkpoint_id": f"checkpoint-{index + 1}",
                "family_key": "family-a",
                "share_code": "000049",
                "upload_info_id": f"notice-{index + 1}",
                "known_at": f"2020-0{index + 1}-02",
                "effective_date": f"2020-0{index + 1}-01",
                "trigger_type": "MATERIAL_SECTION_CHANGE",
                "source_document": f"notice-{index + 1}.pdf",
            }
        )
    return pd.DataFrame(rows)


def _adjudication_sections(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _parsed_state(upload_info_id: str, *, low: float = 60.0, high: float = 95.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "upload_info_id": upload_info_id,
                "status": "success",
                "parser_version": 16,
                "document": {
                    "evidence": "股票资产占基金资产的比例为60%-95%",
                    "source_document": f"{upload_info_id}.pdf",
                    "equity_min_pct": low,
                    "equity_max_pct": high,
                    "fund_type": "混合型-偏股",
                },
            }
        ]
    )


def _lifecycle_checkpoint(trigger_type: str = "INCEPTION", known_at: str = "2020-01-02") -> dict[str, object]:
    return {
        "checkpoint_id": "lifecycle-1",
        "family_key": "family-a",
        "share_code": "000049",
        "upload_info_id": f"{trigger_type}:000049:{known_at or 'UNKNOWN'}",
        "known_at": known_at,
        "effective_date": known_at,
        "trigger_type": trigger_type,
        "source_document": "fund_master:000049",
    }


def _lifecycle_section(
    upload_info_id: str,
    *,
    family_key: str = "family-a",
    known_at: str = "2020-01-01",
    clause: str = "股票资产占基金资产的比例为60%-95%。",
    source_document: str | None = None,
) -> dict[str, object]:
    source = source_document or f"{upload_info_id}.pdf"
    return {
        "upload_info_id": upload_info_id,
        "family_key": family_key,
        "known_at": known_at,
        "source_document": source,
        "evidence_location": f"{source}:p4",
        "section_text": clause,
        "extraction_status": "success",
    }


def _lifecycle_evidence(upload_info_id: str, **overrides: object) -> dict[str, object]:
    section = _lifecycle_section(upload_info_id)
    section.update(
        {
            "status": "success",
            "parser_version": 16,
            "evidence": "股票资产占基金资产的比例为60%-95%。",
            "equity_min_pct": 60.0,
            "equity_max_pct": 95.0,
            "fund_type": "混合型-偏股",
            "parsed_upload_info_id": section["upload_info_id"],
            "parsed_known_at": section["known_at"],
            "parsed_source_document": section["source_document"],
            "parsed_evidence_location": section["evidence_location"],
            "parsed_evidence_raw": section["section_text"],
            "parsed_normalized_clause": normalize_investment_clause(str(section["section_text"])),
        }
    )
    section.update(overrides)
    if "parsed_normalized_clause" not in overrides:
        section["parsed_normalized_clause"] = normalize_investment_clause(
            str(section["parsed_evidence_raw"])
        )
    return section


def test_lifecycle_resolver_rejects_cross_family_and_late_candidates() -> None:
    checkpoint = _lifecycle_checkpoint()
    candidates = [
        _lifecycle_evidence("cross", family_key="family-b"),
        _lifecycle_evidence("late", known_at="2020-01-03", parsed_known_at="2020-01-03"),
    ]

    selected, failure = classification_checkpoints.resolve_lifecycle_evidence(checkpoint, candidates)

    assert selected == []
    assert failure == "SECTION_EXTRACTION_FAILED"


def test_lifecycle_resolver_rejects_candidates_missing_required_provenance() -> None:
    checkpoint = _lifecycle_checkpoint()
    candidate = _lifecycle_evidence("candidate")
    for missing in ("known_at", "source_document", "evidence_location", "section_text"):
        incomplete = dict(candidate)
        incomplete[missing] = ""
        selected, failure = classification_checkpoints.resolve_lifecycle_evidence(checkpoint, [incomplete])
        assert selected == []
        assert failure == "SECTION_EXTRACTION_FAILED"


def test_lifecycle_resolver_reports_same_priority_clause_ambiguity() -> None:
    checkpoint = _lifecycle_checkpoint()
    candidates = [
        _lifecycle_evidence("first", known_at="2020-01-01"),
        _lifecycle_evidence(
            "second",
            known_at="2020-01-01",
            clause="股票资产占基金资产的比例为0%-30%。",
            section_text="股票资产占基金资产的比例为0%-30%。",
            evidence="股票资产占基金资产的比例为0%-30%。",
            parsed_evidence_raw="股票资产占基金资产的比例为0%-30%。",
            equity_min_pct=0.0,
            equity_max_pct=30.0,
        ),
    ]

    selected, failure = classification_checkpoints.resolve_lifecycle_evidence(checkpoint, candidates)

    assert selected == []
    assert failure == "SOURCE_CONFLICT"


def test_lifecycle_resolver_prefers_latest_causal_date_and_sorts_ties() -> None:
    checkpoint = _lifecycle_checkpoint()
    candidates = [
        _lifecycle_evidence("older", known_at="2019-12-01", parsed_known_at="2019-12-01"),
        _lifecycle_evidence("z", source_document="z.pdf", parsed_source_document="z.pdf"),
        _lifecycle_evidence("a", source_document="a.pdf", parsed_source_document="a.pdf"),
    ]

    first, first_failure = classification_checkpoints.resolve_lifecycle_evidence(checkpoint, candidates)
    second, second_failure = classification_checkpoints.resolve_lifecycle_evidence(checkpoint, list(reversed(candidates)))

    assert first_failure == second_failure == ""
    assert [row["source_document"] for row in first] == ["a.pdf", "z.pdf"]
    assert first == second


def test_ordinary_checkpoint_never_falls_back_to_family_candidate() -> None:
    checkpoint = _adjudication_checkpoints().iloc[0].to_dict()
    candidate = _lifecycle_section("other-upload")
    sections = _adjudication_sections(candidate)

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), sections, _parsed_state("other-upload")
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SECTION_EXTRACTION_FAILED"


def test_terminal_without_known_at_does_not_guess_a_candidate() -> None:
    checkpoint = _lifecycle_checkpoint("TERMINAL_STATE", "")
    sections = _adjudication_sections(_lifecycle_section("terminal-doc"))

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), sections, _parsed_state("terminal-doc")
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SECTION_EXTRACTION_FAILED"


def test_dated_terminal_uses_only_latest_causal_same_family_candidate() -> None:
    checkpoint = _lifecycle_checkpoint("TERMINAL_STATE", "2020-01-02")
    lifecycle_evidence = pd.DataFrame(
        [
            _lifecycle_evidence("older", known_at="2019-12-01", parsed_known_at="2019-12-01"),
            _lifecycle_evidence("selected", known_at="2020-01-02", parsed_known_at="2020-01-02"),
            _lifecycle_evidence("later", known_at="2020-01-03", parsed_known_at="2020-01-03"),
        ]
    )

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=lifecycle_evidence,
    )

    assert result.loc[0, "disposition"] == Disposition.PARSED_STATE.value
    assert result.loc[0, "evidence_location"] == "selected.pdf:p4"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parsed_upload_info_id", "other-upload"),
        ("parsed_known_at", "2019-12-31"),
        ("parsed_source_document", "other.pdf"),
        ("parsed_evidence_location", "other.pdf:p4"),
        ("parsed_evidence_raw", "股票资产占基金资产的比例为0%-30%。"),
        ("parsed_normalized_clause", "股票资产占基金资产的比例为0%-30%。"),
    ],
)
def test_lifecycle_state_provenance_mismatch_fails_closed(field: str, value: str) -> None:
    checkpoint = _lifecycle_checkpoint()
    evidence = pd.DataFrame([_lifecycle_evidence("candidate", **{field: value})])

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=evidence,
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value


def test_lifecycle_success_reports_selected_pdf_provenance_but_keeps_checkpoint_identity() -> None:
    checkpoint = _lifecycle_checkpoint()
    evidence = pd.DataFrame([_lifecycle_evidence("official-7")])

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=evidence,
    )

    row = result.iloc[0]
    assert row.checkpoint_id == "lifecycle-1"
    assert row.trigger_type == "INCEPTION"
    assert row.upload_info_id == "INCEPTION:000049:2020-01-02"
    assert row.source_document == "fund_master:000049"
    assert row.known_at == "2020-01-02"
    assert row.effective_date == "2020-01-02"
    assert row.family_key == "family-a"
    assert row.evidence_upload_info_id == "official-7"
    assert row.evidence_source_document == "official-7.pdf"
    assert row.evidence_known_at == "2020-01-01"
    assert row.evidence_effective_date == ""
    assert row.evidence_location == "official-7.pdf:p4"


def test_same_date_same_clause_prefers_source_with_bound_state_deterministically() -> None:
    checkpoint = _lifecycle_checkpoint()
    invalid = _lifecycle_evidence("a-invalid", status="failure", source_document="a.pdf", parsed_source_document="a.pdf")
    valid = _lifecycle_evidence("z-valid", source_document="z.pdf", parsed_source_document="z.pdf")

    first = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=pd.DataFrame([invalid, valid]),
    )
    second = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=pd.DataFrame([valid, invalid]),
    )

    assert first.loc[0, "disposition"] == Disposition.PARSED_STATE.value
    assert first.loc[0, "evidence_source_document"] == "z.pdf"
    pd.testing.assert_frame_equal(first, second)


def test_same_priority_bound_state_conflict_fails_closed() -> None:
    checkpoint = _lifecycle_checkpoint()
    first = _lifecycle_evidence("first")
    second = _lifecycle_evidence(
        "second", equity_min_pct=0.0, equity_max_pct=30.0,
        evidence="股票资产占基金资产的比例为0%-30%。",
        section_text="股票资产占基金资产的比例为0%-30%。",
        parsed_evidence_raw="股票资产占基金资产的比例为0%-30%。",
    )

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), pd.DataFrame(), pd.DataFrame(),
        lifecycle_evidence=pd.DataFrame([first, second]),
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SOURCE_CONFLICT"


def test_lifecycle_binding_still_requires_explicit_quote_state_support() -> None:
    checkpoint = _lifecycle_checkpoint()
    sections = _adjudication_sections(_lifecycle_section("candidate"))

    result = adjudicate_checkpoints(
        pd.DataFrame([checkpoint]), sections, _parsed_state("candidate")
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "disposition"] != Disposition.VERIFIED_NOOP.value


def test_normalization_preserves_numbers_negation_comparators_and_denominator():
    raw = "第 12 页\n股票资产 不低于 基金资产净值的４０％。\n第 13 页"
    normalized = normalize_investment_clause(raw)
    assert NORMALIZATION_VERSION == 1
    assert "不低于" in normalized
    assert "基金资产净值" in normalized
    assert "40%" in normalized
    assert "第12页" not in normalized


def test_clause_fingerprint_is_deterministic_and_enum_is_closed():
    normalized = normalize_investment_clause("股票资产占基金资产的60%-95%")
    assert fingerprint_clause(normalized) == fingerprint_clause(normalized)
    assert len(fingerprint_clause(normalized)) == 64
    assert {item.value for item in Disposition} == {
        "PARSED_STATE", "VERIFIED_NOOP", "UNRESOLVED"
    }


def _aliases() -> pd.DataFrame:
    return pd.DataFrame({"query_code": ["49"], "family_key": ["family-a"]})


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fundCode": ["49"],
            "uploadInfoId": ["notice-1"],
            "reportSendDate": ["2020-01-02"],
            "reportName": ["基金转型有关事项公告"],
            "reportCode": ["FC010001"],
            "inception_date": ["2020-01-01"],
            "end_date": [None],
        }
    )


def _sections(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_candidate_ids_ignore_parser_status_and_disposition() -> None:
    sections = _sections(
        [
            {
                "share_code": "000049",
                "upload_info_id": "section-1",
                "known_at": "2020-01-03",
                "section_name": "investment_range",
                "section_text": "股票资产占基金资产的60%-95%",
                "parser_status": "success",
                "disposition": "PARSED_STATE",
            }
        ]
    )
    changed = sections.assign(parser_status="failure", disposition="UNRESOLVED")

    first = generate_required_checkpoints(_metadata(), _aliases(), sections)
    second = generate_required_checkpoints(_metadata(), _aliases(), changed)

    assert GENERATION_RULE_VERSION == 2
    assert first.checkpoint_id.tolist() == second.checkpoint_id.tolist()


def test_sample_authority_emits_all_60_exact_trailing_hyphen_family_keys() -> None:
    sample = pd.DataFrame(
        [
            {
                "share_code": f"{index:06d}",
                "family_key": f"manager-{index}|family-{index}-",
                "status": "存续",
                "inception_date": "2020-01-01",
            }
            for index in range(1, 61)
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "query_code": f"{index:06d}",
                "family_key": f"manager-{index}|family-{index}",
            }
            for index in range(1, 61)
        ]
    )

    checkpoints = generate_required_checkpoints(
        pd.DataFrame(), aliases, family_sample=sample
    )

    inception = checkpoints.loc[checkpoints.trigger_type.eq("INCEPTION")]
    assert len(inception) == 60
    assert set(inception.family_key) == set(sample.family_key)
    assert set(inception.share_code) == {f"{index:06d}" for index in range(1, 61)}


def test_share_class_suffix_aliases_map_to_exact_sample_family_identity() -> None:
    sample = pd.DataFrame(
        [
            {
                "share_code": "150011",
                "family_key": "manager-a|fund-a-进取",
                "status": "清盘",
                "inception_date": "2010-02-10",
            },
            {
                "share_code": "150017",
                "family_key": "manager-b|fund-b-B",
                "status": "清盘",
                "inception_date": "2010-04-22",
            },
        ]
    )
    aliases = pd.DataFrame(
        [
            {"query_code": "160212", "family_key": "manager-a|fund-a"},
            {"query_code": "150016", "family_key": "manager-b|fund-b"},
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "fundCode": "160212",
                "uploadInfoId": "transform-a",
                "reportSendDate": "2020-01-02",
                "reportName": "基金转型公告",
            },
            {
                "fundCode": "150016",
                "uploadInfoId": "merge-b",
                "reportSendDate": "2020-01-03",
                "reportName": "基金合并公告",
            },
        ]
    )

    checkpoints = generate_required_checkpoints(
        metadata, aliases, family_sample=sample
    )

    events = checkpoints.loc[
        checkpoints.trigger_type.isin(["TRANSFORMATION", "MERGER"])
    ]
    assert set(events.family_key) == {"manager-a|fund-a-进取", "manager-b|fund-b-B"}
    assert set(checkpoints.family_key) == {
        "manager-a|fund-a-进取",
        "manager-b|fund-b-B",
    }


def test_metadata_empty_sample_families_keep_critical_lifecycle_rows() -> None:
    sample = pd.DataFrame(
        [
            {
                "share_code": "000001",
                "family_key": "manager|active-family-",
                "status": "存续",
                "inception_date": "2018-05-04",
            },
            {
                "share_code": "000002",
                "family_key": "manager|liquidated-family-",
                "status": "清盘",
                "inception_date": "2019-06-05",
            },
        ]
    )
    aliases = pd.DataFrame(columns=["query_code", "family_key"])

    checkpoints = generate_required_checkpoints(
        pd.DataFrame(), aliases, family_sample=sample
    )

    assert checkpoints.trigger_type.value_counts().to_dict() == {
        "INCEPTION": 2,
        "TERMINAL_STATE": 1,
    }
    terminal = checkpoints.loc[checkpoints.trigger_type.eq("TERMINAL_STATE")].iloc[0]
    assert terminal.family_key == "manager|liquidated-family-"
    assert terminal.share_code == "000002"
    assert terminal.known_at == ""
    assert terminal.effective_date == ""
    assert checkpoints.is_critical.map(bool).all()
    assert set(checkpoints.generation_rule_version) == {2}
    assert set(checkpoints.normalization_version) == {1}


def test_ambiguous_normalized_alias_match_is_rejected() -> None:
    sample = pd.DataFrame(
        [
            {
                "share_code": "000001",
                "family_key": "manager|same-family-A",
                "status": "存续",
                "inception_date": "2020-01-01",
            },
            {
                "share_code": "000002",
                "family_key": "manager|same-family-B",
                "status": "存续",
                "inception_date": "2020-01-02",
            },
        ]
    )
    aliases = pd.DataFrame(
        [{"query_code": "999999", "family_key": "manager|same-family"}]
    )

    with pytest.raises(ValueError, match="ambiguous normalized alias"):
        generate_required_checkpoints(pd.DataFrame(), aliases, family_sample=sample)


def test_failed_critical_section_generates_uncomparable_denominator_row() -> None:
    sections = _sections(
        [
            {
                "share_code": "000049",
                "upload_info_id": "broken-section",
                "known_at": "2020-01-03",
                "section_name": "investment_range",
                "section_text": None,
                "section_comparable": False,
                "parser_status": "failure",
            }
        ]
    )

    checkpoints = generate_required_checkpoints(_metadata(), _aliases(), sections)

    row = checkpoints.loc[checkpoints.trigger_type.eq("SECTION_UNCOMPARABLE")].iloc[0]
    assert bool(row.is_critical) is True
    assert row.upload_info_id == "broken-section"


def test_explicit_transformation_is_critical() -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())

    row = checkpoints.loc[checkpoints.trigger_type.eq("TRANSFORMATION")].iloc[0]
    assert bool(row.is_critical) is True
    assert row.share_code == "000049"


def test_material_sections_only_add_first_and_changed_numerical_boundary() -> None:
    sections = _sections(
        [
            {
                "share_code": "000049",
                "upload_info_id": "one",
                "known_at": "2020-01-03",
                "section_name": "investment_range",
                "section_text": "股票资产占基金资产的60%-95%",
            },
            {
                "share_code": "000049",
                "upload_info_id": "two",
                "known_at": "2020-02-03",
                "section_name": "investment_range",
                "section_text": "股票资产占基金资产的60%-95%",
            },
            {
                "share_code": "000049",
                "upload_info_id": "three",
                "known_at": "2020-03-03",
                "section_name": "investment_range",
                "section_text": "股票资产占基金资产的40%-95%",
            },
        ]
    )

    checkpoints = generate_required_checkpoints(_metadata().iloc[0:0], _aliases(), sections)

    changed = checkpoints.loc[checkpoints.trigger_type.eq("MATERIAL_SECTION_CHANGE")]
    assert changed.upload_info_id.tolist() == ["one", "three"]


def test_checkpoint_ids_are_stable_when_inputs_are_reordered() -> None:
    metadata = pd.concat([_metadata(), _metadata().assign(uploadInfoId="notice-2")])
    sections = _sections(
        [
            {
                "share_code": "000049",
                "upload_info_id": "section-1",
                "known_at": "2020-01-03",
                "section_text": "股票资产占基金资产的60%-95%",
            },
            {
                "share_code": "000049",
                "upload_info_id": "section-2",
                "known_at": "2020-02-03",
                "section_text": "股票资产占基金资产的40%-95%",
            },
        ]
    )

    first = generate_required_checkpoints(metadata, _aliases(), sections)
    second = generate_required_checkpoints(
        metadata.iloc[::-1], _aliases().iloc[::-1], sections.iloc[::-1]
    )

    assert first.checkpoint_id.tolist() == second.checkpoint_id.tolist()


def test_later_terminal_metadata_generates_terminal_state_checkpoint() -> None:
    metadata = pd.DataFrame(
        {
            "fundCode": ["49", "49"],
            "uploadInfoId": ["active-notice", "terminal-notice"],
            "reportSendDate": ["2020-01-02", "2022-01-02"],
            "reportName": ["定期报告", "终止公告"],
            "inception_date": ["2020-01-01", "2020-01-01"],
            "end_date": [None, "2021-12-31"],
            "status": ["存续", "终止"],
        }
    )

    checkpoints = generate_required_checkpoints(metadata, _aliases())

    terminal = checkpoints.loc[checkpoints.trigger_type.eq("TERMINAL_STATE")].iloc[0]
    assert terminal.effective_date == "2021-12-31"
    assert terminal.known_at == "2022-01-02"


def test_duplicate_checkpoint_content_collapses_deterministically_under_reordering() -> None:
    metadata = pd.DataFrame(
        {
            "fundCode": ["49", "49"],
            "uploadInfoId": ["same-upload", "same-upload"],
            "reportSendDate": ["2020-01-02", "2020-01-02"],
            "reportName": ["基金转型有关事项公告", "基金转型有关事项公告"],
            "source_document": ["z-source.pdf", "a-source.pdf"],
        }
    )

    first = generate_required_checkpoints(metadata, _aliases())
    second = generate_required_checkpoints(metadata.iloc[::-1], _aliases())

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first.trigger_type.eq("TRANSFORMATION"), "source_document"].item() == "a-source.pdf"


def test_freeze_round_trips_leading_zero_codes(tmp_path: Path) -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())
    path = freeze_checkpoint_manifest(checkpoints, tmp_path / "manifest.csv")

    restored = pd.read_csv(path, dtype={"share_code": str})

    assert restored.share_code.tolist() == ["000049"] * len(restored)


def test_freeze_flushes_and_fsyncs_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def recording_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def recording_replace(source: Path, destination: Path) -> Path:
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(Path, "replace", recording_replace)

    path = freeze_checkpoint_manifest(checkpoints, tmp_path / "manifest.csv")

    assert events == ["fsync", "replace"]
    assert pd.read_csv(path, dtype={"share_code": str}).share_code.iloc[0] == "000049"


def test_freeze_retries_a_transient_windows_replace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())
    attempts = 0
    real_replace = Path.replace

    def fail_once(source: Path, destination: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "synthetic transient lock", str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_once)

    path = freeze_checkpoint_manifest(checkpoints, tmp_path / "manifest.csv")

    assert attempts == 2
    assert path.is_file()


def test_freeze_rejects_duplicate_deletion_and_mutation(tmp_path: Path) -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())
    path = tmp_path / "manifest.csv"
    freeze_checkpoint_manifest(checkpoints, path)

    with pytest.raises(ValueError, match="duplicate"):
        freeze_checkpoint_manifest(pd.concat([checkpoints, checkpoints.iloc[[0]]]), path)
    with pytest.raises(ValueError, match="deletion"):
        freeze_checkpoint_manifest(checkpoints.iloc[1:], path)
    mutated = checkpoints.copy()
    mutated.loc[0, "source_document"] = "mutated.pdf"
    with pytest.raises(ValueError, match="mutation"):
        freeze_checkpoint_manifest(mutated, path)


def test_freeze_accepts_append_only_revision_chain(tmp_path: Path) -> None:
    checkpoints = generate_required_checkpoints(_metadata(), _aliases())
    path = tmp_path / "manifest.csv"
    freeze_checkpoint_manifest(checkpoints, path)
    replacement = checkpoints.iloc[[0]].copy()
    replacement.loc[:, "checkpoint_id"] = "a" * 64
    replacement.loc[:, "manifest_revision"] = 2
    replacement.loc[:, "supersedes_checkpoint_id"] = checkpoints.iloc[0].checkpoint_id
    replacement.loc[:, "revision_reason"] = "corrected source label"
    replacement.loc[:, "source_document"] = "corrected.pdf"

    freeze_checkpoint_manifest(pd.concat([checkpoints, replacement], ignore_index=True), path)
    restored = pd.read_csv(path, dtype=str, keep_default_na=False)

    assert restored.checkpoint_id.tolist()[-1] == "a" * 64
    assert restored.checkpoint_id.tolist()[: len(checkpoints)] == checkpoints.checkpoint_id.tolist()


def test_regex_miss_without_extracted_evidence_is_unresolved() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "failure",
            "section_text": "",
            "evidence_location": "notice-1.pdf:p4",
        }
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, pd.DataFrame())

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SECTION_EXTRACTION_FAILED"


def test_first_explicit_state_then_exact_section_match_is_verified_noop() -> None:
    checkpoints = _adjudication_checkpoints(2)
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        },
        {
            "upload_info_id": "notice-2",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-2.pdf:p5",
        },
    )

    result = adjudicate_checkpoints(checkpoints, sections, _parsed_state("notice-1"))

    assert result.disposition.tolist() == [
        Disposition.PARSED_STATE.value,
        Disposition.VERIFIED_NOOP.value,
    ]
    assert result.loc[1, "predecessor_checkpoint_id"] == "checkpoint-1"
    assert result.loc[1, "clause_sha256"] == result.loc[0, "clause_sha256"]
    assert result.loc[1, "evidence_raw"]


def test_changed_numeric_boundary_cannot_inherit_noop() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        },
        {
            "upload_info_id": "notice-2",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为40%-95%。",
            "evidence_location": "notice-2.pdf:p5",
        },
    )

    result = adjudicate_checkpoints(
        _adjudication_checkpoints(2), sections, _parsed_state("notice-1")
    )

    assert result.loc[1, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[1, "failure_reason"] == "POSSIBLE_STATE_CHANGE"


def test_amendment_scope_cannot_override_a_changed_numeric_boundary() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        },
        {
            "upload_info_id": "notice-2",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为40%-95%。",
            "evidence_location": "notice-2.pdf:p5",
            "amendment_scope": "non-classification",
            "comparison_status": "success",
            "comparison_result": "unchanged",
        },
    )

    result = adjudicate_checkpoints(
        _adjudication_checkpoints(2), sections, _parsed_state("notice-1")
    )

    assert result.loc[1, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[1, "failure_reason"] == "POSSIBLE_STATE_CHANGE"


@pytest.mark.parametrize(
    "unsupported_fields",
    [
        {"tushare_fund_type": "混合型-偏股"},
        {"fund_name": "偏股精选混合"},
        {"model_confidence": 0.99},
    ],
)
def test_non_quoted_inference_never_resolves_a_checkpoint(
    unsupported_fields: dict[str, object],
) -> None:
    checkpoints = _adjudication_checkpoints()
    for column, value in unsupported_fields.items():
        checkpoints[column] = value
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        }
    )

    result = adjudicate_checkpoints(checkpoints, sections, pd.DataFrame())

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value


def test_later_document_cannot_be_used_as_predecessor() -> None:
    checkpoints = _adjudication_checkpoints(2).iloc[::-1].reset_index(drop=True)
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        },
        {
            "upload_info_id": "notice-2",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-2.pdf:p5",
        },
    )

    result = adjudicate_checkpoints(checkpoints, sections, _parsed_state("notice-2"))

    earlier = result.loc[result.checkpoint_id.eq("checkpoint-1")].iloc[0]
    assert earlier.disposition == Disposition.UNRESOLVED.value
    assert earlier.failure_reason == "MISSING_PREDECESSOR"


@pytest.mark.parametrize(
    ("section", "expected_reason"),
    [
        ({"pdf_status": "missing"}, "PDF_MISSING"),
        ({"asset_denominator_status": "ambiguous"}, "AMBIGUOUS_ASSET_DENOMINATOR"),
    ],
)
def test_explicit_evidence_failures_use_frozen_reason_codes(
    section: dict[str, object], expected_reason: str
) -> None:
    section.update(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        }
    )

    result = adjudicate_checkpoints(
        _adjudication_checkpoints(), _adjudication_sections(section), pd.DataFrame()
    )

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == expected_reason


def test_nested_parsed_document_pdf_failure_uses_pdf_missing_reason() -> None:
    parsed = pd.DataFrame(
        [
            {
                "upload_info_id": "notice-1",
                "status": "failure",
                "failure": {"reason": "PDF missing from official archive"},
            }
        ]
    )
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        }
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "failure_reason"] == "PDF_MISSING"


def test_conflicting_official_states_for_one_checkpoint_are_not_voted() -> None:
    parsed = pd.concat(
        [_parsed_state("notice-1", low=60, high=95), _parsed_state("notice-1", low=0, high=30)],
        ignore_index=True,
    )
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        }
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SOURCE_CONFLICT"


def test_clause_clusters_are_deterministic_and_retain_every_source_member() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-2",
            "known_at": "2020-02-02",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
        },
        {
            "upload_info_id": "notice-1",
            "known_at": "2020-01-02",
            "section_text": " 股票资产 占基金资产的比例为60%-95% 。 ",
        },
        {
            "upload_info_id": "notice-3",
            "known_at": "2020-03-02",
            "section_text": "",
        },
    )

    first = build_clause_clusters(sections)
    second = build_clause_clusters(sections.iloc[::-1])

    pd.testing.assert_frame_equal(first, second)
    assert first.upload_info_id.tolist() == ["notice-1", "notice-2"]
    assert first.cluster_member_count.tolist() == [2, 2]
    assert first.cluster_id.nunique() == 1


def test_successes_always_preserve_raw_normalized_fingerprint_and_location() -> None:
    result = adjudicate_checkpoints(
        _adjudication_checkpoints(),
        _adjudication_sections(
            {
                "upload_info_id": "notice-1",
                "extraction_status": "success",
                "section_text": "股票资产占基金资产的比例为60%-95%。",
                "evidence_location": "notice-1.pdf:p4",
            }
        ),
        _parsed_state("notice-1"),
    )

    successful = result.loc[result.disposition.ne(Disposition.UNRESOLVED.value)]
    assert successful[["evidence_raw", "evidence_normalized", "clause_sha256", "evidence_location"]].ne("").all().all()


def test_nonclassification_assertion_cannot_noop_when_critical_subject_changes() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
            "source_document": "notice-1.pdf",
        },
        {
            "upload_info_id": "notice-2",
            "extraction_status": "success",
            "section_text": "债券资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-2.pdf:p4",
            "source_document": "notice-2.pdf",
            "amendment_scope": "non-classification",
            "comparison_status": "success",
            "comparison_result": "unchanged",
        },
    )

    result = adjudicate_checkpoints(
        _adjudication_checkpoints(2), sections, _parsed_state("notice-1")
    )

    assert result.loc[1, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[1, "failure_reason"] == "POSSIBLE_STATE_CHANGE"


def test_parser_state_requires_current_document_and_current_clause_quote() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "current.pdf:p4",
            "source_document": "current.pdf",
        }
    )
    parsed = pd.DataFrame(
        [
            {
                "upload_info_id": "notice-1",
                "status": "success",
                "parser_version": 16,
                "document": {
                    "evidence": "债券资产占基金资产的比例为0%-30%",
                    "source_document": "unrelated.pdf",
                    "equity_min_pct": 0.0,
                    "equity_max_pct": 30.0,
                },
            }
        ]
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value


def test_clause_clusters_break_complete_ties_deterministically() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "same-upload",
            "known_at": "2020-01-02",
            "effective_date": "2020-01-01",
            "source_document": "z-source.pdf",
            "evidence_location": "z-source.pdf:p4",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
        },
        {
            "upload_info_id": "same-upload",
            "known_at": "2020-01-02",
            "effective_date": "2020-01-01",
            "source_document": "a-source.pdf",
            "evidence_location": "a-source.pdf:p4",
            "section_text": "股票资产 占基金资产的比例为60%-95%。",
        },
    )

    first = build_clause_clusters(sections)
    second = build_clause_clusters(sections.iloc[::-1])

    pd.testing.assert_frame_equal(first, second)
    assert first.source_document.tolist() == ["a-source.pdf", "z-source.pdf"]


@pytest.mark.parametrize("incoming_conflict", ["", "SOURCE_CONFLICT"])
def test_conflicting_critical_sections_preserve_source_conflict_priority(
    incoming_conflict: str,
) -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
            "failure_reason": incoming_conflict,
        },
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为0%-30%。",
            "evidence_location": "notice-1.pdf:p5",
        },
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, pd.DataFrame())

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value
    assert result.loc[0, "failure_reason"] == "SOURCE_CONFLICT"


@pytest.mark.parametrize(
    ("quote", "low", "high"),
    [
        ("股票资产", 0.0, 30.0),
        ("股票资产占基金资产的比例为60%-95%", 0.0, 30.0),
    ],
)
def test_same_source_quote_must_explicitly_support_parsed_state_values(
    quote: str, low: float, high: float
) -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
            "source_document": "notice-1.pdf",
        }
    )
    parsed = pd.DataFrame(
        [
            {
                "upload_info_id": "notice-1",
                "status": "success",
                "parser_version": 16,
                "document": {
                    "evidence": quote,
                    "source_document": "notice-1.pdf",
                    "equity_min_pct": low,
                    "equity_max_pct": high,
                },
            }
        ]
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "disposition"] == Disposition.UNRESOLVED.value


def test_exact_same_source_quote_supports_parsed_state_values() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
            "source_document": "notice-1.pdf",
        }
    )
    parsed = pd.DataFrame(
        [
            {
                "upload_info_id": "notice-1",
                "status": "success",
                "parser_version": 16,
                "document": {
                    "evidence": "股票资产占基金资产的比例为60%-95%",
                    "source_document": "notice-1.pdf",
                    "equity_min_pct": 60.0,
                    "equity_max_pct": 95.0,
                    "fund_type": "混合型-偏股",
                },
            }
        ]
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "disposition"] == Disposition.PARSED_STATE.value
    assert result.loc[0, "equity_min_pct"] == 60.0
    assert result.loc[0, "equity_max_pct"] == 95.0


def test_nested_failure_reason_source_conflict_is_preserved() -> None:
    sections = _adjudication_sections(
        {
            "upload_info_id": "notice-1",
            "extraction_status": "success",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
            "evidence_location": "notice-1.pdf:p4",
        }
    )
    parsed = pd.DataFrame(
        [
            {
                "upload_info_id": "notice-1",
                "status": "failure",
                "failure": {"reason": "SOURCE_CONFLICT"},
            }
        ]
    )

    result = adjudicate_checkpoints(_adjudication_checkpoints(), sections, parsed)

    assert result.loc[0, "failure_reason"] == "SOURCE_CONFLICT"


def test_timeline_noop_copies_only_explicit_predecessor_after_known_at() -> None:
    dispositions = _timeline_dispositions()
    timeline = build_checkpoint_timeline(dispositions)

    noop = timeline.loc[timeline.checkpoint_id.eq("a-2")].iloc[0]
    assert noop.state_source_checkpoint_id == "a-1"
    assert noop.checkpoint_id == "a-2"
    assert noop.equity_min_pct == 60.0
    assert noop.effective_from == pd.Timestamp("2020-02-10")


def test_later_known_document_quoting_historical_date_does_not_backfill_history() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "known_at"] = "2020-03-10"
    dispositions.loc[0, "effective_date"] = "2020-01-01"

    timeline = build_checkpoint_timeline(dispositions)

    assert timeline.loc[0, "effective_from"] == pd.Timestamp("2020-03-10")
    assert bool(timeline.loc[0, "retrospective_effective_date"]) is True
    assert timeline.loc[0, "causal_violation"] == False
    assert validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)["causal_violations"] == 0


def test_timeline_start_before_source_known_at_is_a_causal_violation() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "known_at"] = "2020-03-10"
    dispositions.loc[0, "effective_date"] = "2020-01-01"
    timeline = build_checkpoint_timeline(dispositions)
    timeline.loc[0, "effective_from"] = pd.Timestamp("2020-01-01")
    timeline.loc[0, "causal_violation"] = False

    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert counts["causal_violations"] == 1


def test_future_legal_effective_date_delays_state_without_causal_violation() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "known_at"] = "2020-01-10"
    dispositions.loc[0, "effective_date"] = "2020-03-10"

    timeline = build_checkpoint_timeline(dispositions)

    assert timeline.loc[0, "effective_from"] == pd.Timestamp("2020-03-10")
    assert bool(timeline.loc[0, "retrospective_effective_date"]) is False
    assert validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)["causal_violations"] == 0


def test_future_known_state_source_is_a_causal_violation() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "effective_date"] = dispositions.loc[0, "known_at"]
    future = dispositions.iloc[[0]].copy()
    future.loc[:, "checkpoint_id"] = "a-future"
    future.loc[:, "known_at"] = "2020-04-10"
    future.loc[:, "effective_date"] = "2020-04-10"
    dispositions = pd.concat([dispositions, future], ignore_index=True)
    timeline = build_checkpoint_timeline(dispositions)
    timeline.loc[timeline.checkpoint_id.eq("a-1"), "state_source_checkpoint_id"] = "a-future"

    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert counts["causal_violations"] == 1


def test_parsed_state_without_known_at_is_not_usable_and_is_causal() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "known_at"] = ""
    dispositions.loc[0, "effective_date"] = "2020-01-10"
    dispositions.loc[0, "evidence_raw"] = "股票资产占基金资产的比例为60%-95%"

    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert pd.isna(timeline.loc[0, "effective_from"])
    assert timeline.loc[0, "effective_date"] == "2020-01-10"
    assert timeline.loc[0, "evidence_raw"] == "股票资产占基金资产的比例为60%-95%"
    assert bool(timeline.loc[0, "causal_violation"]) is True
    assert counts["causal_violations"] == 1


def test_noop_without_known_at_is_not_usable_and_is_causal() -> None:
    dispositions = _timeline_dispositions()
    dispositions.loc[1, "known_at"] = ""

    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)
    noop = timeline.loc[timeline.checkpoint_id.eq("a-2")].iloc[0]

    assert pd.isna(noop.effective_from)
    assert bool(noop.causal_violation) is True
    assert counts["causal_violations"] == 1


@pytest.mark.parametrize(
    ("source_id", "source_disposition"),
    [
        ("", None),
        ("missing-source", None),
        ("a-source", "UNRESOLVED"),
        ("a-source", "VERIFIED_NOOP"),
    ],
)
def test_usable_state_requires_final_parsed_state_source(
    source_id: str, source_disposition: str | None
) -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "effective_date"] = dispositions.loc[0, "known_at"]
    if source_disposition is not None:
        source = dispositions.iloc[[0]].copy()
        source.loc[:, "checkpoint_id"] = "a-source"
        source.loc[:, "known_at"] = "2020-01-01"
        source.loc[:, "effective_date"] = "2020-01-01"
        source.loc[:, "disposition"] = source_disposition
        if source_disposition == "VERIFIED_NOOP":
            origin = dispositions.iloc[[0]].copy()
            origin.loc[:, "checkpoint_id"] = "a-origin"
            origin.loc[:, "known_at"] = "2019-12-01"
            origin.loc[:, "effective_date"] = "2019-12-01"
            source.loc[:, "predecessor_checkpoint_id"] = "a-origin"
            dispositions = pd.concat([origin, source, dispositions], ignore_index=True)
        else:
            source.loc[:, "predecessor_checkpoint_id"] = ""
            dispositions = pd.concat([source, dispositions], ignore_index=True)
    timeline = build_checkpoint_timeline(dispositions)
    timeline.loc[timeline.checkpoint_id.eq("a-1"), "state_source_checkpoint_id"] = source_id

    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert counts["causal_violations"] == 1


def test_cross_family_and_later_predecessor_are_rejected_and_counted() -> None:
    dispositions = _timeline_dispositions().iloc[[0]].copy()
    dispositions.loc[0, "effective_date"] = dispositions.loc[0, "known_at"]
    later = dispositions.iloc[[0]].copy()
    later.loc[:, "checkpoint_id"] = "a-3"
    later.loc[:, "known_at"] = "2020-03-10"
    later.loc[:, "effective_date"] = "2020-03-10"
    later.loc[:, "disposition"] = "PARSED_STATE"
    cross = later.copy()
    cross.loc[:, "checkpoint_id"] = "b-1"
    cross.loc[:, "family_key"] = "family-b"
    cross.loc[:, "disposition"] = "VERIFIED_NOOP"
    cross.loc[:, "predecessor_checkpoint_id"] = "a-1"
    dispositions = pd.concat([dispositions, later, cross], ignore_index=True)
    dispositions.loc[1, "disposition"] = "VERIFIED_NOOP"
    dispositions.loc[1, "predecessor_checkpoint_id"] = "a-3"

    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert timeline.loc[timeline.checkpoint_id.isin(["a-3", "b-1"]), "effective_from"].isna().all()
    assert counts["causal_violations"] == 2
    assert counts["orphan_noops"] == 2


def test_missing_or_unresolved_predecessor_increments_orphan_noop() -> None:
    dispositions = _timeline_dispositions().iloc[[0]].copy()
    missing = dispositions.iloc[[0]].copy()
    missing.loc[:, "checkpoint_id"] = "a-2"
    missing.loc[:, "known_at"] = "2020-02-10"
    missing.loc[:, "effective_date"] = "2020-02-10"
    missing.loc[:, "disposition"] = "VERIFIED_NOOP"
    missing.loc[:, "predecessor_checkpoint_id"] = "not-present"
    unresolved = missing.copy()
    unresolved.loc[:, "checkpoint_id"] = "a-4"
    unresolved.loc[:, "known_at"] = "2020-02-15"
    unresolved.loc[:, "effective_date"] = "2020-02-15"
    unresolved.loc[:, "predecessor_checkpoint_id"] = ""
    unresolved.loc[:, "disposition"] = "UNRESOLVED"
    noop_of_unresolved = missing.copy()
    noop_of_unresolved.loc[:, "checkpoint_id"] = "a-3"
    noop_of_unresolved.loc[:, "known_at"] = "2020-03-10"
    noop_of_unresolved.loc[:, "effective_date"] = "2020-03-10"
    noop_of_unresolved.loc[:, "predecessor_checkpoint_id"] = "a-4"
    dispositions = pd.concat([dispositions, missing, unresolved, noop_of_unresolved], ignore_index=True)

    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert counts["orphan_noops"] == 2


def test_source_conflict_has_no_usable_interval_and_counts_unresolved_conflict() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "disposition"] = "UNRESOLVED"
    dispositions.loc[0, "failure_reason"] = "SOURCE_CONFLICT"

    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert timeline.loc[0, "effective_from"] is pd.NaT or pd.isna(timeline.loc[0, "effective_from"])
    assert counts["unresolved_conflicts"] == 1


def test_critical_unresolved_is_joined_from_frozen_manifest() -> None:
    dispositions = _timeline_dispositions().iloc[:1].copy()
    dispositions.loc[0, "disposition"] = "UNRESOLVED"
    dispositions.loc[0, "failure_reason"] = "SECTION_EXTRACTION_FAILED"
    manifest = _manifest_for(dispositions)
    noncritical = manifest.copy()
    noncritical.loc[:, "is_critical"] = False

    critical_timeline = build_checkpoint_timeline(dispositions)
    noncritical_counts = validate_checkpoint_invariants(noncritical, dispositions, critical_timeline)
    critical_counts = validate_checkpoint_invariants(manifest, dispositions, critical_timeline)

    assert noncritical_counts["critical_unresolved"] == 0
    assert critical_counts["critical_unresolved"] == 1


def test_same_time_states_share_interval_until_next_strictly_later_state() -> None:
    first = _timeline_dispositions().iloc[[0]].copy()
    first.loc[:, "effective_date"] = "2020-01-10"
    same_time = first.copy()
    same_time.loc[:, "checkpoint_id"] = "a-2"
    later = first.copy()
    later.loc[:, "checkpoint_id"] = "a-3"
    later.loc[:, "known_at"] = "2020-03-10"
    later.loc[:, "effective_date"] = "2020-03-10"
    dispositions = pd.concat([first, same_time, later], ignore_index=True)
    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    same_time_rows = timeline.loc[timeline.checkpoint_id.isin(["a-1", "a-2"])]
    assert same_time_rows["effective_from"].tolist() == [
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-10"),
    ]
    assert same_time_rows["effective_to"].tolist() == [
        pd.Timestamp("2020-03-10"),
        pd.Timestamp("2020-03-10"),
    ]
    assert pd.isna(timeline.loc[timeline.checkpoint_id.eq("a-3"), "effective_to"].item())
    assert counts["nonpositive_intervals"] == 0


def test_multiple_same_time_groups_use_next_strictly_later_time_per_family() -> None:
    first = _timeline_dispositions().iloc[[0]].copy()
    first.loc[:, "effective_date"] = "2020-01-10"
    rows = []
    for checkpoint_id, family_key, date, low, high in [
        ("a-1", "family-a", "2020-01-10", 60.0, 100.0),
        ("a-2", "family-a", "2020-01-10", 60.0, 95.0),
        ("a-3", "family-a", "2020-02-10", 70.0, 95.0),
        ("a-4", "family-a", "2020-02-10", 70.0, 90.0),
        ("b-1", "family-b", "2020-01-20", 40.0, 80.0),
    ]:
        row = first.copy()
        row.loc[:, "checkpoint_id"] = checkpoint_id
        row.loc[:, "family_key"] = family_key
        row.loc[:, "known_at"] = date
        row.loc[:, "effective_date"] = date
        row.loc[:, "equity_min_pct"] = low
        row.loc[:, "equity_max_pct"] = high
        rows.append(row)
    dispositions = pd.concat(rows, ignore_index=True)

    timeline = build_checkpoint_timeline(dispositions)

    first_group = timeline.loc[timeline.checkpoint_id.isin(["a-1", "a-2"])]
    second_group = timeline.loc[timeline.checkpoint_id.isin(["a-3", "a-4"])]
    family_b = timeline.loc[timeline.checkpoint_id.eq("b-1")].iloc[0]
    assert set(first_group["equity_min_pct"]) == {60.0}
    assert set(first_group["equity_max_pct"]) == {95.0}
    assert set(first_group["effective_to"]) == {pd.Timestamp("2020-02-10")}
    assert set(second_group["equity_min_pct"]) == {70.0}
    assert set(second_group["equity_max_pct"]) == {90.0}
    assert second_group["effective_to"].isna().all()
    assert pd.isna(family_b.effective_to)


def test_interval_boundary_uses_effective_from_order_not_known_at_order() -> None:
    first = _timeline_dispositions().iloc[[0]].copy()
    rows = []
    for checkpoint_id, known_at, effective_date in [
        ("later-effective-first-known", "2020-01-01", "2020-03-01"),
        ("earlier-effective-later-known", "2020-02-01", "2020-02-01"),
    ]:
        row = first.copy()
        row.loc[:, "checkpoint_id"] = checkpoint_id
        row.loc[:, "known_at"] = known_at
        row.loc[:, "effective_date"] = effective_date
        rows.append(row)

    timeline = build_checkpoint_timeline(pd.concat(rows, ignore_index=True))

    earlier = timeline.loc[
        timeline.checkpoint_id.eq("earlier-effective-later-known")
    ].iloc[0]
    later = timeline.loc[
        timeline.checkpoint_id.eq("later-effective-first-known")
    ].iloc[0]
    assert earlier.effective_from == pd.Timestamp("2020-02-01")
    assert earlier.effective_to == pd.Timestamp("2020-03-01")
    assert pd.isna(later.effective_to)


def test_reversed_effective_from_sequence_maps_every_row_to_next_strict_time() -> None:
    first = _timeline_dispositions().iloc[[0]].copy()
    rows = []
    for checkpoint_id, known_at, effective_date in [
        ("at-may", "2020-01-01", "2020-05-01"),
        ("at-april", "2020-02-01", "2020-04-01"),
        ("at-june", "2020-06-01", "2020-06-01"),
    ]:
        row = first.copy()
        row.loc[:, "checkpoint_id"] = checkpoint_id
        row.loc[:, "known_at"] = known_at
        row.loc[:, "effective_date"] = effective_date
        rows.append(row)

    timeline = build_checkpoint_timeline(pd.concat(rows, ignore_index=True))
    by_id = timeline.set_index("checkpoint_id")

    assert by_id.loc["at-april", "effective_to"] == pd.Timestamp("2020-05-01")
    assert by_id.loc["at-may", "effective_to"] == pd.Timestamp("2020-06-01")
    assert pd.isna(by_id.loc["at-june", "effective_to"])


def test_same_time_noop_keeps_inherited_state_and_shares_later_boundary() -> None:
    parsed = _timeline_dispositions().iloc[[0]].copy()
    parsed.loc[:, "known_at"] = "2020-01-10"
    parsed.loc[:, "effective_date"] = "2020-01-10"
    noop = parsed.copy()
    noop.loc[:, "checkpoint_id"] = "a-2"
    noop.loc[:, "disposition"] = "VERIFIED_NOOP"
    noop.loc[:, "predecessor_checkpoint_id"] = "a-1"
    noop.loc[:, "fund_type"] = ""
    noop.loc[:, "equity_min_pct"] = float("nan")
    noop.loc[:, "equity_max_pct"] = float("nan")
    later = parsed.copy()
    later.loc[:, "checkpoint_id"] = "a-3"
    later.loc[:, "known_at"] = "2020-02-10"
    later.loc[:, "effective_date"] = "2020-02-10"

    dispositions = pd.concat([parsed, noop, later], ignore_index=True)
    timeline = build_checkpoint_timeline(dispositions)
    noop_row = timeline.loc[timeline.checkpoint_id.eq("a-2")].iloc[0]

    assert noop_row.state_source_checkpoint_id == "a-1"
    assert noop_row.equity_min_pct == 60.0
    assert noop_row.equity_max_pct == 95.0
    assert noop_row.effective_from == pd.Timestamp("2020-01-10")
    assert noop_row.effective_to == pd.Timestamp("2020-02-10")


def test_timeline_is_deterministic_under_input_reordering() -> None:
    dispositions = _timeline_dispositions()

    first = build_checkpoint_timeline(dispositions)
    second = build_checkpoint_timeline(dispositions.iloc[::-1].reset_index(drop=True))

    pd.testing.assert_frame_equal(first, second)


def test_same_time_compatible_intersection_is_tightened_and_conflict_unresolved() -> None:
    compatible = _timeline_dispositions().iloc[:1].copy()
    compatible.loc[:, "checkpoint_id"] = "a-2"
    compatible.loc[:, "equity_min_pct"] = 70.0
    compatible.loc[:, "equity_max_pct"] = 90.0
    combined = pd.concat([_timeline_dispositions().iloc[:1], compatible], ignore_index=True)
    timeline = build_checkpoint_timeline(combined)
    assert set(timeline.equity_min_pct.dropna()) == {70.0}
    assert set(timeline.equity_max_pct.dropna()) == {90.0}

    conflict = compatible.copy()
    conflict.loc[:, "checkpoint_id"] = "a-3"
    conflict.loc[:, "equity_min_pct"] = 96.0
    conflict.loc[:, "equity_max_pct"] = 100.0
    conflict_frame = pd.concat([_timeline_dispositions().iloc[:1], conflict], ignore_index=True)
    conflict_timeline = build_checkpoint_timeline(conflict_frame)
    assert conflict_timeline.effective_from.isna().all()


def test_clean_fixture_has_zero_invariant_counts() -> None:
    clean = _timeline_dispositions().copy()
    clean.loc[0, "effective_date"] = clean.loc[0, "known_at"]
    timeline = build_checkpoint_timeline(clean)

    assert validate_checkpoint_invariants(_manifest_for(clean), clean, timeline) == {
        "causal_violations": 0,
        "unresolved_conflicts": 0,
        "nonpositive_intervals": 0,
        "orphan_noops": 0,
        "critical_unresolved": 0,
    }


def test_same_time_intersection_does_not_mutate_verified_noop_state() -> None:
    parsed = _timeline_dispositions().iloc[[0]].copy()
    parsed.loc[:, "checkpoint_id"] = "p1"
    parsed.loc[:, "known_at"] = "2020-01-01"
    parsed.loc[:, "effective_date"] = "2020-01-01"
    noop = parsed.copy()
    noop.loc[:, "checkpoint_id"] = "n2"
    noop.loc[:, "known_at"] = "2020-02-01"
    noop.loc[:, "effective_date"] = "2020-02-01"
    noop.loc[:, "disposition"] = "VERIFIED_NOOP"
    noop.loc[:, "predecessor_checkpoint_id"] = "p1"
    noop.loc[:, "equity_min_pct"] = float("nan")
    noop.loc[:, "equity_max_pct"] = float("nan")
    parsed_later = parsed.copy()
    parsed_later.loc[:, "checkpoint_id"] = "p3"
    parsed_later.loc[:, "known_at"] = "2020-02-01"
    parsed_later.loc[:, "effective_date"] = "2020-02-01"
    parsed_later.loc[:, "equity_min_pct"] = 70.0
    parsed_later.loc[:, "equity_max_pct"] = 90.0

    timeline = build_checkpoint_timeline(pd.concat([parsed, noop, parsed_later], ignore_index=True))
    noop_row = timeline.loc[timeline.checkpoint_id.eq("n2")].iloc[0]

    assert noop_row.state_source_checkpoint_id == "p1"
    assert noop_row.equity_min_pct == 60.0
    assert noop_row.equity_max_pct == 95.0


def test_same_time_parsed_fund_type_conflict_is_unresolved() -> None:
    first = _timeline_dispositions().iloc[[0]].copy()
    second = first.copy()
    second.loc[:, "checkpoint_id"] = "a-2"
    second.loc[:, "fund_type"] = "混合型-偏债"

    dispositions = pd.concat([first, second], ignore_index=True)
    timeline = build_checkpoint_timeline(dispositions)
    counts = validate_checkpoint_invariants(_manifest_for(dispositions), dispositions, timeline)

    assert timeline.effective_from.isna().all()
    assert timeline.failure_reason.tolist() == ["SOURCE_CONFLICT", "SOURCE_CONFLICT"]
    assert counts["unresolved_conflicts"] == 2
