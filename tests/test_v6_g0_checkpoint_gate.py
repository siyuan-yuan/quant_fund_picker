from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v6 import g0_checkpoint_gate as gate_module
from v6.g0_checkpoint_gate import (
    GateInputs,
    evaluate_gate,
    run_checkpoint_gate,
    write_gate_artifacts,
)


STRATA = (
    ("存续", "pre2013"),
    ("存续", "2013_2019"),
    ("存续", "2020_2026"),
    ("清盘", "pre2013"),
    ("清盘", "2013_2019"),
    ("清盘", "2020_2026"),
)


def _sleep_then_pid(seconds: float) -> int:
    time.sleep(seconds)
    return os.getpid()


def _crash_or_pid(command: str) -> int:
    if command == "crash":
        os._exit(17)
    return os.getpid()


def _sample() -> pd.DataFrame:
    rows = []
    code = 1
    for status, era in STRATA:
        for _ in range(10):
            rows.append(
                {
                    "share_code": f"{code:06d}",
                    "family_key": f"family-{code:03d}",
                    "status": status,
                    "inception_era": era,
                }
            )
            code += 1
    return pd.DataFrame(rows)


def _aliases(sample: pd.DataFrame) -> pd.DataFrame:
    aliases = sample[["share_code", "family_key"]].rename(
        columns={"share_code": "query_code"}
    )
    aliases["sample_share_code"] = sample["share_code"].tolist()
    duplicate_aliases = aliases.copy()
    duplicate_aliases["query_code"] = [f"{code:06d}" for code in range(100001, 100061)]
    return pd.concat([aliases, duplicate_aliases], ignore_index=True)


def _gate_frames(
    *,
    resolved_checkpoints: int = 95,
    critical_unresolved: int = 0,
    covered_by_stratum: tuple[int, ...] = (8, 8, 9, 9, 10, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = _sample()
    aliases = _aliases(sample)
    checkpoints = pd.DataFrame(
        {
            "checkpoint_id": [f"cp-{index:03d}" for index in range(100)],
            "family_key": [f"family-{index % 60 + 1:03d}" for index in range(100)],
            "share_code": [f"{index % 60 + 1:06d}" for index in range(100)],
            "is_critical": [index < 90 for index in range(100)],
        }
    )
    dispositions = checkpoints.copy()
    dispositions["disposition"] = [
        "PARSED_STATE" if index < resolved_checkpoints else "UNRESOLVED"
        for index in range(100)
    ]
    if critical_unresolved:
        dispositions.loc[0, "disposition"] = "UNRESOLVED"
    timeline_rows = []
    offset = 0
    for covered in covered_by_stratum:
        for family_index in range(offset, offset + covered):
            timeline_rows.append(
                {
                    "checkpoint_id": f"timeline-{family_index}",
                    "family_key": sample.loc[family_index, "family_key"],
                    "effective_from": "2020-01-01",
                    "disposition": "PARSED_STATE",
                }
            )
        offset += 10
    return checkpoints, dispositions, pd.DataFrame(timeline_rows), sample, aliases


def _evaluate(
    *,
    resolved_checkpoints: int = 95,
    critical_unresolved: int = 0,
    covered_by_stratum: tuple[int, ...] = (8, 8, 9, 9, 10, 10),
    invariant_override: tuple[str, int] | None = None,
):
    checkpoints, dispositions, timeline, sample, aliases = _gate_frames(
        resolved_checkpoints=resolved_checkpoints,
        critical_unresolved=critical_unresolved,
        covered_by_stratum=covered_by_stratum,
    )
    invariants = {
        "causal_violations": 0,
        "unresolved_conflicts": 0,
        "nonpositive_intervals": 0,
        "orphan_noops": 0,
        "critical_unresolved": critical_unresolved,
    }
    if invariant_override:
        invariants[invariant_override[0]] = invariant_override[1]
    return evaluate_gate(
        checkpoints=checkpoints,
        dispositions=dispositions,
        timeline=timeline,
        family_sample=sample,
        aliases=aliases,
        invariant_counts=invariants,
        legacy_document_counts={"success": 1676, "valid": 1993, "failure": 317},
    )


def test_exact_preregistered_boundary_passes_and_document_rate_is_diagnostic() -> None:
    result = _evaluate()

    assert result.passed is True
    assert result.metric("checkpoint_coverage").actual == pytest.approx(0.95)
    assert result.metric("critical_checkpoint_resolution").actual == pytest.approx(1.0)
    assert result.metric("fund_coverage_overall").actual == pytest.approx(0.90)
    assert result.metric("coverage_存续_pre2013").actual == pytest.approx(0.80)
    assert result.metric("coverage_存续_2013_2019").actual == pytest.approx(0.80)
    diagnostic = result.metric("document_parser_coverage")
    assert diagnostic.role == "DIAGNOSTIC"
    assert diagnostic.numerator == 1676
    assert diagnostic.denominator == 1993
    assert diagnostic.failures == 317
    assert diagnostic.actual == pytest.approx(1676 / 1993)


@pytest.mark.parametrize(
    "metric",
    [
        "checkpoint_coverage",
        "critical_checkpoint_resolution",
        "fund_coverage_overall",
        "coverage_存续_pre2013",
        "coverage_存续_2013_2019",
        "coverage_存续_2020_2026",
        "coverage_清盘_pre2013",
        "coverage_清盘_2013_2019",
        "coverage_清盘_2020_2026",
        "causal_violations",
        "unresolved_conflicts",
        "nonpositive_intervals",
        "orphan_noops",
        "critical_unresolved",
    ],
)
def test_each_primary_metric_can_fail_gate_by_itself(metric: str) -> None:
    if metric == "checkpoint_coverage":
        result = _evaluate(resolved_checkpoints=94)
    elif metric == "critical_checkpoint_resolution":
        result = _evaluate(resolved_checkpoints=100, critical_unresolved=1)
    elif metric == "fund_coverage_overall":
        result = _evaluate(covered_by_stratum=(8, 8, 8, 8, 8, 8))
    elif metric.startswith("coverage_"):
        target = metric.removeprefix("coverage_")
        counts = [10] * 6
        counts[[f"{status}_{era}" for status, era in STRATA].index(target)] = 7
        result = _evaluate(covered_by_stratum=tuple(counts))
    else:
        result = _evaluate(invariant_override=(metric, 1))

    assert result.passed is False
    assert result.metric(metric).result == "FAIL"
    other_primary = result.metrics.loc[
        result.metrics.role.eq("PRIMARY") & result.metrics.metric.ne(metric), "result"
    ]
    permitted_linked_failure = (
        {"critical_unresolved"} if metric == "critical_checkpoint_resolution" else set()
    )
    failed_others = set(
        result.metrics.loc[
            result.metrics.role.eq("PRIMARY")
            & result.metrics.metric.ne(metric)
            & result.metrics.result.eq("FAIL"),
            "metric",
        ]
    )
    assert failed_others == permitted_linked_failure


def test_alias_rows_cannot_inflate_unique_family_coverage() -> None:
    checkpoints, dispositions, _, sample, aliases = _gate_frames()
    timeline = pd.DataFrame(
        [
            {
                "checkpoint_id": "one",
                "share_code": aliases.loc[0, "query_code"],
                "effective_from": "2020-01-01",
                "disposition": "PARSED_STATE",
            },
            {
                "checkpoint_id": "two",
                "share_code": aliases.loc[60, "query_code"],
                "effective_from": "2020-02-01",
                "disposition": "PARSED_STATE",
            },
        ]
    )

    result = evaluate_gate(
        checkpoints=checkpoints,
        dispositions=dispositions,
        timeline=timeline,
        family_sample=sample,
        aliases=aliases,
        invariant_counts={name: 0 for name in (
            "causal_violations", "unresolved_conflicts", "nonpositive_intervals",
            "orphan_noops", "critical_unresolved",
        )},
        legacy_document_counts={"success": 1, "valid": 1, "failure": 0},
    )

    assert result.metric("fund_coverage_overall").numerator == 1
    assert {f"coverage_{status}_{era}" for status, era in STRATA}.issubset(
        set(result.metrics.metric)
    )


def test_review_lists_every_metric_and_final_protocol_decision() -> None:
    result = _evaluate()

    for metric in result.metrics.metric:
        assert metric in result.review_markdown
    assert "G0 PASS under V6-G0-A1" in result.review_markdown
    assert "162703/420632" in result.review_markdown


def test_six_artifacts_are_atomic_reloadable_and_do_not_touch_legacy_paths(
    tmp_path: Path,
) -> None:
    result = _evaluate()
    result.checkpoints["share_code"] = "000049"
    legacy_metrics = tmp_path / "g0_gate_metrics_v17.csv"
    legacy_timeline = tmp_path / "classification_timeline.csv"
    legacy_metrics.write_text("legacy metrics\n", encoding="utf-8")
    legacy_timeline.write_text("legacy timeline\n", encoding="utf-8")

    paths = write_gate_artifacts(result, tmp_path)

    assert set(paths) == {
        "required_classification_checkpoints",
        "document_dispositions",
        "classification_clause_clusters",
        "classification_state_timeline",
        "g0_amendment_a1_gate_metrics",
        "g0_amendment_a1_gate_review",
    }
    assert all(path.is_file() for path in paths.values())
    assert not list(tmp_path.glob("*.tmp"))
    for key, path in paths.items():
        if key == "g0_amendment_a1_gate_review":
            assert path.read_text(encoding="utf-8").startswith("# G0 Amendment A1")
        else:
            assert isinstance(pd.read_csv(path), pd.DataFrame)
    restored = pd.read_csv(
        paths["required_classification_checkpoints"], dtype={"share_code": str}
    )
    assert set(restored.share_code) == {"000049"}
    metrics = pd.read_csv(paths["g0_amendment_a1_gate_metrics"])
    assert set(metrics.protocol_version) == {"V6-G0-A1"}
    assert legacy_metrics.read_text(encoding="utf-8") == "legacy metrics\n"
    assert legacy_timeline.read_text(encoding="utf-8") == "legacy timeline\n"


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_lifecycle_evidence_builder_uses_explicit_alias_family_and_exact_provenance() -> None:
    aliases = pd.DataFrame([{
        "query_code": "160049", "sample_share_code": "000049", "family_key": "family-a"
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a"}])
    sections = pd.DataFrame(
        [{
            "share_code": "160049", "upload_info_id": "notice-1",
            "known_at": "2020-01-01", "source_document": "000049_notice-1.pdf",
            "evidence_location": "000049_notice-1.pdf:p4",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
        }]
    )
    parsed = pd.DataFrame(
        [{
            "status": "success", "parser_version": 16,
            "document": {
                "share_code": "160049", "upload_info_id": "notice-1",
                "known_at": "2020-01-01", "source_document": "000049_notice-1.pdf",
                "evidence_location": "000049_notice-1.pdf:p4",
                "evidence": "股票资产占基金资产的比例为60%-95%。",
                "equity_min_pct": 60.0, "equity_max_pct": 95.0,
            },
        }]
    )

    evidence = gate_module._build_lifecycle_evidence(sections, parsed, aliases, sample)

    assert evidence.loc[0, "family_key"] == "family-a"
    assert evidence.loc[0, "parsed_source_document"] == "000049_notice-1.pdf"
    assert evidence.loc[0, "parsed_evidence_location"] == "000049_notice-1.pdf:p4"


def test_lifecycle_evidence_builder_rejects_ambiguous_authoritative_alias() -> None:
    aliases = pd.DataFrame(
        [
            {"query_code": "000049", "sample_share_code": "000049", "family_key": "family-a"},
            {"query_code": "000049", "sample_share_code": "000050", "family_key": "family-b"},
        ]
    )
    sample = pd.DataFrame(
        [
            {"share_code": "000049", "family_key": "family-a"},
            {"share_code": "000050", "family_key": "family-b"},
        ]
    )

    with pytest.raises(ValueError, match="ambiguous.*000049"):
        gate_module._build_lifecycle_evidence(
            pd.DataFrame(), pd.DataFrame(), aliases, sample
        )


def test_lifecycle_evidence_builder_does_not_combine_mismatched_section_and_parser() -> None:
    aliases = pd.DataFrame([{
        "query_code": "000049", "sample_share_code": "000049", "family_key": "family-a"
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a"}])
    sections = pd.DataFrame(
        [{
            "share_code": "000049", "upload_info_id": "notice-1",
            "known_at": "2020-01-01", "source_document": "000049_notice-1.pdf",
            "evidence_location": "000049_notice-1.pdf:p4",
            "section_text": "股票资产占基金资产的比例为60%-95%。",
        }]
    )
    parsed = pd.DataFrame(
        [{
            "status": "success", "parser_version": 16,
            "document": {
                "share_code": "000049", "upload_info_id": "notice-1",
                "known_at": "2020-01-01", "source_document": "other.pdf",
                "evidence_location": "other.pdf:p4",
                "evidence": "股票资产占基金资产的比例为60%-95%。",
                "equity_min_pct": 60.0, "equity_max_pct": 95.0,
            },
        }]
    )

    evidence = gate_module._build_lifecycle_evidence(sections, parsed, aliases, sample)

    assert evidence.empty


def test_lifecycle_evidence_builder_uses_source_as_formal_parser_location_fallback() -> None:
    aliases = pd.DataFrame([{
        "query_code": "160049", "sample_share_code": "000049", "family_key": "family-a"
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a-"}])
    source = "160049_notice-1.pdf"
    raw = "股票资产占基金资产的比例为60%-95%。"
    sections = pd.DataFrame([{
        "share_code": "160049", "upload_info_id": "notice-1", "known_at": "2020-01-01",
        "source_document": source, "evidence_location": source, "section_text": raw,
    }])
    parsed = pd.DataFrame([{
        "status": "success", "parser_version": 16, "source_document": source,
        "document": {
            "share_code": "160049", "upload_info_id": "notice-1", "known_at": "2020-01-01",
            "source_document": source, "evidence": raw,
            "equity_min_pct": 60.0, "equity_max_pct": 95.0,
        },
    }])

    evidence = gate_module._build_lifecycle_evidence(sections, parsed, aliases, sample)

    assert evidence.loc[0, "family_key"] == "family-a-"
    assert evidence.loc[0, "parsed_evidence_location"] == source


def test_lifecycle_evidence_builder_rejects_missing_parser_source() -> None:
    aliases = pd.DataFrame([{
        "query_code": "000049", "sample_share_code": "000049", "family_key": "family-a"
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a"}])
    raw = "股票资产占基金资产的比例为60%-95%。"
    sections = pd.DataFrame([{
        "share_code": "000049", "upload_info_id": "notice-1", "known_at": "2020-01-01",
        "source_document": "000049_notice-1.pdf", "evidence_location": "000049_notice-1.pdf",
        "section_text": raw,
    }])
    parsed = pd.DataFrame([{
        "status": "success", "parser_version": 16,
        "document": {
            "share_code": "000049", "upload_info_id": "notice-1", "known_at": "2020-01-01",
            "source_document": "", "evidence": raw,
            "equity_min_pct": 60.0, "equity_max_pct": 95.0,
        },
    }])

    assert gate_module._build_lifecycle_evidence(sections, parsed, aliases, sample).empty


@pytest.mark.parametrize("sample_share_code", ["", "999999"])
def test_authoritative_alias_mapping_rejects_missing_or_unknown_sample_share_code(
    sample_share_code: str,
) -> None:
    aliases = pd.DataFrame([{
        "query_code": "160049", "sample_share_code": sample_share_code,
        "family_key": "family-a",
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a-"}])

    with pytest.raises(ValueError, match="sample_share_code"):
        gate_module._build_lifecycle_evidence(pd.DataFrame(), pd.DataFrame(), aliases, sample)


def _cli_fixture(tmp_path: Path, *, success: bool) -> dict[str, Path]:
    metadata = tmp_path / "metadata.jsonl"
    parsed = tmp_path / "parsed.jsonl"
    sample_path = tmp_path / "sample.csv"
    aliases_path = tmp_path / "aliases.csv"
    failures = tmp_path / "failures.csv"
    pdf_root = tmp_path / "pdf"
    out = tmp_path / "a1"
    pdf_root.mkdir()
    metadata_rows = []
    parsed_rows = []
    sample_rows = []
    alias_rows = []
    for index, (status, era) in enumerate(STRATA, start=1):
        code = f"{index:06d}"
        upload_id = f"notice-{index}"
        source = f"{code}_{upload_id}.pdf"
        metadata_rows.append({
            "fundCode": code,
            "uploadInfoId": upload_id,
            "reportSendDate": "2020-01-02",
            "effective_date": "2020-01-02",
            "reportCode": "FA010010",
            "reportName": "基金定期文件",
        })
        if success:
            parsed_rows.append({
                "status": "success",
                "parser_version": 16,
                "source_document": source,
                "document": {
                    "share_code": code,
                    "upload_info_id": upload_id,
                    "known_at": "2020-01-02",
                    "effective_date": "2020-01-02",
                    "source_document": source,
                    "evidence": "股票资产占基金资产的比例为60%-95%",
                    "fund_type": "混合型-偏股",
                    "equity_min_pct": 60.0,
                    "equity_max_pct": 95.0,
                },
            })
        else:
            parsed_rows.append({
                "status": "failure",
                "parser_version": 16,
                "source_document": source,
                "failure": {
                    "share_code": code,
                    "upload_info_id": upload_id,
                    "reason": "SECTION_EXTRACTION_FAILED",
                },
            })
        sample_rows.append({
            "share_code": code,
            "family_key": f"family-{index}",
            "status": status,
            "inception_era": era,
        })
        alias_rows.append({
            "query_code": code, "sample_share_code": code, "family_key": f"family-{index}"
        })
    _write_jsonl(metadata, metadata_rows)
    _write_jsonl(parsed, parsed_rows)
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    pd.DataFrame(alias_rows).to_csv(aliases_path, index=False)
    pd.DataFrame(
        [] if success else [{"share_code": "000001", "reason": "failed"}],
        columns=["share_code", "reason"],
    ).to_csv(failures, index=False)
    return {
        "metadata": metadata,
        "parsed": parsed,
        "sample": sample_path,
        "aliases": aliases_path,
        "failures": failures,
        "pdf_root": pdf_root,
        "out": out,
    }


def _cli_argv(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--metadata", str(paths["metadata"]),
        "--parsed-documents", str(paths["parsed"]),
        "--pdf-root", str(paths["pdf_root"]),
        "--family-sample", str(paths["sample"]),
        "--aliases", str(paths["aliases"]),
        "--legacy-failures", str(paths["failures"]),
        "--out", str(paths["out"]),
        *extra,
    ]


def _run_cli(paths: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "v6.g0_checkpoint_gate",
        *_cli_argv(paths, *extra),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


@pytest.mark.parametrize("success", [True, False])
def test_normal_cli_accepts_v2_schema_with_unresolved_lifecycle_gate_failure(
    tmp_path: Path, success: bool
) -> None:
    paths = _cli_fixture(tmp_path, success=success)

    completed = _run_cli(paths)

    assert completed.returncode == 2, completed.stderr
    assert len(list(paths["out"].iterdir())) == 6


def test_freeze_only_never_opens_parser_or_failure_and_writes_only_manifest(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path, success=True)
    sample = pd.read_csv(paths["sample"], dtype={"share_code": str})
    sample["family_key"] = sample["family_key"] + "-"
    sample["inception_date"] = [f"2020-01-{index:02d}" for index in range(1, 7)]
    sample.to_csv(paths["sample"], index=False)
    paths["parsed"].unlink()
    paths["failures"].unlink()

    completed = _run_cli(paths, "--freeze-checkpoints-only")

    assert completed.returncode == 0, completed.stderr
    assert [path.name for path in paths["out"].iterdir()] == [
        "required_classification_checkpoints.csv"
    ]
    restored = pd.read_csv(
        paths["out"] / "required_classification_checkpoints.csv",
        dtype={"share_code": str},
    )
    assert set(restored.family_key) == set(sample.family_key)
    assert restored.family_key.nunique() == 6
    assert set(restored.trigger_type) == {
        "INCEPTION",
        "SECTION_UNCOMPARABLE",
        "TERMINAL_STATE",
    }
    lifecycle = restored.loc[restored.trigger_type.eq("INCEPTION")]
    assert lifecycle.share_code.tolist() == [f"{index:06d}" for index in range(1, 7)]
    assert lifecycle.is_critical.astype(str).str.lower().eq("true").all()
    assert set(restored.generation_rule_version) == {2}
    assert set(restored.normalization_version) == {1}


def test_requested_frozen_manifest_is_immutable_and_mismatch_aborts_before_outputs(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path, success=True)
    frozen_out = tmp_path / "frozen"
    freeze_paths = {**paths, "out": frozen_out}
    frozen_run = _run_cli(freeze_paths, "--freeze-checkpoints-only")
    assert frozen_run.returncode == 0, frozen_run.stderr
    frozen = frozen_out / "required_classification_checkpoints.csv"
    original = pd.read_csv(frozen, dtype=str, keep_default_na=False)

    result = run_checkpoint_gate(
        GateInputs(
            metadata_jsonl=paths["metadata"],
            parsed_documents_jsonl=paths["parsed"],
            pdf_root=paths["pdf_root"],
            family_sample_csv=paths["sample"],
            aliases_csv=paths["aliases"],
            legacy_parse_failures_csv=paths["failures"],
        ),
        frozen_checkpoints=frozen,
    )
    assert result.checkpoints.checkpoint_id.tolist() == original.checkpoint_id.tolist()
    assert result.checkpoints.share_code.tolist() == original.share_code.tolist()
    assert result.checkpoints.is_critical.astype(str).str.lower().eq("true").all()
    before = frozen.read_bytes()
    changed_metadata = [
        json.loads(line)
        for line in paths["metadata"].read_text(encoding="utf-8").splitlines()
    ]
    changed_metadata[0]["uploadInfoId"] = "changed-id"
    _write_jsonl(paths["metadata"], changed_metadata)

    with pytest.raises(ValueError, match="frozen checkpoint manifest mismatch"):
        run_checkpoint_gate(
            GateInputs(
                metadata_jsonl=paths["metadata"],
                parsed_documents_jsonl=paths["parsed"],
                pdf_root=paths["pdf_root"],
                family_sample_csv=paths["sample"],
                aliases_csv=paths["aliases"],
                legacy_parse_failures_csv=paths["failures"],
            ),
            frozen_checkpoints=frozen,
        )
    assert frozen.read_bytes() == before
    assert not paths["out"].exists()


@pytest.mark.parametrize(
    ("history_order", "expected_disposition", "expected_successes"),
    [
        ("failure_then_success", "PARSED_STATE", 6),
        ("success_then_failure", "UNRESOLVED", 5),
    ],
)
def test_append_only_cache_uses_last_source_entry_not_highest_parser_version(
    tmp_path: Path,
    history_order: str,
    expected_disposition: str,
    expected_successes: int,
) -> None:
    paths = _cli_fixture(tmp_path, success=True)
    records = [
        json.loads(line)
        for line in paths["parsed"].read_text(encoding="utf-8").splitlines()
    ]
    current = records[0]
    failure = {
        "status": "failure",
        "parser_version": 99,
        "source_document": current["source_document"],
        "failure": {
            "share_code": "000001",
            "upload_info_id": "notice-1",
            "report_code": "FA010010",
            "reason": "SECTION_EXTRACTION_FAILED",
        },
    }
    if history_order == "failure_then_success":
        records.insert(0, failure)
    else:
        records.append(failure)
    _write_jsonl(paths["parsed"], records)
    for source in {str(record["source_document"]) for record in records}:
        (paths["pdf_root"] / source).touch()

    result = run_checkpoint_gate(
        GateInputs(
            metadata_jsonl=paths["metadata"],
            parsed_documents_jsonl=paths["parsed"],
            pdf_root=paths["pdf_root"],
            family_sample_csv=paths["sample"],
            aliases_csv=paths["aliases"],
            legacy_parse_failures_csv=paths["failures"],
        )
    )

    row = result.dispositions.loc[result.dispositions.upload_info_id.eq("notice-1")].iloc[0]
    assert row.disposition == expected_disposition
    diagnostic = result.metric("document_parser_coverage")
    assert diagnostic.numerator == expected_successes
    assert diagnostic.denominator == 6
    assert diagnostic.failures == 6 - expected_successes


def test_normal_orchestration_reconciles_frozen_1676_of_1993_fa_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    parsed_path = tmp_path / "parsed.jsonl"
    sample_path = tmp_path / "sample.csv"
    aliases_path = tmp_path / "aliases.csv"
    failures_path = tmp_path / "failures.csv"
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    metadata_rows = []
    parsed_rows = []
    failure_rows = []
    for index in range(1993):
        upload_id = f"snapshot-{index:04d}"
        source = f"000001_{upload_id}.pdf"
        metadata_rows.append(
            {
                "fundCode": "000001",
                "uploadInfoId": upload_id,
                "reportSendDate": "2020-01-02",
                "reportCode": "FA010010",
                "reportName": "基金定期文件",
            }
        )
        (pdf_root / source).touch()
        if index < 1676:
            success = {
                "status": "success",
                "parser_version": 16,
                "source_document": source,
                "document": {
                    "share_code": "000001",
                    "upload_info_id": upload_id,
                    "known_at": "2020-01-02",
                    "source_document": source,
                    "report_code": "FA010010",
                    "evidence": "股票资产占基金资产的比例为60%-95%",
                    "fund_type": "混合型-偏股",
                    "equity_min_pct": 60.0,
                    "equity_max_pct": 95.0,
                },
            }
            if index < 5:
                parsed_rows.append(
                    {
                        "status": "failure",
                        "parser_version": 98,
                        "source_document": source,
                        "failure": {
                            "share_code": "000001",
                            "upload_info_id": upload_id,
                            "report_code": "FA010010",
                            "reason": "old failure",
                        },
                    }
                )
            parsed_rows.append(success)
        else:
            failure = {
                "status": "failure",
                "parser_version": 16,
                "source_document": source,
                "failure": {
                    "share_code": "000001",
                    "upload_info_id": upload_id,
                    "report_code": "FA010010",
                    "reason": "SECTION_EXTRACTION_FAILED",
                },
            }
            if index < 1681:
                parsed_rows.append(
                    {
                        "status": "success",
                        "parser_version": 97,
                        "source_document": source,
                        "document": {
                            "share_code": "000001",
                            "upload_info_id": upload_id,
                            "known_at": "2020-01-02",
                            "source_document": source,
                            "report_code": "FA010010",
                            "evidence": "股票资产占基金资产的比例为60%-95%",
                            "fund_type": "混合型-偏股",
                            "equity_min_pct": 60.0,
                            "equity_max_pct": 95.0,
                        },
                    }
                )
            parsed_rows.append(failure)
            failure_rows.append(failure["failure"])
    _write_jsonl(metadata_path, metadata_rows)
    _write_jsonl(parsed_path, parsed_rows)
    pd.DataFrame(
        [
            {
                "share_code": f"{index:06d}",
                "family_key": f"family-{index}",
                "status": status,
                "inception_era": era,
            }
            for index, (status, era) in enumerate(STRATA, start=1)
        ]
    ).to_csv(sample_path, index=False)
    pd.DataFrame([{
        "query_code": "999999", "sample_share_code": "000001", "family_key": "family-1"
    }]).to_csv(
        aliases_path, index=False
    )
    pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
    monkeypatch.setattr(
        gate_module,
        "extract_pdf_text",
        lambda _path: (_ for _ in ()).throw(ValueError("synthetic invalid PDF")),
    )

    result = run_checkpoint_gate(
        GateInputs(
            metadata_jsonl=metadata_path,
            parsed_documents_jsonl=parsed_path,
            pdf_root=pdf_root,
            family_sample_csv=sample_path,
            aliases_csv=aliases_path,
            legacy_parse_failures_csv=failures_path,
        )
    )

    diagnostic = result.metric("document_parser_coverage")
    assert diagnostic.numerator == 1676
    assert diagnostic.denominator == 1993
    assert diagnostic.failures == 317
    assert diagnostic.actual == pytest.approx(1676 / 1993)


def test_normal_cli_preserves_same_output_frozen_manifest(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path, success=True)
    assert gate_module.main(_cli_argv(paths, "--freeze-checkpoints-only")) == 0
    frozen = paths["out"] / "required_classification_checkpoints.csv"
    before_bytes = frozen.read_bytes()
    old_timestamp = 1_000_000_000
    os.utime(frozen, (old_timestamp, old_timestamp))
    before_mtime = frozen.stat().st_mtime_ns

    exit_code = gate_module.main(
        _cli_argv(paths, "--frozen-checkpoints", str(frozen))
    )

    assert exit_code == 2
    assert frozen.read_bytes() == before_bytes
    assert frozen.stat().st_mtime_ns == before_mtime


def test_denominator_section_cache_resumes_success_and_failure_without_reopening_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    metadata = pd.DataFrame(
        [
            {"fundCode": "000001", "uploadInfoId": "ok", "reportSendDate": "2020-01-01"},
            {"fundCode": "000002", "uploadInfoId": "bad", "reportSendDate": "2020-01-02"},
        ]
    )
    for name in ("000001_ok.pdf", "000002_bad.pdf"):
        (pdf_root / name).touch()
    opened: list[str] = []

    def extract(path: Path) -> str:
        opened.append(path.name)
        if path.name == "000002_bad.pdf":
            raise ValueError("broken")
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    cache = tmp_path / "sections.jsonl"

    first = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)
    capsys.readouterr()
    second = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)
    resumed_progress = capsys.readouterr().err

    assert opened == ["000001_ok.pdf", "000002_bad.pdf"]
    assert first.to_dict("records") == second.to_dict("records")
    assert set(second.extraction_status) == {"success", "failure"}
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2
    assert not cache.with_suffix(cache.suffix + ".tmp").exists()
    assert "2/2 cached=2" in resumed_progress


def test_denominator_pipeline_reports_mutually_exclusive_document_section_clause_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    rows = []
    texts = {
        "manager": "关于基金经理变更的公告。",
        "no-section": "本基金法律文件，但正文没有投资章节。",
        "combined": "第八部分 投资范围\n股票、债券等资产合计占基金资产的80%-95%。",
        "parsed": "第八部分 投资范围\n股票资产占基金资产的比例为60%-95%。",
    }
    names = {
        "manager": "关于基金经理变更的公告",
        "no-section": "基金合同更新",
        "combined": "招募说明书更新",
        "parsed": "招募说明书更新",
    }
    for upload_id in texts:
        (pdf_root / f"000001_{upload_id}.pdf").write_bytes(b"pdf")
        rows.append({
            "fundCode": "000001", "uploadInfoId": upload_id,
            "reportSendDate": "2020-01-01", "reportCode": "FC900090",
            "reportName": names[upload_id],
        })
    monkeypatch.setattr(
        gate_module, "extract_pdf_text", lambda path: texts[path.stem.split("_", 1)[1]]
    )

    result = gate_module._extract_denominator_sections(pd.DataFrame(rows), pdf_root)
    by_id = result.set_index("upload_info_id")

    assert by_id.loc["manager", "root_cause_reason"] == "DOCUMENT_NOT_APPLICABLE"
    assert by_id.loc["no-section", "root_cause_reason"] == "INVESTMENT_SECTION_NOT_FOUND"
    assert by_id.loc["combined", "root_cause_reason"] == "EQUITY_CLAUSE_NOT_FOUND"
    assert by_id.loc["parsed", "root_cause_reason"] == ""
    assert by_id.loc["parsed", "section_heading"] == "投资范围"
    assert by_id.loc["parsed", "evidence_location"] == "000001_parsed.pdf"
    assert by_id.loc["parsed", "section_locator"].endswith("#page=1;heading=投资范围")


def test_denominator_pipeline_fails_closed_for_same_section_state_ambiguity_and_damaged_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    for upload_id in ("ambiguous", "damaged"):
        (pdf_root / f"000001_{upload_id}.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([
        {"fundCode": "000001", "uploadInfoId": "ambiguous", "reportSendDate": "2020-01-01",
         "reportName": "招募说明书更新"},
        {"fundCode": "000001", "uploadInfoId": "damaged", "reportSendDate": "2020-01-02",
         "reportName": "招募说明书更新"},
    ])
    texts = {
        "ambiguous": (
            "第八部分 投资范围\n原文第一行 股票资产占基金资产的比例为60%-95%；\n"
            "原文第二行 股票资产占基金资产的比例为0%-30%。"
        ),
        "damaged": "\ufffd" * 20,
    }
    monkeypatch.setattr(
        gate_module, "extract_pdf_text", lambda path: texts[path.stem.split("_", 1)[1]]
    )

    result = gate_module._extract_denominator_sections(metadata, pdf_root).set_index("upload_info_id")

    assert result.loc["ambiguous", "root_cause_reason"] == "EQUITY_CLAUSE_AMBIGUOUS"
    assert result.loc["damaged", "root_cause_reason"] == "TEXT_EXTRACTION_DAMAGED"


def test_denominator_pipeline_preserves_canonical_join_location_and_blocks_pending_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    for upload_id in ("structured", "proposal", "fallback"):
        (pdf_root / f"000001_{upload_id}.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([
        {"fundCode": "000001", "uploadInfoId": "structured", "reportSendDate": "2020-01-01",
         "reportName": "招募说明书更新"},
        {"fundCode": "000001", "uploadInfoId": "proposal", "reportSendDate": "2020-01-02",
         "reportName": "持有人大会拟修改基金合同的公告"},
        {"fundCode": "000001", "uploadInfoId": "fallback", "reportSendDate": "2020-01-03",
         "reportName": "历史法律文件"},
    ])
    texts = {
        "structured": "首页\f八、投资范围\n股票资产占基金资产的比例为60%-95%。",
        "proposal": (
            "八、投资范围\n股票资产占基金资产的比例为60%-95%。"
            "本次拟修改基金合同，尚需持有人大会表决通过后方可生效。"
        ),
        "fallback": "封面\f股票资产占基金资产的比例为0%-30%。",
    }
    monkeypatch.setattr(
        gate_module, "extract_pdf_text", lambda path: texts[path.stem.split("_", 1)[1]]
    )

    result = gate_module._extract_denominator_sections(metadata, pdf_root).set_index("upload_info_id")

    assert result.loc["structured", "evidence_location"] == "000001_structured.pdf"
    assert result.loc["structured", "section_locator"].endswith("#page=2;heading=投资范围")
    assert result.loc["proposal", "root_cause_reason"] == "PROPOSED_STATE_NOT_EFFECTIVE"
    assert result.loc["fallback", "section_page_start"] == "2"
    assert result.loc["fallback", "section_locator"].endswith("#page=2;heading=DOCUMENT_FALLBACK")


def test_structured_section_output_joins_existing_parser_provenance_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000049_notice.pdf").write_bytes(b"pdf")
    raw = "股票资产占基金资产的比例为60%-95%"
    metadata = pd.DataFrame([{
        "fundCode": "000049", "uploadInfoId": "notice", "reportSendDate": "2020-01-01",
        "reportName": "招募说明书更新",
    }])
    monkeypatch.setattr(
        gate_module, "extract_pdf_text", lambda _path: f"八、投资范围\n{raw}。"
    )
    sections = gate_module._extract_denominator_sections(metadata, pdf_root)
    parsed = pd.DataFrame([{
        "status": "success", "parser_version": 16,
        "document": {
            "share_code": "000049", "upload_info_id": "notice", "known_at": "2020-01-01",
            "source_document": "000049_notice.pdf", "evidence": raw,
            "equity_min_pct": 60.0, "equity_max_pct": 95.0,
        },
    }])
    aliases = pd.DataFrame([{
        "query_code": "000049", "sample_share_code": "000049", "family_key": "family-a"
    }])
    sample = pd.DataFrame([{"share_code": "000049", "family_key": "family-a"}])

    evidence = gate_module._build_lifecycle_evidence(sections, parsed, aliases, sample)

    assert len(evidence) == 1
    assert evidence.loc[0, "evidence_location"] == "000049_notice.pdf"
    assert evidence.loc[0, "section_locator"].endswith("#page=1;heading=投资范围")


def test_rejected_result_notice_cannot_emit_comparable_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_rejected.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([{
        "fundCode": "000001", "uploadInfoId": "rejected", "reportSendDate": "2025-01-01",
        "reportName": "关于基金份额持有人大会表决结果公告",
    }])
    monkeypatch.setattr(
        gate_module,
        "extract_pdf_text",
        lambda _path: (
            "八、投资范围\n股票资产占基金资产的比例为60%-95%。\n"
            "本议案未经持有人大会表决通过。"
        ),
    )

    row = gate_module._extract_denominator_sections(metadata, pdf_root).iloc[0]

    assert row.extraction_status == "failure"
    assert not bool(row.section_comparable)
    assert row.root_cause_reason == "PROPOSED_STATE_NOT_EFFECTIVE"


def test_section_cache_invalidates_on_metadata_and_pdf_content_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    pdf = pdf_root / "000001_notice.pdf"
    pdf.write_bytes(b"first")
    metadata = pd.DataFrame([{
        "fundCode": "000001",
        "uploadInfoId": "notice",
        "reportSendDate": "2020-01-01",
        "effective_date": "2020-01-01",
    }])
    opened: list[bytes] = []

    def extract(path: Path) -> str:
        opened.append(path.read_bytes())
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    cache = tmp_path / "sections.jsonl"
    gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)

    changed_metadata = metadata.copy()
    changed_metadata.loc[0, "reportSendDate"] = "2020-02-01"
    gate_module._extract_denominator_sections(changed_metadata, pdf_root, cache_path=cache)
    old_stat = pdf.stat()
    pdf.write_bytes(b"other")
    os.utime(pdf, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    gate_module._extract_denominator_sections(changed_metadata, pdf_root, cache_path=cache)

    assert opened == [b"first", b"first", b"other"]
    cached = json.loads(cache.read_text(encoding="utf-8").strip())
    assert cached["cache_schema_version"]
    assert cached["extractor_version"]
    assert cached["metadata_fingerprint"]
    assert cached["pdf_fingerprint"]["sha256"]


def test_timestamp_metadata_uses_the_same_stable_json_values_for_fingerprint_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A date-like metadata value must not crash cache writes or miss its own cache."""

    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_notice.pdf").write_bytes(b"pdf")
    timestamp_metadata = pd.DataFrame([{
        "fundCode": "000001",
        "uploadInfoId": "notice",
        "reportSendDate": pd.Timestamp("2020-01-02 16:30:00"),
        "effective_date": pd.Timestamp("2020-01-03 08:00:00"),
        "reportCode": "FA010010",
        "reportName": "招募说明书",
    }])
    equivalent_metadata = pd.DataFrame([{
        "fundCode": "000001",
        "uploadInfoId": "notice",
        "reportSendDate": date(2020, 1, 2),
        "effective_date": datetime(2020, 1, 3, 8, 0),
        "reportCode": "FA010010",
        "reportName": "招募说明书",
    }])
    opened: list[str] = []

    def extract(path: Path) -> str:
        opened.append(path.name)
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    monkeypatch.setattr(
        gate_module, "extract_effective_date", lambda _text: pd.Timestamp("2020-01-04 12:00:00")
    )
    cache = tmp_path / "sections.jsonl"

    first = gate_module._extract_denominator_sections(
        timestamp_metadata, pdf_root, cache_path=cache
    )
    second = gate_module._extract_denominator_sections(
        equivalent_metadata, pdf_root, cache_path=cache
    )

    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert opened == ["000001_notice.pdf"]
    assert first.iloc[0].known_at == "2020-01-02"
    assert first.iloc[0].effective_date == "2020-01-04"
    assert second.to_dict("records") == first.to_dict("records")
    assert cached["metadata_fingerprint"] == gate_module._fingerprint_metadata(
        equivalent_metadata.iloc[0].to_dict(), "000001", "notice"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.Timestamp("2020-01-02 16:30:00"), "2020-01-02"),
        (datetime(2020, 1, 2, 16, 30), "2020-01-02"),
        (date(2020, 1, 2), "2020-01-02"),
        (pd.NaT, None),
        (pd.NA, None),
        (np.float64("nan"), None),
        (np.int64(7), 7),
        (np.float64(1.25), 1.25),
        (np.bool_(True), True),
        ("plain", "plain"),
        (3, 3),
        (2.5, 2.5),
        (False, False),
        (None, None),
    ],
)
def test_stable_json_scalar_accepts_only_supported_metadata_values(
    value: object, expected: object
) -> None:
    """The JSON boundary rejects unknown values instead of stringifying them implicitly."""

    assert gate_module._stable_json_scalar(value) == expected


def test_stable_json_scalar_fails_closed_for_unknown_metadata_value() -> None:
    with pytest.raises(TypeError, match="unsupported JSON scalar"):
        gate_module._stable_json_scalar(object())


def test_stable_json_scalar_normalizes_numpy_datetime_units_without_epoch_conversion() -> None:
    """NumPy day and nanosecond dates represent calendar days, never epoch integers."""

    day = np.datetime64("2020-01-02", "D")
    nanosecond = np.datetime64("2020-01-02T23:59:59", "ns")

    assert gate_module._stable_json_scalar(day) == "2020-01-02"
    assert gate_module._stable_json_scalar(nanosecond) == "2020-01-02"
    assert gate_module._stable_json_scalar(np.datetime64("NaT")) is None


@pytest.mark.parametrize("value", [np.longdouble("1.25"), np.clongdouble("1.25+0j")])
def test_stable_json_scalar_fails_closed_when_numpy_item_remains_numpy_scalar(
    value: object,
) -> None:
    """Long precision/complex NumPy scalars must not recurse forever or stringify."""

    with pytest.raises(TypeError, match="unsupported JSON scalar"):
        gate_module._stable_json_scalar(value)


def test_stable_json_scalar_preserves_timezone_aware_input_calendar_date() -> None:
    """Timezone-aware values retain their expressed local day instead of converting to UTC."""

    offset = timezone(timedelta(hours=14))
    values = [
        pd.Timestamp("2020-01-02 00:30:00+14:00"),
        datetime(2020, 1, 2, 0, 30, tzinfo=offset),
    ]

    assert [gate_module._stable_json_scalar(value) for value in values] == [
        "2020-01-02",
        "2020-01-02",
    ]


def test_failed_missing_pdf_cache_invalidates_when_file_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    metadata = pd.DataFrame([{
        "fundCode": "000001", "uploadInfoId": "notice", "reportSendDate": "2020-01-01"
    }])
    opened: list[bool] = []

    def extract(path: Path) -> str:
        opened.append(path.is_file())
        if not path.is_file():
            raise FileNotFoundError(path)
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    cache = tmp_path / "sections.jsonl"
    first = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)
    (pdf_root / "000001_notice.pdf").write_bytes(b"now-present")
    second = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)

    assert opened == [False, True]
    assert first.iloc[0].extraction_status == "failure"
    assert second.iloc[0].extraction_status == "success"


def test_fully_cached_run_does_not_start_pdf_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_notice.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([{
        "fundCode": "000001", "uploadInfoId": "notice", "reportSendDate": "2020-01-01"
    }])
    monkeypatch.setattr(
        gate_module, "extract_pdf_text", lambda _path: "股票资产占基金资产的比例为60%-95%"
    )
    cache = tmp_path / "sections.jsonl"
    expected = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)

    class UnexpectedWorker:
        def __init__(self, _function: object):
            raise AssertionError("worker started without a cache miss")

    monkeypatch.setattr(gate_module, "_IsolatedWorker", UnexpectedWorker)
    actual = gate_module._extract_denominator_sections(
        metadata, pdf_root, cache_path=cache, timeout_seconds=1.0
    )

    assert actual.to_dict("records") == expected.to_dict("records")


def test_section_cache_with_old_extractor_version_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_notice.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([{
        "fundCode": "000001", "uploadInfoId": "notice", "reportSendDate": "2020-01-01"
    }])
    opened: list[str] = []

    def extract(path: Path) -> str:
        opened.append(path.name)
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    cache = tmp_path / "sections.jsonl"
    gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)
    cached = json.loads(cache.read_text(encoding="utf-8").strip())
    cached["extractor_version"] = "obsolete"
    cache.write_text(json.dumps(cached, ensure_ascii=False) + "\n", encoding="utf-8")

    gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)

    assert opened == ["000001_notice.pdf", "000001_notice.pdf"]


def test_nan_primary_metadata_uses_aliases_and_alias_change_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    pdf = pdf_root / "000001_notice.pdf"
    pdf.write_bytes(b"pdf")
    metadata = pd.DataFrame([{
        "fundCode": float("nan"),
        "share_code": "000001",
        "uploadInfoId": float("nan"),
        "upload_info_id": "notice",
        "reportSendDate": float("nan"),
        "known_at": "2020-01-01",
        "effective_date": float("nan"),
        "effectiveDate": "2020-01-02",
        "reportCode": float("nan"),
        "report_code": "FA010010",
        "reportName": float("nan"),
        "report_name": "招募说明书",
    }])
    opened: list[str] = []

    def extract(path: Path) -> str:
        opened.append(path.name)
        return "股票资产占基金资产的比例为60%-95%"

    monkeypatch.setattr(gate_module, "extract_pdf_text", extract)
    cache = tmp_path / "sections.jsonl"
    first = gate_module._extract_denominator_sections(metadata, pdf_root, cache_path=cache)
    changed = metadata.copy()
    changed.loc[0, "known_at"] = "2020-02-01"
    second = gate_module._extract_denominator_sections(changed, pdf_root, cache_path=cache)

    assert opened == ["000001_notice.pdf", "000001_notice.pdf"]
    assert first.iloc[0].share_code == "000001"
    assert first.iloc[0].upload_info_id == "notice"
    assert first.iloc[0].known_at == "2020-01-01"
    assert first.iloc[0].effective_date == "2020-01-02"
    assert second.iloc[0].known_at == "2020-02-01"


def test_nan_primary_metadata_aliases_are_used_by_pdf_preflight_counts(
    tmp_path: Path,
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "000001_notice.pdf").write_bytes(b"pdf")
    metadata = pd.DataFrame([{
        "fundCode": float("nan"),
        "share_code": "000001",
        "uploadInfoId": float("nan"),
        "upload_info_id": "notice",
        "reportCode": float("nan"),
        "report_code": "FA010010",
    }])

    assert gate_module._pdf_counts(metadata, pdf_root) == {
        "available": 1, "required": 1, "missing": 0
    }
    assert gate_module._legacy_counts(pd.DataFrame(), metadata, pdf_root) == {
        "success": 0, "failure": 1, "valid": 1
    }


def test_freeze_only_validates_aliases_before_opening_any_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _cli_fixture(tmp_path, success=True)
    aliases = pd.read_csv(paths["aliases"], dtype=str)
    aliases.loc[len(aliases)] = {"query_code": "000001", "family_key": "family-2"}
    aliases.to_csv(paths["aliases"], index=False)

    def unexpected_extract(_path: Path) -> str:
        raise AssertionError("PDF extraction started before fast validation")

    monkeypatch.setattr(gate_module, "extract_pdf_text", unexpected_extract)

    with pytest.raises(ValueError, match="maps to multiple sample families"):
        gate_module._freeze_only(
            GateInputs(
                metadata_jsonl=paths["metadata"],
                parsed_documents_jsonl=None,
                pdf_root=paths["pdf_root"],
                family_sample_csv=paths["sample"],
                aliases_csv=paths["aliases"],
                legacy_parse_failures_csv=None,
            ),
            paths["out"],
        )


def test_isolated_call_terminates_work_that_exceeds_timeout() -> None:
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="timed out"):
        gate_module._run_isolated(time.sleep, (2.0,), timeout_seconds=0.05)

    assert time.monotonic() - started < 1.0


def test_isolated_worker_reuses_one_process_for_multiple_documents() -> None:
    with gate_module._IsolatedWorker(os.getpid) as worker:
        first_pid = worker.call((), timeout_seconds=5.0)
        second_pid = worker.call((), timeout_seconds=5.0)

    assert first_pid == second_pid
    assert first_pid != os.getpid()


def test_isolated_worker_restarts_after_one_document_times_out() -> None:
    with gate_module._IsolatedWorker(_sleep_then_pid) as worker:
        with pytest.raises(TimeoutError):
            worker.call((2.0,), timeout_seconds=0.05)
        assert worker.call((0.0,), timeout_seconds=5.0) != os.getpid()


def test_isolated_worker_restarts_after_child_process_crashes() -> None:
    with gate_module._IsolatedWorker(_crash_or_pid) as worker:
        with pytest.raises(RuntimeError, match="worker.*terminated"):
            worker.call(("crash",), timeout_seconds=5.0)
        assert worker.call(("ok",), timeout_seconds=5.0) != os.getpid()


@pytest.mark.parametrize(
    ("share_code", "upload_id"),
    [
        ("../000001", "notice"),
        ("000001", "../notice"),
        ("000001", "..\\notice"),
        ("C:\\000001", "notice"),
        ("000001", "/absolute"),
    ],
)
def test_pdf_preflight_rejects_unsafe_document_tokens(
    tmp_path: Path, share_code: str, upload_id: str
) -> None:
    metadata = pd.DataFrame([{"fundCode": share_code, "uploadInfoId": upload_id}])

    with pytest.raises(ValueError, match="unsafe metadata document token"):
        gate_module._pdf_counts(metadata, tmp_path)


def test_process_cleanup_kills_child_that_survives_terminate() -> None:
    events: list[str] = []

    class StubbornProcess:
        def is_alive(self) -> bool:
            return "kill" not in events

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

        def join(self, timeout: float | None = None) -> None:
            events.append(f"join:{timeout}")

    gate_module._stop_process(StubbornProcess())

    assert events == ["terminate", "join:1.0", "kill", "join:1.0"]
