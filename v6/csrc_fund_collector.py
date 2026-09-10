from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
from pypdf import PdfReader

from v6.csrc_fund_disclosure import (
    EquityConstraint,
    classify_mixed_fund,
    extract_effective_date,
    extract_equity_constraint,
    extract_pdf_text,
)


DEFAULT_REPORT_TYPES = (
    "FA010010",  # 初始招募说明书
    "FA010030",  # 招募说明书更新
    "FA010070",  # 基金产品资料概要
    "FA010080",  # 基金产品资料概要更新
    "FA020010",  # 基金合同
    "FA020020",  # 基金合同更新
    "FC050010",  # 转换基金运作方式
    "FC050020",  # 转换基金运作方式其他公告
    "FC220010",  # 修改基金合同
    "FC220030",  # 修改招募说明书
    "FC900090",  # 历史临时公告
)
PARSER_VERSION = 16
SUCCESS_CACHE_COMPATIBILITY_FLOOR = 15


class MetadataClient(Protocol):
    def fetch_page(
        self, fund_code: str, report_type: str, start: int, page_size: int
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PdfValidation:
    path: Path
    size_bytes: int
    page_count: int


def _datatables_params(
    fund_code: str, report_type: str, start: int, page_size: int
) -> list[dict[str, Any]]:
    values: list[tuple[str, Any]] = [
        ("sEcho", 1),
        ("iColumns", 6),
        ("sColumns", ""),
        ("iDisplayStart", start),
        ("iDisplayLength", page_size),
    ]
    values.extend((f"mDataProp_{i}", name) for i, name in enumerate(
        ("fundCode", "fundId", "reportName", "organName", "reportDesp", "reportSendDate")
    ))
    values.extend(
        [
            ("fundType", ""),
            ("reportType", report_type),
            ("reportYear", ""),
            ("fundCompanyShortName", ""),
            ("fundCode", str(fund_code).zfill(6)),
            ("fundShortName", ""),
            ("startUploadDate", "1997-01-01"),
            ("endUploadDate", pd.Timestamp.now().strftime("%Y-%m-%d")),
        ]
    )
    return [{"name": name, "value": value} for name, value in values]


class CsrcDisclosureClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        base_url: str = "http://eid.csrc.gov.cn",
        timeout: float = 30.0,
    ):
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "User-Agent": "quant-fund-picker-v6/1.0 (public disclosure research)"
        }

    def fetch_page(
        self, fund_code: str, report_type: str, start: int, page_size: int
    ) -> dict[str, Any]:
        ao_data = _datatables_params(fund_code, report_type, start, page_size)
        response = self.session.get(
            self.base_url + "/fund/disclose/advanced_search_report.do",
            params={"aoData": json.dumps(ao_data, ensure_ascii=False, separators=(",", ":"))},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "aaData" not in payload:
            raise ValueError("Unexpected CSRC metadata response")
        return payload

    def download_pdf(
        self, upload_info_id: str, destination: str | os.PathLike[str]
    ) -> PdfValidation:
        destination_path = Path(destination)
        if destination_path.exists():
            return validate_pdf(destination_path)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        partial = destination_path.with_suffix(destination_path.suffix + ".part")
        try:
            response = self.session.get(
                self.base_url + "/fund/disclose/instance_show_pdf_id.do",
                params={"instanceid": str(upload_info_id)},
                headers=self.headers,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        stream.write(chunk)
            result = validate_pdf(partial)
            partial.replace(destination_path)
            return PdfValidation(
                destination_path, result.size_bytes, result.page_count
            )
        except Exception:
            partial.unlink(missing_ok=True)
            raise


class CheckpointStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._state = self._load()

    @staticmethod
    def _key(fund_code: str, report_type: str) -> str:
        return f"{fund_code}|{report_type}"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "jobs": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _job(self, fund_code: str, report_type: str) -> dict[str, Any]:
        key = self._key(fund_code, report_type)
        return self._state.setdefault("jobs", {}).setdefault(
            key, {"next_start": 0, "complete": False}
        )

    def next_start(self, fund_code: str, report_type: str) -> int:
        return int(self._job(fund_code, report_type)["next_start"])

    def is_complete(self, fund_code: str, report_type: str) -> bool:
        return bool(self._job(fund_code, report_type)["complete"])

    def save_progress(
        self,
        fund_code: str,
        report_type: str,
        next_start: int,
        *,
        complete: bool,
    ) -> None:
        job = self._job(fund_code, report_type)
        job.update(next_start=int(next_start), complete=bool(complete))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary.replace(self.path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (2**attempt))


def _record_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("uploadInfoId") or ""),
        str(row.get("uploadInfoDetailId") or ""),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect_report_metadata(
    client: MetadataClient,
    fund_code: str,
    report_type: str,
    output_path: str | os.PathLike[str],
    checkpoint: CheckpointStore,
    *,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    output = Path(output_path)
    existing = _read_jsonl(output)
    seen = {_record_key(row) for row in existing}
    if checkpoint.is_complete(fund_code, report_type):
        return existing

    start = checkpoint.next_start(fund_code, report_type)
    while True:
        payload = client.fetch_page(fund_code, report_type, start, page_size)
        rows = list(payload.get("aaData") or [])
        total = int(payload.get("iTotalRecords") or payload.get("iTotalDisplayRecords") or 0)
        if rows:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    key = _record_key(row)
                    if key in seen:
                        continue
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    seen.add(key)
                    existing.append(row)

        next_start = start + len(rows)
        complete = not rows or next_start >= total
        checkpoint.save_progress(
            fund_code, report_type, next_start, complete=complete
        )
        if complete:
            break
        if next_start <= start:
            raise RuntimeError("CSRC pagination made no progress")
        start = next_start

    return existing


def validate_pdf(path: str | os.PathLike[str]) -> PdfValidation:
    pdf_path = Path(path)
    with pdf_path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError(f"PDF signature missing: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
    except Exception as exc:
        raise ValueError(f"Unreadable PDF: {pdf_path}") from exc
    if page_count < 1:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    return PdfValidation(pdf_path, pdf_path.stat().st_size, page_count)


def build_classification_timeline(documents: pd.DataFrame) -> pd.DataFrame:
    required = {
        "share_code",
        "known_at",
        "effective_date",
        "equity_min_pct",
        "equity_max_pct",
        "evidence",
        "source_document",
    }
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(f"Missing timeline columns: {sorted(missing)}")

    timeline = documents.copy()
    timeline["share_code"] = timeline["share_code"].astype(str).str.zfill(6)
    timeline["known_at"] = pd.to_datetime(timeline["known_at"], errors="raise")
    timeline["effective_date"] = pd.to_datetime(
        timeline["effective_date"], errors="coerce"
    )
    timeline["effective_from"] = timeline[["effective_date", "known_at"]].max(axis=1)
    if "inception_date" in timeline:
        timeline["inception_date"] = pd.to_datetime(
            timeline["inception_date"], errors="coerce"
        )
        invalid = (
            timeline["effective_from"] < timeline["inception_date"]
        ) | (
            timeline["effective_date"].notna()
            & (timeline["effective_date"] < timeline["inception_date"])
        )
        if invalid.any():
            codes = timeline.loc[invalid, "share_code"].tolist()
            raise ValueError(f"Classification effective date before inception: {codes}")

    def classify(row: pd.Series) -> str | None:
        override = row.get("fund_type_override")
        if pd.notna(override) and str(override):
            return str(override)
        constraint = EquityConstraint(
            float(row["equity_min_pct"]),
            float(row["equity_max_pct"]),
            str(row["evidence"]),
        )
        return classify_mixed_fund(constraint)

    timeline["fund_type"] = timeline.apply(classify, axis=1)
    timeline = timeline.sort_values(
        ["share_code", "effective_from", "known_at", "source_document"],
        kind="stable",
    ).reset_index(drop=True)
    state_columns = ["fund_type", "equity_min_pct", "equity_max_pct"]
    resolved_indices: list[int] = []
    for key, group in timeline.groupby(["share_code", "effective_from"], dropna=False):
        if len(group[state_columns].drop_duplicates()) == 1:
            resolved_indices.extend(group.index.tolist())
            continue
        tight_min = group["equity_min_pct"].max()
        tight_max = group["equity_max_pct"].min()
        tightest = group[
            group["equity_min_pct"].eq(tight_min)
            & group["equity_max_pct"].eq(tight_max)
        ]
        if tight_min > tight_max or tightest.empty:
            raise ValueError(f"Conflicting classifications at {key}")
        resolved_indices.extend(tightest.index.tolist())
    timeline = timeline.loc[resolved_indices].reset_index(drop=True)
    timeline = timeline.drop_duplicates(
        ["share_code", "effective_from", *state_columns], keep="last"
    ).reset_index(drop=True)
    timeline["effective_to"] = timeline.groupby("share_code")[
        "effective_from"
    ].shift(-1)
    preferred = [
        "share_code", "known_at", "effective_date", "effective_from", "effective_to",
        "fund_type", "equity_min_pct", "equity_max_pct", "evidence",
        "source_document", "report_code", "report_name", "upload_info_id",
        "inception_date", "fund_type_override",
    ]
    ordered = [column for column in preferred if column in timeline.columns]
    ordered.extend(column for column in timeline.columns if column not in ordered)
    return timeline[ordered]


def build_timeline_from_downloads(
    metadata: list[dict[str, Any]] | pd.DataFrame,
    pdf_root: str | os.PathLike[str],
    *,
    inception_dates: dict[str, Any] | None = None,
    parse_cache_path: str | os.PathLike[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = metadata.to_dict("records") if isinstance(metadata, pd.DataFrame) else metadata
    root = Path(pdf_root)
    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cache_path = Path(parse_cache_path) if parse_cache_path is not None else None
    cached_entries = _read_jsonl(cache_path) if cache_path is not None else []

    def cache_is_compatible(entry: dict[str, Any]) -> bool:
        version = entry.get("parser_version")
        if version == PARSER_VERSION:
            return True
        if (
            entry.get("status") == "success"
            and isinstance(version, int)
            and SUCCESS_CACHE_COMPATIBILITY_FLOOR <= version < PARSER_VERSION
        ):
            return True
        payload = entry.get("document") or entry.get("failure") or {}
        return version in {4, 5, 6, 7, 8, 9, 10, 11, 12} and str(payload.get("report_code") or "").startswith("FC")

    cache = {
        str(entry.get("source_document")): entry
        for entry in cached_entries
        if cache_is_compatible(entry)
    }

    def cache_entry(entry: dict[str, Any]) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()

    for record in records:
        share_code = str(record.get("fundCode") or record.get("share_code") or "").zfill(6)
        upload_id = str(record.get("uploadInfoId") or record.get("upload_info_id") or "")
        path = root / f"{share_code}_{upload_id}.pdf"
        cached = cache.get(path.name)
        if cached is not None:
            if cached["status"] == "success":
                documents.append(cached["document"])
            else:
                failures.append(cached["failure"])
            continue
        report_name = str(record.get("reportName") or "")
        phase_hint = "closed" if re.search(r"(?:定期开放|\d+年封闭)", report_name) else None
        try:
            text = extract_pdf_text(path)
            constraint = extract_equity_constraint(text, phase_hint=phase_hint)
        except Exception as exc:
            failure = {
                "share_code": share_code,
                "upload_info_id": upload_id,
                "report_code": str(record.get("reportCode") or ""),
                "report_name": str(record.get("reportName") or ""),
                "reason": str(exc),
            }
            failures.append(failure)
            if path.exists():
                cache_entry(
                    {
                        "parser_version": PARSER_VERSION,
                        "source_document": path.name,
                        "status": "failure",
                        "failure": failure,
                    }
                )
            continue
        known_at = record.get("reportSendDate") or record.get("known_at")
        effective_date = extract_effective_date(text)
        report_code = str(record.get("reportCode") or "")
        if report_code.startswith("FC") and effective_date is None:
            failure = {
                "share_code": share_code,
                "upload_info_id": upload_id,
                "report_code": report_code,
                "report_name": str(record.get("reportName") or ""),
                "reason": "Event notice has no explicit effective date",
            }
            failures.append(failure)
            cache_entry(
                {
                    "parser_version": PARSER_VERSION,
                    "source_document": path.name,
                    "status": "failure",
                    "failure": failure,
                }
            )
            continue
        row: dict[str, Any] = {
            "share_code": share_code,
            "known_at": known_at,
            "effective_date": (
                effective_date.strftime("%Y-%m-%d")
                if effective_date is not None
                else None
            ),
            "equity_min_pct": constraint.equity_min_pct,
            "equity_max_pct": constraint.equity_max_pct,
            "evidence": constraint.evidence,
            "source_document": path.name,
            "report_code": record.get("reportCode"),
            "report_name": record.get("reportName"),
            "upload_info_id": upload_id,
        }
        is_official_index = bool(
            report_code.startswith("FA")
            and re.search(r"指数(?:证券投资)?基金", report_name)
        )
        if is_official_index and constraint.equity_min_pct >= 60:
            row["fund_type_override"] = "指数型-股票"
        if inception_dates is not None:
            row["inception_date"] = inception_dates.get(share_code)
        documents.append(row)
        cache_entry(
            {
                "parser_version": PARSER_VERSION,
                "source_document": path.name,
                "status": "success",
                "document": row,
            }
        )

    if documents:
        timeline_parts = []
        document_frame = pd.DataFrame(documents)
        for share_code, group in document_frame.groupby("share_code", sort=True):
            try:
                timeline_parts.append(build_classification_timeline(group))
            except ValueError as exc:
                if not str(exc).startswith("Conflicting classifications at "):
                    raise
                failures.append(
                    {
                        "share_code": str(share_code),
                        "upload_info_id": ",".join(
                            group["upload_info_id"].astype(str).tolist()
                        ),
                        "report_code": "TIMELINE_CONFLICT",
                        "report_name": "",
                        "reason": str(exc),
                    }
                )
        timeline = (
            pd.concat(timeline_parts, ignore_index=True)
            if timeline_parts
            else pd.DataFrame()
        )
    else:
        timeline = pd.DataFrame(
            columns=[
                "share_code", "known_at", "effective_date", "equity_min_pct",
                "equity_max_pct", "evidence", "source_document", "report_code",
                "report_name", "upload_info_id", "effective_from", "fund_type",
                "effective_to",
            ]
        )
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "share_code", "upload_info_id", "report_code", "report_name", "reason"
        ],
    )
    return timeline, failure_frame


def load_fund_codes(path: str | os.PathLike[str]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        code = value.split(".", 1)[0].zfill(6)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def download_metadata_pdfs(
    client: CsrcDisclosureClient,
    metadata: list[dict[str, Any]],
    pdf_root: str | os.PathLike[str],
    *,
    delay_seconds: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(pdf_root)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in metadata:
        share_code = str(record.get("fundCode") or "").zfill(6)
        upload_id = str(record.get("uploadInfoId") or "")
        key = (share_code, upload_id)
        if not share_code or not upload_id or key in seen:
            continue
        seen.add(key)
        destination = root / f"{share_code}_{upload_id}.pdf"
        try:
            validation = client.download_pdf(upload_id, destination)
            successes.append(
                {
                    "share_code": share_code,
                    "upload_info_id": upload_id,
                    "path": str(validation.path),
                    "size_bytes": validation.size_bytes,
                    "page_count": validation.page_count,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "share_code": share_code,
                    "upload_info_id": upload_id,
                    "reason": str(exc),
                }
            )
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return pd.DataFrame(successes), pd.DataFrame(
        failures, columns=["share_code", "upload_info_id", "reason"]
    )


def collect_fund_disclosures(
    codes: list[str],
    report_types: list[str] | tuple[str, ...],
    out_root: str | os.PathLike[str],
    *,
    client: CsrcDisclosureClient | None = None,
    page_size: int = 20,
    delay_seconds: float = 0.2,
    download_pdfs: bool = True,
    parse_only: bool = False,
    checkpoint_gate: bool = False,
    family_sample_path: str | os.PathLike[str] | None = None,
    aliases_path: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    if checkpoint_gate:
        if family_sample_path is None:
            raise ValueError("checkpoint_gate requires family_sample_path")
        if aliases_path is None:
            raise ValueError("checkpoint_gate requires aliases_path")
        if not Path(family_sample_path).is_file():
            raise ValueError(
                f"checkpoint_gate family_sample_path not found: {family_sample_path}"
            )
        if not Path(aliases_path).is_file():
            raise ValueError(f"checkpoint_gate aliases_path not found: {aliases_path}")

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    checkpoint = CheckpointStore(root / "checkpoint.json")
    disclosure_client = client or CsrcDisclosureClient()

    if not parse_only:
        for code in codes:
            for report_type in report_types:
                collect_report_metadata(
                    disclosure_client,
                    str(code).zfill(6),
                    report_type,
                    metadata_path,
                    checkpoint,
                    page_size=page_size,
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    elif not metadata_path.exists():
        raise FileNotFoundError(f"parse-only metadata not found: {metadata_path}")

    metadata = _read_jsonl(metadata_path)
    metadata_csv = root / "metadata.csv"
    pd.DataFrame(metadata).to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    artifacts = {
        "metadata_jsonl": metadata_path,
        "metadata_csv": metadata_csv,
        "checkpoint": checkpoint.path,
    }
    if not download_pdfs:
        if checkpoint_gate:
            raise ValueError("checkpoint_gate requires legacy parse outputs")
        return artifacts

    pdf_root = root / "pdf"
    download_index_path = root / "pdf_index.csv"
    download_failures_path = root / "pdf_download_failures.csv"
    if not parse_only:
        download_index, download_failures = download_metadata_pdfs(
            disclosure_client,
            metadata,
            pdf_root,
            delay_seconds=delay_seconds,
        )
        download_index.to_csv(download_index_path, index=False, encoding="utf-8-sig")
        download_failures.to_csv(
            download_failures_path, index=False, encoding="utf-8-sig"
        )

    parse_cache_path = root / "parsed_documents.jsonl"
    timeline, parse_failures = build_timeline_from_downloads(
        metadata, pdf_root, parse_cache_path=parse_cache_path
    )
    timeline_path = root / "classification_timeline.csv"
    parse_failures_path = root / "classification_parse_failures.csv"
    timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")
    parse_failures.to_csv(parse_failures_path, index=False, encoding="utf-8-sig")
    artifacts.update(
        pdf_root=pdf_root,
        pdf_index=download_index_path,
        pdf_download_failures=download_failures_path,
        classification_timeline=timeline_path,
        classification_parse_failures=parse_failures_path,
        parsed_documents=parse_cache_path,
    )
    if checkpoint_gate:
        from v6.g0_checkpoint_gate import (
            GateInputs,
            run_checkpoint_gate,
            write_gate_artifacts,
        )

        a1_root = root / "a1"
        frozen_checkpoints = a1_root / "required_classification_checkpoints.csv"
        frozen = frozen_checkpoints if frozen_checkpoints.is_file() else None
        result = run_checkpoint_gate(
            GateInputs(
                metadata_jsonl=metadata_path,
                parsed_documents_jsonl=parse_cache_path,
                pdf_root=pdf_root,
                family_sample_csv=Path(family_sample_path),
                aliases_csv=Path(aliases_path),
                legacy_parse_failures_csv=parse_failures_path,
            ),
            frozen_checkpoints=frozen,
        )
        a1_artifacts = write_gate_artifacts(
            result, a1_root, frozen_checkpoints=frozen
        )
        artifacts.update(
            a1_required_classification_checkpoints=a1_artifacts[
                "required_classification_checkpoints"
            ],
            a1_document_dispositions=a1_artifacts["document_dispositions"],
            a1_classification_clause_clusters=a1_artifacts[
                "classification_clause_clusters"
            ],
            a1_classification_state_timeline=a1_artifacts[
                "classification_state_timeline"
            ],
            a1_gate_metrics=a1_artifacts["g0_amendment_a1_gate_metrics"],
            a1_gate_review=a1_artifacts["g0_amendment_a1_gate_review"],
        )
    return artifacts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect resumable CSRC fund disclosures and build PIT classifications."
    )
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--report-types", nargs="+", default=list(DEFAULT_REPORT_TYPES)
    )
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--max-codes", type=int)
    parser.add_argument("--checkpoint-gate", action="store_true")
    parser.add_argument("--family-sample")
    parser.add_argument("--aliases")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--parse-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    codes = load_fund_codes(args.codes_file)
    if args.max_codes is not None:
        codes = codes[: args.max_codes]
    artifacts = collect_fund_disclosures(
        codes,
        args.report_types,
        args.out,
        page_size=args.page_size,
        delay_seconds=args.delay_seconds,
        download_pdfs=not args.metadata_only,
        parse_only=args.parse_only,
        checkpoint_gate=args.checkpoint_gate,
        family_sample_path=args.family_sample,
        aliases_path=args.aliases,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
