"""Resumable, pre-adjudication A2 phase runner.

This module records deterministic phase state only.  It deliberately does not
call ``write_a2_artifacts`` and has no adjudication mode; denominator material
is produced only by the later validation/freeze task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .a2_events import generate_legal_state_events
from .a2_schedule import build_a2_monthly_schedule
from .a2_validation import (
    build_development_event_audit,
    build_family_validation_audit,
    freeze_external_event_validation_cohort,
    freeze_reviewed_denominators,
    validate_external_event_labels,
)


RUNNER_PHASE = "a2_implementation_task_7_runner"
STATE_FILENAME = "A2_IMPLEMENTATION_STATE.json"
MODES = (
    "shadow-events",
    "validate-development",
    "freeze-external-event-validation",
    "validate-external-events",
    "validate-families",
    "freeze-denominators",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_schedule(trade_calendar: Path, schedule: Path | None) -> pd.DataFrame:
    if schedule is not None:
        schedule = _require_file(schedule, "schedule")
        frame = pd.read_csv(schedule)
        if "decision_date" not in frame.columns:
            raise ValueError("schedule missing decision_date")
        dates = pd.to_datetime(frame["decision_date"], errors="coerce").dt.normalize()
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("schedule contains invalid or duplicate decision dates")
        return pd.DataFrame({"decision_date": dates.sort_values().reset_index(drop=True)})
    calendar = pd.read_csv(_require_file(trade_calendar, "trade calendar"))
    return build_a2_monthly_schedule(calendar)


def _require_complete_monthly_schedule(schedule: pd.DataFrame) -> None:
    if len(schedule) != 243:
        raise ValueError("freeze requires 243 monthly decision dates")
    dates = pd.to_datetime(schedule["decision_date"], errors="coerce").dt.normalize()
    expected = pd.period_range("2006-01", "2026-03", freq="M").astype(str).tolist()
    actual = dates.dt.to_period("M").astype(str).tolist()
    if dates.isna().any() or actual != expected:
        raise ValueError("freeze requires the complete ordered 243 monthly decision dates")


def _code_test_manifest() -> tuple[str, list[dict[str, str]]]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted(
        set(root.glob("v6/a2_*.py")) | set(root.glob("tests/test_v6_a2_*.py")),
        key=lambda path: path.as_posix(),
    )
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
        for path in paths
        if path.is_file()
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload), records


def _input_hash(paths: Iterable[tuple[str, Path]]) -> str:
    records = [
        {"label": label, "path": path.as_posix(), "sha256": _sha256_file(path)}
        for label, path in sorted(paths, key=lambda item: item[0])
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _reject_immutable_output(output: Path) -> None:
    parts = {part.lower() for part in output.resolve().parts}
    if "a1" in parts and "v2" in parts:
        raise ValueError("a1/v2 is immutable; A2 runner output is forbidden there")


def _validate_development_isolation(split: pd.DataFrame) -> None:
    role_columns = [column for column in ("split_role", "role", "split") if column in split.columns]
    if not role_columns:
        raise ValueError("split missing development/validation role")
    values = {
        str(value).strip().lower().replace("_", "-")
        for column in role_columns
        for value in split[column].dropna()
    }
    has_development = bool(values & {"development", "dev"})
    leakage = {
        value
        for value in values
        if value not in {"development", "dev"}
        and ("validation" in value or "external" in value or value in {"untouched", "test"})
    }
    if leakage and not has_development:
        raise ValueError("untouched-validation leakage in development mode")


def _next_command(mode: str, paths: dict[str, Path], output: Path) -> str:
    flag = f"--{mode}"
    command = [
        "python",
        "-m",
        "v6.a2_runner",
        flag,
        "--sample",
        paths["sample"].as_posix(),
        "--aliases",
        paths["aliases"].as_posix(),
        "--metadata",
        paths["metadata"].as_posix(),
        "--sections",
        paths["sections"].as_posix(),
        "--split",
        paths["split"].as_posix(),
        "--trade-calendar",
        paths["trade_calendar"].as_posix(),
        "--output",
        output.as_posix(),
    ]
    if paths.get("schedule") is not None:
        command.extend(["--schedule", paths["schedule"].as_posix()])
    if paths.get("code_manifest") is not None:
        command.extend(["--code-manifest", paths["code_manifest"].as_posix()])
    if paths.get("external_cohort") is not None:
        command.extend(["--external-cohort", paths["external_cohort"].as_posix()])
    if paths.get("external_labels") is not None:
        command.extend(["--external-labels", paths["external_labels"].as_posix()])
    return " ".join(command)


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = Path(tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)[1])
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _check_existing_state(path: Path, current: dict[str, object]) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("existing A2 implementation state is invalid") from exc
    for key in ("input_sha256", "split_sha256", "code_test_manifest_sha256", "row_counts"):
        if existing.get(key) != current.get(key):
            raise ValueError(f"hash mismatch in existing A2 implementation state: {key}")
    return existing


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _task8_output_dir(output: Path) -> Path:
    return output / "validation"


def _run_task8_phase(
    mode: str,
    output: Path,
    sample: pd.DataFrame,
    aliases: pd.DataFrame,
    metadata: pd.DataFrame,
    sections: pd.DataFrame,
    split: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    external_cohort: Path | None,
    external_labels: Path | None,
) -> None:
    """Run Task 8 validation/freeze work without state dispositions or Gate metrics."""

    validation_dir = _task8_output_dir(output)
    if mode == "validate-development":
        audit = build_development_event_audit(
            sample, metadata, aliases, split, sections, allow_validation=True
        )
        _atomic_csv(audit, validation_dir / "development_event_audit.csv")
        return
    if mode == "freeze-external-event-validation":
        cohort = freeze_external_event_validation_cohort(metadata, sample, aliases)
        _atomic_csv(cohort, validation_dir / "external_event_validation_frozen.csv")
        return
    if mode == "validate-external-events":
        if external_cohort is None or external_labels is None:
            raise FileNotFoundError("frozen external cohort and labels are required for external validation")
        cohort = pd.read_csv(external_cohort)
        labels = pd.read_csv(external_labels)
        result = validate_external_event_labels(cohort, labels)
        _atomic_csv(result, validation_dir / "external_event_validation_labels.csv")
        return
    if mode == "validate-families":
        audit = build_family_validation_audit(
            sample, metadata, aliases, split, sections
        )
        _atomic_csv(audit, validation_dir / "family_validation_audit.csv")
        return
    if mode == "freeze-denominators":
        events, evidence = generate_legal_state_events(sample, metadata, aliases)
        sample_by_family = sample.set_index(sample["family_key"].astype(str))
        dates = pd.to_datetime(schedule["decision_date"], errors="coerce").dt.normalize()
        observations = []
        for family in split["family_key"].astype(str).drop_duplicates():
            master = sample_by_family.loc[family]
            inception = pd.to_datetime(master.get("inception_date"), errors="coerce")
            termination = pd.to_datetime(master.get("termination_date"), errors="coerce")
            for decision in dates:
                active = bool(
                    pd.notna(inception)
                    and decision >= inception
                    and (pd.isna(termination) or decision < termination)
                )
                observations.append(
                    {
                        "family_key": family,
                        "decision_date": decision,
                        "active_eligibility": active,
                    }
                )
        freeze_reviewed_denominators(
            events,
            evidence,
            pd.DataFrame(observations),
            split,
            schedule,
            output / "frozen",
        )


def run_a2(
    *,
    mode: str,
    output: str | Path,
    sample: str | Path,
    aliases: str | Path,
    metadata: str | Path,
    sections: str | Path,
    split: str | Path,
    trade_calendar: str | Path,
    schedule: str | Path | None = None,
    code_manifest: str | Path | None = None,
    external_cohort: str | Path | None = None,
    external_labels: str | Path | None = None,
) -> dict[str, object]:
    """Run one resumable A2 pre-adjudication phase and write only its state."""

    if mode == "adjudicate":
        raise ValueError("adjudicate is not implemented in Task 7")
    if mode not in MODES:
        raise ValueError(f"unknown A2 runner mode: {mode}")
    output_path = Path(output)
    _reject_immutable_output(output_path)

    path_values = {
        "sample": Path(sample),
        "aliases": Path(aliases),
        "metadata": Path(metadata),
        "sections": Path(sections),
        "split": Path(split),
        "trade_calendar": Path(trade_calendar),
    }
    optional_values = {
        "schedule": Path(schedule) if schedule is not None else None,
        "code_manifest": Path(code_manifest) if code_manifest is not None else None,
        "external_cohort": Path(external_cohort) if external_cohort is not None else None,
        "external_labels": Path(external_labels) if external_labels is not None else None,
    }
    for label, path in path_values.items():
        _require_file(path, label)
    for label, path in optional_values.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    if mode == "validate-development":
        _validate_development_isolation(pd.read_csv(path_values["split"]))
    if mode == "validate-external-events" and optional_values["code_manifest"] is None:
        raise FileNotFoundError("frozen code manifest is required for external validation")
    if mode == "validate-external-events" and optional_values["external_cohort"] is None:
        raise FileNotFoundError("frozen external event cohort is required for external validation")
    if mode == "validate-external-events" and optional_values["external_labels"] is None:
        raise FileNotFoundError("external event labels are required for external validation")

    schedule_frame = _read_schedule(path_values["trade_calendar"], optional_values["schedule"])
    if mode == "freeze-denominators":
        _require_complete_monthly_schedule(schedule_frame)

    sample_frame = pd.read_csv(path_values["sample"])
    aliases_frame = pd.read_csv(path_values["aliases"])
    split_frame = pd.read_csv(path_values["split"])
    metadata_rows = _read_jsonl(path_values["metadata"])
    section_rows = _read_jsonl(path_values["sections"])
    metadata_frame = pd.DataFrame(metadata_rows)
    sections_frame = pd.DataFrame(section_rows)
    source_paths = list(path_values.items()) + [
        (label, path) for label, path in optional_values.items() if path is not None
    ]
    current = {
        "schema_version": 1,
        "phase": RUNNER_PHASE,
        "status": "RUNNER_PHASE_COMPLETE",
        "mode": mode,
        "input_sha256": _input_hash(source_paths),
        "split_sha256": _sha256_file(path_values["split"]),
        "code_test_manifest_sha256": _code_test_manifest()[0],
        "row_counts": {
            "sample": len(sample_frame),
            "aliases": len(aliases_frame),
            "metadata": len(metadata_rows),
            "sections": len(section_rows),
            "split": len(split_frame),
            "decision_dates": len(schedule_frame),
        },
        "last_completed_phase": mode,
        "completed_phases": [mode],
        "next_command": _next_command(
            mode,
            {**path_values, **optional_values},
            output_path,
        ),
    }
    state_path = output_path / STATE_FILENAME
    existing = _check_existing_state(state_path, current)
    if mode != "shadow-events":
        _run_task8_phase(
            mode,
            output_path,
            sample_frame,
            aliases_frame,
            metadata_frame,
            sections_frame,
            split_frame,
            schedule_frame,
            external_cohort=optional_values["external_cohort"],
            external_labels=optional_values["external_labels"],
        )
    if existing is not None:
        if existing.get("last_completed_phase") == mode:
            return existing
        completed = list(existing.get("completed_phases", []))
        if mode not in completed:
            completed.append(mode)
        current["completed_phases"] = completed
        _write_state(state_path, current)
        return current
    _write_state(state_path, current)
    return current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one A2 pre-adjudication phase")
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true", dest=mode.replace("-", "_"))
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--aliases", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--sections", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--trade-calendar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--external-cohort", type=Path)
    parser.add_argument("--external-labels", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    mode = next(mode for mode in MODES if getattr(args, mode.replace("-", "_")))
    values = vars(args).copy()
    values.pop(mode.replace("-", "_"), None)
    for candidate in MODES:
        values.pop(candidate.replace("-", "_"), None)
    run_a2(mode=mode, **values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
