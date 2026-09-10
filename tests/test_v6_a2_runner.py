from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from v6.a2_runner import run_a2


@dataclass(frozen=True)
class RunnerInputs:
    mode: str
    output: Path
    sample: Path
    aliases: Path
    metadata: Path
    sections: Path
    split: Path
    trade_calendar: Path
    schedule: Path | None = None
    code_manifest: Path | None = None
    external_cohort: Path | None = None

    def as_kwargs(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "output": self.output,
            "sample": self.sample,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "sections": self.sections,
            "split": self.split,
            "trade_calendar": self.trade_calendar,
            "schedule": self.schedule,
            "code_manifest": self.code_manifest,
            "external_cohort": self.external_cohort,
        }

    def with_mode(self, mode: str, output: Path | None = None) -> RunnerInputs:
        return replace(self, mode=mode, output=output or self.output)

    def with_output(self, output: Path) -> RunnerInputs:
        return replace(self, output=output)

    def with_development_rows(self, role: str, output: Path | None = None) -> RunnerInputs:
        frame = pd.read_csv(self.split)
        frame.loc[0, "split_role"] = role
        split = self.split.parent / f"split-{role}.csv"
        frame.to_csv(split, index=False)
        return replace(self, split=split, output=output or self.output)

    def with_schedule_count(self, count: int, output: Path | None = None) -> RunnerInputs:
        dates = pd.date_range("2006-01-31", periods=count, freq="ME")
        schedule = self.split.parent / f"schedule-{count}.csv"
        pd.DataFrame({"decision_date": dates}).to_csv(schedule, index=False)
        return replace(self, schedule=schedule, output=output or self.output)


@pytest.fixture
def runner_inputs(tmp_path: Path) -> RunnerInputs:
    sample = tmp_path / "sample.csv"
    pd.DataFrame(
        [{"family_key": "manager|fund-dev", "share_code": "000001"}]
    ).to_csv(sample, index=False)

    aliases = tmp_path / "aliases.csv"
    pd.DataFrame(
        [{"sample_share_code": "000001", "family_key": "manager|fund-dev"}]
    ).to_csv(aliases, index=False)

    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_document": "dev-event.pdf",
                        "share_code": "000001",
                        "family_key": "manager|fund-dev",
                        "split_role": "development",
                        "title": "基金成立公告",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source_document": "external-1.pdf",
                        "share_code": "000002",
                        "family_key": "external|fund-1",
                        "split_role": "external",
                        "title": "基金转换业务公告",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sections = tmp_path / "sections.jsonl"
    sections.write_text(
        json.dumps(
            {
                "source_document": "dev-event.pdf",
                "family_key": "manager|fund-dev",
                "extraction_status": "success",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    split = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "family_key": "manager|fund-dev",
                "split_role": "development",
                "stratum": "stratum-1",
            }
        ]
    ).to_csv(split, index=False)

    calendar = tmp_path / "trade_cal.csv"
    dates = pd.date_range("2006-01-31", periods=243, freq="ME")
    pd.DataFrame(
        {
            "exchange": ["SSE"] * len(dates),
            "cal_date": dates.strftime("%Y%m%d"),
            "is_open": [1] * len(dates),
        }
    ).to_csv(calendar, index=False)

    return RunnerInputs(
        mode="shadow-events",
        output=tmp_path / "a2",
        sample=sample,
        aliases=aliases,
        metadata=metadata,
        sections=sections,
        split=split,
        trade_calendar=calendar,
    )


def test_shadow_events_never_writes_denominator_or_verdict(
    runner_inputs: RunnerInputs,
) -> None:
    run_a2(**runner_inputs.as_kwargs())

    assert not list(runner_inputs.output.glob("*verdict*"))
    assert not (runner_inputs.output / "frozen").exists()
    assert not list(runner_inputs.output.rglob("legal_state_events.csv"))
    assert (runner_inputs.output / "A2_IMPLEMENTATION_STATE.json").is_file()


def test_development_mode_rejects_untouched_family_rows(
    runner_inputs: RunnerInputs,
) -> None:
    inputs = runner_inputs.with_development_rows("untouched_validation")

    with pytest.raises(ValueError, match="untouched-validation leakage"):
        run_a2(**inputs.with_mode("validate-development").as_kwargs())


def test_external_validation_requires_frozen_code_manifest(
    runner_inputs: RunnerInputs,
) -> None:
    inputs = runner_inputs.with_mode("validate-external-events")

    with pytest.raises(FileNotFoundError, match="frozen code manifest"):
        run_a2(**inputs.as_kwargs())


def test_freeze_requires_complete_243_month_schedule(
    runner_inputs: RunnerInputs,
) -> None:
    inputs = runner_inputs.with_schedule_count(242).with_mode("freeze-denominators")

    with pytest.raises(ValueError, match="243 monthly decision dates"):
        run_a2(**inputs.as_kwargs())


def test_runner_refuses_a1_v2_output_directory(runner_inputs: RunnerInputs) -> None:
    output = runner_inputs.output / "output" / "v6" / "g0_full_pit" / "a1" / "v2"

    with pytest.raises(ValueError, match="a1/v2 is immutable"):
        run_a2(**runner_inputs.with_output(output).as_kwargs())


def test_state_records_hashes_rows_completed_phase_and_exact_next_command(
    runner_inputs: RunnerInputs,
) -> None:
    run_a2(**runner_inputs.as_kwargs())
    state_path = runner_inputs.output / "A2_IMPLEMENTATION_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["last_completed_phase"] == "shadow-events"
    assert state["next_command"].startswith("python -m v6.a2_runner")
    assert state["input_sha256"]
    assert state["split_sha256"]
    assert state["code_test_manifest_sha256"]
    assert state["row_counts"]["sample"] == 1
    assert state["row_counts"]["metadata"] == 2
    assert state["row_counts"]["decision_dates"] == 243


def test_same_hash_rerun_is_byte_identical_and_resumable(
    runner_inputs: RunnerInputs,
) -> None:
    run_a2(**runner_inputs.as_kwargs())
    state_path = runner_inputs.output / "A2_IMPLEMENTATION_STATE.json"
    before = state_path.read_bytes()

    run_a2(**runner_inputs.as_kwargs())

    assert state_path.read_bytes() == before


def test_changed_input_hash_fails_closed_before_state_mutation(
    runner_inputs: RunnerInputs,
) -> None:
    run_a2(**runner_inputs.as_kwargs())
    state_path = runner_inputs.output / "A2_IMPLEMENTATION_STATE.json"
    before = state_path.read_bytes()
    runner_inputs.sample.write_text(
        runner_inputs.sample.read_text(encoding="utf-8") + "000002,manager|fund-new\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        run_a2(**runner_inputs.as_kwargs())

    assert state_path.read_bytes() == before


def test_phase_transition_preserves_completed_phases(
    runner_inputs: RunnerInputs,
) -> None:
    run_a2(**runner_inputs.as_kwargs())
    run_a2(**runner_inputs.with_mode("validate-development").as_kwargs())

    state = json.loads(
        (runner_inputs.output / "A2_IMPLEMENTATION_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["last_completed_phase"] == "validate-development"
    assert state["completed_phases"] == ["shadow-events", "validate-development"]


def test_runner_has_no_adjudicate_mode(runner_inputs: RunnerInputs) -> None:
    with pytest.raises(ValueError, match="adjudicate is not implemented"):
        run_a2(**runner_inputs.with_mode("adjudicate").as_kwargs())
