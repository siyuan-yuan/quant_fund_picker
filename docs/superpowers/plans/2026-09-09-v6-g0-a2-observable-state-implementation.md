# V6-G0-A2 Observable Legal-State Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic A2 Gate that measures official legal-state events and monthly decision-date state availability without treating routine disclosure documents as critical events.

**Architecture:** Add five focused A2 modules: eligibility mapping, monthly schedule, event generation, evidence/state resolution, and Gate/artifact orchestration. Reuse the existing A1 section extraction and causal timeline primitives only where their semantics match A2; never write into `a1/v2`. Freeze development rules first, open untouched validation once, then freeze event and decision-observation denominators before any formal adjudication.

**Tech Stack:** Python 3.11, pandas, pytest, existing V6 CSV/JSONL and atomic artifact utilities.

**Spec:** `docs/superpowers/specs/2026-09-09-v6-g0-a2-observable-state-design.md`

## Global Constraints

- Preserve `a1/v2` as immutable first official A1 FAIL; checkpoint SHA-256 stays `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd`.
- Do not read G1–G4 results while developing or validating A2.
- Do not generate an A2 denominator until development code/tests and the 24/36 split are frozen.
- Do not run formal A2 adjudication in this plan; stop after reviewed denominator freeze.
- `event_id` excludes every source-document field; events own evidence candidates 1:N.
- Monthly dates are the 243 SSE last trading days from 2006-01 through 2026-03, inclusive.
- Eligibility mapping version is exactly `state_to_universe_eligibility_v1`.
- The original 36 validation families cannot create one-off family rules and are not called untouched for event-title semantics because their A1 transformation metadata was previously inspected.
- Freeze a separate 120-row external event-title validation cohort exactly as spec section 6 before manual labelling.
- Current workspace has no Git metadata. Never fabricate commits; after each task record file hashes and `NO_GIT` in `docs/V6_实验协议与执行台账.md`. If execution moves to a real Git checkout, replace that snapshot step with the stated commit.

---

### Task 1: Freeze legal state → universe eligibility

**Files:**
- Create: `v6/a2_eligibility.py`
- Create: `tests/test_v6_a2_eligibility.py`
- Modify: `v6/contracts.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: `legal_fund_type: str`, `equity_min_pct: float | None`, `equity_max_pct: float | None`.
- Produces: `EligibilityResult(status: str, target_type: str, exclusion_reason: str)` and `state_to_universe_eligibility_v1(legal_fund_type: str, equity_min_pct: float | None, equity_max_pct: float | None) -> EligibilityResult`.

- [ ] **Step 1: Write exact boundary and exclusion tests**

```python
import pytest
from v6.a2_eligibility import state_to_universe_eligibility_v1

@pytest.mark.parametrize("minimum,expected", [(59.999, "UNRESOLVED_ELIGIBILITY"), (60.0, "ELIGIBLE")])
def test_mixed_equity_boundary_is_frozen(minimum, expected):
    result = state_to_universe_eligibility_v1("混合型", minimum, 95.0)
    assert result.status == expected
    assert result.target_type == ("混合型-偏股" if expected == "ELIGIBLE" else "")

@pytest.mark.parametrize("maximum,expected", [(30.0, "EXCLUDED"), (30.001, "ELIGIBLE")])
def test_flexible_boundary_is_frozen(maximum, expected):
    result = state_to_universe_eligibility_v1("混合型-灵活", 0.0, maximum)
    assert result.status == expected

def test_explicit_non_target_precedes_numeric_rule():
    result = state_to_universe_eligibility_v1("保本混合型", 60.0, 95.0)
    assert (result.status, result.exclusion_reason) == ("EXCLUDED", "EXPLICIT_NON_TARGET_MANDATE")

def test_invalid_or_missing_bounds_never_enter_universe():
    assert state_to_universe_eligibility_v1("混合型", None, 95.0).status == "UNRESOLVED_ELIGIBILITY"
    assert state_to_universe_eligibility_v1("股票型", 96.0, 95.0).status == "UNRESOLVED_ELIGIBILITY"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_eligibility.py -q`

Expected: import failure because `v6.a2_eligibility` does not exist.

- [ ] **Step 3: Implement the frozen mapping**

Create an immutable dataclass and implement precedence exactly as spec section 7. Add `A2_ELIGIBILITY_VERSION = "state_to_universe_eligibility_v1"` and `A2_TARGET_TYPES` to `v6/contracts.py`; do not reuse name heuristics from current fund metadata.

```python
@dataclass(frozen=True)
class EligibilityResult:
    status: str
    target_type: str = ""
    exclusion_reason: str = ""

def state_to_universe_eligibility_v1(
    legal_fund_type: str,
    equity_min_pct: float | None,
    equity_max_pct: float | None,
) -> EligibilityResult:
    """Apply the complete precedence table from A2 specification section 7."""
    # The implementation added in this step must contain every table branch;
    # this declaration fixes the public signature and is not copied verbatim.
    raise NotImplementedError
```

- [ ] **Step 4: Run GREEN and regression**

Run: `python -m pytest tests/test_v6_a2_eligibility.py tests/test_v6_fund_master.py -q`

Expected: all tests pass with exit 0; existing `TARGET_TYPES` remain unchanged.

- [ ] **Step 5: Freeze task evidence**

Record hashes and test count in the ledger. Git alternative: `git commit -m "feat(v6): freeze A2 universe eligibility mapping"`.

---

### Task 2: Generate the exact 243-date monthly schedule

**Files:**
- Create: `v6/a2_schedule.py`
- Create: `tests/test_v6_a2_schedule.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: Tushare `trade_cal.csv` with `exchange`, `cal_date`, `is_open`.
- Produces: `build_a2_monthly_schedule(trade_calendar: pd.DataFrame) -> pd.DataFrame` with `decision_date`, `calendar_month`, `schedule_version`.

- [ ] **Step 1: Write schedule tests**

```python
def test_schedule_uses_last_open_sse_day_per_month():
    calendar = pd.DataFrame({
        "exchange": ["SSE", "SSE", "SSE", "SSE"],
        "cal_date": ["20200123", "20200124", "20200131", "20200228"],
        "is_open": [1, 0, 0, 1],
    })
    result = build_a2_monthly_schedule(calendar, start="2020-01-01", end="2020-02-29")
    assert result.decision_date.dt.strftime("%Y-%m-%d").tolist() == ["2020-01-23", "2020-02-28"]

def test_full_schedule_has_frozen_range_and_count(real_trade_calendar):
    result = build_a2_monthly_schedule(real_trade_calendar)
    assert len(result) == 243
    assert result.calendar_month.iloc[[0, -1]].tolist() == ["2006-01", "2026-03"]
    assert result.decision_date.is_monotonic_increasing
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_schedule.py -q`

Expected: import failure.

- [ ] **Step 3: Implement deterministic SSE calendar handling**

Filter `exchange == "SSE"` and `is_open == 1`, normalize `cal_date`, select the maximum open date in every calendar month from `DECISION_START` through `DECISION_END`, and fail if any month is missing. Set `A2_SCHEDULE_VERSION = "monthly-sse-last-open-v1"`.

- [ ] **Step 4: Run GREEN and compare historical invariant**

Run: `python -m pytest tests/test_v6_a2_schedule.py tests/test_research_invariants.py -q`

Expected: exit 0 and January 2020 resolves to 2020-01-23.

- [ ] **Step 5: Freeze task evidence**

Record code/test hashes and the input calendar path, row count and hash. Git alternative: `git commit -m "feat(v6): add frozen A2 monthly decision schedule"`.

---

### Task 3: Generate genuine legal-state events with document-independent IDs

**Files:**
- Create: `v6/a2_events.py`
- Create: `tests/test_v6_a2_events.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Produces `LegalStateEvent(event_id, family_key, event_type, legal_subject, event_anchor_date, generation_version)`.
- Produces `EvidenceCandidate(event_id, source_document, known_at, document_stage, report_code, report_name)`.
- Produces `generate_legal_state_events(master, metadata, aliases, split_role) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write identity, 1:N, positive-event and exclusion tests**

```python
def test_multiple_documents_share_one_event_id():
    rows = transformation_fixture(files=["notice.pdf", "contract.pdf", "prospectus.pdf"])
    events, evidence = generate_legal_state_events(**rows)
    assert len(events) == 1
    assert evidence.event_id.nunique() == 1
    assert set(evidence.source_document) == {"notice.pdf", "contract.pdf", "prospectus.pdf"}

@pytest.mark.parametrize("title", [
    "暂停大额申购和转换转入业务的公告",
    "新增代销机构并开通基金转换业务的公告",
    "调整基金转换费率的公告",
])
def test_transaction_conversion_is_not_legal_transformation(title):
    events, _ = generate_legal_state_events(**metadata_fixture(title))
    assert "LEGAL_TRANSFORMATION" not in set(events.event_type)

def test_explicit_effective_transformation_is_event():
    events, _ = generate_legal_state_events(**metadata_fixture(
        "保本周期到期并于2017年5月11日起转型为灵活配置混合型基金"
    ))
    assert events.event_type.tolist() == ["LEGAL_TRANSFORMATION"]
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_events.py -q`

Expected: import failure.

- [ ] **Step 3: Implement closed taxonomy and event-anchor rules**

Use the six spec event types. Implement positive legal-change phrases and explicit transaction exclusions before positive matching. Hash only the five frozen identity fields. Keep every file field out of `LegalStateEvent`; attach files only through `EvidenceCandidate.event_id`.

- [ ] **Step 4: Add adversarial regression for the 49 A1 transformation titles**

Create a frozen fixture containing title, expected `LEGAL_TRANSFORMATION`/`NOT_EVENT`, and review status. Assert the 33 definite transaction-conversion false positives do not create events. The seven previously uncertain titles remain fail-closed `REVIEW_REQUIRED` until development-only legal context labels them; do not infer from A1 disposition.

- [ ] **Step 5: Run GREEN**

Run: `python -m pytest tests/test_v6_a2_events.py tests/test_v6_classification_checkpoints.py -q`

Expected: exit 0; A1 tests remain unchanged.

- [ ] **Step 6: Freeze task evidence**

Record hashes, event-rule version and fixture counts. Git alternative: `git commit -m "feat(v6): generate document-independent legal state events"`.

---

### Task 4: Attach evidence and preserve incompatible same-event conflicts

**Files:**
- Create: `v6/a2_evidence.py`
- Create: `tests/test_v6_a2_evidence.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: event rows, evidence candidates, existing structured-section-v3 results.
- Produces: `resolve_event_evidence(events, candidates, sections, *, predecessor_states=None) -> pd.DataFrame` with result in `NEW_STATE`, `VERIFIED_CONTINUITY`, `TERMINATED`, `UNRESOLVED` and explicit `failure_reason`.

- [ ] **Step 1: Write legal-priority and conflict tests**

```python
def test_priority_cannot_override_disjoint_states():
    evidence = same_event_evidence(contract=(60, 95), prospectus=(0, 30))
    row = resolve_event_evidence(*evidence).iloc[0]
    assert row.result == "UNRESOLVED"
    assert row.failure_reason == "UNRESOLVED_EVENT_CONFLICT"

def test_compatible_constraints_keep_all_sources():
    evidence = same_event_evidence(contract=(40, 95), prospectus=(40, 100))
    row = resolve_event_evidence(*evidence).iloc[0]
    assert (row.equity_min_pct, row.equity_max_pct) == (40, 95)
    assert set(row.contributing_sources.split("|")) == {"contract.pdf", "prospectus.pdf"}

def test_document_pointer_is_not_state_clause():
    row = resolve_event_evidence(*pointer_only_fixture()).iloc[0]
    assert row.failure_reason == "GOVERNING_DOCUMENT_REQUIRED"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_evidence.py -q`

Expected: import failure.

- [ ] **Step 3: Implement resolver using existing causal/provenance rules**

Require exact family/alias provenance, `known_at`, legal stage, evidence location and source checksum. Apply priority only after compatibility. `VERIFIED_CONTINUITY` requires affirmative unchanged text plus a resolved causal predecessor; operational silence remains unresolved/not applicable.

- [ ] **Step 4: Run GREEN with existing section and A1 causal suites**

Run: `python -m pytest tests/test_v6_a2_evidence.py tests/test_v6_csrc_fund_disclosure.py tests/test_v6_classification_checkpoints.py -q`

Expected: exit 0.

- [ ] **Step 5: Freeze task evidence**

Record hashes and test summary. Git alternative: `git commit -m "feat(v6): resolve A2 event evidence without conflict override"`.

---

### Task 5: Build causal intervals and monthly decision observations

**Files:**
- Create: `v6/a2_timeline.py`
- Create: `tests/test_v6_a2_timeline.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Produces `build_legal_state_timeline(event_dispositions) -> pd.DataFrame`.
- Produces `build_decision_observations(families, schedule, timeline) -> pd.DataFrame`.
- Observation statuses: `OBSERVED_STATE`, `PRE_OBSERVABLE`, `UNRESOLVED_EVENT_CONFLICT`; termination is a boundary record.

- [ ] **Step 1: Write publication/effectiveness and termination tests**

```python
def test_state_starts_at_later_of_known_and_legal_dates():
    timeline = build_legal_state_timeline(events_fixture(known_at="2019-02-01", legal_effective_from="2018-01-01"))
    assert timeline.usable_from.iloc[0] == pd.Timestamp("2019-02-01")

def test_preobservable_month_is_counted_but_not_backfilled():
    obs = build_decision_observations(families_fixture(), schedule_fixture(["2019-01-31", "2019-02-28"]), state_fixture("2019-02-01"))
    assert obs.observation_status.tolist() == ["PRE_OBSERVABLE", "OBSERVED_STATE"]

def test_post_termination_month_is_not_active_denominator():
    obs = build_decision_observations(terminated_family_fixture("2020-01-15"), schedule_fixture(["2019-12-31", "2020-01-23"]), state_fixture("2019-01-01"))
    assert obs.loc[obs.decision_date.eq("2020-01-23"), "active_eligibility"].item() is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_timeline.py -q`

Expected: import failure.

- [ ] **Step 3: Implement timeline and query logic**

Use `usable_from=max(known_at, legal_effective_from)`, strictly later interval boundaries, same-time compatible state intersection, explicit conflict closure, and no cross-family predecessor. Apply `state_to_universe_eligibility_v1` at query time and retain its version/result fields.

- [ ] **Step 4: Run GREEN and invariant regressions**

Run: `python -m pytest tests/test_v6_a2_timeline.py tests/test_v6_classification_checkpoints.py tests/test_research_invariants.py -q`

Expected: exit 0; causal, nonpositive and conflict fixtures behave independently.

- [ ] **Step 5: Freeze task evidence**

Record hashes and test summary. Git alternative: `git commit -m "feat(v6): build A2 causal monthly state observations"`.

---

### Task 6: Implement Gate metrics and atomic artifacts

**Files:**
- Create: `v6/a2_gate.py`
- Create: `tests/test_v6_a2_gate.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Produces `compute_a2_gate(events, dispositions, timeline, observations, split) -> A2GateResult`.
- Produces `write_a2_artifacts(result, output_dir, mode)` with `mode in {shadow, freeze_denominator, adjudicate}`.

- [ ] **Step 1: Write every Gate boundary test**

```python
def test_gate_fails_below_event_boundary():
    assert compute_a2_gate(**gate_fixture(event_resolved=949, event_total=1000)).passed is False

def test_gate_passes_event_boundary_when_every_other_condition_passes():
    assert compute_a2_gate(**gate_fixture(event_resolved=950, event_total=1000)).event_coverage_passed is True

def test_any_unresolved_inception_fails():
    result = compute_a2_gate(**gate_fixture(critical_unresolved=1))
    assert (result.passed, result.critical_events_passed) == (False, False)

def test_decision_observation_boundary_is_exact():
    assert compute_a2_gate(**gate_fixture(observed=949, active_observations=1000)).decision_coverage_passed is False
    assert compute_a2_gate(**gate_fixture(observed=950, active_observations=1000)).decision_coverage_passed is True

@pytest.mark.parametrize("field", ["causal_violations", "nonpositive_intervals", "unresolved_conflicts"])
def test_each_invariant_failure_blocks_gate(field):
    result = compute_a2_gate(**gate_fixture(**{field: 1}))
    assert result.passed is False

def test_family_and_stratum_boundaries_are_exact():
    assert compute_a2_gate(**gate_fixture(covered_families=53, total_families=60)).family_coverage_passed is False
    assert compute_a2_gate(**gate_fixture(covered_families=54, total_families=60)).family_coverage_passed is True
    assert compute_a2_gate(**gate_fixture(stratum_counts=[(7, 10)] + [(8, 10)] * 5)).strata_passed is False
```

Define `gate_fixture` in the same test file with passing defaults: 950/1000 events, zero critical unresolved, 54/60 families, six `(8,10)` strata, 950/1000 observed active pairs, and all invariant counts zero.

Each fixture supplies exact integer numerators and denominators; assertions check both boolean verdict and reported fraction.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_gate.py -q`

Expected: import failure.

- [ ] **Step 3: Implement exact counters and atomic write boundary**

Write all eight spec artifacts through a temporary directory, flush and fsync files, then atomically rename. `shadow` cannot write a verdict or dispositions; `freeze_denominator` writes only event/evidence/observation identities plus manifest; `adjudicate` refuses unless frozen hashes match.

- [ ] **Step 4: Add immutability and mode tests**

Assert an existing adjudication directory cannot be overwritten, shadow mode cannot create Gate files, denominator freeze is byte-identical on repeated input, and an input/hash mismatch aborts before output mutation.

- [ ] **Step 5: Run GREEN**

Run: `python -m pytest tests/test_v6_a2_gate.py tests/test_v6_g0_checkpoint_gate.py -q`

Expected: exit 0 and no A1 artifact changes.

- [ ] **Step 6: Freeze task evidence**

Record hashes and test summary. Git alternative: `git commit -m "feat(v6): add immutable A2 Gate artifacts"`.

---

### Task 7: Wire a resumable A2 runner without adjudication

**Files:**
- Create: `v6/a2_runner.py`
- Create: `tests/test_v6_a2_runner.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- CLI modes: `--shadow-events`, `--validate-development`, `--freeze-external-event-validation`, `--validate-external-events`, `--validate-families`, `--freeze-denominators`.
- No `--adjudicate` mode is implemented in this task.

- [ ] **Step 1: Write CLI boundary tests**

```python
def test_shadow_events_never_writes_denominator_or_verdict(tmp_path, runner_inputs):
    run_a2(runner_inputs.with_mode("shadow-events", tmp_path))
    assert not list(tmp_path.glob("*verdict*"))
    assert not (tmp_path / "frozen").exists()

def test_development_mode_rejects_untouched_family_rows(tmp_path, runner_inputs):
    with pytest.raises(ValueError, match="untouched-validation leakage"):
        run_a2(runner_inputs.with_development_rows(role="untouched_validation", output=tmp_path))

def test_external_validation_requires_frozen_code_and_cohort(tmp_path, runner_inputs):
    with pytest.raises(FileNotFoundError, match="frozen code manifest"):
        run_a2(runner_inputs.with_mode("validate-external-events", tmp_path))

def test_freeze_requires_validation_and_243_dates(tmp_path, runner_inputs):
    with pytest.raises(ValueError, match="243 monthly decision dates"):
        run_a2(runner_inputs.with_schedule_count(242, output=tmp_path))

def test_runner_refuses_a1_v2_output_directory(runner_inputs):
    with pytest.raises(ValueError, match="a1/v2 is immutable"):
        run_a2(runner_inputs.with_output(Path("output/v6/g0_full_pit/csrc_stratified/a1/v2")))
```

Define `runner_inputs` as a pytest fixture that writes minimal sample, alias, metadata, section-cache, split and trade-calendar inputs under `tmp_path`; its default mode has one development family, external non-sample metadata rows, and a complete synthetic 243-month calendar.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v6_a2_runner.py -q`

Expected: import failure.

- [ ] **Step 3: Implement resumable phase machine**

Persist `A2_IMPLEMENTATION_STATE.json` after every phase with input hashes, code/test manifest, split hash, row counts, last completed phase and exact next command. Repeated runs skip hash-matching completed work; mismatches fail closed.

- [ ] **Step 4: Run GREEN and full deterministic test groups**

Run these separately and preserve JUnit plus exit codes:

```text
python -m pytest tests/test_v6_a2_eligibility.py tests/test_v6_a2_schedule.py --junitxml=output/v6/a2/test-eligibility-schedule.xml -q
python -m pytest tests/test_v6_a2_events.py tests/test_v6_a2_evidence.py --junitxml=output/v6/a2/test-events-evidence.xml -q
python -m pytest tests/test_v6_a2_timeline.py tests/test_v6_a2_gate.py tests/test_v6_a2_runner.py --junitxml=output/v6/a2/test-timeline-gate-runner.xml -q
python -m pytest tests/test_v6_csrc_fund_disclosure.py tests/test_v6_classification_checkpoints.py tests/test_v6_g0_checkpoint_gate.py --junitxml=output/v6/a2/test-a1-regression.xml -q
```

Expected: every command exit 0; record each test count rather than combining console output.

- [ ] **Step 5: Freeze implementation manifest**

Hash every A2 source/test/runner, JUnit file, spec revision, A1 frozen checkpoint, split and calendar input. Git alternative: `git commit -m "feat(v6): wire resumable A2 pre-adjudication runner"`.

---

### Task 8: Development validation, external event validation, family validation, and denominator freeze

**Files:**
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/validation/development_event_audit.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/validation/external_event_validation_frozen.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/validation/external_event_validation_labels.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/validation/family_validation_audit.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/frozen/legal_state_events.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/frozen/event_evidence_candidates.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/frozen/decision_observation_ids.csv`
- Create: `output/v6/g0_full_pit/csrc_stratified/a2/frozen/a2_denominator_manifest.sha256`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes only frozen code/tests, official inputs, 24/36 split, the mechanically frozen 120-row external cohort, and SSE calendar.
- Produces reviewed denominators; no state dispositions, coverage score or Gate verdict.

- [ ] **Step 1: Run development-only event audit**

Run:

```text
python -m v6.a2_runner --validate-development --sample output/v6/g0_full_pit/csrc_stratified_sample.csv --aliases output/v6/g0_full_pit/csrc_stratified_aliases.csv --metadata output/v6/g0_full_pit/csrc_stratified/metadata.jsonl --sections output/v6/g0_full_pit/csrc_stratified/a1/root_cause_audit/structured_section_v3_shadow/section_cache.jsonl --split output/v6/g0_full_pit/csrc_stratified/a1/root_cause_audit/structured_section_v3_dev_validation_split.csv --trade-calendar cache/tushare/bulk/trade_cal.csv --output output/v6/g0_full_pit/csrc_stratified/a2
```

Manually label every development candidate as closed-taxonomy event, explicit non-event, or review-required with exact title/text evidence. Fix only general rules through new RED/GREEN tests; refreeze code after each revision.

- [ ] **Step 2: Obtain independent development review**

Reviewer checks all positive events, explicit exclusions, event anchoring, duplicate grouping and zero access to untouched content. Any blocker returns to Task 3 or 4 with a new revision.

- [ ] **Step 3: Freeze and label external title validation once**

Run the same command as Step 1 with `--validate-development` replaced by `--freeze-external-event-validation`; assert exactly 120 metadata rows outside the 60 sampled families, no more than three per share code, deterministic seed/hash ordering, and no label column. Record the file hash before manual labels are added. Then label the frozen rows and run `--validate-external-events` without changing code.

Record precision/recall by event type and false-positive category. Do not add rules from these rows in the same revision. If validation fails the spec stop condition, record failure and do not freeze denominators.

- [ ] **Step 4: Run 36-family end-to-end validation**

Run the same command as Step 1 with `--validate-development` replaced by `--validate-families`. This validates alias, evidence, timeline and observation behavior on the original 36-family side, while explicitly reporting that its transformation title metadata was previously exposed.

- [ ] **Step 5: Freeze the two denominators**

Run the same command as Step 1 with `--validate-development` replaced by `--freeze-denominators`; all paths and output remain identical.

Assert 243 ordered dates, exact 60-family/six-stratum membership, unique event IDs, 1:N evidence integrity, no source field in event identity, complete active family/date enumeration, and byte-stable repeat output.

- [ ] **Step 6: Obtain two independent read-only reviews**

Specification review checks every A2 rule and denominator identity. Quality review independently recomputes event IDs, monthly dates, split membership, identity/evidence cardinality and hashes. Both must approve.

- [ ] **Step 7: Stop before formal adjudication**

Update `A2_IMPLEMENTATION_STATE.json` to `DENOMINATORS_FROZEN_AWAITING_USER_APPROVAL`, record exact next command but do not run it, append final hashes and `NO_GIT` to the ledger, and return control to the user.

---

## Plan completion boundary

This plan is complete only when A2 code/tests generalize to untouched validation and both denominators are independently approved and frozen. It intentionally stops before reading A2 state outcomes, computing coverage, writing a Gate verdict, or entering G1.
