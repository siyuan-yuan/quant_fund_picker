# V6-G0-A1 Classification Checkpoint Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the preregistered V6-G0-A1 classification-checkpoint evidence pipeline and replace the document parser rate as the primary G0(c) decision metric while retaining it unchanged as a diagnostic.

**Architecture:** Add a focused `v6/classification_checkpoints.py` domain module that deterministically normalizes evidence, freezes required checkpoints, adjudicates `PARSED_STATE`/`VERIFIED_NOOP`/`UNRESOLVED`, and builds the PIT state timeline. Add `v6/g0_checkpoint_gate.py` as a reproducible artifact-and-Gate CLI; keep PDF acquisition and legacy parsing in `v6/csrc_fund_collector.py`, with only a narrow orchestration hook so historical v13–v17 outputs remain untouched.

**Tech Stack:** Python 3.11, pandas, hashlib/JSON/regex/Unicode standard library, pytest; existing pypdf and PaddleOCR fallback remain optional upstream extractors and are not new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-06-v6-g0-a1-state-checkpoint-design.md`

## Global Constraints

- Generate and persist the required-checkpoint denominator before reading parser success/failure outcomes or adjudicating states.
- The only disposition values are `PARSED_STATE`, `VERIFIED_NOOP`, and `UNRESOLVED`.
- A regex miss, absent extracted ratio, fund name, current Tushare type, later document, or low-confidence model result can never create `VERIFIED_NOOP`.
- `C_checkpoint = (N_PARSED_STATE + N_VERIFIED_NOOP) / N_REQUIRED_CHECKPOINT >= 95%`.
- Every `CRITICAL_CHECKPOINT` must be resolved: critical resolution = 100%.
- Fund timeline coverage must be >=90% overall and >=80% in each of the six preregistered strata.
- Causal violations = 0 and unresolved conflicts = 0.
- Continue reporting the frozen document diagnostic with denominator 1,993, current success 1,676, failures 317, and `C_doc=84.14%`; do not rewrite v13–v17 Gate artifacts.
- Preserve raw evidence, normalized evidence, source location, normalization version, and SHA-256 together; a hash never substitutes for evidence text.
- Never carry evidence backward in time or across unrelated fund families.
- Any critical extraction/comparison failure remains `UNRESOLVED` and makes G0 FAIL.
- G1–G4 remain frozen unless every amended G0 condition passes.
- Append every completed task and Gate decision to `docs/V6_实验协议与执行台账.md`.
- This working copy has no `.git`; run commit steps only in a real Git checkout. In this copy, record the exact changed-file list and test output in the ledger and never invent a commit id.

## File Structure

- Create `v6/classification_checkpoints.py`: constants, schemas, deterministic normalization/fingerprinting, checkpoint generation, dispositions, state timeline construction, and invariant validation.
- Create `v6/g0_checkpoint_gate.py`: load frozen inputs, write six amendment artifacts, calculate all amended Gate metrics, and expose the CLI.
- Modify `v6/csrc_fund_collector.py`: optional post-parse call into the new pipeline; preserve existing default behavior and v13–v17 outputs.
- Create `tests/test_v6_classification_checkpoints.py`: unit tests for denominator independence, normalization, checkpoint triggers, NO-OP evidence rules, conflict and causal invariants.
- Create `tests/test_v6_g0_checkpoint_gate.py`: artifact, Gate, six-strata, diagnostic continuity, and CLI integration tests.
- Modify `tests/test_v6_csrc_collector.py`: prove the new orchestration mode does not change legacy parse-only behavior.
- Modify `docs/V6_实验协议与执行台账.md`: append RED/GREEN/Gate evidence after every task.
- Generate under `output/v6/g0_full_pit/csrc_stratified/a1/`: the six frozen CSV/Markdown outputs named in the spec; never overwrite legacy files.

---

### Task 1: Freeze Domain Types and Evidence Normalization

**Files:**
- Create: `v6/classification_checkpoints.py`
- Create: `tests/test_v6_classification_checkpoints.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: raw extracted investment-clause text as `str`.
- Produces: `NORMALIZATION_VERSION: int = 1`, `Disposition(str, Enum)`, `normalize_investment_clause(text: str) -> str`, and `fingerprint_clause(normalized_text: str) -> str`.

- [ ] **Step 1: Write the failing normalization and enum tests**

```python
from v6.classification_checkpoints import (
    NORMALIZATION_VERSION,
    Disposition,
    fingerprint_clause,
    normalize_investment_clause,
)


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
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -v
```

Expected: collection fails because `v6.classification_checkpoints` does not exist.

- [ ] **Step 3: Implement the minimal closed types and deterministic normalizer**

```python
from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import Enum

NORMALIZATION_VERSION = 1


class Disposition(str, Enum):
    PARSED_STATE = "PARSED_STATE"
    VERIFIED_NOOP = "VERIFIED_NOOP"
    UNRESOLVED = "UNRESOLVED"


def normalize_investment_clause(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"第\s*\d+\s*页", "", value)
    value = value.replace("％", "%").replace("—", "-").replace("–", "-")
    value = re.sub(r"\s+", "", value)
    return value


def fingerprint_clause(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run focused tests and the existing parser suite**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py tests/test_v6_csrc_fund_disclosure.py -q
```

Expected: all tests PASS; existing clause extraction behavior is unchanged.

- [ ] **Step 5: Record Gate 1 and commit where Git exists**

Append normalization version, exact commands, pass counts, and changed files to the ledger. Gate 1 passes only if prohibited semantic tokens remain present and fingerprints are stable.

```powershell
git add v6/classification_checkpoints.py tests/test_v6_classification_checkpoints.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): freeze checkpoint evidence normalization"
```

---

### Task 2: Generate and Freeze the Required-Checkpoint Denominator

**Files:**
- Modify: `v6/classification_checkpoints.py`
- Modify: `tests/test_v6_classification_checkpoints.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: `metadata: pd.DataFrame`, `fund_aliases: pd.DataFrame`, `document_sections: pd.DataFrame | None`.
- Produces: `generate_required_checkpoints(metadata, fund_aliases, document_sections=None) -> pd.DataFrame` and `freeze_checkpoint_manifest(checkpoints, path) -> Path`.
- Required output columns: `checkpoint_id`, `family_key`, `share_code`, `upload_info_id`, `known_at`, `effective_date`, `trigger_type`, `is_critical`, `source_document`, `generation_rule_version`, `manifest_revision`, `supersedes_checkpoint_id`, `revision_reason`.

- [ ] **Step 1: Write RED tests proving parser-outcome independence and trigger coverage**

```python
def test_checkpoint_generation_ignores_parser_outcomes_and_keeps_failed_extraction():
    metadata = pd.DataFrame([
        {"fundCode": "000001", "uploadInfoId": "10", "reportCode": "FA010010",
         "reportName": "基金招募说明书", "reportSendDate": "2019-01-01"},
        {"fundCode": "000001", "uploadInfoId": "11", "reportCode": "FC020000",
         "reportName": "基金转型公告", "reportSendDate": "2020-01-01"},
    ])
    aliases = pd.DataFrame([
        {"family_key": "m|f", "query_code": "000001", "status": "存续",
         "inception_era": "2013_2019"}
    ])
    sections = pd.DataFrame([
        {"upload_info_id": "10", "extraction_status": "failure", "normalized_clause": "",
         "parser_status": "failure"},
        {"upload_info_id": "11", "extraction_status": "success", "normalized_clause": "股票资产60%-95%",
         "parser_status": "success"},
    ])
    first = generate_required_checkpoints(metadata, aliases, sections)
    second = generate_required_checkpoints(
        metadata,
        aliases,
        sections.assign(parser_status=["success", "failure"]),
    )
    assert first["checkpoint_id"].tolist() == second["checkpoint_id"].tolist()
    assert set(first["trigger_type"]) >= {"INCEPTION", "EXPLICIT_TRANSFORMATION", "SECTION_UNCOMPARABLE"}
    assert first.loc[first.trigger_type == "EXPLICIT_TRANSFORMATION", "is_critical"].all()


def test_changed_boundary_creates_checkpoint_but_identical_section_does_not():
    # Three chronologically ordered documents: 60%-95%, identical, then 40%-95%.
    # Expected section triggers are first material section and changed numeric boundary only.
    result = generate_required_checkpoints(metadata_fixture(), aliases_fixture(), sections_fixture())
    material = result[result.trigger_type == "MATERIAL_SECTION_CHANGE"]
    assert material.upload_info_id.tolist() == ["1", "3"]
```

- [ ] **Step 2: Run the two tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -k "checkpoint_generation or changed_boundary" -v
```

Expected: FAIL because generator interfaces are absent.

- [ ] **Step 3: Implement deterministic candidate generation**

Implement title/report-code trigger constants and the exact sequence below:

```python
GENERATION_RULE_VERSION = 1
CRITICAL_TITLE_PATTERN = re.compile(r"转型|合并|终止|清算|基金类别|投资范围|资产配置")


def generate_required_checkpoints(
    metadata: pd.DataFrame,
    fund_aliases: pd.DataFrame,
    document_sections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    docs = metadata.copy()
    docs["share_code"] = docs["fundCode"].astype(str).str.zfill(6)
    docs["upload_info_id"] = docs["uploadInfoId"].astype(str)
    docs["known_at"] = pd.to_datetime(docs["reportSendDate"], errors="raise")
    alias_map = fund_aliases.assign(
        query_code=fund_aliases["query_code"].astype(str).str.zfill(6)
    )[["query_code", "family_key", "status", "inception_era"]].drop_duplicates()
    docs = docs.merge(alias_map, left_on="share_code", right_on="query_code", validate="many_to_one")
    docs = docs.sort_values(["family_key", "known_at", "upload_info_id"])
    sections = pd.DataFrame(columns=["upload_info_id", "normalized_clause", "extraction_status"])
    if document_sections is not None:
        sections = document_sections.copy()
        sections["upload_info_id"] = sections["upload_info_id"].astype(str)
    sections = sections.drop(columns=["parser_status", "disposition"], errors="ignore")
    docs = docs.merge(sections, on="upload_info_id", how="left")
    docs["clause_sha256"] = docs["normalized_clause"].fillna("").map(
        lambda value: fingerprint_clause(value) if value else ""
    )
    docs["previous_sha256"] = docs.groupby("family_key")["clause_sha256"].shift()
    rows: list[dict[str, object]] = []
    for family_key, group in docs.groupby("family_key", sort=True):
        first = group.iloc[0]
        rows.append(_checkpoint_row(first, family_key, "INCEPTION", True))
        for _, row in group.iterrows():
            title = f"{row.get('reportName', '')}|{row.get('reportCode', '')}"
            if CRITICAL_TITLE_PATTERN.search(title):
                rows.append(_checkpoint_row(row, family_key, "EXPLICIT_TRANSFORMATION", True))
            if row.get("extraction_status") != "success" or not row.get("clause_sha256"):
                rows.append(_checkpoint_row(row, family_key, "SECTION_UNCOMPARABLE", True))
            elif row.get("previous_sha256") != row.get("clause_sha256"):
                rows.append(_checkpoint_row(row, family_key, "MATERIAL_SECTION_CHANGE", True))
        if str(first["status"]) == "清盘":
            rows.append(_checkpoint_row(group.iloc[-1], family_key, "TERMINAL_STATE", True))
    result = pd.DataFrame(rows).drop_duplicates("checkpoint_id").sort_values(
        ["family_key", "known_at", "checkpoint_id"]
    )
    return result.reset_index(drop=True)


def freeze_checkpoint_manifest(checkpoints: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    if checkpoints["checkpoint_id"].duplicated().any():
        raise ValueError("duplicate checkpoint_id")
    if destination.exists():
        frozen = pd.read_csv(destination, dtype=str)
        current = checkpoints.astype(str)
        old_ids = set(frozen["checkpoint_id"])
        new_ids = set(current["checkpoint_id"])
        if not old_ids.issubset(new_ids):
            raise ValueError("frozen checkpoints cannot be removed")
        comparable = [column for column in frozen.columns if column not in {"manifest_revision", "revision_reason"}]
        old_rows = frozen.set_index("checkpoint_id")[comparable].sort_index()
        new_rows = current.set_index("checkpoint_id").loc[old_rows.index, comparable].sort_index()
        if not old_rows.equals(new_rows):
            raise ValueError("frozen checkpoints cannot be mutated")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    checkpoints.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(destination)
    return destination
```

Implement `_checkpoint_row` in the same module so `checkpoint_id` is the SHA-256 of a pipe-delimited immutable source identity (`family_key`, `share_code`, `upload_info_id`, `trigger_type`, rule version), never row order. The public signatures and columns are frozen by this task.

- [ ] **Step 4: Add append-only manifest tests and make the task GREEN**

Add tests that write revision 1, reject removal/mutation, and accept an appended correction with `manifest_revision=2`, `supersedes_checkpoint_id`, and non-empty `revision_reason`.

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -v
```

Expected: all Task 1–2 tests PASS.

- [ ] **Step 5: Record Gate 2 and commit where Git exists**

Gate 2 passes only if changing parser outcomes leaves checkpoint IDs unchanged, all extraction failures remain in the denominator, and manifest rows cannot be deleted or silently changed.

```powershell
git add v6/classification_checkpoints.py tests/test_v6_classification_checkpoints.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): freeze required classification checkpoints"
```

---

### Task 3: Extract, Cluster, and Adjudicate Evidence

**Files:**
- Modify: `v6/classification_checkpoints.py`
- Modify: `tests/test_v6_classification_checkpoints.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: frozen checkpoints, metadata, legacy `parsed_documents.jsonl` rows, and extracted critical-section rows.
- Produces: `build_clause_clusters(sections: pd.DataFrame) -> pd.DataFrame` and `adjudicate_checkpoints(checkpoints, sections, parsed_documents) -> pd.DataFrame`.
- Disposition output includes `checkpoint_id`, `disposition`, `decision_method`, `evidence_raw`, `evidence_normalized`, `evidence_location`, `clause_sha256`, `normalization_version`, `predecessor_checkpoint_id`, `fund_type`, `equity_min_pct`, `equity_max_pct`, `parser_version`, `reviewer_version`, `failure_reason`.

- [ ] **Step 1: Write RED tests for all three dispositions**

```python
def test_regex_miss_never_becomes_verified_noop():
    result = adjudicate_checkpoints(one_checkpoint(), failed_section(), pd.DataFrame())
    assert result.loc[0, "disposition"] == "UNRESOLVED"
    assert result.loc[0, "failure_reason"] == "SECTION_EXTRACTION_FAILED"


def test_exact_section_match_can_be_verified_noop_with_predecessor_chain():
    result = adjudicate_checkpoints(two_checkpoints(), identical_sections(), parsed_first_state())
    assert result.disposition.tolist() == ["PARSED_STATE", "VERIFIED_NOOP"]
    assert result.loc[1, "predecessor_checkpoint_id"] == result.loc[0, "checkpoint_id"]
    assert result.loc[1, "clause_sha256"] == result.loc[0, "clause_sha256"]
    assert result.loc[1, "evidence_raw"]


def test_changed_numeric_boundary_cannot_inherit_noop():
    result = adjudicate_checkpoints(two_checkpoints(), changed_sections(), parsed_first_state())
    assert result.loc[1, "disposition"] != "VERIFIED_NOOP"
```

- [ ] **Step 2: Run the disposition tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -k "regex_miss or exact_section or numeric_boundary" -v
```

Expected: FAIL because cluster/adjudication functions are absent.

- [ ] **Step 3: Implement clustering and conservative adjudication**

Implement these rules exactly:

```python
def build_clause_clusters(sections: pd.DataFrame) -> pd.DataFrame:
    valid = sections.loc[sections["normalized_clause"].fillna("").ne("")].copy()
    valid["normalization_version"] = NORMALIZATION_VERSION
    valid["clause_sha256"] = valid["normalized_clause"].map(fingerprint_clause)
    valid["cluster_id"] = (
        "n" + valid["normalization_version"].astype(str) + "-" + valid["clause_sha256"]
    )
    valid["cluster_member_count"] = valid.groupby("cluster_id")["cluster_id"].transform("size")
    return valid.sort_values(["cluster_id", "known_at", "upload_info_id"]).reset_index(drop=True)


def adjudicate_checkpoints(
    checkpoints: pd.DataFrame,
    sections: pd.DataFrame,
    parsed_documents: pd.DataFrame,
) -> pd.DataFrame:
    merged = checkpoints.merge(sections, on="upload_info_id", how="left").merge(
        parsed_documents, on="upload_info_id", how="left", suffixes=("", "_parsed")
    )
    decisions: list[dict[str, object]] = []
    previous_by_family: dict[str, dict[str, object]] = {}
    for row in merged.sort_values(["family_key", "known_at", "checkpoint_id"]).to_dict("records"):
        previous = previous_by_family.get(str(row["family_key"]))
        decision = dict(row)
        has_raw = bool(str(row.get("evidence_raw") or "").strip())
        has_state = pd.notna(row.get("equity_min_pct")) and pd.notna(row.get("equity_max_pct"))
        same_clause = bool(previous) and row.get("clause_sha256") == previous.get("clause_sha256")
        if has_raw and has_state:
            decision.update(disposition=Disposition.PARSED_STATE.value, decision_method="EXPLICIT_STATE", failure_reason="")
            previous_by_family[str(row["family_key"])] = decision
        elif has_raw and same_clause:
            decision.update(
                disposition=Disposition.VERIFIED_NOOP.value,
                decision_method="EXACT_CRITICAL_SECTION_MATCH",
                predecessor_checkpoint_id=previous["checkpoint_id"],
                fund_type=previous.get("fund_type"),
                equity_min_pct=previous.get("equity_min_pct"),
                equity_max_pct=previous.get("equity_max_pct"),
                failure_reason="",
            )
            previous_by_family[str(row["family_key"])] = decision
        else:
            reason = "SECTION_EXTRACTION_FAILED" if row.get("extraction_status") != "success" else "POSSIBLE_STATE_CHANGE"
            decision.update(disposition=Disposition.UNRESOLVED.value, decision_method="", failure_reason=reason)
        decisions.append(decision)
    return pd.DataFrame(decisions)
```

Freeze failure reasons as `PDF_MISSING`, `SECTION_EXTRACTION_FAILED`, `SECTION_COMPARISON_FAILED`, `AMBIGUOUS_ASSET_DENOMINATOR`, `POSSIBLE_STATE_CHANGE`, `SOURCE_CONFLICT`, and `MISSING_PREDECESSOR`.

- [ ] **Step 4: Add tests for forbidden evidence and source conflicts**

Add parameterized cases showing that current Tushare type, fund-name inference, later-document evidence, and model confidence without a quoted source all remain `UNRESOLVED`. Add a contradictory official-source case that produces `SOURCE_CONFLICT`.

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Record Gate 3 and commit where Git exists**

Gate 3 passes only when every success has raw evidence and either explicit state fields or an auditable predecessor; every unproven case remains unresolved.

```powershell
git add v6/classification_checkpoints.py tests/test_v6_classification_checkpoints.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): adjudicate checkpoint evidence conservatively"
```

---

### Task 4: Build the Causal State Timeline and Validate Invariants

**Files:**
- Modify: `v6/classification_checkpoints.py`
- Modify: `tests/test_v6_classification_checkpoints.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: disposition rows from Task 3.
- Produces: `build_checkpoint_timeline(dispositions: pd.DataFrame) -> pd.DataFrame` and `validate_checkpoint_invariants(checkpoints, dispositions, timeline) -> dict[str, int]`.

- [ ] **Step 1: Write RED tests for PIT timing, NO-OP carry-forward, and conflicts**

```python
def test_noop_carries_state_forward_only_after_known_at():
    timeline = build_checkpoint_timeline(dispositions_with_noop())
    assert timeline.loc[1, "state_source_checkpoint_id"] == timeline.loc[0, "checkpoint_id"]
    assert timeline.loc[1, "effective_from"] >= timeline.loc[1, "known_at"]


def test_future_document_never_backfills_past_state():
    timeline = build_checkpoint_timeline(dispositions_with_historical_effective_date())
    assert timeline.loc[0, "effective_from"] == pd.Timestamp("2021-12-20")


def test_unresolved_conflict_is_counted_not_voted_away():
    metrics = validate_checkpoint_invariants(
        conflict_checkpoints(), conflict_dispositions(), pd.DataFrame()
    )
    assert metrics["unresolved_conflicts"] == 1
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py -k "carries_state or backfills or conflict_is_counted" -v
```

Expected: FAIL because timeline interfaces are absent.

- [ ] **Step 3: Implement the timeline without backward fill**

For each `family_key`, sort by `known_at`, `effective_date`, then immutable checkpoint ID. For `PARSED_STATE`, set `effective_from=max(known_at,effective_date)`; for `VERIFIED_NOOP`, copy state fields only from its explicit predecessor and retain both checkpoint IDs. Exclude `UNRESOLVED` from usable intervals but retain it in invariant metrics. Derive `effective_to` from the next usable state without producing zero/negative intervals.

- [ ] **Step 4: Implement invariant counters and run regression**

Return exact integer counters: `causal_violations`, `unresolved_conflicts`, `nonpositive_intervals`, `orphan_noops`, `critical_unresolved`. Assert the legacy compatible-intersection behavior remains covered by `tests/test_v6_csrc_collector.py`.

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_classification_checkpoints.py tests/test_v6_csrc_collector.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Record Gate 4 and commit where Git exists**

Gate 4 requires all synthetic valid fixtures to have zero causal violations, unresolved conflicts, nonpositive intervals, orphan NO-OPs, and critical unresolved checkpoints; each injected violation must increment its matching counter.

```powershell
git add v6/classification_checkpoints.py tests/test_v6_classification_checkpoints.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): build causal checkpoint state timeline"
```

---

### Task 5: Implement the Reproducible Amended Gate and Artifacts

**Files:**
- Create: `v6/g0_checkpoint_gate.py`
- Create: `tests/test_v6_g0_checkpoint_gate.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes paths to `metadata.jsonl`, `parsed_documents.jsonl`, PDF root, frozen family sample, alias mapping, and output directory.
- Produces: `run_checkpoint_gate(inputs: GateInputs) -> GateResult`, `write_gate_artifacts(result, out_dir) -> dict[str, Path]`, and module CLI.
- `GateResult` contains `passed: bool`, `metrics: pd.DataFrame`, checkpoints, dispositions, clusters, timeline, and review markdown.

- [ ] **Step 1: Write RED unit tests for every simultaneous threshold**

```python
def test_gate_pass_requires_all_preregistered_conditions():
    result = evaluate_gate(
        checkpoints=checkpoint_fixture(resolved=95, total=100, critical_unresolved=0),
        dispositions=disposition_fixture(resolved=95, total=100),
        timeline=timeline_fixture(overall=0.90, each_stratum=0.80),
        aliases=aliases_six_strata(),
        invariant_counts={"causal_violations": 0, "unresolved_conflicts": 0,
                          "nonpositive_intervals": 0, "orphan_noops": 0,
                          "critical_unresolved": 0},
        legacy_document_counts={"valid": 1993, "success": 1676, "failure": 317},
    )
    assert result.passed is True
    assert result.metric("document_parser_coverage").is_primary is False


@pytest.mark.parametrize("metric", [
    "checkpoint_coverage", "critical_checkpoint_resolution",
    "fund_coverage_overall", "coverage_存续_pre2013",
    "coverage_存续_2013_2019", "coverage_存续_2020_2026",
    "coverage_清盘_pre2013", "coverage_清盘_2013_2019",
    "coverage_清盘_2020_2026", "causal_violations", "unresolved_conflicts",
])
def test_each_primary_condition_can_fail_gate(metric):
    result = valid_gate_fixture().with_failed_metric(metric)
    assert result.passed is False
```

- [ ] **Step 2: Run Gate tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_g0_checkpoint_gate.py -v
```

Expected: collection fails because `v6.g0_checkpoint_gate` does not exist.

- [ ] **Step 3: Implement typed Gate inputs/results and exact metrics**

Use dataclasses:

```python
@dataclass(frozen=True)
class GateInputs:
    metadata_jsonl: Path
    parsed_documents_jsonl: Path
    pdf_root: Path
    family_sample_csv: Path
    aliases_csv: Path
    legacy_parse_failures_csv: Path


@dataclass
class GateResult:
    passed: bool
    metrics: pd.DataFrame
    checkpoints: pd.DataFrame
    dispositions: pd.DataFrame
    clusters: pd.DataFrame
    timeline: pd.DataFrame
    review_markdown: str
```

Metric rows must include `gate`, `metric`, `role`, `threshold`, `numerator`, `denominator`, `actual`, `result`, and `protocol_version`. Use `role=PRIMARY` for amended decision rows and `role=DIAGNOSTIC` for `document_parser_coverage` and PDF engineering diagnostics.

- [ ] **Step 4: Write all six artifacts atomically under the A1 directory**

`write_gate_artifacts` must write:

```text
required_classification_checkpoints.csv
document_dispositions.csv
classification_clause_clusters.csv
classification_state_timeline.csv
g0_amendment_a1_gate_metrics.csv
g0_amendment_a1_gate_review.md
```

Write to a sibling `.tmp`, flush and close, then use `Path.replace`; retry transient Windows `PermissionError` using the existing bounded retry pattern. Never write legacy `g0_gate_metrics_v17.csv` or `classification_timeline.csv`.

- [ ] **Step 5: Add CLI integration test and implementation**

CLI:

```powershell
python -m v6.g0_checkpoint_gate `
  --metadata output/v6/g0_full_pit/csrc_stratified/metadata.jsonl `
  --parsed-documents output/v6/g0_full_pit/csrc_stratified/parsed_documents.jsonl `
  --pdf-root output/v6/g0_full_pit/csrc_stratified/pdf `
  --family-sample output/v6/g0_full_pit/csrc_stratified_sample.csv `
  --aliases output/v6/g0_full_pit/csrc_stratified_aliases.csv `
  --legacy-failures output/v6/g0_full_pit/csrc_stratified/classification_parse_failures.csv `
  --out output/v6/g0_full_pit/csrc_stratified/a1
```

The integration test uses `tmp_path`, verifies all six files, reloads them, checks leading-zero codes, confirms `protocol_version=V6-G0-A1`, and asserts the process exit code is 0 only on PASS and 2 on Gate FAIL.

- [ ] **Step 6: Run Gate suite plus all V6 tests**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_g0_checkpoint_gate.py -v
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_*.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Record Gate 5 and commit where Git exists**

Gate 5 requires threshold-boundary tests, all six strata, diagnostic continuity, atomic outputs, and CLI exit semantics to pass.

```powershell
git add v6/g0_checkpoint_gate.py tests/test_v6_g0_checkpoint_gate.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): add preregistered G0 checkpoint gate"
```

---

### Task 6: Add the Narrow Collector Orchestration Hook

**Files:**
- Modify: `v6/csrc_fund_collector.py`
- Modify: `tests/test_v6_csrc_collector.py`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: existing collector output directory and optional `--checkpoint-gate` plus required sample/alias paths.
- Produces: the same legacy artifact dictionary by default; when explicitly enabled, adds six `a1_*` paths without renaming legacy keys.

- [ ] **Step 1: Write RED tests for opt-in behavior and legacy stability**

```python
def test_default_parse_only_does_not_create_a1_artifacts(tmp_path, monkeypatch):
    artifacts = collect_fund_disclosures(
        ["000001"], ["FA"], tmp_path, parse_only=True
    )
    assert "a1_gate_metrics" not in artifacts
    assert not (tmp_path / "a1").exists()


def test_checkpoint_gate_is_explicit_and_preserves_legacy_paths(tmp_path, monkeypatch):
    artifacts = collect_fund_disclosures(
        ["000001"], ["FA"], tmp_path, parse_only=True,
        checkpoint_gate=True,
        family_sample_path=sample_csv(tmp_path),
        aliases_path=aliases_csv(tmp_path),
    )
    assert artifacts["classification_timeline"] == tmp_path / "classification_timeline.csv"
    assert artifacts["a1_gate_metrics"] == tmp_path / "a1/g0_amendment_a1_gate_metrics.csv"
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_csrc_collector.py -k "checkpoint_gate or default_parse_only" -v
```

Expected: FAIL because new keyword arguments are not accepted.

- [ ] **Step 3: Add the opt-in arguments and delegate to the Gate module**

Extend `collect_fund_disclosures` with keyword-only `checkpoint_gate: bool = False`, `family_sample_path: str | os.PathLike[str] | None = None`, and `aliases_path: str | os.PathLike[str] | None = None`. Reject enabled mode when either path is missing. Add mutually compatible CLI flags `--checkpoint-gate`, `--family-sample`, and `--aliases`; retain the existing `--metadata-only`/`--parse-only` mutual exclusion.

- [ ] **Step 4: Run collector and complete V6 regression**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_csrc_collector.py -q
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_v6_*.py -q
```

Expected: all tests PASS and legacy parse-only tests remain unchanged.

- [ ] **Step 5: Record Gate 6 and commit where Git exists**

Gate 6 passes only if default behavior is byte-path compatible, A1 generation is explicit, and missing prerequisites fail before writing partial A1 output.

```powershell
git add v6/csrc_fund_collector.py tests/test_v6_csrc_collector.py docs/V6_实验协议与执行台账.md
git commit -m "feat(v6): wire opt-in checkpoint gate into collector"
```

---

### Task 7: Freeze the Real Denominator Before Adjudication

**Files:**
- Generate: `output/v6/g0_full_pit/csrc_stratified/a1/required_classification_checkpoints.csv`
- Generate: `output/v6/g0_full_pit/csrc_stratified/a1/checkpoint_manifest_v1.sha256`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: existing 1,067 metadata jobs, 2,807 valid local PDFs, 60-family frozen sample and alias mapping.
- Produces: immutable amendment denominator and its SHA-256 before any real-data disposition is calculated.

- [ ] **Step 1: Run denominator-only mode**

Add and use `--freeze-checkpoints-only`, which may extract/compare critical sections only to identify mechanical change/uncomparable triggers but must not load `parsed_documents.jsonl` or emit dispositions.

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m v6.g0_checkpoint_gate `
  --metadata output/v6/g0_full_pit/csrc_stratified/metadata.jsonl `
  --pdf-root output/v6/g0_full_pit/csrc_stratified/pdf `
  --family-sample output/v6/g0_full_pit/csrc_stratified_sample.csv `
  --aliases output/v6/g0_full_pit/csrc_stratified_aliases.csv `
  --out output/v6/g0_full_pit/csrc_stratified/a1 `
  --freeze-checkpoints-only
```

- [ ] **Step 2: Audit the frozen denominator mechanically**

Run checks that checkpoint IDs are unique; every sampled family appears; every explicit transition/merger/liquidation title produces a critical row; every missing/unreadable comparison produces `SECTION_UNCOMPARABLE`; no parser status column is present.

- [ ] **Step 3: Freeze and record the denominator hash**

```powershell
Get-FileHash -Algorithm SHA256 output/v6/g0_full_pit/csrc_stratified/a1/required_classification_checkpoints.csv
```

Write the digest, row count, trigger counts, critical count, generation-rule version, normalization version, command, and timestamp into `checkpoint_manifest_v1.sha256` and the ledger. After this step, corrections must append a revision and never replace v1.

- [ ] **Step 4: Stop for Gate 7 review**

Gate 7 is a mandatory review pause. Do not run real-data adjudication until the denominator audit passes and the frozen hash is recorded.

- [ ] **Step 5: Commit where Git exists**

```powershell
git add output/v6/g0_full_pit/csrc_stratified/a1/required_classification_checkpoints.csv output/v6/g0_full_pit/csrc_stratified/a1/checkpoint_manifest_v1.sha256 docs/V6_实验协议与执行台账.md
git commit -m "data(v6): freeze G0 A1 checkpoint denominator"
```

---

### Task 8: Run Real Adjudication, Resolve Residual Clusters, and Recheck G0

**Files:**
- Generate: all six files under `output/v6/g0_full_pit/csrc_stratified/a1/`
- Create if residual review is needed: `data/v6/official/checkpoint_adjudication_overrides.csv`
- Modify: `docs/V6_实验协议与执行台账.md`

**Interfaces:**
- Consumes: frozen denominator hash from Task 7, existing parser cache, local PDFs, and evidence-preserving human/model review rows if required.
- Produces: the first official V6-G0-A1 Gate decision.

- [ ] **Step 1: Run deterministic clustering and adjudication**

Run the Task 5 CLI against the frozen denominator. It must verify the denominator hash before loading parser results. If the hash differs, exit nonzero without producing a Gate review.

- [ ] **Step 2: Review residuals by unique evidence cluster, not document count**

Sort unresolved rows by `is_critical DESC`, affected family count DESC, cluster member count DESC. Use PaddleOCR only for unreadable/image PDFs. Any manual/model resolution must be appended to `checkpoint_adjudication_overrides.csv` with `checkpoint_id`, exact quoted evidence, evidence location, decision, state fields or predecessor, reviewer, reviewed_at, and source-document checksum. Reject rows missing quoted evidence.

- [ ] **Step 3: Re-run after each bounded residual batch**

Each batch must target a closed set of unique clusters. After a batch, rerun unit tests, rebuild dispositions/timeline/Gate, append before/after counts to the ledger, and stop if the next batch lacks a mechanically auditable decision rule.

- [ ] **Step 4: Execute the final amended Gate**

Verify and report all conditions separately:

```text
checkpoint coverage >= 95%
critical checkpoint resolution = 100%
fund coverage overall >= 90%
each six-stratum coverage >= 80%
causal violations = 0
unresolved conflicts = 0
legacy document diagnostic = 1676 / 1993 = 84.14% unless independently changed by a versioned parser run
```

If any primary condition fails, write `FINAL: G0 FAIL` and keep G1–G4 frozen. If all pass, write `FINAL: G0 PASS under V6-G0-A1`, preserve the old v17 FAIL artifacts, and stop for user review before starting G1.

- [ ] **Step 5: Run the full verification suite**

Run:

```powershell
& 'C:\Users\10941\AppData\Local\Programs\Python\Python311\python.exe' -m pytest -q
```

Expected: all tests PASS. Then verify every output reloads, required IDs are unique, disposition counts reconcile to denominator rows, and timeline families reconcile to Gate numerators.

- [ ] **Step 6: Record Gate 8 and commit where Git exists**

Append exact metrics, artifact paths, hashes, test output, changed files, and PASS/FAIL decision to the ledger.

```powershell
git add v6 tests data/v6/official/checkpoint_adjudication_overrides.csv output/v6/g0_full_pit/csrc_stratified/a1 docs/V6_实验协议与执行台账.md
git commit -m "data(v6): adjudicate and decide G0 under amendment A1"
```

Do not add the overrides path if no override file was needed.

---

## Execution Stop Conditions

- Stop immediately after each numbered Gate for independent review and ledger entry.
- Stop before adjudication if the denominator was not frozen independently of parser outcomes.
- Stop before accepting any `VERIFIED_NOOP` without raw evidence, normalized evidence, matching predecessor, and decision method.
- Stop and keep G0 FAIL if any critical checkpoint remains unresolved, any causal violation or unresolved conflict is nonzero, or fund/stratum coverage misses its threshold.
- Stop after the first official V6-G0-A1 decision; do not automatically enter G1 even on PASS.
