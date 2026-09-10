# A1 structured section pipeline — implementation report

Date: 2026-09-09. Scope: the document/section/clause front half used by the
A1 denominator-section extractor. No adjudication was run and no `a1/v2`
artifact, checkpoint definition, threshold, NOOP rule, or parser classification
regex was changed.

## Implemented boundary

- `assess_document_applicability` separates obvious operational notices from
  documents whose title or substantive text can define an investment state.
  Unknown cases remain `UNDETERMINED`; they are not silently discarded.
- `extract_investment_sections` removes standalone PDF page headers, respects
  page breaks, recognizes the generic legal headings 投资目标、投资范围、投资策略、
  资产配置、投资限制, and returns bounded candidates with heading and start page.
- The Gate now evaluates `Document -> Section -> Clause`. It records
  `document_applicability`, `section_heading`, `section_page_start`,
  `section_raw`, and an exclusive `root_cause_reason` while retaining the
  preregistered `failure_reason=SECTION_EXTRACTION_FAILED` boundary for A1
  compatibility.
- Successful evidence retains canonical source PDF for the existing strict
  parser join, while `section_locator` separately records page and heading;
  bounded raw and normalized section text are separate fields. Documents without
  headings retain the prior direct-clause fallback only when an explicit equity
  constraint is actually parsable, and its page is located from the evidence
  instead of being assumed to be page 1.
- Multiple section candidates with different parsed equity states fail closed as
  `EQUITY_CLAUSE_AMBIGUOUS`. Combined stock-and-bond constraints are not treated
  as stock allocation evidence.
- Section cache extractor version advanced to `structured-section-v3`, so old
  cache entries cannot masquerade as structured results.

The exclusive diagnostic stages introduced in this increment are:

1. `DOCUMENT_NOT_APPLICABLE`
2. `INVESTMENT_SECTION_NOT_FOUND`
3. `TEXT_EXTRACTION_DAMAGED`
4. `EQUITY_CLAUSE_NOT_FOUND`
5. `EQUITY_CLAUSE_AMBIGUOUS`
6. `PROPOSED_STATE_NOT_EFFECTIVE`
7. `EVIDENCE_LOCATION_UNRESOLVED`

Pending proposal language such as a contract change that still requires a vote
before becoming effective is rejected before clause adjudication. Common section
forms including `第八部分 投资范围`, `八、投资范围`, `（八）投资范围`, and
`8. 投资范围` share the same structural path.

The pending test is locally bound to `本次`/`本议案`/proposal-title semantics;
generic governance language in an already-effective contract is not treated as
a pending state. Explicit operational titles remain non-state-changing when they
merely repeat product terms, unless the title or locally bound text identifies a
contract/transition change. For phase-structured clauses, post-conversion text is
selected before the within-phase ambiguity check, preserving the existing
closed-period-to-LOF semantics.

## TDD and regression evidence

- Initial focused collection failed because the new pipeline API did not exist:
  genuine RED.
- Initial new behavior tests: 4 passed.
- First review rejected four specification gaps and quality review rejected
  three additional gaps. New counterexamples were RED before the fix: same-section
  state ambiguity, blank/damaged extraction, raw-vs-normalized provenance,
  operational-title evidence precedence, canonical parser join, pending proposals,
  common heading formats, and fallback page location.
- Fix-round focused behavior tests: 10 passed.
- Full disclosure parser plus classification checkpoint modules: 136 passed.
- Gate lifecycle/provenance/cache focused regression: 14 passed.
- Structured-section to existing-parser provenance end-to-end: 9 passed in its
  focused group (including the lifecycle builder controls).
- Second quality review found three additional blocking counterexamples:
  generic contract-governance false pending, operational notices carrying
  repeated product terms, and legal closed-period/post-conversion ranges. All
  three were RED, then fixed locally.
- Final cumulative pipeline counterexamples: 13 passed.
- Final full disclosure plus classification modules: 138 passed.
- Final Gate provenance/cache/pipeline focused regression: 15 passed.
- A further quality pass found two boundary cases before approval: a proposal
  split across adjacent sentences, and a later open-period cash clause appearing
  after the post-conversion equity clause. Both were reproduced RED. Proposal
  windows now allow bounded cross-sentence references only when anchored by
  `本议案`/`本次`/`上述事项` or a meeting-call title. Phase selection now binds
  each parsed equity constraint to its nearest preceding phase marker; a later
  non-equity marker cannot truncate an earlier current equity clause.
- Final disclosure plus classification modules after this fix: 139 passed.
- Final Gate focused regression remains 15 passed.
- The next quality pass tightened pending semantics once more. A generic `本次`
  update can no longer bind to a distant contract-governance sentence: the local
  window also requires a proposal/change/review action. Conversely, a title that
  explicitly says `草案`/`拟修改`/`拟变更`/`拟修订` is pending by default unless
  the same material states both approval and a concrete effective date. These
  counterexamples were RED, then GREEN (3 focused passed); the final related
  module counts remain 139 passed and 15 passed respectively.
- Final review also required the release from proposal status to be causal, not
  a document-wide conjunction. Proposal-title defaults now include `议案` and
  `征求意见稿`. They are released only when one bounded chain anchored to the
  current proposal states affirmative approval and a concrete effective date;
  unrelated historical approvals/dates, `未经`, and conditional `若/如/拟`
  effectiveness cannot release a new draft. The complete pending positive and
  negative fixture is green; final module counts remain 139 and 15.
- The final conditional-scope counterexamples place `若/如` both immediately
  before the current-proposal anchor and between the anchor and approval. The
  entire bounded resolution chain, plus its immediately preceding character,
  is now checked for conditional mood; only the unconditional control releases
  pending status. The pending fixture and the 139-module/15-Gate regressions pass.
- A fresh independent QA still reproduced cross-sentence referent leakage, so
  the release and pending logic was simplified from whole-document windows to
  sentence-scoped current-item chains. Only the immediately following sentence
  may join, and only when it begins with `上述事项`. This prevents a current draft
  from borrowing an old proposal's approval/date and prevents a routine update
  from binding to a later generic governance rule. `拟经`, `未经`, and conditional
  forms remain pending. The expanded fixture is green; the 139-module and
  15-Gate regression counts remain green.
- The last document-wide pending fallback was removed after a review showed that
  it could reclassify an already approved current item from an unrelated later
  governance sentence. A meeting-call title is now pending by default unless the
  same sentence-scoped current-item chain proves approval and effective date.
  The new approved-current-item plus unrelated-later-rule counterexample passes;
  the disclosure/classification suite remains 139 passed.
- Final proposal release is deliberately whitelist-based: only a result/effective
  announcement title plus an unconditional current-item approval/date chain can
  release a proposal title. `拟提交`, `拟经`, `没有经`, old proposals/contracts,
  and dates marked for later notice all fail closed. Routine updates containing
  an explicit `不涉及变更` statement cannot bind to a generic governance clause;
  the redundant whole-sentence pending shortcut was removed. The expanded
  fixture and final 139-module/15-Gate regressions pass.
- Result-announcement titles are also fail closed when they mention the current
  item but do not satisfy the affirmative release chain. This covers rejected
  (`未经`) and conditional result notices even when the title contains none of
  the proposal keywords. A full synthetic Section pipeline test proves that such
  a notice emits `PROPOSED_STATE_NOT_EFFECTIVE`, not comparable evidence. Final
  evidence: 139 disclosure/classification tests and 16 Gate-focused tests pass.
- Python syntax compilation passed.
- A combined full Gate-module attempt again entered the known Windows
  spawn-heavy path and did not return a final pytest summary. It produced no
  assertion failure and is not recorded as a PASS. Directly affected
  deterministic Gate tests are included in the 10-passed run above.

## Frozen A1 v2 integrity

| Artifact | SHA-256 |
|---|---|
| `required_classification_checkpoints.csv` | `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd` |
| `document_dispositions.csv` | `61c01da72082769382562071e2e19de636a25f5b6aeb662c5d6b6a806ecd5991` |
| `classification_clause_clusters.csv` | `d6a43c1a95c3c189ca106a3e42d80b9f51df7bd6be7ed62af13105bf2c8f362a` |
| `classification_state_timeline.csv` | `fc2c1ea87565e1424d6bde608c5e5842f8c1d8086ce87e3086738ada316aa7f4` |
| `g0_amendment_a1_gate_metrics.csv` | `f2e71aea7ec74db1893f34275f18f425e68ef56c624c36b07d79c3252adf7074` |
| `g0_amendment_a1_gate_review.md` | `a026b99d58d44a52a5ba774977f0f81bd9301626f0018744fef26f31082fa5b4` |

The first official A1 decision remains the immutable v2 G0 FAIL. This increment
does not support a new formal adjudication by itself.
