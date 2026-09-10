# Protocol Amendment V6-G0-A2 — Observable Legal-State Gate

## 1. Status and purpose

- Amendment: `V6-G0-A2`
- Status: design revision 2 approved in principle; implementation planning is authorized, but denominator generation and adjudication are not.
- Purpose: determine whether the historical legal classification and equity-allocation state required by the V6 FULL-PIT universe is observable at each downstream decision date.
- Downstream status: G1–G4 remain frozen until A2 is implemented, independently validated, and passes.

A2 is a diagnostic-driven successor protocol, not a correction to A1. The first official A1 adjudication remains immutable:

`a1/v2 = FINAL G0 FAIL`, 1,371 checkpoints, SHA-256 `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd`.

The reason for A2 is structural. A1 marked every generated checkpoint critical, including documents whose investment section was absent. This turned routine sales, subscription, redemption, fee, dividend, and transaction-conversion notices into mandatory legal-state events. Such documents normally contain neither a new state nor affirmative unchanged evidence, so resolving them is not equivalent to restoring a PIT state.

## 2. Scientific question

A2 asks:

> At every fund/date observation that V6 may use, can the system recover a legal fund type and equity-allocation state from official evidence that was already public and legally applicable on that date?

The measurement unit is therefore an observable state interval used by a decision date, not an arbitrary disclosure document.

## 3. Core objects

### 3.1 Legal-state event

A `LEGAL_STATE_EVENT` is an event capable of creating, changing, inheriting, or terminating the fund's classification or permitted equity-allocation range. The closed event taxonomy is:

1. `INCEPTION`
2. `LEGAL_TRANSFORMATION`
3. `MERGER_OR_SUCCESSION`
4. `FUND_TYPE_CHANGE`
5. `INVESTMENT_MANDATE_CHANGE`
6. `TERMINATION`

Routine operations are not legal-state events: sales-channel changes, fee discounts, dividend notices, purchase/redemption limits, ordinary conversion services, manager appointments, NAV announcements, and periodic portfolio reports.

### 3.2 State evidence

A state requires official evidence for:

- legal fund type;
- equity allocation lower and upper bounds, including an explicit open bound where legally applicable;
- `known_at`, the first date the evidence was publicly available;
- `legal_effective_from`, when the state legally applies;
- exact source document and evidence location;
- raw clause, normalized clause, and parser/reviewer provenance.

Silence in an operational document is never state evidence and never proves NOOP.

### 3.3 Observable state interval

For a state supported by evidence `e`:

`usable_from = max(e.known_at, e.legal_effective_from)`

The state is queryable on `[usable_from, next_usable_from)` unless a termination or unresolved conflicting legal event closes it earlier. A retrospective legal date is recorded but cannot backfill the period before `known_at`.

### 3.4 Decision-date observation

A `DECISION_OBSERVATION` is a `(family_key, decision_date)` pair produced by the frozen downstream V6 schedule. Active-eligibility begins at the authoritative inception date and ends before an authoritative termination date. The freeze retains the termination boundary and its reason, but post-termination dates are not investable observations and are not included in the active coverage denominator. Each active-eligibility pair returns exactly one of:

- `OBSERVED_STATE`
- `PRE_OBSERVABLE` — no official causal state is available yet;
- `UNRESOLVED_EVENT_CONFLICT`

`PRE_OBSERVABLE` and `UNRESOLVED_EVENT_CONFLICT` observations cannot enter the investable universe. They remain in the active coverage denominator and may not be backfilled. `TERMINATED` is recorded at the boundary but is not counted as a missing active observation.

## 4. Event generation

Event candidates must be generated before reading parsed state values. Candidate generation may use only official metadata, frozen fund-master lifecycle fields, authoritative alias/family mappings, and document title/type semantics.

### 4.1 Positive event rules

- `INCEPTION`: one per sampled legal family, using the authoritative establishment/effective date.
- `TERMINATION`: one per terminated family when an authoritative termination, liquidation, merger, or delisting date is known; unknown dates remain explicitly unresolved.
- Other events require both:
  1. a positive legal-change expression tied to the current fund; and
  2. an authoritative document class or text stage capable of making the change effective.

Qualifying evidence includes an effective fund contract/prospectus, holder-meeting resolution and effectiveness announcement, regulator approval/effectiveness notice, or a transformation announcement that identifies the succeeding legal state and effective date.

### 4.2 Explicit false-positive exclusions

The token `转换` alone is insufficient. The following do not create `LEGAL_TRANSFORMATION`:

- 申购/赎回/转换转入/转换转出;
- 开通或暂停基金转换业务;
- conversion fee or sales-channel notices;
- share-class conversion that does not change the legal mandate;
- narrative references to a historical transformation without a current effective change.

A legal transformation requires semantics such as `转型为`, `变更为`, `保本周期到期后转为`, closed-end-to-open-end legal conversion, merger/succession, or an equivalent explicit change, plus the applicable document stage.

### 4.3 Deterministic identity and deduplication

Event identity is derived from:

`event_id = SHA256(family_key | event_type | legal_subject | event_anchor_date | A2_generation_version)`

`event_anchor_date` is the authoritative announced effective date when present; otherwise it is the official event announcement date. Its selection rule and provenance are frozen before adjudication. `canonical_source_document` is expressly excluded from `event_id`.

Duplicate share codes or duplicate metadata rows referencing the same canonical official PDF create one evidence candidate. Multiple documents for the same event remain evidence candidates attached by `event_id`; they never create separate events merely because their files differ. The relationship is strictly `LEGAL_STATE_EVENT 1:N EVENT_EVIDENCE_CANDIDATE`. A source document exists only in the evidence-candidate layer.

## 5. Evidence resolution

Resolution follows:

`event -> authoritative family/alias -> official metadata -> governing document -> section -> clause -> state -> interval`

### 5.1 Source priority

Priority is legal, not merely chronological:

1. effective fund contract or regulator/holder-resolution effectiveness document;
2. prospectus effective for the event;
3. transformation/merger announcement containing the operative state;
4. later official document only as a locator or corroboration, never as hindsight state evidence.

Source priority may select operative evidence only when all same-event candidate evidence is legally compatible. It must never silently override an incompatible state. Disjoint constraints such as 60%–95% versus 0%–30%, incompatible legal fund types, or irreconcilable effective dates produce `UNRESOLVED_EVENT_CONFLICT` regardless of source rank. Compatible constraints may be intersected only when the resulting interval is non-empty and the method and contributing sources are retained.

An announcement that points to a separately published contract/prospectus triggers retrieval of that governing document. The pointer itself does not supply the missing allocation clause.

### 5.2 State decisions

Each event receives exactly one result:

- `NEW_STATE`: current official evidence directly establishes the state;
- `VERIFIED_CONTINUITY`: affirmative unchanged evidence plus a causally valid predecessor establishes continuity;
- `TERMINATED`: authoritative evidence closes the state;
- `UNRESOLVED`: evidence missing, ambiguous, conflicting, not yet effective, or not locatable.

`DOCUMENT_NOT_APPLICABLE` and `PROPOSED_STATE_NOT_EFFECTIVE` can only explain why a document is unusable; neither implies `VERIFIED_CONTINUITY`.

## 6. Development and validation isolation

The existing six-stratum 24-development/36-validation family split remains frozen for section, clause, timeline and end-to-end family validation. During the A1 necessity audit, title metadata for all 49 transformation-labelled rows—including 14 rows from the 36-family side—was inspected. Therefore the 36 families must not be described as untouched for A2 title-event semantics.

- New title semantics, document-stage rules, alias logic, and extraction fallbacks may be discovered only from development families.
- A separate external event-title validation cohort is frozen before its titles are manually reviewed. Eligible rows are official metadata outside all 60 sampled families, published no later than 2026-03-31, whose title contains at least one broad retrieval token from `{转型, 转换, 合并, 变更, 到期, 终止}`. Sort by `SHA256("A2-EVENT-OOS-20260909" | canonical_source_document)`, keep at most three documents per share code, and select the first 120 rows. Freeze IDs, titles and source metadata before labels are added.
- The 120 external rows are manually labelled once as a closed-taxonomy event type, `NOT_EVENT`, or `REVIEW_REQUIRED`; they may measure event-rule generalization but cannot create new rules in the same revision.
- Failure on either validation cohort reopens development only after recording the validation result and freezing a new rule revision. Validation examples may not be copied into one-off family exceptions.
- The final formal Gate still covers all 60 families.

## 7. Frozen state-to-universe mapping

Before any A2 result is viewed, implementation must expose and freeze the deterministic function `state_to_universe_eligibility_v1(legal_fund_type, equity_min_pct, equity_max_pct)`. It returns `eligibility_status`, `target_type`, and `exclusion_reason`.

The mapping is:

| Legal state | Additional bound rule | Result |
|---|---|---|
| explicit non-index `股票型` | `equity_min_pct >= 60` | eligible as `股票型` |
| explicit domestic equity-index mandate | `equity_min_pct >= 60` | eligible as `指数型-股票` |
| explicit overseas/global equity-index mandate | `equity_min_pct >= 60` | eligible as `指数型-海外股票` |
| `混合型` | `equity_min_pct >= 60` | eligible as `混合型-偏股` |
| explicit legal subtype `混合型-灵活` / `灵活配置混合型` | `equity_max_pct > 30`; no minimum required | eligible as `混合型-灵活` |
| any mixed state | `equity_max_pct <= 30` | excluded as `混合型-偏债` |
| bond, money-market, FOF, commodity, REIT, capital-protected, balanced, stable-return, or other non-target mandate | any | excluded with its explicit legal reason |
| missing/ambiguous type or bound needed by a rule | — | `UNRESOLVED_ELIGIBILITY`; never eligible |

Rule precedence is: explicit non-target mandate exclusion; missing/invalid state; mixed upper-bound exclusion; then positive target rules. Bounds must satisfy `0 <= lower <= upper <= 100`. Names, current fund-master labels, realized holdings, benchmark composition, returns, or downstream coverage cannot override the legal state. The target type set is frozen as `{股票型, 指数型-股票, 指数型-海外股票, 混合型-偏股, 混合型-灵活}`.

Tests must pin every boundary: 59.999/60 for minimum rules and 30/30.001 for flexible-versus-bond-biased treatment, plus missing values, invalid intervals, index geography, and explicit excluded mandates.

## 8. Frozen monthly decision schedule

The A2 decision-observation denominator uses every monthly decision date consumed by the V6 training or prediction panel, not only quarterly rebalance dates.

- Range: January 2006 through March 2026, inclusive.
- Calendar rule: for each calendar month, use the last trading day present in the frozen canonical trading calendar.
- Expected count: 243 monthly decision dates.
- A monthly date remains in the denominator even when the cross-section is thin or no trade is executed.
- Quarterly rebalance dates are a strict execution subset and do not create an alternative G0 denominator.
- The manifest freezes the calendar source, calendar hash, start/end dates, ordered 243 dates, and schedule-rule version before event/state adjudication.

This schedule is used to generate all active-eligibility `(family_key, decision_date)` pairs before state results are read.

## 9. Frozen denominator and anti-gaming rules

A2 uses two denominators, frozen before state adjudication:

1. all generated `LEGAL_STATE_EVENT` checkpoints;
2. all reachable `DECISION_OBSERVATION` pairs produced by the frozen downstream schedule and the frozen sample.

The freeze manifest records rules, code/test hashes, input hashes, event counts, decision-date counts, family/stratum counts, exclusions by reason, and the exact command. After freezing:

- event IDs, decision dates, sample families, strata, thresholds, and state definitions cannot change within that adjudication version;
- corrections create an append-only A2 denominator revision and a new adjudication version;
- no failed event or observation may be deleted to improve coverage;
- A1 artifacts and A1 failure remain untouched.

## 10. A2 Gate

A2 passes only if all conditions hold:

1. `LEGAL_STATE_EVENT` resolution >= 95% overall;
2. `INCEPTION`, genuine legal transformation/merger/type/mandate change, and dated termination events are 100% resolved;
3. at least one observable state interval exists for >=90% of the 60 families;
4. family coverage in each of the six frozen strata is >=80%;
5. `OBSERVED_STATE / all frozen active-eligibility DECISION_OBSERVATION pairs >= 95%`;
6. causal violations = 0;
7. nonpositive intervals = 0;
8. unresolved same-event state conflicts = 0;
9. no observation uses evidence with `known_at > decision_date` or `legal_effective_from > decision_date`;
10. every unavailable active family/date remains explicit as `PRE_OBSERVABLE` or `UNRESOLVED_EVENT_CONFLICT`, and every termination boundary remains explicit as `TERMINATED`.

Condition 2 prevents frequent ordinary events from hiding missing lifecycle states. Condition 5 measures the actual data surface consumed by the backtest. A family may begin contributing only from its first observable causal state; earlier dates remain counted as unavailable and are never backfilled.

## 11. Required outputs

Each formal A2 adjudication version writes, atomically and append-only:

1. `legal_state_events.csv`
2. `event_evidence_candidates.csv`
3. `event_dispositions.csv`
4. `legal_state_timeline.csv`
5. `decision_observation_coverage.csv`
6. `a2_gate_metrics.csv`
7. `a2_gate_review.md`
8. `a2_manifest.sha256`

The report includes the A1-to-A2 trigger audit, especially false `转换` triggers and operational `SECTION_UNCOMPARABLE` documents, but never rewrites A1 dispositions.

## 12. Verification requirements

Before the denominator freeze:

- unit tests cover every positive event type and every explicit false-positive exclusion;
- adversarial tests distinguish legal transformation from transaction conversion;
- duplicate canonical PDFs and share aliases collapse deterministically;
- lifecycle evidence cannot cross families or move backward from future knowledge;
- decision-date queries enforce both publication and legal-effectiveness dates;
- `state_to_universe_eligibility_v1` passes all exact boundary and exclusion tests;
- the frozen monthly schedule contains exactly the expected ordered 243 last-trading-day dates;
- one event may own multiple evidence documents without changing `event_id`;
- incompatible same-event evidence remains unresolved regardless of source priority;
- development/validation membership is asserted at runtime.

Before formal adjudication:

- all deterministic test groups have explicit exit 0 and machine-readable summaries;
- code, tests, inputs, split, denominator, and runner hashes are frozen;
- an independent specification review and an independent quality review both approve;
- no G1–G4 metric has been read or used to modify A2.

## 13. Stop conditions

- If untouched validation shows that genuine event detection does not generalize, stop and revise the event model before freezing the denominator.
- If official archives cannot provide inception or genuine transformation evidence at material scale, stop with A2 FAIL and report the pre-observable historical region; do not infer or backfill state.
- If any Gate condition fails, G1–G4 remain frozen.
- Even if A2 passes, stop for user review before starting G1.

## 14. Non-goals

A2 does not attempt to classify every disclosure, maximize parser coverage, infer unchanged state from silence, repair missing official archives with third-party claims, alter the 60-family sample, tune downstream returns, or erase the diagnostic value of A1 FAIL.
