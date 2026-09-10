# V6 A1 remaining-work necessity assessment

## Decision

Do **not** continue the planned undifferentiated remediation of all 1,251 A1 unresolved checkpoints. It is not aligned with the actual FULL-PIT requirement and cannot reliably distinguish evidence recovery from satisfying false checkpoints.

Continue only work that establishes the historical state at fund inception and at genuine legal state-change events. Preserve A1/v2 as an immutable FAIL and treat the current evidence as grounds to design a separate A2 observability protocol before any further formal adjudication.

## Why the remaining A1 workload is structurally problematic

1. All 1,371 frozen A1 checkpoints have `is_critical=True`. The nominal 95% coverage threshold is therefore dominated by the stricter rule that every critical checkpoint must resolve: operationally A1 requires 1,371/1,371.
2. `_candidate()` writes `is_critical=True` for every generated row. When a document section is empty or uncomparable, the generator creates `SECTION_UNCOMPARABLE`; consequently ordinary operational documents become mandatory critical state events.
3. In the sealed development trace, 301/318 `INVESTMENT_SECTION_NOT_FOUND` checkpoints contain no investment-structure term. The other 17 contain only narrative references, governance text, or pointers to separate legal documents. None contains a missed standalone governing investment section.
4. The 49 frozen `TRANSFORMATION` checkpoints are contaminated by lexical over-triggering: `_event_trigger()` treats any title containing “转换” as a transformation. Title-level inspection identifies at least 33 definite subscription/redemption or fund-transaction conversion notices, 9 plausible legal transformations, and 7 requiring review; five of the seven are also ordinary daily subscription/conversion notices, while two concern legacy fund-share conversion and need legal-context review.
5. A1 `VERIFIED_NOOP` requires affirmative unchanged evidence plus a causal predecessor. Ordinary sales and transaction notices normally provide neither. No extractor improvement can legitimately turn their silence into NOOP.

## What the project actually needs

The FULL-PIT universe needs a causally usable state at:

- fund inception or the first investable decision date;
- genuine legal transformations, mergers, type changes, and investment-range changes;
- the last usable state before termination;
- each downstream portfolio decision date, using only evidence already public and effective.

It does not need every sales-channel, fee-discount, subscription-limit, dividend, or routine operating announcement to be independently adjudicated as a critical classification checkpoint.

## Necessary versus unnecessary remaining work

| Work item | Decision | Reason |
|---|---|---|
| Recover inception governing evidence | Necessary | Without an initial state, a family cannot enter a causal PIT universe. |
| Audit genuine legal transformations | Necessary | These events can change eligibility and are a primary survivor/PIT risk. |
| Resolve timeline causal/nonpositive invariants | Necessary and already completed in shadow | A usable PIT state cannot contain hindsight or invalid intervals. |
| Continue generic section regex iteration | Stop for now | Development evidence found no missed governing sections in the linked documents. |
| Resolve every operational `SECTION_UNCOMPARABLE` row | Unnecessary for the research system | These rows usually do not represent state events and cannot support affirmative NOOP. |
| Run A1/v3 now | Not justified | A1 remains structurally contaminated; a new score would not answer the scientific question. |
| Define an A2 observable-state protocol | Necessary before further adjudication | It must separate true legal state events from operational-document noise without rewriting A1 history. |

## Recommended next design

Create V6-G0-A2 as a new, explicitly diagnostic-driven amendment. Preserve A1/v2 and its FAIL. A2 should generate checkpoints only from lifecycle boundaries and positively identified legal state-change events, require a state to be available at every downstream decision date, and retain ordinary documents as provenance/search inputs rather than automatic critical checkpoints. Event detection must distinguish legal transformation from transaction “conversion” using document type, title context, and authoritative effective-date evidence.

No A2 code, denominator, thresholds, or adjudication should be produced until the design is reviewed and approved.

