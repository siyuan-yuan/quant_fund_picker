# A1 lifecycle checkpoint evidence resolver implementation

Date: 2026-09-08

## Scope and result

Implemented the first root-cause fix only: an explicit, deterministic,
fail-closed evidence resolver for `INCEPTION` and `TERMINAL_STATE` checkpoints,
including review fix round 1 for end-to-end provenance binding.
No adjudication was run and no timeline, date extraction, section extraction,
NOOP rule, Gate threshold, or frozen `a1/v2` artifact was changed.

## API and adjudication behavior

`resolve_lifecycle_evidence(checkpoint, candidate_evidence)` returns
`(selected_records, failure_reason)`.

- It operates only on `INCEPTION` and `TERMINAL_STATE`.
- The checkpoint must have a non-empty authoritative `family_key` and
  `known_at`.
- Every eligible candidate must explicitly carry the identical `family_key`,
  a valid non-later `known_at`, non-empty `upload_info_id`, `source_document`,
  `evidence_location`, and raw section text. Its parser-side upload, date,
  source, location, raw text, and normalized text must match those fields.
- The closest candidate date at or before the checkpoint is selected. Ties are
  sorted by source document, upload ID, evidence location, and normalized
  clause, independent of input order.
- Different normalized clauses at the selected date, or an explicit source
  conflict, return `SOURCE_CONFLICT`.
- A terminal checkpoint without `known_at` remains unresolved. Later documents
  are never backfilled.
- All ordinary document checkpoints retain exact `upload_info_id` matching.
- Existing three-argument callers remain valid and do not infer lifecycle
  evidence from ordinary `sections` or `parsed_documents`.
- The resolver binds candidate evidence only. Existing same-source quote,
  explicit parser state, numeric support, and evidence-location requirements
  still decide `PARSED_STATE`. Lifecycle resolution cannot create
  `VERIFIED_NOOP`.

`_build_lifecycle_evidence(sections, parsed, aliases, family_sample)` is the
Gate-side authority boundary. It maps `query_code` through explicit
`sample_share_code` to the exact `family_sample.family_key`, raises on missing,
unknown, or ambiguous sample mappings, and combines
section/parser records only when code, upload ID, known date, source document,
evidence location, and normalized evidence all match. `alias.family_key` is only
diagnostic and cannot replace the sample identity. `run_checkpoint_gate` passes this
self-contained frame through the optional `lifecycle_evidence` argument.

For equal-date/equal-clause sources, adjudication evaluates each bound parser
state separately, chooses deterministically among supported identical states,
and returns `SOURCE_CONFLICT` for different supported states. A successful
lifecycle decision retains every frozen manifest field. The independent
`evidence_upload_info_id`, `evidence_known_at`, `evidence_effective_date`,
`evidence_source_document`, and `evidence_location` fields identify the selected
official PDF.

## TDD evidence

The focused RED run failed for the missing resolver and missing lifecycle join:
5 failed and 3 passed. The final focused GREEN run passed 8 tests. Coverage
includes cross-family rejection, later-document rejection, missing provenance,
same-priority ambiguity, deterministic selection, ordinary checkpoint exact
matching, undated terminal refusal, dated terminal causal selection, and the
continued explicit-state requirement.

- `lifecycle_resolver_red.log`
- `lifecycle_resolver_green.log`
- `classification_checkpoints_full.log`: 61 passed
- `all_v6_tests.log`: 211 passed

Review fix round 1 added a second genuine RED: 6 failed and 7 passed. Final
verification after the fix:

- `lifecycle_resolver_fix1_red.log`: 6 failed, 7 passed
- `lifecycle_resolver_fix1_green.log`: 20 passed
- `lifecycle_resolver_fix1_focused_full.log`: 132 passed before the final
  provenance parameter-matrix expansion
- `lifecycle_resolver_fix1_all_v6.log`: 223 passed in 64.08 seconds

Review fix round 2 corrected the formal input contract and frozen identity
separation. Its initial RED was 5 failed and 19 passed. The production parser
shape may omit `evidence_location`; in that exact case the builder uses the same
explicit fallback as `_sections_from_parsed`, namely non-empty
`source_document`. A missing source remains ineligible.

- `lifecycle_resolver_fix2_red.log`: 5 failed, 19 passed
- `lifecycle_resolver_fix2_green.log`: 24 passed
- `lifecycle_resolver_fix2_all_v6.log`: 227 passed in 61.36 seconds

The first fix-round-2 V6 regression reached 226 passed and 1 failed. The lone
failure was the pre-existing 1,993-document orchestration fixture, whose local
alias row still used the former two-column shape and omitted the newly required
`sample_share_code`. Adding the explicit `sample_share_code=000001` fixture
binding made that test pass (1 passed, 70 deselected); the subsequent complete
V6 run produced the 227-pass result above. No production relaxation was made
for the obsolete fixture shape.

## Frozen artifact verification

The six official `a1/v2` artifact hashes remain identical to the recorded
root-cause-audit baseline:

| Artifact | SHA-256 |
|---|---|
| required_classification_checkpoints.csv | 477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd |
| document_dispositions.csv | 61c01da72082769382562071e2e19de636a25f5b6aeb662c5d6b6a806ecd5991 |
| classification_clause_clusters.csv | d6a43c1a95c3c189ca106a3e42d80b9f51df7bd6be7ed62af13105bf2c8f362a |
| classification_state_timeline.csv | fc2c1ea87565e1424d6bde608c5e5842f8c1d8086ce87e3086738ada316aa7f4 |
| g0_amendment_a1_gate_metrics.csv | f2e71aea7ec74db1893f34275f18f425e68ef56c624c36b07d79c3252adf7074 |
| g0_amendment_a1_gate_review.md | a026b99d58d44a52a5ba774977f0f81bd9301626f0018744fef26f31082fa5b4 |

## Remaining integration boundary

Parser records without explicit evidence location or another required
provenance field remain unresolved. Alias rows outside the exact sampled family
set are ignored. This intentionally prevents ordinary parser/section caches,
fund names, or unreviewed code similarity from becoming lifecycle authority.
