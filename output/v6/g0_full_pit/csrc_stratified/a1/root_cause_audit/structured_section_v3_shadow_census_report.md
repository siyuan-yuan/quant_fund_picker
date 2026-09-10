# A1 structured-section-v3 shadow root-cause census

## Status and scope

- Status: **APPROVED by two independent read-only reviews**.
- Scope: remediation diagnostic only; this is **not** a formal A1 adjudication and creates no Gate verdict.
- Frozen denominator: the exact 1,371 checkpoint IDs from `a1/v2`, SHA-256 `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd`.
- Immutable history: `a1/v2` remains the first official adjudication and its `G0 FAIL` is unchanged.
- Shadow census SHA-256: `a21d5b9b3ff574a2a644285536081a377db9001be95c2a5eae9430b1242e53cc`.

## Trigger × root-cause result

| Trigger | Total | Main diagnostic result |
|---|---:|---|
| INCEPTION | 60 | 36 `PARSED_CLAUSE`; 11 `INVESTMENT_SECTION_NOT_FOUND`; 7 `OFFICIAL_DOCUMENT_NOT_FOUND`; 4 `EQUITY_CLAUSE_NOT_FOUND`; 2 `NO_CAUSAL_SOURCE_DOCUMENT` |
| TRANSFORMATION | 49 | 37 `INVESTMENT_SECTION_NOT_FOUND`; 9 `DOCUMENT_NOT_APPLICABLE`; 3 `PARSED_CLAUSE` |
| SECTION_UNCOMPARABLE | 1,110 | 606 `INVESTMENT_SECTION_NOT_FOUND`; 307 `EQUITY_CLAUSE_NOT_FOUND`; 176 `DOCUMENT_NOT_APPLICABLE`; 18 `PROPOSED_STATE_NOT_EFFECTIVE`; 2 `TEXT_EXTRACTION_DAMAGED`; 1 `PDF_MISSING` |
| MATERIAL_SECTION_CHANGE | 121 | 95 `PARSED_CLAUSE`; 13 `EQUITY_CLAUSE_NOT_FOUND`; 10 `EQUITY_CLAUSE_AMBIGUOUS`; 2 `PROPOSED_STATE_NOT_EFFECTIVE`; 1 `TEXT_EXTRACTION_FAILED` |
| TERMINAL_STATE | 30 | 30 `DATE_OR_STAGE_UNDETERMINED` |
| TYPE_CHANGE | 1 | 1 `INVESTMENT_SECTION_NOT_FOUND` |

All rows close exactly to 1,371. The three generated pivots were independently recomputed with zero differences.

## Interpretation

The leading failure is in section discovery/boundaries, not classification regex: `INVESTMENT_SECTION_NOT_FOUND` accounts for 655 checkpoints, including 37/49 transformations and 606/1,110 formerly `SECTION_UNCOMPARABLE` checkpoints. The next major category is `EQUITY_CLAUSE_NOT_FOUND` (324), which requires document-level sampling before any parser expansion. `DOCUMENT_NOT_APPLICABLE` (185) must not be converted to `VERIFIED_NOOP`.

The 36/60 INCEPTION `PARSED_CLAUSE` count means only that a parsable causal candidate exists under the shadow diagnostic's successful-first selection. It is not a legal-state adjudication, does not prove the nearest governing document, and must not be reported as 36 resolved or as a PASS.

## Engineering route

The next authorized step is to inspect `INVESTMENT_SECTION_NOT_FOUND` examples drawn only from the pre-frozen 24 development families, stratified by trigger, fund/report code, title pattern, and observed heading structure. Untouched-validation failures remain sealed until rules and tests derived from development data are frozen. No extractor change is authorized by the aggregate census alone.

## Verification record

- Runner r1 canonical-source deduplication and exact frozen `source_document` matching: approved.
- Corrected checkpoint `f26d9b...904e`: `EQUITY_CLAUSE_AMBIGUOUS`, not source ambiguity.
- Checkpoint set: 1,371 rows, 1,371 unique IDs, exact equality with v2 IDs.
- Causal selection: no selected source known after its checkpoint; ordinary source mismatch count zero.
- Development/validation split: six strata each contain 4 development and 6 untouched-validation families.
- Tests: disclosure 56, classification 83, full Gate module 76; all explicit exit 0 / JUnit zero failures and errors.
- Output boundary: no disposition, adjudication, verdict, or formal Gate artifact was generated.

