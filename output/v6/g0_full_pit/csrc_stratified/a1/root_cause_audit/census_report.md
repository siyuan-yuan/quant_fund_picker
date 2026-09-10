# A1 unresolved checkpoint root-cause census

This local-cache audit preserves the original 1,371 checkpoints and the six v2 adjudication artifacts. It is evidence triage, not a second adjudication. The script is `scripts/audit_a1_unresolved.py`; `census_summary.json` records before/after SHA-256 values for all six frozen artifacts.

## Checkpoint-level census

| Primary observed blocker | Checkpoints |
|---|---:|
| Lifecycle evidence join not implemented | 90 |
| Section versus semantics undetermined | 1,158 |
| Missing adjudication predecessor | 1 |
| Local PDF missing | 1 |
| Date or phase undetermined | 1 |
| Total unresolved | 1,251 |

The primary bucket is the earliest demonstrable blocker in the current processing path. Causes can coexist; secondary flags preserve additional observations. `UNDETERMINED_SECTION_VS_SEMANTICS` explicitly does not decide whether a legal section is absent, unreadable, omitted by extraction, or semantically unsupported. Existing cached errors conflate these stages. No row claims the official historical record lacks evidence.

`unresolved_checkpoint_census.csv` has exactly one row per unresolved checkpoint. It includes original disposition, source, dates, alias chain, stratum, parser and section-cache status, primary bucket, and secondary flags. No original disposition is changed.

## Inception and terminal lifecycle join

All 60 INCEPTION and 30 TERMINAL_STATE checkpoints use synthetic upload identities and `fund_master:` sources. `adjudicate_checkpoints` indexes sections and parsed records by exact official `upload_info_id`, then queries that index with the synthetic identity. There is no lifecycle-to-official-document resolver in that path. This is a demonstrated implementation gap, independent of whether suitable evidence ultimately exists.

Among the 60 inception checkpoints, 15 have a locally indexed family-alias prospectus candidate dated at or before the checkpoint; 11 have a parsed allocation clause in such a candidate. These counts are candidate discovery, not evidence approval: title matching does not prove it is the legally applicable initial prospectus, metadata dates are not independently verified publication dates, and alias relationships require legal review. The other 45 lack such a candidate in this local index; that is not an official-archive absence finding.

`inception_trace.csv` covers all 60 checkpoints. `lifecycle_candidate_documents.csv` lists their family-alias prospectus candidates, metadata publication/upload dates, whether known by checkpoint, local PDF availability, cached extraction/parse outcome, allocation clause, and metadata locator. The explicit blank official URL means the original metadata did not provide one. Later candidates remain in the list for diagnosis but cannot be used retroactively without valid causal evidence.

## Transformation

`transformation_trace.csv` covers all 49 frozen transformation checkpoints, including the 2 baseline successes. The other 47 have exact local document matches but return `No explicit equity allocation constraint found`; that error alone cannot identify a section-location problem, missing classification evidence, or a future-phase/date problem. Their exact-source candidates appear in the candidate table with cached clauses and dates. Legal human review remains pending and is explicitly marked in every automated trace.

## SECTION_UNCOMPARABLE and measurement design

Of 1,110 checkpoints, 772 have report code FC900090, 318 have FA report codes, and 20 have other FC codes. The filename/title and report code are retained per checkpoint. Many documents may legitimately lack a full investment chapter; report type alone does not prove NOOP or official evidence absence.

The frozen denominator section extractor calls `extract_equity_constraint` and stores its short evidence clause as `section_text`. Adjudication sections similarly come from parsed allocation evidence. Consequently the existing field and failure labels are not measurements of a generic, independently located legal section. The 1,158 combined errors need text-level review before assigning finer extraction-versus-semantic buckets. The high FC count makes universal-section-extractor failure an unproven explanation for the whole denominator.

One missing-predecessor checkpoint already has the clause `股票50％－70％` (`162207_419441.pdf`) but remains unresolved. This warrants state/adjudication review; it does not justify relaxing the A1 NOOP definition.

This census supports targeted implementation investigation and targeted legal evidence review. It does not yet justify either an A1 recovery PASS or an A2 protocol revision.
