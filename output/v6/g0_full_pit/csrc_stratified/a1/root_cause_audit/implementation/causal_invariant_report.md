# A1 causal invariant semantics implementation report

Date: 2026-09-08. Scope: `v6/classification_checkpoints.py` and `tests/test_v6_classification_checkpoints.py`. No adjudication was rerun and no frozen `a1/v2` artifact was modified.

## Result

The timeline now distinguishes a historical legal effective date from actual point-in-time backfill. A parsed state remains usable only from `max(known_at, effective_date)`. When `effective_date < known_at`, the row is marked `retrospective_effective_date=true` for audit, but this fact alone is not a causal violation.

The causal metric now counts observable timeline violations:

- `effective_from` precedes the checkpoint evidence `known_at`;
- a state source belongs to another family or was not known by `effective_from`;
- a `VERIFIED_NOOP` names a cross-family or non-causal predecessor, retaining the existing rejection and orphan behavior.

Quality review round 1 made this fail closed. A `PARSED_STATE` or `VERIFIED_NOOP` with no `known_at` cannot form a usable interval and is counted as causal. Every usable row must name a source that exists, belongs to the same family, has final disposition `PARSED_STATE`, has a non-empty `known_at`, and was known no later than `effective_from`. Empty, missing, unresolved, intermediate NOOP, cross-family, and future-known sources are rejected. Evidence and quoted legal dates remain on unusable rows for audit.

A future legal effective date still delays `effective_from` until that date and is not a violation. The Gate metric schema and threshold are unchanged; only the meaning of an already-computed causal violation now matches the protocol rule against using evidence before it was public and effective.

## TDD evidence

Five focused behaviors cover historical dates, actual backfill, future legal effectiveness, future-known state sources, and cross-family/later NOOP predecessors. Before implementation, the focused run produced `2 failed, 3 passed, 72 deselected`; both failures were the missing retrospective audit field. After the implementation, the same selection produced `5 passed, 72 deselected`.

After quality review round 1, six new negative cases were observed RED (`6 failed, 77 deselected`) before production changes. The expanded focused selection then passed (`11 passed, 72 deselected`), and the complete classification checkpoint file passed: `83 passed`. Raw command summaries are retained in `causal_invariant_round1_red.log` and `causal_invariant_round1_green.log`.

An earlier complete V6 selection produced `242 passed, 2 failed`. The exact failures were `test_isolated_worker_restarts_after_one_document_times_out` and `test_isolated_worker_restarts_after_child_process_crashes`; both timed out at `v6/g0_checkpoint_gate.py:142` while waiting five seconds for a restarted spawn worker. A focused rerun reproduced `2 failed, 69 deselected`. A later no-concurrency full V6 run after round 1 stopped making progress beyond 57% in the Windows spawn-heavy segment and was interrupted after several minutes; it is incomplete. Process inspection during diagnosis showed one responsive Python 3.11 process (`PID 5940`, CPU 4.72 seconds at observation), but this does not establish whether the delay came from residual load or child import/startup latency. The worker uses `multiprocessing.get_context("spawn")`; no worker code was modified.

## Frozen v2 in-memory probe

The probe read the frozen 1,371-row manifest and 1,371-row dispositions, rebuilt the timeline in memory, and ran the invariant validator without writing a formal adjudication artifact:

- timeline rows / unique checkpoint IDs: `1371 / 1371`;
- manifest and timeline checkpoint ID sets: identical;
- `retrospective_effective_dates = 6`;
- `causal_violations = 0`;
- `nonpositive_intervals = 0`.

This converts the six audited historical-date flags into explicit retrospective-date diagnostics while preserving their actual PIT start at `known_at`.

## Frozen A1 v2 verification

The six formal artifact hashes remain unchanged:

| Artifact | SHA-256 |
|---|---|
| `required_classification_checkpoints.csv` | `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd` |
| `document_dispositions.csv` | `61c01da72082769382562071e2e19de636a25f5b6aeb662c5d6b6a806ecd5991` |
| `classification_clause_clusters.csv` | `d6a43c1a95c3c189ca106a3e42d80b9f51df7bd6be7ed62af13105bf2c8f362a` |
| `classification_state_timeline.csv` | `fc2c1ea87565e1424d6bde608c5e5842f8c1d8086ce87e3086738ada316aa7f4` |
| `g0_amendment_a1_gate_metrics.csv` | `f2e71aea7ec74db1893f34275f18f425e68ef56c624c36b07d79c3252adf7074` |
| `g0_amendment_a1_gate_review.md` | `a026b99d58d44a52a5ba774977f0f81bd9301626f0018744fef26f31082fa5b4` |

Repository metadata is absent in this delivered workspace, so version-control status is `NO_GIT`.
