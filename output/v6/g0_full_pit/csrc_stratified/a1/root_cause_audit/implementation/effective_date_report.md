# A1 effective-date extraction implementation report

Date: 2026-09-08. Scope: `v6/csrc_fund_disclosure.py` and `tests/test_v6_csrc_fund_disclosure.py`. This is an implementation fix only; it does not rerun adjudication or modify timeline, invariant, NOOP, section extraction, checkpoint definitions, or any `a1/v2` artifact.

## Root cause and implemented behavior

The previous extractor removed whitespace and allowed a date to span an unlimited non-sentence region before finding `基金合同` and `生效`. In `400013_321232.pdf`, PDF extraction removed useful layout boundaries and allowed the 2004-06-01 implementation date of the Securities Investment Fund Law to bind to a much later contract-definition phrase.

The replacement accepts dates only through bounded local grammar:

- a named fund contract immediately followed by `自/于 DATE 起 ... 生效`;
- `自 DATE 起 ... 基金合同 ... 生效` within a bounded clause;
- the explicit predicate `自 DATE 起 ... 转型 ... 生效`.

It rejects unresolved `YYYY年XX月XX日` contract-effective placeholders. If more than one distinct locally qualified date remains, it returns `None` instead of selecting one by document order. Invalid calendar dates are also ignored.

This preserves the required positive forms `自2017年5月11日起转型生效` and `基金合同自2007年2月12日起生效`, as well as the existing form where an old contract expires and a named replacement contract becomes effective on the same explicit date.

## Evidence and tests

The RED cases use the audited 321700 text and the real local official `400013_321232.pdf`. They demonstrate the two incorrect historical dates and the required positive forms before implementation. A separate ambiguity case proves that two distinct qualified dates yield `None`.

Verification after implementation:

- focused effective-date regression: 6 passed, 35 deselected;
- `tests/test_v6_csrc_fund_disclosure.py`: 41 passed;
- all `tests/test_v6_*.py`: 232 passed;
- direct real-PDF checks: both `400013_321232.pdf` and `400013_321700.pdf` return `None`.

The local desktop Python runtime did not include pytest or requests. They were installed only under the user's temporary directory and injected into the test process; project dependencies and source files were not changed for this environment setup.

## Frozen A1 v2 verification

The six frozen-artifact SHA-256 values match `census_summary.json` exactly after the implementation and tests:

| Artifact | SHA-256 |
|---|---|
| `required_classification_checkpoints.csv` | `477b944f504b04bbeb2a24faa4192764fb3eba2ce0ccd5ea3e588bab33172efd` |
| `document_dispositions.csv` | `61c01da72082769382562071e2e19de636a25f5b6aeb662c5d6b6a806ecd5991` |
| `classification_clause_clusters.csv` | `d6a43c1a95c3c189ca106a3e42d80b9f51df7bd6be7ed62af13105bf2c8f362a` |
| `classification_state_timeline.csv` | `fc2c1ea87565e1424d6bde608c5e5842f8c1d8086ce87e3086738ada316aa7f4` |
| `g0_amendment_a1_gate_metrics.csv` | `f2e71aea7ec74db1893f34275f18f425e68ef56c624c36b07d79c3252adf7074` |
| `g0_amendment_a1_gate_review.md` | `a026b99d58d44a52a5ba774977f0f81bd9301626f0018744fef26f31082fa5b4` |

## Remaining 321700 limitation

Returning `None` is the only supported conclusion at this extraction layer. The PDF contains historical 2017-05-11 transformation text, a proposed replacement state, and an unresolved `2018年XX月XX日` effective date; its extracted text also interleaves table columns. Determining which 0-95 clause belongs to which legal stage still requires layout-aware table reconstruction and targeted legal verification. This implementation does not classify that clause or alter its adjudication.

Git status is unavailable because this delivered workspace has no `.git` repository (`NO_GIT`).

## Review fix round 1

Independent review found three remaining grammar gaps. The date-first branch could still bind a short law sentence such as `《证券投资基金法》自...施行` to a following fund-contract sentence because bounded distance did not prove syntactic ownership. Placeholder rejection covered date-first wording but not contract-first `合同自/于 PLACEHOLDER 生效` wording. The named-contract expression also required a character before `基金合同`, excluding the shorthand `《基金合同》`.

The date-first branch now requires the target contract subject immediately after the date, apart from punctuation and the explicit `原基金合同失效` bridge. Thus `施行，根据...` cannot cross-bind even when the text is short. Placeholder grammar mirrors both positive word orders and recognizes placeholder components in the year, month, or day. The target-subject grammar accepts `《基金合同》`, `本基金基金合同`, `新基金合同`, and longer book-title contract names.

Review RED was observed before the production change: 5 failed and 3 passed. Final verification was 11 focused review tests passed, 48 tests passed across the disclosure file and the two affected collector contracts, and all 237 V6 tests passed. The six frozen A1 v2 hashes remained identical to the values above.

The first full V6 run during this review round produced 2 failures and 235 passes. Both failures used the established `自2019年1月11日起，新基金合同生效` form. The initially narrowed subject expression accidentally composed `新基金` with an additional `基金合同`, so it matched `新基金基金合同` instead of `新基金合同`. Replacing that construction with explicit subject alternatives restored both collector contracts; the final full V6 run then passed all 237 tests.
