# G0 Amendment A1 Gate Review (V6-G0-A1)

| Role | Metric | Threshold | Numerator | Denominator | Actual | Failures | Result |
|---|---|---:|---:|---:|---:|---:|---|
| PRIMARY | checkpoint_coverage | >= 0.95 | 120 | 1371 | 0.0875273523 | 1251 | FAIL |
| PRIMARY | critical_checkpoint_resolution | == 1.0 | 120 | 1371 | 0.0875273523 | 1251 | FAIL |
| PRIMARY | fund_coverage_overall | >= 0.90 | 45 | 60 | 0.75 | 15 | FAIL |
| PRIMARY | coverage_存续_pre2013 | >= 0.80 | 7 | 10 | 0.7 | 3 | FAIL |
| PRIMARY | coverage_存续_2013_2019 | >= 0.80 | 10 | 10 | 1 | 0 | PASS |
| PRIMARY | coverage_存续_2020_2026 | >= 0.80 | 4 | 10 | 0.4 | 6 | FAIL |
| PRIMARY | coverage_清盘_pre2013 | >= 0.80 | 8 | 10 | 0.8 | 2 | PASS |
| PRIMARY | coverage_清盘_2013_2019 | >= 0.80 | 8 | 10 | 0.8 | 2 | PASS |
| PRIMARY | coverage_清盘_2020_2026 | >= 0.80 | 8 | 10 | 0.8 | 2 | PASS |
| PRIMARY | causal_violations | == 0 | 6 | 1 | 6 | 6 | FAIL |
| PRIMARY | unresolved_conflicts | == 0 | 0 | 1 | 0 | 0 | PASS |
| PRIMARY | nonpositive_intervals | == 0 | 15 | 1 | 15 | 15 | FAIL |
| PRIMARY | orphan_noops | == 0 | 0 | 1 | 0 | 0 | PASS |
| PRIMARY | critical_unresolved | == 0 | 1251 | 1 | 1251 | 1251 | FAIL |
| DIAGNOSTIC | document_parser_coverage | diagnostic only | 1676 | 1993 | 0.8409433016 | 317 | PASS |
| DIAGNOSTIC | pdf_acquisition_coverage | diagnostic only | 2807 | 2808 | 0.9996438746 | 1 | PASS |
| DIAGNOSTIC | historical_missing_pdf_count | diagnostic only | 1 | 1 | 1 | 1 | PASS |

Historical missing PDF remains disclosed: `162703/420632`.

FINAL: G0 FAIL under V6-G0-A1
