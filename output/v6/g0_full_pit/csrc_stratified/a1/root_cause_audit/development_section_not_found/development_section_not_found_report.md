# Development-only `INVESTMENT_SECTION_NOT_FOUND` trace

## Result

The dominant failure is source-document mapping, not a missing generic heading pattern.

- Scope: 318 checkpoints from 16 pre-frozen development families only.
- Documents: 287 unique PDFs; all 287 text scans succeeded.
- Untouched-validation leakage: zero.
- No investment-structure term at all: 301/318 checkpoints.
- At least one investment-structure term: 17/318 checkpoints, but every hit is a narrative mention, governance clause, résumé phrase, or pointer to a separate contract/prospectus; none is a standalone governing investment section in the linked document.

## Trigger view

| Trigger | Checkpoints | No structure term | Narrative/reference hits |
|---|---:|---:|---:|
| INCEPTION | 5 | 5 | 0 |
| TRANSFORMATION | 26 | 23 | 3 |
| SECTION_UNCOMPARABLE | 287 | 273 | 14 |

The five INCEPTION sources are an allocation-confirmation notice, launch reminder, or sales-channel announcements—not initial prospectuses. Of 26 TRANSFORMATION rows, 24 sources are ordinary subscription/redemption, sales, or other operational notices. Only two titles are direct transformation/expiry documents.

The clearest direct transformation example, `400013_321924.pdf`, does not contain the new allocation clause. Lines 138–143 say that investment range, strategy and allocation operate according to the fund contract and prospectus published separately on 2017-04-27. Therefore adding another heading regex to the current announcement would be incorrect; the missing step is to retrieve and causally join those referenced governing documents.

Two `160212` conversion notices contain affirmative language that the investment objective/range/strategy did not change and that allocation follows the contract. This is a candidate for a separate A1 NOOP-evidence review, but it must not be automatically promoted: the frozen rule still requires affirmative unchanged evidence plus a causally valid predecessor.

## Title pattern evidence

For the 287 `SECTION_UNCOMPARABLE` rows, 130 are sales/channel notices, 71 other operational notices, 25 subscription/redemption operations, 22 launch/subscription notices, 14 contract/amendment notices, 14 transformation/expiry notices, 9 distribution notices, and only 1 prospectus/summary. This explains why a legal investment section is normally absent.

## Decision

Do not change the structured section extractor from this evidence. Open a source-document mapping audit for lifecycle and transformation checkpoints, beginning with the 60 INCEPTION and 49 TRANSFORMATION frozen checkpoints. The audit must distinguish:

1. governing prospectus/contract exists in local official metadata but was not linked;
2. announcement explicitly points to a separate governing document/date;
3. official governing document is absent from the collected metadata;
4. affirmative unchanged language may qualify only after predecessor validation;
5. genuinely unobservable under A1.

## Frozen artifacts

- Inventory SHA-256: `9b862f1d9866ead2bed79d4312a758224d54fdccc5a9e954796d0808a0ac186e`
- Scan cache SHA-256: `fa4a347d02e4936b3ed9cfe414ec358beb11e507c86b7b0e26c09e6d3089ef48`
- Trace SHA-256: `ac7cd7a14ebbaf1c3801ca25b6ba96def5abbe245980bcdc828d6e6cf7d86a61`
- Summary SHA-256: `b4492dcd77accea3f6ef1b713cf5f986a397768d41bbec16f458c07e8b3e65bd`
- Runner SHA-256: `2d798cd6a659952d7553d7bfb3567979359a9468533434b9a6fe83a5fc93ac0b`

