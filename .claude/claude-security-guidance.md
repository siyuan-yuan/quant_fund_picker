# Quant Fund Picker security rules

- Treat `output/ledger.json`, `output/dca_state.json`, and `output/fee_settings.json` as private local financial data. Never expose their filesystem paths through a generic file-serving endpoint.
- Browser requests that mutate ledger/settings or start expensive scans must remain same-origin. Do not restore wildcard CORS on write APIs.
- Validate fund codes as exactly six ASCII digits before using them in cache paths or outbound data-source requests.
- Values from fund providers, cached scan files, API error messages, and imported ledger files are untrusted. Escape them before inserting into `innerHTML`; prefer `textContent` when markup is unnecessary.
- Keep request-body and collection-size limits on public API inputs. Reject non-finite monetary values (`NaN`, `Infinity`).
- External data requests must use fixed provider hosts and bounded timeouts; never fetch a user-supplied URL.
- Persist local financial data with atomic replacement. Preserve existing validation when loading legacy or manually edited files.
- Do not add secrets, account credentials, broker cookies, or API tokens to source, cache, logs, or documentation.
