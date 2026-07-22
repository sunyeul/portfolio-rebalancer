# Phase 1 Toss Account Observation Design

## Scope

Phase 1 establishes a reproducible, read-only Toss account observation pipeline. It reads account metadata, holdings, KRW/USD cash buying power, USD/KRW exchange rate, and closed-order history, normalizes the official OpenAPI response shapes, reconciles the currency totals, and persists immutable local snapshots.

This phase does not place, modify, cancel, or size orders. It does not calculate investment return, realized profit/loss, cash policy status, or buy/sell actions; those belong to later phases. It also does not persist access tokens, client secrets, account numbers, or `accountSeq` values.

## Decisions

### Read path and account discovery

`GET /api/v1/accounts` is called with the OAuth bearer token but without `X-Tossinvest-Account`; it is the bootstrap endpoint that returns the configured account's `accountSeq`. All other account-context reads use the configured sequence in `X-Tossinvest-Account`. The configured sequence must match one returned `BROKERAGE` account or the synchronization fails without persistence.

The allowlisted transport remains the only network boundary. Pagination is implemented only for `GET /api/v1/orders?status=CLOSED`, using `cursor`, `limit`, `from`, and `to`; repeated cursors and duplicate order IDs are rejected or deduplicated deterministically.

### Normalized observation

Each sync produces one normalized snapshot identified by a SHA-256 fingerprint over sorted normalized values. The snapshot stores an alias (`toss-brokerage`), source call timestamps, normalized holdings, native-currency cash, applied USD/KRW rate, aggregated order executions, reconciliation diagnostics, and a data-quality state:

- `complete`: all required calls and currency checks succeed;
- `partial`: a usable holding/cash observation exists but a required call, pagination page, or reconciliation check is missing/inconsistent;
- `stale`: source timestamps are older than the configured freshness window;
- `failed`: no usable holdings snapshot can be produced.

Raw account numbers, account sequence values, credentials, access tokens, and raw response bodies never enter SQLite, API responses, or CLI output.

### Currency and reconciliation policy

The API explicitly returns currency-separated `Price` values and `cashBuyingPower` values. The normalizer verifies that the KRW and USD buying-power responses identify the requested currency, that the USD/KRW rate is positive, and that each holding's native currency agrees with its market-country/currency contract. Only then is USD cash converted to KRW using the returned `rate`; the snapshot records `cash_currency_check=verified_by_currency_labels` and the original native values.

Holdings reconcile independently by currency: summed item market values must equal the holdings overview `marketValue` amounts within `0.01` native units. KRW-normalized totals are calculated only from verified native values and the applied rate. A mismatch creates `partial`, not a silently corrected value.

### Persistence and idempotency

Schema version 2 adds `broker_account_snapshots`, `broker_holdings`, `broker_cash_observations`, `broker_exchange_rates`, and `broker_orders`. Parent rows are immutable after insertion. A unique `(account_alias, fingerprint)` constraint makes repeated identical syncs return the original snapshot instead of creating duplicates. Only a `complete` snapshot may become the current evaluable snapshot; a later `partial`, `stale`, or `failed` snapshot is retained as diagnostic evidence and cannot replace it.

### User-facing entry points

The agent-facing CLI gains:

- `ips-pilot toss-health`: live config, OAuth, account discovery, and account-sequence match check; no persistence;
- `ips-pilot toss-sync [--from YYYY-MM-DD] [--to YYYY-MM-DD]`: read and persist one immutable snapshot;
- `ips-pilot toss-snapshots [--latest | --snapshot-id ID]`: inspect normalized local snapshots without contacting Toss.

Every command emits one machine-readable JSON object. Errors contain a stable stage/code and sanitized message only. No public FastAPI sync route is added in this phase because the current app has no authentication boundary for a broker-reading endpoint.

## Verification

Tests use `httpx.MockTransport` fixtures only. They cover official response shapes, account bootstrap without an account header, OAuth/header behavior, KRW/USD reconciliation, exchange-rate conversion, pagination/cursor deduplication, stable fingerprints, partial/failure states, stale snapshots, latest-complete preservation, CLI JSON output, and absence of secret/account-number persistence.

The Phase 1 exit gate requires the full backend suite, focused observation tests, ruff checks on changed files, and a grep proving that no order mutation method or raw account identifier is introduced.
