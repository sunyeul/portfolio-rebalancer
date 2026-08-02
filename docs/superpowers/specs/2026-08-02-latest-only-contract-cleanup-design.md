# Latest-only contract cleanup

## Goal

Remove obsolete IPS policy, evaluation, and market-candle compatibility paths.
The workbench accepts and exposes only the current inspection contract.  It
retains the market-history points needed by the active risk-evidence window.

## Scope

- Delete the unused `allocation_review` validator family and reject unknown
  policy top-level fields instead of silently dropping them.
- Remove policy default injection, template fallbacks, evaluation payload
  hydration, engine policy defaults, and candle value-equivalence fallback.
- Require the current `phase5-v2` evaluation result shape at persistence and
  use the persisted result unchanged at the API boundary.
- Make candle replays idempotent by looking up the full identity plus
  fingerprint, while treating a changed fingerprint as a new revision.
- Add a one-off, explicit local database cleanup command that deletes
  non-current policy/evaluation records and non-latest candle revisions while
  retaining the current active policy and full distinct-date candle history.

## Data cleanup semantics

The command runs in one transaction after validating the active policy against
the strict policy contract.  It retains the active policy version and deletes
superseded policy versions only when no retained row references them.  It
deletes all evaluations except current `phase5-v2` evaluations whose result
passes the new persistence contract.  It retains the latest candle revision
for each `(source_kind, market_country, symbol, interval, candle_at, adjusted)`
identity; this preserves the historical series required for risk evidence.

If a retained row blocks policy-version deletion, the command fails closed and
reports it.  It does not mutate Toss snapshots, orders, account performance,
or active-policy values.

## Contract boundaries

`validate_policy` rejects unknown top-level fields and requires the existing
current policy fields.  `insert_evaluation_run` rejects malformed `phase5-v2`
results, including absent account/source/allocation/layer/instrument/queue
fields.  The API only reports `contract_supported=true` for this exact engine
version and shape; it performs no adaptation.

The v2 statuses (`OK`, `Watch`, `Review`, `Action`) and the inspection-only
guardrails remain unchanged.  No order, price, quantity, or execution field is
introduced.

## Verification

- Policy tests cover unknown-field rejection and reject legacy
  `allocation_review` input.
- Evaluation tests cover valid current v2 persistence and malformed v2
  rejection.
- Market-store tests cover same-fingerprint replay, fingerprint revisions, and
  `A -> B -> A` replay.
- Cleanup tests prove retained versus deleted rows and rollback on a blocked
  reference.
- Run Ruff and the full test suite.
