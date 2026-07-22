# Phase 2 Account-Value and Performance History Design

## Status

Approved design. This phase extends the existing IPS Pilot portfolio workflow; it does not replace manual portfolio snapshots, layer evaluation, Review Queue, or journal behavior.

## Purpose

Phase 2 turns the immutable Toss account observations from Phase 1 into a reproducible account-value and performance history. It establishes a user-confirmed tracking baseline, keeps external cash flows separate from investment return, calculates supported realized and unrealized profit/loss, and exposes history through a local agent-facing CLI.

The phase is deliberately evidence-first. It must not infer lifetime principal, silently treat unexplained balance changes as return, or create buy/sell/order instructions.

## Scope

### In scope

- One immutable tracking baseline per Toss account alias, initialized from complete `snapshot_id=4` after explicit value confirmation.
- Append-only external cash-flow candidates and user decisions.
- Versioned, idempotent performance projection runs linked to immutable account snapshots.
- Account value, invested market value, deployable cash, current holdings cost basis, unrealized profit/loss, supported realized profit/loss, cumulative gain, simple baseline-relative return, and evaluable TWR segments.
- Separate actual-cost and post-baseline tracking-cost ledgers using moving-average accounting.
- Native-currency execution records and optional KRW translation using a source observation exchange rate.
- Foreign-currency remeasurement diagnostics that are not presented as exact return attribution.
- Local JSON CLI commands for baseline confirmation, projection refresh, candidate review, flow decisions, and history inspection.
- SQLite schema migration from version 2 to version 3 with preservation of all existing data.

### Out of scope

- Automatic orders, order sizing, buy/sell recommendations, take-profit/stop-loss rules, or execution flags.
- Market-regime logic, reserve-policy statuses, or profit/loss review statuses; those belong to Phases 3–5.
- Public API endpoints or frontend pages; Phase 6 will add authenticated workbench surfaces.
- A claim of lifetime return or tax-grade realized profit/loss.
- Automatic classification of unexplained cash or position changes.
- Corporate-action reconstruction when the broker does not provide sufficient evidence.
- A background scheduler; the user explicitly runs refresh after a Toss sync.

## Existing-System Boundary

Phase 0 and Phase 1 remain the source-of-truth boundaries:

- `broker_account_snapshots` and child tables stay immutable after insertion.
- Only `complete` snapshots can be performance inputs.
- `partial`, `stale`, and `failed` snapshots remain diagnostic records and cannot become performance points.
- Existing `portfolio_snapshots`, `snapshot_positions`, evaluation runs, Review Queue, and journals are not mutated by Phase 2.
- The existing invested-weight semantics remain unchanged. Account cash is modeled separately and is not inserted into asset or layer weights.
- The Phase 2 CLI emits one JSON object and never returns raw account numbers, credentials, access tokens, or raw broker payloads.

## Definitions

### Tracking baseline

The tracking baseline is the user's confirmed starting point for this history. For this project it is:

- `account_alias`: `toss-brokerage`
- `baseline_snapshot_id`: `4`
- `tracking_started_at`: the `synced_at` value of snapshot 4
- `initial_principal_krw`: `120802745.17802304` KRW, confirmed by matching the stored snapshot value

The stored baseline is immutable. A correction requires a new explicit baseline record in a future version; no update or delete command is provided in Phase 2.

### Principal and cost basis

- `tracking_principal_krw`: the confirmed initial principal plus confirmed external deposits minus confirmed external withdrawals.
- `current_cost_basis`: the broker-reported cost basis of currently held positions, translated at the observation exchange rate when possible. It is never labeled lifetime principal.
- `actual_cost_ledger`: moving-average basis seeded from each baseline holding's native `cost_native`.
- `tracking_cost_ledger`: moving-average basis seeded from each baseline holding's native `market_value_native`, so post-baseline realized tracking profit/loss excludes pre-baseline appreciation.

### Performance values

- `total_value_krw`: Phase 1's complete snapshot account value.
- `invested_value_krw`: Phase 1's invested market value.
- `cash_value_krw`: Phase 1's validated KRW-translated buying power.
- `unrealized_pnl_krw`: broker-reported current holdings profit/loss translated by the observation rate.
- `account_gain_krw`: `total_value_krw - initial_principal_krw - cumulative_confirmed_net_external_flow_krw`.
- `simple_return`: `account_gain_krw / initial_principal_krw` only while no confirmed external flow exists. It is null after a flow because the denominator would be misleading.
- `interval_twr`: return between two adjacent complete snapshots when the interval has no unresolved flow or reconciliation issue.
- `segment_twr`: chain-linked TWR within an uninterrupted evaluable segment. A confirmed flow without adequate bracketing ends the current segment; later evaluable points start a new segment.
- `tracked_realized_pnl`: realized result against the post-baseline tracking cost ledger.
- `actual_realized_pnl`: realized result against the broker-style actual cost ledger. It can include appreciation that occurred before the tracking baseline and is therefore not the tracking return.

### Data quality

Performance points use `evaluable` or `non_evaluable` as data-quality states. These are not IPS inspection statuses. A point is `non_evaluable` when an unresolved material cash residual, unexplained position quantity change, conflicting execution, missing required valuation, or missing required FX conversion prevents a defensible calculation.

## Persistence Model (Schema Version 3)

Migration 3 is forward-only and must preserve all version 1 and version 2 rows.

### `account_tracking_baselines`

One immutable confirmed baseline per account alias.

```sql
CREATE TABLE IF NOT EXISTS account_tracking_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL UNIQUE,
    baseline_snapshot_id INTEGER NOT NULL UNIQUE
        REFERENCES broker_account_snapshots(id),
    tracking_started_at TEXT NOT NULL,
    initial_principal_krw REAL NOT NULL,
    baseline_fx_rate REAL,
    confirmed_at TEXT NOT NULL,
    confirmation_fingerprint TEXT NOT NULL UNIQUE
);
```

The confirm command accepts an expected principal and requires it to match the complete snapshot's stored `total_value_krw` within `0.01` KRW. It rejects a missing, non-complete, or non-evaluable snapshot.

### `account_cash_flow_candidates`

One immutable candidate per baseline, snapshot interval, currency, and residual fingerprint.

```sql
CREATE TABLE IF NOT EXISTS account_cash_flow_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER NOT NULL REFERENCES account_tracking_baselines(id),
    from_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    to_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    currency TEXT NOT NULL,
    observed_delta_native REAL NOT NULL,
    explained_trade_delta_native REAL NOT NULL,
    residual_native REAL NOT NULL,
    residual_krw REAL,
    materiality_threshold_krw REAL NOT NULL,
    bridge_basis TEXT NOT NULL CHECK(bridge_basis IN ('filled_at', 'settlement_date', 'none')),
    candidate_fingerprint TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

The candidate detector compares both fill-time and settlement-time trade bridges. If either bridge explains the balance within tolerance, no material candidate is created. Residual materiality is `max(10000 KRW, 1 basis point of the prior total account value)`, with USD thresholds converted using the ending complete snapshot's FX rate. Sub-material residuals remain in run diagnostics.

### `account_cash_flow_decisions`

User decisions are append-only. A new decision supersedes an older decision for calculation purposes without deleting the old row.

```sql
CREATE TABLE IF NOT EXISTS account_cash_flow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES account_cash_flow_candidates(id),
    classification TEXT NOT NULL CHECK(classification IN (
        'external_deposit', 'external_withdrawal', 'investment_income',
        'fee_or_tax', 'internal_fx', 'rounding_or_false_positive',
        'other_non_external'
    )),
    confirmed_amount_native REAL,
    confirmed_amount_krw REAL,
    effective_at TEXT,
    note TEXT NOT NULL DEFAULT '',
    decided_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Only `external_deposit` and `external_withdrawal` change tracking principal. Other classifications explicitly attest that the movement is not an external capital flow; the movement remains part of account performance. A candidate with no latest decision remains unresolved.

For an external deposit or withdrawal, omitted confirmation amounts default to the candidate's signed residual in the candidate currency. If a user supplies an amount, it must have the same sign and be within the candidate's configured tolerance; otherwise the decision is rejected. A decision without an effective timestamp remains usable for principal accounting but keeps the affected TWR interval non-evaluable.

### `account_performance_runs`

An immutable calculation execution. Repeating the same inputs returns the existing run.

```sql
CREATE TABLE IF NOT EXISTS account_performance_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id INTEGER NOT NULL REFERENCES account_tracking_baselines(id),
    through_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    input_fingerprint TEXT NOT NULL UNIQUE,
    engine_version TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('complete', 'partial', 'blocked')),
    data_quality_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

`state=complete` means every included point is usable or explicitly segmented. `partial` means a newer complete snapshot exists but some intervals are non-evaluable. `blocked` means no new point can be produced because the required source or baseline is missing.

### `account_performance_points`

One point per source snapshot and performance run.

```sql
CREATE TABLE IF NOT EXISTS account_performance_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES account_performance_runs(id),
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    previous_snapshot_id INTEGER REFERENCES broker_account_snapshots(id),
    point_at TEXT NOT NULL,
    evaluation_state TEXT NOT NULL CHECK(evaluation_state IN ('evaluable', 'non_evaluable')),
    evaluation_reason TEXT,
    total_value_krw REAL,
    invested_value_krw REAL,
    cash_value_krw REAL,
    current_cost_basis_krw REAL,
    unrealized_pnl_krw REAL,
    tracking_principal_krw REAL,
    cumulative_external_flow_krw REAL,
    account_gain_krw REAL,
    simple_return REAL,
    interval_twr REAL,
    segment_id INTEGER,
    segment_twr REAL,
    tracked_realized_pnl_krw REAL,
    actual_realized_pnl_krw REAL,
    fx_remeasurement_krw REAL,
    UNIQUE(run_id, snapshot_id)
);
```

### `account_execution_ledger`

Derived execution rows used to make realized P&L auditable.

```sql
CREATE TABLE IF NOT EXISTS account_execution_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES account_performance_runs(id),
    source_snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id),
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    currency TEXT NOT NULL,
    side TEXT NOT NULL,
    filled_at TEXT,
    settlement_date TEXT,
    filled_quantity_native REAL NOT NULL,
    filled_amount_native REAL,
    commission_native REAL,
    tax_native REAL,
    actual_basis_before_native REAL,
    tracking_basis_before_native REAL,
    actual_realized_pnl_native REAL,
    tracking_realized_pnl_native REAL,
    realized_pnl_krw REAL,
    krw_conversion_snapshot_id INTEGER REFERENCES broker_account_snapshots(id),
    UNIQUE(run_id, order_id)
);
```

Zero-fill orders are excluded. If the same `order_id` has conflicting final execution fields across complete snapshots, the run is blocked and no realized P&L is emitted for that order.

## Projection Algorithm

1. Load the immutable baseline and all complete account snapshots at or after the baseline, sorted by `synced_at`, then `id`.
2. Verify every source snapshot has total value, invested value, cash, and required FX data. Keep the source snapshot unchanged.
3. Build a canonical closed-execution set by `order_id`. Identical duplicates are collapsed; conflicting duplicates create a blocked diagnostic.
4. Seed each baseline holding's actual native moving-average basis from `cost_native` and tracking native basis from `market_value_native`.
5. For each interval, compare holding quantity changes with filled executions. An unexplained material quantity change marks the point `non_evaluable` and records a position-reconciliation issue.
6. Compare native buying-power changes against trade cash deltas using both `filled_at` and `settlement_date` bridges. Create candidates only for material residuals that neither bridge explains.
7. Apply the latest candidate decisions. External deposit/withdrawal decisions adjust tracking principal; all other explicit non-external decisions leave principal unchanged.
8. Process post-baseline executions in deterministic order `(filled_at or ordered_at, order_id)`. Buys add quantity and total cost to both moving-average ledgers. Sells remove quantity at the prior average basis and produce actual and tracking realized P&L. Commissions and taxes reduce proceeds/cost according to the execution side and remain visible in the ledger.
9. Compute account gain, cost basis, unrealized P&L, TWR, and the bounded FX remeasurement diagnostic for the point. Do not force components to sum when income, corporate action, or missing execution evidence prevents a complete bridge.
10. Persist one immutable performance run, its points, and its execution ledger in a transaction. A repeated input fingerprint returns the existing run.

## CLI Contract

All commands return exactly one JSON object and preserve machine-readable errors.

- `ips-pilot performance baseline-preview --snapshot-id ID`
- `ips-pilot performance baseline-confirm --snapshot-id ID --expected-principal-krw VALUE`
- `ips-pilot performance refresh`
- `ips-pilot performance candidates --run-id ID`
- `ips-pilot performance decide-flow --candidate-id ID --classification CLASSIFICATION [--amount-native VALUE] [--effective-at ISO] [--note TEXT]`
- `ips-pilot performance history --latest | --run-id ID`

The baseline confirm and flow decision commands are the only Phase 2 persistence commands that require explicit user intent. Refresh persists an immutable derived run but never changes a broker snapshot or broker state.

## Failure and Safety Rules

- No automatic fallback to a prior performance result when the requested source is missing or stale.
- No calculation from `partial`, `stale`, or `failed` account snapshots.
- No automatic cash-flow classification.
- No silent correction of quantity, cash, order, or FX mismatches.
- No use of current holdings cost basis as tracking principal.
- No tax-grade or lifetime-performance claim.
- No buy, sell, execute, order-size, or execution fields in performance inspection output.
- The execution ledger's source `side` is internal audit data and is not emitted by the regular history summary.
- All user decisions and all source snapshot IDs remain replayable.

## Verification Strategy

Tests use local SQLite fixtures and existing `FakeReader`/`MockTransport` patterns; no live Toss or market-data call is required.

### Migration and persistence

- v2 to v3 migration preserves all existing rows and is forward-only.
- Baseline creation requires a complete snapshot and exact principal confirmation.
- Duplicate baseline confirmation is rejected without modifying the first baseline.
- Candidate, decision, run, point, and execution rows remain after later refreshes.
- Identical refresh inputs return the same run ID.

### Calculation

- Baseline point starts at zero gain and zero TWR.
- No-flow snapshots produce reproducible account gain, simple return, and interval/segment TWR.
- External deposit and withdrawal decisions alter tracking principal but do not fabricate return.
- Unresolved candidates make only the affected interval non-evaluable.
- Moving-average buy/sell ledger handles commissions, taxes, zero-fill orders, repeated order IDs, and deterministic ordering.
- Actual and tracking realized P&L differ correctly for positions held before the baseline.
- Unexplained quantity changes and conflicting executions block realized P&L.
- KRW/USD native calculations and bounded FX remeasurement are deterministic.

### CLI and security

- Every command emits one JSON object on success and failure.
- Latest history reports the actual `run_id`, `baseline_snapshot_id`, and source snapshot IDs.
- Raw account number, `accountSeq`, client secret, access token, and raw broker response are absent from persisted rows and CLI output.
- Existing portfolio/evaluation tests continue to pass unchanged.

## Adversarial Review Checklist

- Confirm that the baseline total value is never read as lifetime deposits.
- Confirm that a new complete snapshot cannot replace an old performance run in place.
- Confirm that a partial/stale snapshot cannot advance the current evaluable history.
- Confirm that current FX translation of cost basis is not labeled historical principal.
- Confirm that pre-baseline appreciation is not counted as post-baseline tracking realized P&L.
- Confirm that unexplained dividends, fees, currency conversions, and corporate actions are not silently external deposits or investment return components.
- Confirm that order mutation transport remains unchanged and unavailable.
- Confirm that the CLI cannot emit order sizing or direct action language.
- Confirm that migration and refresh are transactional and safe when interrupted.

## Exit Criteria

- A user-confirmed baseline exists for `toss-brokerage` without changing broker or existing portfolio state.
- At least one reproducible complete performance run can be generated from snapshot 4 and later complete snapshots.
- Each displayed value links to immutable source snapshot IDs and a calculation run fingerprint.
- External cash-flow ambiguity produces a visible candidate and a non-evaluable interval rather than fabricated return.
- Actual cost basis, tracking principal, unrealized P&L, and tracked realized P&L are distinct in storage and output.
- Full backend tests, Phase 2 focused tests, Ruff checks, migration checks, and CLI redaction checks pass.
