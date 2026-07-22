# Phase 1 Toss Account Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read, normalize, reconcile, and persist immutable Toss account observations with safe CLI health/sync/snapshot commands.

**Architecture:** Extend the existing allowlisted Toss transport with an account-bootstrap read that omits the account header, then keep normalization and synchronization in `integrations/toss/observation.py`. Persist only normalized, redacted observations through a version-2 SQLite migration and expose the workflow through the JSON-only CLI; no unauthenticated broker-reading API route is added.

**Tech Stack:** Python 3.12, SQLite, httpx.MockTransport, dataclasses, SHA-256 fingerprints, pytest, Typer, ruff, uv.

---

## Preconditions and boundary

- Work on branch `codex/phase-1-toss-account-observation`, based on the completed Phase 0 branch.
- Never use live credentials in tests or fixtures. Live CLI commands are user-invoked only.
- Do not persist raw Toss response bodies, account numbers, `accountSeq`, access tokens, or secrets.
- Do not add order creation, modification, cancellation, sizing, or execution fields to any command.

## File responsibility map

### Create

- `integrations/toss/observation.py` — official response normalization, reconciliation, pagination, quality classification, and stable fingerprints.
- `storage/account_observation_store.py` — immutable snapshot inserts, idempotent lookup, latest-complete selection, and read serialization.
- `tests/test_toss_observation.py` — mocked official response fixtures and normalization/sync tests.
- `tests/test_account_observation_store.py` — schema persistence, idempotency, quality preservation, and redaction tests.
- `tests/test_toss_cli.py` — JSON-only health/sync/snapshot command tests.

### Modify

- `storage/schema.py` — add migration 2 tables and indexes without changing migration 1.
- `integrations/toss/auth.py` — permit account bootstrap reads without `X-Tossinvest-Account`.
- `integrations/toss/__init__.py` — export observation types and service.
- `cli.py` — add `toss-health`, `toss-sync`, and `toss-snapshots` commands.
- `tests/test_toss_transport.py` — cover account bootstrap header omission.
- `tests/test_database_migrations.py` — assert version 2 tables appear and legacy data remains.
- `README.md` — document Phase 1 commands and normalized snapshot boundary.

## Task 1: Add schema version 2 observation tables

**Files:** `storage/schema.py`, `tests/test_database_migrations.py`

- [ ] Write tests that a fresh database reaches version 2, a version-1 database gains all five broker tables, and legacy portfolio rows remain unchanged.
- [ ] Add `MIGRATION_2_SQL` with these exact tables:

```sql
CREATE TABLE IF NOT EXISTS broker_account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL,
    sync_started_at TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('complete', 'partial', 'stale', 'failed')),
    is_current_evaluable INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT NOT NULL,
    source_timestamps_json TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    reconciliation_json TEXT NOT NULL,
    total_value_krw REAL,
    invested_value_krw REAL,
    cash_value_krw REAL,
    UNIQUE(account_alias, source_fingerprint)
);

CREATE TABLE IF NOT EXISTS broker_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    market_country TEXT NOT NULL,
    currency TEXT NOT NULL,
    quantity REAL NOT NULL,
    last_price REAL NOT NULL,
    average_purchase_price REAL NOT NULL,
    market_value_native REAL NOT NULL,
    market_value_krw REAL NOT NULL,
    cost_native REAL NOT NULL,
    cost_krw REAL NOT NULL,
    profit_loss_native REAL NOT NULL,
    profit_loss_krw REAL NOT NULL,
    daily_profit_loss_native REAL NOT NULL,
    daily_profit_loss_krw REAL NOT NULL,
    UNIQUE(snapshot_id, symbol)
);

CREATE TABLE IF NOT EXISTS broker_cash_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    buying_power_native REAL NOT NULL,
    buying_power_krw REAL NOT NULL,
    UNIQUE(snapshot_id, currency)
);

CREATE TABLE IF NOT EXISTS broker_exchange_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    rate REAL NOT NULL,
    mid_rate REAL,
    valid_from TEXT,
    valid_until TEXT
);

CREATE TABLE IF NOT EXISTS broker_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES broker_account_snapshots(id) ON DELETE CASCADE,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    currency TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    ordered_at TEXT,
    canceled_at TEXT,
    quantity REAL NOT NULL,
    order_price_native REAL,
    order_amount_native REAL,
    filled_quantity REAL NOT NULL,
    average_filled_price_native REAL,
    filled_amount_native REAL,
    commission_native REAL,
    tax_native REAL,
    filled_at TEXT,
    settlement_date TEXT,
    UNIQUE(snapshot_id, order_id)
);

CREATE INDEX IF NOT EXISTS idx_broker_snapshots_latest
    ON broker_account_snapshots(account_alias, synced_at, id);
CREATE INDEX IF NOT EXISTS idx_broker_holdings_snapshot
    ON broker_holdings(snapshot_id, symbol);
CREATE INDEX IF NOT EXISTS idx_broker_orders_snapshot
    ON broker_orders(snapshot_id, order_id);
```

- [ ] Set `LATEST_SCHEMA_VERSION = 2`, add `MIGRATIONS[2]`, and leave migration 1 text unchanged.
- [ ] Run focused migration tests and commit `feat: add broker observation schema`.

## Task 2: Make account bootstrap and response normalization explicit

**Files:** `integrations/toss/auth.py`, `integrations/toss/observation.py`, `tests/test_toss_transport.py`, `tests/test_toss_observation.py`

- [ ] Add a test that `reader.get_json('/api/v1/accounts', include_account_header=False)` sends OAuth authorization but no `x-tossinvest-account` header; existing account-context reads still send it.
- [ ] Implement `include_account_header: bool = True` in `TossAuthorizedReader.get_json`; omit the account header only when false.
- [ ] Define frozen dataclasses `NormalizedHolding`, `NormalizedCash`, `NormalizedFxRate`, `NormalizedOrder`, `NormalizedSnapshot` and `SyncState` literal values.
- [ ] Parse all decimal strings with `Decimal`, convert to finite floats only at the persistence boundary, and reject missing/negative quantities or rates with a diagnostic error.
- [ ] Normalize `HoldingsOverview.items`, `BuyingPowerResponse.result`, `ExchangeRateResponse.result`, and `PaginatedOrderResponse.result` into the dataclasses. Preserve native currency values and calculate KRW values using USD/KRW `rate` only for USD records.
- [ ] Reconcile each native-currency holding sum against the overview `marketValue.krw`/`marketValue.usd` within `Decimal('0.01')`; verify requested cash response currency and positive USD/KRW rate. Mark a mismatch as `partial`.
- [ ] Paginate closed orders with `status=CLOSED`, `limit=100`, optional `from`/`to`, and `cursor`; stop on `hasNext == false`, reject repeated cursors, and deduplicate by `orderId` while preserving first-seen order.
- [ ] Build the fingerprint from the normalized holdings, cash, FX, and order dictionaries with `json.dumps(normalized_payload, sort_keys=True, separators=(',', ':'))` and `hashlib.sha256`, excluding sync timestamps and account identifiers.
- [ ] Add tests for exact official fixtures, USD conversion, mismatched currency, holding reconciliation, repeated cursor, duplicate order, and stable fingerprint.
- [ ] Run focused observation tests and commit `feat: normalize toss account observations`.

## Task 3: Implement sync orchestration and immutable storage

**Files:** `integrations/toss/observation.py`, `storage/account_observation_store.py`, `tests/test_account_observation_store.py`, `tests/test_toss_observation.py`

- [ ] Implement `TossObservationService.sync(from_date=None, to_date=None, max_order_pages=100)`:
  1. Load `TossApiConfig.from_env` and issue a token.
  2. Read `/api/v1/accounts` without account header and require the configured sequence to match a `BROKERAGE` account.
  3. Read holdings, KRW buying power, USD buying power, USD/KRW exchange rate, and paginated closed orders.
  4. Classify `complete`, `partial`, or `failed`; use `stale` only when a source timestamp is present and older than the freshness window.
  5. Persist normalized values through `insert_snapshot`, or return the existing row for an identical fingerprint.
- [ ] Treat account/holdings failure as `failed`; retain a snapshot with diagnostics only when normalized holdings are available. Treat cash/order/rate failures as `partial` and never promote it.
- [ ] Implement `insert_snapshot` in one transaction: `INSERT OR IGNORE` parent, insert child rows only for a new parent, set prior complete rows to `is_current_evaluable=0` only when the new state is `complete`, and return serialized normalized data without account identifiers.
- [ ] Implement `latest_complete(account_alias='toss-brokerage')`, `list_snapshots(limit=20)`, and `get_snapshot(snapshot_id)` with child rows and no raw secrets/account numbers.
- [ ] Add tests proving identical syncs are idempotent, partial/failure snapshots do not replace the latest complete snapshot, and all persisted/output fields exclude account numbers, account sequence, token, and secret.
- [ ] Run focused sync/storage tests and commit `feat: persist immutable toss observations`.

## Task 4: Add JSON-only CLI health, sync, and snapshot inspection

**Files:** `cli.py`, `tests/test_toss_cli.py`, `README.md`

- [ ] Add `toss-health` that runs config, OAuth, account discovery, and account-sequence match checks without persistence; output `{ok, command, checks, account_count, error}` only.
- [ ] Add `toss-sync` with `--from`, `--to`, and `--max-order-pages`; output the normalized snapshot and `{ok, command, snapshot_id, state, error}`.
- [ ] Add `toss-snapshots` with `--latest`, `--snapshot-id`, and `--limit`; output local normalized snapshots only.
- [ ] Route all exceptions through sanitized `CliError` stages. Never print exception reprs containing config or raw response bodies.
- [ ] Add `README.md` examples showing `ips-pilot toss-health`, `ips-pilot toss-sync`, and `ips-pilot toss-snapshots --latest`; document that sync is read-only and that only complete snapshots are evaluable.
- [ ] Add `CliRunner` tests with monkeypatched service/config; assert exactly one JSON object and no token/account number in stdout.
- [ ] Run focused CLI tests and commit `feat: add toss observation cli commands`.

## Task 5: Phase 1 adversarial completion gate

- [ ] Run `rtk uv run pytest -q` and require all tests to pass.
- [ ] Run ruff checks on every changed Python file.
- [ ] Prove order mutation remains unreachable:

```bash
rtk rg -n "place_order|create_order|cancel_order|modify_order|POST.*orders|DELETE.*orders|PATCH.*orders" integrations api cli.py tests/test_toss_observation.py tests/test_toss_cli.py
```

Only negative tests and the existing allowlist references may match.

- [ ] Prove account identifiers and secrets are absent from persistence/output code:

```bash
rtk rg -n "accountNo|account_seq|TOSS_OPEN_API_CLIENT_SECRET|access_token|X-Tossinvest-Account" storage api frontend cli.py integrations/toss tests/test_toss_observation.py tests/test_toss_cli.py
```

Matches must be limited to environment/header handling and redaction assertions; no SQLite columns or CLI/API output may contain them.

- [ ] Run `rtk git diff --check`, inspect `rtk git status --short`, and commit any concrete gate-only correction with a literal file list.
- [ ] Report schema version, sync states, pagination/currency reconciliation coverage, complete-snapshot promotion behavior, test results, and the remaining Phase 2 boundary. Stop after Phase 1; do not implement performance history or cash-policy signals in this plan.
