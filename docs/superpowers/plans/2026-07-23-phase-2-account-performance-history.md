# Phase 2 Account-Value and Performance History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible account-value and performance history from the user-confirmed Toss snapshot-4 baseline while preserving existing portfolio evaluation behavior.

**Architecture:** Keep Phase 1 account snapshots immutable. Add schema-v3 tracking, flow-decision, execution-ledger, and versioned performance-run tables. Put calculations in a pure service, persistence in a dedicated store, and expose explicit local Typer commands under `ips-pilot performance`.

**Tech Stack:** Python 3.12, SQLite, dataclasses, `Decimal`, Typer, pytest, Ruff.

---

## File Map

- Modify: `storage/schema.py` — migration 3 and `LATEST_SCHEMA_VERSION`.
- Modify: `storage/account_observation_store.py` — deterministic complete-snapshot listing.
- Create: `storage/performance_store.py` — baseline, candidate, decision, run, point, and execution persistence.
- Create: `services/account_performance.py` — pure baseline, reconciliation, moving-average, and TWR calculations.
- Modify: `cli.py` — `performance` Typer subcommands and JSON serialization.
- Modify: `Taskfile.yml` and `README.md` — local refresh/history entry points and usage.
- Modify: `tests/test_database_migrations.py` and `tests/test_account_observation_store.py`.
- Create: `tests/test_account_performance.py`, `tests/test_performance_store.py`, and `tests/test_performance_cli.py`.

## Task 1: Schema v3 migration

**Files:** `storage/schema.py`, `tests/test_database_migrations.py`

- [ ] **Step 1: Add failing migration assertions.** Assert fresh and v2-upgraded databases contain `account_tracking_baselines`, `account_cash_flow_candidates`, `account_cash_flow_decisions`, `account_performance_runs`, `account_performance_points`, and `account_execution_ledger`; assert a v2 broker snapshot survives and `PRAGMA user_version == 3`.
- [ ] **Step 2: Run:**

```bash
rtk uv run pytest tests/test_database_migrations.py -q
```

Expected: failure because schema version remains 2.

- [ ] **Step 3: Implement `MIGRATION_3_SQL`.** Set `LATEST_SCHEMA_VERSION = 3`, add all six tables and indexes from `docs/superpowers/specs/2026-07-23-phase-2-account-performance-history-design.md`, register `MIGRATIONS[3]`, and preserve the existing forward-only transaction wrapper. Keep the exact CHECK enums from the spec for candidate bridge basis, decision classification, run state, and point state.
- [ ] **Step 4: Verify and commit:**

```bash
rtk uv run pytest tests/test_database_migrations.py -q
rtk git diff --check
rtk git add storage/schema.py tests/test_database_migrations.py
rtk git commit -m "feat: add phase two performance schema"
```

## Task 2: Persistence primitives

**Files:** `storage/account_observation_store.py`, `storage/performance_store.py`, `tests/test_account_observation_store.py`, `tests/test_performance_store.py`

- [ ] **Step 1: Test deterministic complete snapshots.** Insert complete and partial fixtures and assert new `list_complete_snapshots(account_alias="toss-brokerage")` excludes non-complete rows and orders by `synced_at ASC, id ASC`.
- [ ] **Step 2: Implement `list_complete_snapshots`.** Reuse `_row_to_snapshot`; query `state='complete'` with the deterministic order and return hydrated dictionaries.
- [ ] **Step 3: Test and implement baseline functions:** `preview_baseline(snapshot_id)`, `create_baseline(snapshot_id, expected_principal_krw, confirmed_at=None)`, and `get_baseline(account_alias="toss-brokerage")`. Require complete/current/non-null total value, compare Decimal values within 0.01 KRW, reject a second baseline, and insert the immutable row transactionally.
- [ ] **Step 4: Test and implement candidate/decision functions:** `insert_cash_flow_candidate`, `list_cash_flow_candidates`, `append_cash_flow_decision`, and `latest_cash_flow_decisions`. Candidate fingerprint repeats return the existing row; decisions are append-only and latest selection is by `decided_at, id`.
- [ ] **Step 5: Test and implement run functions:** `insert_performance_run`, `get_performance_run`, and `latest_performance_run`. The input fingerprint must return an existing fully hydrated run; parent, point, and execution rows insert in one transaction; no earlier row is updated or deleted.
- [ ] **Step 6: Verify and commit:**

```bash
rtk uv run pytest tests/test_account_observation_store.py tests/test_performance_store.py -q
rtk git diff --check
rtk git add storage/account_observation_store.py storage/performance_store.py tests/test_account_observation_store.py tests/test_performance_store.py
rtk git commit -m "feat: persist phase two performance inputs"
```

## Task 3: Pure calculation engine

**Files:** `services/account_performance.py`, `tests/test_account_performance.py`

- [ ] **Step 1: Add frozen dataclasses `TrackingBaseline`, `CashFlowCandidate`, and `PerformanceProjection`; add failing tests for the zero baseline point, no-flow 8.2m→8.4m interval, 15,000 KRW residual, unresolved-flow non-evaluable state, duplicate order collapse, and conflicting order blocking.
- [ ] **Step 2: Implement `validate_baseline_snapshot`, `materiality_threshold_krw`, and `fingerprint_inputs`.** Use `Decimal(str(value))`, reject non-finite values, and calculate the threshold as `max(10000, previous_total_krw * 0.0001)`.
- [ ] **Step 3: Implement `canonical_executions` and `apply_execution`.** Collapse identical `order_id` rows; conflict on differing final execution fields; seed actual basis from baseline `cost_native` and tracking basis from baseline `market_value_native`; process non-zero fills by `(filled_at or ordered_at, order_id)`; reject overselling instead of creating negative quantities.
- [ ] **Step 4: Implement `detect_cash_candidates` and `detect_quantity_issues`.** Compare both fill-time and settlement-time bridges; emit a candidate only when both exceed materiality; record missing timestamp basis instead of inventing one; flag unexplained quantity changes.
- [ ] **Step 5: Implement `build_projection`.** Apply confirmed external deposits/withdrawals to tracking principal; keep other explicit classifications as performance; calculate account gain, nullable simple return, interval/segment TWR, current cost basis, actual/tracking realized P&L, and diagnostic FX remeasurement; leave insufficient evidence non-evaluable.
- [ ] **Step 6: Verify and commit:**

```bash
rtk uv run pytest tests/test_account_performance.py -q
rtk uv run ruff check services/account_performance.py tests/test_account_performance.py
rtk uv run ruff format --check services/account_performance.py tests/test_account_performance.py
rtk git diff --check
rtk git add services/account_performance.py tests/test_account_performance.py
rtk git commit -m "feat: calculate account performance history"
```

## Task 4: Projection orchestration

**Files:** `storage/performance_store.py`, `services/account_performance.py`, `tests/test_performance_store.py`

- [ ] **Step 1: Add an idempotent refresh test.** Insert a baseline and later complete snapshot, call `refresh_performance()`, assert points contain both source IDs, call it again, and assert the same run ID with no extra run/point/ledger rows.
- [ ] **Step 2: Implement `refresh_performance(account_alias="toss-brokerage")`.** Load the immutable baseline and complete snapshots, collect latest decisions, build candidates and projection, insert candidates transactionally, recompute the fingerprint including candidate IDs/decisions, and insert or return the immutable run. Never modify broker snapshots or existing portfolio tables.
- [ ] **Step 3: Sanitize regular history hydration.** Return baseline metadata, points, data-quality diagnostics, candidate summaries, and execution counts; omit raw account identifiers, broker payloads, and the execution ledger `side` from regular history output.
- [ ] **Step 4: Verify and commit:**

```bash
rtk uv run pytest tests/test_performance_store.py -q
rtk git diff --check
rtk git add services/account_performance.py storage/performance_store.py tests/test_performance_store.py
rtk git commit -m "feat: orchestrate immutable performance runs"
```

## Task 5: CLI, Taskfile, and README

**Files:** `cli.py`, `Taskfile.yml`, `README.md`, `tests/test_performance_cli.py`

- [ ] **Step 1: Add CLI tests with `CliRunner`.** Cover `performance baseline-preview --snapshot-id 4`, `baseline-confirm --snapshot-id 4 --expected-principal-krw 120802745.17802304`, `refresh`, `candidates --run-id 1`, `decide-flow --candidate-id 1 --classification internal_fx`, and `history --latest`. Assert one JSON object per command, machine-readable errors, and absence of `accountNo`, `accountSeq`, `access_token`, and `client_secret`.
- [ ] **Step 2: Register `performance_app = typer.Typer(...)` and implement commands `baseline-preview`, `baseline-confirm`, `refresh`, `candidates`, `decide-flow`, and `history`. Use `initialize_database()` and `_exit_with_command_error`; reject `--latest` with `--run-id`; validate the seven schema classifications before persistence; keep output JSON-only.
- [ ] **Step 3: Add `toss-performance-refresh` and `toss-performance-history` tasks. Document `toss-sync` → baseline preview → baseline confirm → refresh → history; state Phase 2 does not mutate Toss or existing portfolio snapshots.
- [ ] **Step 4: Verify and commit:**

```bash
rtk uv run pytest tests/test_performance_cli.py -q
rtk uv run ruff check cli.py tests/test_performance_cli.py
rtk uv run ruff format --check cli.py tests/test_performance_cli.py
rtk git diff --check
rtk git add cli.py Taskfile.yml README.md tests/test_performance_cli.py
rtk git commit -m "feat: expose performance history cli"
```

## Task 6: Full verification and adversarial review

**Files:** all Phase 2 files and focused tests

- [ ] **Step 1: Run focused tests:**

```bash
rtk uv run pytest tests/test_database_migrations.py tests/test_account_observation_store.py tests/test_account_performance.py tests/test_performance_store.py tests/test_performance_cli.py -q
```

- [ ] **Step 2: Run guardrail searches:**

```bash
rtk rg -n "POST /api/v1/orders|/cancel|/modify|order_size|execute" services/account_performance.py storage/performance_store.py cli.py
rtk rg -n "accountNo|accountSeq|access_token|client_secret" services/account_performance.py storage/performance_store.py cli.py tests/test_performance_cli.py
```

Expected: no new order mutation path, order sizing, direct execution output, or raw secret/account identifier.

- [ ] **Step 3: Run the complete verification set:**

```bash
rtk uv run pytest -q
rtk uv run ruff check cli.py storage/schema.py storage/account_observation_store.py storage/performance_store.py services/account_performance.py tests/test_database_migrations.py tests/test_account_observation_store.py tests/test_account_performance.py tests/test_performance_store.py tests/test_performance_cli.py
rtk uv run ruff format --check cli.py storage/schema.py storage/account_observation_store.py storage/performance_store.py services/account_performance.py tests/test_database_migrations.py tests/test_account_observation_store.py tests/test_account_performance.py tests/test_performance_store.py tests/test_performance_cli.py
rtk git diff --check
```

- [ ] **Step 4: Review final boundaries.** Confirm baseline, current cost basis, and tracking principal are distinct; incomplete evidence remains non-evaluable; Phase 0/1 tests remain green; and `rtk git status --short` is clean. If a concrete integration correction is required, stage only the named files reported by that review and commit it as `test: verify phase two integration boundaries`; otherwise make no additional commit.

## Completion Checklist

- [ ] Schema v3 migrates without data loss.
- [ ] Baseline confirmation is explicit, exact, and immutable.
- [ ] Complete snapshots are the only performance inputs.
- [ ] Cash-flow candidates and decisions are append-only and replayable.
- [ ] Actual and tracking moving-average cost ledgers remain distinct.
- [ ] TWR becomes non-evaluable when evidence is insufficient.
- [ ] Projection runs are immutable and fingerprint-idempotent.
- [ ] CLI output is machine-readable and sanitized.
- [ ] Existing Phase 0/1 and portfolio evaluation tests remain green.
