# Phase 2.5 Toss-only Foundation Convergence Design

**Date:** 2026-07-23  
**Status:** Approved  
**Product:** IPS Pilot

## Context

Phases 0–2 added a verified, read-only Toss Securities integration, immutable
account snapshots, and reproducible account-performance history alongside the
older generic portfolio workflow. The product is now intentionally narrowing:
Toss Securities becomes the only source of account facts and market data.

The old manual rows, CSV/TSV upload, user-entered ticker workflow, yfinance
price history, generic saved portfolios, and Japan-account accommodations are
not retained as compatibility paths. Phase 2.5 is a clean convergence phase
between the completed account-history work and the cash-policy MVP.

IPS Pilot remains an inspection workbench. The source change does not authorize
orders, order sizing, automatic profit taking, automatic stop loss, or direct
buy/sell recommendations. Human-facing judgments continue to use only `OK`,
`Watch`, `Review`, and `Action`, where `Action` means inspect a possible
exceptional intervention.

## Outcome

After Phase 2.5:

- Toss Open API observations are the only account-data source;
- immutable Toss account snapshots are the only factual portfolio history;
- app-owned IPS metadata is keyed to Toss-observed instruments;
- the active IPS policy is versioned and replayable;
- one deterministic projection converts a complete Toss snapshot into gross,
  invested, cash, layer, and instrument weights;
- the CLI is the only user-facing runtime until the Toss-only workbench is
  rebuilt in Phase 6;
- generic API, frontend, persistence, analysis, and market-data code is gone;
- the runtime dependency graph no longer includes yfinance, pandas, NumPy,
  SciPy, FastAPI, Pydantic, PyYAML, multipart parsing, or signed sessions.

## Non-goals

- Implementing Phase 3 cash-policy statuses.
- Fetching Toss candles or market indicators; that belongs to Phase 4.
- Rebuilding the browser workbench; that belongs to Phase 6.
- Migrating generic portfolio rows into Toss profiles.
- Preserving a read-only legacy viewer or compatibility API.
- Placing, preparing, modifying, canceling, or sizing an order.

## Source-of-truth Boundaries

### Toss account facts

The following values come only from normalized Toss Open API responses:

- account holdings and quantities;
- current prices, average purchase prices, market values, and cost basis;
- currency and market-country identifiers;
- KRW and foreign-currency buying power;
- exchange rates applied to account normalization;
- closed orders, fills, commissions, taxes, and settlement evidence.

Users cannot create or edit these facts. Each synchronization produces an
immutable snapshot. The current account state is the newest `complete` snapshot
selected by `(synced_at, id)`. `partial`, `stale`, and `failed` observations
remain diagnostic evidence and cannot become evaluable current state.

### App-owned IPS state

The app persists only information the broker cannot supply:

- `core`, `satellite`, or `experiment` classification;
- thesis state: `unknown`, `valid`, `watch`, or `broken`;
- a short thesis note;
- versioned cash and layer policy limits;
- later evaluation and human-decision records linked to exact source snapshot
  and policy IDs.

This state is not a second portfolio. It annotates Toss-observed instruments and
cannot override quantity, value, cost, return, or cash.

## Schema v4

### Tables retained unchanged

Phase 1 and Phase 2 evidence remains authoritative:

- `broker_account_snapshots`
- `broker_holdings`
- `broker_cash_observations`
- `broker_exchange_rates`
- `broker_orders`
- `account_tracking_baselines`
- `account_cash_flow_candidates`
- `account_cash_flow_decisions`
- `account_performance_runs`
- `account_performance_points`
- `account_execution_ledger`

### Tables removed

Schema v4 drops all generic-portfolio persistence:

- `portfolios`
- `assets`
- `thesis_statuses`
- `portfolio_snapshots`
- `portfolio_current_states`
- `snapshot_positions`
- `snapshot_evaluation_runs`
- `ips_target_allocations`
- `ips_action_priorities`
- `ips_rules`
- `journal_entries`
- historical `analysis_runs`, when present

No row is migrated or archived. Foreign-key children are dropped before their
parents inside the v4 transaction.

### `ips_instrument_profiles`

An IPS profile uses the composite Toss identity
`(account_alias, market_country, symbol)`. A display name is not an identity and
may change between snapshots.

Required fields are:

- `account_alias`
- `market_country`
- `symbol`
- `layer`, constrained to `core`, `satellite`, or `experiment`
- `thesis_status`, constrained to `unknown`, `valid`, `watch`, or `broken`
- `thesis_note`
- `created_at`
- `updated_at`

A profile can be created only after the same identity has appeared in a stored
Toss holding. Profiles remain when a position disappears so a later re-entry
does not silently lose its IPS classification.

No existing ticker row is auto-mapped. A newly observed holding without a
profile remains explicitly unclassified. Phase 3 will surface that condition as
`Review`; Phase 2.5 exposes it as projection data quality rather than inventing
an evaluation status early.

### `ips_policy_versions`

Policy is immutable and versioned per account alias. Each row stores:

- monotonic version;
- canonical policy JSON;
- SHA-256 policy hash;
- `active` or `superseded` state;
- creation timestamp.

The initial policy contains the approved cash reserve
`minimum=0.10`, `target=0.15`, `maximum=0.20` and the existing first-class layer
limits. Only one version may be active for an account. Later approvals create a
new version and supersede the old row; they never update history in place.

## Destructive Migration Policy

The v4 cutover is intentionally destructive for generic portfolio data.

- The migration runs in `BEGIN IMMEDIATE` and rolls back the entire v4 step on
  any SQL failure.
- Initialization runs SQLite integrity and foreign-key checks before accepting
  the database for use.
- The application stops creating adjacent long-lived `.bak` files before
  migrations.
- Existing, precisely identified migration backups are removed only in the
  explicit Phase 2.5 cleanup step after v4 integrity, Toss snapshot, baseline,
  and performance verification succeeds.
- Unknown files and arbitrary database tables are not deleted.

The removal is not reversible through the application. Git preserves source
history, but deleted local generic portfolio data is intentionally not retained.

## Toss Portfolio Projection

`services/account_projection.py` becomes the sole bridge between persisted
broker facts and later IPS judgment.

The projection accepts an explicit snapshot ID or resolves the latest complete
snapshot. It rejects any non-complete snapshot and returns:

- source snapshot ID and timestamps;
- total account, invested, and deployable-cash values in KRW;
- gross cash weight against total account value;
- each holding's gross weight against total account value;
- each holding's invested weight against invested market value;
- profile layer and thesis state when present;
- aggregated layer invested weights;
- a deterministic list of unclassified instrument identities;
- data-quality and reconciliation evidence copied from the source snapshot.

Denominators stay explicit. Cash policy uses gross weight. Layer and asset
inspection uses invested weight. Missing KRW values, zero or inconsistent
denominators, non-complete state, or missing profiles never default to zero or
`OK`.

The projection is pure and is not persisted. Later evaluation runs persist the
source snapshot ID, active policy version ID, engine version, and derived result
so they can be replayed.

## User-facing Runtime

### CLI retained

- `toss-health`
- `toss-sync`
- `toss-snapshots`
- all `performance` subcommands

Each command continues to emit exactly one JSON object on stdout and sanitized,
machine-readable errors.

### CLI added

- `profiles list [--snapshot-id ID]`
- `profiles set --symbol SYMBOL --market-country COUNTRY --layer LAYER
  --thesis-status STATUS [--note TEXT]`
- `account-view [--snapshot-id ID]`

`profiles set` changes only IPS annotation. It verifies that the instrument was
observed by Toss and never changes a broker snapshot. `account-view` emits the
pure Toss portfolio projection and contains no judgment or trade language.

### Runtime removed

- `evaluate`, `agent-brief`, `review-queue`, and `risk`
- generic `portfolios` and `snapshots` command groups
- file, text, CSV/TSV, portfolio-ID, and manual-snapshot inputs
- the FastAPI application, session middleware, and all v1 routes
- the React/Vite frontend and its build tasks

The temporary Phase 2.5 product is intentionally CLI-only. Keeping the old web
surface hidden or read-only would preserve its generic contracts and undermine
the clean cut. Phase 6 creates a new Toss-only API and workbench from the new
projection and evaluation contracts.

## Code and Dependency Removal

The cutover deletes the generic analysis and evaluation implementation rather
than leaving dead modules:

- `api/`, `frontend/`, `middleware/`, and `main.py`
- `core/`
- `services/analysis_service.py`
- `services/evaluation_engine.py`
- `services/evaluation_period.py`
- `services/evaluation_status.py`
- `services/evaluation_units.py`
- `services/portfolio_service.py`
- `storage/config_store.py`
- `storage/journal_store.py`
- `storage/portfolio_store.py`
- yfinance and generic metric utilities under `utils/`
- `config/ips.yaml`
- tests that only specify the removed behavior

`OK`, `Watch`, `Review`, and `Action` are reintroduced from one backend source
of truth when Phase 3 adds Toss-only evaluation. They are not kept alive through
the old pandas-based evaluator.

The remaining runtime uses the standard library plus `httpx` and `typer`.
Ruff and pytest remain development dependencies.

## Error and Safety Behavior

- Missing Toss configuration fails before network access.
- A live Toss failure never falls back to manual or yfinance data.
- A missing complete snapshot produces a machine-readable source error.
- An unclassified holding is visible and cannot be evaluated as `OK`.
- Profile writes reject unknown identities and invalid layer or thesis values.
- Policy seed or projection failures cannot mutate broker snapshots.
- The transport allowlist continues to reject every order mutation endpoint.
- No output contains `buy`, `sell`, `execute`, order quantities, or order-sized
  cash gaps as proposed actions.

## Verification

### Migration

- A v3 fixture with generic and Toss rows upgrades to v4.
- Every generic table and `analysis_runs` is absent after the upgrade.
- Every Toss observation, performance baseline, cash-flow decision, performance
  point, and execution-ledger row remains byte-for-byte equivalent.
- A fresh database creates only Toss/performance/profile/policy tables.
- A forced v4 SQL failure leaves schema version and all v3 rows unchanged.
- No new `.bak` file appears.
- SQLite `integrity_check` returns `ok` and `foreign_key_check` is empty.

### Profiles and projection

- A Toss-observed KR and US instrument can be profiled independently.
- An unseen instrument is rejected.
- Invalid layer and thesis values are rejected.
- Profile updates do not change broker rows.
- Gross and invested weights use different documented denominators.
- Unclassified identities are deterministic.
- Partial, stale, and failed snapshots are rejected.
- Zero, missing, and inconsistent totals fail closed.

### Surface and dependency deletion

- CLI help exposes only Toss, performance, profiles, and account-view commands.
- Removed commands return Typer usage errors and no JSON evaluation fallback.
- Repository scans find no yfinance import, manual upload route, generic
  portfolio command, Japanese-account branch, or frontend source.
- The lockfile has no yfinance, pandas, NumPy, SciPy, FastAPI, Pydantic, PyYAML,
  multipart, or itsdangerous runtime dependency.
- The focused Toss and performance suites and the full remaining test suite pass
  without live credentials or market data.

## Adversarial Review Findings

1. **Data loss is intentional but must be bounded.** Only named generic tables
   and named migration backups are deleted; Toss evidence is asserted before
   cleanup.
2. **Ticker relabeling can corrupt IPS metadata.** Profiles use the Toss symbol
   plus market country and require prior broker observation; names are display
   only.
3. **Removing yfinance temporarily removes risk analytics.** Phase 2.5 does not
   fake equivalent analytics from sparse account snapshots. Phase 4 restores
   market context from official Toss candles.
4. **Keeping the old frontend would retain generic contracts.** The web surface
   is deleted and intentionally rebuilt later.
5. **A snapshot alone cannot preserve IPS intent.** Versioned policies and
   instrument profiles remain separate, app-owned state.
6. **New holdings must not inherit optimistic defaults.** Missing profiles are
   explicit and block later `OK` classification.
7. **A clean dependency removal can accidentally affect performance math.** The
   Phase 2 calculator is standard-library based and receives dedicated
   regression coverage before numerical packages are removed.

## Exit Criteria

- Schema v4 contains only Toss/performance evidence and Toss-keyed IPS state.
- Existing Toss snapshot 4, baseline 1, and performance run 1 remain readable
  in the current local database after migration.
- No generic portfolio data or adjacent legacy migration backup remains after
  the explicit cleanup gate.
- `account-view` reproducibly projects the latest complete Toss snapshot.
- All current holdings are either explicitly profiled or listed as
  unclassified.
- The CLI and dependency graph contain no manual, yfinance, Japan-account, API,
  or frontend compatibility path.
- Read-only broker transport and secret-redaction tests remain green.
- Phase 3 can consume the projection without redefining account facts or weight
  denominators.
