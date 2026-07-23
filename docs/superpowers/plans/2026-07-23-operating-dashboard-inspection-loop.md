# Operating Dashboard and Inspection Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the approved Toss-only operating design into a deterministic inspection engine, immutable evaluation history, JSON CLI, and read-only local dashboard.

**Architecture:** Keep Toss snapshots and Phase 2 performance history as the only broker facts. Store app-owned target policy versions separately, calculate all statuses in one framework-independent inspection service, persist a replayable evaluation run, and expose that result through CLI and a read-only FastAPI/React workbench.

**Tech Stack:** Python 3.12, SQLite migrations, Typer, FastAPI/Pydantic, React/TypeScript/Vite, local CSS, lucide-react, native SVG, pytest, ruff.

---

### Task 1: Extend the Toss-only persistence contract

**Files:**
- Modify: `storage/schema.py`
- Modify: `storage/policy_store.py`
- Create: `storage/evaluation_store.py`
- Test: `tests/test_database_migrations.py`
- Test: `tests/test_policy_store.py`
- Test: `tests/test_evaluation_store.py`

- [ ] **Step 1: Add the v5 migration test.** Assert a fresh database and a v4 fixture contain `ips_evaluation_runs`, with foreign keys, the allowed `complete/not_evaluable/failed` states, and the source/policy/performance fingerprint columns.
- [ ] **Step 2: Add `MIGRATION_5_SQL` and set `LATEST_SCHEMA_VERSION = 5`.** Create the immutable run table with account alias, snapshot id, optional performance run id, policy version/hash, profile snapshot JSON/hash, engine version, state/reason, result JSON, evaluation fingerprint, timestamps, and deterministic indexes. Register migration 5 without changing Toss evidence tables.
- [ ] **Step 3: Expand `DEFAULT_POLICY` with performance and cadence metadata.** Keep the existing cash/layer values, add `performance.annual_return_target = 0.10`, `measurement = trailing_12_month_twr`, `minimum_history_days = 365`, and weekly/monthly cadence. Do not add invented instrument targets.
- [ ] **Step 4: Implement evaluation-store insert/get/latest helpers.** Canonicalize JSON, make duplicate fingerprints idempotent, return the actual ids and decoded result, and reject malformed state/fingerprint inputs.
- [ ] **Step 5: Run focused migration/store tests.** `rtk uv run pytest -q tests/test_database_migrations.py tests/test_policy_store.py tests/test_evaluation_store.py` must pass.
- [ ] **Step 6: Commit.** `git add storage/schema.py storage/policy_store.py storage/evaluation_store.py tests/test_database_migrations.py tests/test_policy_store.py tests/test_evaluation_store.py && git commit -m "feat: persist Toss inspection evaluations"`.

### Task 2: Implement policy validation and activation

**Files:**
- Modify: `storage/policy_store.py`
- Create: `services/policy_validation.py`
- Modify: `storage/instrument_profile_store.py` only if identity normalization is needed
- Test: `tests/test_policy_validation.py`

- [ ] **Step 1: Write table-driven failing tests.** Cover non-finite/out-of-range values, minimum-target-maximum ordering, layer target sum, duplicate `(market_country, symbol)` identities, grouped instrument target/range feasibility, unseen Toss identities, and activation compare-and-swap conflicts.
- [ ] **Step 2: Implement `validate_policy(policy, observed_identities)` returning canonical normalized policy plus structured errors.** Normalize country/symbol/layer/thesis casing, require all policy sections, use a documented `1e-9` sum tolerance, and reject targets for identities not present in any Toss snapshot. Existing default policy remains valid as a template but is non-evaluable until held instruments receive targets/profiles.
- [ ] **Step 3: Implement `activate_policy(policy, expected_current_version)` atomically.** Validate first, confirm the active version exactly matches the expected version, supersede it, insert one immutable new version, and return the new version/hash. Never partially persist instruments or mutate broker facts.
- [ ] **Step 4: Implement template generation.** Read the selected/latest complete Toss snapshot, emit all observed identities with null profile/target placeholders, and include the active policy metadata without copying quantities, prices, or values into the policy file.
- [ ] **Step 5: Run focused tests and commit.** `rtk uv run pytest -q tests/test_policy_validation.py tests/test_policy_store.py` must pass, then commit with `feat: validate and activate Toss target policies`.

### Task 3: Build the deterministic inspection engine

**Files:**
- Create: `services/inspection_engine.py`
- Modify: `services/account_projection.py` only for missing explicit denominator fields
- Modify: `storage/performance_store.py` only for a stable performance summary helper
- Test: `tests/test_inspection_engine.py`

- [ ] **Step 1: Write failing return-window and denominator tests.** Assert exact 365-day history is eligible, shorter history is `Watch` with `annual_return_history_insufficient`, cumulative TWR stays separate, cash uses gross value, and layers/instruments use invested value.
- [ ] **Step 2: Write table-driven status tests for every approved matrix row.** Include source failure, unclassified holdings, range breaches, watched thesis, broken thesis without hard breach, broken thesis with hard maximum breach, annual underperformance, and gain/loss/cash/shortfall-only guardrails. Assert only `OK`, `Watch`, `Review`, `Action` appear and no result key contains order sizing or execution semantics.
- [ ] **Step 3: Implement pure evaluation functions.** Accept projection, performance run, active policy, and canonical profiles; return source facts, performance facts, cash gap, separate layer and instrument rows, stable triggers, plain-language meaning, allowed next-step posture, and Review Queue. Use severity precedence `Action > Review > Watch > OK`, stable sorting, and `not_evaluable` source gates.
- [ ] **Step 4: Implement deterministic fingerprints.** Hash the canonical snapshot/performance/policy/profile/engine inputs and ensure identical inputs produce byte-identical result JSON and evaluation fingerprint.
- [ ] **Step 5: Run the engine suite and commit.** `rtk uv run pytest -q tests/test_inspection_engine.py tests/test_account_projection.py` must pass, then commit with `feat: add deterministic IPS inspection engine`.

### Task 4: Expose policy and inspection through the JSON CLI

**Files:**
- Modify: `cli.py`
- Modify: `Taskfile.yml`
- Modify: `README.md`
- Test: `tests/test_inspection_cli.py`

- [ ] **Step 1: Add failing CLI tests.** Exercise `policy show --active`, `policy template`, `policy validate --file`, `policy activate --file --expected-current-version`, `inspection run --snapshot-id`, and `inspection show --latest/--run-id`; assert one JSON object on stdout, actual snapshot/performance/policy ids, and machine-readable failures.
- [ ] **Step 2: Add Typer subcommands.** Initialize the Toss-only database, call framework-independent stores/services, serialize sanitized errors, and reject mutually exclusive options. `inspection run` must reuse an existing fingerprint and must not silently choose a failed snapshot.
- [ ] **Step 3: Add Taskfile shortcuts and README examples.** Document `toss-policy-template`, `toss-policy-validate`, `toss-inspection-run`, `toss-inspection-latest`; state that output is inspection-only and annual target is 10% trailing-12-month TWR after 365 supported days.
- [ ] **Step 4: Run CLI tests and commit.** `rtk uv run pytest -q tests/test_inspection_cli.py tests/test_cli.py` must pass, then commit with `feat: expose Toss inspection CLI`.

### Task 5: Add the read-only Toss local API

**Files:**
- Create: `api/__init__.py`
- Create: `api/app.py`
- Create: `api/schemas.py`
- Modify: `pyproject.toml`
- Test: `tests/test_api_contract.py`

- [ ] **Step 1: Add FastAPI/Pydantic dependencies and failing contract tests.** Test source health, latest snapshot/projection, performance, active policy, profiles, latest evaluation, and Review Queue; test missing data, stale/failed snapshots, and sanitized errors.
- [ ] **Step 2: Implement loopback-only API routes.** Return existing service/store shapes without duplicating status logic. Create a per-process HttpOnly SameSite session cookie, never serialize Toss credentials/account secrets, and leave all write routes absent in this phase.
- [ ] **Step 3: Add a local server command.** Add `web`/`serve` CLI entry that binds `127.0.0.1` by default and serves the built frontend when present; Vite development proxy remains API-only.
- [ ] **Step 4: Run API tests and commit.** `rtk uv run pytest -q tests/test_api_contract.py` must pass, then commit with `feat: add Toss read-only local API`.

### Task 6: Build the read-only operating dashboard

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx` or `frontend/src/lib/api.test.ts`

- [ ] **Step 1: Scaffold the Toss-only React/Vite app.** Use TypeScript, local CSS, lucide-react, and native SVG only; do not restore removed generic portfolio contracts or add a charting framework.
- [ ] **Step 2: Implement the three-column workbench.** Left navigation covers overview/performance/allocation/profiles/policy/source health; center shows source state, actual snapshot id, principal, account value, cumulative TWR, annual target/history state, cash band, separate layer/instrument gaps; right shows the ordered Review Queue with raw trigger, meaning, and verification task.
- [ ] **Step 3: Implement presentation states and sorting.** Render backend statuses only, preserve API order when inactive, sort numeric raw values client-side with missing values last, and visibly handle loading, partial, stale, failed, unclassified, insufficient-history, and invalid-policy states. Do not add mutation controls.
- [ ] **Step 4: Verify narrow and desktop layouts.** Run `npm run typecheck`, `npm run build`, and focused frontend tests; inspect the running page at loopback when available. Commit with `feat: add Toss operating dashboard`.

### Task 7: Adversarial verification and handoff

**Files:**
- Modify: `AGENTS.md` only if a durable implementation lesson is discovered
- Modify: `README.md` for final run instructions
- Test: repository-wide test and forbidden-language scans

- [ ] **Step 1: Run the complete backend suite and lint.** `rtk uv run pytest -q` and `rtk uv run ruff check .` must pass.
- [ ] **Step 2: Run frontend typecheck/build and OpenAPI contract checks.** Ensure generated or checked response types match the backend and no browser secret enters the bundle.
- [ ] **Step 3: Scan for forbidden compatibility paths.** Search for manual portfolio uploads, yfinance, generic broker/Japan fallbacks, order-sized fields, execution flags, and direct trading recommendation routes.
- [ ] **Step 4: Perform an adversarial review against the approved risks.** Confirm no external cash assumption, no annualization before 365 days, immutable profile snapshots, actual latest snapshot ids, gross/invested denominator labels, and `Action` only for broken thesis plus hard breach.
- [ ] **Step 5: Commit the verified implementation and report the exact commands, tests, and any explicitly deferred Phase 4–6 work.**
