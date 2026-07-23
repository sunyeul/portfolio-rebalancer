# Investment Principal Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with TDD checkpoints.

**Goal:** Rename the account principal contract to `investment_principal_krw`, expose backend-owned account profit and principal-relative return, and make Overview follow the investment-principal → account-value → return mental model.

**Architecture:** Preserve the existing baseline and classified external-flow calculation, but rename its persisted performance-point field through a SQLite migration. Derive account profit and return in the backend inspection result from the latest evaluable performance point; the frontend only formats those fields and leaves YTD TWR as a separate annual-objective metric.

**Tech Stack:** Python 3.12, SQLite migrations, pytest, FastAPI/CLI JSON contracts, React + TypeScript + Bun/Vite.

---

### Task 1: Migrate the persisted principal column safely

**Files:**
- Modify: `storage/schema.py`
- Modify: `tests/test_database_migrations.py`

- [ ] **Step 1: Write the failing migration assertions**

Extend migration tests to expect schema version `9`, an
`account_performance_points.investment_principal_krw` column, and no
`tracking_principal_krw` column. Add an upgrade test that creates a version-8
database with one performance point containing `tracking_principal_krw = 8200000`
and asserts the version-9 row contains `investment_principal_krw = 8200000`.

- [ ] **Step 2: Run the focused tests to verify failure**

Run `uv run pytest tests/test_database_migrations.py -q`.
Expected: FAIL because the current schema version is 8 and the old column remains.

- [ ] **Step 3: Add migration 9**

Set `LATEST_SCHEMA_VERSION = 9`, add `MIGRATION_9_SQL`, and register it in
`MIGRATIONS`. Rebuild only `account_performance_points` with the same
columns, foreign keys, indexes, and constraints, replacing
`tracking_principal_krw` with `investment_principal_krw`. Copy every row,
drop the temporary table, and rename it back. Do not add a compatibility alias.

- [ ] **Step 4: Run migration tests to verify preservation**

Run `uv run pytest tests/test_database_migrations.py -q`.
Expected: all migration tests pass, including value preservation and newer-schema refusal.

- [ ] **Step 5: Commit**

```bash
git add storage/schema.py tests/test_database_migrations.py
git commit -m "feat: migrate tracking principal to investment principal"
```

### Task 2: Rename performance calculation and storage contracts

**Files:**
- Modify: `services/account_performance.py`
- Modify: `storage/performance_store.py`
- Modify: `tests/test_account_performance.py`
- Modify: `tests/test_performance_store.py`

- [ ] **Step 1: Write failing principal and return tests**

Update existing point expectations from `tracking_principal_krw` to
`investment_principal_krw`. Add a classified external-flow test:

```python
assert point["investment_principal_krw"] == pytest.approx(initial + flow)
assert point["account_gain_krw"] == pytest.approx(total - initial - flow)
assert point["simple_return"] == pytest.approx(
    point["account_gain_krw"] / point["investment_principal_krw"]
)
```

Also cover a non-positive investment principal by asserting
`simple_return is None`, while `account_gain_krw` remains available when the
account value is valid.

- [ ] **Step 2: Run focused tests to verify failure**

Run `uv run pytest tests/test_account_performance.py tests/test_performance_store.py -q`.
Expected: FAIL on the renamed field and on return calculation when an external flow exists.

- [ ] **Step 3: Rename calculation and persistence keys**

In `build_projection`, rename the point key to `investment_principal_krw`.
Compute current principal as `baseline.initial_principal_krw + cumulative_flow`
and compute `simple_return` as `gain / current_principal` only for finite
positive principal. In `insert_performance_run`, replace the point insert key
with `investment_principal_krw`. Keep `account_gain_krw` as the internal
immutable point fact; the inspection/API layer will expose the human-facing
`account_profit_krw` name.

- [ ] **Step 4: Run focused tests to verify the implementation**

Run `uv run pytest tests/test_account_performance.py tests/test_performance_store.py -q`.
Expected: PASS, with internal cash/holding movements not changing principal and
classified external flows changing it exactly once.

- [ ] **Step 5: Commit**

```bash
git add services/account_performance.py storage/performance_store.py tests/test_account_performance.py tests/test_performance_store.py
git commit -m "feat: calculate investment principal return"
```

### Task 3: Expose backend-owned account principal, profit, and return

**Files:**
- Modify: `services/inspection_engine.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `tests/test_inspection_engine.py`
- Modify: `tests/test_api_contract.py`

- [ ] **Step 1: Write failing inspection contract tests**

Add an evaluable performance fixture whose latest point contains
`investment_principal_krw = 1000000`, `account_gain_krw = 120000`, and
`simple_return = 0.12`. Assert:

```python
assert result["account"]["investment_principal_krw"] == 1000000
assert result["account"]["account_profit_krw"] == 120000
assert result["account"]["account_return"] == pytest.approx(0.12)
assert "tracking_principal_krw" not in result["account"]
```

Add a missing/non-evaluable fixture asserting the new values are unavailable,
never numeric zero. Extend the API test to assert the fields pass through without
frontend/API reclassification.

- [ ] **Step 2: Run focused tests to verify failure**

Run `uv run pytest tests/test_inspection_engine.py tests/test_api_contract.py -q`.
Expected: FAIL because the engine emits only `tracking_principal_krw`.

- [ ] **Step 3: Add the backend result fields**

In `evaluate_inspection`, read the latest evaluable performance point and set:

```python
{
    "total_value_krw": projection.get("total_value_krw"),
    "invested_value_krw": projection.get("invested_value_krw"),
    "cash_value_krw": projection.get("cash_value_krw"),
    "cash_weight_gross": projection.get("cash_weight_gross"),
    "investment_principal_krw": latest_point.get("investment_principal_krw"),
    "account_profit_krw": latest_point.get("account_gain_krw"),
    "account_return": latest_point.get("simple_return"),
}
```

If source or performance is not evaluable, preserve null evidence and do not
fabricate zero. Update the TypeScript `Evaluation` contract with the fields.

- [ ] **Step 4: Run focused tests to verify the implementation**

Run `uv run pytest tests/test_inspection_engine.py tests/test_api_contract.py -q`.
Expected: PASS, with Action/Review classification unchanged because these are
descriptive account facts only.

- [ ] **Step 5: Commit**

```bash
git add services/inspection_engine.py frontend/src/lib/api.ts tests/test_inspection_engine.py tests/test_api_contract.py
git commit -m "feat: expose account principal return contract"
```

### Task 4: Rebuild the Overview mental model in the frontend

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/lib/presentation.ts`
- Modify: `frontend/tests/presentation.test.ts`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write failing presentation tests**

Add tests for formatting zero account profit/return as a real value and missing
account principal/return as `자료 없음`. Keep the existing whole-won assertions.

- [ ] **Step 2: Run frontend tests to verify failure**

Run `bun test frontend/tests/presentation.test.ts`.
Expected: FAIL because Overview still reads `account.tracking_principal_krw`,
derives principal delta in the component, and presents holdings unrealized
return as the primary return card.

- [ ] **Step 3: Update the Overview facts**

Read only `account.investment_principal_krw`,
`account.total_value_krw`, `account.account_profit_krw`, and
`account.account_return` from the backend. Render the primary facts in this
order: `투자 원금`, `계좌 평가금`, `원금 대비 계좌 수익률`, `YTD 계좌 TWR`,
`예외 검토`. Show account profit in the return card's supporting text. Keep
holdings unrealized return in Performance/Instrument surfaces. Rename the
Performance points table header and field to `투자 원금` and
`investment_principal_krw`. Do not calculate account return in React.

- [ ] **Step 4: Run frontend tests and compile checks**

Run:

```bash
bun test frontend/tests
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

Expected: all tests pass and Vite produces a production build.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/lib/presentation.ts frontend/tests/presentation.test.ts frontend/src/styles.css
git commit -m "feat: align overview with investment principal"
```

### Task 5: Update documentation and verify the full contract

**Files:**
- Modify: `README.md`
- Test: `tests/test_account_performance.py`, `tests/test_inspection_service.py`, `tests/test_api_contract.py`

- [ ] **Step 1: Add the user-facing definition**

Document that investment principal is baseline plus classified external flows,
account value includes cash, principal-relative return is separate from YTD TWR,
and holdings cost basis remains a position-level metric.

- [ ] **Step 2: Run the complete offline verification set**

Run:

```bash
uv run pytest -q
bun test frontend/tests
bun run --cwd frontend typecheck
bun run --cwd frontend build
git diff --check
```

Expected: all Python and frontend tests pass, build succeeds, and there are no
whitespace errors.

- [ ] **Step 3: Verify the browser contract**

Start `task toss-dashboard-api` and `task toss-dashboard-dev`, open the Vite
port shown by the task, and confirm Overview reads `투자 원금`, `계좌 평가금`,
`원금 대비 계좌 수익률`, `YTD 계좌 TWR`, and `예외 검토` in that order.
Confirm Performance still shows holding unrealized return separately and no
UI/API text introduces order sizing or direct trade instructions.

- [ ] **Step 4: Commit**

```bash
git add README.md tests/test_account_performance.py tests/test_inspection_service.py tests/test_api_contract.py
git commit -m "docs: define investment principal semantics"
```
