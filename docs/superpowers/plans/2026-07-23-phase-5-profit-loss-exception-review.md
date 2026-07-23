# Phase 5 Profit/Loss and Exceptional-Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible profit/loss, drawdown, and structured exceptional-review evidence to the Toss-only IPS inspection flow while keeping `Action` limited to a broken thesis plus the same instrument's policy maximum breach.

**Architecture:** A new pure `services/risk_evidence.py` module calculates facts and evidence quality from the selected Toss snapshot, performance run, and adjusted Toss candles. `services/inspection_service.py` owns input selection and immutable fingerprints, while `services/inspection_engine.py` remains the only source of `OK`, `Watch`, `Review`, and `Action`. CLI, API, and React render persisted backend results without reclassifying them.

**Tech Stack:** Python 3.12, SQLite, Typer, FastAPI, pytest, React, TypeScript, Bun, Vite

---

## Preflight and execution boundaries

- [ ] Start only after the existing YTD/dashboard worktree changes are committed or intentionally separated. Do not stage unrelated files, especially `node_modules/`, the HTML mockup, or root package artifacts.
- [ ] Read `AGENTS.md`, then use `ips-judgment-filter` before changing engine wording or status behavior.
- [ ] Run every test with a temporary `PORTFOLIO_DB_PATH`; no test may call live Toss APIs.
- [ ] Treat `market sync`, profile-factor population, policy activation, and the first persisted `phase5-v1` run as operational changes. Stop for explicit user approval before those rollout steps.
- [ ] Do not add order quantity, transaction value, execution flags, or direct buy/sell language to any result.

## Task 1: Add schema version 8 and structured profile fields

**Files:**

- Modify: `storage/schema.py`
- Modify: `storage/instrument_profile_store.py`
- Modify: `tests/test_database_migrations.py`
- Modify: `tests/test_instrument_profile_store.py`

- [ ] **Step 1: Write migration preservation tests**

Add a v7 fixture containing one profile and one evaluation, migrate it, and assert:

```python
assert _schema_version(path) == LATEST_SCHEMA_VERSION == 8
assert profile == (
    "unknown", "unknown", "unknown", "unknown", ""
)
assert evaluation_result_json == '{"status":"Review"}'
assert market_evidence_fingerprint is None
assert json.loads(market_evidence_json) == {}
```

Also assert fresh databases expose all five new profile columns and both evaluation-evidence columns.

- [ ] **Step 2: Run the migration tests and confirm failure**

Run:

```bash
uv run pytest tests/test_database_migrations.py -q
```

Expected: failures show schema version 7 and missing columns.

- [ ] **Step 3: Add the forward-only migration**

Set `LATEST_SCHEMA_VERSION = 8`, add `MIGRATION_8_SQL`, and register it:

```python
MIGRATION_8_SQL = """
ALTER TABLE ips_instrument_profiles ADD COLUMN overlap_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK(overlap_status IN ('unknown', 'clear', 'review'));
ALTER TABLE ips_instrument_profiles ADD COLUMN management_burden_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK(management_burden_status IN ('unknown', 'clear', 'review'));
ALTER TABLE ips_instrument_profiles ADD COLUMN holdability_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK(holdability_status IN ('unknown', 'clear', 'review'));
ALTER TABLE ips_instrument_profiles ADD COLUMN etf_substitution_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK(etf_substitution_status IN ('unknown', 'not_applicable', 'clear', 'review'));
ALTER TABLE ips_instrument_profiles ADD COLUMN review_factors_note TEXT NOT NULL DEFAULT '';
ALTER TABLE ips_evaluation_runs ADD COLUMN market_evidence_fingerprint TEXT;
ALTER TABLE ips_evaluation_runs ADD COLUMN market_evidence_json TEXT NOT NULL DEFAULT '{}';
"""
```

The migration must not rewrite historical `result_json`, `engine_version`, or policy rows.

- [ ] **Step 4: Write profile validation and round-trip tests**

Cover every accepted enum, invalid enum rejection, defaults for an existing call, note normalization, deterministic listing, and proof that broker holdings remain unchanged.

```python
profile = upsert_profile(
    "AAPL", "US", "satellite", "watch",
    overlap_status="review",
    management_burden_status="clear",
    holdability_status="unknown",
    etf_substitution_status="not_applicable",
    review_factors_note="ETF alternative requires review",
)
assert profile["overlap_status"] == "review"
assert profile["etf_substitution_status"] == "not_applicable"
```

- [ ] **Step 5: Extend the profile store**

Add constants for the three-state factors and ETF's four-state factor. Extend `_validate_profile`, `_row_to_profile`, `upsert_profile`, `get_profile`, and `list_profiles`. Keep defaults so existing callers persist `unknown` without breakage:

```python
def upsert_profile(
    symbol: str,
    market_country: str,
    layer: str,
    thesis_status: str,
    thesis_note: str = "",
    account_alias: str = "toss-brokerage",
    *,
    overlap_status: str = "unknown",
    management_burden_status: str = "unknown",
    holdability_status: str = "unknown",
    etf_substitution_status: str = "unknown",
    review_factors_note: str = "",
) -> dict[str, Any]:
```

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest tests/test_database_migrations.py tests/test_instrument_profile_store.py -q
git diff --check
git add storage/schema.py storage/instrument_profile_store.py tests/test_database_migrations.py tests/test_instrument_profile_store.py
git commit -m "feat: add phase 5 profile evidence schema"
```

## Task 2: Validate and template the immutable risk-review policy

**Files:**

- Modify: `services/policy_validation.py`
- Modify: `storage/policy_store.py`
- Modify: `docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json`
- Modify: `docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md`
- Modify: `tests/test_policy_store.py`

- [ ] **Step 1: Add failing policy-validation tests**

Test the approved block exactly:

```python
"risk_review": {
    "lookback_sessions": 252,
    "minimum_history_points": 200,
    "max_data_age_days": 7,
    "max_gap_days": 7,
    "account_drawdown_review": -0.15,
    "instrument_drawdown_review": {
        "core": -0.25,
        "satellite": -0.20,
        "experiment": -0.15,
    },
}
```

Add parametrized failures for booleans, zero or negative integer limits, `minimum_history_points > lookback_sessions`, drawdown values at `-1`, `0`, positive, non-finite, missing or extra layer keys, and a missing `risk_review` object.

- [ ] **Step 2: Run the policy tests and confirm failure**

Run:

```bash
uv run pytest tests/test_policy_store.py -q
```

Expected: the normalized policy omits `risk_review`, and invalid risk policies are accepted.

- [ ] **Step 3: Implement strict risk-policy normalization**

Add dedicated helpers instead of widening the existing allocation `_number` helper:

```python
def _positive_integer(value: Any, path: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{path} must be a positive integer")
        return None
    return value


def _negative_rate(value: Any, path: str, errors: list[str]) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    if not math.isfinite(number) or not -1 < number < 0:
        errors.append(f"{path} must be greater than -1 and less than 0")
        return None
    return number
```

Return the normalized `risk_review` block from `validate_policy` and enforce exact keys for `instrument_drawdown_review`.

- [ ] **Step 4: Seed new databases without silently upgrading current policy**

Add `risk_review` to `DEFAULT_POLICY` so a fresh database is complete. Keep `ensure_default_policy`'s legacy auto-upgrade set limited to `performance` and `cadence`; do not add `risk_review` to `required_defaults`. This preserves the explicit approval boundary for existing active policies.

Ensure `policy_template()` carries `risk_review` from the active policy or fresh default. Update the Pattern B draft JSON and its explanatory Markdown, but do not call `activate_policy`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_policy_store.py tests/test_cli.py -q
git diff --check
git add services/policy_validation.py storage/policy_store.py docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md tests/test_policy_store.py
git commit -m "feat: define phase 5 risk review policy"
```

## Task 3: Build pure account and instrument risk evidence

**Files:**

- Create: `services/risk_evidence.py`
- Create: `tests/test_risk_evidence.py`

- [ ] **Step 1: Write failing safe-ratio and account-evidence tests**

Cover finite positive cost basis, zero/missing cost basis, supported and unsupported realized P/L, fewer than two connected points, non-evaluable boundaries, and current versus maximum drawdown.

Use this curve assertion:

```python
evidence = build_account_profit_loss(
    snapshot_id=7,
    performance_run={
        "id": 3,
        "state": "complete",
        "execution_count": 1,
        "points": [
            _point("2026-01-01T00:00:00Z", 0.00),
            _point("2026-02-01T00:00:00Z", 0.20),
            _point("2026-03-01T00:00:00Z", -0.25),
            _point("2026-04-01T00:00:00Z", 0.10),
        ],
    },
)
assert evidence["drawdown"]["maximum"] == pytest.approx(-0.25)
assert evidence["drawdown"]["current"] == pytest.approx(-0.175)
assert evidence["realized_pnl_supported"] is True
```

When `execution_count == 0`, assert both realized amounts are `None`, even if the stored point contains numeric zero.

- [ ] **Step 2: Write failing candle-quality and drawdown tests**

Generate deterministic daily candles and cover:

- latest 252 selected from a longer range;
- at least 200 required;
- snapshot timestamp cutoff excludes future candles;
- adjusted and `adjusted_supported` are both required;
- duplicate timestamps, invalid or non-positive closes, staleness over seven calendar days, and a gap over seven calendar days each return a named unavailable state;
- KR and US identities produce separate descriptors;
- ordered source fingerprints change the combined market fingerprint.

```python
assert result["drawdown"]["state"] == "complete"
assert result["drawdown"]["history_points"] == 252
assert result["drawdown"]["current"] == pytest.approx(-0.10)
assert result["drawdown"]["maximum"] == pytest.approx(-0.30)
```

- [ ] **Step 3: Run the new tests and confirm import failure**

Run:

```bash
uv run pytest tests/test_risk_evidence.py -q
```

Expected: `services.risk_evidence` does not exist.

- [ ] **Step 4: Implement facts and evidence quality without statuses**

Use a stable public entry point:

```python
def build_risk_evidence(
    projection: Mapping[str, Any] | None,
    performance_run: Mapping[str, Any] | None,
    candles_by_identity: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    risk_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return source-linked facts only; never assign IPS status."""
```

The returned shape must contain:

```python
{
    "schema_version": 1,
    "account_profit_loss": {...},
    "instruments": {"US/AAPL": {...}},
    "market_evidence": {"US/AAPL": {...}},
    "market_evidence_fingerprint": "...",
}
```

Implement account drawdown from the compounded evaluable TWR curve without
silently bridging a non-evaluable interval. Implement instrument drawdown from
native adjusted closes. Use snapshot `synced_at` as the upper time bound. Do
not import the inspection engine or status vocabulary in this module.

For a missing projection or performance run, return the same top-level shape
with explicit `source_unavailable` or `performance_unavailable` states, empty
instrument/market descriptors, and the canonical fingerprint of that empty
market descriptor. This keeps failed-source evaluations deterministic without
inventing zero values.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_risk_evidence.py -q
git diff --check
git add services/risk_evidence.py tests/test_risk_evidence.py
git commit -m "feat: calculate phase 5 risk evidence"
```

## Task 4: Select exact Toss candle ranges for each held instrument

**Files:**

- Modify: `storage/market_store.py`
- Modify: `tests/test_market_store.py`
- Modify: `services/inspection_service.py`
- Modify: `tests/test_inspection_service.py`

- [ ] **Step 1: Add failing bounded-selection tests**

Persist adjusted and unadjusted candles before and after a snapshot timestamp. Assert the selector returns only adjusted supported stock candles for the requested `(market_country, symbol)`, ordered oldest to newest, capped by limit, and never after `through_at`.

```python
candles = list_adjusted_stock_candles(
    market_country="US",
    symbol="AAPL",
    through_at="2026-07-22T16:01:04+00:00",
    limit=252,
)
assert all(row["source_kind"] == "stock" for row in candles)
assert all(row["adjusted"] == 1 and row["adjusted_supported"] == 1 for row in candles)
assert [row["candle_at"] for row in candles] == sorted(row["candle_at"] for row in candles)
```

- [ ] **Step 2: Implement one bounded store query**

Add:

```python
def list_adjusted_stock_candles(
    *,
    market_country: str,
    symbol: str,
    through_at: str,
    limit: int,
) -> list[dict[str, Any]]:
```

Use `datetime(candle_at) <= datetime(?)` so equivalent ISO timestamps with
different offsets are compared chronologically. Select by
`datetime(candle_at) DESC, id DESC` in a subquery for the limit, then return by
`datetime(candle_at), id`. Keep `list_candles` unchanged for Phase 4 market
context.

- [ ] **Step 3: Add failing service selection tests**

Patch the store selector and assert `inspection_service` asks once per held identity with:

- the projection's `synced_at`;
- `lookback_sessions` as the limit;
- no benchmark or unheld symbol;
- deterministic identity order.

- [ ] **Step 4: Add a pure input selector in the service**

Create a private helper:

```python
def _select_market_candles(
    projection: Mapping[str, Any], risk_policy: Mapping[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
```

This helper only reads local normalized Toss candles. It must not call the Toss transport or sync data during inspection.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_market_store.py tests/test_inspection_service.py -q
git diff --check
git add storage/market_store.py services/inspection_service.py tests/test_market_store.py tests/test_inspection_service.py
git commit -m "feat: select bounded Toss drawdown evidence"
```

## Task 5: Persist market evidence identity and support read-only preview

**Files:**

- Modify: `storage/evaluation_store.py`
- Modify: `services/inspection_service.py`
- Modify: `services/inspection_engine.py`
- Modify: `tests/test_evaluation_store.py`
- Modify: `tests/test_inspection_service.py`

- [ ] **Step 1: Add failing persistence and fingerprint tests**

Extend the evaluation fixture with:

```python
"market_evidence_fingerprint": "market-v1",
"market_evidence": {
    "US/AAA": {
        "first_candle_at": "2025-07-22T00:00:00Z",
        "latest_candle_at": "2026-07-22T00:00:00Z",
        "history_points": 252,
        "source_fingerprint": "range-a",
    }
},
```

Assert canonical JSON round-trips and a changed market fingerprint creates a different `evaluation_fingerprint`.

- [ ] **Step 2: Extend immutable evaluation storage**

Require and persist `market_evidence_fingerprint` and `market_evidence` for new writes. Decode `market_evidence_json` to `market_evidence` on reads. Historical rows migrate as `{}` and `None` without changing their result.

Extend the fingerprint contract:

```python
def evaluation_fingerprint(
    *,
    source_fingerprint: str,
    performance_fingerprint: str | None,
    policy_hash: str,
    profile_hash: str,
    market_evidence_fingerprint: str,
    engine_version: str = ENGINE_VERSION,
) -> str:
```

- [ ] **Step 3: Refactor service input assembly once**

Create `_evaluate_inputs(...)` so persisted runs and previews use identical selection and calculation. Set `ENGINE_VERSION = "phase5-v1"`. Refuse a persisted Phase 5 run when the active policy lacks `risk_review`:

```python
if "risk_review" not in active["policy"]:
    raise RuntimeError(
        "active IPS policy lacks risk_review; validate and explicitly activate a Phase 5 policy"
    )
```

Pass `risk_evidence` into `evaluate_inspection`, store its compact `market_evidence`, and include its fingerprint in the run fingerprint.

Build an `evidence_refs` mapping for the engine containing the selected
`snapshot_id`, `performance_run_id`, active `policy_version_id` or preview
`None`, `policy_hash`, `profile_hash`, and run-level market fingerprint. The
preview and persisted paths must use the same mapping shape.

- [ ] **Step 4: Add and test non-persisting preview**

Expose:

```python
def preview_inspection(
    policy: Mapping[str, Any],
    *,
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
```

The result must include `persisted: False`, `policy_version_id: None`, the policy hash, snapshot ID, evidence fingerprints, and the evaluation result. Test the row count of `ips_evaluation_runs` before and after preview; it must not change.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_evaluation_store.py tests/test_inspection_service.py -q
git diff --check
git add storage/evaluation_store.py services/inspection_service.py services/inspection_engine.py tests/test_evaluation_store.py tests/test_inspection_service.py
git commit -m "feat: persist phase 5 evidence identity"
```

## Task 6: Centralize the Phase 5 status truth table in the engine

**Files:**

- Modify: `services/inspection_engine.py`
- Modify: `tests/test_inspection_engine.py`

- [ ] **Step 1: Add the adversarial truth-table tests before rules**

Use parametrized cases that assert both final status and exact trigger sets:

| Case | Expected maximum |
|---|---|
| gain alone | `OK` |
| loss alone | `OK` |
| account drawdown unavailable | account `Watch` |
| account current drawdown crosses policy | account `Review` |
| instrument drawdown unavailable | `Watch` |
| core drawdown crosses policy | `Watch` |
| satellite/experiment drawdown crosses policy | `Review` |
| instrument maximum breach, thesis valid | `Review` |
| gain plus instrument maximum breach | `Review` |
| loss plus thesis or holdability concern | `Review` |
| broken thesis inside instrument maximum | `Review` |
| broken thesis plus layer breach only | `Review` |
| broken thesis plus same-instrument maximum breach | `Action` |
| core unknown factors | no factor escalation |
| satellite/experiment unknown factor | one combined `Review` unit |
| any explicit factor review | `Review` |

For every case, recursively inspect result keys and strings and reject `buy`, `sell`, `execute`, `order_quantity`, `transaction_value`, `stop_loss`, and `take_profit`.

- [ ] **Step 2: Run the engine tests and confirm failures**

Run:

```bash
uv run pytest tests/test_inspection_engine.py -q
```

Expected: evidence is not attached and Phase 5 triggers are absent.

- [ ] **Step 3: Add evidence to account and instrument units**

Change the engine entry point to require the pure bundle:

```python
def evaluate_inspection(
    projection: Mapping[str, Any] | None,
    performance_run: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    risk_evidence: Mapping[str, Any] | None = None,
    evidence_refs: Mapping[str, Any] | None = None,
    *,
    source_error: str | None = None,
) -> dict[str, Any]:
```

Place account evidence at `result["account_profit_loss"]`. Place each instrument's facts under its existing unit's `evidence` key. Evidence values alone must not call `_raise_status`.

Make `account_profit_loss` one inspection unit with `kind = "account_risk"`,
`identity = "account_profit_loss"`, status, triggers, meaning, verification
task, and the factual values. Unsupported account drawdown adds
`account_drawdown_evidence_unavailable` at `Watch`; a current drawdown at or
below the policy threshold adds `account_drawdown_review_threshold` at
`Review`. Account profit/loss amounts never escalate this unit.
When the overall inspection is already `not_evaluable` because the account
source or performance run failed, include the unavailable factual object but
retain the existing source/performance queue items instead of adding a second
account-risk queue item.

- [ ] **Step 4: Apply the approved rule order and queue priorities**

Keep one instrument unit and one queue item per identity. Preserve the existing
below-minimum allocation `Review`; distinguish the hard maximum predicate for
the exceptional rule. Accumulate factor triggers, then raise only to the
allowed maximum. Use these exact factor triggers:

```python
overlap_unknown
overlap_review
management_burden_unknown
management_burden_review
holdability_unknown
holdability_review
etf_substitution_unknown
etf_substitution_review
```

For core, only the four `*_review` triggers escalate. For satellite and
experiment, `unknown` and `review` both escalate once on the combined
instrument unit; `etf_substitution_status = "not_applicable"` is complete.
`loss_with_thesis_or_holdability_concern` requires negative unrealized P/L and
either `thesis_status in {"watch", "broken"}` or
`holdability_status == "review"`. The `Action` branch must use exactly:

```python
hard_maximum_breach = (
    current is not None
    and target_ready
    and float(current) > float(target["maximum"])
)
if thesis == "broken" and hard_maximum_breach:
    status = _raise_status(status, "Action")
    triggers.append("broken_thesis_and_hard_maximum_breach")
```

A layer maximum is never referenced in this predicate. Missing drawdown remains a `Watch` trigger even when an independently supported condition is stronger. Attach compact evidence references to queue items:

```python
"evidence_refs": {
    **dict(evidence_refs or {}),
    "market_source_fingerprint": instrument_evidence["drawdown"].get("source_fingerprint"),
}
```

- [ ] **Step 5: Verify invariant coverage and commit**

Run:

```bash
uv run pytest tests/test_inspection_engine.py tests/test_risk_evidence.py -q
git diff --check
git add services/inspection_engine.py tests/test_inspection_engine.py
git commit -m "feat: classify phase 5 review evidence"
```

## Task 7: Extend the agent-facing CLI without adding mutations

**Files:**

- Modify: `cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing profile-option tests**

Invoke `profiles set` with:

```text
--overlap-status review
--management-burden-status clear
--holdability-status unknown
--etf-substitution-status not_applicable
--review-factors-note "ETF 대체 검토"
```

Assert all values reach `upsert_profile`, stdout contains one JSON object, and no holding or order facts are added to the profile response.

- [ ] **Step 2: Add failing preview command tests**

Patch `validate_policy`, `policy_metadata`, and `preview_inspection`. Assert:

```python
result = runner.invoke(
    app,
    ["inspection", "preview", "--policy-file", str(policy_path), "--snapshot-id", "7"],
)
payload = json.loads(result.stdout)
assert result.exit_code == 0
assert payload["command"] == "inspection preview"
assert payload["persisted"] is False
assert payload["snapshot_id"] == 7
```

Also test invalid JSON, missing file, invalid policy, and service errors as a single machine-readable object.

- [ ] **Step 3: Implement CLI options and read-only preview**

Add typed options with the store defaults. Implement `inspection preview --policy-file` by loading JSON, unwrapping an optional top-level `policy`, validating against `list_observed_identities()`, and calling `preview_inspection`. Do not call `activate_policy` or `insert_evaluation_run`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run pytest tests/test_cli.py -q
git diff --check
git add cli.py tests/test_cli.py
git commit -m "feat: expose phase 5 inspection preview"
```

## Task 8: Render persisted Phase 5 evidence in existing dashboard tabs

**Files:**

- Modify: `api/app.py`
- Modify: `tests/test_api_contract.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/presentation.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/tests/presentation.test.ts`

- [ ] **Step 1: Lock the read-only API contract**

Add an API test proving `/api/inspection` returns the stored `result`, `market_evidence`, and statuses byte-for-structure without calling the engine. Add a `/api/profiles` assertion for the five new profile fields. No write route is added.

- [ ] **Step 2: Add failing presentation tests for zero versus unavailable**

Add pure helpers:

```typescript
export function evidenceValue(
  value: unknown,
  state: unknown,
  formatter: (value: unknown) => string,
): string {
  if (finiteNumber(value) !== null) return formatter(value);
  if (typeof state === "string" && state.length) return state;
  return "자료 없음";
}
```

Test that numeric zero renders as `0`, while `null` with `insufficient_history` renders the evidence state. Keep KRW values whole won.

- [ ] **Step 3: Extend frontend types only enough to name the evidence**

Add `market_evidence?: JsonObject` to `Evaluation` and `account_profit_loss?: JsonObject` to `Evaluation.result`. Continue accepting unknown nested fields as `JsonObject`; do not duplicate the engine's status model in TypeScript.

- [ ] **Step 4: Update existing panels**

Implement the approved placements:

- Overview: count queue items whose backend status is `Action` and show the
  highest backend status among them; render zero and `자료 없음` without
  inferring another status.
- Performance: account current/max drawdown, realized support state, and realized amounts beside existing YTD/trailing/cumulative views.
- Allocation: instrument unrealized return, current drawdown, evidence state, and compact profile-factor summary.
- Profiles & policy: the four factor statuses and shared rationale.
- Review Queue: backend triggers, meaning, verification task, and a collapsed `evidence_refs` detail.

Use `Status` only with backend-provided values. Do not derive severity, thresholds, or queue ordering in the browser.

- [ ] **Step 5: Verify API, presentation, typecheck, and production build**

Run:

```bash
uv run pytest tests/test_api_contract.py -q
bun test frontend/tests/presentation.test.ts
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

- [ ] **Step 6: Perform local browser verification**

Start the API with `task toss-dashboard-api` and the Vite UI with
`task toss-dashboard-dev` in separate terminals, then open
`http://127.0.0.1:5173`. Verify wide and narrow layouts, sidebar
collapsed/expanded, each existing tab, horizontal table scrolling, missing
evidence labels, zero values, and collapsed evidence references. Confirm
Overview is not dominated by source metadata.

- [ ] **Step 7: Commit**

```bash
git diff --check
git add api/app.py tests/test_api_contract.py frontend/src/lib/api.ts frontend/src/lib/presentation.ts frontend/src/App.tsx frontend/src/styles.css frontend/tests/presentation.test.ts
git commit -m "feat: present phase 5 review evidence"
```

## Task 9: Complete offline integration, documentation, and the approval gate

**Files:**

- Modify: `docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md`
- Modify: `README.md`
- Modify: `tests/test_inspection_service.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add one offline end-to-end fixture**

Build a temporary database containing:

- complete reconciled Toss snapshots;
- a complete performance run;
- KR and US adjusted stock candles;
- core, satellite, and experiment profiles with structured factors;
- an inactive validated policy payload containing `risk_review`.

Run `inspection preview` twice and assert deterministic equality, no inserted evaluation row, exact snapshot and evidence fingerprints, one combined queue item per unit, and the approved `Action` invariant.

- [ ] **Step 2: Add a forbidden-semantics assertion**

Recursively collect result keys and string values. Reject the following field
semantics and direct-action phrases while allowing the approved phrase
`future regular-purchase policy` and its Korean equivalent:

```python
FORBIDDEN_KEYS = {
    "buy", "sell", "execute", "order_quantity", "transaction_value",
    "stop_loss", "take_profit",
}
FORBIDDEN_PHRASES = {"buy now", "sell now", "즉시 매수", "즉시 매도", "주문 수량"}
```

This assertion covers the preview and persisted-result fixtures.

- [ ] **Step 3: Update user documentation and roadmap**

Mark Phase 5 implementation complete only after all offline checks pass. Document:

```text
uv run ips-pilot inspection preview --policy-file docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json
uv run ips-pilot profiles set --symbol NBIS --market-country US --layer satellite --thesis-status watch --overlap-status review --management-burden-status clear --holdability-status clear --etf-substitution-status review --note "위성 투자 논지 재검토" --review-factors-note "중복과 ETF 대체 가능성 확인"
```

State explicitly that preview does not activate policy or persist an evaluation, `Action` is an exceptional inspection signal, and market sync/profile updates/policy activation require deliberate operator action.

- [ ] **Step 4: Run the full offline verification set**

Run:

```bash
uv run pytest -q
bun test frontend/tests
bun run --cwd frontend typecheck
bun run --cwd frontend build
git diff --check
```

- [ ] **Step 5: Adversarially inspect the preview output**

Review the full JSON and dashboard against these questions:

1. Can gain, loss, drawdown, cash, or a layer breach create `Action` without the same instrument's hard maximum breach?
2. Does missing evidence ever render as numeric zero?
3. Does any UI or CLI string imply permission to trade?
4. Can stale, future, unadjusted, duplicate, or gapped candles produce a drawdown number?
5. Can preview alter the active policy, profile rows, evaluation rows, or broker facts?
6. Does a market evidence change alter the immutable evaluation fingerprint?

Record failures as test cases before correcting implementation.

- [ ] **Step 6: Commit the completed offline implementation**

```bash
git add docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md README.md tests/test_inspection_service.py tests/test_cli.py
git commit -m "docs: complete phase 5 offline rollout"
```

- [ ] **Step 7: Stop before operational mutation**

Report the offline preview, evidence gaps, and proposed commands. Obtain explicit user approval before any of the following:

1. syncing live Toss stock candles;
2. setting the 16 holdings' structured factor judgments;
3. activating the new immutable policy version;
4. persisting the first `phase5-v1` inspection run.

After approval, perform those actions in that order and verify the selected `snapshot_id`, policy version, market evidence fingerprint, review queue, and browser rendering.
