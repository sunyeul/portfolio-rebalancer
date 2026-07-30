# Risk-adjusted Instrument Target Ranges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add review-only, evidence-backed instrument target ranges without treating legacy policy ratios, holdings returns, or allocation gaps as current target recommendations.

**Architecture:** Keep the existing benchmark regime selector and add one pure instrument-review stage in `services/dynamic_allocation.py`. The stage validates adjusted Toss candles, maps four explainable evidence signals to policy-anchored ranges, fits a deterministic reference inside feasible layer ranges, and returns no policy when evidence is incomplete or ranges are infeasible. CLI and API explicitly supply all policy-instrument candle series; activation remains unchanged and separate.

**Tech Stack:** Python 3.12, standard library, Typer CLI, FastAPI, SQLite-backed Toss market store, pytest.

---

### Task 1: Pure instrument evidence, classification, and range fitting

**Files:**
- Modify: `services/dynamic_allocation.py:280-419`
- Modify: `tests/test_dynamic_allocation.py:1-177`

- [ ] **Step 1: Add failing unit-test helpers and classification tests**

Extend the test candle helper so stock evidence is explicitly adjusted:

```python
def _candles(values, *, end=NOW, adjusted=True):
    start = end - timedelta(days=len(values) - 1)
    return [
        {
            "candle_at": (start + timedelta(days=index)).isoformat(),
            "close_price": value,
            "adjusted": adjusted,
            "adjusted_supported": adjusted,
        }
        for index, value in enumerate(values)
    ]


def _instrument_series(policy, values):
    return {
        f"{item['market_country']}/{item['symbol']}": _candles(values)
        for item in policy["instruments"]
    }
```

Import `build_instrument_target_reviews` and add focused tests:

```python
def test_instrument_signals_map_to_policy_anchored_ranges():
    policy = build_neutral_policy(_active_policy())
    regime_policy = scale_policy_to_regime(policy, "neutral")
    series = _instrument_series(policy, [100.0] * 220)
    series["KR/069500"] = _candles([220 - index * 0.7 for index in range(220)])

    result = build_instrument_target_reviews(
        series, active_policy=policy, regime_policy=regime_policy, now=NOW
    )

    kode = next(item for item in result["reviews"] if item["identity"] == "KR/069500")
    assert kode["signal"] == "severe"
    assert kode["analysis_range"] == {
        "minimum": 0.0,
        "maximum": pytest.approx(0.044444444444),
    }
    assert kode["role_review_required"] is True
```

Add separate cases for `supportive`, `neutral`, `adverse`, severe-volatility
precedence, unadjusted candles, and a role-specific drawdown threshold.

- [ ] **Step 2: Run the new classification tests and verify they fail**

Run:

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -k "instrument_signal or instrument_target" -v
```

Expected: collection or assertion failure because
`build_instrument_target_reviews` does not exist.

- [ ] **Step 3: Implement the pure evidence and range helpers**

Add these focused helpers to `services/dynamic_allocation.py`:

```python
def _instrument_key(item: Mapping[str, Any]) -> str:
    return "/".join(_identity(item))


def _instrument_signal(evidence: Mapping[str, Any]) -> str:
    if evidence["severe_risk"]:
        return "severe"
    if evidence["trend_direction"] < 0:
        return "adverse"
    if evidence["trend_direction"] > 0:
        return "supportive"
    return "neutral"


def _analysis_range(
    signal: str,
    policy_item: Mapping[str, Any],
    regime_item: Mapping[str, Any],
) -> dict[str, float]:
    minimum = float(policy_item["minimum"])
    target = float(regime_item["target"])
    maximum = float(policy_item["maximum"])
    bounds = {
        "supportive": (target, maximum),
        "neutral": (minimum, maximum),
        "adverse": (minimum, target),
        "severe": (0.0, minimum),
    }[signal]
    return {"minimum": bounds[0], "maximum": bounds[1]}
```

Add `_fit_layer_references(reviews, layer_target)` using only standard-library
math: clamp each regime anchor to its range, calculate the remaining difference,
and distribute it proportionally to available headroom in stable identity order.
Return `None` when lower-bound sum exceeds the layer target or upper-bound sum is
below it. Round only at the final 12-decimal assignment and place residual on the
last eligible identity.

Add:

```python
def build_instrument_target_reviews(
    series_by_identity: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    active_policy: Mapping[str, Any],
    regime_policy: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Build evidence-backed review ranges without activating a policy."""
```

For every ordered policy instrument:

1. require adjusted and adjustment-supported candles;
2. call `_series_evidence` with `risk_review` quality settings, the instrument's
   role-specific drawdown threshold, and the existing allocation volatility
   threshold;
3. emit `policy_anchor`, `regime_anchor`, evidence details, signal, range,
   `role_review_required`, reasons, and verification task; and
4. set range/reference to `None` when evidence is invalid.

When all evidence is complete and all layers are feasible, deep-copy the regime
policy and replace each instrument's minimum, target, and maximum with the
analysis lower bound, fitted reference, and analysis upper bound. Validate it
with `validate_policy`. Return:

```python
{
    "state": "complete",
    "reason": "instrument_target_ranges_complete",
    "reviews": reviews,
    "proposed_policy": validated_policy,
}
```

Incomplete evidence returns `state="incomplete"`; infeasible ranges return
`state="infeasible"`. Both return `proposed_policy=None`.

- [ ] **Step 4: Run the focused unit tests**

Run:

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -k "instrument_signal or instrument_target" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the pure target-range engine**

```bash
rtk git add services/dynamic_allocation.py tests/test_dynamic_allocation.py
rtk git commit -m "feat: add risk-adjusted instrument target ranges"
```

### Task 2: Compose instrument ranges into market-context candidates

**Files:**
- Modify: `services/dynamic_allocation.py:405-531`
- Modify: `tests/test_dynamic_allocation.py:178-350`

- [ ] **Step 1: Add failing evaluator contract tests**

Add tests that pass `instrument_series_by_identity` explicitly:

```python
def test_complete_instrument_ranges_create_review_candidate():
    policy = build_neutral_policy(_active_policy())
    flat = _series_map([100.0] * 220)
    instruments = _instrument_series(policy, [100.0] * 220)
    instruments["KR/069500"] = _candles(
        [220 - index * 0.7 for index in range(220)]
    )

    result = evaluate_dynamic_allocation(
        flat,
        instrument_series_by_identity=instruments,
        active_policy=policy,
        last_change_at="2026-05-01T00:00:00+00:00",
        now=NOW,
    )

    assert result["status"] == "Review"
    assert result["candidate_state"] == "candidate"
    assert result["proposed_policy"] is not None
    assert result["instrument_target_reviews"]
```

Add tests for incomplete instrument evidence (`Watch`, `observe`, null policy),
infeasible ranges (`Review`, `observe`, null policy), cooldown on range-only
changes, and the absence of order/execution vocabulary.

- [ ] **Step 2: Run the evaluator tests and verify they fail**

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -k "complete_instrument or incomplete_instrument or infeasible_ranges or range_only" -v
```

Expected: failure because the evaluator has no instrument-series parameter or
review fields.

- [ ] **Step 3: Integrate the pure review stage**

Change the evaluator signature to:

```python
def evaluate_dynamic_allocation(
    series_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    active_policy: Mapping[str, Any],
    instrument_series_by_identity: Mapping[
        str, Sequence[Mapping[str, Any]]
    ] | None = None,
    last_change_at: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
```

After regime selection, construct the regime policy and, when the optional map
is supplied, call `build_instrument_target_reviews`. Always include
`instrument_target_reviews` in the result. Preserve the legacy benchmark-only
path when the optional map is `None` for research replay compatibility.

For incomplete evidence return:

```python
{
    "status": "Watch",
    "candidate_state": "observe",
    "reason": "instrument_evidence_incomplete",
    "proposed_targets": target_summary(regime_policy),
    "proposed_policy": None,
}
```

For infeasible ranges return `status="Review"`, `candidate_state="observe"`,
and `reason="instrument_target_ranges_infeasible"` with no proposed policy.

Replace summary-only change detection with a comparison of cash target, layer
targets, and each instrument's minimum/target/maximum. Apply the existing
cooldown after this complete comparison.

- [ ] **Step 4: Run dynamic-allocation regression tests**

```bash
rtk uv run pytest tests/test_dynamic_allocation.py -v
```

Expected: all tests pass, including existing benchmark-only callers.

- [ ] **Step 5: Commit evaluator composition**

```bash
rtk git add services/dynamic_allocation.py tests/test_dynamic_allocation.py
rtk git commit -m "feat: compose instrument ranges into policy candidates"
```

### Task 3: Supply policy-instrument candles from CLI and API

**Files:**
- Modify: `cli.py:1-48,223-234,651-716`
- Modify: `api/app.py:1-34,107-118,413-438`
- Modify: `tests/test_cli.py:328-400`
- Modify: `tests/test_api_contract.py:78-121`

- [ ] **Step 1: Add failing CLI and API series tests**

Expand the fixture policies to include:

```python
"risk_review": {"lookback_sessions": 252},
"instruments": [
    {"market_country": "KR", "symbol": "069500", "layer": "core"},
    {"market_country": "US", "symbol": "VOO", "layer": "core"},
],
```

Monkeypatch `list_adjusted_stock_candles`, capture the evaluator arguments, and
assert:

```python
assert set(captured["instrument_series_by_identity"]) == {
    "KR/069500",
    "US/VOO",
}
```

For market sync, retain the existing deduplication assertion so SPY or QQQ is
collected once even when it is both a benchmark and a policy instrument.

- [ ] **Step 2: Run the CLI/API tests and verify they fail**

```bash
rtk uv run pytest tests/test_cli.py::test_market_context_persists_composite_candidate_without_activation tests/test_api_contract.py::test_market_context_passes_composite_series_and_policy_timestamp -v
```

Expected: failure because only benchmark series reach the evaluator.

- [ ] **Step 3: Add the read-only instrument-series loaders**

Import `datetime`, `timezone`, and `list_adjusted_stock_candles` in both entry
points. Add the same small local helper next to `_stored_allocation_series`:

```python
def _stored_instrument_series(
    policy: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    through_at = datetime.now(timezone.utc).isoformat()
    limit = int(policy["risk_review"]["lookback_sessions"])
    return {
        f"{item['market_country'].upper()}/{item['symbol'].upper()}": (
            list_adjusted_stock_candles(
                market_country=item["market_country"],
                symbol=item["symbol"],
                through_at=through_at,
                limit=limit,
            )
        )
        for item in policy.get("instruments", [])
    }
```

Pass the result as `instrument_series_by_identity` from CLI `market sync`, CLI
`market context`, and API `/api/market-context`. Do not change candidate storage
or activation behavior.

- [ ] **Step 4: Run CLI/API contract tests**

```bash
rtk uv run pytest tests/test_cli.py tests/test_api_contract.py -v
```

Expected: all tests pass and no write route or automatic activation appears.

- [ ] **Step 5: Commit runtime wiring**

```bash
rtk git add cli.py api/app.py tests/test_cli.py tests/test_api_contract.py
rtk git commit -m "feat: supply instrument evidence to target reviews"
```

### Task 4: Verify contracts and document the delivered behavior

**Files:**
- Modify if required by verified output only: `README.md`
- Test: `tests/test_dynamic_allocation.py`
- Test: `tests/test_policy_store.py`
- Test: `tests/test_market_store.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_api_contract.py`
- Test when shared evidence changes: `tests/test_risk_evidence.py`
- Test when research compatibility changes: `tests/research/test_qlib_replay.py`

- [ ] **Step 1: Run formatting and the focused verification set**

```bash
rtk uv run ruff format --check services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py
rtk uv run pytest tests/test_dynamic_allocation.py tests/test_policy_store.py tests/test_market_store.py tests/test_cli.py tests/test_api_contract.py -q
```

Expected: formatting check and focused tests pass.

- [ ] **Step 2: Run related inspection and research regressions**

```bash
rtk uv run pytest tests/test_risk_evidence.py tests/test_inspection_service.py tests/research/test_qlib_replay.py -q
```

Expected: all tests pass; benchmark-only research replay remains supported.

- [ ] **Step 3: Run a guardrail scan on serialized output**

```python
encoded = json.dumps(result, ensure_ascii=False).lower()
for forbidden in ("buy", "sell", "execute", "quantity", "price"):
    assert forbidden not in encoded
```

Keep this assertion in `tests/test_dynamic_allocation.py`; do not add execution
fields to make the scan pass conditionally.

- [ ] **Step 4: Inspect the final diff and commit any verification-only fixes**

```bash
rtk git diff --check
rtk git status --short
```

If verification required a tracked fix, stage only files owned by this plan and
commit:

```bash
rtk git add README.md services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py
rtk git commit -m "test: verify instrument target range contracts"
```

Do not stage the user's unrelated `.codex`, `AGENTS.md`, ACE-hook, or ACE-test
changes.
