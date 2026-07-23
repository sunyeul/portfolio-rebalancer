# Dynamic Market Allocation Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and roll out a Toss-backed three-regime IPS review that proposes coordinated cash and layer targets without creating orders or activating market-derived candidates automatically.

**Architecture:** Add a focused `dynamic_allocation` service that owns neutral-policy construction, regime scaling, per-series evidence, and composite classification. Extend policy validation for a replayable `allocation_review` block, then replace the CLI's single-KOSPI context path with four required Toss benchmarks while reusing immutable candidate storage and explicit policy activation.

**Tech Stack:** Python 3.12, Typer, SQLite, pytest, existing Toss market normalization and IPS policy services.

---

## File map

- Create `services/dynamic_allocation.py`: allocation configuration, deterministic policy scaling, benchmark evidence, and three-regime evaluation.
- Create `tests/test_dynamic_allocation.py`: pure policy-builder and evaluator coverage.
- Modify `services/policy_validation.py`: validate and preserve the optional `allocation_review` policy block.
- Modify `tests/test_policy_store.py`: policy-validation tests for allocation configuration.
- Modify `cli.py`: collect four primary benchmarks and call the composite evaluator.
- Modify `tests/test_cli.py`: composite context and candidate-persistence contract tests.
- Modify `tests/test_market_store.py`: remove its dependency on the retired cash-only evaluator.
- Delete `services/market_context.py`: remove the superseded single-KOSPI cash evaluator.
- Delete `tests/test_market_context.py`: replace with composite evaluator tests.
- Create `docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json`: reviewed neutral policy used for explicit rollout.
- Modify `README.md`: document the composite review and human-approval boundary.

### Task 1: Validate replayable allocation-review policy configuration

**Files:**
- Modify: `services/policy_validation.py`
- Modify: `tests/test_policy_store.py`

- [ ] **Step 1: Write failing validation tests**

Add a `_allocation_review()` fixture with all four benchmark definitions, thresholds, cooldown, and three presets. Add one passing test asserting the normalized policy preserves that object and parameterized failures for duplicate benchmark identities, weights not summing to one, missing regimes, preset layer totals not summing to one, and preset targets outside the policy hard ranges.

```python
def test_policy_validation_preserves_allocation_review():
    policy = _valid_policy()
    policy["allocation_review"] = _allocation_review()

    normalized = validate_policy(policy, [("US", "SPY")])

    assert normalized["allocation_review"] == _allocation_review()


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value["benchmarks"].append(dict(value["benchmarks"][0])), "duplicate"),
        (lambda value: value["benchmarks"][0].update(weight=0.10), "sum to 1"),
        (lambda value: value["regimes"].pop("risk_off"), "exactly risk_on, neutral, risk_off"),
        (lambda value: value["regimes"]["neutral"]["layers"].update(core=0.61), "sum to 1"),
        (lambda value: value["regimes"]["risk_on"].update(cash_target=0.02), "cash_reserve range"),
    ],
)
def test_policy_validation_rejects_invalid_allocation_review(mutator, message):
    policy = _valid_policy()
    policy["allocation_review"] = _allocation_review()
    mutator(policy["allocation_review"])

    with pytest.raises(PolicyValidationError, match=message):
        validate_policy(policy, [("US", "SPY")])
```

- [ ] **Step 2: Run the new validation tests and confirm failure**

Run: `rtk .venv/bin/python -m pytest tests/test_policy_store.py -q`

Expected: the preservation assertion fails because `validate_policy` currently drops `allocation_review`.

- [ ] **Step 3: Implement allocation-review validation**

Add a private validator that enforces finite values, unique keys and identities, benchmark weights summing to one, the exact regime vocabulary, and preset targets within already-validated cash/layer hard ranges. Keep the block optional so historical policies remain replayable.

```python
def _allocation_review(
    value: Any,
    cash: dict[str, float] | None,
    layers: dict[str, dict[str, float]],
    errors: list[str],
) -> dict[str, Any] | None:
    start_error_count = len(errors)
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append("allocation_review must be an object")
        return None
    strategy = str(value.get("strategy", "")).strip()
    if strategy != "us_kr_three_regime_v1":
        errors.append("allocation_review.strategy must be us_kr_three_regime_v1")
    cooldown_days = _positive_integer(
        value.get("cooldown_days"), "allocation_review.cooldown_days", errors
    )
    scalar_paths = {
        "minimum_history_points": _positive_integer,
        "max_data_age_days": _positive_integer,
        "max_gap_days": _positive_integer,
    }
    scalars = {
        key: parser(value.get(key), f"allocation_review.{key}", errors)
        for key, parser in scalar_paths.items()
    }
    thresholds = {
        "drawdown_review": _signed_rate(value.get("drawdown_review"), "allocation_review.drawdown_review", errors),
        "volatility_review": _number(value.get("volatility_review"), "allocation_review.volatility_review", errors),
        "risk_on_trend": _signed_unit(value.get("risk_on_trend"), "allocation_review.risk_on_trend", errors),
        "risk_off_trend": _signed_unit(value.get("risk_off_trend"), "allocation_review.risk_off_trend", errors),
        "risk_on_max_risk_weight": _number(value.get("risk_on_max_risk_weight"), "allocation_review.risk_on_max_risk_weight", errors),
        "risk_off_risk_weight": _number(value.get("risk_off_risk_weight"), "allocation_review.risk_off_risk_weight", errors),
    }
    benchmarks = _benchmark_rows(value.get("benchmarks"), errors)
    regimes = _regime_rows(value.get("regimes"), cash, layers, errors)
    if len(errors) != start_error_count:
        return None
    return {
        "strategy": strategy,
        "cooldown_days": cooldown_days,
        **scalars,
        **thresholds,
        "benchmarks": benchmarks,
        "regimes": regimes,
    }
```

Implement `_signed_rate`, `_signed_unit`, `_benchmark_rows`, and `_regime_rows` directly above this function. `_benchmark_rows` accepts only `stock` and `market_indicator`, normalizes key/country/symbol to uppercase, rejects duplicate keys and identities, and requires weights to sum to one with `SUM_TOLERANCE`. `_regime_rows` requires exactly `risk_on`, `neutral`, and `risk_off`, normalizes `cash_target` and the three layer values with `_number`, checks their sum with `SUM_TOLERANCE`, and checks every target against `cash` and `layers`.

Insert the normalized object into the final result only when the input contains it:

```python
normalized = {
    "cash_reserve": cash,
    "performance": normalized_performance,
    "risk_review": risk_review,
    "cadence": normalized_cadence,
    "layers": layers,
    "instruments": normalized_instruments,
}
if allocation_review is not None:
    normalized["allocation_review"] = allocation_review
return normalized
```

- [ ] **Step 4: Run focused tests**

Run: `rtk .venv/bin/python -m pytest tests/test_policy_store.py -q`

Expected: all policy-store tests pass.

- [ ] **Step 5: Commit the isolated validation change**

```bash
rtk git add services/policy_validation.py tests/test_policy_store.py
rtk git commit -m "feat: validate dynamic allocation policy"
```

### Task 2: Build the approved neutral policy and regime-scaled policies

**Files:**
- Create: `services/dynamic_allocation.py`
- Create: `tests/test_dynamic_allocation.py`

- [ ] **Step 1: Write failing neutral-policy tests**

Create an active-policy fixture with the current sixteen Toss identities. Assert the builder produces the approved hard ranges, neutral targets, GLD Core target of 10%, exact layer instrument sums, and the retained ratios for Core excluding GLD, Satellite excluding GLD, and Experiment.

```python
def test_build_neutral_policy_reclassifies_gld_and_preserves_layer_ratios():
    result = build_neutral_policy(_active_policy())

    assert result["cash_reserve"] == {
        "minimum": 0.03, "target": 0.05, "maximum": 0.10
    }
    assert result["layers"] == {
        "core": {"minimum": 0.50, "target": 0.60, "maximum": 0.70},
        "satellite": {"minimum": 0.28, "target": 0.38, "maximum": 0.48},
        "experiment": {"minimum": 0.00, "target": 0.02, "maximum": 0.04},
    }
    gld = _instrument(result, "US", "GLD")
    assert gld["layer"] == "core"
    assert gld["target"] == pytest.approx(0.10)
    for layer in ("core", "satellite", "experiment"):
        assert sum(i["target"] for i in result["instruments"] if i["layer"] == layer) \
            == pytest.approx(result["layers"][layer]["target"])
```

Add scaling tests for all three regimes and failures for missing GLD or an empty source layer.

- [ ] **Step 2: Run the new tests and confirm import failure**

Run: `rtk .venv/bin/python -m pytest tests/test_dynamic_allocation.py -q`

Expected: collection fails because `services.dynamic_allocation` does not exist.

- [ ] **Step 3: Implement configuration and deterministic scaling**

Define `DEFAULT_ALLOCATION_REVIEW`, `build_neutral_policy(active_policy)`, `scale_policy_to_regime(policy, regime)`, and `target_summary(policy)`. Use `copy.deepcopy`; stable-sort instruments by `(market_country, symbol)`; calculate all but the last target by proportional share and assign the exact remainder to the last instrument.

```python
REGIME_TARGETS = {
    "risk_on": {"cash_target": 0.03, "layers": {"core": 0.52, "satellite": 0.44, "experiment": 0.04}},
    "neutral": {"cash_target": 0.05, "layers": {"core": 0.60, "satellite": 0.38, "experiment": 0.02}},
    "risk_off": {"cash_target": 0.10, "layers": {"core": 0.70, "satellite": 0.29, "experiment": 0.01}},
}


def _allocate(items, total, source_weights):
    ordered = sorted(items, key=lambda item: (item["market_country"], item["symbol"]))
    denominator = sum(source_weights[(i["market_country"], i["symbol"])] for i in ordered)
    if denominator <= 0:
        raise DynamicAllocationError("source layer target must be positive")
    allocated = {}
    subtotal = 0.0
    for item in ordered[:-1]:
        identity = (item["market_country"], item["symbol"])
        value = total * source_weights[identity] / denominator
        allocated[identity] = value
        subtotal += value
    last = ordered[-1]
    allocated[(last["market_country"], last["symbol"])] = total - subtotal
    return allocated
```

After neutral targets are assigned, compute each instrument's minimum and maximum from its neutral within-layer share and the approved layer bounds. Attach a deep copy of `DEFAULT_ALLOCATION_REVIEW`.

- [ ] **Step 4: Run policy-builder tests**

Run: `rtk .venv/bin/python -m pytest tests/test_dynamic_allocation.py -q`

Expected: all policy-builder tests pass.

- [ ] **Step 5: Commit the pure builder**

```bash
rtk git add services/dynamic_allocation.py tests/test_dynamic_allocation.py
rtk git commit -m "feat: build dynamic allocation policies"
```

### Task 3: Evaluate four-market evidence and produce review-only candidates

**Files:**
- Modify: `services/dynamic_allocation.py`
- Modify: `tests/test_dynamic_allocation.py`
- Delete: `services/market_context.py`
- Delete: `tests/test_market_context.py`

- [ ] **Step 1: Write failing regime and guardrail tests**

Generate timestamped 220-point series ending at an injected `now`. Cover broad positive trend (`risk_on`), broad negative trend (`risk_off`), mixed trend (`neutral`), severe-risk override, unchanged targets (`OK`), changed targets (`Review` candidate), cooldown (`Watch` observe), and each data-quality failure (`Watch` observe with no policy).

```python
def test_broad_positive_trend_proposes_risk_on_without_activation():
    policy = build_neutral_policy(_active_policy())
    result = evaluate_dynamic_allocation(
        _series_map(_rising_values()),
        active_policy=policy,
        last_change_at="2026-05-01T00:00:00+00:00",
        now=NOW,
    )

    assert result["status"] == "Review"
    assert result["candidate_state"] == "candidate"
    assert result["regime"] == "risk_on"
    assert result["proposed_targets"] == REGIME_TARGETS["risk_on"]
    assert "buy" not in json.dumps(result).lower()
    assert "sell" not in json.dumps(result).lower()


def test_stale_required_benchmark_fails_closed():
    series = _series_map(_flat_values())
    series["KR/KOSDAQ"] = series["KR/KOSDAQ"][:-10]

    result = evaluate_dynamic_allocation(series, active_policy=policy, now=NOW)

    assert result["status"] == "Watch"
    assert result["candidate_state"] == "observe"
    assert result["proposed_policy"] is None
    assert result["failed_benchmarks"] == ["KR/KOSDAQ"]
```

- [ ] **Step 2: Run the evaluator tests and confirm failure**

Run: `rtk .venv/bin/python -m pytest tests/test_dynamic_allocation.py -q`

Expected: failures show the evaluator and evidence functions are missing.

- [ ] **Step 3: Implement per-series evidence and composite classification**

Port timestamp/close normalization, drawdown, and 20-return annualized volatility from the retired evaluator into private focused functions. Evaluate all configured benchmarks before classifying so output contains evidence for every required source.

```python
def _classify(weighted_trend: float, severe_risk_weight: float, config: Mapping[str, Any]) -> str:
    if (
        weighted_trend <= config["risk_off_trend"]
        or severe_risk_weight >= config["risk_off_risk_weight"]
    ):
        return "risk_off"
    if (
        weighted_trend >= config["risk_on_trend"]
        and severe_risk_weight < config["risk_on_max_risk_weight"]
    ):
        return "risk_on"
    return "neutral"
```

Return `proposed_policy=None` when any primary input fails. Otherwise build the complete regime policy, compare only cash/layer targets, enforce the 30-day cooldown, and return `OK`, `Watch`, or `Review` with `observe` or `candidate`.

- [ ] **Step 4: Remove the retired evaluator and run focused tests**

Run: `rtk .venv/bin/python -m pytest tests/test_dynamic_allocation.py tests/test_policy_store.py -q`

Expected: all tests pass and no source imports `services.market_context`.

- [ ] **Step 5: Commit the composite evaluator**

```bash
rtk git add services/dynamic_allocation.py tests/test_dynamic_allocation.py services/market_context.py tests/test_market_context.py
rtk git commit -m "feat: evaluate US and Korea allocation regimes"
```

### Task 4: Wire composite evidence into the CLI and candidate store

**Files:**
- Modify: `cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_market_store.py`

- [ ] **Step 1: Rewrite the CLI tests around four series**

Mock `list_candles` by `(source_kind, market_country, symbol)`, assert `evaluate_dynamic_allocation` receives `US/SPY`, `US/QQQ`, `KR/KOSPI`, and `KR/KOSDAQ`, and assert a candidate is persisted with the active policy id while activation remains unchanged.

```python
def test_market_context_persists_composite_candidate_without_activation(monkeypatch):
    seen = {}
    monkeypatch.setattr("cli.get_active_policy", lambda: _dynamic_active_policy())
    monkeypatch.setattr("cli.list_candles", lambda **kwargs: [kwargs])

    def evaluate(series, **kwargs):
        seen.update(series)
        return {"status": "Review", "candidate_state": "candidate", "regime": "risk_on"}

    monkeypatch.setattr("cli.evaluate_dynamic_allocation", evaluate)
    result = runner.invoke(app, ["market", "context"])

    assert result.exit_code == 0
    assert set(seen) == {"US/SPY", "US/QQQ", "KR/KOSPI", "KR/KOSDAQ"}
    assert _payload(result)["activation"] == "human approval required; active policy unchanged"
```

Add a sync test proving requested holdings are deduplicated with required SPY/QQQ stocks and both Korean indicators are fetched.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `rtk .venv/bin/python -m pytest tests/test_cli.py tests/test_market_store.py -q`

Expected: failures reference the old single-series evaluator and benchmark option.

- [ ] **Step 3: Replace CLI integration**

Import `evaluate_dynamic_allocation`. Add one helper that loads configured series from storage:

```python
def _stored_allocation_series(policy: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    config = policy.get("allocation_review") or {}
    return {
        item["key"]: list_candles(
            source_kind=item["source_kind"],
            market_country=item["market_country"],
            symbol=item["symbol"],
        )
        for item in config.get("benchmarks", [])
    }
```

In `market sync`, stable-deduplicate requested/held stocks with required configured stock benchmarks, collect both configured indicators, persist all normalized observations, evaluate once, and persist only a `candidate` result. In `market context`, remove `--benchmark`, load all configured series, evaluate once, and preserve the one-object stdout contract.

Update the market-store idempotence test to use a literal candidate JSON object so storage tests no longer depend on signal logic.

- [ ] **Step 4: Run CLI and storage tests**

Run: `rtk .venv/bin/python -m pytest tests/test_cli.py tests/test_market_store.py tests/test_toss_market.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit only task-owned hunks**

`cli.py` and `tests/test_cli.py` already contain unrelated user changes. Inspect the diff, stage only the dynamic-allocation hunks, and leave every pre-existing hunk unstaged.

```bash
rtk git diff -- cli.py tests/test_cli.py tests/test_market_store.py
rtk git add -p cli.py tests/test_cli.py
rtk git add tests/test_market_store.py
rtk git diff --cached --check
rtk git commit -m "feat: expose composite market allocation context"
```

### Task 5: Generate, preview, and explicitly activate the neutral baseline

**Files:**
- Create: `docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json`
- Modify: `README.md`

- [ ] **Step 1: Generate the recommended policy from active version 5**

Run the builder against `get_active_policy()["policy"]`, serialize with sorted keys and indentation, and place that exact output in `docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json`. Do not source identities, targets, prices, or holdings from anywhere except the active IPS policy and normalized Toss snapshot.

Expected key checks:

```json
{
  "cash_reserve": {"minimum": 0.03, "target": 0.05, "maximum": 0.10},
  "layers": {
    "core": {"minimum": 0.50, "target": 0.60, "maximum": 0.70},
    "satellite": {"minimum": 0.28, "target": 0.38, "maximum": 0.48},
    "experiment": {"minimum": 0.00, "target": 0.02, "maximum": 0.04}
  }
}
```

- [ ] **Step 2: Validate and preview without changing policy state**

Run:

```bash
rtk .venv/bin/python -m cli policy validate --file docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json
rtk .venv/bin/python -m cli inspection preview --policy-file docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json --snapshot-id 4
```

Expected: each command returns one JSON object with `ok=true`; preview references snapshot 4 and does not change the active policy.

- [ ] **Step 3: Activate with optimistic concurrency**

Run:

```bash
rtk .venv/bin/python -m cli policy activate --file docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json --expected-current-version 5
rtk .venv/bin/python -m cli policy show --active
```

Expected: the first command activates version 6; the readback shows cash 5%, layer targets 60/38/2, GLD Core 10%, and the allocation-review configuration. No policy candidate is approved automatically.

- [ ] **Step 4: Update the README and commit the policy artifact**

Document the four primary benchmarks, preset targets, cooldown, `market sync`/`market context` flow, and explicit activation boundary. Avoid describing target gaps as orders.

```bash
rtk git add README.md docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json
rtk git commit -m "docs: add neutral dynamic allocation policy"
```

### Task 6: Verify the integrated system and current saved data

**Files:**
- Modify only if a failing test exposes a defect in a task-owned path.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
rtk .venv/bin/python -m ruff check services/dynamic_allocation.py services/policy_validation.py cli.py tests/test_dynamic_allocation.py tests/test_policy_store.py tests/test_cli.py tests/test_market_store.py
rtk .venv/bin/python -m ruff format --check services/dynamic_allocation.py services/policy_validation.py cli.py tests/test_dynamic_allocation.py tests/test_policy_store.py tests/test_cli.py tests/test_market_store.py
```

Expected: both commands exit zero.

- [ ] **Step 2: Run focused integration tests**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_dynamic_allocation.py tests/test_policy_store.py tests/test_market_store.py tests/test_toss_market.py tests/test_cli.py tests/test_inspection_service.py tests/test_inspection_engine.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the full test suite**

Run: `rtk .venv/bin/python -m pytest -q`

Expected: all tests pass without live-market access.

- [ ] **Step 4: Evaluate saved market evidence**

Run: `rtk .venv/bin/python -m cli market context`

Expected: one JSON object. If one or more required stored benchmarks are missing or stale, result is `Watch`/`observe` with no candidate; if all are current and the cooldown has expired, an eligible target difference may create a review candidate. In every case the active policy remains unchanged.

- [ ] **Step 5: Run a fresh saved-snapshot inspection and read it back**

Run:

```bash
rtk .venv/bin/python -m cli inspection run --snapshot-id 4
rtk .venv/bin/python -m cli inspection show --latest
```

Expected: both commands return `ok=true`, use active policy version 6, preserve the exact v2 status vocabulary, and contain no order fields.

- [ ] **Step 6: Audit the final diff and repository state**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; unrelated pre-existing user changes remain intact and clearly distinguishable from this feature's files/hunks.
