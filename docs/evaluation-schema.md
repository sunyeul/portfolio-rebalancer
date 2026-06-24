# Evaluation v2 Schema

## Domain Objects

```python
EvaluationPeriod(
    label="3M",
    start_date=date(2026, 3, 23),
    end_date=date(2026, 6, 23),
)
```

```python
EvaluationUnit(
    level="asset",
    name="QLD",
    parent_layer="experiment",
    benchmark="QQQ",
    target_weight=0.03,
    allowed_mdd=-0.25,
    allowed_volatility=0.45,
    max_weight=0.05,
    min_efficiency=0.2,
    thesis="leveraged Nasdaq experiment",
    counter_scenario="MDD exceeds experiment rule",
    check_frequency="monthly",
    manual_intervention_allowed=True,
    evaluation_period=period,
)
```

```python
EvaluationOutput(
    current_weight=0.02,
    weight_gap=-0.01,
    layer_internal_weight=1.0,
    period_return=0.04,
    cagr=0.17,
    benchmark_return=0.03,
    benchmark_excess_return=0.01,
    mdd=-0.12,
    volatility=0.35,
    concentration=0.02,
    risk_contribution=0.07,
    return_mdd_ratio=0.33,
    cagr_mdd_ratio=1.42,
    sharpe=0.41,
    sortino=0.52,
    thesis_status="watch",
    burden="high",
    status="Review",
)
```

## CLI Envelope

```json
{
  "ok": true,
  "command": "evaluate",
  "input": {
    "source": "snapshot",
    "snapshot_id": 14,
    "period": "YTD",
    "rf": 0.025,
    "bench": "SPY:80,QQQ:20"
  },
  "evaluation_period": {
    "label": "YTD",
    "start_date": "2026-01-01",
    "end_date": "2026-06-23"
  },
  "layer_evaluations": [],
  "asset_evaluations": [],
  "review_queue": [],
  "journal_draft": [],
  "warnings": [],
  "guardrails": {
    "not_investment_advice": true,
    "no_immediate_order_instruction": true
  },
  "error": null
}
```

## Review Item

```python
ReviewItem(
    level="asset",
    name="QLD",
    parent_layer="experiment",
    status="Review",
    triggered_by=["mdd_exceeded", "high_burden"],
    metrics_snapshot={"mdd": -0.32, "cagr_mdd_ratio": 0.4},
    thesis="leveraged Nasdaq experiment",
    counter_scenario="MDD exceeds experiment rule",
    suggested_next_step="Review thesis and experiment rule before changing exposure.",
)
```

The schema intentionally avoids `buy`, `sell`, or `execute` fields in v2 inspection outputs.
