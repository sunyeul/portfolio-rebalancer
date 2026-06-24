# Auxiliary Score Visibility Design

## Context

The portfolio evaluation already calculates supporting performance and efficiency signals such as `efficiency_score`, `return_total_pct`, Sharpe-related metrics, return contribution, detailed IPS component scores, and `efficiency_warning`. The current UI exposes many of these values only in dense tables, so the primary visible judgment can feel dominated by target allocation and risk contribution.

## Goal

Make return and efficiency signals visible beside IPS allocation/risk judgments without turning the app into a short-term performance-chasing recommender.

## Approved Direction

Use a hybrid of:

- Option C: a compact summary area plus a dedicated supporting-score table.
- Option A: inline supporting-signal chips inside the existing action summary.

## UX Behavior

The evaluation result should show a new supporting-score view that makes these questions easy to answer:

- Which increase candidates have weak efficiency?
- Which candidates are supported by both allocation gap and efficiency?
- Which assets have strong recent return but weak efficiency or risk posture?
- Which low-efficiency assets are still allowed by IPS but should receive lower priority or closer review?

The action summary should also include small chips for the most important supporting signals:

- Efficiency warning.
- Strong or weak period return.
- High or medium IPS fit.
- Risk contribution overage.

These chips are explanatory, not a separate buy/sell trigger.

## Data

Use data already available in `EvaluationResponse`:

- `evaluation.proposal`: `efficiency_score`, `return_total_pct`, `ips_fit_score`, `ips_fit_band`, `risk_over`, `efficiency_warning`, `gap_pct`, `suggested_trade_pct`, `action_reason`.
- `evaluation.ips_actions`: Korean CSV-compatible fields such as `E`, `return_total%`, `IPS적합도`, `IPS등급`, `risk_over`, `efficiency_warning`, `action_label`, and `decision_summary`.

No backend schema change is required for the first implementation.

## Components

Add focused frontend helpers inside `frontend/src/App.tsx`, matching the current app style:

- A formatter for compact score labels.
- A chip component for supporting signals.
- A supporting-score summary function that counts efficiency warnings, risk overages, high IPS fit rows, and strong/weak period return rows.
- A supporting-score table column set for `evaluation.proposal`.

## UI Placement

In the evaluation section:

- Keep the current tabs: `플레이북`, `액션 요약`, `로직 확인`, `점수 구성`.
- Add a new tab named `성과·효율`.
- In `액션 요약`, add a compact signal-chip row inside the recommended action cell.
- In `성과·효율`, show summary cards followed by the supporting-score table.

## Visual Rules

- Use restrained card styling consistent with the existing dashboard.
- Use green for supportive signals, amber for attention signals, red for risk/negative signals, and blue for informational IPS fit.
- Keep text short enough for table cells and mobile wrapping.
- Avoid implying that high return alone is a buy signal.

## Testing

Add frontend unit coverage if an existing test harness is available. If not, run the existing frontend type/build check. Also run the backend tests that cover evaluation serialization to confirm the API remains compatible.

## Out of Scope

- Changing IPS scoring weights.
- Adding new backend metrics.
- Rewriting the evaluation decision logic.
- Treating return or efficiency as a standalone action trigger.
