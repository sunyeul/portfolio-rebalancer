---
name: ips-pilot-cli-review
description: Use when Codex needs to review an IPS Pilot portfolio snapshot through the v2 CLI, including latest snapshot checks, Review Queue summaries, risk checks, and safe agent-facing summaries.
---

# IPS Pilot CLI Review

## Purpose

Use the IPS Pilot CLI as the source of truth for saved portfolio reviews. Summaries are inspection signals only, not trading advice.

Pair with `ips-judgment-filter` whenever the response includes buy, sell, DCA, drawdown, or allocation language.

## Workflow

1. If the portfolio is unclear, run `uv run ips-pilot portfolios list`.
2. If the snapshot is unclear, run `uv run ips-pilot snapshots list --portfolio-id <portfolio_id>`.
3. For `latest`, choose the newest `created_at`; if missing or tied, choose the largest `id`. Always state the actual `snapshot_id`.
4. Use `uv run ips-pilot agent-brief --snapshot-id <snapshot_id>` by default.
5. Use `review-queue`, `risk`, or `evaluate` only when the user asks for that narrower or fuller surface.

## Current Output Contract

The CLI prints one JSON object to stdout. Failures must remain machine-readable.

For `agent-brief`, read `input`, `evaluation_period`, `status_summary`, `review_queue`, `guardrails`, `warnings`, and `error`. `status_summary` is the combined layer plus asset count for `OK`, `Watch`, `Review`, and `Action`.

For separate layer and asset records, run `evaluate` and read `layer_evaluations`, `asset_evaluations`, `review_queue`, `journal_draft`, `warnings`, and `guardrails`.

For `risk`, read `risk_review`, not `review_queue`.

Review Queue items use `level`, `name`, `parent_layer`, `status`, `triggered_by`, `metrics_snapshot`, and `suggested_next_step`.

## Interpretation

Keep raw labels for traceability, but explain them in human-readable language. Do not show only `triggered_by` codes unless the user asks for raw output.

| Status | Meaning |
| --- | --- |
| `OK` | Inside the IPS inspection frame |
| `Watch` | Soft warning for next review |
| `Review` | Human verification item |
| `Action` | Inspect possible exceptional intervention; not permission to trade |

Common trigger explanations:

| Trigger | Human explanation |
| --- | --- |
| `target_gap_outside_tolerance` | Current weight is meaningfully away from the IPS target; review future regular-purchase policy. |
| `max_weight_exceeded` | Exposure is above the layer or asset cap; inspect allocation and risk concentration. |
| `risk_contribution_high` | This unit contributes a large share of portfolio risk; check whether the burden is intentional. |
| `mdd_exceeded` | Drawdown exceeded the configured tolerance; verify thesis and risk limit fit. |
| `volatility_exceeded` | Volatility exceeded the configured tolerance; review whether this still fits the layer role. |
| `thesis_watch` | Thesis is marked watch; verify whether the reason to hold remains strong enough. |
| `efficiency_below_threshold` | Risk-adjusted efficiency is weak for this frame; use as a review prompt, not a trade signal. |
| `high_burden` | Position or layer has high management burden; inspect whether it is worth the complexity. |
| `insufficient_performance_data` | Data is not enough for a strong judgment; verify data before interpreting the status. |

When a trigger is unknown, preserve the raw label and explain only what can be inferred from the surrounding metrics.

## Response Contract

1. State the evaluated `snapshot_id`.
2. State the evaluation period.
3. Summarize `status_summary`, or layer/asset records when using `evaluate`.
4. List Review Queue or risk items with status, raw trigger labels, human explanation, and suggested next step.
5. End with verification work, not trading instructions.

Use concise Korean by default when the user writes in Korean.

## Safety Rules

- Do not create buy, sell, execute, order-size, or execution flags.
- Do not edit snapshots, config, journal entries, or persistent state unless explicitly asked.
- Treat stale, missing, or ambiguous data as a reason to verify or observe.
