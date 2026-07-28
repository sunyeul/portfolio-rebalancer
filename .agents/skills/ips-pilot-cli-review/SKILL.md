---
name: ips-pilot-cli-review
description: Use when reviewing a saved IPS Pilot Toss inspection through the CLI, including latest evaluations, Review Queue summaries, risk context, or a safe Korean agent brief.
---

# IPS Pilot CLI Review

Use persisted inspection output as evidence, never as trading advice. Pair with
`ips-judgment-filter` whenever discussing allocation, drawdown, regular
purchases, or exceptional intervention.

## Choose the surface

- Saved evaluation: `uv run ips-pilot inspection show --latest` or
  `--run-id <id>`. `--latest` is the newest evaluation by `created_at`, then
  `id`; report both `run_id` and `snapshot_id`.
- Observed Toss snapshot: `uv run ips-pilot toss-snapshots --latest`. Do not
  treat its result as the latest evaluation without comparing `snapshot_id`.
- Proposed policy: `inspection preview --policy-file <path>` evaluates the
  supplied policy and does not create an evaluation run.
- New or refreshed evaluation: `inspection run` can persist or reuse an
  evaluation. Run it only when the user explicitly asks to create or refresh
  an evaluation.

`toss-snapshots`, `inspection preview`, and `inspection show` initialize the
local database before reading. If schema migration, database creation, or
default-policy seeding is not authorized, explain that no state-free CLI read
surface exists and request permission instead.

## Read the contract

The CLI emits one JSON object. Check `ok`, `error`, and
`contract_supported` before interpreting `evaluation`. A supported persisted
evaluation contains `run_id`, `snapshot_id`, `state`, and `evaluation`; read
the inspection result from `evaluation.result`.

Use these result fields as separate axes:

- `allocation_state`: whether allocation is evaluable.
- `status`: exactly `OK`, `Watch`, `Review`, or `Action`.
- `priority`: when to revisit the item (`P1`–`P4`).
- `queue_class`: `blocking`, `adjustment`, or `observation`.
- `suggestion`: the single adjustment direction, never an order.

Each `review_queue` item contains `priority`, `priority_label`, `queue_class`,
`kind`, `identity`, `status`, `triggers`, `suggestion`, `meaning`,
`verification_task`, and `evidence_refs`. Blocking items have null
priority and suggestion. Do not infer allocation metrics or a trade from a
queue item; inspect the linked cash, layer, or instrument record when needed.

## Response shape

1. State `run_id`, `snapshot_id`, and whether the contract is supported.
2. Summarize the four status counts and material allocation state.
3. For non-`OK` findings, give the raw trigger, plain Korean meaning, the
   adjustment direction, and the verification task.
4. End with verification or future regular-purchase-policy review—not a buy,
   sell, quantity, price, or execution instruction.

Missing, stale, unreconciled, or incomplete source data is a blocking
verification item. `Action` means inspect a possible exceptional intervention;
it does not authorize a trade.
