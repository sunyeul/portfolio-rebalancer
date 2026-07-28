---
name: ips-rebalance-priority-table
description: Create consistent Korean IPS Pilot portfolio-rebalancing priority tables from saved Toss snapshots and the active IPS policy. Use when users ask to rank holdings by layer, compare current weight, target, gap, return, and profit/loss, review cash-reserve rebalancing, or apply request-scoped role and priority overrides.
---

# IPS Rebalance Priority Table

Use this skill for inspection tables, never order plans. Read `ips-judgment-filter` and `ips-pilot-cli-review` before analyzing allocation, cash, or portfolio actions.

## Read-only data workflow

1. List saved snapshots and select the latest `complete` snapshot. Prefer `created_at`; when unavailable use `synced_at`; break a timestamp tie by the larger ID.

   ```bash
   uv run ips-pilot toss-snapshots --limit 20 \
     | jq '[.snapshots[] | {id, created_at, synced_at, state, is_current_evaluable}]'
   ```

2. Read the selected account projection and active policy.

   ```bash
   uv run ips-pilot account-view --snapshot-id <snapshot_id>
   uv run ips-pilot policy show --active
   ```

3. Read the latest persisted inspection only when its `snapshot_id` matches the selected snapshot.

   ```bash
   uv run ips-pilot inspection show --latest
   ```

4. State the actual `snapshot_id`, `synced_at`, and whether the source is complete and reconciled. If the source is stale, partial, failed, or unreconciled, report the data issue and do not create a confident priority ranking.

Do not run `toss-sync`, `policy activate`, or any command that changes snapshots, policy, journal, or orders.

## Request-scoped review inputs

Before ranking, extract and restate the inputs supplied in the current request. Inputs can include:

- a cash objective;
- a user-defined role, such as treating an instrument as a core hedge;
- holdings to preserve, simplify, or compare for overlap;
- tax, liquidity, or currency constraints;
- a candidate instrument that may fill a distinct role or return driver.

Use these inputs only for this response. They are analysis inputs, not current
account facts. Do not persist them as metadata or change the active policy. If
an input reclassifies a holding, show both `정책 레이어` and `분석 레이어`.
Retain the policy target and gap unless the user supplies an analysis target;
otherwise label the analysis target `미정`. Treat policy targets as directional
vectors and ranges, not mandatory destinations for every holding.

If the selected snapshot is partial, stale, failed, or unreconciled, do not
assert current weights, gaps, returns, profit/loss, or candidate holdings. You
may still show a clearly labeled conditional scenario—for example, “가정: 총계좌
평가액 기준 현금 10%”—and list the verification needed before using it as a
current-account conclusion.

## Calculations and table contract

- Cash weight and cash gap use gross account value.
- Layer and instrument weights and gaps use invested account value.
- Calculate `갭 = 현재 비중 - 목표 비중`.
- Round percentages and percentage-point gaps to one decimal place. Present profit/loss in 만원 or 백만원 consistently within a response.
- Treat return and unrealized profit/loss as descriptive evidence. Never make gains, losses, drawdowns, or an annual-return target a standalone trade trigger.

Start with this layer summary:

| 구분 | 현재 | 목표 | 갭 | 분모 |
| --- | ---: | ---: | ---: | --- |
| 현금 | | | | 총계좌 |
| Core | | | | 투자금 |
| Satellite | | | | 투자금 |
| Experiment | | | | 투자금 |

Then create one table for each analysis layer that has holdings. Use these columns:

| 순위 | 종목 | 정책 레이어 | 분석 레이어 | 현재 | 목표 | 갭 | 수익률 / 평가손익 | 우선 이유 |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |

Generate `우선 이유` from the current user inputs first. Use role fit, concentration, overlap, target gap, return, and profit/loss only as supporting evidence. Explain that the rank is a human review order, not a transaction instruction. Do not reuse a previous response's rationale after the user supplies a new input.

Give each relevant holding a plain analysis direction such as cash direction,
maintain, future-allocation direction, concentration-normalization review, or
exceptional-review. Label the table ordering as `분석 순위` and keep it
separate from persisted backend `priority` (`P1`–`P4`); do not reclassify or
overwrite the backend priority. When a persisted inspection or Review Queue
item is used, reproduce backend-owned `status`, `priority`, `queue_class`, and
`suggestion` unchanged; the analysis rank and direction are supplemental.

When a candidate instrument is supplied or discovered through clearly labeled
research, show it in a separate `후보 종목` section. A candidate is not a
current holding: do not give it current weight, gap, return, profit/loss,
status, queue class, or backend priority until it appears in a normalized Toss
snapshot. Screen its distinct role, cost, overlap, thesis, burden, liquidity,
tax, and policy fit before carrying it into a future policy comparison.

## Guardrails

- Preserve `OK`, `Watch`, `Review`, and `Action` exactly when reporting inspection results.
- Do not use retired manual instrument metadata such as `thesis_status`, profile notes, overlap status, or management-burden status.
- Do not give an order side, quantity, price, timing, execution instruction, or automatic buy/sell recommendation.
- End with verification work, such as confirming the user role input, policy-target mismatch, tax treatment, liquidity, or current-data freshness.
