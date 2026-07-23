# IPS Rebalance Priority Table Skill Design

## Goal

Add a project-local skill that produces a consistent Korean review table for
cash-reserve and portfolio-rebalancing discussions. The table is an inspection
artifact, not an order plan.

## Scope

Create `.agents/skills/ips-rebalance-priority-table/SKILL.md` only. No script,
API, database, snapshot, policy, or journal change is part of this work.

The skill reads the latest complete Toss snapshot and active IPS policy through
the existing CLI. It reports the actual `snapshot_id` and keeps cash
percentages separate from layer and instrument percentages:

- Cash uses gross account value.
- Layers and instruments use invested account value.
- A positive gap means the current weight exceeds the active policy target;
  a negative gap means it is below target.

## Workflow

1. Read `ips-judgment-filter` and `ips-pilot-cli-review`.
2. Discover the available CLI commands when necessary; use only read-only
   commands for snapshots, account projection, active policy, and persisted
   inspection results.
3. Select the latest complete snapshot by timestamp, resolving a tie by larger
   ID, and state the selected `snapshot_id` and saved-data time.
4. Extract the user's current review inputs before ranking. Inputs can include
   an analysis-layer override, an intended role such as a hedge, a cash goal,
   overlap concerns, a preference to preserve or simplify a holding, and tax or
   liquidity constraints. Restate the inputs used in the result.
5. Produce a layer summary and one table per `core`, `satellite`, and
   `experiment` layer. Every table has these columns: rank, instrument, current
   weight, target weight, gap, return, unrealized profit/loss, and review
   rationale.
6. Generate the review rationale and rank from the current user inputs first,
   then use role fit, concentration, overlap, target gap, and
   return/profit-loss as descriptive evidence. Never make return or loss a
   standalone trigger. Do not reuse a prior request's rationale when the user
   supplies new inputs.
7. Treat user-specified classification as an analysis override, not a policy
   mutation. Show the active policy layer and the analysis layer separately.
   Keep the policy target and gap unless the user explicitly supplies an
   override target; otherwise mark an override target as unspecified.
8. End with verification tasks and never provide order side, quantity, price,
   timing, execution instructions, or automatic trade recommendations.

## Guardrails

- Preserve the four v2 statuses: `OK`, `Watch`, `Review`, and `Action`.
- Do not use retired manual instrument metadata such as `thesis_status`,
  profile notes, overlap status, or management-burden status. The current
  runtime does not store them.
- Treat user review inputs as request-scoped context only. Never persist them
  to profiles, policies, snapshots, or journals.
- Do not change policies, snapshots, journal entries, or persistent state.
- Treat stale, incomplete, or unreconciled data as a verification issue.
- Explain that a rank is a human review order, not a transaction instruction.

## Validation

Validate the generated skill folder with the standard skill validator. Check a
representative request for: the actual snapshot ID, the denominator note,
mandatory columns, a separate layer summary, an explicit analysis-override
label when used, and no trade-sizing language.
