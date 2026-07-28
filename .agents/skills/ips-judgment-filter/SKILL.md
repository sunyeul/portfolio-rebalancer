---
name: ips-judgment-filter
description: Use when designing, reviewing, or explaining IPS Pilot judgment about regular purchases, allocation, drawdowns, core/satellite/experiment roles, or exceptional human intervention.
---

# IPS Judgment Filter

IPS Pilot is an inspection workbench, not a trading recommender. It can make a
specific, evidence-based case for what a person should verify or reconsider;
it must not create an order, size, side, price, or execution instruction.

## Preserve the v2 frame

Keep `core`, `satellite`, and `experiment` first-class. The only top-level
statuses are:

- `OK`: inside the inspection frame.
- `Watch`: a soft observation for the next review.
- `Review`: a human verification item.
- `Action`: inspect a possible exceptional intervention; never permission to
  trade.

Do not add a fifth status to express detail. Use the independent result axes
instead:

- `allocation_state`: `complete`, `partial`, or `not_evaluable`.
- `priority`: `P1` before the next regular purchase, `P2` in the monthly
  review, `P3` in the next regular-purchase allocation, or `P4` observe.
- `queue_class`: `blocking`, `adjustment`, or `observation`.
- `suggestion`: the backend-owned, user-visible adjustment direction.

Use `suggestion` as the one action-level answer: secure cash by reducing or
pausing future regular purchases, increase their pace when cash is excessive,
increase an underweight allocation, normalize an overweight allocation, hold
and observe, or inspect a possible exceptional intervention. It adds no order
side, quantity, price, or execution authority.

## Judgment posture

Use the normalized Toss snapshot and persisted policy for all account and
policy facts. Keep three input classes separate:

- account facts: holdings, cash, cost, price, order, and execution facts from
  the current normalized Toss snapshot;
- explicit user inputs: a role, target, constraint, or objective supplied for
  this response's analysis layer;
- analysis assumptions: clearly labeled scenario inputs or external research
  that support comparison but never replace account facts or the active policy.

Explain the raw trigger, its IPS meaning, the adjustment direction, and the
next verification task in plain Korean. A user-defined role or target can guide
the analysis layer without silently changing the policy or backend result.

- Underweight exposure can justify reviewing future regular-purchase allocation
  when the source and cash policy support it.
- Overweight or higher-risk exposure may support reduced/paused future regular
  purchases, maintain/observe, simplification, hedging, concentration review,
  or exceptional-review paths. Choose among them from the supported evidence
  and explicit user intent; state the direction as inspection guidance, never
  as a sell instruction.
- Cash is a strategy asset: assess it against gross-account weight; assess
  layers and instruments against invested-account weight.
- Performance, gains, losses, and drawdown are evidence for observation or
  risk review. They are not standalone trade triggers.
- For a backend evaluation, missing, stale, unreconciled, or incomplete source
  data remains blocking: `allocation_state=not_evaluable`, null priority, and
  no suggestion. In explanatory prose, state the unavailable fact and its
  verification task; this does not prevent a clearly labeled conditional
  scenario that avoids claiming current account weights.

The annual-return target is the active policy's
`performance.annual_return_target`; it is descriptive context, not a trigger.

## Evidence-based flexibility

Do not reduce a judgment to “hold” when the supported evidence permits a more
useful conditional analysis. An explanation may compare regular-purchase
adjustment, observation, concentration review, simplification, hedging, and an
exceptional-review path, then name the evidence still required. Give each
relevant holding a plain direction such as cash direction, maintain, future
allocation direction, or review for normalization. A direction is a human
review vector, not an order.

When the user supplies an explicit role, target, or constraint, show it as a
response-scoped analysis input and distinguish it from the active policy. When
the evidence does not select one path, compare alternatives rather than
collapsing to generic observation. A suitable new instrument may be discussed
as a separate candidate when it adds a distinct role, return driver, hedge, or
diversification benefit; it must not be treated as a current holding fact.

It may say that an exceptional intervention is worth inspecting only when
explicitly supported evidence exists. It must not turn that analysis into a
transaction recommendation.

Satellite and experiment exposure receive stricter review of the evidence the
runtime actually supports, including role, allocation range, and drawdown
thresholds. Do not invent retired manual thesis, overlap, burden, holdability,
or ETF-substitution metadata to force a result.

## Hard stops

- Do not add buy, sell, execute, order-size, price, or execution fields.
- Do not reimplement or override status classification outside the backend.
- Do not convert an allocation gap, cash breach, performance result, or
  drawdown alone into `Action`.
- Do not infer layer or instrument advice from an all-cash `partial` result or
  a blocking `not_evaluable` result.
