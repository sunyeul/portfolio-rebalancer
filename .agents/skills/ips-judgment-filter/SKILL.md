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

Use the normalized Toss snapshot and persisted policy as the only factual
source. Explain the raw trigger, its IPS meaning, the adjustment direction,
and the next verification task in plain Korean.

- Underweight exposure can justify reviewing future regular-purchase allocation
  when the source and cash policy support it.
- Overweight or higher-risk exposure should first prompt reduced/paused future
  regular purchases or overweight normalization review, not a sell instruction.
- Cash is a strategy asset: assess it against gross-account weight; assess
  layers and instruments against invested-account weight.
- Performance, gains, losses, and drawdown are evidence for observation or
  risk review. They are not standalone trade triggers.
- Missing, stale, unreconciled, or incomplete source data is blocking:
  `allocation_state=not_evaluable`, null priority, and no suggestion.

The annual-return target is the active policy's
`performance.annual_return_target`; it is descriptive context, not a trigger.

## Evidence-based flexibility

Do not reduce a judgment to “hold” when the supported evidence permits a more
useful conditional analysis. An explanation may compare regular-purchase
adjustment, observation, concentration review, and an exceptional-review path,
then name the evidence still required. It may say that an exceptional
intervention is worth inspecting only when explicitly supported evidence
exists. It must not turn that analysis into a transaction recommendation.

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
