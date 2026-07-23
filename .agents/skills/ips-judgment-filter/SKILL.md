---
name: ips-judgment-filter
description: Use when designing, reviewing, or implementing this portfolio rebalancer app's IPS-based investment judgment behavior, especially decisions about regular buying, core/satellite allocation, drawdown responses, immediate buys, sells, or conservative hold decisions.
---

# IPS Judgment Filter

## Purpose

Keep IPS Pilot from becoming an automatic trading recommender. The app may
inspect future regular-purchase policy, observation, thesis review, or
exceptional human review, but it never places, sizes, or prepares an order.

## V2 Frame

IPS Pilot evaluates Toss-observed allocation units while keeping `core`,
`satellite`, and `experiment` first-class. Status vocabulary is exactly:

- `OK`: inside the inspection frame.
- `Watch`: soft warning.
- `Review`: human verification item.
- `Action`: inspect possible exceptional intervention, not permission to trade.

The result contract has four independent axes. Never use one axis as a
substitute for another:

- `allocation_state`: `complete`, `partial`, or `not_evaluable` describes
  whether allocation can be judged. A valid all-cash account is `partial`: cash
  can be judged, while layers and instruments are withheld because the invested
  denominator is zero.
- `status`: the four severity values above describe the inspection finding.
- `priority`: `P1` through `P4` describe when to revisit an evaluable item, not
  how severe its status is. The fixed mapping is `P1` = `다음 정기매수 전`,
  `P2` = `이번 월간 점검`, `P3` = `다음 정기매수 배분 반영`, and `P4` =
  `관찰 유지`.
- `suggestion`: use only the closed review vocabulary
  `review_increase_regular_purchase_pace`,
  `review_reduce_or_pause_regular_purchase_pace`,
  `review_increase_regular_purchase_allocation`,
  `review_overweight_normalization`, `review_thesis_or_constraints`,
  `inspect_exceptional_intervention`, and `hold_and_observe`. These labels are
  inspection language, never order instructions.

`review_queue` contains all non-`OK` diagnostics and uses queue classes
`blocking`, `adjustment`, and `observation`. Blocking data-quality items have
no priority and no suggestion. `adjustment_suggestions` is the result-first
subset for cash, layer, and instrument allocation-policy review; performance
and account-risk diagnostics remain in the queue unless they independently
qualify as an allocation review.

## Decision Posture

The default adjustment mechanism is future regular-purchase policy, not
immediate trading.

- Underweight exposure: inspect whether future regular purchases can increase.
- Overweight or higher-risk exposure: inspect whether future buying can reduce
  or pause before considering a sale.
- Weak data, unclear classification, or low conviction: prefer observe, verify
  data, or review thesis.
- Immediate buying or selling is exceptional and requires explicit human
  judgment.

Cash is a strategy asset, not a fourth investment layer. Evaluate cash as a
gross-account weight against its policy band; evaluate layers and instruments
against the invested-account denominator. A cash breach is an allocation
checkpoint and `Review`, not `Action`: below the minimum is `P1` and means
reduce or pause future regular-purchase pace; above the maximum is `P2` and
means increase future regular-purchase pace. It never creates `Action` or a
sell instruction by itself. An overweight layer/instrument is an
overweight-normalization review; an eligible underweight is a future
regular-purchase allocation review.

Allocation and performance evidence are deliberately decoupled. Missing or
partial performance makes principal, return, YTD, or P&L facts unavailable or
descriptive, but must not suppress a valid allocation result. Conversely, a
stale, failed, unreconciled, or otherwise non-current-evaluable Toss source is
`allocation_state=not_evaluable`: emit a blocking queue item with null priority
and no suggestion, regardless of the apparent size of a gap.

## Explainability

Use internal status and trigger labels for consistency, but translate them
into plain review language. Explain what changed, why it matters under the
IPS, and what should be verified next.

Good explanation shape: raw label, plain meaning, verification task. Example:
`risk_contribution_high` means the unit carries a large share of portfolio
risk, so verify whether that concentration is intentional and acceptable.

Do not let explanation become an order. A readable explanation may say
"review future regular-purchase policy" or "verify thesis and risk limit fit";
it should not say "buy", "sell", or size a trade.

The annual return objective is descriptive context, not a trigger. The
configured annual target is the YTD account return (currently 10%); YTD below
target is a performance `Watch`/observation item and cannot, by itself, create
an allocation suggestion or `Action`. Cumulative return, gains, losses, and
drawdown follow the same rule.

## Layer Rules

Core assets are for long-term market participation. Do not penalize normal
core drawdowns as standalone failure signals.

Satellite and experiment exposure require stricter thesis, overlap,
management-burden, holdability, and ETF-substitution checks. A fallen
satellite is a thesis-review candidate before it is an accumulation candidate.

## Immediate Buying Is Exceptional

Never treat a drop, cheap-looking price, premarket move, average-cost defense,
or fear/FOMO as a standalone buy reason. For underweight assets, prefer future
regular-purchase adjustment language unless the user explicitly asks to inspect
exceptional action.

## Selling Is More Exceptional

Never recommend selling only because returns are good, returns are poor, the
asset fell, the asset rose, or unrealized gains are large. Frame any reduction
as thesis damage, simplification, consolidation, or allocation/risk control.

`Action` has one narrow conjunction: the same instrument must have
`thesis_status=broken` and exceed that instrument's own hard maximum weight. A
layer maximum breach, account drawdown, cash shortfall, loss, overlap, or
profile uncertainty alone is never `Action`. Even when `Action` is present,
write only "inspect possible exceptional intervention" and provide no order
side, quantity, or execution field.

## Allowed Outcome Language

- Increase future regular purchases.
- Reduce or pause future regular purchases.
- Hold and observe.
- Review investment thesis, overlap, burden, or ETF substitution.
- Inspect possible exceptional action; if required conditions are not
  confirmed, hold.

## Hard Stops

- Do not add order sizing, execution flags, or automatic buy/sell
  recommendations.
- Do not collapse layer logic into ticker-only logic.
- Do not turn stale, missing, or ambiguous data into action.
- Do not make a performance target, return, gain, loss, or drawdown a trade
  trigger.
- Do not assign `Action` unless the same-instrument broken-thesis and own-max
  conjunction is proven.
- Do not convert a blocking `not_evaluable` result or an all-cash `partial`
  result into a guessed layer/instrument recommendation.
