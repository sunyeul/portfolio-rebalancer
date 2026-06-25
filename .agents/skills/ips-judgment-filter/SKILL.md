---
name: ips-judgment-filter
description: Use when designing, reviewing, or implementing this portfolio rebalancer app's IPS-based investment judgment behavior, especially decisions about regular buying, core/satellite allocation, drawdown responses, immediate buys, sells, or conservative hold decisions.
---

# IPS Judgment Filter

## Purpose

Keep IPS Pilot from becoming an automatic trading recommender. The app may inspect whether the IPS suggests future regular-purchase changes, observation, thesis review, or exceptional human review.

## V2 Frame

IPS Pilot v2 evaluates both layers and assets as inspection units. Keep `core`, `satellite`, and `experiment` as first-class layers, and use only the status vocabulary `OK`, `Watch`, `Review`, and `Action`.

- `OK`: inside the inspection frame.
- `Watch`: soft warning.
- `Review`: human verification item.
- `Action`: broken thesis plus hard limit breach; inspect possible exceptional intervention, not permission to trade.

## Decision Posture

The default adjustment mechanism is future regular-purchase policy, not immediate trading.

- Underweight exposure: inspect whether future regular purchases can increase.
- Overweight or higher-risk exposure: inspect whether future buying can reduce or pause before considering a sale.
- Weak data, unclear classification, or low conviction: prefer observe, verify data, or review thesis.
- Immediate buying or selling is exceptional and requires explicit human judgment.

## Explainability

Use internal status and trigger labels for consistency, but translate them into plain review language for humans. Explain what changed, why it matters under the IPS, and what should be verified next.

Good explanation shape: raw label, plain meaning, verification task. Example: `risk_contribution_high` means the unit is carrying a large share of portfolio risk, so verify whether that concentration is intentional and still acceptable.

Do not let explanation become an order. A readable explanation may say "review future regular-purchase policy" or "verify thesis and risk limit fit"; it should not say "buy", "sell", or size a trade.

## Layer Rules

Core assets are for long-term market participation. Do not penalize normal core drawdowns as standalone failure signals.

Satellite and experiment exposure require stricter thesis, overlap, management-burden, holdability, and ETF-substitution checks. A fallen satellite is a thesis-review candidate before it is an accumulation candidate.

## Immediate Buying Is Exceptional

Never treat a drop, cheap-looking price, premarket move, average-cost defense, or fear/FOMO as a standalone buy reason. For underweight assets, prefer regular-purchase adjustment language unless the user explicitly asks to inspect exceptional action.

## Selling Is More Exceptional

Never recommend selling only because returns are good, returns are poor, the asset fell, the asset rose, or unrealized gains are large. Selling language must be framed as thesis damage, simplification, consolidation, or allocation/risk control.

## Allowed Outcome Language

- Increase future regular purchases.
- Reduce or pause future regular purchases.
- Hold and observe.
- Review investment thesis, overlap, burden, or ETF substitution.
- Inspect possible exceptional action; if required conditions are not confirmed, hold.

## Hard Stops

- Do not add order sizing, execution flags, or automatic buy/sell recommendations.
- Do not collapse layer logic into ticker-only logic.
- Do not turn stale, missing, or ambiguous data into action.
