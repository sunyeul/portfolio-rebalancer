---
name: ips-judgment-filter
description: Use when designing, reviewing, or implementing this portfolio rebalancer app's IPS-based investment judgment behavior, especially decisions about regular buying, core/satellite allocation, drawdown responses, immediate buys, sells, or conservative hold decisions.
---

# IPS Judgment Filter

## Core Role

Use this skill as the app's investment policy filter. The app should not behave like an automatic trading recommender. It should help the user decide whether the current portfolio state calls for increasing regular purchases, reducing regular purchases, observing, rechecking the investment thesis, or treating an immediate action as an exception.

The central question is:

> Given the current portfolio, does the user's investment policy say to adjust future buying, wait, recheck the thesis, or only exceptionally consider immediate buy/sell action?

## Default Posture

The default adjustment mechanism is regular purchase allocation, not immediate trading.

- If an asset is below target, first ask whether future regular purchases can be increased.
- If an asset is above target or risk has risen, first ask whether future buying can be reduced or paused.
- Immediate buying or selling is considered only after regular purchase adjustment is insufficient.
- When data is weak, classification is unclear, or conviction is low, prefer hold, observe, or thesis review.

## Judgment Order

Follow this order before recommending any strong action:

1. Check whether the portfolio is missing core exposure.
2. Check whether the investment thesis still holds.
3. Check whether regular purchase adjustment can address the gap.
4. Consider exceptional immediate buying only if regular purchase adjustment is not enough.
5. Treat satellite assets more strictly: confirm long-term holdability, thesis quality, overlap, management burden, and whether an ETF substitute is better.

In ambiguous situations, choose core reinforcement over new satellite exposure. If capital must be deployed but the satellite case is unclear, the default destination is core, not the satellite idea.

## Core Assets

Core assets are for long-term market participation, not short-term performance chasing.

- If a core asset is below target and its thesis remains intact, prioritize reinforcing it through regular purchase increases.
- In a drawdown, weak short-term performance or low efficiency does not make a core asset bad by itself.
- A drawdown is usually a time to secure sound core exposure at better prices, not a reason to expand satellite bets first.
- For underweight core assets in a drawdown, prefer "increase regular purchases" over "buy immediately."

## Satellite Assets

Satellite assets require stricter judgment. The first question is not how much upside remains, but whether the user can responsibly hold the asset for a long time.

Before increasing satellite exposure, verify:

- The industry, company, or theme thesis remains alive.
- The user can hold it for at least five years.
- The user can actually manage the position.
- An ETF would not be a cleaner substitute.
- It does not duplicate existing positions unnecessarily.
- The motivation is not simply that the price looks cheaper.

In a drawdown, a satellite asset is usually a thesis review candidate before it is an additional purchase candidate.

## Drawdown Behavior

In falling markets, the app should not encourage indiscriminate dip-buying.

The preferred drawdown response is:

1. Determine whether core exposure is below target.
2. Confirm the thesis is intact.
3. Use regular purchase increases where possible.
4. Consider exceptional additional buying only after the above checks.
5. Apply stricter thesis and holdability checks to satellites.

The key behavior is "core-first regular purchase reinforcement," not "buy whatever fell."

## Immediate Buying Is Exceptional

The app should not say to buy just because an asset dropped sharply.

Only consider immediate buying when all of these are reasonably true:

- The investment thesis remains intact.
- There is room under the target allocation.
- The price is attractive enough relative to long-term expectation.
- Regular purchase increases are insufficient.
- The action does not increase management complexity materially.
- The motivation is not FOMO.
- The goal is not merely lowering average cost.

Do not treat these as valid standalone buy reasons:

- A one-day sharp drop.
- A large premarket move.
- A desire to lower average cost.
- Other investors being fearful.
- The asset simply looking cheap.

The relevant IPS focus is long-term expected return and target allocation, not average cost defense.

## Selling Is More Exceptional

The app should not recommend selling only because returns are good or bad.

Selling can be considered when:

- The investment thesis is damaged.
- The position can be consolidated into a better asset.
- The portfolio needs simplification.
- The position is far above target allocation.

Do not treat these as valid standalone sell reasons:

- The asset rose a lot.
- The asset recently fell.
- Recent performance was poor for a few months.
- There is a short-term negative event.
- There are large unrealized gains.

Selling should be framed as thesis damage, simplification, or allocation control, not emotional response.

## Insufficient Data

When data is incomplete, stale, unreliable, or classification is unclear, the app should become more conservative.

The right outcome is usually hold, observe, or thesis review. Do not turn unreliable data into buy or sell recommendations.

## Final Judgment Categories

Use these five natural-language outcomes as the app's decision vocabulary:

1. Increase regular purchases: the asset is below target and can be accumulated under the IPS, but the adjustment should happen through future buying rather than immediate purchase.
2. Reduce or pause regular purchases: the asset is above target or risk has risen, so new buying should be reduced before considering a sale.
3. Hold and observe: there is no need to act now; defer investment judgment until the next review.
4. Review investment thesis: numbers alone are insufficient; recheck the reason to hold, ETF substitution, overlap, and management burden.
5. Consider exceptional action or hold: immediate buy/sell is exceptional, and if the required conditions are not confirmed, hold.

## Common Mistakes

- Treating underweight assets as automatic immediate buys.
- Treating overweight assets as automatic sells.
- Penalizing core assets for normal drawdown behavior.
- Treating fallen satellites as opportunities before thesis review.
- Using average cost, premarket movement, or one-day price changes as decisive reasons.
- Forcing a buy/sell decision when the policy answer is to wait.
