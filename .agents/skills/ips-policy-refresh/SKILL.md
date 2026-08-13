---
name: ips-policy-refresh
description: Use when recalculating or reviewing IPS target weights, creating a versioned policy candidate, or changing cash, instrument, or layer allocations. Calculate targets jointly at instrument level first, derive layer targets by aggregation, validate the candidate, and keep activation separate.
---

# IPS Policy Refresh

Create a reproducible, instrument-first IPS policy candidate. Pair this skill
with `ips-judgment-filter`; policy research never grants order or execution
authority.

## Preserve the boundary

- Treat refresh as read-only research until the user explicitly requests
  activation.
- Keep the active immutable policy unchanged while calculating, validating, and
  previewing a candidate.
- Use normalized Toss data only for observed account identities and account
  facts. Keep external research and market estimates labeled as analysis
  inputs.
- Do not use current weights, allocation gaps, cost basis, profit/loss, or
  drawdown to manufacture strategic targets. They may appear only in a separate
  transition or stability comparison.
- Do not call a weight "calculated" unless a reproducible calculator or fully
  specified equation, dated inputs, and constraints produced it. If that
  capability or evidence is absent, return `calculation_state=not_evaluable`
  and name the missing input or implementation.

## Calculate the candidate

1. Read the active policy version and its hash. Discover the current CLI and
   schema before using them. Treat CLI reads that initialize, migrate, or seed
   the database as stateful; use a read-only database connection when such
   changes are not authorized.
2. Freeze the calculation timestamp, eligible instrument universe, instrument
   identities, layer roles, cash rule, estimation window, return and risk
   estimates, overlap treatment, constraints, and calculation method.
3. Calculate cash against gross account value. State whether it comes from a
   liquidity reserve, risk rule, or another explicit method.
4. Calculate every instrument target jointly against invested account value.
   Make instrument targets the primary decision variables; do not calculate
   layer targets first and distribute them by inherited shares.
5. Derive each layer target as the sum of its instrument targets, using the
   repository policy tolerance for floating-point comparison:

   ```text
   layer_target[L] = sum(instrument_target[i] for i in layer L)
   ```

6. Treat layer minimums and maximums as strategic constraints or sensitivity
   envelopes, not as the source of target weights. If an aggregated layer
   violates a guardrail, rerun the instrument-level calculation with the
   constraint; never clip a layer total and proportionally rewrite instruments
   after the fact.
7. Derive instrument and layer ranges with a named, predeclared sensitivity or
   uncertainty method. Do not invent bands merely to make validation pass.
8. Convert invested-account targets to gross-account equivalents:

   ```text
   gross_instrument_target[i] = (1 - cash_target) * instrument_target[i]
   gross_layer_target[L] = (1 - cash_target) * layer_target[L]
   ```

9. Verify instrument-to-layer aggregation, layer target sum of `1`, and gross
   cash-plus-assets sum of `1` within the repository policy tolerance. Also
   verify role coverage, identity coverage, bounds,
   concentration, overlap, risk contribution, and stability across reasonable
   input perturbations.
10. Validate the candidate with the repository-owned policy validator. Use
    `policy validate` only when its possible database initialization is
    authorized.
11. Preview the candidate on an explicit snapshot and, when the user supplies a
    regular-contribution scenario, run the read-only historical
    `inspection candidate-preview`. Keep both previews non-persistent with
    respect to policy activation.
12. Produce a candidate artifact and stop. Run `policy activate` only in a
    separate step after an explicit activation request, using the expected
    current version to prevent races.

## Output the evidence before the conclusion

Return these sections in order:

1. Candidate summary: base version and hash, calculation timestamp,
   `calculation_state`, validation result, preview result, and activation state.
2. Calculation provenance: method, dated inputs, estimation windows, data
   quality, constraints, and unresolved assumptions.
3. Cash target: gross-account minimum, target, maximum, and derivation.
4. Instrument targets: identity, layer, minimum, target, maximum,
   gross-account equivalent, risk contribution, and calculation evidence.
5. Layer aggregation: the member-instrument arithmetic, derived target,
   guardrail, and gross-account equivalent.
6. Active-to-candidate differences: changes in cash, instruments, layers,
   method, roles, and constraints.
7. Validation and stability: policy invariants, feasibility, concentration,
   overlap, perturbation results, and candidate fingerprints.
8. Unresolved decisions and the next human approval step.

Use `valid`, `invalid`, or `not_evaluable` for candidate validation. Do not add
IPS inspection `status`, `priority`, `queue_class`, or `suggestion` fields to
policy-candidate research. Do not include order side, quantity, price, timing,
execution, or automatic activation fields.

## Fail closed

Return no complete policy candidate when any of these conditions holds:

- the eligible universe or layer role is unresolved;
- required market evidence is missing, stale, malformed, or temporally
  inconsistent;
- the calculation method, parameters, or constraints are not reproducible;
- an instrument target or range is guessed from the current account gap;
- instrument targets do not aggregate to their derived layer targets within
  the repository policy tolerance;
- ranges or layer constraints are infeasible;
- the policy validator rejects the result; or
- a removed allocation has no explicitly calculated destination.

Show partial diagnostics when useful, but label the candidate
`calculation_state=not_evaluable`, set the proposed policy to null, and state
the verification task.
