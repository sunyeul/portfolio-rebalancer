# Flexible IPS Skill Guardrails Design

## Context

The project-local IPS skills currently protect the inspection-only product
boundary, but several absolute phrases also push valid analysis toward generic
observation or regular-purchase language. This conflicts with the desired
collaboration style: use current account evidence, incorporate explicit user
intent, compare alternatives, and state a clear direction for each instrument
without turning that direction into an order.

The change will make judgment and presentation more flexible. It will not
expand execution authority, mutate persisted state without approval, change
the backend status contract, or introduce an alternative source of account
facts.

## Goals

- Separate non-negotiable product boundaries from evidence-dependent judgment
  rules and response-style defaults.
- Allow explicit per-instrument direction, conditional scenarios, new-instrument
  comparisons, and user-defined analytical roles or targets.
- Treat policy targets as directional vectors rather than mandatory endpoints.
- Permit conditional analysis when data is incomplete while preventing stale or
  unsupported facts from being presented as current account truth.
- Keep all outputs within the IPS inspection and human-review frame.

## Non-goals

- Adding or implying automatic trading, order side, quantity, price, timing, or
  execution permission.
- Changing the `OK`, `Watch`, `Review`, and `Action` vocabulary or reclassifying
  backend results in a skill, CLI, or frontend.
- Using manual holdings, yfinance, another broker, or a foreign-account fallback
  as account facts.
- Changing snapshots, policies, journals, evaluations, or other persistent
  state without an explicit user request.
- Modifying backend classification or API contracts.

## Guardrail model

The skills will distinguish four kinds of instruction instead of expressing
all caution as absolute prohibitions.

### 1. Product invariants

These remain hard boundaries:

- Normalized Toss snapshots are the source of holdings, cash, cost, price,
  order, and execution facts.
- Persisted backend output owns status, priority, queue class, and suggestion.
- Skills do not create an order, order-sized field, execution flag, or automatic
  buy/sell recommendation.
- Persistent state changes require an explicit user request.

### 2. Evidence discipline

Account facts, user inputs, and analysis assumptions must be kept distinct:

- Account facts come from a current, complete, reconciled Toss snapshot.
- Explicit user inputs may define an analytical role, target, constraint, or
  objective for the current response.
- Analyst assumptions and external research may support scenarios or candidate
  comparisons when clearly labeled; they do not replace account facts or the
  active policy.
- Missing or stale data blocks confident claims about the current account. It
  does not block a conditional scenario whose assumptions and verification
  needs are stated.

### 3. Judgment flexibility

Within those boundaries, responses should be specific:

- State a plain direction for each relevant holding: cash direction, maintain,
  future-allocation direction, concentration normalization, or exceptional
  intervention review.
- Compare more than one plausible path when the evidence does not select a
  single answer.
- Consider suitable new instruments when they add a distinct return driver,
  hedge, diversification benefit, or role improvement.
- Use policy targets as vectors. Do not mechanically force every holding to its
  target or treat a target gap as a standalone trade trigger.
- An `Action` result can support a concrete explanation of why exceptional
  human intervention deserves review, but never authorizes a transaction.

### 4. Side-effect authority

Interpretation of already available evidence may proceed autonomously. A CLI
command that initializes or migrates the database is not state-free merely
because its user-facing purpose is reading. Commands that can initialize or
change storage, create or refresh evaluations, sync snapshots, activate policy,
or otherwise persist state remain gated by an explicit user request. A
user-authorized state change is still subject to the repository contract and is
outside this skill-editing scope.

## Skill changes

### `ips-judgment-filter`

- Replace the single "only factual source" posture with the three-way split of
  account facts, explicit user inputs, and labeled analysis assumptions.
- Reframe regular-purchase adjustment as one preferred path, not the automatic
  conclusion for every overweight, drawdown, or cash issue.
- Allow explicit directional conclusions and comparison of regular-purchase,
  allocation, simplification, hedging, observation, and exceptional-review
  paths when supported.
- Keep the status vocabulary, source boundary, and execution hard stops intact.

### `ips-pilot-cli-review`

- Preserve the saved-result contract and actual `run_id`/`snapshot_id`
  reporting.
- Change missing-data guidance so it blocks definitive current-account claims
  but still permits labeled conditional scenarios and a verification list.
- Allow the agent brief to explain alternatives and a clear direction without
  inferring a transaction from a Review Queue item.
- Keep evaluation creation or refresh behind explicit authorization.

### `ips-rebalance-priority-table`

- Accept request-scoped analytical roles, targets, constraints, and candidate
  instruments as first-class table inputs.
- Show policy and analysis targets separately when they differ; do not silently
  overwrite or persist the policy.
- Rank human review directions using current account evidence and user intent,
  with return, profit/loss, concentration, overlap, and gap as supporting
  evidence rather than mechanical triggers.
- Label a request-scoped analysis rank separately from persisted backend
  `priority`; never present one as the other.
- Permit a separate candidate-instrument section when a new instrument has a
  distinct role. Do not mix a non-held candidate into current-weight or
  unrealized-profit calculations.

### `ips-pilot-frontend-workbench`

The frontend skill already protects a presentation boundary rather than
forcing conservative judgment. It will be changed only if wording alignment is
needed after pressure testing. It must continue to render backend-owned status
and priority without deriving replacements client-side.

## Response flow

1. Establish whether current account facts are complete and reconciled.
2. Restate explicit user roles, objectives, targets, and constraints.
3. Separate policy facts from response-scoped analytical assumptions.
4. Present per-instrument direction and portfolio-level trade-offs.
5. Compare alternatives or new candidates when they materially improve the
   portfolio role structure.
6. State uncertainty and the next verification task.
7. End with a human review boundary, not an order instruction.

## Error and ambiguity handling

- If account data is stale, partial, failed, or unreconciled, identify the
  unavailable fact and avoid a confident current-weight conclusion.
- If the user's analytical target conflicts with active policy, show both and
  label the difference rather than choosing one silently.
- If a requested direction would require an order, provide the inspection
  rationale and decision criteria but omit side, quantity, price, and timing.
- If a new candidate lacks enough role, cost, overlap, or liquidity evidence,
  keep it as a research candidate rather than placing it in the ranked holdings
  table.

## Skill TDD and verification

Before editing each skill, run baseline scenarios against the current text and
record where it becomes unnecessarily noncommittal. Re-run the same scenarios
after the minimal edit.

Flexibility scenarios must verify that the revised skill can:

- use an explicit user-defined hedge or core role without changing policy;
- give a distinct direction for every holding instead of assigning one generic
  hold response;
- compare an existing allocation with a suitable new candidate;
- provide a conditional scenario when the snapshot is incomplete;
- treat targets as vectors rather than compulsory destinations.

Safety scenarios must verify that the revised skill still refuses to:

- invent current account facts from assumptions or external sources;
- emit order side, quantity, price, timing, or execution permission;
- translate `Action` into permission to trade;
- persist a role, target, snapshot, policy, or evaluation without authorization;
- reclassify backend status, priority, queue class, or suggestion.

Because these are discipline and reference skills, verification will use
pressure scenarios before and after each skill edit, followed by frontmatter
validation, cross-reference checks, and a focused diff review.

## Acceptance criteria

- The three judgment-facing skills use the same fact/input/assumption model.
- Responses may be explicit and directional without becoming transaction
  instructions.
- Incomplete data blocks unsupported facts, not all conditional analysis.
- New-instrument analysis is supported without contaminating current-holding
  calculations or persisted policy.
- All repository-level product guardrails remain unchanged.
- Pressure tests demonstrate both increased flexibility and preserved safety.
