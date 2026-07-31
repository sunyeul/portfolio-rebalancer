# Balanced IPS judgment axes

## Goal

Keep the IPS inspection strict about the currently active policy while making
evidence-backed policy-change candidates visible. A market or technical signal
must not silently reclassify a current portfolio evaluation or authorize a
trade.

## Decision model

The backend exposes two independent questions.

| Question | Inputs | Result | May change the current Review Queue? |
| --- | --- | --- | --- |
| Is the active policy being followed? | latest normalized Toss snapshot, active policy, allocation ranges, supported drawdown rules | primary inspection evaluation | Yes |
| Is there evidence to reconsider the policy? | complete market and technical evidence plus candidate-policy feasibility | policy-change candidate assessment | No |

The primary evaluation remains the only source for `OK`, `Watch`, `Review`,
and `Action` in the Review Queue. Candidate assessment is a separately named
result shown with Market Context and never replaces or mutates that evaluation.

## Fixed boundaries

- A snapshot is current only when it is complete, evaluable, and no more than
  72 hours old at evaluation time. Otherwise the allocation result is
  `not_evaluable`, with no priority and no adjustment suggestion.
- Current-policy compliance continues to use cash, layer, and instrument
  allocation ranges, plus the existing runtime-owned drawdown checks.
- Incomplete, stale, or ambiguous market/technical evidence yields no policy
  candidate. It is an observation/data-verification outcome, not a fallback
  allocation decision.
- Drawdown remains evidence for observation or risk review; it does not by
  itself create `Action` or a transaction instruction.
- `Action` remains reserved for an explicitly modeled exceptional-intervention
  rule. No automatic `Action` producer is added in this change.

The 72-hour boundary is deliberately a single backend rule, not a new
per-screen setting. It is long enough for ordinary non-trading days but short
enough to prevent an old complete snapshot from being presented as current.

## Candidate-policy visibility

Market Context displays the active-policy values beside the candidate-policy
values for each affected cash, layer, or instrument target/range. It also
displays the candidate evidence state and feasibility result.

Only a candidate with complete evidence and a feasible allocation is eligible
for explicit human approval. Approval remains a separate existing policy
activation flow; the candidate view does not persist, activate, or overwrite a
policy as part of inspection.

This preserves flexibility: supported evidence can name a precise future
regular-purchase allocation direction, while the active policy continues to
govern the account until a human changes it.

## Shared backend source of truth

Move currentness calculation into one backend helper used by API and CLI.
It receives the selected snapshot, active policy, and evaluation timestamp;
it returns structured reasons including snapshot age. API and CLI only render
that result and do not classify currentness independently.

The helper must retain the existing latest-snapshot selection rule: newest
`synced_at`, then largest `id` for a missing or tied timestamp.

## Status and queue contract

- Do not add status values or action/order fields.
- `OK`, `Watch`, `Review`, and `Action` remain backend-owned classifications.
- A stale source is blocking (`allocation_state=not_evaluable`), so it cannot
  produce adjustment suggestions from stale weights.
- Candidate-policy differences are informational to a human policy review;
  they do not add an item to the primary Review Queue merely because a target
  differs.

## Error handling and tests

Tests cover a fresh 72-hour boundary, an expired snapshot, and an incomplete
source. API and CLI tests assert identical currentness payloads from the shared
helper. Market Context tests assert that candidate differences are visible but
do not change the current inspection status, queue class, priority, or
suggestion. Existing drawdown tests continue to prove that drawdown alone does
not produce `Action`.

No schema migration, snapshot mutation, configuration write, new dependency,
or frontend status-classification logic is required.
