# Review Queue Red-Team Design

## Goal

Give every persisted Review Queue item a concise counterargument and the
evidence still required, without changing its IPS status, priority,
suggestion, queue order, policy, or source data.

## Design

`services.inspection_engine` will add a read-only `red_team` object while it
projects evaluated items into the Review Queue. The object has two Korean text
fields:

- `counterargument`: why the observed trigger alone is insufficient to make a
  decision;
- `evidence_needed`: the existing verification task that must be completed.

The counterargument is deterministic and selected only from the queue item
kind and whether the item blocks evaluation. It does not call an LLM, read an
external source, or introduce account facts. This keeps the first version
auditable and usable when the application has no model-provider configuration.

The React Review Queue renders the object as a compact `반대 관점` block. It
does not derive a replacement status, priority, suggestion, or queue order.

## Cases

- Blocking source item: current allocation judgment is unavailable until the
  Toss source and policy coverage are verified.
- Cash, layer, or instrument item: an allocation-range signal alone does not
  establish a transaction or exceptional intervention.
- Performance or account-risk item: return, profit/loss, or drawdown alone
  does not establish an allocation change or exceptional intervention.
- Any other item: the observed signal remains a review item and needs the
  linked verification task before it can support a conclusion.

## Non-goals

- Adding an LLM provider, prompts, model keys, background jobs, notifications,
  policy interviews, or persistence.
- Replacing the Toss snapshot as the account-fact source.
- Adding buy, sell, price, quantity, timing, execution, or automatic-trading
  language.
- Reclassifying `OK`, `Watch`, `Review`, or `Action`, or changing priority,
  suggestion, queue class, or queue order.

## Verification

- Focused engine tests prove that blocking, allocation, and performance queue
  items receive the expected red-team text while their decision fields remain
  unchanged.
- API contract tests verify that persisted red-team text is returned without a
  frontend or API reclassification.
- Frontend TypeScript checking and production build verify the new typed,
  read-only presentation.
