# Policy Preflight and Change Brief Design

## Goal

Add two read-only decision-support surfaces:

- questions a person must answer before preparing a policy change;
- a change-only brief comparing the newest evaluation with the previous one.

## Policy preflight

`services.policy_preflight` produces a structured question list from the
active policy and latest persisted evaluation. It asks for the intended scope,
objective, duration, supporting evidence, and reversal condition. If the
latest evaluation cannot support an allocation judgment, it adds source
verification as the first question.

The API and dashboard present these questions with browser-local answer fields.
Answers are not sent to the server, persisted, applied to policy, or used to
reclassify an evaluation. This is a deterministic first step rather than an
LLM integration.

## Change brief

`services.change_brief` compares the two most recent immutable evaluations.
It returns only:

- new, changed, or resolved Review Queue items;
- a current source alert when the newest evaluation source is not complete or
  its allocation state is not evaluable;
- current and previous run/snapshot identifiers.

Its queue comparison uses kind and identity as the stable item key. It compares
status, priority, queue class, suggestion code, and trigger set without
deriving a replacement IPS result. The brief is available from a read-only API
endpoint, the `inspection brief` JSON CLI command, and a dashboard tab. It
does not schedule, send, or recommend a transaction.

## Non-goals

- LLM provider calls, credentials, prompt storage, chat history, or model-made
  account facts.
- Email, Kakao, push notifications, a scheduler, or automatic sending.
- Persisting interview answers or changing policy, snapshot, evaluation, status,
  priority, suggestion, queue order, order, quantity, price, or execution.
- Treating external research as a Toss account fact.

## Verification

- Service tests cover source-blocked preflight, stable baseline brief, queue
  additions/changes/resolutions, and source alerts.
- API tests verify both responses remain read-only and preserve source run
  identity.
- CLI tests verify one JSON object is emitted for the change brief.
- Frontend typechecking, tests, and production build verify local-only display.
