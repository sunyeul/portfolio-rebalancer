# ACE Shared Collaboration Lessons

This Git-tracked file contains small, reusable rules for future contributors. Keep entries concrete and scoped to an observable trigger. Use `Active` or `Superseded`; preserve stable IDs and lifecycle history.

## Entry schema (not a lesson)

Add a `## lesson-...` entry only after a reusable lesson has evidence. Each entry must contain these fields:

- lesson ID: stable `lesson-...` identifier that is never reused
- title: short rule title
- status: `Active` or `Superseded`
- scope: repository area or workflow
- trigger: observable condition that activates the rule
- rule: action a future contributor can execute
- evidence: test, review, or reproducible observation supporting the rule
- helpful: how applying the rule prevents recurrence
- harmful: cost or failure mode if the rule is applied too broadly

No shared lessons are recorded by the bootstrap itself.
