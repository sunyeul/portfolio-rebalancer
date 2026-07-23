# Overview Priority Queue Removal Design

**Created:** 2026-07-24
**Status:** Approved for implementation

## Purpose

Remove the compact `우선 확인 항목` panel from Overview. It repeats much of the
same priority, status, suggestion, meaning, and verification information already
shown by `다음 조정 검토`, making the primary screen harder to scan.

## Decision

Overview keeps backend-owned `adjustment_suggestions` as its single actionable
inspection summary. It no longer renders `ReviewQueue` with the first three
`review_queue` entries.

The full `전체 확인 항목` tab remains unchanged. It continues to be the place for
the complete Review Queue, including non-allocation evidence such as source or
policy blockers, account-risk findings, and performance observations.

## Scope

- Modify `frontend/src/App.tsx` to remove only the compact Overview
  `ReviewQueue` invocation and its now-unnecessary wrapper.
- Adjust the affected Overview layout styling only if the removed wrapper leaves
  stale grid behavior.
- Preserve the existing `다음 조정 검토` blocked-state notice so an
  `allocation_state = not_evaluable` result remains visible on Overview.
- Preserve all API payloads, backend ordering, status vocabulary, and Review
  Queue data.

## Alternatives Considered

1. Keep both panels and make their scope labels more explicit. This retains
   useful non-allocation visibility, but does not resolve repeated allocation
   rows.
2. Merge both panels into a new Overview queue. This would blur allocation
   suggestions with diagnostic evidence and would require a new presentation
   contract.
3. Remove only the compact Overview queue while retaining the full Review tab.
   This is the selected smallest change: one clear summary on Overview and one
   detailed diagnostic destination elsewhere.

## Verification

Run `npm run typecheck` in `frontend`. Confirm manually that Overview has no
compact Review Queue while the `전체 확인 항목` tab still renders the full list.
