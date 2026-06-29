export const IPS_PILOT_REVIEW_CATALOG_ID = 'ips-pilot-review/v1' as const;

export const appSurfaceTargetValues = ['review_queue', 'journal_draft'] as const;

export const appSurfaceModes = ['preview', 'replace', 'append'] as const;

export const allowedReviewSurfaces = [
  'ReviewQueueTriageSurface',
  'JournalDraftComposerSurface',
  'MetricExplanationPopover',
  'GuardrailNotice'
] as const;

export const allowedReviewComponents = [
  'ReviewQueueTriageSurface',
  'ReviewItemCard',
  'TriggerExplanationList',
  'ReviewQuestionList',
  'DispositionSelector',
  'JournalDraftComposerSurface',
  'EvidenceLinkedDraftBlock',
  'FollowUpChecklist',
  'GuardrailNotice'
] as const;

export const reviewDispositionValues = [
  'include_in_journal',
  'observe',
  'review_thesis',
  'defer_until_next_review'
] as const;

export const forbiddenReviewActionValues = [
  'buy',
  'sell',
  'increase_position',
  'decrease_position',
  'rebalance_now',
  'calculate_order_size',
  'place_order'
] as const;

export const triageStatusValues = ['Action', 'Review', 'Watch'] as const;

export const decisionContextValues = [
  'regular_review',
  'market_correction',
  'sharp_drop_review',
  'rebalance_review'
] as const;

export const journalDraftSectionValues = ['context', 'observation', 'interpretation', 'decision', 'follow_up'] as const;

export const journalEvidenceFieldValues = [
  'status',
  'triggered_by',
  'metrics_snapshot',
  'suggested_next_step',
  'thesis'
] as const;
