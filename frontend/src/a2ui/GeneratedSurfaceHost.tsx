import { JournalDraftComposerSurface } from './renderers/JournalDraftComposerSurface';
import { ReviewQueueTriageSurface } from './renderers/ReviewQueueTriageSurface';
import type { AppSurfaceTarget } from './types';
import { useGenerativeUi } from './GenerativeUiContext';

export function GeneratedSurfaceHost({ target }: { target: AppSurfaceTarget }) {
  const { journalDraftSurface, reviewDecisions, reviewQueueSurface, updateReviewDecision } = useGenerativeUi();

  if (target === 'review_queue') {
    if (!reviewQueueSurface) return null;
    return (
      <ReviewQueueTriageSurface
        surface={reviewQueueSurface}
        decisions={reviewDecisions}
        onDecisionChange={updateReviewDecision}
      />
    );
  }

  if (!journalDraftSurface) return null;
  return <JournalDraftComposerSurface surface={journalDraftSurface} />;
}
