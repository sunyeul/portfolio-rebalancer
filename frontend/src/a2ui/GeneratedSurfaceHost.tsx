import { JournalDraftComposerSurface } from './renderers/JournalDraftComposerSurface';
import { EvaluationGraphSurface } from './renderers/EvaluationGraphSurface';
import { ReviewQueueTriageSurface } from './renderers/ReviewQueueTriageSurface';
import type { AppSurfaceTarget } from './types';
import { useGenerativeUi } from './GenerativeUiContext';
import type { EvaluationResponse } from '../lib/api';

export function GeneratedSurfaceHost({
  evaluation,
  focusedLayer,
  focusedTicker,
  onLayerFocus,
  onTickerFocus,
  onTickerSelect,
  target
}: {
  evaluation?: EvaluationResponse | null;
  focusedLayer?: string | null;
  focusedTicker?: string | null;
  onLayerFocus?: (layer: string | null) => void;
  onTickerFocus?: (ticker: string | null) => void;
  onTickerSelect?: (ticker: string) => void;
  target: AppSurfaceTarget;
}) {
  const {
    evaluationGraphSurface,
    journalDraftSurface,
    reviewDecisions,
    reviewQueueSurface,
    updateReviewDecision
  } = useGenerativeUi();

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

  if (target === 'evaluation_graphs') {
    if (!evaluationGraphSurface) return null;
    return (
      <EvaluationGraphSurface
        evaluation={evaluation ?? null}
        focusedLayer={focusedLayer ?? null}
        focusedTicker={focusedTicker ?? null}
        onLayerFocus={onLayerFocus}
        onTickerFocus={onTickerFocus}
        onTickerSelect={onTickerSelect}
        surface={evaluationGraphSurface}
      />
    );
  }

  if (!journalDraftSurface) return null;
  return <JournalDraftComposerSurface surface={journalDraftSurface} />;
}
