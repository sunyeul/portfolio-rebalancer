import { createContext, useCallback, useContext, useMemo, useReducer } from 'react';
import type { ReactNode } from 'react';

import type {
  AppSurfaceTarget,
  GenerativeSurfacePatch,
  JournalDraftComposerSurfacePayload,
  ReviewDecision,
  ReviewQueueTriageSurfacePayload
} from './types';

export type GenerativeUiState = {
  reviewQueueSurface: ReviewQueueTriageSurfacePayload | null;
  journalDraftSurface: JournalDraftComposerSurfacePayload | null;
  reviewDecisions: ReviewDecision[];
};

export type GenerativeUiAction =
  | { type: 'applySurfacePatch'; patch: GenerativeSurfacePatch }
  | { type: 'clearSurface'; target: AppSurfaceTarget }
  | { type: 'updateReviewDecision'; decision: ReviewDecision }
  | { type: 'setReviewDecisions'; decisions: ReviewDecision[] };

export const initialGenerativeUiState: GenerativeUiState = {
  reviewQueueSurface: null,
  journalDraftSurface: null,
  reviewDecisions: []
};

export function generativeUiReducer(
  state: GenerativeUiState,
  action: GenerativeUiAction
): GenerativeUiState {
  if (action.type === 'applySurfacePatch') {
    if (action.patch.target === 'review_queue') {
      return { ...state, reviewQueueSurface: action.patch.payload };
    }
    return { ...state, journalDraftSurface: action.patch.payload };
  }

  if (action.type === 'clearSurface') {
    if (action.target === 'review_queue') {
      return { ...state, reviewQueueSurface: null };
    }
    return { ...state, journalDraftSurface: null };
  }

  if (action.type === 'updateReviewDecision') {
    const withoutExisting = state.reviewDecisions.filter(
      (candidate) => candidate.review_item_id !== action.decision.review_item_id
    );
    return { ...state, reviewDecisions: [...withoutExisting, action.decision] };
  }

  return { ...state, reviewDecisions: action.decisions };
}

type GenerativeUiContextValue = GenerativeUiState & {
  applySurfacePatch: (patch: GenerativeSurfacePatch) => void;
  clearSurface: (target: AppSurfaceTarget) => void;
  updateReviewDecision: (decision: ReviewDecision) => void;
  setReviewDecisions: (decisions: ReviewDecision[]) => void;
};

const GenerativeUiContext = createContext<GenerativeUiContextValue | null>(null);

export function GenerativeUiProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(generativeUiReducer, initialGenerativeUiState);
  const applySurfacePatch = useCallback((patch: GenerativeSurfacePatch) => {
    dispatch({ type: 'applySurfacePatch', patch });
  }, []);
  const clearSurface = useCallback((target: AppSurfaceTarget) => {
    dispatch({ type: 'clearSurface', target });
  }, []);
  const updateReviewDecision = useCallback((decision: ReviewDecision) => {
    dispatch({ type: 'updateReviewDecision', decision });
  }, []);
  const setReviewDecisions = useCallback((decisions: ReviewDecision[]) => {
    dispatch({ type: 'setReviewDecisions', decisions });
  }, []);
  const value = useMemo<GenerativeUiContextValue>(
    () => ({
      ...state,
      applySurfacePatch,
      clearSurface,
      updateReviewDecision,
      setReviewDecisions
    }),
    [applySurfacePatch, clearSurface, setReviewDecisions, state, updateReviewDecision]
  );

  return <GenerativeUiContext.Provider value={value}>{children}</GenerativeUiContext.Provider>;
}

export function useGenerativeUi() {
  const value = useContext(GenerativeUiContext);
  if (!value) {
    throw new Error('useGenerativeUi must be used inside GenerativeUiProvider.');
  }
  return value;
}
