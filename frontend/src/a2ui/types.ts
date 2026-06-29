import type { z } from 'zod';

import type { IPS_PILOT_REVIEW_CATALOG_ID } from './catalogs/ipsPilotReviewCatalog';
import type { reviewDecisionSchema } from './schemas/journalDraftComposer.schema';
import type { journalDraftComposerSurfaceSchema } from './schemas/journalDraftComposer.schema';
import type {
  reviewDispositionSchema,
  agentExplanationSourceSchema,
  reviewItemCardSchema,
  reviewQueueTriageSurfaceSchema
} from './schemas/reviewQueueTriage.schema';

export type AgentExplanationSource = z.infer<typeof agentExplanationSourceSchema>;
export type ReviewDisposition = z.infer<typeof reviewDispositionSchema>;
export type ReviewItemCardPayload = z.infer<typeof reviewItemCardSchema>;
export type ReviewQueueTriageSurfacePayload = z.infer<typeof reviewQueueTriageSurfaceSchema>;
export type ReviewDecision = z.infer<typeof reviewDecisionSchema>;
export type JournalDraftComposerSurfacePayload = z.infer<typeof journalDraftComposerSurfaceSchema>;

export type ReviewA2UISurfacePayload = ReviewQueueTriageSurfacePayload | JournalDraftComposerSurfacePayload;

export type AppSurfaceTarget = 'review_queue' | 'journal_draft';
export type AppSurfaceMode = 'preview' | 'replace' | 'append';
export type AppSurfaceComponent = ReviewA2UISurfacePayload['component'];

export type A2UiAppSurfaceEnvelope = {
  catalog_id: typeof IPS_PILOT_REVIEW_CATALOG_ID;
  target: AppSurfaceTarget;
  mode: AppSurfaceMode;
  surface: ReviewA2UISurfacePayload;
  source: {
    agent: 'reviewCopilot';
    created_at: string;
  };
};

export type GenerativeSurfacePatch =
  | {
      target: 'review_queue';
      surface: 'ReviewQueueTriageSurface';
      mode: AppSurfaceMode;
      payload: ReviewQueueTriageSurfacePayload;
    }
  | {
      target: 'journal_draft';
      surface: 'JournalDraftComposerSurface';
      mode: AppSurfaceMode;
      payload: JournalDraftComposerSurfacePayload;
    };

export type ReviewA2UIEnvelope =
  | {
      ok: true;
      surface: ReviewA2UISurfacePayload;
    }
  | {
      ok: false;
      fallback_text: string;
      validation_errors: string[];
    };
