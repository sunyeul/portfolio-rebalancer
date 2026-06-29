import { z } from 'zod';

import {
  IPS_PILOT_REVIEW_CATALOG_ID,
  decisionContextValues,
  journalDraftSectionValues,
  journalEvidenceFieldValues,
  triageStatusValues
} from '../catalogs/ipsPilotReviewCatalog';
import { reviewDispositionSchema } from './reviewQueueTriage.schema';

export const reviewDecisionSchema = z.object({
  review_item_id: z.string().min(1),
  disposition: reviewDispositionSchema,
  user_note: z.string().optional()
});

export const evidenceLinkedDraftBlockSchema = z.object({
  id: z.string().min(1),
  section: z.enum(journalDraftSectionValues),
  title: z.string().min(1),
  draft_text: z.string().min(1),
  evidence: z
    .array(
      z.object({
        review_item_id: z.string().min(1),
        field: z.enum(journalEvidenceFieldValues),
        value: z.string().min(1)
      })
    )
    .min(1),
  editable: z.literal(true)
});

export const journalDraftComposerSurfaceSchema = z.object({
  component: z.literal('JournalDraftComposerSurface'),
  catalog_id: z.literal(IPS_PILOT_REVIEW_CATALOG_ID),
  title: z.string().min(1),
  decision_context: z.enum(decisionContextValues),
  included_items: z.array(
    z.object({
      review_item_id: z.string().min(1),
      name: z.string().min(1),
      status: z.enum(triageStatusValues),
      disposition: reviewDispositionSchema
    })
  ),
  draft_blocks: z.array(evidenceLinkedDraftBlockSchema).min(1),
  follow_up_checklist: z.array(
    z.object({
      id: z.string().min(1),
      text: z.string().min(1),
      source_review_item_ids: z.array(z.string().min(1)).min(1)
    })
  ),
  guardrail_notice: z.object({
    text: z.string().min(1)
  })
});
