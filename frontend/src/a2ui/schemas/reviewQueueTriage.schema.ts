import { z } from 'zod';

import {
  IPS_PILOT_REVIEW_CATALOG_ID,
  reviewDispositionValues,
  triageStatusValues
} from '../catalogs/ipsPilotReviewCatalog';

export const reviewDispositionSchema = z.enum(reviewDispositionValues);
export const agentExplanationSourceSchema = z.enum(['automatic', 'requested']);

export const agentExplanationSchema = z.object({
  source: agentExplanationSourceSchema,
  text: z.string().min(1),
  created_at: z.string().min(1)
});

export const reviewItemCardSchema = z.object({
  id: z.string().min(1),
  level: z.enum(['layer', 'asset']),
  name: z.string().min(1),
  parent_layer: z.string().nullable(),
  status: z.enum(triageStatusValues),
  status_label_ko: z.string().min(1),
  triggered_by: z.array(z.string()),
  trigger_explanations: z
    .array(
      z.object({
        code: z.string().min(1),
        explanation: z.string().min(1)
      })
    )
    .min(1),
  agent_summary: z.string().min(1),
  ips_interpretation: z.string().min(1),
  verification_focus: z.string().min(1),
  agent_explanations: z.array(agentExplanationSchema).optional(),
  review_questions: z.array(z.string().min(1)).min(1),
  suggested_next_step: z.string().min(1),
  next_review_note: z.string().min(1),
  allowed_dispositions: z.array(reviewDispositionSchema).min(1)
});

export const reviewQueueTriageSurfaceSchema = z.object({
  component: z.literal('ReviewQueueTriageSurface'),
  catalog_id: z.literal(IPS_PILOT_REVIEW_CATALOG_ID),
  title: z.string().min(1),
  evaluation_period: z.object({
    label: z.enum(['1M', '3M', '6M', 'YTD', '1Y', 'Max', 'custom']),
    start_date: z.string().min(1),
    end_date: z.string().min(1)
  }),
  guardrail_notice: z.object({
    text: z.string().min(1)
  }),
  agent_overview: agentExplanationSchema.optional(),
  groups: z.array(
    z.object({
      status: z.enum(triageStatusValues),
      summary: z.string().min(1),
      items: z.array(reviewItemCardSchema)
    })
  )
});
