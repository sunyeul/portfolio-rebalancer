import { z } from 'zod';

export const layerValues = ['core', 'satellite', 'experiment'] as const;
export const categoryValues = [
  'core_market',
  'core_gold',
  'satellite_ai_infra',
  'satellite_ai_software',
  'satellite_space',
  'satellite_nextgen',
  'experiment_leverage',
  'experiment_momentum'
] as const;
export const thesisStatusValues = ['valid', 'watch', 'broken', 'unknown', 'intact'] as const;

export type LayerType = (typeof layerValues)[number];
export type CategoryType = (typeof categoryValues)[number];
export type ThesisStatusInput = (typeof thesisStatusValues)[number];

export const portfolioRowSchema = z.object({
  ticker: z.string().trim().toUpperCase(),
  allocation: z.union([z.number(), z.string()]).optional().nullable(),
  return_total: z.union([z.number(), z.string()]).optional().nullable(),
  layer: z.union([z.enum(layerValues), z.literal('')]).optional().nullable(),
  category: z.union([z.enum(categoryValues), z.literal('')]).optional().nullable(),
  dca_enabled: z.union([z.boolean(), z.string()]).optional().nullable(),
  thesis_status: z.union([z.enum(thesisStatusValues), z.literal('')]).optional().nullable()
});

export const settingsSchema = z.object({
  period: z.enum(['1M', '3M', '6M', 'YTD', '1Y', 'Max']),
  rfPct: z.coerce.number(),
  bench: z.string().trim().min(1)
});

export type PortfolioRowInput = z.infer<typeof portfolioRowSchema>;
export type SettingsValues = z.infer<typeof settingsSchema>;
