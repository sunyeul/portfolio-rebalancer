import { z } from 'zod';

export const layerValues = ['core', 'satellite', 'experiment'] as const;
export const thesisStatusValues = ['valid', 'watch', 'broken', 'unknown'] as const;

export type LayerType = (typeof layerValues)[number];
export type ThesisStatusInput = (typeof thesisStatusValues)[number];

export const portfolioRowSchema = z.object({
  ticker: z.string().trim().toUpperCase(),
  allocation: z.union([z.number(), z.string()]).optional().nullable(),
  return_total: z.union([z.number(), z.string()]).optional().nullable(),
  layer: z.union([z.enum(layerValues), z.literal('')]).optional().nullable(),
  thesis_status: z.union([z.enum(thesisStatusValues), z.literal('')]).optional().nullable()
});

export const settingsSchema = z.object({
  period: z.enum(['1M', '3M', '6M', 'YTD', '1Y', 'Max']),
  bench: z.string().trim().min(1)
});

export type PortfolioRowInput = z.infer<typeof portfolioRowSchema>;
export type SettingsValues = z.infer<typeof settingsSchema>;
