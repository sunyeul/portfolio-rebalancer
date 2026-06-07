import { z } from 'zod';

export const portfolioRowSchema = z.object({
  ticker: z.string().trim().toUpperCase(),
  allocation: z.union([z.number(), z.string()]).optional().nullable(),
  return_total: z.union([z.number(), z.string()]).optional().nullable(),
  group: z.string().optional().nullable(),
  role: z.string().optional().nullable(),
  dca_enabled: z.union([z.boolean(), z.string()]).optional().nullable(),
  thesis_status: z.string().optional().nullable()
});

export const settingsSchema = z.object({
  periodMode: z.enum(['months', 'YTD', 'Max']),
  months: z.coerce.number().int().min(1).max(120),
  rfPct: z.coerce.number(),
  bench: z.string().trim().min(1),
  momentumWeight: z.coerce.number().min(0).max(0.5),
  rcOverThreshPct: z.coerce.number().min(0),
  eThresh: z.coerce.number().min(0).max(1)
});

export type PortfolioRowInput = z.infer<typeof portfolioRowSchema>;
export type SettingsValues = z.infer<typeof settingsSchema>;

