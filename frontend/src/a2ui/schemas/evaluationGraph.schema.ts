import { z } from 'zod';

import {
  IPS_PILOT_REVIEW_CATALOG_ID,
  appSurfaceModes,
  assetEvaluationMetricValues,
  evaluationGraphChartTypeValues,
  evaluationGraphSourceValues,
  layerEvaluationMetricValues
} from '../catalogs/ipsPilotReviewCatalog';

const evaluationStatusSchema = z.enum(['OK', 'Watch', 'Review', 'Action']);
const thesisStatusSchema = z.enum(['valid', 'watch', 'broken', 'unknown']);
const layerTypeSchema = z.enum(['core', 'satellite', 'experiment']);

export const layerEvaluationMetricSchema = z.enum(layerEvaluationMetricValues);
export const assetEvaluationMetricSchema = z.enum(assetEvaluationMetricValues);
export const evaluationGraphSourceSchema = z.enum(evaluationGraphSourceValues);
export const evaluationGraphChartTypeSchema = z.enum(evaluationGraphChartTypeValues);

export const evaluationGraphFilterSchema = z
  .object({
    status: z.array(evaluationStatusSchema).min(1).optional(),
    layer: z.array(layerTypeSchema).min(1).optional(),
    thesis_status: z.array(thesisStatusSchema).min(1).optional()
  })
  .strict();

const layerWeightGapChartSchema = z
  .object({
    id: z.string().min(1),
    chart_type: z.literal('layer_weight_gap_bar'),
    title: z.string().min(1),
    description: z.string().min(1).optional(),
    source: z.literal('layer_evaluations'),
    filter: evaluationGraphFilterSchema.optional()
  })
  .strict();

const assetRiskScatterChartSchema = z
  .object({
    id: z.string().min(1),
    chart_type: z.literal('asset_risk_scatter'),
    title: z.string().min(1),
    description: z.string().min(1).optional(),
    source: z.literal('asset_evaluations'),
    filter: evaluationGraphFilterSchema.optional()
  })
  .strict();

const metricBarChartSchema = z
  .object({
    id: z.string().min(1),
    chart_type: z.literal('metric_bar'),
    title: z.string().min(1),
    description: z.string().min(1).optional(),
    source: evaluationGraphSourceSchema,
    metric: z.union([layerEvaluationMetricSchema, assetEvaluationMetricSchema]),
    filter: evaluationGraphFilterSchema.optional(),
    sort: z
      .object({
        by: z.union([layerEvaluationMetricSchema, assetEvaluationMetricSchema]),
        direction: z.enum(['asc', 'desc'])
      })
      .strict()
      .optional(),
    limit: z.number().int().min(1).max(30).optional()
  })
  .strict()
  .superRefine((chart, context) => {
    const allowedMetrics: readonly string[] =
      chart.source === 'layer_evaluations' ? layerEvaluationMetricValues : assetEvaluationMetricValues;
    if (!allowedMetrics.includes(chart.metric)) {
      context.addIssue({
        code: 'custom',
        path: ['metric'],
        message: `${chart.metric} is not valid for ${chart.source}`
      });
    }
    if (chart.sort && !allowedMetrics.includes(chart.sort.by)) {
      context.addIssue({
        code: 'custom',
        path: ['sort', 'by'],
        message: `${chart.sort.by} is not valid for ${chart.source}`
      });
    }
  });

export const evaluationGraphChartSchema = z.union([
  layerWeightGapChartSchema,
  assetRiskScatterChartSchema,
  metricBarChartSchema
]);

export const evaluationGraphSurfaceSchema = z
  .object({
    component: z.literal('EvaluationGraphSurface'),
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
    mode_hint: z.enum(appSurfaceModes).optional(),
    charts: z.array(evaluationGraphChartSchema).min(1).max(8)
  })
  .strict();
