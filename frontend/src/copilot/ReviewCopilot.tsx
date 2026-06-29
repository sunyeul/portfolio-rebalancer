import { CopilotKit, CopilotSidebar, useAgentContext, useFrontendTool } from '@copilotkit/react-core/v2';
import type { JsonSerializable } from '@copilotkit/react-core/v2';
import '@copilotkit/react-core/v2/styles.css';
import type { ReactNode } from 'react';
import { useMemo } from 'react';
import { z } from 'zod';

import { IPS_PILOT_REVIEW_CATALOG_ID } from '../a2ui/catalogs/ipsPilotReviewCatalog';
import { useGenerativeUi } from '../a2ui/GenerativeUiContext';
import { reviewDecisionSchema } from '../a2ui/schemas/journalDraftComposer.schema';
import type {
  A2UiAppSurfaceEnvelope,
  AppSurfaceTarget,
  ReviewA2UISurfacePayload,
  ReviewDecision
} from '../a2ui/types';
import {
  buildJournalDraftComposerSurface,
  buildReviewQueueTriageSurface,
  defaultReviewDecisionsFromEvaluation,
  mergeReviewQueueAgentExplanations,
  reviewItemId
} from '../a2ui/utils/builders';
import { applyA2UiAppSurfacePayload } from '../a2ui/utils/validation';
import {
  type AnalysisResponse,
  type AssetRow,
  type EvaluationResponse,
  runAnalysis,
  runEvaluation
} from '../lib/api';
import type { LayerType } from '../lib/schemas';

const REVIEW_COPILOT_LABELS = {
  modalHeaderTitle: 'Review Copilot',
  welcomeMessageText: '평가 결과나 Review Queue에 대해 물어보세요.',
  chatInputPlaceholder: '평가 결과를 점검 관점으로 질문하세요...',
  chatDisclaimerText: 'Inspection only. 매수/매도 지시나 주문 수량은 제공하지 않습니다.',
  chatToggleOpenLabel: 'Review Copilot 열기',
  chatToggleCloseLabel: 'Review Copilot 닫기'
};

const runEvaluationInputSchema = z.object({
  confirmed_current_settings: z.boolean(),
  period: z.enum(['1M', '3M', '6M', 'YTD', '1Y', 'Max']).optional(),
  as_of_date: z.string().optional(),
  bench: z.string().optional(),
  layer_benchmarks: z.record(z.string(), z.string()).optional()
});

const draftJournalInputSchema = z.object({
  decision_context: z.enum(['regular_review', 'market_correction', 'sharp_drop_review', 'rebalance_review']),
  selected_review_items: z.array(z.string()).optional(),
  tone: z.enum(['brief', 'detailed']).optional()
});

const agentExplanationInputSchema = z.object({
  review_item_id: z.string(),
  text: z.string()
});

const createReviewQueueTriageInputSchema = z.object({
  agent_overview: z.string().optional(),
  agent_explanations: z.array(agentExplanationInputSchema).optional()
});

const createJournalDraftComposerInputSchema = z.object({
  decision_context: z.enum(['regular_review', 'market_correction', 'sharp_drop_review', 'rebalance_review']),
  review_decisions: z.array(reviewDecisionSchema).optional(),
  include_observe_and_deferred: z.boolean().optional()
});

export type ReviewCopilotSettings = {
  period: '1M' | '3M' | '6M' | 'YTD' | '1Y' | 'Max';
  asOfDate: string;
  layerBenchmarks: Record<LayerType, string>;
  analysisBenchmark: string;
};

export type ReviewCopilotHostProps = {
  children: ReactNode;
};

export type ReviewCopilotProps = {
  portfolio: AssetRow[];
  evaluation: EvaluationResponse | null;
  settings: ReviewCopilotSettings;
  onAnalysis: (analysis: AnalysisResponse) => void;
  onEvaluation: (evaluation: EvaluationResponse) => void;
};

function analysisPeriodFromEvaluationPeriod(period: ReviewCopilotSettings['period']) {
  if (period === '1Y') return 12;
  if (period === 'YTD' || period === 'Max') return period;
  return Number(period.replace('M', ''));
}

function appSurfaceEnvelope(target: AppSurfaceTarget, surface: ReviewA2UISurfacePayload): A2UiAppSurfaceEnvelope {
  return {
    catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
    target,
    mode: 'replace',
    surface,
    source: {
      agent: 'reviewCopilot',
      created_at: new Date().toISOString()
    }
  };
}

function surfaceApplicationResult(
  message: string,
  validation: ReturnType<typeof applyA2UiAppSurfacePayload>
): string {
  if (!validation.ok) {
    return validation.fallback_text;
  }

  return message;
}

function ToolStatusMessage({
  pendingLabel,
  result,
  status
}: {
  pendingLabel: string;
  result: unknown;
  status: string;
}) {
  if (status !== 'complete') {
    return (
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm font-semibold text-slate-600">
        {pendingLabel}
      </div>
    );
  }

  if (typeof result !== 'string') {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm font-semibold text-amber-800">
        Generated UI를 앱 본문에 반영하지 못했습니다. 대신 텍스트 요약으로 표시합니다.
      </div>
    );
  }

  if (result.includes('반영하지 못했습니다')) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm font-semibold text-amber-800">
        {result}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-cyan-200 bg-cyan-50 p-3 text-sm font-semibold text-cyan-900">
      {result}
    </div>
  );
}

function compactEvaluationContext(
  portfolio: AssetRow[],
  evaluation: EvaluationResponse | null,
  settings: ReviewCopilotSettings
) {
  return {
    portfolio,
    evaluation_period: evaluation?.evaluation_period ?? null,
    layer_evaluations: evaluation?.layer_evaluations ?? [],
    asset_evaluations: evaluation?.asset_evaluations ?? [],
    review_queue: evaluation?.review_queue ?? [],
    warnings: evaluation?.warnings ?? [],
    guardrails:
      evaluation?.guardrails ?? {
        not_investment_advice: true,
        no_immediate_order_instruction: true
      },
    settings: {
      period: settings.period,
      as_of_date: settings.asOfDate,
      bench: settings.analysisBenchmark,
      layer_benchmarks: settings.layerBenchmarks
    }
  };
}

function draftJournalNote(
  context: ReturnType<typeof compactEvaluationContext>,
  input: z.infer<typeof draftJournalInputSchema>
) {
  const selected = new Set(input.selected_review_items ?? []);
  const reviewItems =
    selected.size === 0
      ? context.review_queue
      : context.review_queue.filter((item) => selected.has(item.name) || selected.has(reviewItemId(item)));
  const tone = input.tone ?? 'brief';
  const scopeLine =
    reviewItems.length === 0
      ? '이번 평가에서 별도 Review Queue 항목은 확인되지 않았다.'
      : `이번 평가는 ${reviewItems.map((item) => `${item.name}(${item.status})`).join(', ')}를 중심으로 점검한다.`;
  const checks = reviewItems.map((item) => {
    const layer = item.parent_layer ? ` / ${item.parent_layer}` : '';
    return `- ${item.name}${layer}: ${item.suggested_next_step} Trigger: ${item.triggered_by.join(', ') || 'none'}.`;
  });
  const noteLines =
    tone === 'detailed'
      ? [
          `[${input.decision_context}] ${context.evaluation_period?.label ?? 'current'} 평가 점검 메모`,
          scopeLine,
          '즉시 매매 판단이 아니라 IPS 적합성, 논리 상태, 위험 기여, 계층 내 부담을 확인하는 리뷰로 기록한다.',
          ...checks,
          '다음 단계는 필요한 데이터와 논리 상태를 확인한 뒤 향후 정기매수 정책 또는 관찰 여부를 검토하는 것이다.'
        ]
      : [
          `[${input.decision_context}] ${context.evaluation_period?.label ?? 'current'} 리뷰: ${scopeLine}`,
          '매수/매도 지시가 아니라 점검 기록으로 남긴다.',
          ...checks.slice(0, 4)
        ];

  return {
    decision_context: input.decision_context,
    decision_note: noteLines.join('\n'),
    included_items: reviewItems.map(reviewItemId)
  };
}

export function ReviewCopilotHost({ children }: ReviewCopilotHostProps) {
  return (
    <CopilotKit
      runtimeUrl="/copilotkit"
      useSingleEndpoint={false}
      onError={({ error }) => {
        console.error('[review-copilot]', error);
      }}
    >
      {children}
    </CopilotKit>
  );
}

export function ReviewCopilot({ portfolio, evaluation, settings, onAnalysis, onEvaluation }: ReviewCopilotProps) {
  const { applySurfacePatch, reviewDecisions, reviewQueueSurface, setReviewDecisions } = useGenerativeUi();
  const context = useMemo(
    () => compactEvaluationContext(portfolio, evaluation, settings),
    [portfolio, evaluation, settings]
  );
  const agentContext = useMemo(() => ({ ...context, review_decisions: reviewDecisions }), [context, reviewDecisions]);

  useAgentContext({
    description:
      'Current IPS Pilot workbench state. Treat statuses as inspection labels only and never as order instructions.',
    value: agentContext as unknown as JsonSerializable
  });

  useFrontendTool(
    {
      name: 'getEvaluationContext',
      description:
        'Read the current visible IPS Pilot evaluation context, including portfolio, layer evaluations, asset evaluations, Review Queue, review decisions, warnings, guardrails, and settings.',
      parameters: z.object({}),
      handler: async () => agentContext
    },
    [agentContext]
  );

  useFrontendTool(
    {
      name: 'createReviewQueueTriageSurface',
      description:
        'Create and apply an ips-pilot-review/v1 ReviewQueueTriageSurface to the Review Queue app surface. Use this for requests to organize, triage, or prioritize Review Queue items.',
      parameters: createReviewQueueTriageInputSchema,
      followUp: false,
      handler: async ({ agent_overview, agent_explanations }) => {
        const baseSurface = reviewQueueSurface ?? buildReviewQueueTriageSurface(evaluation);
        const surface =
          agent_overview || (agent_explanations && agent_explanations.length > 0)
            ? mergeReviewQueueAgentExplanations(baseSurface, {
                source: 'requested',
                created_at: new Date().toISOString(),
                overview: agent_overview,
                explanations: agent_explanations
              })
            : baseSurface;
        const validation = applyA2UiAppSurfacePayload(appSurfaceEnvelope('review_queue', surface), applySurfacePatch);
        if (!validation.ok) {
          console.error('[review-copilot:a2ui]', validation.validation_errors);
          return surfaceApplicationResult('', validation);
        }
        setReviewDecisions(defaultReviewDecisionsFromEvaluation(evaluation));
        return surfaceApplicationResult(
          'Review Queue에 generated triage board를 생성했습니다. 앱 본문에서 각 항목의 처리 방침을 선택하세요.',
          validation
        );
      },
      render: ({ status, result }) => {
        return <ToolStatusMessage pendingLabel="Review Queue triage surface 준비 중..." result={result} status={status} />;
      }
    },
    [applySurfacePatch, evaluation, reviewQueueSurface, setReviewDecisions]
  );

  useFrontendTool(
    {
      name: 'createJournalDraftComposerSurface',
      description:
        'Create and apply an ips-pilot-review/v1 JournalDraftComposerSurface to the Journal Draft app surface from selected Review Queue dispositions. This creates editable, copyable journal draft blocks only and does not save journal state.',
      parameters: createJournalDraftComposerInputSchema,
      followUp: false,
      handler: async ({ decision_context, review_decisions, include_observe_and_deferred }) => {
        const decisions = review_decisions ?? reviewDecisions;
        const surface = buildJournalDraftComposerSurface({
          evaluation,
          decisions,
          decisionContext: decision_context,
          includeObserveAndDeferred: include_observe_and_deferred
        });
        const validation = applyA2UiAppSurfacePayload(appSurfaceEnvelope('journal_draft', surface), applySurfacePatch);
        if (!validation.ok) {
          console.error('[review-copilot:a2ui]', validation.validation_errors);
        }
        return surfaceApplicationResult(
          '선택된 review decision을 기반으로 Journal Draft를 생성했습니다. Journal Draft 영역에서 초안을 확인하세요.',
          validation
        );
      },
      render: ({ status, result }) => {
        return <ToolStatusMessage pendingLabel="Journal draft composer 준비 중..." result={result} status={status} />;
      }
    },
    [applySurfacePatch, evaluation, reviewDecisions]
  );

  useFrontendTool(
    {
      name: 'runEvaluation',
      description:
        'Rerun IPS Pilot analysis and evaluation only after the user explicitly confirms the current settings. This produces inspection outputs only.',
      parameters: runEvaluationInputSchema,
      handler: async ({ confirmed_current_settings, period, as_of_date, bench, layer_benchmarks }, { signal }) => {
        if (!confirmed_current_settings) {
          return {
            ok: false,
            message:
              'Evaluation was not run. Ask the user to confirm the period, as-of date, benchmark, and layer benchmarks first.'
          };
        }

        const nextPeriod = period ?? settings.period;
        const nextAsOfDate = as_of_date ?? settings.asOfDate;
        const nextLayerBenchmarks = layer_benchmarks ?? settings.layerBenchmarks;
        const nextBench = bench ?? settings.analysisBenchmark;
        const analysis = await runAnalysis({
          period: analysisPeriodFromEvaluationPeriod(nextPeriod),
          as_of_date: nextAsOfDate,
          rf: 0.025,
          bench: nextBench,
          layer_benchmarks: nextLayerBenchmarks
        }, signal);
        onAnalysis(analysis);

        const nextEvaluation = await runEvaluation({
          period: nextPeriod,
          as_of_date: nextAsOfDate,
          bench: nextBench,
          layer_benchmarks: nextLayerBenchmarks
        }, signal);
        onEvaluation(nextEvaluation);

        return {
          ok: true,
          evaluation_period: nextEvaluation.evaluation_period,
          review_queue_count: nextEvaluation.review_queue.length,
          guardrails: nextEvaluation.guardrails
        };
      }
    },
    [settings, onAnalysis, onEvaluation]
  );

  useFrontendTool(
    {
      name: 'draftJournalNote',
      description:
        'Create a copyable journal note draft from the visible Review Queue. This does not save or mutate journal state.',
      parameters: draftJournalInputSchema,
      handler: async (input) => draftJournalNote(context, input)
    },
    [context]
  );

  return (
    <CopilotSidebar
      header={{ closeButton: { 'aria-label': 'Review Copilot 접기' } }}
      labels={REVIEW_COPILOT_LABELS}
      defaultOpen={true}
      position="right"
      width={420}
    />
  );
}
