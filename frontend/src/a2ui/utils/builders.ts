import type { EvaluationResponse, ReviewItem } from '../../lib/api';
import { IPS_PILOT_REVIEW_CATALOG_ID, reviewDispositionValues } from '../catalogs/ipsPilotReviewCatalog';
import type {
  AgentExplanationSource,
  JournalDraftComposerSurfacePayload,
  ReviewDecision,
  ReviewDisposition,
  ReviewItemCardPayload,
  ReviewQueueTriageSurfacePayload
} from '../types';

const GUARDRAIL_NOTICE =
  '이 화면은 IPS 점검과 저널 기록을 위한 구조화 UI이며, 매수/매도 지시나 주문 수량 산정이 아닙니다.';

const statusOrder = ['Action', 'Review', 'Watch'] as const;

export type ReviewQueueAgentExplanationInput = {
  review_item_id: string;
  text: string;
};

export type ReviewQueueAgentExplanationPatch = {
  source: AgentExplanationSource;
  created_at: string;
  overview?: string | null;
  explanations?: ReviewQueueAgentExplanationInput[];
};

const statusLabelsKo: Record<ReviewItem['status'], string> = {
  Action: '조치 검토',
  Review: '중점 점검',
  Watch: '관찰 점검'
};

const triggerExplanationKo: Record<string, string> = {
  risk_contribution: '포트폴리오 위험 기여도가 커서, 이 노출이 의도된 부담인지 확인해야 합니다.',
  risk_contribution_high: '포트폴리오 위험 기여도가 커서, 이 노출이 의도된 부담인지 확인해야 합니다.',
  target_gap_outside_tolerance: '현재 비중이 IPS의 목표 범위에서 벗어났는지 확인해야 합니다.',
  thesis_watch: '기록된 thesis가 관찰 또는 재검토 상태입니다.',
  thesis_broken: '기록된 thesis가 훼손됐을 가능성이 있어 근거를 다시 확인해야 합니다.',
  volatility_exceeded: '변동성이 허용 기준보다 커졌을 수 있습니다.',
  mdd_exceeded: '낙폭이 점검 기준을 넘었을 수 있습니다.',
  efficiency_below_threshold: '성과 대비 위험 효율이 낮아졌을 수 있습니다.',
  high_burden: '관리 부담이나 계층 내 부담이 커졌을 수 있습니다.',
  max_weight_exceeded: '단일 항목 또는 계층 비중이 상한을 넘었는지 확인해야 합니다.'
};

export function reviewItemId(item: ReviewItem) {
  return `${item.level}:${item.parent_layer ?? 'portfolio'}:${item.name}`;
}

function particle(value: string) {
  const last = value.charCodeAt(value.length - 1);
  if (last < 0xac00 || last > 0xd7a3) return '은';
  return (last - 0xac00) % 28 === 0 ? '는' : '은';
}

function layerLabelKo(item: ReviewItem) {
  if (item.level === 'layer') return `${item.name} 계층`;
  if (item.parent_layer) return `${item.parent_layer} 계층의 ${item.name}`;
  return `${item.name} 항목`;
}

function parentLayer(item: ReviewItem) {
  if (item.level === 'layer') return item.name;
  return item.parent_layer ?? 'unassigned';
}

function statusExplanation(item: ReviewItem) {
  if (item.status === 'Action') {
    return '조치 검토 상태이지만 매매 허가가 아니라 예외적 개입 가능성을 사람 검토로 확인하라는 뜻입니다.';
  }
  if (item.status === 'Review') {
    return '중점 점검 상태로, thesis와 위험 부담, 데이터 근거를 확인해야 합니다.';
  }
  return '관찰 점검 상태로, 다음 정기 리뷰까지 같은 신호가 이어지는지 확인하는 항목입니다.';
}

function triggerExplanations(item: ReviewItem) {
  if (item.triggered_by.length === 0) {
    return [
      {
        code: 'none',
        explanation: '명시적인 trigger code는 없지만 Review Queue에 포함되어 점검 대상 여부를 확인해야 합니다.'
      }
    ];
  }
  return item.triggered_by.map((code) => ({
    code,
    explanation:
      triggerExplanationKo[code] ??
      `기록된 trigger code를 기준으로 ${layerLabelKo(item)}의 IPS 적합성과 데이터 근거를 확인해야 합니다.`
  }));
}

function safeSummary(item: ReviewItem) {
  const subject = `${item.name}${particle(item.name)}`;
  const triggers =
    item.triggered_by.length > 0
      ? `${item.triggered_by.length}개의 점검 신호가 확인되어 Review Queue에 올라왔습니다.`
      : '명시적인 trigger code는 없지만 Review Queue 점검 대상으로 표시됐습니다.';
  return `${subject} ${item.status}(${statusLabelsKo[item.status]}) 상태입니다. ${triggers} 이는 즉시 행동 지시가 아니라 IPS 기준과 데이터 근거를 확인하기 위한 항목입니다.`;
}

function ipsInterpretation(item: ReviewItem) {
  const layer = parentLayer(item);
  const subject = layerLabelKo(item);
  if (layer === 'core') {
    return `${subject}은 core 노출로서 정상적인 시장 변동과 구조적 thesis 문제를 구분해 봐야 합니다. ${statusExplanation(item)}`;
  }
  if (layer === 'satellite') {
    return `${subject}은 satellite 노출이므로 thesis 지속성, core와의 overlap, 위험 부담, ETF 대체 가능성을 더 엄격히 확인해야 합니다. ${statusExplanation(item)}`;
  }
  if (layer === 'experiment') {
    return `${subject}은 experiment 노출이므로 관리 부담, 변동성, 보유 가능성을 보수적으로 확인해야 합니다. ${statusExplanation(item)}`;
  }
  return `${subject}은 계층 정보가 불명확하므로 먼저 분류와 데이터 근거를 확인한 뒤 IPS 적합성을 점검해야 합니다. ${statusExplanation(item)}`;
}

function verificationFocus(item: ReviewItem) {
  const layer = parentLayer(item);
  if (layer === 'core') {
    return '정상적인 core 변동인지, 장기 참여 논리를 훼손한 구조적 변화인지 구분할 근거를 확인합니다.';
  }
  if (layer === 'satellite') {
    return 'thesis, overlap, risk burden, ETF substitution 검토가 필요한지 확인합니다.';
  }
  if (layer === 'experiment') {
    return '관리 부담, 변동성, 보유 가능성이 다음 정기 리뷰까지 감당 가능한지 확인합니다.';
  }
  return '계층 분류, trigger 산출 근거, 데이터 최신성을 먼저 확인합니다.';
}

function reviewQuestions(item: ReviewItem) {
  const questions = [
    `${layerLabelKo(item)}의 계층 내 역할이 아직 분명한가?`,
    '현재 trigger가 일시적 데이터 문제인지 구조적 변화인지 구분할 근거가 있는가?',
    '다음 정기 리뷰까지 관찰로 충분한가, thesis 재검토가 필요한가?'
  ];
  if (item.thesis) {
    questions.splice(1, 0, `기록된 thesis가 현재 ${item.status}(${statusLabelsKo[item.status]}) 상태와 여전히 맞는가?`);
  }
  const layer = parentLayer(item);
  if (layer === 'satellite') {
    questions.push('core 노출과 중복되거나 ETF로 단순화할 여지가 있는가?');
  }
  if (layer === 'experiment') {
    questions.push('관리 부담과 변동성을 감안해 계속 보유 가능성을 점검할 근거가 있는가?');
  }
  return questions;
}

function nextReviewNote(item: ReviewItem) {
  const layer = parentLayer(item);
  if (item.status === 'Action') {
    return `${item.name}은 예외적 개입 가능성까지 점검할 항목으로 저널에 남기되, 먼저 데이터와 thesis 훼손 여부를 확인한다.`;
  }
  if (layer === 'core') {
    return `${item.name}은 core 역할이 유지되는지 관찰하고, 필요하면 다음 정기 리뷰에서 향후 정기매수 정책 점검 대상으로 기록한다.`;
  }
  if (layer === 'satellite') {
    return `${item.name}은 satellite thesis, overlap, 위험 부담, ETF 대체 가능성을 다음 리뷰 메모에 포함한다.`;
  }
  if (layer === 'experiment') {
    return `${item.name}은 관리 부담, 변동성, 보유 가능성을 별도 확인하고 관찰 유지 또는 thesis 재검토 여부를 기록한다.`;
  }
  return `${item.name}은 계층 분류와 데이터 근거를 확인한 뒤 관찰 유지 또는 thesis 재검토 여부를 기록한다.`;
}

function itemToCard(item: ReviewItem): ReviewItemCardPayload {
  return {
    id: reviewItemId(item),
    level: item.level,
    name: item.name,
    parent_layer: item.parent_layer,
    status: item.status,
    status_label_ko: statusLabelsKo[item.status],
    triggered_by: item.triggered_by,
    trigger_explanations: triggerExplanations(item),
    agent_summary: safeSummary(item),
    ips_interpretation: ipsInterpretation(item),
    verification_focus: verificationFocus(item),
    review_questions: reviewQuestions(item),
    suggested_next_step: nextReviewNote(item),
    next_review_note: nextReviewNote(item),
    allowed_dispositions: [...reviewDispositionValues]
  };
}

function groupSummary(status: ReviewItem['status'], count: number) {
  if (status === 'Action') {
    return `${count}개 항목은 조치 검토 대상입니다. 이는 매매 허가가 아니라 예외적 개입 가능성 점검입니다.`;
  }
  if (status === 'Review') {
    return `${count}개 항목은 thesis, risk 부담, overlap, 데이터 근거 확인이 필요합니다.`;
  }
  return `${count}개 항목은 다음 리뷰까지 관찰하며 근거가 더 분명해지는지 확인합니다.`;
}

export function buildReviewQueueTriageSurface(evaluation: EvaluationResponse | null): ReviewQueueTriageSurfacePayload {
  const period = evaluation?.evaluation_period ?? {
    label: 'custom' as const,
    start_date: 'N/A',
    end_date: 'N/A'
  };
  const queue = evaluation?.review_queue ?? [];

  return {
    component: 'ReviewQueueTriageSurface',
    catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
    title: 'Review Queue 점검 해석 보드',
    evaluation_period: period,
    guardrail_notice: { text: GUARDRAIL_NOTICE },
    groups: statusOrder.map((status) => {
      const items = queue.filter((item) => item.status === status).map(itemToCard);
      return {
        status,
        summary: groupSummary(status, items.length),
        items
      };
    })
  };
}

export function mergeReviewQueueAgentExplanations(
  surface: ReviewQueueTriageSurfacePayload,
  patch: ReviewQueueAgentExplanationPatch
): ReviewQueueTriageSurfacePayload {
  const normalizedExplanations = new Map(
    (patch.explanations ?? [])
      .map((explanation) => [explanation.review_item_id, explanation.text.trim()] as const)
      .filter(([, text]) => text.length > 0)
  );
  const nextOverview = patch.overview?.trim();

  return {
    ...surface,
    agent_overview: nextOverview
      ? {
          source: patch.source,
          text: nextOverview,
          created_at: patch.created_at
        }
      : surface.agent_overview,
    groups: surface.groups.map((group) => ({
      ...group,
      items: group.items.map((item) => {
        const text = normalizedExplanations.get(item.id);
        if (!text) return item;

        const previous = item.agent_explanations ?? [];
        const keptPrevious =
          patch.source === 'automatic'
            ? previous.filter((explanation) => explanation.source !== 'automatic')
            : previous;
        return {
          ...item,
          agent_explanations: [
            ...keptPrevious,
            {
              source: patch.source,
              text,
              created_at: patch.created_at
            }
          ]
        };
      })
    }))
  };
}

function findItem(evaluation: EvaluationResponse | null, id: string) {
  return evaluation?.review_queue.find((item) => reviewItemId(item) === id || item.name === id) ?? null;
}

export function defaultReviewDecisionsFromEvaluation(evaluation: EvaluationResponse | null): ReviewDecision[] {
  return (evaluation?.review_queue ?? []).map((item) => ({
    review_item_id: reviewItemId(item),
    disposition: item.status === 'Watch' ? 'observe' : 'include_in_journal'
  }));
}

function shouldInclude(disposition: ReviewDisposition, includeObserveAndDeferred: boolean) {
  if (disposition === 'include_in_journal' || disposition === 'review_thesis') return true;
  return includeObserveAndDeferred && (disposition === 'observe' || disposition === 'defer_until_next_review');
}

function stringifyEvidence(value: unknown) {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function dispositionSentence(decision: ReviewDecision, item: ReviewItem) {
  const note = decision.user_note ? ` 메모: ${decision.user_note}` : '';
  if (decision.disposition === 'include_in_journal') {
    return `${item.name}은 이번 저널에 점검 항목으로 포함한다.${note}`;
  }
  if (decision.disposition === 'review_thesis') {
    return `${item.name}은 thesis와 계층 내 역할을 재검토할 항목으로 기록한다.${note}`;
  }
  if (decision.disposition === 'observe') {
    return `${item.name}은 즉시 판단하지 않고 관찰을 유지할 항목으로 기록한다.${note}`;
  }
  return `${item.name}은 다음 정기 리뷰까지 결정을 보류할 항목으로 기록한다.${note}`;
}

function ensureEvidence(
  block: JournalDraftComposerSurfacePayload['draft_blocks'][number]
): JournalDraftComposerSurfacePayload['draft_blocks'][number] {
  if (block.evidence.length > 0) return block;
  return {
    ...block,
    evidence: [
      {
        review_item_id: 'review-queue',
        field: 'metrics_snapshot',
        value: 'No selected Review Queue item.'
      }
    ]
  };
}

export function buildJournalDraftComposerSurface({
  evaluation,
  decisions,
  decisionContext,
  includeObserveAndDeferred = false
}: {
  evaluation: EvaluationResponse | null;
  decisions?: ReviewDecision[];
  decisionContext: JournalDraftComposerSurfacePayload['decision_context'];
  includeObserveAndDeferred?: boolean;
}): JournalDraftComposerSurfacePayload {
  const allDecisions = decisions && decisions.length > 0 ? decisions : defaultReviewDecisionsFromEvaluation(evaluation);
  const includedDecisions = allDecisions.filter((decision) =>
    shouldInclude(decision.disposition, includeObserveAndDeferred)
  );
  const includedItems = includedDecisions
    .map((decision) => ({ decision, item: findItem(evaluation, decision.review_item_id) }))
    .filter((entry): entry is { decision: ReviewDecision; item: ReviewItem } => Boolean(entry.item));
  const period = evaluation?.evaluation_period;
  const primary = includedItems[0]?.item;
  const sourceIds = includedItems.map(({ item }) => reviewItemId(item));
  const dispositionLines = includedItems.map(({ decision, item }) => dispositionSentence(decision, item));

  const rawDraftBlocks: JournalDraftComposerSurfacePayload['draft_blocks'] = [
    {
      id: 'context',
      section: 'context',
      title: 'Context',
      draft_text: `이번 평가는 ${period?.label ?? 'current'} 기준 IPS 정기 점검이다.`,
      evidence: [
        {
          review_item_id: primary ? reviewItemId(primary) : 'evaluation-period',
          field: 'metrics_snapshot',
          value: `${period?.start_date ?? 'N/A'} ~ ${period?.end_date ?? 'N/A'}`
        }
      ],
      editable: true
    },
    {
      id: 'observation',
      section: 'observation',
      title: 'Observation',
      draft_text:
        includedItems.length > 0
          ? `${includedItems.map(({ item }) => `${item.name}(${item.status})`).join(', ')} 항목이 Review Queue에 올라왔다.`
          : '이번 평가에서 저널에 포함할 Review Queue 항목은 선택되지 않았다.',
      evidence: includedItems.slice(0, 4).map(({ item }) => ({
        review_item_id: reviewItemId(item),
        field: 'triggered_by',
        value: item.triggered_by.join(', ') || 'none'
      })),
      editable: true
    },
    {
      id: 'interpretation',
      section: 'interpretation',
      title: 'Interpretation',
      draft_text:
        '해당 항목은 즉시 매매 대상이 아니라 thesis, risk 부담, 계층 내 역할, 데이터 상태를 재확인할 검토 대상이다.',
      evidence: includedItems.slice(0, 4).map(({ item }) => ({
        review_item_id: reviewItemId(item),
        field: 'status',
        value: item.status
      })),
      editable: true
    },
    {
      id: 'decision',
      section: 'decision',
      title: 'Decision',
      draft_text:
        dispositionLines.length > 0
          ? [
              '이번 기록은 주문 지시가 아니라 점검 기록으로 남긴다.',
              ...dispositionLines,
              '필요한 경우 향후 정기매수 정책, 관찰 유지, thesis 재검토 여부를 별도로 확인한다.'
            ].join(' ')
          : '이번 기록은 주문 지시가 아니라 점검 기록으로 남긴다. 선택된 항목이 없으므로 추가 판단은 보류한다.',
      evidence: includedItems.slice(0, 4).map(({ item }) => ({
        review_item_id: reviewItemId(item),
        field: 'suggested_next_step',
        value: item.suggested_next_step
      })),
      editable: true
    },
    {
      id: 'follow-up',
      section: 'follow_up',
      title: 'Follow-up',
      draft_text: '다음 평가에서 동일 trigger, risk contribution, thesis_status, 계층 부담이 완화 또는 악화됐는지 확인한다.',
      evidence: includedItems.slice(0, 4).map(({ item }) => ({
        review_item_id: reviewItemId(item),
        field: 'metrics_snapshot',
        value: stringifyEvidence(item.metrics_snapshot)
      })),
      editable: true
    }
  ];
  const draftBlocks: JournalDraftComposerSurfacePayload['draft_blocks'] = rawDraftBlocks.map(ensureEvidence);

  return {
    component: 'JournalDraftComposerSurface',
    catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
    title: 'Decision Journal Composer',
    decision_context: decisionContext,
    included_items: includedItems.map(({ decision, item }) => ({
      review_item_id: reviewItemId(item),
      name: item.name,
      status: item.status,
      disposition: decision.disposition
    })),
    draft_blocks: draftBlocks,
    follow_up_checklist: includedItems.map(({ item }) => ({
      id: `follow-up:${reviewItemId(item)}`,
      text: `${item.name}: ${item.suggested_next_step}`,
      source_review_item_ids: [reviewItemId(item)]
    })),
    guardrail_notice: { text: GUARDRAIL_NOTICE }
  };
}
