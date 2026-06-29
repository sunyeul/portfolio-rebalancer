import { describe, expect, test } from 'bun:test';

import {
  initialGenerativeUiState,
  generativeUiReducer
} from '../src/a2ui/GenerativeUiContext';
import { IPS_PILOT_REVIEW_CATALOG_ID } from '../src/a2ui/catalogs/ipsPilotReviewCatalog';
import type {
  A2UiAppSurfaceEnvelope,
  EvaluationGraphSurfacePayload,
  GenerativeSurfacePatch,
  JournalDraftComposerSurfacePayload,
  ReviewQueueTriageSurfacePayload
} from '../src/a2ui/types';
import {
  applyA2UiAppSurfacePayload,
  convertA2UiToSurfacePatch,
  validateA2UiAppSurfaceEnvelope
} from '../src/a2ui/utils/validation';
import {
  buildReviewQueueTriageSurface,
  defaultReviewDecisionsFromEvaluation,
  mergeReviewQueueAgentExplanations
} from '../src/a2ui/utils/builders';
import type { EvaluationResponse } from '../src/lib/api';

const reviewSurface: ReviewQueueTriageSurfacePayload = {
  component: 'ReviewQueueTriageSurface',
  catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
  title: 'Review Queue Triage',
  evaluation_period: {
    label: '1M',
    start_date: '2026-06-01',
    end_date: '2026-06-27'
  },
  guardrail_notice: {
    text: 'Inspection only.'
  },
  groups: [
    {
      status: 'Action',
      summary: 'Inspect exceptional intervention candidates.',
      items: [
        {
          id: 'asset:satellite:SMH',
          level: 'asset',
          name: 'SMH',
          parent_layer: 'satellite',
          status: 'Action',
          triggered_by: ['risk_contribution'],
          status_label_ko: '조치 검토',
          trigger_explanations: [
            {
              code: 'risk_contribution',
              explanation: '포트폴리오 위험 기여도가 커서, 이 노출이 의도된 부담인지 확인해야 합니다.'
            }
          ],
          agent_summary: 'SMH은 Action 상태로 점검 대상입니다.',
          ips_interpretation: 'SMH은 satellite 노출이므로 thesis, overlap, risk burden을 확인해야 합니다.',
          verification_focus: 'thesis, overlap, risk burden, ETF substitution 검토가 필요한지 확인합니다.',
          review_questions: ['이 항목의 계층 내 역할이 아직 분명한가?'],
          suggested_next_step: 'SMH은 thesis와 위험 부담을 다음 리뷰 메모에 포함한다.',
          next_review_note: 'SMH은 thesis와 위험 부담을 다음 리뷰 메모에 포함한다.',
          allowed_dispositions: ['include_in_journal', 'review_thesis', 'observe']
        }
      ]
    }
  ]
};

const journalSurface: JournalDraftComposerSurfacePayload = {
  component: 'JournalDraftComposerSurface',
  catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
  title: 'Decision Journal Composer',
  decision_context: 'regular_review',
  included_items: [
    {
      review_item_id: 'asset:satellite:SMH',
      name: 'SMH',
      status: 'Review',
      disposition: 'review_thesis'
    }
  ],
  draft_blocks: [
    {
      id: 'decision',
      section: 'decision',
      title: 'Decision',
      draft_text: 'Record this as an inspection note.',
      evidence: [
        {
          review_item_id: 'asset:satellite:SMH',
          field: 'status',
          value: 'Review'
        }
      ],
      editable: true
    }
  ],
  follow_up_checklist: [
    {
      id: 'follow-up:smh',
      text: 'Verify thesis.',
      source_review_item_ids: ['asset:satellite:SMH']
    }
  ],
  guardrail_notice: {
    text: 'Inspection only.'
  }
};

const graphSurface: EvaluationGraphSurfacePayload = {
  component: 'EvaluationGraphSurface',
  catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
  title: '계층/종목 평가 그래프',
  evaluation_period: {
    label: '1M',
    start_date: '2026-06-01',
    end_date: '2026-06-27'
  },
  guardrail_notice: {
    text: 'Inspection only.'
  },
  charts: [
    {
      id: 'layer-gap',
      chart_type: 'layer_weight_gap_bar',
      title: '계층 비중과 목표 대비 차이',
      source: 'layer_evaluations'
    },
    {
      id: 'risk-scatter',
      chart_type: 'asset_risk_scatter',
      title: '종목 비중과 위험 기여',
      source: 'asset_evaluations'
    },
    {
      id: 'risk-bars',
      chart_type: 'metric_bar',
      title: '위험 기여 상위 종목',
      source: 'asset_evaluations',
      metric: 'risk_contribution',
      sort: {
        by: 'risk_contribution',
        direction: 'desc'
      },
      limit: 10
    }
  ]
};

function envelope(
  overrides: Partial<A2UiAppSurfaceEnvelope> = {}
): A2UiAppSurfaceEnvelope {
  return {
    catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
    target: 'review_queue',
    mode: 'replace',
    surface: reviewSurface,
    source: {
      agent: 'reviewCopilot',
      created_at: '2026-06-27T00:00:00.000Z'
    },
    ...overrides
  };
}

describe('A2UI app surface validation', () => {
  test('accepts ReviewQueueTriageSurface with review_queue target', () => {
    const result = validateA2UiAppSurfaceEnvelope(envelope());
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(convertA2UiToSurfacePatch(result.envelope)).toMatchObject({
        target: 'review_queue',
        surface: 'ReviewQueueTriageSurface'
      });
    }
  });

  test('builds ReviewQueueTriageSurface with Korean interpretation fields', () => {
    const evaluation: EvaluationResponse = {
      evaluation_period: {
        label: '1M',
        start_date: '2026-06-01',
        end_date: '2026-06-27'
      },
      layer_evaluations: [],
      asset_evaluations: [],
      review_queue: [
        {
          level: 'asset',
          name: 'SMH',
          parent_layer: 'satellite',
          status: 'Action',
          triggered_by: ['risk_contribution_high', 'thesis_watch'],
          metrics_snapshot: {},
          thesis: 'Semiconductor cycle satellite exposure.',
          counter_scenario: null,
          suggested_next_step: 'Verify thesis and risk burden.'
        }
      ],
      journal_draft: [],
      warnings: [],
      guardrails: {
        not_investment_advice: true,
        no_immediate_order_instruction: true
      }
    };

    const surface = buildReviewQueueTriageSurface(evaluation);
    const item = surface.groups.find((group) => group.status === 'Action')?.items[0];

    expect(surface.title).toBe('Review Queue 점검 해석 보드');
    expect(item?.status_label_ko).toBe('조치 검토');
    expect(item?.trigger_explanations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'risk_contribution_high',
          explanation: expect.stringContaining('포트폴리오 위험 기여도')
        })
      ])
    );
    expect(item?.ips_interpretation).toContain('satellite');
    expect(item?.verification_focus).toContain('ETF substitution');
    expect(item?.next_review_note).toContain('저널');
    expect(item?.next_review_note).toContain('확인');
    expect(validateA2UiAppSurfaceEnvelope(envelope({ surface })).ok).toBe(true);
  });

  test('merges automatic and requested agent explanations into ReviewQueueTriageSurface', () => {
    const automaticSurface = mergeReviewQueueAgentExplanations(reviewSurface, {
      source: 'automatic',
      created_at: '2026-06-27T12:00:00.000Z',
      overview: '이번 보드는 thesis와 위험 부담 확인을 우선합니다.',
      explanations: [
        {
          review_item_id: 'asset:satellite:SMH',
          text: 'SMH는 satellite 노출이므로 thesis 지속성과 위험 부담을 함께 확인하는 항목입니다.'
        }
      ]
    });
    const requestedSurface = mergeReviewQueueAgentExplanations(automaticSurface, {
      source: 'requested',
      created_at: '2026-06-27T12:05:00.000Z',
      explanations: [
        {
          review_item_id: 'asset:satellite:SMH',
          text: '추가 설명에서는 ETF 대체 가능성과 core 중복 여부를 함께 점검합니다.'
        }
      ]
    });
    const item = requestedSurface.groups[0].items[0];

    expect(requestedSurface.agent_overview?.source).toBe('automatic');
    expect(item.agent_explanations).toEqual([
      {
        source: 'automatic',
        created_at: '2026-06-27T12:00:00.000Z',
        text: 'SMH는 satellite 노출이므로 thesis 지속성과 위험 부담을 함께 확인하는 항목입니다.'
      },
      {
        source: 'requested',
        created_at: '2026-06-27T12:05:00.000Z',
        text: '추가 설명에서는 ETF 대체 가능성과 core 중복 여부를 함께 점검합니다.'
      }
    ]);
    expect(validateA2UiAppSurfaceEnvelope(envelope({ surface: requestedSurface })).ok).toBe(true);
  });

  test('rejects forbidden vocabulary in agent explanation fields', () => {
    const surface = mergeReviewQueueAgentExplanations(reviewSurface, {
      source: 'requested',
      created_at: '2026-06-27T12:05:00.000Z',
      overview: 'Do not place_order from this board.',
      explanations: [
        {
          review_item_id: 'asset:satellite:SMH',
          text: 'Do not sell from this inspection board.'
        }
      ]
    });

    expect(validateA2UiAppSurfaceEnvelope(envelope({ surface })).ok).toBe(false);
  });

  test('builds default Review Queue decisions from status', () => {
    const evaluation: EvaluationResponse = {
      evaluation_period: {
        label: '1M',
        start_date: '2026-06-01',
        end_date: '2026-06-27'
      },
      layer_evaluations: [],
      asset_evaluations: [],
      review_queue: [
        {
          level: 'asset',
          name: 'SMH',
          parent_layer: 'satellite',
          status: 'Action',
          triggered_by: ['risk_contribution_high'],
          metrics_snapshot: {},
          thesis: null,
          counter_scenario: null,
          suggested_next_step: 'Verify thesis and risk burden.'
        },
        {
          level: 'asset',
          name: 'QQQ',
          parent_layer: 'core',
          status: 'Review',
          triggered_by: ['target_gap_outside_tolerance'],
          metrics_snapshot: {},
          thesis: null,
          counter_scenario: null,
          suggested_next_step: 'Verify core role.'
        },
        {
          level: 'asset',
          name: 'GLD',
          parent_layer: 'satellite',
          status: 'Watch',
          triggered_by: ['thesis_watch'],
          metrics_snapshot: {},
          thesis: null,
          counter_scenario: null,
          suggested_next_step: 'Observe until next review.'
        }
      ],
      journal_draft: [],
      warnings: [],
      guardrails: {
        not_investment_advice: true,
        no_immediate_order_instruction: true
      }
    };

    expect(defaultReviewDecisionsFromEvaluation(evaluation)).toEqual([
      {
        review_item_id: 'asset:satellite:SMH',
        disposition: 'include_in_journal'
      },
      {
        review_item_id: 'asset:core:QQQ',
        disposition: 'include_in_journal'
      },
      {
        review_item_id: 'asset:satellite:GLD',
        disposition: 'observe'
      }
    ]);
    expect(defaultReviewDecisionsFromEvaluation(null)).toEqual([]);
  });

  test('accepts JournalDraftComposerSurface with journal_draft target', () => {
    const result = validateA2UiAppSurfaceEnvelope(
      envelope({
        target: 'journal_draft',
        surface: journalSurface
      })
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(convertA2UiToSurfacePatch(result.envelope)).toMatchObject({
        target: 'journal_draft',
        surface: 'JournalDraftComposerSurface'
      });
    }
  });

  test('rejects ReviewQueueTriageSurface with journal_draft target', () => {
    const result = validateA2UiAppSurfaceEnvelope(envelope({ target: 'journal_draft' }));
    expect(result.ok).toBe(false);
  });

  test('accepts EvaluationGraphSurface with evaluation_graphs target', () => {
    const result = validateA2UiAppSurfaceEnvelope(
      envelope({
        target: 'evaluation_graphs',
        surface: graphSurface
      })
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(convertA2UiToSurfacePatch(result.envelope)).toMatchObject({
        target: 'evaluation_graphs',
        surface: 'EvaluationGraphSurface'
      });
    }
  });

  test('rejects EvaluationGraphSurface with non-graph targets', () => {
    expect(validateA2UiAppSurfaceEnvelope(envelope({ surface: graphSurface })).ok).toBe(false);
    expect(
      validateA2UiAppSurfaceEnvelope(
        envelope({
          target: 'journal_draft',
          surface: graphSurface
        })
      ).ok
    ).toBe(false);
  });

  test('rejects invalid EvaluationGraphSurface chart source, chart type, and metric', () => {
    expect(
      validateA2UiAppSurfaceEnvelope(
        envelope({
          target: 'evaluation_graphs',
          surface: {
            ...graphSurface,
            charts: [
              {
                id: 'bad-source',
                chart_type: 'asset_risk_scatter',
                title: 'Bad source',
                source: 'layer_evaluations'
              }
            ]
          }
        })
      ).ok
    ).toBe(false);
    expect(
      validateA2UiAppSurfaceEnvelope(
        envelope({
          target: 'evaluation_graphs',
          surface: {
            ...graphSurface,
            charts: [
              {
                id: 'bad-chart',
                chart_type: 'pie',
                title: 'Bad chart',
                source: 'asset_evaluations'
              }
            ]
          }
        })
      ).ok
    ).toBe(false);
    expect(
      validateA2UiAppSurfaceEnvelope(
        envelope({
          target: 'evaluation_graphs',
          surface: {
            ...graphSurface,
            charts: [
              {
                id: 'bad-metric',
                chart_type: 'metric_bar',
                title: 'Bad metric',
                source: 'layer_evaluations',
                metric: 'layer_internal_weight'
              }
            ]
          }
        })
      ).ok
    ).toBe(false);
  });

  test('rejects missing targets and unknown components', () => {
    expect(validateA2UiAppSurfaceEnvelope({ ...envelope(), target: undefined }).ok).toBe(false);
    expect(
      validateA2UiAppSurfaceEnvelope({
        ...envelope(),
        surface: { component: 'UnknownSurface' }
      }).ok
    ).toBe(false);
  });

  for (const word of ['buy', 'sell', 'calculate_order_size']) {
    test(`rejects forbidden action vocabulary: ${word}`, () => {
      const result = validateA2UiAppSurfaceEnvelope(
        envelope({
          surface: {
            ...reviewSurface,
            groups: [
              {
                ...reviewSurface.groups[0],
                items: [
                  {
                    ...reviewSurface.groups[0].items[0],
                    suggested_next_step: `Do not ${word}.`
                  }
                ]
              }
            ]
          }
        })
      );
      expect(result.ok).toBe(false);
    });
  }

  for (const word of ['buy', 'sell', 'calculate_order_size', 'place_order']) {
    test(`rejects forbidden action vocabulary in graph surface: ${word}`, () => {
      const result = validateA2UiAppSurfaceEnvelope(
        envelope({
          target: 'evaluation_graphs',
          surface: {
            ...graphSurface,
            title: `Do not ${word} from this graph.`
          }
        })
      );
      expect(result.ok).toBe(false);
    });
  }

  test('rejects forbidden action vocabulary in explanation fields', () => {
    const result = validateA2UiAppSurfaceEnvelope(
      envelope({
        surface: {
          ...reviewSurface,
          groups: [
            {
              ...reviewSurface.groups[0],
              items: [
                {
                  ...reviewSurface.groups[0].items[0],
                  ips_interpretation: 'Do not place_order from this inspection board.'
                }
              ]
            }
          ]
        }
      })
    );
    expect(result.ok).toBe(false);
  });

  test('rejects journal draft blocks without evidence', () => {
    const result = validateA2UiAppSurfaceEnvelope(
      envelope({
        target: 'journal_draft',
        surface: {
          ...journalSurface,
          draft_blocks: [
            {
              ...journalSurface.draft_blocks[0],
              evidence: []
            }
          ]
        }
      })
    );
    expect(result.ok).toBe(false);
  });

  test('applies valid payloads as surface patches and preserves review decisions', () => {
    const patches: GenerativeSurfacePatch[] = [];
    const result = applyA2UiAppSurfacePayload(envelope(), (patch) => patches.push(patch));
    expect(result.ok).toBe(true);
    expect(patches).toHaveLength(1);

    const withSurface = generativeUiReducer(initialGenerativeUiState, {
      type: 'applySurfacePatch',
      patch: patches[0]
    });
    const withDecision = generativeUiReducer(withSurface, {
      type: 'updateReviewDecision',
      decision: {
        review_item_id: 'asset:satellite:SMH',
        disposition: 'review_thesis'
      }
    });

    expect(withDecision.reviewQueueSurface?.component).toBe('ReviewQueueTriageSurface');
    expect(withDecision.reviewDecisions).toEqual([
      {
        review_item_id: 'asset:satellite:SMH',
        disposition: 'review_thesis'
      }
    ]);
  });

  test('reducer replaces and appends EvaluationGraphSurface charts', () => {
    const replacePatch: GenerativeSurfacePatch = {
      target: 'evaluation_graphs',
      surface: 'EvaluationGraphSurface',
      mode: 'replace',
      payload: graphSurface
    };
    const appendPatch: GenerativeSurfacePatch = {
      target: 'evaluation_graphs',
      surface: 'EvaluationGraphSurface',
      mode: 'append',
      payload: {
        ...graphSurface,
        title: '추가 그래프',
        charts: [
          {
            id: 'mdd-bars',
            chart_type: 'metric_bar',
            title: 'MDD 점검',
            source: 'asset_evaluations',
            metric: 'mdd',
            sort: {
              by: 'mdd',
              direction: 'asc'
            }
          }
        ]
      }
    };

    const withGraph = generativeUiReducer(initialGenerativeUiState, {
      type: 'applySurfacePatch',
      patch: replacePatch
    });
    const withAppend = generativeUiReducer(withGraph, {
      type: 'applySurfacePatch',
      patch: appendPatch
    });

    expect(withGraph.evaluationGraphSurface?.charts).toHaveLength(3);
    expect(withAppend.evaluationGraphSurface?.charts.map((chart) => chart.id)).toEqual([
      'layer-gap',
      'risk-scatter',
      'risk-bars',
      'mdd-bars'
    ]);
  });
});
