import { describe, expect, test } from 'bun:test';

import type { ReviewItem } from '../src/lib/api';
import { describeReviewTrigger, groupReviewQueueItems } from '../src/lib/reviewQueue';

function item(status: ReviewItem['status'], name: string): ReviewItem {
  return {
    level: 'asset',
    name,
    parent_layer: 'core',
    status,
    triggered_by: [],
    metrics_snapshot: {},
    thesis: null,
    counter_scenario: null,
    suggested_next_step: '다음 정기 리뷰에서 다시 확인'
  };
}

describe('review queue helpers', () => {
  test('uses Action, Review, Watch order and preserves each group order', () => {
    const groups = groupReviewQueueItems([
      item('Watch', 'GLD'),
      item('Action', 'QQQ'),
      item('Review', 'SMH'),
      item('Action', 'VOO')
    ]);

    expect(groups.map((group) => [group.status, group.items.map((entry) => entry.name)])).toEqual([
      ['Action', ['QQQ', 'VOO']],
      ['Review', ['SMH']],
      ['Watch', ['GLD']]
    ]);
  });

  test('explains known and unknown trigger codes without order language', () => {
    expect(describeReviewTrigger('risk_contribution')).toContain('위험 기여도');
    expect(describeReviewTrigger('new_trigger')).toContain('데이터 근거');
    expect(describeReviewTrigger('risk_contribution')).not.toMatch(/매수|매도|주문/);
  });
});
