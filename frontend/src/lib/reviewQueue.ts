import type { ReviewItem } from './api';

export const reviewQueueStatusOrder = ['Action', 'Review', 'Watch'] as const;

const reviewTriggerExplanations: Record<string, string> = {
  risk_contribution: '위험 기여도가 커서 이 부담이 의도된 것인지 확인합니다.',
  risk_contribution_high: '위험 기여도가 커서 이 부담이 의도된 것인지 확인합니다.',
  target_gap_outside_tolerance: '현재 비중과 IPS 목표 범위의 차이를 확인합니다.',
  thesis_watch: '기록된 투자 논리가 관찰 또는 재검토 상태인지 확인합니다.',
  thesis_broken: '투자 논리가 훼손됐는지 근거를 다시 확인합니다.',
  volatility_exceeded: '변동성이 허용 기준을 넘었는지 확인합니다.',
  mdd_exceeded: '낙폭이 점검 기준을 넘었는지 확인합니다.',
  efficiency_below_threshold: '성과 대비 위험 효율이 낮아졌는지 확인합니다.',
  high_burden: '관리 부담이나 계층 내 부담이 커졌는지 확인합니다.',
  max_weight_exceeded: '항목 또는 계층 비중이 상한을 넘었는지 확인합니다.'
};

export function describeReviewTrigger(code: string) {
  return reviewTriggerExplanations[code] ?? '기록된 점검 신호와 데이터 근거를 확인합니다.';
}

export function groupReviewQueueItems(queue: ReviewItem[]) {
  return reviewQueueStatusOrder.map((status) => ({
    status,
    items: queue.filter((item) => item.status === status)
  }));
}
