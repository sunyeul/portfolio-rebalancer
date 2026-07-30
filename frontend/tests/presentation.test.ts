import { expect, test } from "bun:test";

import { currentEvaluationResult } from "../src/lib/api";
import { evidenceValue, formatAccountReturn, formatAllocationReason, formatEvaluationCurrentness, formatKrw, formatQueuePriority, formatSignedKrw, supportedRate } from "../src/lib/presentation";

test("KRW formatting removes fractional won", () => {
  expect(formatKrw(120802745.17802304)).toBe("120,802,745 KRW");
  expect(formatKrw(-141765.95742144)).toBe("-141,766 KRW");
  expect(formatSignedKrw(6020.9)).toBe("+6,021 KRW");
  expect(formatKrw(null)).toBe("—");
});

test("supportedRate requires finite values and a positive basis", () => {
  expect(supportedRate(700000, 6500000)).toBeCloseTo(0.1076923077);
  expect(supportedRate(null, 6500000)).toBeNull();
  expect(supportedRate(0, 0)).toBeNull();
});

test("evidenceValue distinguishes zero from unavailable evidence", () => {
  expect(evidenceValue(0, "complete", formatKrw)).toBe("0 KRW");
  expect(evidenceValue(null, "insufficient_history", formatKrw)).toBe(
    "insufficient_history",
  );
  expect(evidenceValue(null, null, formatKrw)).toBe("자료 없음");
});

test("account return formatting distinguishes zero from missing principal evidence", () => {
  expect(formatAccountReturn(0)).toBe("0.0%");
  expect(formatAccountReturn(null)).toBe("자료 없음");
});

test("allocation blocking reasons remain human-readable without inventing a decision", () => {
  expect(formatAllocationReason("gross_account_denominator_invalid")).toBe(
    "총계좌 평가금 분모를 확인할 수 없습니다.",
  );
  expect(formatAllocationReason("invested_denominator_unavailable")).toBe(
    "투자금 평가금 분모를 확인할 수 없습니다.",
  );
  expect(formatAllocationReason("unknown_reason")).toBe("unknown_reason");
  expect(formatAllocationReason(null)).toContain("비중 조정 판단");
});

test("blocking queue items keep their priority label even without a priority code", () => {
  expect(formatQueuePriority(null, "평가 차단")).toBe("평가 차단");
  expect(formatQueuePriority("P1", "다음 정기매수 전")).toBe("P1 · 다음 정기매수 전");
});

test("evaluation currentness explains stale snapshot and policy IDs", () => {
  expect(formatEvaluationCurrentness({
    is_current: false,
    reasons: ["snapshot_mismatch", "policy_version_mismatch"],
    evaluation_snapshot_id: 7,
    current_snapshot_id: 8,
    evaluation_policy_version_id: 3,
    active_policy_version_id: 4,
  })).toBe(
    "저장 평가 스냅샷 #7 · 현재 스냅샷 #8 · 저장 평가 정책 #3 · 활성 정책 #4. 최신 Toss 스냅샷과 활성 정책으로 inspection run을 명시적으로 실행한 뒤 다시 불러오세요.",
  );
});

test("stale evaluation result is not available to workbench panels", () => {
  const evaluation = { result: { review_queue: [{ identity: "US/AAA" }] } };
  const currentness = {
    is_current: false,
    reasons: ["snapshot_mismatch"],
    evaluation_snapshot_id: 7,
    current_snapshot_id: 8,
    evaluation_policy_version_id: 4,
    active_policy_version_id: 4,
  };

  expect(currentEvaluationResult(evaluation, currentness)).toBeUndefined();
  expect(currentEvaluationResult(evaluation, { ...currentness, is_current: true }))
    .toBe(evaluation.result);
});
