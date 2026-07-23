import { expect, test } from "bun:test";

import { evidenceValue, formatAccountReturn, formatAllocationReason, formatKrw, formatSignedKrw, supportedRate } from "../src/lib/presentation";

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
  expect(formatAllocationReason("invested_denominator_unavailable")).toBe(
    "투자금 평가금 분모를 확인할 수 없습니다.",
  );
  expect(formatAllocationReason("unknown_reason")).toBe("unknown_reason");
  expect(formatAllocationReason(null)).toContain("비중 조정 판단");
});
