import { expect, test } from "bun:test";

import { evidenceValue, formatKrw, formatSignedKrw, supportedRate } from "../src/lib/presentation";

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
