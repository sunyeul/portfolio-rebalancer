const wholeWon = new Intl.NumberFormat("ko-KR", {
  maximumFractionDigits: 0,
  minimumFractionDigits: 0,
});

export function finiteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function formatPercent(value: unknown) {
  const number = finiteNumber(value);
  return number === null ? "—" : `${(number * 100).toFixed(1)}%`;
}

export function formatAccountReturn(value: unknown) {
  return finiteNumber(value) === null ? "자료 없음" : formatPercent(value);
}

export function formatKrw(value: unknown) {
  const number = finiteNumber(value);
  return number === null ? "—" : `${wholeWon.format(number)} KRW`;
}

export function formatSignedKrw(value: unknown) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  return `${number > 0 ? "+" : ""}${wholeWon.format(number)} KRW`;
}

export function supportedRate(numerator: unknown, denominator: unknown) {
  const supportedNumerator = finiteNumber(numerator);
  const supportedDenominator = finiteNumber(denominator);
  if (supportedNumerator === null || supportedDenominator === null || supportedDenominator <= 0) return null;
  return supportedNumerator / supportedDenominator;
}

export function evidenceValue(
  value: unknown,
  state: unknown,
  formatter: (value: unknown) => string,
) {
  if (finiteNumber(value) !== null) return formatter(value);
  if (typeof state === "string" && state.length) return state;
  return "자료 없음";
}

const allocationReasonLabels: Record<string, string> = {
  source_not_current_evaluable: "현재 Toss 스냅샷이 평가 가능 상태가 아닙니다.",
  holdings_reconciliation_failed: "보유 종목 조정 검증이 완료되지 않았습니다.",
  gross_denominator_invalid: "총계좌 평가금 분모를 확인할 수 없습니다.",
  invested_denominator_unavailable: "투자금 평가금 분모를 확인할 수 없습니다.",
  policy_coverage_incomplete: "정책의 종목·레이어 커버리지가 완전하지 않습니다.",
};

export function formatAllocationReason(value: unknown) {
  if (typeof value !== "string" || !value.length) return "현재 비중 조정 판단에 필요한 자료를 확인할 수 없습니다.";
  return allocationReasonLabels[value] ?? value;
}

export function formatQueuePriority(priority: unknown, label: unknown) {
  const readableLabel = typeof label === "string" && label.length ? label : null;
  const readablePriority = typeof priority === "string" && priority.length ? priority : null;
  if (readableLabel) return readablePriority ? `${readablePriority} · ${readableLabel}` : readableLabel;
  return readablePriority ?? "검토 시점 미상";
}
