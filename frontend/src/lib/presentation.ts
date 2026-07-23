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
