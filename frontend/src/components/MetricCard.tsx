type MetricCardProps = {
  label: string;
  value: number | null | undefined;
  format?: 'pct' | 'number';
};

export function MetricCard({ label, value, format = 'pct' }: MetricCardProps) {
  const display =
    value === null || value === undefined
      ? 'N/A'
      : format === 'pct'
        ? `${(value * 100).toFixed(2)}%`
        : value.toFixed(2);

  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{display}</strong>
    </div>
  );
}

