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
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <span className="block text-sm font-semibold text-slate-500">{label}</span>
      <strong className="mt-2 block text-2xl font-bold text-slate-950">{display}</strong>
    </div>
  );
}
