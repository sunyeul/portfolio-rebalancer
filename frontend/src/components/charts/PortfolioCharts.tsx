import type { EvaluationResponse } from '../../lib/api';

export function EvaluationCharts({ evaluation }: { evaluation: EvaluationResponse }) {
  const counts = evaluation.asset_evaluations.reduce<Record<string, number>>((acc, row) => {
    const status = row.output.status;
    acc[status] = (acc[status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="grid gap-3 sm:grid-cols-4">
      {(['OK', 'Watch', 'Review', 'Action'] as const).map((status) => (
        <div key={status} className="rounded-lg border border-slate-200 bg-white p-4">
          <span className="block text-sm font-bold text-slate-500">{status}</span>
          <strong className="mt-2 block text-2xl font-bold text-slate-950">{counts[status] ?? 0}</strong>
        </div>
      ))}
    </div>
  );
}

export function ReliabilityRiskChart() {
  return null;
}
