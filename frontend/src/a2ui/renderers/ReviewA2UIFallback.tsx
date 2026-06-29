import type { ReviewA2UIEnvelope } from '../types';

export function ReviewA2UIFallback({ envelope }: { envelope?: ReviewA2UIEnvelope }) {
  const fallbackText =
    !envelope || envelope.ok
      ? 'A2UI validation failed. 검토용 plain text fallback을 사용하세요.'
      : envelope.fallback_text;

  return (
    <section className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
      <strong className="block font-extrabold">A2UI fallback</strong>
      <p className="m-0 mt-1 font-semibold">{fallbackText}</p>
    </section>
  );
}
