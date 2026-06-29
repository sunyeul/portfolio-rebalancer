import { ShieldCheck } from 'lucide-react';

export function GuardrailNotice({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-teal-200 bg-teal-50 px-3 py-2 text-sm font-semibold text-teal-900">
      <ShieldCheck className="mt-0.5 h-4 w-4 flex-none" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}
