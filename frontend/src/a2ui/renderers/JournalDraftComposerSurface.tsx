import { CheckSquare, ClipboardCopy, FileText } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { JournalDraftComposerSurfacePayload } from '../types';
import { GuardrailNotice } from './GuardrailNotice';

const sectionLabels: Record<JournalDraftComposerSurfacePayload['draft_blocks'][number]['section'], string> = {
  context: 'Context',
  observation: 'Observation',
  interpretation: 'Interpretation',
  decision: 'Decision',
  follow_up: 'Follow-up'
};

export function JournalDraftComposerSurface({ surface }: { surface: JournalDraftComposerSurfacePayload }) {
  const [draftBlocks, setDraftBlocks] = useState(surface.draft_blocks);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const copyText = useMemo(
    () =>
      draftBlocks
        .map((block) => `[${sectionLabels[block.section]}]\n${block.draft_text}`)
        .concat(
          surface.follow_up_checklist.length > 0
            ? [`[Follow-up Checklist]\n${surface.follow_up_checklist.map((item) => `- ${item.text}`).join('\n')}`]
            : []
        )
        .join('\n\n'),
    [draftBlocks, surface.follow_up_checklist]
  );

  async function copyDraft() {
    try {
      await navigator.clipboard.writeText(copyText);
      setCopyState('copied');
    } catch {
      setCopyState('failed');
    }
  }

  return (
    <section className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <header className="flex items-start gap-2">
        <FileText className="mt-1 h-5 w-5 flex-none text-slate-700" aria-hidden="true" />
        <div className="min-w-0">
          <h3 className="m-0 text-base font-extrabold text-slate-950">{surface.title}</h3>
          <p className="m-0 text-sm font-semibold text-slate-500">{surface.decision_context}</p>
        </div>
      </header>

      <GuardrailNotice text={surface.guardrail_notice.text} />

      <section className="rounded-lg border border-slate-200 bg-white p-3">
        <h4 className="m-0 text-sm font-extrabold text-slate-900">Included Review Items</h4>
        <div className="mt-2 grid gap-2">
          {surface.included_items.length > 0 ? (
            surface.included_items.map((item) => (
              <div key={item.review_item_id} className="rounded-md bg-slate-50 px-3 py-2 text-sm">
                <strong className="text-slate-900">{item.name}</strong>
                <span className="ml-2 font-bold text-slate-500">{item.status}</span>
                <span className="ml-2 font-semibold text-slate-500">{item.disposition}</span>
              </div>
            ))
          ) : (
            <p className="m-0 text-sm font-semibold text-slate-500">No selected Review Queue items.</p>
          )}
        </div>
      </section>

      <section className="grid gap-2">
        {draftBlocks.map((block, index) => (
          <article key={block.id} className="rounded-lg border border-slate-200 bg-white p-3">
            <label className="grid gap-2">
              <span className="text-sm font-extrabold text-slate-900">{block.title}</span>
              <textarea
                className="min-h-24 resize-y rounded-md border border-slate-300 bg-white px-3 py-2 text-sm leading-6 text-slate-800"
                value={block.draft_text}
                onChange={(event) =>
                  setDraftBlocks((current) =>
                    current.map((candidate, candidateIndex) =>
                      candidateIndex === index ? { ...candidate, draft_text: event.target.value } : candidate
                    )
                  )
                }
              />
            </label>
            <details className="mt-2">
              <summary className="cursor-pointer text-xs font-extrabold uppercase text-slate-500">Evidence</summary>
              <ul className="mt-2 grid gap-1 text-xs font-semibold text-slate-600">
                {block.evidence.map((evidence) => (
                  <li key={`${block.id}:${evidence.review_item_id}:${evidence.field}`}>
                    {evidence.review_item_id} · {evidence.field}: {evidence.value}
                  </li>
                ))}
              </ul>
            </details>
          </article>
        ))}
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-3">
        <h4 className="m-0 flex items-center gap-2 text-sm font-extrabold text-slate-900">
          <CheckSquare className="h-4 w-4" aria-hidden="true" />
          Follow-up Checklist
        </h4>
        <ul className="mt-2 grid gap-1 text-sm leading-5 text-slate-700">
          {surface.follow_up_checklist.map((item) => (
            <li key={item.id}>- {item.text}</li>
          ))}
        </ul>
      </section>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-3 py-2 text-sm font-extrabold text-white"
          onClick={copyDraft}
        >
          <ClipboardCopy className="h-4 w-4" aria-hidden="true" />
          Copy
        </button>
        <span className="text-sm font-semibold text-slate-500">
          {copyState === 'copied' ? 'Copied.' : copyState === 'failed' ? 'Copy failed.' : 'Preview only. Not saved.'}
        </span>
      </div>
    </section>
  );
}
