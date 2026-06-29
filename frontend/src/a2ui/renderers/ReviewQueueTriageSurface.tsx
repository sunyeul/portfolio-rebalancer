import { ClipboardList } from 'lucide-react';

import type {
  ReviewDecision,
  ReviewDisposition,
  ReviewItemCardPayload,
  ReviewQueueTriageSurfacePayload
} from '../types';
import { GuardrailNotice } from './GuardrailNotice';

const dispositionLabels: Record<ReviewDisposition, string> = {
  include_in_journal: '저널에 포함',
  observe: '관찰 유지',
  review_thesis: '논리 재검토',
  defer_until_next_review: '다음 리뷰로 보류'
};

const statusStyles = {
  Action: 'border-red-200 bg-red-50 text-red-800',
  Review: 'border-blue-200 bg-blue-50 text-blue-800',
  Watch: 'border-amber-200 bg-amber-50 text-amber-800'
} as const;

const groupLabels = {
  Action: 'Action · 조치 검토',
  Review: 'Review · 중점 점검',
  Watch: 'Watch · 관찰 점검'
} as const;

function ReviewItemCard({
  item,
  disposition,
  onDispositionChange
}: {
  item: ReviewItemCardPayload;
  disposition: ReviewDisposition;
  onDispositionChange: (disposition: ReviewDisposition) => void;
}) {
  const agentExplanationTexts =
    item.agent_explanations && item.agent_explanations.length > 0
      ? item.agent_explanations.map((explanation) => explanation.text)
      : [item.agent_summary];

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <strong className="text-sm text-slate-950">{item.name}</strong>
            <span className={`rounded-full border px-2 py-0.5 text-xs font-bold ${statusStyles[item.status]}`}>
              {item.status} · {item.status_label_ko}
            </span>
          </div>
          <p className="mt-1 text-xs font-semibold text-slate-500">
            {item.level}
            {item.parent_layer ? ` / ${item.parent_layer}` : ''}
          </p>
        </div>
        <label className="grid gap-1 text-xs font-bold text-slate-600">
          <span>처리 방침</span>
          <select
            className="min-w-36 rounded-md border border-slate-300 bg-white px-2 py-1 text-sm font-semibold text-slate-900"
            value={disposition}
            onChange={(event) => onDispositionChange(event.target.value as ReviewDisposition)}
          >
            {item.allowed_dispositions.map((value) => (
              <option key={value} value={value}>
                {dispositionLabels[value]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-3 grid gap-3">
        <div className="grid gap-1">
          <p className="text-xs font-extrabold text-slate-500">에이전트 해석</p>
          <div className="grid gap-2">
            {agentExplanationTexts.map((text, index) => (
              <p
                key={`${item.id}:agent-explanation:${index}`}
                className="rounded-md border border-cyan-100 bg-cyan-50 px-3 py-2 text-sm leading-6 text-slate-700"
              >
                {text}
              </p>
            ))}
          </div>
        </div>

        <div className="grid gap-1">
          <p className="text-xs font-extrabold text-slate-500">왜 올라왔나</p>
          <div className="grid gap-2">
            {item.trigger_explanations.map((trigger) => (
              <div key={trigger.code} className="rounded-md bg-slate-50 px-3 py-2">
                <p className="text-sm leading-6 text-slate-700">{trigger.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}

export function ReviewQueueTriageSurface({
  surface,
  decisions,
  onDecisionChange
}: {
  surface: ReviewQueueTriageSurfacePayload;
  decisions: ReviewDecision[];
  onDecisionChange: (decision: ReviewDecision) => void;
}) {
  const dispositionById = new Map(decisions.map((decision) => [decision.review_item_id, decision.disposition]));
  const itemCount = surface.groups.reduce((count, group) => count + group.items.length, 0);

  return (
    <section className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <header className="flex items-start gap-2">
        <ClipboardList className="mt-1 h-5 w-5 flex-none text-slate-700" aria-hidden="true" />
        <div className="min-w-0">
          <h3 className="m-0 text-base font-extrabold text-slate-950">{surface.title}</h3>
          <p className="m-0 text-sm font-semibold text-slate-500">
            {surface.evaluation_period.label}: {surface.evaluation_period.start_date} ~{' '}
            {surface.evaluation_period.end_date} · {itemCount}개 항목
          </p>
        </div>
      </header>

      <GuardrailNotice text={surface.guardrail_notice.text} />

      <div className="grid gap-4">
        {surface.groups.map((group) => (
          <section key={group.status} className="grid gap-2">
            <div>
              <h4 className="m-0 text-sm font-extrabold text-slate-900">{groupLabels[group.status]}</h4>
            </div>
            {group.items.length > 0 ? (
              <div className="grid gap-2">
                {group.items.map((item) => (
                  <ReviewItemCard
                    key={item.id}
                    item={item}
                    disposition={dispositionById.get(item.id) ?? item.allowed_dispositions[0]}
                    onDispositionChange={(disposition) =>
                      onDecisionChange({
                        review_item_id: item.id,
                        disposition
                      })
                    }
                  />
                ))}
              </div>
            ) : (
              <div className="rounded-md border border-dashed border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-500">
                해당 상태의 점검 항목이 없습니다.
              </div>
            )}
          </section>
        ))}
      </div>
    </section>
  );
}
