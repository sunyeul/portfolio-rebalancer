import type { AgentExplanationSource, ReviewQueueTriageSurfacePayload } from '../types';
import type { ReviewQueueAgentExplanationPatch } from './builders';

type AgentExplanationResponse = {
  overview?: string | null;
  explanations?: Array<{
    review_item_id: string;
    text: string;
  }>;
};

function itemPayload(surface: ReviewQueueTriageSurfacePayload) {
  return surface.groups.flatMap((group) =>
    group.items.map((item) => ({
      id: item.id,
      level: item.level,
      name: item.name,
      parent_layer: item.parent_layer,
      status: item.status,
      triggered_by: item.triggered_by,
      summary: item.agent_summary,
      ips_interpretation: item.ips_interpretation,
      verification_focus: item.verification_focus,
      next_review_note: item.next_review_note
    }))
  );
}

export async function requestReviewQueueAgentExplanations({
  signal,
  source,
  surface
}: {
  signal?: AbortSignal;
  source: AgentExplanationSource;
  surface: ReviewQueueTriageSurfacePayload;
}): Promise<ReviewQueueAgentExplanationPatch> {
  const response = await fetch('/copilotkit/review-queue/explanations', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source,
      evaluation_period: surface.evaluation_period,
      groups: surface.groups.map((group) => ({
        status: group.status,
        summary: group.summary
      })),
      items: itemPayload(surface)
    })
  });

  if (!response.ok) {
    throw new Error(`Agent explanation request failed: ${response.status}`);
  }

  const payload = (await response.json()) as AgentExplanationResponse;
  return {
    source,
    created_at: new Date().toISOString(),
    overview: payload.overview ?? null,
    explanations: (payload.explanations ?? []).filter(
      (explanation) => explanation.review_item_id.trim() !== '' && explanation.text.trim() !== ''
    )
  };
}
