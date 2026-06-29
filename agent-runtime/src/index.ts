import {
  BuiltInAgent,
  CopilotRuntime,
  convertMessagesToVercelAISDKMessages,
  convertToolsToVercelAITools,
  createCopilotRuntimeHandler
} from "@copilotkit/runtime/v2";
import { openai } from "@ai-sdk/openai";
import { generateText, streamText, stepCountIs } from "ai";

import { systemPrompt } from "./prompts";

const port = Number(process.env.COPILOT_RUNTIME_PORT ?? "3001");
const model = (process.env.COPILOT_MODEL ?? "gpt-4o").replace(/^openai[/:]/, "");

const forbiddenActionWords = [
  "buy",
  "sell",
  "increase_position",
  "decrease_position",
  "rebalance_now",
  "calculate_order_size",
  "place_order"
];

type ReviewQueueExplanationRequest = {
  source?: "automatic" | "requested";
  evaluation_period?: {
    label?: string;
    start_date?: string;
    end_date?: string;
  };
  groups?: Array<{
    status?: string;
    summary?: string;
  }>;
  items?: Array<{
    id?: string;
    level?: string;
    name?: string;
    parent_layer?: string | null;
    status?: string;
    triggered_by?: string[];
    summary?: string;
    ips_interpretation?: string;
    verification_focus?: string;
    next_review_note?: string;
  }>;
};

type ModelExplanationResponse = {
  overview?: unknown;
  explanations?: Array<{
    review_item_id?: unknown;
    text?: unknown;
  }>;
};

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Content-Type": "application/json"
    }
  });
}

function hasForbiddenActionText(value: string) {
  const haystack = value.toLowerCase();
  return forbiddenActionWords.some((word) => {
    if (word === "buy" || word === "sell") {
      return new RegExp(`\\b${word}\\b`).test(haystack);
    }
    return haystack.includes(word);
  });
}

function parseJsonObject(text: string): ModelExplanationResponse {
  const trimmed = text.trim();
  const withoutFence = trimmed
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();
  const start = withoutFence.indexOf("{");
  const end = withoutFence.lastIndexOf("}");
  if (start < 0 || end < start) {
    throw new Error("Model response did not include a JSON object.");
  }
  return JSON.parse(withoutFence.slice(start, end + 1)) as ModelExplanationResponse;
}

function compactItemForPrompt(item: NonNullable<ReviewQueueExplanationRequest["items"]>[number]) {
  return {
    id: String(item.id ?? ""),
    level: item.level,
    name: item.name,
    parent_layer: item.parent_layer,
    status: item.status,
    triggered_by: item.triggered_by ?? [],
    summary: item.summary,
    ips_interpretation: item.ips_interpretation,
    verification_focus: item.verification_focus,
    next_review_note: item.next_review_note
  };
}

async function handleReviewQueueExplanations(request: Request) {
  if (request.method === "OPTIONS") {
    return jsonResponse({ ok: true });
  }
  if (request.method !== "POST") {
    return jsonResponse({ detail: "Method not allowed." }, 405);
  }
  if (!process.env.OPENAI_API_KEY) {
    return jsonResponse({ detail: "OPENAI_API_KEY is required for agent explanations." }, 503);
  }

  const payload = (await request.json().catch(() => null)) as ReviewQueueExplanationRequest | null;
  const items = (payload?.items ?? []).filter((item) => item.id && item.name).slice(0, 12);
  if (items.length === 0) {
    return jsonResponse({ overview: null, explanations: [] });
  }

  const allowedIds = new Set(items.map((item) => String(item.id)));
  const promptInput = {
    source: payload?.source === "requested" ? "requested" : "automatic",
    evaluation_period: payload?.evaluation_period,
    groups: payload?.groups ?? [],
    items: items.map(compactItemForPrompt)
  };

  let parsed: ModelExplanationResponse;
  try {
    const { text } = await generateText({
      model: openai(model),
      maxOutputTokens: 1400,
      messages: [
        {
          role: "system",
          content: [
            "You write concise Korean inspection explanations for IPS Pilot Review Queue boards.",
            "Return JSON only. Do not use Markdown.",
            "This is not investment advice and not an order instruction.",
            "Do not recommend buying, selling, execution, order sizing, or immediate rebalancing.",
            "Frame every explanation as data verification, thesis review, risk burden review, observation, or future regular-purchase policy review.",
            "Use only review_item_id values provided by the input."
          ].join(" ")
        },
        {
          role: "user",
          content: [
            "Create one short board overview and one short explanation per item.",
            "Each explanation must be 1-3 Korean sentences.",
            'Return shape: {"overview":"...","explanations":[{"review_item_id":"...","text":"..."}]}',
            JSON.stringify(promptInput)
          ].join("\n\n")
        }
      ]
    });
    parsed = parseJsonObject(text);
  } catch (error) {
    console.error("[review-copilot:explanations]", error);
    return jsonResponse({ detail: "Failed to generate Review Queue explanations." }, 502);
  }
  const overview = typeof parsed.overview === "string" && !hasForbiddenActionText(parsed.overview)
    ? parsed.overview.trim()
    : null;
  const explanations = (Array.isArray(parsed.explanations) ? parsed.explanations : [])
    .map((explanation) => ({
      review_item_id: String(explanation.review_item_id ?? ""),
      text: String(explanation.text ?? "").trim()
    }))
    .filter(
      (explanation) =>
        allowedIds.has(explanation.review_item_id) &&
        explanation.text.length > 0 &&
        !hasForbiddenActionText(explanation.text)
    );

  return jsonResponse({ overview, explanations });
}

const runtime = new CopilotRuntime({
  agents: {
    default: new BuiltInAgent({
      type: "aisdk",
      factory: ({ input, abortSignal }) => {
        const messages = [
          { role: "system" as const, content: systemPrompt() },
          ...convertMessagesToVercelAISDKMessages(input.messages, {
            forwardSystemMessages: true
          })
        ];
        const tools = convertToolsToVercelAITools(input.tools);
        return streamText({
          model: openai(model),
          messages,
          tools,
          abortSignal,
          stopWhen: stepCountIs(5)
        });
      }
    })
  }
});

const handler = createCopilotRuntimeHandler({
  runtime,
  basePath: "/copilotkit",
  cors: true
});

Bun.serve({
  port,
  fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/copilotkit/review-queue/explanations") {
      return handleReviewQueueExplanations(request);
    }
    return handler(request);
  }
});

console.log(`Review Copilot runtime listening on http://localhost:${port}/copilotkit`);
