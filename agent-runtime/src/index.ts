import {
  BuiltInAgent,
  CopilotRuntime,
  convertInputToTanStackAI,
  createCopilotRuntimeHandler
} from "@copilotkit/runtime/v2";
import { chat } from "@tanstack/ai";
import { openaiText } from "@tanstack/ai-openai";

import { systemPrompt } from "./prompts";

const port = Number(process.env.COPILOT_RUNTIME_PORT ?? "3001");
const model = (process.env.COPILOT_MODEL ?? "gpt-4o") as Parameters<typeof openaiText>[0];

const runtime = new CopilotRuntime({
  agents: {
    default: new BuiltInAgent({
      type: "tanstack",
      factory: ({ input, abortController }) => {
        const { messages, systemPrompts } = convertInputToTanStackAI(input);
        systemPrompts.unshift(systemPrompt());
        return chat({
          adapter: openaiText(model),
          messages,
          systemPrompts,
          abortController
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
  fetch: handler
});

console.log(`Review Copilot runtime listening on http://localhost:${port}/copilotkit`);
