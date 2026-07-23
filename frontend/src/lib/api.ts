export type Envelope<T> = { ok: boolean; data: T | null; error: { message: string } | null };
export type JsonObject = Record<string, unknown>;

export type PerformanceRun = JsonObject & {
  id?: number;
  state?: string;
  points?: Array<JsonObject>;
};

export type AccountSummary = JsonObject & {
  total_value_krw?: number | null;
  invested_value_krw?: number | null;
  cash_value_krw?: number | null;
  cash_weight_gross?: number | null;
  investment_principal_krw?: number | null;
  account_profit_krw?: number | null;
  account_return?: number | null;
};

export type InspectionSuggestion = {
  code?: string;
  label?: string;
};

export type InspectionItem = JsonObject & {
  priority?: "P1" | "P2" | "P3" | "P4" | null;
  priority_label?: string | null;
  queue_class?: "blocking" | "adjustment" | "observation";
  suggestion?: InspectionSuggestion | null;
  status?: "OK" | "Watch" | "Review" | "Action" | string;
  kind?: string;
  identity?: string;
  current?: number | null;
  minimum?: number | null;
  target?: number | null;
  maximum?: number | null;
  gap?: number | null;
  denominator?: string | null;
};

export type InspectionResult = {
  engine_version?: string;
  allocation_state?: "complete" | "partial" | "not_evaluable" | string;
  allocation_reason?: string | null;
  account?: AccountSummary;
  account_profit_loss?: JsonObject;
  source?: Record<string, unknown>;
  performance?: JsonObject;
  cash?: InspectionItem | null;
  layers?: InspectionItem[];
  instruments?: InspectionItem[];
  adjustment_suggestions?: InspectionItem[];
  review_queue?: InspectionItem[];
  evidence_refs?: JsonObject;
};

export type InspectionData = {
  evaluation: Evaluation | null;
  contract_supported: boolean;
};

export type Evaluation = {
  id?: number;
  performance_run_id?: number | null;
  policy_version_id?: number;
  engine_version?: string;
  state?: "complete" | "not_evaluable" | "failed";
  non_evaluable_reason?: string | null;
  profile_snapshot?: Array<Record<string, unknown>>;
  account?: AccountSummary;
  result?: InspectionResult;
  market_evidence?: JsonObject;
  snapshot_id?: number;
};

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin" });
  const payload = await response.json() as Envelope<T>;
  if (!response.ok || !payload.ok) throw new Error(payload.error?.message ?? "자료를 불러오지 못했습니다.");
  return payload.data as T;
}
