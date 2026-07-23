export type Envelope<T> = { ok: boolean; data: T | null; error: { message: string } | null };
export type JsonObject = Record<string, unknown>;

export type PerformanceRun = JsonObject & {
  id?: number;
  state?: string;
  points?: Array<JsonObject>;
};

export type Evaluation = {
  id?: number;
  performance_run_id?: number | null;
  policy_version_id?: number;
  engine_version?: string;
  state?: "complete" | "not_evaluable" | "failed";
  non_evaluable_reason?: string | null;
  profile_snapshot?: Array<Record<string, unknown>>;
  account?: JsonObject;
  result?: {
    engine_version?: string;
    account?: JsonObject;
    state?: string;
    source?: Record<string, unknown>;
    performance?: JsonObject;
    cash?: JsonObject;
    layers?: Array<JsonObject>;
    instruments?: Array<JsonObject>;
    review_queue?: Array<JsonObject>;
  };
  snapshot_id?: number;
};

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin" });
  const payload = await response.json() as Envelope<T>;
  if (!response.ok || !payload.ok) throw new Error(payload.error?.message ?? "자료를 불러오지 못했습니다.");
  return payload.data as T;
}
