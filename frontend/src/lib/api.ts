export type Envelope<T> = { ok: boolean; data: T | null; error: { message: string } | null };

export type Evaluation = {
  id?: number;
  performance_run_id?: number | null;
  policy_version_id?: number;
  engine_version?: string;
  state?: "complete" | "not_evaluable" | "failed";
  non_evaluable_reason?: string | null;
  account?: Record<string, unknown>;
  result?: {
    engine_version?: string;
    account?: Record<string, unknown>;
    state?: string;
    source?: Record<string, unknown>;
    performance?: Record<string, unknown>;
    cash?: Record<string, unknown>;
    layers?: Array<Record<string, unknown>>;
    instruments?: Array<Record<string, unknown>>;
    review_queue?: Array<Record<string, unknown>>;
  };
  snapshot_id?: number;
};

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin" });
  const payload = await response.json() as Envelope<T>;
  if (!response.ok || !payload.ok) throw new Error(payload.error?.message ?? "자료를 불러오지 못했습니다.");
  return payload.data as T;
}
