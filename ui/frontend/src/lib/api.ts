const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface StartEvalRequest {
  target_model: string;
  concern: string;
  target_api_key?: string;
  target_api_base?: string;
  evaluator_model?: string;
  evaluator_api_key?: string;
  evaluator_api_base?: string;
  embedding_api_key?: string;
  num_rollouts?: number;
  max_turns?: number;
  max_concurrency?: number;
}

export interface EvalListItem {
  run_id: string;
  created_at: string;
  status: "running" | "completed" | "failed";
  target_model: string;
  concern: string;
  error?: string;
}

export interface BehaviorReport {
  behavior_name: string;
  description: string;
  num_rollouts: number;
  elicitation_rate: number;
  average_score: number;
  score_distribution: Record<string, number>;
  meta_judge_summary: string;
}

export interface FullReport {
  run_id: string;
  timestamp: string;
  target_model: string;
  evaluator_model: string;
  concern: string;
  num_rollouts_per_behavior: number;
  diversity: number;
  behavior_reports: BehaviorReport[];
  executive_summary: string;
  methodology_notes: string;
}

export interface EvalRecord extends EvalListItem {
  config?: StartEvalRequest;
  report?: FullReport;
}

export async function startEvaluation(req: StartEvalRequest): Promise<{ run_id: string; created_at: string }> {
  const res = await fetch(`${BASE}/api/evaluations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listEvaluations(): Promise<EvalListItem[]> {
  const res = await fetch(`${BASE}/api/evaluations`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getEvaluation(runId: string): Promise<EvalRecord> {
  const res = await fetch(`${BASE}/api/evaluations/${runId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteEvaluation(runId: string): Promise<void> {
  await fetch(`${BASE}/api/evaluations/${runId}`, { method: "DELETE" });
}

export function streamProgress(
  runId: string,
  onStage: (data: StageEvent) => void,
  onComplete: (data: CompleteEvent) => void,
  onError: (msg: string) => void,
): EventSource {
  const es = new EventSource(`${BASE}/api/evaluations/${runId}/stream`);

  es.addEventListener("stage", (e) => {
    try { onStage(JSON.parse((e as MessageEvent).data)); } catch {}
  });
  es.addEventListener("complete", (e) => {
    try {
      onComplete(JSON.parse((e as MessageEvent).data));
      es.close();
    } catch {}
  });
  es.addEventListener("error", (e) => {
    try {
      const d = JSON.parse((e as MessageEvent).data ?? "{}");
      onError(d.message ?? "Unknown error");
    } catch {
      onError("Stream connection error");
    }
    es.close();
  });

  return es;
}

export interface StageEvent {
  stage: number;
  name: string;
  status: "running" | "complete";
  detail: string;
  behaviors?: Array<{ name: string; description: string; modality: string }>;
}

export interface CompleteEvent {
  run_id: string;
  report: FullReport;
}
