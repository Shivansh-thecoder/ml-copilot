// Client for the Copilot endpoints (backend/app/api/routers/copilot.py).
// Three functions, matching the three endpoints exactly — no fetch()
// calls anywhere outside this service layer (see CopilotTab.tsx, which
// only ever calls these).

import { apiRequest } from "@/services/api";
import type { CopilotResult, CopilotRun, CopilotRunStatus } from "@/lib/types";

// Wire shapes from backend/app/api/routers/copilot.py's `_summarize`/
// CopilotFinalResponse — snake_case, exactly as FastAPI/Pydantic
// serializes them. Kept distinct from the CopilotRun/CopilotResult types
// in lib/types.ts on purpose: those are this app's camelCase convention
// (matching Dataset/Project), these are what's actually on the wire.
interface CopilotResultDto {
  success: boolean;
  objective: string;
  dataset_id: string;
  target_column: string | null;
  task_type: string | null;
  selected_model: string | null;
  primary_metric: string | null;
  validation_performance: string;
  comparison_summary: string;
  overfitting_diagnostic: string | null;
  tuning_performed: boolean;
  tuning_explanation: string;
  saved_model_location: string | null;
  caveats: string[];
  failure_reason: string | null;
}

interface CopilotRunDto {
  run_id: string;
  status: CopilotRunStatus;
  question: string | null;
  result: CopilotResultDto | null;
}

function mapResult(dto: CopilotResultDto): CopilotResult {
  return {
    success: dto.success,
    objective: dto.objective,
    datasetId: dto.dataset_id,
    targetColumn: dto.target_column,
    taskType: dto.task_type,
    selectedModel: dto.selected_model,
    primaryMetric: dto.primary_metric,
    validationPerformance: dto.validation_performance,
    comparisonSummary: dto.comparison_summary,
    overfittingDiagnostic: dto.overfitting_diagnostic,
    tuningPerformed: dto.tuning_performed,
    tuningExplanation: dto.tuning_explanation,
    savedModelLocation: dto.saved_model_location,
    caveats: dto.caveats,
    failureReason: dto.failure_reason,
  };
}

function mapRun(dto: CopilotRunDto): CopilotRun {
  return {
    runId: dto.run_id,
    status: dto.status,
    question: dto.question,
    result: dto.result ? mapResult(dto.result) : null,
  };
}

// POST /copilot/runs — starts a new run. Blocks (the backend itself
// blocks on graph.invoke()) until the workflow pauses for clarification
// or completes; see CopilotTab.tsx for the loading state shown while
// this is in flight.
export async function startCopilotRun(datasetId: string, objective: string): Promise<CopilotRun> {
  const dto = await apiRequest<CopilotRunDto>("/copilot/runs", {
    method: "POST",
    body: JSON.stringify({ dataset_id: datasetId, objective }),
  });
  return mapRun(dto);
}

// POST /copilot/runs/{id}/continue — resumes a run paused for
// clarification with the user's answer. Does NOT restart the workflow;
// the backend resumes LangGraph's checkpointed state from exactly where
// it paused (see graphs/copilot_graph.py's clarification_node).
export async function continueCopilotRun(runId: string, answer: string): Promise<CopilotRun> {
  const dto = await apiRequest<CopilotRunDto>(`/copilot/runs/${runId}/continue`, {
    method: "POST",
    body: JSON.stringify({ answer }),
  });
  return mapRun(dto);
}

// GET /copilot/runs/{id} — current status/result without advancing the run.
export async function getCopilotRun(runId: string): Promise<CopilotRun> {
  const dto = await apiRequest<CopilotRunDto>(`/copilot/runs/${runId}`);
  return mapRun(dto);
}
