// Client for the saved-model endpoints (backend/app/api/routers/models.py).
// Same split as copilotService.ts: the *Dto interfaces are the snake_case
// wire shapes FastAPI serializes, the exported functions hand back this
// app's camelCase types.

import { apiRequest } from "@/services/api";
import type { ModelSchema, PredictionResult, SavedModel } from "@/lib/types";

interface SavedModelDto {
  dataset_id: string;
  model_name: string;
  task_type: string;
  target_column: string;
  primary_metric: string;
  test_metrics: Record<string, number>;
  saved_at: string;
}

interface ModelSchemaDto extends SavedModelDto {
  features: { name: string; kind: "numeric" | "categorical" | "unknown"; categories?: (string | number)[] }[];
  classes: (string | number)[] | null;
  supports_probabilities: boolean;
}

interface PredictionResultDto {
  task_type: string;
  target_column: string;
  predictions: {
    row: number;
    prediction: string | number;
    probabilities?: Record<string, number>;
    confidence?: number;
  }[];
  ignored_columns: string[];
  warnings: string[];
}

function mapSavedModel(dto: SavedModelDto): SavedModel {
  return {
    datasetId: dto.dataset_id,
    modelName: dto.model_name,
    taskType: dto.task_type,
    targetColumn: dto.target_column,
    primaryMetric: dto.primary_metric,
    testMetrics: dto.test_metrics ?? {},
    savedAt: dto.saved_at,
  };
}

// GET /models — the datasets that currently have a saved model.
export async function listSavedModels(): Promise<SavedModel[]> {
  const dto = await apiRequest<{ count: number; models: SavedModelDto[] }>("/models");
  return dto.models.map(mapSavedModel);
}

// GET /models/{datasetId}/schema — the feature columns the saved model
// expects, read off its fitted pipeline. PredictModal builds its form
// from this rather than from any hardcoded column list.
export async function getModelSchema(datasetId: string): Promise<ModelSchema> {
  const dto = await apiRequest<ModelSchemaDto>(`/models/${datasetId}/schema`);
  return {
    ...mapSavedModel(dto),
    features: dto.features,
    classes: dto.classes,
    supportsProbabilities: dto.supports_probabilities,
  };
}

// POST /models/{datasetId}/predict — values are sent as the user typed
// them; the backend does the coercion and validation, since it's the
// side that knows each column's fitted type.
export async function predictWithModel(
  datasetId: string,
  rows: Record<string, string>[]
): Promise<PredictionResult> {
  const dto = await apiRequest<PredictionResultDto>(`/models/${datasetId}/predict`, {
    method: "POST",
    body: JSON.stringify({ rows }),
  });
  return {
    taskType: dto.task_type,
    targetColumn: dto.target_column,
    predictions: dto.predictions.map((p) => ({
      row: p.row,
      prediction: p.prediction,
      probabilities: p.probabilities ?? null,
      confidence: p.confidence ?? null,
    })),
    ignoredColumns: dto.ignored_columns,
    warnings: dto.warnings,
  };
}
