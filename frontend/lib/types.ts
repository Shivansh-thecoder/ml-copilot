export type ProjectStatus = "active" | "completed" | "archived";

export interface Project {
  id: string;
  name: string;
  description: string;
  datasetName: string;
  lastOpened: string;
  status: ProjectStatus;
}

// Shape collected by ProjectModal's form and handed up to whoever owns
// the projects array — kept separate from `Project` since the caller
// (not the form) assigns id/lastOpened/status.
export interface NewProjectInput {
  name: string;
  description: string;
  datasetName: string;
}

export interface ActivityLogEntry {
  id: string;
  message: string;
  timestamp: string;
}

// Fields a user can edit from the Settings tab — narrower than `Project`
// since id/datasetName/lastOpened aren't user-editable.
export type ProjectSettingsUpdate = Pick<Project, "name" | "description" | "status">;

export type ProjectTabId = "overview" | "dataset" | "models" | "experiments" | "copilot" | "settings";

// Shared color vocabulary for Badge. Kept here (not in the component) so
// every status domain below can reference the same type without Badge
// needing to know what any of them mean.
export type BadgeTone = "green" | "blue" | "gray" | "amber" | "red";

export type PageId = "projects" | "datasets" | "models" | "experiments" | "settings";

export interface Dataset {
  id: string;
  name: string;
  rows: number;
  columns: number;
  targetColumn: string;
  sizeLabel: string;
  uploadedAt: string;
}

// Shape collected by UploadDatasetModal's form — narrower than `Dataset`
// since row/column/size/upload-date metadata isn't known until the
// backend has actually processed the uploaded file.
export interface NewDatasetInput {
  name: string;
  targetColumn: string;
}

// --- Copilot (backend/app/api/routers/copilot.py) ---
// Mirrors backend/langgraph/schemas/copilot.py's CopilotFinalResponse,
// field-for-field, translated from the wire format's snake_case (see
// services/copilotService.ts's mapping functions) to this codebase's
// camelCase convention — same treatment Dataset gets for
// target_column/size_label above.

export type CopilotRunStatus = "needs_clarification" | "completed" | "in_progress";

export interface CopilotResult {
  success: boolean;
  objective: string;
  datasetId: string;
  targetColumn: string | null;
  taskType: string | null;
  selectedModel: string | null;
  primaryMetric: string | null;
  validationPerformance: string;
  comparisonSummary: string;
  overfittingDiagnostic: string | null;
  tuningPerformed: boolean;
  tuningExplanation: string;
  savedModelLocation: string | null;
  caveats: string[];
  failureReason: string | null;
}

// What every /copilot/runs* endpoint returns — `result` is null while a
// run is `in_progress`/`needs_clarification`, and only `question` is set
// while `needs_clarification` (see CopilotTab.tsx for how the UI branches
// on `status`).
export interface CopilotRun {
  runId: string;
  status: CopilotRunStatus;
  question: string | null;
  result: CopilotResult | null;
}

// --- Saved models / prediction (backend/app/api/routers/models.py) ---
// Every field here originates in the fitted pipeline stored in the saved
// .joblib artifact, so nothing in the UI knows any dataset's column
// names — the form is built from whatever `features` comes back with.

export type FeatureKind = "numeric" | "categorical" | "unknown";

export interface ModelFeature {
  name: string;
  kind: FeatureKind;
  categories?: (string | number)[];
}

export interface SavedModel {
  datasetId: string;
  modelName: string;
  taskType: string;
  targetColumn: string;
  primaryMetric: string;
  testMetrics: Record<string, number>;
  savedAt: string;
}

export interface ModelSchema extends SavedModel {
  features: ModelFeature[];
  classes: (string | number)[] | null;
  supportsProbabilities: boolean;
}

export interface Prediction {
  row: number;
  prediction: string | number;
  probabilities: Record<string, number> | null;
  confidence: number | null;
}

export interface PredictionResult {
  taskType: string;
  targetColumn: string;
  predictions: Prediction[];
  ignoredColumns: string[];
  warnings: string[];
}
