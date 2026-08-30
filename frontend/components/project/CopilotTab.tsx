"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import Button from "@/components/ui/Button";
import Select from "@/components/ui/Select";
import Textarea from "@/components/ui/Textarea";
import Field from "@/components/ui/Field";
import Badge from "@/components/ui/Badge";
import EmptyState from "@/components/ui/EmptyState";
import { listDatasets } from "@/services/datasetService";
import { continueCopilotRun, startCopilotRun } from "@/services/copilotService";
import { ApiError } from "@/services/api";
import type { CopilotRun, Dataset } from "@/lib/types";

// Local UI state machine -- NOT the backend's own workflow state
// (CopilotState, backend/langgraph/state.py). This only tracks what the
// PAGE is doing right now: is a request in flight, did it fail, is it
// waiting on the user for clarification, or does it have a finished
// result to show. `CopilotRun.status` (from the backend) drives the
// "clarification" vs "result" branches once a request succeeds.
type ViewState = "idle" | "loading" | "clarification" | "result" | "error";

// Reused by both the "start" and "continue" error paths -- there's only
// one way this tab surfaces a failure to the user.
function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
      {message}
    </div>
  );
}

export default function CopilotTab() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [objective, setObjective] = useState("");

  const [viewState, setViewState] = useState<ViewState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [run, setRun] = useState<CopilotRun | null>(null);
  const [clarificationAnswer, setClarificationAnswer] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Datasets come from the SAME backend endpoint the Datasets page uses
  // (services/datasetService.ts -> GET /datasets) -- not a second dataset
  // system, just this tab's own copy of "what's available to pick from."
  useEffect(() => {
    let cancelled = false;
    listDatasets()
      .then((data) => {
        if (cancelled) return;
        setDatasets(data);
        if (data.length > 0) setSelectedDatasetId((prev) => prev || data[0].id);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetsError(error instanceof ApiError ? error.message : "Could not load datasets.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleStart() {
    if (!selectedDatasetId || !objective.trim()) return;
    setViewState("loading");
    setErrorMessage("");
    setElapsedSeconds(0);
    try {
      const result = await startCopilotRun(selectedDatasetId, objective.trim());
      setRun(result);
      setViewState(result.status === "needs_clarification" ? "clarification" : "result");
    } catch (error: unknown) {
      setErrorMessage(error instanceof ApiError ? error.message : "The Copilot run failed unexpectedly.");
      setViewState("error");
    }
  }

  async function handleContinue() {
    if (!run || !clarificationAnswer.trim()) return;
    setViewState("loading");
    setErrorMessage("");
    setElapsedSeconds(0);
    try {
      // Resumes the SAME run (run.runId) -- the backend continues its
      // checkpointed LangGraph state from where it paused, not a fresh
      // run. See services/copilotService.ts's continueCopilotRun.
      const result = await continueCopilotRun(run.runId, clarificationAnswer.trim());
      setRun(result);
      setClarificationAnswer("");
      setViewState(result.status === "needs_clarification" ? "clarification" : "result");
    } catch (error: unknown) {
      setErrorMessage(error instanceof ApiError ? error.message : "Could not submit the answer.");
      setViewState("error");
    }
  }

  function handleStartOver() {
    setRun(null);
    setErrorMessage("");
    setClarificationAnswer("");
    setViewState("idle");
  }

  const isBusy = viewState === "loading";

  // A real elapsed-time counter, not a fake progress bar -- there's no
  // meaningful "% done" for a multi-stage LLM+ML pipeline, but a ticking
  // clock is honest evidence the request is still alive, not frozen.
  // Reset happens where a request actually starts (handleStart/
  // handleContinue), not here -- this effect only runs the ticker.
  useEffect(() => {
    if (!isBusy) return;
    const interval = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [isBusy]);

  return (
    <div className="flex flex-col gap-6">
      {datasetsError && <ErrorBanner message={datasetsError} />}

      {datasets.length === 0 && !datasetsError ? (
        <EmptyState message="No datasets uploaded yet -- upload one from the Datasets page first." />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Select
              label="Dataset"
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={isBusy || viewState === "clarification"}
            >
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </Select>
          </div>

          <Textarea
            label="ML objective"
            placeholder="e.g. Build a model to predict customer churn."
            rows={3}
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            disabled={isBusy || viewState === "clarification" || viewState === "result"}
          />

          {viewState === "idle" || viewState === "error" ? (
            <div>
              <Button
                type="button"
                onClick={handleStart}
                disabled={!selectedDatasetId || !objective.trim()}
              >
                Start Copilot
              </Button>
            </div>
          ) : null}

          {errorMessage && <ErrorBanner message={errorMessage} />}

          {isBusy && (
            <div className="flex items-center gap-2 rounded-md border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-600 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-400">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              <span>
                The Copilot is working -- inspecting the dataset, training and evaluating models.
                This can take a minute or more; there is no progress percentage to show, since real
                model training is happening. ({elapsedSeconds}s elapsed)
              </span>
            </div>
          )}

          {viewState === "clarification" && run?.question && (
            <div className="flex flex-col gap-3 rounded-md border border-amber-200 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950">
              <p className="text-sm text-amber-900 dark:text-amber-200">{run.question}</p>
              <div className="flex items-end gap-2">
                <div className="flex-1">
                  <Textarea
                    label="Your answer"
                    rows={1}
                    value={clarificationAnswer}
                    onChange={(e) => setClarificationAnswer(e.target.value)}
                  />
                </div>
                <Button type="button" onClick={handleContinue} disabled={!clarificationAnswer.trim()}>
                  Submit
                </Button>
              </div>
            </div>
          )}

          {viewState === "result" && run?.result && <CopilotResultView result={run.result} />}

          {(viewState === "result" || viewState === "error") && (
            <div>
              <Button type="button" variant="secondary" onClick={handleStartOver}>
                Start a new run
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function CopilotResultView({ result }: { result: NonNullable<CopilotRun["result"]> }) {
  if (!result.success) {
    return (
      <div className="flex flex-col gap-2">
        <Badge label="Did not complete" tone="red" />
        <ErrorBanner message={result.failureReason ?? "The workflow could not complete, for an unknown reason."} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 rounded-lg border border-gray-200 p-5 dark:border-gray-800">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">Result</h3>
        <Badge label="Completed" tone="green" />
      </div>

      <dl className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <Field label="Objective" value={result.objective} className="sm:col-span-2" />
        <Field label="Dataset" value={result.datasetId} />
        <Field label="Target column" value={result.targetColumn ?? "—"} />
        <Field label="Task" value={result.taskType ?? "—"} />
        <Field label="Selected model" value={result.selectedModel ?? "—"} />
        <Field label="Primary metric" value={result.primaryMetric ?? "—"} />
        <Field label="Validation / CV performance" value={result.validationPerformance} className="sm:col-span-2" />
        <Field label="Model comparison" value={result.comparisonSummary} className="sm:col-span-2" />
        <Field label="Overfitting / underfitting" value={result.overfittingDiagnostic ?? "—"} />
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
            Tuning
          </dt>
          <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
            <Badge label={result.tuningPerformed ? "Performed" : "Skipped"} tone={result.tuningPerformed ? "blue" : "gray"} />
            <span className="ml-2 text-gray-600 dark:text-gray-400">{result.tuningExplanation}</span>
          </dd>
        </div>
        <Field label="Saved model" value={result.savedModelLocation ?? "Not saved"} className="sm:col-span-2" />
      </dl>

      {result.caveats.length > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950">
          <p className="text-xs font-medium uppercase tracking-wide text-amber-700 dark:text-amber-400">
            Caveats
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-900 dark:text-amber-200">
            {result.caveats.map((caveat, index) => (
              <li key={index}>{caveat}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
