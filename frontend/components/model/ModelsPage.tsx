"use client";

import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import Button from "@/components/ui/Button";
import EmptyState from "@/components/ui/EmptyState";
import SectionHeader from "@/components/ui/SectionHeader";
import Table from "@/components/ui/Table";
import PredictModal from "@/components/model/PredictModal";
import { ApiError } from "@/services/api";
import { listSavedModels } from "@/services/modelService";
import type { SavedModel } from "@/lib/types";

interface ModelsPageProps {
  searchQuery: string;
}

export default function ModelsPage({ searchQuery }: ModelsPageProps) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testingModel, setTestingModel] = useState<SavedModel | null>(null);

  useEffect(() => {
    let cancelled = false;
    listSavedModels()
      .then((data) => {
        if (!cancelled) setModels(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load models.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleModels = models.filter((model) =>
    `${model.modelName} ${model.targetColumn}`.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <section className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-5xl">
        <SectionHeader
          title="Models"
          description="Models the ML Copilot trained and saved. Test one with your own feature values."
        />

        {error && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            {error}
          </div>
        )}

        <div className="mt-8">
          {isLoading ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">Loading models…</p>
          ) : visibleModels.length === 0 ? (
            <EmptyState
              message={
                models.length === 0
                  ? "No saved models yet — run the Copilot on a dataset to train and save one."
                  : "No models match your search."
              }
            />
          ) : (
            <Table
              rows={visibleModels}
              getRowKey={(model) => model.datasetId}
              columns={[
                {
                  key: "model",
                  header: "Model",
                  render: (model) => (
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {model.modelName}
                    </span>
                  ),
                },
                { key: "task", header: "Task", render: (model) => model.taskType },
                { key: "target", header: "Target", render: (model) => model.targetColumn },
                {
                  key: "score",
                  header: "Test Score",
                  render: (model) => {
                    const score = model.testMetrics[model.primaryMetric];
                    return score === undefined
                      ? "—"
                      : `${model.primaryMetric} ${score.toFixed(3)}`;
                  },
                },
                {
                  key: "saved",
                  header: "Saved",
                  render: (model) => new Date(model.savedAt).toLocaleDateString(),
                },
                {
                  key: "actions",
                  header: "",
                  render: (model) => (
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={() => setTestingModel(model)}
                      icon={<FlaskConical className="h-4 w-4" aria-hidden />}
                    >
                      Test Model
                    </Button>
                  ),
                },
              ]}
            />
          )}
        </div>
      </div>

      {testingModel && (
        <PredictModal model={testingModel} onClose={() => setTestingModel(null)} />
      )}
    </section>
  );
}
