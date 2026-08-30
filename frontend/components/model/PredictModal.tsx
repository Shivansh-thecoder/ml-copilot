"use client";

import { useEffect, useState } from "react";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import Modal from "@/components/ui/Modal";
import Select from "@/components/ui/Select";
import { ApiError } from "@/services/api";
import { getModelSchema, predictWithModel } from "@/services/modelService";
import type { ModelSchema, PredictionResult, SavedModel } from "@/lib/types";

interface PredictModalProps {
  model: SavedModel;
  onClose: () => void;
}

export default function PredictModal({ model, onClose }: PredictModalProps) {
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoadingSchema, setIsLoadingSchema] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The form's fields come from the model itself, not from anything this
  // component knows about the dataset — a different saved model renders a
  // different form with no code change here.
  useEffect(() => {
    let cancelled = false;
    getModelSchema(model.datasetId)
      .then((loaded) => {
        if (cancelled) return;
        setSchema(loaded);
        setValues(
          Object.fromEntries(
            loaded.features.map((feature) => [
              feature.name,
              feature.kind === "categorical" && feature.categories?.length
                ? String(feature.categories[0])
                : "",
            ])
          )
        );
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Could not load this model's features.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoadingSchema(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model.datasetId]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);
    setIsPredicting(true);
    try {
      setResult(await predictWithModel(model.datasetId, [values]));
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err.message : "Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }

  const prediction = result?.predictions[0] ?? null;

  return (
    <Modal title={`Test ${model.modelName}`} onClose={onClose}>
      {isLoadingSchema ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading model features…</p>
      ) : schema ? (
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Enter feature values to predict{" "}
            <span className="font-medium text-gray-900 dark:text-gray-100">{schema.targetColumn}</span>.
          </p>

          <div className="grid max-h-72 grid-cols-2 gap-3 overflow-y-auto pr-1">
            {schema.features.map((feature) =>
              feature.categories?.length ? (
                <Select
                  key={feature.name}
                  label={feature.name}
                  value={values[feature.name] ?? ""}
                  onChange={(e) => setValues((prev) => ({ ...prev, [feature.name]: e.target.value }))}
                >
                  {feature.categories.map((category) => (
                    <option key={String(category)} value={String(category)}>
                      {String(category)}
                    </option>
                  ))}
                </Select>
              ) : (
                <Input
                  key={feature.name}
                  label={feature.name}
                  value={values[feature.name] ?? ""}
                  // `inputMode` (not type="number") so the value stays a
                  // plain string: the backend owns numeric validation, and
                  // a number input silently blanks out unparseable text.
                  inputMode={feature.kind === "numeric" ? "decimal" : "text"}
                  placeholder={feature.kind === "numeric" ? "number" : "text"}
                  onChange={(e) => setValues((prev) => ({ ...prev, [feature.name]: e.target.value }))}
                />
              )
            )}
          </div>

          {error && (
            <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
              {error}
            </p>
          )}

          {prediction && (
            <div className="rounded-md border border-gray-200 bg-gray-50 p-4 dark:border-gray-800 dark:bg-gray-950">
              <p className="text-xs font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
                Predicted {result?.targetColumn}
              </p>
              <p className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100">
                {typeof prediction.prediction === "number"
                  ? Number(prediction.prediction.toFixed(4))
                  : prediction.prediction}
              </p>

              {prediction.confidence !== null && (
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </p>
              )}

              {prediction.probabilities && (
                <dl className="mt-3 flex flex-col gap-1">
                  {Object.entries(prediction.probabilities).map(([className, probability]) => (
                    <div key={className} className="flex items-center gap-2 text-xs">
                      <dt className="w-20 shrink-0 truncate text-gray-500 dark:text-gray-400">
                        {className}
                      </dt>
                      <dd className="flex flex-1 items-center gap-2">
                        <span className="h-1.5 flex-1 rounded-full bg-gray-200 dark:bg-gray-800">
                          <span
                            className="block h-1.5 rounded-full bg-gray-900 dark:bg-gray-100"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </span>
                        <span className="w-12 text-right text-gray-600 dark:text-gray-400">
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </dd>
                    </div>
                  ))}
                </dl>
              )}

              {result && result.warnings.length > 0 && (
                <ul className="mt-3 list-disc pl-4 text-xs text-amber-700 dark:text-amber-500">
                  {result.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="flex justify-end gap-2">
            <Button type="button" variant="secondary" onClick={onClose}>
              Close
            </Button>
            <Button type="submit" disabled={isPredicting}>
              {isPredicting ? "Predicting…" : "Predict"}
            </Button>
          </div>
        </form>
      ) : (
        <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
          {error ?? "Could not load this model."}
        </p>
      )}
    </Modal>
  );
}
