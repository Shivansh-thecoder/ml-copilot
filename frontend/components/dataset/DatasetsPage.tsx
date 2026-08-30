"use client";

import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import Button from "@/components/ui/Button";
import SectionHeader from "@/components/ui/SectionHeader";
import Table from "@/components/ui/Table";
import EmptyState from "@/components/ui/EmptyState";
import UploadDatasetModal from "@/components/dataset/UploadDatasetModal";
import { ApiError } from "@/services/api";
import { deleteDataset, listDatasets } from "@/services/datasetService";
import type { Dataset } from "@/lib/types";

interface DatasetsPageProps {
  searchQuery: string;
}

export default function DatasetsPage({ searchQuery }: DatasetsPageProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDatasets()
      .then((data) => {
        if (!cancelled) setDatasets(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load datasets.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleDatasets = datasets.filter((dataset) =>
    dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  function handleUploaded(dataset: Dataset) {
    setDatasets((prev) => [...prev, dataset]);
    setIsUploadModalOpen(false);
  }

  async function handleDelete(id: string) {
    setDeletingId(id);
    try {
      await deleteDataset(id);
      setDatasets((prev) => prev.filter((d) => d.id !== id));
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err.message : "Could not delete dataset.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <SectionHeader
          title="Datasets"
          description="Datasets uploaded for use in your ML projects."
          action={
            <Button
              type="button"
              onClick={() => setIsUploadModalOpen(true)}
              icon={<Plus className="h-4 w-4" aria-hidden />}
            >
              Upload Dataset
            </Button>
          }
        />

        {error && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            {error}
          </div>
        )}

        <div className="mt-8">
          {isLoading ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">Loading datasets…</p>
          ) : visibleDatasets.length === 0 ? (
            <EmptyState
              message={
                datasets.length === 0
                  ? "No datasets yet — upload one to get started."
                  : "No datasets match your search."
              }
            />
          ) : (
            <Table
              rows={visibleDatasets}
              getRowKey={(dataset) => dataset.id}
              columns={[
                {
                  key: "name",
                  header: "Name",
                  render: (dataset) => (
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {dataset.name}
                    </span>
                  ),
                },
                { key: "rows", header: "Rows", render: (dataset) => dataset.rows.toLocaleString() },
                { key: "columns", header: "Columns", render: (dataset) => dataset.columns.toString() },
                { key: "target", header: "Target", render: (dataset) => dataset.targetColumn || "—" },
                { key: "size", header: "Size", render: (dataset) => dataset.sizeLabel },
                { key: "uploaded", header: "Upload Date", render: (dataset) => dataset.uploadedAt },
                {
                  key: "actions",
                  header: "",
                  render: (dataset) => (
                    <button
                      type="button"
                      onClick={() => handleDelete(dataset.id)}
                      disabled={deletingId === dataset.id}
                      title="Delete dataset"
                      className="text-gray-400 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" aria-hidden />
                    </button>
                  ),
                },
              ]}
            />
          )}
        </div>
      </div>

      {isUploadModalOpen && (
        <UploadDatasetModal onCancel={() => setIsUploadModalOpen(false)} onUploaded={handleUploaded} />
      )}
    </section>
  );
}
