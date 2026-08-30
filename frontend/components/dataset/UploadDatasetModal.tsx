"use client";

import { useState, type FormEvent } from "react";
import Modal from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Button from "@/components/ui/Button";
import { ApiError } from "@/services/api";
import { uploadDataset } from "@/services/datasetService";
import type { Dataset } from "@/lib/types";

interface UploadDatasetModalProps {
  onCancel: () => void;
  onUploaded: (dataset: Dataset) => void;
}

export default function UploadDatasetModal({ onCancel, onUploaded }: UploadDatasetModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = file !== null && name.trim().length > 0 && !isUploading;

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    // Pre-fill the name from the filename if the user hasn't typed one yet.
    if (selected && !name.trim()) {
      setName(selected.name.replace(/\.csv$/i, ""));
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file || !canSubmit) return;

    setIsUploading(true);
    setError(null);
    try {
      const dataset = await uploadDataset(file, name.trim(), targetColumn.trim());
      onUploaded(dataset);
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err.message : "Upload failed unexpectedly.");
      setIsUploading(false);
    }
  }

  return (
    <Modal title="Upload Dataset" onClose={onCancel}>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-gray-700 dark:text-gray-300">CSV File</span>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            disabled={isUploading}
            className="rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-2 file:py-1 file:text-sm dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100 dark:file:bg-gray-800"
          />
        </label>

        <Input
          label="Dataset Name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          disabled={isUploading}
        />
        <Input
          label="Target Column (optional)"
          value={targetColumn}
          onChange={(event) => setTargetColumn(event.target.value)}
          disabled={isUploading}
          placeholder="Leave blank to detect it later"
        />

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            {error}
          </div>
        )}

        <div className="mt-2 flex justify-end gap-2">
          <Button type="button" variant="secondary" onClick={onCancel} disabled={isUploading}>
            Cancel
          </Button>
          <Button type="submit" disabled={!canSubmit}>
            {isUploading ? "Uploading…" : "Upload"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
