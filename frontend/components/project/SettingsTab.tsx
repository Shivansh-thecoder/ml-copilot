"use client";

import { useState, type FormEvent } from "react";
import Input from "@/components/ui/Input";
import Textarea from "@/components/ui/Textarea";
import Select from "@/components/ui/Select";
import Button from "@/components/ui/Button";
import type { Project, ProjectSettingsUpdate, ProjectStatus } from "@/lib/types";

interface SettingsTabProps {
  project: Project;
  onSave: (updates: ProjectSettingsUpdate) => void;
  onDelete: () => void;
}

const STATUS_OPTIONS: ProjectStatus[] = ["active", "completed", "archived"];

// The form fields are a local draft, not the shared project data itself
// — they only become real (shared, visible in Overview/ProjectCard) once
// Save calls onSave. Seeding useState from `project` only runs on mount,
// which is safe here because ProjectWorkspace remounts this tab whenever
// the selected project changes, so the draft never goes stale.
export default function SettingsTab({ project, onSave, onDelete }: SettingsTabProps) {
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description);
  const [status, setStatus] = useState<ProjectStatus>(project.status);
  const [justSaved, setJustSaved] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onSave({ name: name.trim() || project.name, description: description.trim(), status });
    setJustSaved(true);
  }

  return (
    <form onSubmit={handleSubmit} className="flex max-w-md flex-col gap-4">
      <Input
        label="Project Name"
        value={name}
        onChange={(event) => {
          setName(event.target.value);
          setJustSaved(false);
        }}
      />

      <Textarea
        label="Description"
        value={description}
        onChange={(event) => {
          setDescription(event.target.value);
          setJustSaved(false);
        }}
        rows={3}
      />

      <Select
        label="Status"
        value={status}
        onChange={(event) => {
          setStatus(event.target.value as ProjectStatus);
          setJustSaved(false);
        }}
      >
        {STATUS_OPTIONS.map((option) => (
          <option key={option} value={option}>
            {option.charAt(0).toUpperCase() + option.slice(1)}
          </option>
        ))}
      </Select>

      <div className="mt-2 flex items-center gap-3">
        <Button type="submit">Save</Button>
        {justSaved && <span className="text-sm text-green-600 dark:text-green-400">Saved</span>}
      </div>

      <div className="mt-6 border-t border-gray-200 pt-6 dark:border-gray-800">
        {confirmingDelete ? (
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-700 dark:text-gray-300">Delete this project?</span>
            <Button type="button" variant="secondary" onClick={onDelete} className="text-red-600 dark:text-red-400">
              Confirm Delete
            </Button>
            <Button type="button" variant="secondary" onClick={() => setConfirmingDelete(false)}>
              Cancel
            </Button>
          </div>
        ) : (
          <Button type="button" variant="secondary" onClick={() => setConfirmingDelete(true)}>
            Delete Project
          </Button>
        )}
      </div>
    </form>
  );
}
