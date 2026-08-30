"use client";

import { useState, type FormEvent } from "react";
import Modal from "@/components/ui/Modal";
import Input from "@/components/ui/Input";
import Textarea from "@/components/ui/Textarea";
import Button from "@/components/ui/Button";
import type { NewProjectInput } from "@/lib/types";

interface ProjectModalProps {
  onCancel: () => void;
  onCreate: (data: NewProjectInput) => void;
}

// Form fields are controlled (value + onChange) and local to this modal
// — they're only meaningful while the modal is open. ProjectsPage
// unmounts this component on cancel/create (conditional rendering),
// which resets the fields for free the next time it opens, so there's no
// manual "clear the form" step to write.
export default function ProjectModal({ onCancel, onCreate }: ProjectModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [datasetName, setDatasetName] = useState("");

  const canSubmit = name.trim().length > 0;

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    onCreate({ name: name.trim(), description: description.trim(), datasetName: datasetName.trim() });
  }

  return (
    <Modal title="New Project" onClose={onCancel}>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <Input
          label="Project Name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          autoFocus
        />
        <Textarea
          label="Description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          rows={2}
        />
        <Input
          label="Dataset Name"
          value={datasetName}
          onChange={(event) => setDatasetName(event.target.value)}
        />

        <div className="mt-2 flex justify-end gap-2">
          <Button type="button" variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={!canSubmit}>
            Create
          </Button>
        </div>
      </form>
    </Modal>
  );
}
