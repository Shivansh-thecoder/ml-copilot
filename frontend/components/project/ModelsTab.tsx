import EmptyState from "@/components/ui/EmptyState";

// No per-project model list yet: a Project isn't linked to a specific
// dataset_id on the backend (just a display name), so there's no way to
// scope a real model query to "this project" without guessing. Trained
// models are visible in the Copilot tab's result after a run.
export default function ModelsTab() {
  return (
    <EmptyState message="No models yet — run the Copilot from this project's Copilot tab to train and save one." />
  );
}
