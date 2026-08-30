import EmptyState from "@/components/ui/EmptyState";

// See ModelsTab.tsx: no per-project experiment list without a real
// project -> dataset_id link. Real results are in the Copilot tab.
export default function ExperimentsTab() {
  return (
    <EmptyState message="No experiments yet — run the Copilot from this project's Copilot tab to record one." />
  );
}
