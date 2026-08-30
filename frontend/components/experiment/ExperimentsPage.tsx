import SectionHeader from "@/components/ui/SectionHeader";
import EmptyState from "@/components/ui/EmptyState";

// Same reasoning as components/model/ModelsPage.tsx: there is no
// cross-project "list all experiments" backend endpoint. Experiment
// history is per-dataset (backend/mcp/storage/experiments.db, surfaced
// via the Copilot's get_experiment_results step) -- see a project's
// Copilot tab after a run for real results, rather than a fabricated
// list here.
export default function ExperimentsPage() {
  return (
    <section className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <SectionHeader title="Experiments" description="Training runs recorded by the ML Copilot." />

        <div className="mt-8">
          <EmptyState message="No experiments yet — run the Copilot on a project's dataset to record one. Results appear in that project's Copilot tab." />
        </div>
      </div>
    </section>
  );
}
