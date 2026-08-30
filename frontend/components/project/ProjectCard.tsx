import Badge from "@/components/ui/Badge";
import { PROJECT_STATUS_META } from "@/lib/statusMeta";
import type { Project } from "@/lib/types";

interface ProjectCardProps {
  project: Project;
  isSelected: boolean;
  onSelect: () => void;
}

// Doesn't reuse the generic ui/Card primitive: Card renders a static
// <div>, but a project card needs button semantics (click + keyboard
// activation) and a selected-state border. Duplicating a few border/
// padding classes here was simpler than teaching Card about selection.
export default function ProjectCard({ project, isSelected, onSelect }: ProjectCardProps) {
  const statusMeta = PROJECT_STATUS_META[project.status];

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={isSelected}
      className={`rounded-lg border p-4 text-left transition-colors ${
        isSelected
          ? "border-gray-900 bg-gray-50 dark:border-gray-100 dark:bg-gray-900"
          : "border-gray-200 bg-white hover:border-gray-300 dark:border-gray-800 dark:bg-gray-950 dark:hover:border-gray-700"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-medium text-gray-900 dark:text-gray-100">{project.name}</h3>
        <Badge label={statusMeta.label} tone={statusMeta.tone} />
      </div>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{project.description}</p>
      <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
        Last opened {project.lastOpened}
      </p>
    </button>
  );
}
