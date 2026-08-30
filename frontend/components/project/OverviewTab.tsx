import Field from "@/components/ui/Field";
import Badge from "@/components/ui/Badge";
import { PROJECT_STATUS_META } from "@/lib/statusMeta";
import type { Project } from "@/lib/types";

interface OverviewTabProps {
  project: Project;
}

// Renders the `project` prop directly — no fetching of its own.
export default function OverviewTab({ project }: OverviewTabProps) {
  const statusMeta = PROJECT_STATUS_META[project.status];

  return (
    <dl className="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <Field label="Name" value={project.name} />
      <Field label="Dataset" value={project.datasetName} />

      {/* Not a Field: the value is a badge, not plain text. */}
      <div>
        <dt className="text-xs font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
          Status
        </dt>
        <dd className="mt-1">
          <Badge label={statusMeta.label} tone={statusMeta.tone} />
        </dd>
      </div>

      <Field label="Last Opened" value={project.lastOpened} />
      <Field label="Description" value={project.description} className="sm:col-span-2" />
    </dl>
  );
}
