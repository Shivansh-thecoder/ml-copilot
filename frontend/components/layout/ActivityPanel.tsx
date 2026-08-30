import Card from "@/components/ui/Card";
import EmptyState from "@/components/ui/EmptyState";
import type { ActivityLogEntry } from "@/lib/types";

interface ActivityPanelProps {
  logs: ActivityLogEntry[];
}

// Read-only feed — receives logs as a prop rather than owning any state
// itself. The log data has to be shared with ProjectsPage's select/
// create actions, so it lives in their common parent (AppLayout) and
// flows down here; ActivityPanel just renders whatever it's given.
export default function ActivityPanel({ logs }: ActivityPanelProps) {
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-4 overflow-y-auto border-l border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-950">
      <Card title="Activity Log">
        {logs.length === 0 ? (
          <EmptyState message="No activity yet." />
        ) : (
          <ul className="flex flex-col gap-3">
            {logs.map((log) => (
              <li key={log.id} className="text-sm">
                <p className="text-gray-700 dark:text-gray-300">{log.message}</p>
                <p className="text-xs text-gray-400 dark:text-gray-500">{log.timestamp}</p>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </aside>
  );
}
