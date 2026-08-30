import type { ReactNode } from "react";

interface EmptyStateProps {
  message: string;
  action?: ReactNode;
}

// Extracted once a third "there's nothing here" message appeared
// (ProjectList and ActivityPanel already had their own ad-hoc versions;
// Datasets/Models/Experiments needed the same pattern, some with an
// action slot for "create the first one").
export default function EmptyState({ message, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-gray-300 p-8 text-center dark:border-gray-700">
      <p className="text-sm text-gray-500 dark:text-gray-400">{message}</p>
      {action}
    </div>
  );
}
