import type { ProjectTabId } from "@/lib/types";

interface ProjectTabsProps {
  activeTab: ProjectTabId;
  onTabChange: (tab: ProjectTabId) => void;
}

const TABS: { id: ProjectTabId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "dataset", label: "Dataset" },
  { id: "copilot", label: "Copilot" },
  { id: "models", label: "Models" },
  { id: "experiments", label: "Experiments" },
  { id: "settings", label: "Settings" },
];

// Purely presentational tab strip — it doesn't know what any tab
// contains. ProjectWorkspace owns `activeTab` and decides what to render;
// this component only displays it and reports clicks.
export default function ProjectTabs({ activeTab, onTabChange }: ProjectTabsProps) {
  return (
    <div className="flex gap-1 border-b border-gray-200 dark:border-gray-800">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onTabChange(tab.id)}
          className={`border-b-2 px-3 py-2 text-sm font-medium ${
            activeTab === tab.id
              ? "border-gray-900 text-gray-900 dark:border-gray-100 dark:text-gray-100"
              : "border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
