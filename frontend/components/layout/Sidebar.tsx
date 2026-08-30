import type { LucideIcon } from "lucide-react";
import { Cpu, Database, FlaskConical, FolderKanban, Settings } from "lucide-react";
import type { PageId } from "@/lib/types";

interface SidebarNavItem {
  id: PageId;
  label: string;
  icon: LucideIcon;
}

interface SidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
}

// Data-driven nav list: adding a new page later means adding one entry
// here rather than duplicating markup for every item.
const NAV_ITEMS: SidebarNavItem[] = [
  { id: "projects", label: "Projects", icon: FolderKanban },
  { id: "datasets", label: "Datasets", icon: Database },
  { id: "models", label: "Models", icon: Cpu },
  { id: "experiments", label: "Experiments", icon: FlaskConical },
  { id: "settings", label: "Settings", icon: Settings },
];

/** Fixed-width left navigation. Clicking an item calls `onNavigate`, and
 * AppLayout swaps which page Workspace renders -- no router involved,
 * just lifted state. */
export default function Sidebar({ activePage, onNavigate }: SidebarProps) {
  return (
    <aside className="flex w-56 shrink-0 flex-col gap-1 border-r border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-950">
      {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          onClick={() => onNavigate(id)}
          className={`flex items-center gap-3 rounded-md px-3 py-2 text-left text-sm transition-colors ${
            activePage === id
              ? "bg-gray-100 font-medium text-gray-900 dark:bg-gray-900 dark:text-gray-100"
              : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-900"
          }`}
        >
          <Icon className="h-4 w-4 shrink-0" aria-hidden />
          {label}
        </button>
      ))}
    </aside>
  );
}
