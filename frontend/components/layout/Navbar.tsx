import SearchBar from "@/components/ui/SearchBar";

interface NavbarProps {
  /** Defaults to "ML Copilot" — kept as a prop (rather than hardcoded)
   * since this bar will likely show a project name here later. */
  title?: string;
  searchQuery: string;
  onSearchChange: (value: string) => void;
}

/** Top navigation bar: brand, a global search box, and an account/
 * settings placeholder. No routing or auth yet — the profile slot is a
 * static placeholder until that's built. The search box is controlled
 * from AppLayout since Projects/Datasets/Models (nested deep under
 * Workspace) all need to read the same query. */
export default function Navbar({ title = "ML Copilot", searchQuery, onSearchChange }: NavbarProps) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-gray-200 bg-white px-4 dark:border-gray-800 dark:bg-gray-950">
      <span className="shrink-0 text-sm font-semibold tracking-tight text-gray-900 dark:text-gray-100">
        {title}
      </span>

      <SearchBar
        value={searchQuery}
        onChange={onSearchChange}
        placeholder="Search projects, datasets, models..."
        className="w-full max-w-sm"
      />

      {/* Placeholder for future user profile / settings menu */}
      <div
        aria-hidden
        className="h-8 w-8 shrink-0 rounded-full border border-gray-200 bg-gray-100 dark:border-gray-800 dark:bg-gray-800"
      />
    </header>
  );
}
