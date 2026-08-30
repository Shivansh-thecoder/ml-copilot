import { Search } from "lucide-react";
import type { InputHTMLAttributes } from "react";

interface SearchBarProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "onChange" | "value"> {
  value: string;
  onChange: (value: string) => void;
}

// Controlled search input reused by the global Navbar search and
// ExperimentsPage's own local search. Reports changes as (value: string)
// rather than the raw ChangeEvent, matching how every other controlled
// field in this codebase reports changes.
export default function SearchBar({ value, onChange, className = "", ...rest }: SearchBarProps) {
  return (
    <div className={`relative flex items-center ${className}`}>
      <Search className="pointer-events-none absolute left-2.5 h-4 w-4 text-gray-400" aria-hidden />
      <input
        type="search"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-md border border-gray-300 bg-white py-1.5 pl-8 pr-3 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100"
        {...rest}
      />
    </div>
  );
}
