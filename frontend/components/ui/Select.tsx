import type { SelectHTMLAttributes } from "react";

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
}

// Same label+field pattern as Input/Textarea — justified once a third
// consumer (SettingsPage's Theme and Default Model selects) needed the
// same wrapper as the project SettingsTab's Status select.
export default function Select({ label, className = "", children, ...rest }: SelectProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-gray-700 dark:text-gray-300">{label}</span>
      <select
        className={`rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100 ${className}`}
        {...rest}
      >
        {children}
      </select>
    </label>
  );
}
