import type { InputHTMLAttributes } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

// Extracted once the same label+input wrapper repeated 3x in ProjectModal
// and again in SettingsTab. Spreads native input props so value/onChange/
// type/placeholder/autoFocus etc. all still work normally.
export default function Input({ label, className = "", ...rest }: InputProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-gray-700 dark:text-gray-300">{label}</span>
      <input
        className={`rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100 ${className}`}
        {...rest}
      />
    </label>
  );
}
