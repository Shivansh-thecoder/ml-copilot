import type { TextareaHTMLAttributes } from "react";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label: string;
}

// Same label+field pattern as Input, split into its own component rather
// than an `as="textarea"` prop on Input — the two elements take different
// native prop sets (rows vs. type/min/max), so a shared prop surface
// would need casting either way.
export default function Textarea({ label, className = "", ...rest }: TextareaProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-gray-700 dark:text-gray-300">{label}</span>
      <textarea
        className={`rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100 ${className}`}
        {...rest}
      />
    </label>
  );
}
