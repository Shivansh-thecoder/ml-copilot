import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: ReactNode;
}

const VARIANT_STYLES: Record<ButtonVariant, string> = {
  primary:
    "bg-gray-900 text-white hover:bg-gray-700 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white",
  secondary:
    "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800",
};

// Extracted once the same primary/secondary button markup repeated across
// Workspace ("New Project"), ProjectModal (Cancel/Create), SettingsTab
// (Save), and now Datasets ("Upload Dataset") and Models ("Run
// Inference"). Extends ButtonHTMLAttributes so `type`, `disabled`,
// `onClick`, `title` etc. all still work without redeclaring them —
// callers must still pass `type` explicitly (no default here) since some
// buttons submit a form and others don't.
export default function Button({
  variant = "primary",
  icon,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`flex items-center justify-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT_STYLES[variant]} ${className}`}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}
