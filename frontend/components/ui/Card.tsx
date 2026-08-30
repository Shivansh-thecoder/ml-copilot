import type { ReactNode } from "react";

interface CardProps {
  /** Heading shown at the top of the card. */
  title: string;
  /** Optional icon rendered next to the title. */
  icon?: ReactNode;
  /** Card body — kept generic via `children` so this one component can
   * serve both the workspace's project cards and the activity panel's
   * status cards. */
  children?: ReactNode;
  className?: string;
}

/**
 * Generic bordered card used across the app shell. Pulled out as its own
 * component instead of repeating the same div/border/padding markup in
 * both Workspace and ActivityPanel — a small example of "extract when
 * you notice repetition" rather than abstracting up front.
 */
export default function Card({ title, icon, children, className = "" }: CardProps) {
  return (
    <div
      className={`rounded-lg border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-gray-900 ${className}`}
    >
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <h3 className="font-medium text-gray-900 dark:text-gray-100">{title}</h3>
      </div>
      {children}
    </div>
  );
}
