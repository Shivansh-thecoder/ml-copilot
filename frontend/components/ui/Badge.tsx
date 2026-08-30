import type { BadgeTone } from "@/lib/types";

interface BadgeProps {
  label: string;
  tone: BadgeTone;
}

const TONE_STYLES: Record<BadgeTone, { dot: string; text: string }> = {
  green: { dot: "bg-green-500", text: "text-green-700 dark:text-green-400" },
  blue: { dot: "bg-blue-500", text: "text-blue-700 dark:text-blue-400" },
  gray: { dot: "bg-gray-400", text: "text-gray-600 dark:text-gray-400" },
  amber: { dot: "bg-amber-500", text: "text-amber-700 dark:text-amber-400" },
  red: { dot: "bg-red-500", text: "text-red-700 dark:text-red-400" },
};

// Takes a label + tone rather than a domain-specific status -- the
// status -> {label, tone} mapping per domain lives in lib/statusMeta.ts.
export default function Badge({ label, tone }: BadgeProps) {
  const { dot, text } = TONE_STYLES[tone];

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium dark:bg-gray-800 ${text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}
