import type { BadgeTone, ProjectStatus } from "@/lib/types";

interface StatusMeta {
  label: string;
  tone: BadgeTone;
}

// status -> {label, tone} mapping, kept out of the shared Badge component
// so Badge itself stays domain-agnostic.
export const PROJECT_STATUS_META: Record<ProjectStatus, StatusMeta> = {
  active: { label: "Active", tone: "green" },
  completed: { label: "Completed", tone: "blue" },
  archived: { label: "Archived", tone: "gray" },
};
