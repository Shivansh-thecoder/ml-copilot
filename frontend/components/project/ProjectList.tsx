import EmptyState from "@/components/ui/EmptyState";
import ProjectCard from "@/components/project/ProjectCard";
import type { Project } from "@/lib/types";

interface ProjectListProps {
  projects: Project[];
  selectedProjectId: string | null;
  onSelectProject: (id: string) => void;
  emptyMessage?: string;
}

// Kept separate from ProjectsPage so ProjectsPage only deals with
// page-level composition (heading, button, modal, search) and this file
// only deals with turning an array into cards. `emptyMessage` is a prop
// (not hardcoded) since the caller knows whether "empty" means "no
// projects exist" or "none match the current search."
export default function ProjectList({
  projects,
  selectedProjectId,
  onSelectProject,
  emptyMessage = "No projects yet — create one to get started.",
}: ProjectListProps) {
  if (projects.length === 0) {
    return <EmptyState message={emptyMessage} />;
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {projects.map((project) => (
        <ProjectCard
          key={project.id}
          project={project}
          isSelected={project.id === selectedProjectId}
          onSelect={() => onSelectProject(project.id)}
        />
      ))}
    </div>
  );
}
