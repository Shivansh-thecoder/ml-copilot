"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import Button from "@/components/ui/Button";
import SectionHeader from "@/components/ui/SectionHeader";
import ProjectList from "@/components/project/ProjectList";
import ProjectModal from "@/components/project/ProjectModal";
import ProjectWorkspace from "@/components/project/ProjectWorkspace";
import type { NewProjectInput, Project, ProjectSettingsUpdate } from "@/lib/types";

interface ProjectsPageProps {
  projects: Project[];
  projectsError: string | null;
  selectedProjectId: string | null;
  onSelectProject: (id: string) => void;
  onCreateProject: (data: NewProjectInput) => void;
  onUpdateProject: (id: string, updates: ProjectSettingsUpdate) => void;
  onDeleteProject: (id: string) => void;
  searchQuery: string;
}

type ProjectsView = "dashboard" | "project";

// Two views live here: the project list ("dashboard") and a single
// project's tabbed workspace ("project"). Which one is showing is UI
// state local to this page — there's no router to own it instead.
export default function ProjectsPage({
  projects,
  projectsError,
  selectedProjectId,
  onSelectProject,
  onCreateProject,
  onUpdateProject,
  onDeleteProject,
  searchQuery,
}: ProjectsPageProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [view, setView] = useState<ProjectsView>("dashboard");

  // Derived, not stored: recomputed from `projects` + `selectedProjectId`
  // on every render instead of living in its own state variable, so it
  // can never disagree with the shared project list.
  const selectedProject = projects.find((project) => project.id === selectedProjectId) ?? null;

  const visibleProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  function handleSelectProject(id: string) {
    onSelectProject(id);
    setView("project");
  }

  function handleCreate(data: NewProjectInput) {
    onCreateProject(data);
    setIsModalOpen(false);
  }

  function handleDelete(id: string) {
    onDeleteProject(id);
    setView("dashboard");
  }

  if (view === "project" && selectedProject) {
    return (
      <ProjectWorkspace
        project={selectedProject}
        onBack={() => setView("dashboard")}
        onSaveSettings={(updates) => onUpdateProject(selectedProject.id, updates)}
        onDelete={() => handleDelete(selectedProject.id)}
      />
    );
  }

  return (
    <section className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <SectionHeader
          title="My Projects"
          description="Create or continue machine learning projects."
          action={
            <Button
              type="button"
              onClick={() => setIsModalOpen(true)}
              icon={<Plus className="h-4 w-4" aria-hidden />}
            >
              New Project
            </Button>
          }
        />

        {projectsError && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            {projectsError}
          </div>
        )}

        <div className="mt-8">
          <ProjectList
            projects={visibleProjects}
            selectedProjectId={selectedProjectId}
            onSelectProject={handleSelectProject}
            emptyMessage={
              projects.length === 0
                ? "No projects yet — create one to get started."
                : "No projects match your search."
            }
          />
        </div>
      </div>

      {isModalOpen && (
        <ProjectModal onCancel={() => setIsModalOpen(false)} onCreate={handleCreate} />
      )}
    </section>
  );
}
