import ProjectsPage from "@/components/project/ProjectsPage";
import DatasetsPage from "@/components/dataset/DatasetsPage";
import ModelsPage from "@/components/model/ModelsPage";
import ExperimentsPage from "@/components/experiment/ExperimentsPage";
import SettingsPage from "@/components/settings/SettingsPage";
import type { NewProjectInput, PageId, Project, ProjectSettingsUpdate } from "@/lib/types";

interface WorkspaceProps {
  activePage: PageId;
  searchQuery: string;
  projects: Project[];
  projectsError: string | null;
  selectedProjectId: string | null;
  onSelectProject: (id: string) => void;
  onCreateProject: (data: NewProjectInput) => void;
  onUpdateProject: (id: string, updates: ProjectSettingsUpdate) => void;
  onDeleteProject: (id: string) => void;
}

// Thin page router for the center region -- which page renders here is
// shared state (Sidebar, a sibling, has to be able to change it), so
// AppLayout owns `activePage` and this just switches on it.
export default function Workspace({
  activePage,
  searchQuery,
  projects,
  projectsError,
  selectedProjectId,
  onSelectProject,
  onCreateProject,
  onUpdateProject,
  onDeleteProject,
}: WorkspaceProps) {
  switch (activePage) {
    case "projects":
      return (
        <ProjectsPage
          projects={projects}
          projectsError={projectsError}
          selectedProjectId={selectedProjectId}
          onSelectProject={onSelectProject}
          onCreateProject={onCreateProject}
          onUpdateProject={onUpdateProject}
          onDeleteProject={onDeleteProject}
          searchQuery={searchQuery}
        />
      );
    case "datasets":
      return <DatasetsPage searchQuery={searchQuery} />;
    case "models":
      return <ModelsPage searchQuery={searchQuery} />;
    case "experiments":
      return <ExperimentsPage />;
    case "settings":
      return <SettingsPage />;
    default:
      return null;
  }
}
