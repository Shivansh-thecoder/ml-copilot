"use client";

import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import ProjectTabs from "@/components/project/ProjectTabs";
import OverviewTab from "@/components/project/OverviewTab";
import DatasetTab from "@/components/project/DatasetTab";
import ModelsTab from "@/components/project/ModelsTab";
import ExperimentsTab from "@/components/project/ExperimentsTab";
import CopilotTab from "@/components/project/CopilotTab";
import SettingsTab from "@/components/project/SettingsTab";
import type { Project, ProjectSettingsUpdate, ProjectTabId } from "@/lib/types";

interface ProjectWorkspaceProps {
  project: Project;
  onBack: () => void;
  onSaveSettings: (updates: ProjectSettingsUpdate) => void;
  onDelete: () => void;
}

// Which tab is active is UI state local to this one workspace view — no
// other component (Workspace, AppLayout, ActivityPanel) needs to know it,
// so it stays here instead of being lifted.
export default function ProjectWorkspace({ project, onBack, onSaveSettings, onDelete }: ProjectWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<ProjectTabId>("overview");

  return (
    <div className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <button
          type="button"
          onClick={onBack}
          className="mb-4 flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden />
          Back to Projects
        </button>

        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">{project.name}</h1>

        <div className="mt-6">
          <ProjectTabs activeTab={activeTab} onTabChange={setActiveTab} />
        </div>

        <div className="mt-6">
          {activeTab === "overview" && <OverviewTab project={project} />}
          {activeTab === "dataset" && <DatasetTab datasetName={project.datasetName} />}
          {activeTab === "copilot" && <CopilotTab />}
          {activeTab === "models" && <ModelsTab />}
          {activeTab === "experiments" && <ExperimentsTab />}
          {activeTab === "settings" && (
            <SettingsTab project={project} onSave={onSaveSettings} onDelete={onDelete} />
          )}
        </div>
      </div>
    </div>
  );
}
