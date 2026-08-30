"use client";

import { useEffect, useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Sidebar from "@/components/layout/Sidebar";
import Workspace from "@/components/layout/Workspace";
import ActivityPanel from "@/components/layout/ActivityPanel";
import { ApiError } from "@/services/api";
import { createProject, deleteProject, listProjects, updateProject } from "@/services/projectService";
import type {
  ActivityLogEntry,
  NewProjectInput,
  PageId,
  Project,
  ProjectSettingsUpdate,
} from "@/lib/types";

/**
 * Top-level application shell: Navbar across the top, then Sidebar,
 * Workspace, and ActivityPanel side by side beneath it.
 *
 * Project-dashboard state lives here (projects, selection, activity log)
 * alongside `activePage` (which sidebar page Workspace renders) and
 * `searchQuery` (the global Navbar search) -- Sidebar/Navbar and
 * Workspace/ActivityPanel are siblings, so their nearest common parent is
 * the only place plain prop-lifting can share state between them.
 */
export default function AppLayout() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [logs, setLogs] = useState<ActivityLogEntry[]>([]);
  const [activePage, setActivePage] = useState<PageId>("projects");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    listProjects()
      .then((data) => {
        if (cancelled) return;
        setProjects(data);
        setSelectedProjectId((prev) => prev ?? data[0]?.id ?? null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setProjectsError(error instanceof ApiError ? error.message : "Could not load projects.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function logAction(message: string) {
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setLogs((prev) => [{ id: crypto.randomUUID(), message, timestamp }, ...prev]);
  }

  function handleSelectProject(id: string) {
    setSelectedProjectId(id);
    const project = projects.find((p) => p.id === id);
    if (project) logAction(`Selected Project: ${project.name}`);
  }

  async function handleCreateProject(data: NewProjectInput) {
    try {
      const newProject = await createProject(data);
      setProjects((prev) => [...prev, newProject]);
      setSelectedProjectId(newProject.id);
      logAction(`Created Project: ${newProject.name}`);
    } catch (error: unknown) {
      logAction(`Failed to create project: ${error instanceof ApiError ? error.message : "unknown error"}`);
    }
  }

  async function handleUpdateProject(id: string, updates: ProjectSettingsUpdate) {
    try {
      const updated = await updateProject(id, updates);
      setProjects((prev) => prev.map((project) => (project.id === id ? updated : project)));
      logAction(`Updated Project: ${updated.name}`);
    } catch (error: unknown) {
      logAction(`Failed to update project: ${error instanceof ApiError ? error.message : "unknown error"}`);
    }
  }

  async function handleDeleteProject(id: string) {
    const project = projects.find((p) => p.id === id);
    try {
      await deleteProject(id);
      setProjects((prev) => prev.filter((p) => p.id !== id));
      setSelectedProjectId((prev) => (prev === id ? null : prev));
      logAction(`Deleted Project: ${project?.name ?? id}`);
    } catch (error: unknown) {
      logAction(`Failed to delete project: ${error instanceof ApiError ? error.message : "unknown error"}`);
    }
  }

  return (
    <div className="flex h-screen flex-col bg-gray-50 dark:bg-gray-950">
      <Navbar searchQuery={searchQuery} onSearchChange={setSearchQuery} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar activePage={activePage} onNavigate={setActivePage} />
        <Workspace
          activePage={activePage}
          searchQuery={searchQuery}
          projects={projects}
          projectsError={projectsError}
          selectedProjectId={selectedProjectId}
          onSelectProject={handleSelectProject}
          onCreateProject={handleCreateProject}
          onUpdateProject={handleUpdateProject}
          onDeleteProject={handleDeleteProject}
        />
        <ActivityPanel logs={logs} />
      </div>
    </div>
  );
}
