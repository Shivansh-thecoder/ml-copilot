// Client for the project endpoints (backend/app/api/routers/projects.py).

import { apiRequest } from "@/services/api";
import type { NewProjectInput, Project, ProjectSettingsUpdate } from "@/lib/types";

// Wire shape from backend/app/schemas/project.py's ProjectResponse.
interface ProjectResponseDto {
  id: string;
  name: string;
  description: string;
  dataset_name: string;
  status: Project["status"];
  last_opened: string;
}

function mapProject(dto: ProjectResponseDto): Project {
  return {
    id: dto.id,
    name: dto.name,
    description: dto.description,
    datasetName: dto.dataset_name,
    status: dto.status,
    lastOpened: dto.last_opened,
  };
}

export async function listProjects(): Promise<Project[]> {
  const dtos = await apiRequest<ProjectResponseDto[]>("/projects");
  return dtos.map(mapProject);
}

export async function createProject(data: NewProjectInput): Promise<Project> {
  const dto = await apiRequest<ProjectResponseDto>("/projects", {
    method: "POST",
    body: JSON.stringify({ name: data.name, description: data.description, dataset_name: data.datasetName }),
  });
  return mapProject(dto);
}

export async function updateProject(id: string, updates: ProjectSettingsUpdate): Promise<Project> {
  const dto = await apiRequest<ProjectResponseDto>(`/projects/${id}`, {
    method: "PUT",
    body: JSON.stringify({ name: updates.name, description: updates.description, status: updates.status }),
  });
  return mapProject(dto);
}

export async function deleteProject(id: string): Promise<void> {
  await apiRequest<void>(`/projects/${id}`, { method: "DELETE" });
}
