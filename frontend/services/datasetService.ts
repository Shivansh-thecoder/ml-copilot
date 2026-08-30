// Client for the dataset endpoints (backend/app/api/routers/datasets.py).
// CopilotTab.tsx also uses `listDatasets()` to offer real, already-
// uploaded datasets for a Copilot run rather than inventing a second way
// to pick or store one.

import { apiRequest } from "@/services/api";
import type { Dataset } from "@/lib/types";

// Wire shape from backend/app/schemas/dataset.py's DatasetResponse —
// snake_case, exactly as FastAPI/Pydantic serializes it.
interface DatasetResponseDto {
  id: string;
  name: string;
  target_column: string;
  rows: number;
  columns: number;
  size_label: string;
  uploaded_at: string;
}

function mapDataset(dto: DatasetResponseDto): Dataset {
  return {
    id: dto.id,
    name: dto.name,
    rows: dto.rows,
    columns: dto.columns,
    targetColumn: dto.target_column,
    sizeLabel: dto.size_label,
    uploadedAt: dto.uploaded_at,
  };
}

export async function listDatasets(): Promise<Dataset[]> {
  const dtos = await apiRequest<DatasetResponseDto[]>("/datasets");
  return dtos.map(mapDataset);
}

// POST /datasets is multipart/form-data (a file plus two form fields),
// not JSON -- FastAPI can't combine a Pydantic body with a file upload
// in one JSON request (see app/api/routers/datasets.py).
export async function uploadDataset(file: File, name: string, targetColumn: string): Promise<Dataset> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", name);
  formData.append("target_column", targetColumn);

  const dto = await apiRequest<DatasetResponseDto>("/datasets", { method: "POST", body: formData });
  return mapDataset(dto);
}

export async function deleteDataset(id: string): Promise<void> {
  await apiRequest<void>(`/datasets/${id}`, { method: "DELETE" });
}
