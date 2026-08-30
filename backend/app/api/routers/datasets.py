"""HTTP layer for the Dataset resource.

The one structural difference from projects.py: creation is `multipart/
form-data` (a file plus two form fields), not JSON, so `create_dataset`
takes `Form(...)`/`File(...)` parameters instead of a single Pydantic
body — FastAPI can't combine a Pydantic model with file upload fields in
one JSON body, since the request itself isn't JSON.
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.dependencies import get_dataset_service
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpdate, to_response
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetResponse])
def list_datasets(
    service: DatasetService = Depends(get_dataset_service),
) -> list[DatasetResponse]:
    return [to_response(dataset) for dataset in service.list_datasets()]


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    return to_response(service.get_dataset(dataset_id))


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    target_column: str = Form(""),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    data = DatasetCreate(name=name, target_column=target_column)
    return to_response(service.create_dataset(data, file))


@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: str,
    data: DatasetUpdate,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    return to_response(service.update_dataset(dataset_id, data))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> None:
    service.delete_dataset(dataset_id)
