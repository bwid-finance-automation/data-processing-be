"""API router for Project operations."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.application.project.project_service import ProjectService
from app.core.dependencies import get_project_service
from app.presentation.schemas.project_schemas import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectSetPasswordRequest,
    ProjectDeleteRequest,
    ProjectVerifyPasswordRequest,
    ProjectResponse,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectVerifyResponse,
    ProjectCaseSummary,
    CaseDetailResponse,
    BankStatementFileItem,
)

router = APIRouter(prefix="/projects", tags=["Projects"])


# ============== Debug endpoint (remove in production) ==============

@router.get("/{project_uuid}/debug-hash")
async def debug_project_hash(
    project_uuid: UUID,
    test_password: str = Query(..., description="Password to test"),
    service: ProjectService = Depends(get_project_service),
):
    """Debug endpoint to check password hash."""
    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    input_hash = ProjectService.hash_password(test_password)
    return {
        "project_uuid": str(project_uuid),
        "is_protected": project.is_protected,
        "stored_hash": project.password_hash,
        "input_hash": input_hash,
        "match": input_hash == project.password_hash if project.password_hash else None,
    }


# ============== Project CRUD ==============

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: ProjectCreateRequest,
    service: ProjectService = Depends(get_project_service),
):
    """
    Create a new project.

    - **project_name**: Name of the project (required)
    - **description**: Optional description
    - **password**: Optional password for protection
    """
    project = await service.create_project(
        project_name=request.project_name,
        description=request.description,
        password=request.password,
    )
    return ProjectResponse.model_validate(project)


@router.get("", response_model=ProjectListResponse)
@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records"),
    search: Optional[str] = Query(None, description="Search by project name"),
    service: ProjectService = Depends(get_project_service),
):
    """
    List all projects with pagination.

    - **skip**: Number of records to skip (default: 0)
    - **limit**: Maximum records to return (default: 50, max: 100)
    - **search**: Optional search term for project name
    """
    projects, total = await service.list_projects(skip, limit, search)
    return ProjectListResponse(
        projects=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
    )


@router.get("/search")
async def search_projects(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    service: ProjectService = Depends(get_project_service),
):
    """
    Quick search projects by name.

    Returns a simple list for autocomplete/dropdown.
    """
    projects = await service.search_projects(q, limit)
    return [
        {
            "uuid": str(p.uuid),
            "project_name": p.project_name,
            "is_protected": p.is_protected,
        }
        for p in projects
    ]


@router.get("/{project_uuid}", response_model=ProjectDetailResponse)
async def get_project(
    project_uuid: UUID,
    service: ProjectService = Depends(get_project_service),
):
    """
    Get project details including all cases.
    """
    project = await service.get_project_with_cases(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    return ProjectDetailResponse(
        uuid=project.uuid,
        project_name=project.project_name,
        description=project.description,
        is_protected=project.is_protected,
        last_accessed_at=project.last_accessed_at,
        created_at=project.created_at,
        updated_at=project.updated_at,
        cases=[ProjectCaseSummary.model_validate(c) for c in project.cases],
    )


@router.put("/{project_uuid}", response_model=ProjectResponse)
async def update_project(
    project_uuid: UUID,
    request: ProjectUpdateRequest,
    service: ProjectService = Depends(get_project_service),
):
    """
    Update project details.

    If project is password protected, current_password is required.
    """
    # Get project to check if protected
    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    # Verify password if protected
    if project.is_protected:
        if not request.current_password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Password required for protected project",
            )
        if not ProjectService.verify_password(request.current_password, project.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password",
            )

    updated_project = await service.update_project(
        uuid=project_uuid,
        project_name=request.project_name,
        description=request.description,
    )
    return ProjectResponse.model_validate(updated_project)


@router.post("/{project_uuid}/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_uuid: UUID,
    request: ProjectDeleteRequest,
    service: ProjectService = Depends(get_project_service),
):
    """
    Delete a project and all its cases.

    If project is password protected, current_password is required.
    """
    # Get project to check if protected
    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    # Verify password if protected
    if project.is_protected:
        if not request.current_password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Password required for protected project",
            )
        if not ProjectService.verify_password(request.current_password, project.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password",
            )

    deleted = await service.delete_project(project_uuid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )


# ============== Password Operations ==============

@router.post("/{project_uuid}/password", response_model=ProjectResponse)
async def set_project_password(
    project_uuid: UUID,
    request: ProjectSetPasswordRequest,
    service: ProjectService = Depends(get_project_service),
):
    """
    Set or remove project password.

    If project is currently protected, current_password is required.
    - Pass password to set/change
    - Pass null/empty to remove password protection
    """
    # Get project to check if protected
    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    # Verify current password if project is protected
    if project.is_protected:
        if not request.current_password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password required for protected project",
            )
        if not ProjectService.verify_password(request.current_password, project.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid current password",
            )

    updated_project = await service.set_password(project_uuid, request.password)
    return ProjectResponse.model_validate(updated_project)


@router.post("/{project_uuid}/verify", response_model=ProjectVerifyResponse)
async def verify_project_password(
    project_uuid: UUID,
    request: ProjectVerifyPasswordRequest,
    service: ProjectService = Depends(get_project_service),
):
    """
    Verify project password.
    """
    import logging
    logger = logging.getLogger(__name__)

    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    if not project.is_protected:
        return ProjectVerifyResponse(verified=True, message="Project is not protected")

    # Debug logging
    input_hash = ProjectService.hash_password(request.password)
    logger.info(f"Verify password for project {project_uuid}")
    logger.info(f"Input password hash: {input_hash[:20]}...")
    logger.info(f"Stored password hash: {project.password_hash[:20] if project.password_hash else 'None'}...")

    verified = ProjectService.verify_password(request.password, project.password_hash)
    logger.info(f"Verification result: {verified}")

    return ProjectVerifyResponse(
        verified=verified,
        message="Password verified" if verified else "Invalid password",
    )


# ============== Case Operations ==============

@router.get("/{project_uuid}/cases")
async def get_project_cases(
    project_uuid: UUID,
    service: ProjectService = Depends(get_project_service),
):
    """
    Get all cases for a project.
    """
    project = await service.get_project(project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_uuid}",
        )

    cases = await service.get_cases_by_project(project_uuid)
    return [ProjectCaseSummary.model_validate(c) for c in cases]


@router.get("/{project_uuid}/cases/bank-statement")
async def get_bank_statement_case(
    project_uuid: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    service: ProjectService = Depends(get_project_service),
):
    """
    Get bank statement parse sessions for a project.

    Returns sessions grouped by parse operation (not individual files).
    Each session contains all files that were processed together.
    """
    case, sessions, total = await service.get_bank_statement_sessions_by_project(
        project_uuid, skip, limit
    )

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No bank statement case found for project: {project_uuid}",
        )

    return {
        "uuid": str(case.uuid),
        "case_type": case.case_type,
        "total_sessions": total,
        "last_processed_at": case.last_processed_at,
        "created_at": case.created_at,
        "sessions": sessions,
    }
