"""Project and ProjectCase schemas for API requests/responses."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# ============== Request Schemas ==============

class ProjectCreateRequest(BaseModel):
    """Request schema for creating a new project."""
    project_name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    password: Optional[str] = Field(None, min_length=4, max_length=100, description="Optional password for protection")


class ProjectUpdateRequest(BaseModel):
    """Request schema for updating a project."""
    project_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    current_password: Optional[str] = Field(None, description="Current password (required if project is protected)")


class ProjectSetPasswordRequest(BaseModel):
    """Request schema for setting/changing project password."""
    current_password: Optional[str] = Field(None, description="Current password (required if project is protected)")
    password: Optional[str] = Field(None, min_length=4, max_length=100, description="New password, null to remove")


class ProjectDeleteRequest(BaseModel):
    """Request schema for deleting a project."""
    current_password: Optional[str] = Field(None, description="Current password (required if project is protected)")


class ProjectVerifyPasswordRequest(BaseModel):
    """Request schema for verifying project password."""
    password: str = Field(..., description="Password to verify")


# ============== Response Schemas ==============

class ProjectCaseSummary(BaseModel):
    """Summary of a project case."""
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    case_type: str
    total_files: int
    last_processed_at: Optional[datetime] = None
    created_at: datetime


class ProjectResponse(BaseModel):
    """Response schema for a project."""
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    project_name: str
    description: Optional[str] = None
    is_protected: bool
    last_accessed_at: datetime
    created_at: datetime
    updated_at: datetime


class ProjectDetailResponse(ProjectResponse):
    """Detailed response schema for a project including cases."""
    cases: List[ProjectCaseSummary] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    """Response schema for listing projects."""
    projects: List[ProjectResponse]
    total: int


class ProjectVerifyResponse(BaseModel):
    """Response schema for password verification."""
    verified: bool
    message: str


# ============== Case Response Schemas ==============

class CaseFileItem(BaseModel):
    """File item in a case history."""
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    file_name: str
    processed_at: Optional[datetime] = None
    created_at: datetime
    # Additional fields based on case type
    extra_info: Optional[dict] = None


class CaseDetailResponse(BaseModel):
    """Detail response for a specific case."""
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    case_type: str
    total_files: int
    last_processed_at: Optional[datetime] = None
    created_at: datetime
    files: List[CaseFileItem] = Field(default_factory=list)


# ============== Bank Statement specific ==============

class BankStatementFileItem(CaseFileItem):
    """Bank statement file item with additional info."""
    bank_name: Optional[str] = None
    total_transactions: Optional[int] = None
    total_accounts: Optional[int] = None


class ParseSessionFileItem(BaseModel):
    """File item within a parse session."""
    uuid: str
    file_name: str
    bank_name: str
    transaction_count: int


class ParseSessionItem(BaseModel):
    """A parse session containing multiple files processed together."""
    session_id: str
    processed_at: Optional[datetime] = None
    file_count: int
    total_transactions: int
    banks: List[str] = Field(default_factory=list)
    files: List[ParseSessionFileItem] = Field(default_factory=list)


class BankStatementSessionsResponse(BaseModel):
    """Response for bank statement parse sessions."""
    uuid: UUID
    case_type: str
    total_sessions: int
    last_processed_at: Optional[datetime] = None
    created_at: datetime
    sessions: List[ParseSessionItem] = Field(default_factory=list)
