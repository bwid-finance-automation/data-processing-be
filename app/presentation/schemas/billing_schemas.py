"""
Pydantic models for Utility Billing request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class FileUploadResponse(BaseModel):
    filename: str
    file_type: str
    size: int
    uploaded_at: datetime
    message: str


class ProcessingRequest(BaseModel):
    input_files: Optional[List[str]] = Field(None, description="Specific files to process, or all if empty")


class ValidationIssue(BaseModel):
    row: int
    issue: str
    site: Optional[str] = None
    unit: Optional[str] = None
    subsidiary: Optional[str] = None
    cs_tenant: Optional[str] = None


class ProcessingStats(BaseModel):
    total_input_records: int
    total_invoices: int
    total_line_items: int
    validation_issues_count: int
    processing_time_seconds: float


class ProcessingResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[ProcessingStats] = None
    output_file: Optional[str] = None
    validation_report: Optional[str] = None
    validation_issues: Optional[List[ValidationIssue]] = None
    error: Optional[str] = None


class FileInfo(BaseModel):
    filename: str
    size: int
    uploaded_at: Optional[datetime] = None
    file_type: str


class SystemStatus(BaseModel):
    status: str
    input_files_count: int
    master_data_files_count: int
    output_files_count: int
    last_processing: Optional[datetime] = None


class MasterDataStatus(BaseModel):
    customers_count: Optional[int] = None
    units_count: Optional[int] = None
    subsidiary_config_exists: bool
    utility_mapping_exists: bool
    last_updated: Optional[datetime] = None


class SessionResponse(BaseModel):
    session_id: str
    message: str
    expires_in_minutes: int
