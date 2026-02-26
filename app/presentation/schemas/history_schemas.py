"""Schemas for file processing history endpoints."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class ModuleSummary(BaseModel):
    """Summary for a single module."""
    module: str
    total_sessions: int
    total_files: int
    last_processed_at: Optional[datetime] = None


class HistorySummaryResponse(BaseModel):
    """Summary of all modules for current user."""
    modules: List[ModuleSummary]
    total_sessions: int
    total_files: int


class BankStatementSessionItem(BaseModel):
    """A single bank statement parse session."""
    session_id: str
    file_count: int
    total_transactions: int
    banks: List[str]
    files: List[str]
    processed_at: Optional[datetime] = None
    download_url: Optional[str] = None


class BankStatementHistoryResponse(BaseModel):
    """Paginated bank statement history."""
    sessions: List[BankStatementSessionItem]
    total: int
    skip: int
    limit: int


class ContractSessionItem(BaseModel):
    """A single contract OCR session."""
    file_name: Optional[str] = None
    contract_number: Optional[str] = None
    contract_title: Optional[str] = None
    tenant: Optional[str] = None
    processed_at: Optional[datetime] = None


class ContractHistoryResponse(BaseModel):
    """Paginated contract history."""
    contracts: List[ContractSessionItem]
    total: int
    skip: int
    limit: int


class GLASessionItem(BaseModel):
    """A single GLA analysis session."""
    file_name: Optional[str] = None
    project_code: str
    project_name: str
    product_type: str
    region: str
    period_label: Optional[str] = None
    processed_at: Optional[datetime] = None


class GLAHistoryResponse(BaseModel):
    """Paginated GLA history."""
    sessions: List[GLASessionItem]
    total: int
    skip: int
    limit: int


class AnalysisSessionItem(BaseModel):
    """A single analysis session (variance, utility billing, excel comparison)."""
    session_id: str
    analysis_type: Optional[str] = None
    status: str
    files_count: int
    processing_details: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None


class AnalysisHistoryResponse(BaseModel):
    """Paginated analysis history."""
    sessions: List[AnalysisSessionItem]
    total: int
    skip: int
    limit: int
