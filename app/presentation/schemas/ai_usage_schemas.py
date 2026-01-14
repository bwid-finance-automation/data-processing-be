"""AI Usage related Pydantic schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AIUsageLogCreate(BaseModel):
    """Schema for creating an AI usage log."""
    project_id: Optional[int] = None
    case_id: Optional[int] = None
    session_id: Optional[str] = None
    provider: str = "gemini"
    model_name: str
    task_type: str = "ocr"
    task_description: Optional[str] = None
    file_name: Optional[str] = None
    file_count: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None


class AIUsageLogResponse(BaseModel):
    """Schema for AI usage log response."""
    id: int
    uuid: str
    project_id: Optional[int] = None
    case_id: Optional[int] = None
    session_id: Optional[str] = None
    provider: str
    model_name: str
    task_type: str
    task_description: Optional[str] = None
    file_name: Optional[str] = None
    file_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    processing_time_ms: float
    processing_time_seconds: float
    estimated_cost_usd: float
    success: bool
    error_message: Optional[str] = None
    requested_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AIUsageListResponse(BaseModel):
    """Schema for paginated AI usage list response."""
    items: List[AIUsageLogResponse]
    total: int
    skip: int
    limit: int


class AIUsageStatsResponse(BaseModel):
    """Schema for aggregated AI usage statistics."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_processing_time_ms: float = 0.0
    total_processing_time_seconds: float = 0.0
    total_cost_usd: float = 0.0
    total_files_processed: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0


class AIUsageByProviderResponse(BaseModel):
    """Schema for usage by provider."""
    provider: str
    request_count: int
    total_tokens: int
    total_cost_usd: float


class AIUsageByTaskTypeResponse(BaseModel):
    """Schema for usage by task type."""
    task_type: str
    request_count: int
    total_tokens: int
    total_cost_usd: float


class AIUsageByUserResponse(BaseModel):
    """Schema for usage by user."""
    user_id: int
    email: str
    full_name: Optional[str] = None
    request_count: int
    total_tokens: int
    total_cost_usd: float


class AIUsageDailyResponse(BaseModel):
    """Schema for daily usage."""
    date: str
    request_count: int
    total_tokens: int
    total_cost_usd: float


class AIUsageDashboardResponse(BaseModel):
    """Schema for complete dashboard data."""
    stats: AIUsageStatsResponse
    by_provider: List[AIUsageByProviderResponse]
    by_task_type: List[AIUsageByTaskTypeResponse]
    by_user: List[AIUsageByUserResponse] = []
    daily_usage: List[AIUsageDailyResponse]
    recent_logs: List[AIUsageLogResponse]


class AIUsageFilterParams(BaseModel):
    """Schema for filter parameters."""
    project_id: Optional[int] = None
    case_id: Optional[int] = None
    provider: Optional[str] = None
    task_type: Optional[str] = None
    success: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)
