# app/models/analysis.py
"""
Comprehensive Pydantic models for analysis requests and responses.

This module provides strongly-typed models for all API operations with
validation, documentation, and standardized error handling.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from enum import Enum
import uuid

# Enums for type safety and validation

class SessionStatus(str, Enum):
    """Session status enumeration."""
    CREATED = "created"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    REVENUE_VARIANCE = "revenue_variance"
    COMPREHENSIVE = "comprehensive"
    AI_POWERED = "ai_powered"
    CUSTOM = "custom"

class FileValidationStatus(str, Enum):
    """File validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"

class ProgressType(str, Enum):
    """Progress message type enumeration."""
    LOG = "log"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Base models with common functionality

class BaseResponseModel(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    success: bool = Field(True, description="Indicates if the request was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")

class BaseRequestModel(BaseModel):
    """Base request model with common validation."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

# Enhanced file and session models

class FileInfo(BaseModel):
    """Comprehensive file information with validation status."""
    filename: str = Field(..., description="Original filename", min_length=1)
    size: int = Field(..., description="File size in bytes", ge=0)
    content_type: Optional[str] = Field(None, description="MIME content type")
    extension: str = Field(..., description="File extension")
    validation_status: FileValidationStatus = Field(default=FileValidationStatus.PENDING, description="Validation status")
    security_scan_passed: bool = Field(default=False, description="Security scan status")
    estimated_sheets: int = Field(default=0, description="Estimated number of sheets", ge=0)
    file_path: Optional[str] = Field(None, description="Server file path")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    created_at: Optional[float] = Field(None, description="File creation timestamp")
    modified_at: Optional[float] = Field(None, description="File modification timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Filename cannot be empty")
        return v.strip()

class SessionInfo(BaseModel):
    """Comprehensive session information."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    status: SessionStatus = Field(default=SessionStatus.CREATED, description="Current session status")
    analysis_type: Optional[AnalysisType] = Field(None, description="Type of analysis being performed")
    files_count: int = Field(default=0, description="Number of files in this session", ge=0)
    progress_percentage: int = Field(default=0, description="Overall progress percentage", ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    timeout_minutes: int = Field(default=60, description="Session timeout in minutes", ge=1, le=1440)
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")
    error_message: Optional[str] = Field(None, description="Error message if session failed")

class AnalysisProgress(BaseModel):
    """Detailed analysis progress information."""
    session_id: str = Field(..., description="Associated session identifier")
    status: SessionStatus = Field(..., description="Current analysis status")
    progress_percentage: int = Field(..., description="Progress percentage", ge=0, le=100)
    current_step: str = Field(..., description="Current processing step")
    total_steps: int = Field(default=10, description="Total number of steps", ge=1)
    current_step_number: int = Field(default=1, description="Current step number", ge=1)
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis start time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    processing_details: Dict[str, Any] = Field(default_factory=dict, description="Processing details")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")

class AnalysisResult(BaseModel):
    """Comprehensive analysis result."""
    session_id: str = Field(..., description="Associated session identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    subsidiary_name: str = Field(..., description="Analyzed subsidiary name")
    filename: str = Field(..., description="Source filename")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Result creation time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time", ge=0)

    # Analysis results
    summary: Dict[str, Any] = Field(default_factory=dict, description="Executive summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from analysis")
    detailed_results: Dict[str, Any] = Field(default_factory=dict, description="Detailed analysis results")
    configuration_used: Dict[str, Any] = Field(default_factory=dict, description="Configuration used")

    # Quality metrics
    data_quality_score: Optional[float] = Field(None, description="Data quality score", ge=0, le=1)
    confidence_score: Optional[float] = Field(None, description="Analysis confidence score", ge=0, le=1)
    risk_assessment: Optional[RiskLevel] = Field(None, description="Overall risk level")

    # File references
    result_files: List[str] = Field(default_factory=list, description="Generated result file paths")
    debug_files: List[str] = Field(default_factory=list, description="Debug file paths")

    # Validation
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    errors: List[str] = Field(default_factory=list, description="Analysis errors")

class DebugFileInfo(BaseModel):
    """Debug file information."""
    key: str = Field(..., description="File key/identifier")
    filename: str = Field(..., description="Debug filename")
    size: int = Field(..., description="File size in bytes", ge=0)
    content_type: str = Field(..., description="File content type")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    description: Optional[str] = Field(None, description="File description")
    download_url: str = Field(..., description="Download URL")

# Request models

class AnalysisConfigRequest(BaseRequestModel):
    """Configuration parameters for analysis."""
    materiality_vnd: Optional[float] = Field(None, description="Materiality threshold in VND")
    recurring_pct_threshold: Optional[float] = Field(None, description="Recurring percentage threshold")
    revenue_opex_pct_threshold: Optional[float] = Field(None, description="Revenue OPEX percentage threshold")
    bs_pct_threshold: Optional[float] = Field(None, description="Balance sheet percentage threshold")
    recurring_code_prefixes: Optional[str] = Field(None, description="Comma-separated recurring code prefixes")
    min_trend_periods: Optional[int] = Field(None, description="Minimum trend periods")
    gm_drop_threshold_pct: Optional[float] = Field(None, description="Gross margin drop threshold percentage")
    dep_pct_only_prefixes: Optional[str] = Field(None, description="Depreciation percentage only prefixes")
    customer_column_hints: Optional[str] = Field(None, description="Customer column hints")

class AnalysisSession(BaseModel):
    """Analysis session information."""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")

# Legacy AnalysisProgress model removed - using enhanced version above

class RevenueVarianceChange(BaseModel):
    """Month-over-month revenue change."""
    period_from: str = Field(..., description="Source period")
    period_to: str = Field(..., description="Target period")
    previous_revenue: float = Field(..., description="Previous period revenue")
    current_revenue: float = Field(..., description="Current period revenue")
    absolute_change: float = Field(..., description="Absolute change in VND")
    percentage_change: float = Field(..., description="Percentage change")
    change_direction: str = Field(..., description="Direction: Increase/Decrease/No Change")

class EntityContribution(BaseModel):
    """Individual entity contribution to changes."""
    entity: str = Field(..., description="Entity/vendor/customer name")
    previous_value: float = Field(..., description="Previous period value")
    current_value: float = Field(..., description="Current period value")
    absolute_change: float = Field(..., description="Absolute change")
    percentage_change: float = Field(..., description="Percentage change")
    contribution_to_period_change: float = Field(..., description="Contribution to total period change (%)")

class NetEffectAnalysis(BaseModel):
    """Net effect analysis showing positive and negative contributors."""
    period_from: str = Field(..., description="Source period")
    period_to: str = Field(..., description="Target period")
    total_change: float = Field(..., description="Total period change")
    net_effect_explanation: str = Field(..., description="Human-readable net effect explanation")
    positive_contributors: List[EntityContribution] = Field(default_factory=list, description="Positive contributors")
    negative_contributors: List[EntityContribution] = Field(default_factory=list, description="Negative contributors")
    total_positive_change: float = Field(..., description="Sum of all positive changes")
    total_negative_change: float = Field(..., description="Sum of all negative changes")
    entities_with_significant_change: int = Field(..., description="Number of entities with significant changes")

class RevenueStreamAnalysis(BaseModel):
    """Revenue stream analysis for individual accounts."""
    account_name: str = Field(..., description="Revenue account name")
    total_entities: int = Field(..., description="Total entities in this stream")
    month_changes: List[RevenueVarianceChange] = Field(default_factory=list, description="Month-over-month changes")
    period_impacts: List[NetEffectAnalysis] = Field(default_factory=list, description="Net effect analysis by period")

class RevenueVarianceAnalysisResponse(BaseModel):
    """Complete revenue variance analysis response."""
    subsidiary: str = Field(..., description="Subsidiary name")
    filename: str = Field(..., description="Analyzed file name")
    months_analyzed: List[str] = Field(..., description="List of months analyzed")

    # Executive Summary
    analysis_summary: Dict[str, int] = Field(..., description="High-level analysis summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from analysis")

    # Core Analysis Results
    total_revenue_analysis: Dict[str, Any] = Field(..., description="Total revenue month-over-month analysis")
    revenue_stream_analysis: Dict[str, Any] = Field(..., description="Individual revenue stream analysis")
    vendor_customer_impact: Dict[str, Any] = Field(..., description="Vendor/customer impact analysis")

    # Configuration
    configuration_used: Dict[str, Any] = Field(..., description="Configuration used for analysis")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

# Legacy FileInfo model removed - using enhanced version above

# Standardized response models

class StandardResponse(BaseResponseModel):
    """Standard API response with data payload."""
    data: Optional[Any] = Field(None, description="Response data payload")
    message: Optional[str] = Field(None, description="Response message")

class ListResponse(BaseResponseModel):
    """Standard list response with pagination."""
    items: List[Any] = Field(default_factory=list, description="List items")
    total_count: int = Field(0, description="Total item count", ge=0)
    page: int = Field(1, description="Current page number", ge=1)
    page_size: int = Field(50, description="Items per page", ge=1, le=1000)
    has_more: bool = Field(False, description="Whether more pages available")

class ErrorResponse(BaseResponseModel):
    """Comprehensive error response with debugging information."""
    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    user_message: Optional[str] = Field(None, description="User-friendly error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: List[str] = Field(default_factory=list, description="Suggested solutions")
    trace_id: Optional[str] = Field(None, description="Error trace identifier")

class HealthResponse(BaseResponseModel):
    """Enhanced health check response."""
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="Application version")
    environment: Optional[str] = Field(None, description="Environment name")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds", ge=0)
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual health check results")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Health metrics")

class SessionResponse(BaseResponseModel):
    """Session-related response."""
    session: SessionInfo = Field(..., description="Session information")

class ProgressResponse(BaseResponseModel):
    """Progress update response."""
    progress: AnalysisProgress = Field(..., description="Progress information")

class AnalysisResultResponse(BaseResponseModel):
    """Analysis result response."""
    result: AnalysisResult = Field(..., description="Analysis result")

class FileUploadResponse(BaseResponseModel):
    """File upload response."""
    file_info: FileInfo = Field(..., description="Uploaded file information")
    session_id: Optional[str] = Field(None, description="Associated session ID")

class DebugFilesResponse(BaseResponseModel):
    """Debug files list response."""
    session_id: str = Field(..., description="Session ID")
    files: List[DebugFileInfo] = Field(..., description="List of available debug files")

class ConfigurationResponse(BaseResponseModel):
    """Configuration information response."""
    configuration: Dict[str, Any] = Field(..., description="Current configuration")
    environment: str = Field(..., description="Environment name")
    validation_status: str = Field(..., description="Configuration validation status")

# Analysis-specific request models

class StartAnalysisRequest(BaseRequestModel):
    """Request to start a new analysis."""
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    configuration: Optional[AnalysisConfigRequest] = Field(None, description="Analysis configuration overrides")
    user_id: Optional[str] = Field(None, description="User identifier")
    timeout_minutes: Optional[int] = Field(None, description="Session timeout override", ge=1, le=1440)

class ProcessAnalysisRequest(BaseRequestModel):
    """Request for processing analysis with files."""
    session_id: Optional[str] = Field(None, description="Existing session ID to use")
    configuration: Optional[AnalysisConfigRequest] = Field(None, description="Analysis configuration")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis")

# Revenue analysis models (enhanced existing ones)

class RevenueVarianceChange(BaseModel):
    """Enhanced month-over-month revenue change."""
    model_config = ConfigDict(str_strip_whitespace=True)

    period_from: str = Field(..., description="Source period")
    period_to: str = Field(..., description="Target period")
    previous_revenue: float = Field(..., description="Previous period revenue")
    current_revenue: float = Field(..., description="Current period revenue")
    absolute_change: float = Field(..., description="Absolute change in VND")
    percentage_change: float = Field(..., description="Percentage change")
    change_direction: str = Field(..., description="Direction: Increase/Decrease/No Change")
    risk_level: Optional[RiskLevel] = Field(None, description="Risk level assessment")
    confidence_score: Optional[float] = Field(None, description="Change confidence score", ge=0, le=1)

class EntityContribution(BaseModel):
    """Enhanced individual entity contribution to changes."""
    model_config = ConfigDict(str_strip_whitespace=True)

    entity: str = Field(..., description="Entity/vendor/customer name")
    previous_value: float = Field(..., description="Previous period value")
    current_value: float = Field(..., description="Current period value")
    absolute_change: float = Field(..., description="Absolute change")
    percentage_change: float = Field(..., description="Percentage change")
    contribution_to_period_change: float = Field(..., description="Contribution to total period change (%)")
    significance_rank: Optional[int] = Field(None, description="Significance ranking", ge=1)
    risk_indicators: List[str] = Field(default_factory=list, description="Risk indicator flags")

class NetEffectAnalysis(BaseModel):
    """Enhanced net effect analysis showing positive and negative contributors."""
    model_config = ConfigDict(str_strip_whitespace=True)

    period_from: str = Field(..., description="Source period")
    period_to: str = Field(..., description="Target period")
    total_change: float = Field(..., description="Total period change")
    net_effect_explanation: str = Field(..., description="Human-readable net effect explanation")
    positive_contributors: List[EntityContribution] = Field(default_factory=list, description="Positive contributors")
    negative_contributors: List[EntityContribution] = Field(default_factory=list, description="Negative contributors")
    total_positive_change: float = Field(..., description="Sum of all positive changes")
    total_negative_change: float = Field(..., description="Sum of all negative changes")
    entities_with_significant_change: int = Field(..., description="Number of entities with significant changes", ge=0)
    volatility_score: Optional[float] = Field(None, description="Period volatility score", ge=0, le=1)
    trend_direction: Optional[str] = Field(None, description="Overall trend direction")

class RevenueStreamAnalysis(BaseModel):
    """Enhanced revenue stream analysis for individual accounts."""
    model_config = ConfigDict(str_strip_whitespace=True)

    account_name: str = Field(..., description="Revenue account name")
    account_code: Optional[str] = Field(None, description="Account code")
    total_entities: int = Field(..., description="Total entities in this stream", ge=0)
    month_changes: List[RevenueVarianceChange] = Field(default_factory=list, description="Month-over-month changes")
    period_impacts: List[NetEffectAnalysis] = Field(default_factory=list, description="Net effect analysis by period")
    trend_analysis: Dict[str, Any] = Field(default_factory=dict, description="Trend analysis results")
    risk_assessment: Optional[RiskLevel] = Field(None, description="Stream risk level")
    data_quality_indicators: Dict[str, Any] = Field(default_factory=dict, description="Data quality metrics")

class RevenueVarianceAnalysisResponse(BaseResponseModel):
    """Enhanced complete revenue variance analysis response."""
    subsidiary: str = Field(..., description="Subsidiary name")
    filename: str = Field(..., description="Analyzed file name")
    analysis_type: AnalysisType = Field(default=AnalysisType.REVENUE_VARIANCE, description="Type of analysis performed")
    months_analyzed: List[str] = Field(..., description="List of months analyzed")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time", ge=0)

    # Executive Summary
    analysis_summary: Dict[str, Any] = Field(..., description="High-level analysis summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from analysis")
    risk_assessment: Optional[RiskLevel] = Field(None, description="Overall risk assessment")

    # Core Analysis Results
    total_revenue_analysis: Dict[str, Any] = Field(..., description="Total revenue month-over-month analysis")
    revenue_stream_analysis: Dict[str, Any] = Field(..., description="Individual revenue stream analysis")
    vendor_customer_impact: Dict[str, Any] = Field(..., description="Vendor/customer impact analysis")

    # Quality and Validation
    data_quality_report: Dict[str, Any] = Field(default_factory=dict, description="Data quality assessment")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

    # Configuration and Metadata
    configuration_used: Dict[str, Any] = Field(..., description="Configuration used for analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis metadata")

# Specialized response models for different endpoints

class SessionListResponse(ListResponse):
    """Session list response with typed items."""
    items: List[SessionInfo] = Field(default_factory=list, description="Session list")

class ProgressStreamResponse(BaseModel):
    """Streaming progress response."""
    model_config = ConfigDict(str_strip_whitespace=True)

    type: ProgressType = Field(..., description="Progress message type")
    message: str = Field(..., description="Progress message")
    percentage: Optional[int] = Field(None, description="Progress percentage", ge=0, le=100)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    session_id: str = Field(..., description="Associated session ID")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional progress data")

class FileValidationResponse(BaseResponseModel):
    """File validation response."""
    file_info: FileInfo = Field(..., description="Validated file information")
    validation_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation results")
    security_scan_results: Dict[str, Any] = Field(default_factory=dict, description="Security scan results")

# Legacy compatibility models (kept for backward compatibility)

class AnalysisSession(BaseModel):
    """Legacy analysis session information (deprecated)."""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")

    @field_validator('created_at', mode='before')
    @classmethod
    def convert_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v