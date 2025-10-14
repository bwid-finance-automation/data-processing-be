# app/services/interfaces.py
"""
Service interfaces and abstractions for better testability and loose coupling.

This module defines abstract base classes and protocols that establish contracts
for service implementations, enabling dependency injection and easy mocking.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, Union, AsyncGenerator
from pathlib import Path
import pandas as pd
from fastapi import UploadFile

from ..models.analysis import (
    AnalysisResult, AnalysisProgress, SessionInfo,
    HealthResponse, FileInfo, DebugFileInfo
)

class IFileProcessingService(Protocol):
    """Interface for file processing operations."""

    async def validate_file(self, file: UploadFile) -> FileInfo:
        """Validate uploaded file and return file information."""
        ...

    async def process_excel_file(self, file: UploadFile) -> Dict[str, pd.DataFrame]:
        """Process Excel file and return sheet data."""
        ...

    async def save_uploaded_file(self, file: UploadFile, session_id: str) -> Path:
        """Save uploaded file to session directory."""
        ...

    def get_file_info(self, file_path: Path) -> FileInfo:
        """Get information about a file."""
        ...

class IAnalysisService(Protocol):
    """Interface for analysis operations."""

    async def start_analysis(
        self,
        files: List[UploadFile],
        config: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Start a new analysis session."""
        ...

    async def get_analysis_progress(self, session_id: str) -> AnalysisProgress:
        """Get progress of an ongoing analysis."""
        ...

    async def get_analysis_result(self, session_id: str) -> Optional[AnalysisResult]:
        """Get completed analysis result."""
        ...

    async def cancel_analysis(self, session_id: str) -> bool:
        """Cancel an ongoing analysis."""
        ...

    def list_sessions(self) -> List[SessionInfo]:
        """List all analysis sessions."""
        ...

    def cleanup_old_sessions(self, max_age_minutes: int) -> int:
        """Clean up old sessions and return count of cleaned sessions."""
        ...

class IRevenueAnalysisService(Protocol):
    """Interface for revenue-specific analysis operations."""

    async def analyze_revenue_variance(
        self,
        excel_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """Perform revenue variance analysis."""
        ...

    async def analyze_revenue_trends(
        self,
        excel_data: Dict[str, pd.DataFrame],
        periods: int = 12
    ) -> Dict[str, Any]:
        """Analyze revenue trends over time."""
        ...

    async def detect_revenue_anomalies(
        self,
        excel_data: Dict[str, pd.DataFrame],
        sensitivity: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in revenue data."""
        ...

class IAIAnalysisService(Protocol):
    """Interface for AI-powered analysis operations."""

    async def analyze_with_ai(
        self,
        data: Dict[str, Any],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Perform AI analysis on data."""
        ...

    async def generate_insights(
        self,
        analysis_result: AnalysisResult
    ) -> Dict[str, Any]:
        """Generate AI insights from analysis results."""
        ...

    async def validate_analysis_quality(
        self,
        analysis_result: AnalysisResult
    ) -> Dict[str, Any]:
        """Validate analysis quality using AI."""
        ...

class ISessionService(Protocol):
    """Interface for session management operations."""

    def create_session(self, user_id: Optional[str] = None) -> SessionInfo:
        """Create a new session."""
        ...

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        ...

    def update_session_progress(self, session_id: str, progress: AnalysisProgress) -> bool:
        """Update session progress."""
        ...

    def end_session(self, session_id: str, result: Optional[AnalysisResult] = None) -> bool:
        """End a session with optional result."""
        ...

    def list_active_sessions(self) -> List[SessionInfo]:
        """List all active sessions."""
        ...

class IHealthService(Protocol):
    """Interface for health monitoring operations."""

    async def check_basic_health(self) -> HealthResponse:
        """Perform basic health check."""
        ...

    async def check_detailed_health(self) -> Dict[str, Any]:
        """Perform detailed health check with all components."""
        ...

    async def check_configuration_health(self) -> Dict[str, Any]:
        """Check configuration health and validity."""
        ...

    async def check_service_dependencies(self) -> Dict[str, bool]:
        """Check if all service dependencies are healthy."""
        ...

class ILoggingService(Protocol):
    """Interface for logging operations."""

    async def stream_logs(self, session_id: str) -> AsyncGenerator[str, None]:
        """Stream logs for a session."""
        ...

    def get_log_file_path(self, session_id: str) -> Optional[Path]:
        """Get log file path for a session."""
        ...

    def archive_session_logs(self, session_id: str) -> bool:
        """Archive logs for a completed session."""
        ...

class IConfigurationService(Protocol):
    """Interface for configuration management operations."""

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get current analysis configuration."""
        ...

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration."""
        ...

    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        ...

    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment configuration information."""
        ...

class IDebugService(Protocol):
    """Interface for debugging and development operations."""

    def list_debug_files(self, session_id: str) -> List[DebugFileInfo]:
        """List debug files for a session."""
        ...

    def get_debug_file_path(self, session_id: str, file_key: str) -> Optional[Path]:
        """Get path to a debug file."""
        ...

    def create_debug_export(self, session_id: str, include_data: bool = True) -> Path:
        """Create debug export package."""
        ...

# Abstract base classes for common service implementations

class BaseService(ABC):
    """Base service class with common functionality."""

    def __init__(self):
        from ..utils.logging_config import get_logger
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name for identification."""
        pass

    def log_operation(self, operation: str, **kwargs):
        """Log a service operation with context."""
        self.logger.info(
            f"{self.get_service_name()}: {operation}",
            extra={
                "service": self.get_service_name(),
                "operation": operation,
                **kwargs
            }
        )

    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log a service error with context."""
        self.logger.error(
            f"{self.get_service_name()}: {operation} failed - {str(error)}",
            extra={
                "service": self.get_service_name(),
                "operation": operation,
                "error": str(error),
                "error_type": type(error).__name__,
                **kwargs
            },
            exc_info=True
        )

class FileProcessingServiceBase(BaseService):
    """Base class for file processing services."""

    def get_service_name(self) -> str:
        return "FileProcessingService"

    @abstractmethod
    async def validate_file(self, file: UploadFile) -> FileInfo:
        """Validate uploaded file and return file information."""
        pass

    @abstractmethod
    async def process_excel_file(self, file: UploadFile) -> Dict[str, pd.DataFrame]:
        """Process Excel file and return sheet data."""
        pass

class AnalysisServiceBase(BaseService):
    """Base class for analysis services."""

    def get_service_name(self) -> str:
        return "AnalysisService"

    @abstractmethod
    async def start_analysis(
        self,
        files: List[UploadFile],
        config: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Start a new analysis session."""
        pass

    @abstractmethod
    async def get_analysis_result(self, session_id: str) -> Optional[AnalysisResult]:
        """Get completed analysis result."""
        pass

class RevenueAnalysisServiceBase(BaseService):
    """Base class for revenue analysis services."""

    def get_service_name(self) -> str:
        return "RevenueAnalysisService"

    @abstractmethod
    async def analyze_revenue_variance(
        self,
        excel_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """Perform revenue variance analysis."""
        pass

class AIAnalysisServiceBase(BaseService):
    """Base class for AI analysis services."""

    def get_service_name(self) -> str:
        return "AIAnalysisService"

    @abstractmethod
    async def analyze_with_ai(
        self,
        data: Dict[str, Any],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Perform AI analysis on data."""
        pass

class SessionServiceBase(BaseService):
    """Base class for session management services."""

    def get_service_name(self) -> str:
        return "SessionService"

    @abstractmethod
    def create_session(self, user_id: Optional[str] = None) -> SessionInfo:
        """Create a new session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        pass