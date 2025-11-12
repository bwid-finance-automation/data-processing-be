# app/core/dependencies.py
"""Dependency injection for FastAPI."""

from functools import lru_cache
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Optional

from .config import get_settings, Settings
from app.application.finance.use_cases.analysis_service import analysis_service, AnalysisService

# Security scheme (optional - for future authentication)
security = HTTPBearer(auto_error=False)

@lru_cache()
def get_analysis_service() -> AnalysisService:
    """Get analysis service singleton."""
    return analysis_service

def get_current_session(session_id: str, service: AnalysisService = Depends(get_analysis_service)):
    """Dependency to get and validate current session."""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )
    return session

async def validate_file_upload(file_content: bytes, filename: str, settings: Settings = Depends(get_settings)):
    """Validate uploaded file."""
    # Check file size
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )

    # Check file extension
    if filename:
        extension = "." + filename.split(".")[-1].lower()
        if extension not in settings.allowed_file_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_extensions)}"
            )

    return True