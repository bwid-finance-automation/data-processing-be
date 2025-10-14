# app/api/health.py
"""Health check endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from ..models.analysis import HealthResponse
from ..core.config import get_settings, Settings
from ..middleware.config_middleware import create_config_health_response

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Basic health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version
    )

@router.get("/health/config")
async def config_health_check():
    """Detailed configuration health check endpoint."""
    health_data = create_config_health_response()

    # Return appropriate HTTP status based on health
    status_code = 200 if health_data["status"] == "healthy" else 503

    return JSONResponse(
        status_code=status_code,
        content=health_data
    )

@router.get("/health/detailed")
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """Comprehensive health check with configuration details."""
    config_health = create_config_health_response()

    return JSONResponse(
        content={
            "status": "ok" if config_health["status"] == "healthy" else "degraded",
            "version": settings.app_version,
            "timestamp": config_health["timestamp"],
            "services": {
                "api": "healthy",
                "configuration": config_health["status"],
                "file_processing": "healthy",
                "ai_analysis": "healthy" if config_health["checks"].get("ai_analysis_enabled") else "disabled"
            },
            "configuration": config_health.get("configuration_summary", {}),
            "environment": config_health.get("environment", "unknown")
        }
    )