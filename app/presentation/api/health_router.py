# app/presentation/api/health_router.py
"""Health check endpoints."""

import os
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.presentation.schemas.analysis import HealthResponse
from app.core.config import get_settings, Settings
from app.presentation.middleware.config_middleware import create_config_health_response

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

@router.get("/ai-config")
async def get_ai_config():
    """Get current AI/OpenAI configuration for display in frontend."""
    # Model info
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    # GPT-5 specific settings
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "medium")
    service_tier = os.getenv("OPENAI_SERVICE_TIER", "auto")

    # Determine model family
    is_gpt5 = model.startswith("gpt-5")
    model_family = "GPT-5 Series" if is_gpt5 else "Legacy GPT-4"

    return JSONResponse(
        content={
            "model": model,
            "model_family": model_family,
            "is_gpt5": is_gpt5,
            "reasoning_effort": reasoning_effort if is_gpt5 else None,
            "service_tier": service_tier,
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
        }
    )