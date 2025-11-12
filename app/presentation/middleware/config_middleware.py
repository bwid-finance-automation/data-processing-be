# app/middleware/config_middleware.py
"""Configuration validation and monitoring middleware."""

import time
from typing import Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import status

from app.core.unified_config import get_unified_config, UnifiedConfig
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

class ConfigValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate and monitor configuration health.

    This middleware ensures the application configuration is valid and
    provides configuration health monitoring.
    """

    def __init__(self, app):
        super().__init__(app)
        self.config_cache = None
        self.last_config_check = 0
        self.config_check_interval = 300  # 5 minutes
        self.config_health = True

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with configuration validation."""
        start_time = time.time()

        try:
            # Check configuration health periodically
            current_time = time.time()
            if (current_time - self.last_config_check) > self.config_check_interval:
                await self._validate_configuration()
                self.last_config_check = current_time

            # If configuration is unhealthy, return error for critical endpoints
            if not self.config_health and self._is_critical_endpoint(request):
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "success": False,
                        "error_code": "CONFIGURATION_ERROR",
                        "message": "Service temporarily unavailable due to configuration issues",
                        "suggestions": [
                            "Please check your configuration settings",
                            "Contact support if the issue persists"
                        ]
                    }
                )

            # Add configuration context to request
            request.state.config = self.config_cache or get_unified_config()

            # Process the request
            response = await call_next(request)

            # Add configuration headers to response (for debugging)
            if hasattr(request.state, 'config') and request.state.config.app.debug:
                response.headers["X-Config-Status"] = "healthy" if self.config_health else "unhealthy"
                response.headers["X-Config-Version"] = request.state.config.app.app_version

            return response

        except Exception as e:
            logger.error(f"Configuration middleware error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error_code": "MIDDLEWARE_ERROR",
                    "message": "Configuration middleware error"
                }
            )

    async def _validate_configuration(self) -> None:
        """Validate current configuration and update health status."""
        try:
            # Load and validate configuration
            config = get_unified_config()
            self.config_cache = config

            # Perform health checks
            validation_errors = []

            # Check file processing limits
            if config.file_processing.max_file_size > 1024 * 1024 * 1024:  # 1GB
                validation_errors.append("File size limit too high (>1GB)")

            # Check AI analysis configuration
            if config.ai_analysis.enable_ai_analysis and not config.ai_analysis.llm_model:
                validation_errors.append("AI analysis enabled but no LLM model specified")

            # Check session configuration
            if config.security.session_timeout_minutes > 1440:  # 24 hours
                validation_errors.append("Session timeout too long (>24 hours)")

            # Check rate limiting configuration
            if config.security.rate_limit_requests_per_minute > 10000:
                validation_errors.append("Rate limit too high (>10000 requests/minute)")

            # Validate thresholds are reasonable
            if config.core_analysis.materiality_vnd > 1e15:
                validation_errors.append("Materiality threshold unreasonably high")

            # Check required directories exist
            if config.app.log_file:
                log_dir = config.app.log_file.split('/')[0] if '/' in config.app.log_file else 'logs'
                try:
                    import os
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    validation_errors.append(f"Cannot create log directory: {e}")

            # Update health status
            if validation_errors:
                logger.warning(f"Configuration validation issues: {'; '.join(validation_errors)}")
                self.config_health = len(validation_errors) < 3  # Healthy if minor issues only
            else:
                self.config_health = True
                logger.debug("Configuration validation passed")

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}", exc_info=True)
            self.config_health = False

    def _is_critical_endpoint(self, request: Request) -> bool:
        """Check if the endpoint is critical and requires healthy configuration."""
        critical_paths = [
            "/api/process",
            "/api/start-analysis",
            "/process",
            "/start_analysis"
        ]

        return any(request.url.path.startswith(path) for path in critical_paths)

class ConfigMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor configuration usage and performance.

    This middleware tracks how configuration is being used and
    identifies potential optimization opportunities.
    """

    def __init__(self, app):
        super().__init__(app)
        self.config_access_count = {}
        self.config_performance_metrics = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Monitor configuration access during request processing."""
        start_time = time.time()

        # Track configuration access
        endpoint = request.url.path
        if endpoint not in self.config_access_count:
            self.config_access_count[endpoint] = 0
        self.config_access_count[endpoint] += 1

        response = await call_next(request)

        # Track performance metrics
        process_time = time.time() - start_time
        if endpoint not in self.config_performance_metrics:
            self.config_performance_metrics[endpoint] = []
        self.config_performance_metrics[endpoint].append(process_time)

        # Log slow requests (> 5 seconds)
        if process_time > 5.0:
            logger.warning(f"Slow request detected: {endpoint} took {process_time:.2f}s")

        return response

    def get_metrics(self) -> Dict[str, Any]:
        """Get configuration monitoring metrics."""
        return {
            "config_access_count": self.config_access_count.copy(),
            "config_performance_metrics": {
                endpoint: {
                    "count": len(times),
                    "avg_time": sum(times) / len(times) if times else 0,
                    "max_time": max(times) if times else 0,
                    "min_time": min(times) if times else 0
                }
                for endpoint, times in self.config_performance_metrics.items()
            }
        }

# Configuration health check endpoint helper
def create_config_health_response() -> Dict[str, Any]:
    """Create a configuration health check response."""
    try:
        config = get_unified_config()

        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": config.app.app_version,
            "environment": "debug" if config.app.debug else "production",
            "checks": {
                "config_loaded": True,
                "ai_analysis_enabled": config.ai_analysis.enable_ai_analysis,
                "file_processing_configured": bool(config.file_processing.max_file_size),
                "security_configured": bool(config.security.session_timeout_minutes),
                "analysis_configured": bool(config.core_analysis.materiality_vnd)
            }
        }

        # Add configuration summary (non-sensitive data only)
        health_data["configuration_summary"] = {
            "max_file_size_mb": config.file_processing.max_file_size / (1024 * 1024),
            "session_timeout_minutes": config.security.session_timeout_minutes,
            "ai_model": config.ai_analysis.llm_model if config.ai_analysis.enable_ai_analysis else None,
            "debug_mode": config.app.debug,
            "rate_limit_per_minute": config.security.rate_limit_requests_per_minute
        }

        return health_data

    except Exception as e:
        logger.error(f"Configuration health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "checks": {
                "config_loaded": False
            }
        }