# app/middleware/validation_middleware.py
"""Validation middleware for API endpoints."""

import time
from typing import Dict, Any
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.exceptions import ValidationError, create_validation_error
from ..utils.input_sanitization import RequestValidator
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate requests and enhance security."""

    def __init__(self, app, max_request_size: int = 100 * 1024 * 1024):  # 100MB
        super().__init__(app)
        self.max_request_size = max_request_size
        self.request_counts: Dict[str, list] = {}  # Simple rate limiting
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_minute = 100

    async def dispatch(self, request: Request, call_next):
        """Process request through validation middleware."""
        start_time = time.time()

        try:
            # 1. Basic request validation
            await self._validate_basic_request(request)

            # 2. Rate limiting
            await self._check_rate_limit(request)

            # 3. Content type validation for specific endpoints
            await self._validate_content_type(request)

            # 4. Request size validation
            await self._validate_request_size(request)

            # Process the request
            response = await call_next(request)

            # 5. Log successful requests
            process_time = time.time() - start_time
            logger.info(f"Request processed: {request.method} {request.url.path}", extra={
                "method": request.method,
                "path": request.url.path,
                "process_time": process_time,
                "status_code": response.status_code,
                "client_ip": self._get_client_ip(request)
            })

            return response

        except HTTPException:
            # Re-raise HTTP exceptions (they're already handled)
            raise

        except Exception as e:
            # Log unexpected errors in middleware
            logger.error(f"Middleware error: {str(e)}", extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": self._get_client_ip(request),
                "process_time": time.time() - start_time
            }, exc_info=True)

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error_code": "MIDDLEWARE_ERROR",
                    "message": "Request processing failed. Please try again.",
                    "suggestions": ["Check your request format and try again"]
                }
            )

    async def _validate_basic_request(self, request: Request) -> None:
        """Validate basic request properties."""
        # Check for suspicious headers
        suspicious_headers = ['x-forwarded-host', 'x-original-url', 'x-rewrite-url']
        for header in suspicious_headers:
            if header in request.headers:
                logger.warning(f"Suspicious header detected: {header}", extra={
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("user-agent", "unknown")
                })

        # Validate User-Agent (basic check)
        user_agent = request.headers.get("user-agent", "")
        if not user_agent:
            logger.warning("Request without User-Agent header", extra={
                "client_ip": self._get_client_ip(request),
                "path": request.url.path
            })

        # Check for excessively long headers
        for name, value in request.headers.items():
            if len(value) > 8192:  # 8KB limit per header
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Header '{name}' is too long"
                )

    async def _check_rate_limit(self, request: Request) -> None:
        """Simple rate limiting by client IP."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Initialize or clean old requests for this IP
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # Remove requests older than the window
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < self.rate_limit_window
        ]

        # Check if rate limit exceeded
        if len(self.request_counts[client_ip]) >= self.max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}", extra={
                "client_ip": client_ip,
                "request_count": len(self.request_counts[client_ip]),
                "path": request.url.path
            })

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait a moment and try again."
            )

        # Add current request timestamp
        self.request_counts[client_ip].append(current_time)

    async def _validate_content_type(self, request: Request) -> None:
        """Validate content type for specific endpoints."""
        path = request.url.path
        method = request.method

        # Only validate content type for POST/PUT requests with body
        if method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").lower()

            # Billing process endpoint uses JSON
            if "/billing/process" in path:
                # Skip validation for billing process endpoint - it handles its own validation
                pass

            # File upload endpoints
            elif "/process" in path or "/start_analysis" in path or "/start-analysis" in path or "/analyze-revenue" in path or "/process-contract" in path:
                expected_types = ["multipart/form-data", "application/x-www-form-urlencoded"]
                if not any(expected in content_type for expected in expected_types):
                    # Allow requests without explicit content-type for file uploads
                    # (some clients don't set it properly for multipart)
                    if content_type and "multipart" not in content_type:
                        RequestValidator.validate_content_type(content_type, expected_types)

            # JSON endpoints (exclude file upload endpoints)
            elif "/api/" in path and "upload" not in path and "/process" not in path and "/compare" not in path:
                if content_type and "application/json" not in content_type:
                    RequestValidator.validate_content_type(content_type, ["application/json"])

    async def _validate_request_size(self, request: Request) -> None:
        """Validate overall request size."""
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                RequestValidator.validate_request_size(size, self.max_request_size)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header"
                )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IP first (from proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Check for real IP
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Check if this is a docs endpoint that needs relaxed CSP for Swagger UI
        is_docs_endpoint = request.url.path in ["/docs", "/redoc"] or request.url.path.startswith("/openapi")

        if is_docs_endpoint:
            # Relaxed CSP for API documentation (Swagger UI/ReDoc)
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
                "img-src 'self' data: fastapi.tiangolo.com; "
                "font-src 'self' data:; "
                "connect-src 'self';"
            )
        else:
            # Strict CSP for all other endpoints
            csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"

        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": csp,
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response

# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        start_time = time.time()

        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url}", extra={
            "method": request.method,
            "url": str(request.url),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_length": request.headers.get("content-length", 0)
        })

        # Process request
        response = await call_next(request)

        # Calculate process time
        process_time = time.time() - start_time

        # Log response
        logger.info(f"Response: {response.status_code}", extra={
            "status_code": response.status_code,
            "process_time": round(process_time, 3),
            "method": request.method,
            "path": request.url.path
        })

        # Add process time header
        response.headers["X-Process-Time"] = str(round(process_time, 3))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"