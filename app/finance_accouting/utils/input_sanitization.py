# app/utils/input_sanitization.py
"""Input sanitization and validation utilities."""

import re
import html
from typing import Any, Optional, List, Dict, Union
from decimal import Decimal, InvalidOperation
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status

from ..core.exceptions import FileProcessingError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class AnalysisParameterValidator(BaseModel):
    """Validated analysis parameters with sanitization."""

    materiality_vnd: Optional[float] = Field(None, ge=0, le=1e15, description="Materiality threshold in VND")
    recurring_pct_threshold: Optional[float] = Field(None, ge=0, le=1, description="Recurring percentage threshold (0-1)")
    revenue_opex_pct_threshold: Optional[float] = Field(None, ge=0, le=1, description="Revenue OPEX percentage threshold (0-1)")
    bs_pct_threshold: Optional[float] = Field(None, ge=0, le=1, description="Balance sheet percentage threshold (0-1)")
    min_trend_periods: Optional[int] = Field(None, ge=1, le=50, description="Minimum trend periods")
    gm_drop_threshold_pct: Optional[float] = Field(None, ge=0, le=1, description="Gross margin drop threshold (0-1)")

    # String fields that need sanitization
    recurring_code_prefixes: Optional[str] = Field(None, max_length=500, description="Comma-separated recurring code prefixes")
    dep_pct_only_prefixes: Optional[str] = Field(None, max_length=500, description="Depreciation percentage only prefixes")
    customer_column_hints: Optional[str] = Field(None, max_length=1000, description="Customer column hints")

    @validator('recurring_code_prefixes', 'dep_pct_only_prefixes', 'customer_column_hints')
    def sanitize_string_fields(cls, v):
        """Sanitize string input fields."""
        if v is None:
            return v
        return sanitize_string_input(v)

    @validator('materiality_vnd')
    def validate_materiality(cls, v):
        """Validate materiality amount."""
        if v is not None and v < 1000:
            logger.warning(f"Very small materiality threshold: {v}")
        return v

class InputSanitizer:
    """Comprehensive input sanitization utility."""

    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
        r"(?i)(or\s+1=1|and\s+1=1|'|\"|\-\-|\/\*|\*\/)",
        r"(?i)(exec|execute|sp_|xp_)"
    ]

    XSS_PATTERNS = [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)<iframe[^>]*>.*?</iframe>",
        r"(?i)javascript:",
        r"(?i)on\w+\s*="
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.[\\/]",
        r"[\\/]\.\.[\\/]",
        r"\.\.\\",
        r"\.\.\/"
    ]

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize uploaded filename."""
        if not filename:
            raise FileProcessingError("Filename cannot be empty")

        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Remove path traversal attempts
        sanitized = re.sub(r'\.\.[\\/]', '_', sanitized)

        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')

        # Ensure it's not empty after sanitization
        if not sanitized or sanitized.isspace():
            raise FileProcessingError("Filename becomes empty after sanitization")

        return sanitized

    @staticmethod
    def detect_injection_attempts(input_string: str) -> List[str]:
        """Detect potential injection attempts in input."""
        threats = []

        # Check for SQL injection
        for pattern in InputSanitizer.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_string):
                threats.append("SQL injection attempt detected")
                break

        # Check for XSS
        for pattern in InputSanitizer.XSS_PATTERNS:
            if re.search(pattern, input_string):
                threats.append("XSS attempt detected")
                break

        # Check for path traversal
        for pattern in InputSanitizer.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, input_string):
                threats.append("Path traversal attempt detected")
                break

        return threats

    @staticmethod
    def sanitize_numeric_input(value: Any, field_name: str = "value") -> Optional[float]:
        """Safely convert and validate numeric input."""
        if value is None or value == "":
            return None

        try:
            # Handle string representations
            if isinstance(value, str):
                # Remove common formatting
                cleaned = value.replace(',', '').replace(' ', '').strip()

                # Check for dangerous patterns
                threats = InputSanitizer.detect_injection_attempts(cleaned)
                if threats:
                    raise FileProcessingError(f"Security threat detected in {field_name}: {threats[0]}")

                if not cleaned:
                    return None

                # Convert to decimal for precision
                decimal_value = Decimal(cleaned)
                float_value = float(decimal_value)

            elif isinstance(value, (int, float)):
                float_value = float(value)
            else:
                raise ValueError(f"Unsupported type for numeric conversion: {type(value)}")

            # Validate range
            if not (-1e15 <= float_value <= 1e15):
                raise ValueError(f"Value out of acceptable range: {float_value}")

            return float_value

        except (ValueError, InvalidOperation, OverflowError) as e:
            raise FileProcessingError(f"Invalid numeric value for {field_name}: {value} ({str(e)})")

    @staticmethod
    def validate_percentage(value: Any, field_name: str = "percentage") -> Optional[float]:
        """Validate percentage values (0-100 or 0-1)."""
        if value is None:
            return None

        numeric_value = InputSanitizer.sanitize_numeric_input(value, field_name)
        if numeric_value is None:
            return None

        # Auto-detect if it's in 0-100 range vs 0-1 range
        if 0 <= numeric_value <= 1:
            return numeric_value
        elif 0 <= numeric_value <= 100:
            return numeric_value / 100.0
        else:
            raise FileProcessingError(f"Percentage value out of range for {field_name}: {numeric_value} (expected 0-100 or 0-1)")

def sanitize_string_input(input_string: str, max_length: int = 1000) -> str:
    """Sanitize string input with security checks."""
    if not input_string:
        return ""

    # Check for security threats
    threats = InputSanitizer.detect_injection_attempts(input_string)
    if threats:
        logger.warning(f"Security threat detected in input: {threats[0]}")
        raise FileProcessingError(f"Invalid input detected: {threats[0]}")

    # HTML escape
    sanitized = html.escape(input_string.strip())

    # Remove control characters except common whitespace
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")

    return sanitized

def sanitize_code_prefixes(prefixes_string: str) -> List[str]:
    """Sanitize and validate account code prefixes."""
    if not prefixes_string:
        return []

    sanitized = sanitize_string_input(prefixes_string, max_length=500)

    # Split and clean individual prefixes
    prefixes = []
    for prefix in sanitized.split(','):
        clean_prefix = prefix.strip()

        # Validate prefix format (should be alphanumeric)
        if not re.match(r'^[a-zA-Z0-9]+$', clean_prefix):
            logger.warning(f"Invalid account code prefix format: {clean_prefix}")
            continue

        if len(clean_prefix) > 10:  # Reasonable limit for account codes
            logger.warning(f"Account code prefix too long: {clean_prefix}")
            continue

        prefixes.append(clean_prefix)

    return prefixes

def validate_analysis_parameters(params: Dict[str, Any]) -> AnalysisParameterValidator:
    """Validate and sanitize analysis parameters."""
    try:
        # Sanitize numeric parameters
        for key in ['materiality_vnd', 'min_trend_periods']:
            if key in params and params[key] is not None:
                params[key] = InputSanitizer.sanitize_numeric_input(params[key], key)

        # Sanitize percentage parameters
        for key in ['recurring_pct_threshold', 'revenue_opex_pct_threshold',
                   'bs_pct_threshold', 'gm_drop_threshold_pct']:
            if key in params and params[key] is not None:
                params[key] = InputSanitizer.validate_percentage(params[key], key)

        # Create validated model
        validated = AnalysisParameterValidator(**params)

        logger.info("Analysis parameters validated successfully")
        return validated

    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid analysis parameters: {str(e)}"
        )

def sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent injection attacks."""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty"
        )

    # Session IDs should only contain alphanumeric characters, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    if len(session_id) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID too long"
        )

    return session_id

# Rate limiting and abuse detection
class RequestValidator:
    """Request-level validation and abuse detection."""

    @staticmethod
    def validate_content_type(content_type: str, expected_types: List[str]) -> None:
        """Validate request content type."""
        if not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content-Type header is required"
            )

        if not any(expected in content_type.lower() for expected in expected_types):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported content type: {content_type}. Expected: {', '.join(expected_types)}"
            )

    @staticmethod
    def validate_request_size(content_length: Optional[int], max_size: int) -> None:
        """Validate overall request size."""
        if content_length and content_length > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = content_length / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large: {actual_mb:.1f}MB. Maximum: {max_mb:.0f}MB"
            )