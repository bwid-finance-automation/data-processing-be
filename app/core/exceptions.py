# app/core/exceptions.py
"""Custom exceptions and error handlers with business-friendly messages."""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Union, Dict, Any, Optional
from datetime import datetime

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

class AnalysisError(Exception):
    """Custom exception for analysis errors with business-friendly messaging."""
    def __init__(
        self,
        message: str,
        details: str = None,
        user_message: str = None,
        error_code: str = None,
        suggestions: list = None
    ):
        self.message = message
        self.details = details
        self.user_message = user_message or self._generate_user_friendly_message(message)
        self.error_code = error_code or self.__class__.__name__.upper()
        self.suggestions = suggestions or []
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)

    def _generate_user_friendly_message(self, technical_message: str) -> str:
        """Generate user-friendly message from technical error."""
        user_friendly_map = {
            "file not found": "The uploaded file could not be found or accessed.",
            "permission denied": "We don't have permission to access this file.",
            "out of memory": "The file is too large to process. Please try a smaller file.",
            "timeout": "The analysis is taking longer than expected. Please try again.",
            "connection": "There's a network connectivity issue. Please check your connection.",
            "invalid format": "The file format is not supported or the file may be corrupted.",
            "missing data": "Required data is missing from the uploaded file.",
            "parsing error": "We couldn't read the data in your file. Please check the file format."
        }

        message_lower = technical_message.lower()
        for key, friendly_msg in user_friendly_map.items():
            if key in message_lower:
                return friendly_msg

        return "An error occurred while processing your request. Please try again or contact support."

class FileProcessingError(AnalysisError):
    """Exception for file processing errors with specific guidance."""
    def __init__(self, message: str, details: str = None, file_name: str = None):
        suggestions = [
            "Ensure the file is a valid Excel file (.xlsx or .xls)",
            "Check that the file contains the required sheets: 'BS Breakdown' and 'PL Breakdown'",
            "Verify the file is not corrupted by opening it in Excel first",
            "Make sure the file size is under 100MB"
        ]

        user_message = self._get_file_error_message(message, file_name)

        super().__init__(
            message=message,
            details=details,
            user_message=user_message,
            error_code="FILE_PROCESSING_ERROR",
            suggestions=suggestions
        )

    def _get_file_error_message(self, message: str, file_name: str = None) -> str:
        """Generate specific file error messages."""
        file_ref = f" in file '{file_name}'" if file_name else ""

        if "empty" in message.lower():
            return f"The uploaded file{file_ref} appears to be empty. Please upload a file with data."
        elif "size" in message.lower():
            return f"The file{file_ref} is too large. Please use a file smaller than 100MB."
        elif "format" in message.lower() or "signature" in message.lower():
            return f"The file{file_ref} is not a valid Excel file. Please upload a .xlsx or .xls file."
        elif "sheet" in message.lower():
            return f"The file{file_ref} is missing required worksheets. Please ensure it contains 'BS Breakdown' and 'PL Breakdown' sheets."
        elif "permission" in message.lower():
            return f"We cannot access the file{file_ref}. Please check file permissions and try again."
        else:
            return f"There was a problem processing the file{file_ref}. Please check the file and try again."

class ConfigurationError(AnalysisError):
    """Exception for configuration errors with helpful guidance."""
    def __init__(self, message: str, details: str = None, parameter: str = None):
        suggestions = [
            "Check that all required parameters are provided",
            "Verify numeric values are within acceptable ranges",
            "Ensure percentage values are between 0 and 100",
            "Contact support if you need help with parameter settings"
        ]

        param_ref = f" for parameter '{parameter}'" if parameter else ""
        user_message = f"There's an issue with the analysis configuration{param_ref}. Please check your settings and try again."

        super().__init__(
            message=message,
            details=details,
            user_message=user_message,
            error_code="CONFIGURATION_ERROR",
            suggestions=suggestions
        )

class SessionError(AnalysisError):
    """Exception for session-related errors."""
    def __init__(self, message: str, details: str = None, session_id: str = None):
        suggestions = [
            "Start a new analysis session",
            "Check that you're using the correct session ID",
            "Sessions expire after 60 minutes of inactivity"
        ]

        session_ref = f" (ID: {session_id})" if session_id else ""
        user_message = f"There's an issue with your analysis session{session_ref}. Please start a new analysis."

        super().__init__(
            message=message,
            details=details,
            user_message=user_message,
            error_code="SESSION_ERROR",
            suggestions=suggestions
        )

class ValidationError(AnalysisError):
    """Exception for input validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        field_ref = f" in field '{field}'" if field else ""
        value_ref = f" (value: {value})" if value else ""

        user_message = f"Invalid input{field_ref}{value_ref}. Please check your data and try again."

        suggestions = [
            "Check that all required fields are filled",
            "Verify numeric values are valid numbers",
            "Ensure dates are in the correct format",
            "Make sure text fields don't contain special characters"
        ]

        super().__init__(
            message=message,
            user_message=user_message,
            error_code="VALIDATION_ERROR",
            suggestions=suggestions
        )

class DataQualityError(AnalysisError):
    """Exception for data quality issues."""
    def __init__(self, message: str, sheet_name: str = None, details: str = None):
        sheet_ref = f" in sheet '{sheet_name}'" if sheet_name else ""

        user_message = f"There's an issue with the data quality{sheet_ref}. The analysis may not be accurate."

        suggestions = [
            "Check for missing or incomplete data in your Excel file",
            "Verify that financial data is in the expected format",
            "Ensure account codes follow the expected pattern",
            "Review period/date columns for consistency"
        ]

        super().__init__(
            message=message,
            details=details,
            user_message=user_message,
            error_code="DATA_QUALITY_ERROR",
            suggestions=suggestions
        )

# Error handlers with enhanced user experience
async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    """Handle custom analysis errors with user-friendly messages."""
    logger.error(f"Analysis error: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "timestamp": exc.timestamp,
        "request_url": str(request.url)
    })

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.user_message,
            "technical_details": exc.details if hasattr(exc, 'details') else None,
            "suggestions": exc.suggestions,
            "timestamp": exc.timestamp,
            "type": exc.__class__.__name__
        }
    )

async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors with user-friendly messages."""
    # Enhanced logging for debugging
    logger.warning(f"Validation error on {request.method} {request.url.path}", extra={
        "request_url": str(request.url),
        "method": request.method,
        "content_type": request.headers.get("content-type"),
        "errors": exc.errors()
    })

    # Log detailed error information for debugging
    logger.error(f"Detailed validation errors: {exc.errors()}")
    for error in exc.errors():
        logger.error(f"Field: {error.get('loc')}, Type: {error.get('type')}, Message: {error.get('msg')}")

    # Convert technical validation errors to user-friendly messages
    user_friendly_errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get('loc', []))
        msg = error.get('msg', '')
        user_msg = _convert_validation_error_to_user_message(field, msg, error.get('type'))
        user_friendly_errors.append({
            "field": field,
            "message": user_msg,
            "type": error.get('type'),
            "technical_message": msg  # Include technical message for debugging
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Please check your input and try again.",
            "validation_errors": user_friendly_errors,
            "timestamp": datetime.now().isoformat(),
            "type": "ValidationError"
        }
    )

async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP errors with consistent format."""
    logger.info(f"HTTP error {exc.status_code}: {exc.detail}", extra={
        "request_url": str(request.url),
        "status_code": exc.status_code
    })

    # Map HTTP status codes to user-friendly messages
    status_messages = {
        400: "Bad request. Please check your input and try again.",
        401: "Authentication required. Please log in.",
        403: "Access denied. You don't have permission for this action.",
        404: "The requested resource was not found.",
        413: "File too large. Please upload a smaller file.",
        415: "Unsupported file type. Please use Excel files (.xlsx or .xls).",
        429: "Too many requests. Please wait a moment and try again.",
        500: "Internal server error. Please try again later.",
        503: "Service temporarily unavailable. Please try again later."
    }

    user_message = status_messages.get(exc.status_code, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": user_message,
            "timestamp": datetime.now().isoformat(),
            "type": "HTTPException"
        }
    )

async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with user-friendly messages."""
    logger.error(f"Unexpected error: {str(exc)}", extra={
        "request_url": str(request.url),
        "exception_type": exc.__class__.__name__
    }, exc_info=True)

    # Don't expose internal details to users
    user_message = "We encountered an unexpected error while processing your request. Our team has been notified."

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": user_message,
            "suggestions": [
                "Please try again in a few moments",
                "If the problem persists, contact our support team",
                "Check if your file meets our requirements"
            ],
            "timestamp": datetime.now().isoformat(),
            "type": "InternalServerError"
        }
    )

def _convert_validation_error_to_user_message(field: str, msg: str, error_type: str) -> str:
    """Convert technical validation errors to user-friendly messages."""
    if "required" in msg.lower():
        return f"The field '{field}' is required. Please provide a value."
    elif "type" in msg.lower():
        if "float" in msg.lower() or "number" in msg.lower():
            return f"The field '{field}' must be a valid number."
        elif "integer" in msg.lower():
            return f"The field '{field}' must be a whole number."
        elif "string" in msg.lower():
            return f"The field '{field}' must be text."
    elif "greater than" in msg.lower():
        return f"The value for '{field}' must be greater than the minimum allowed."
    elif "less than" in msg.lower():
        return f"The value for '{field}' must be less than the maximum allowed."
    elif "max_length" in msg.lower():
        return f"The field '{field}' is too long. Please use fewer characters."
    else:
        return f"The field '{field}' has an invalid value. Please check and try again."

# Utility functions for error handling
def create_business_error(message: str, suggestions: list = None) -> AnalysisError:
    """Create a business-friendly error with suggestions."""
    return AnalysisError(
        message=message,
        suggestions=suggestions or ["Please try again or contact support if the issue persists"]
    )

def create_file_error(message: str, file_name: str = None) -> FileProcessingError:
    """Create a file processing error with helpful guidance."""
    return FileProcessingError(message=message, file_name=file_name)

def create_validation_error(message: str, field: str = None, value: Any = None) -> ValidationError:
    """Create a validation error with field context."""
    return ValidationError(message=message, field=field, value=value)