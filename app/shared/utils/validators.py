"""
Input validation utilities for Data Processing Backend.

This module provides common validation functions for:
- File validation (extension, size, type)
- Bank account number validation
- Currency amount parsing
- Input sanitization
"""

import re
from typing import Optional
from pathlib import Path


# =============================================================================
# File Validation
# =============================================================================

# Allowed file extensions for different file types
ALLOWED_EXTENSIONS = {
    "excel": {".xlsx", ".xls", ".xlsb"},
    "pdf": {".pdf"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp"},
    "all": {".xlsx", ".xls", ".xlsb", ".pdf", ".png", ".jpg", ".jpeg"},
}

# Default max file size: 100 MB
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024


def is_valid_file_extension(
    filename: str,
    allowed_types: Optional[set] = None
) -> bool:
    """
    Check if a filename has a valid extension.

    Args:
        filename: The filename to check
        allowed_types: Set of allowed extensions (defaults to all supported types)

    Returns:
        True if the extension is valid, False otherwise
    """
    if not filename:
        return False

    if allowed_types is None:
        allowed_types = ALLOWED_EXTENSIONS["all"]

    extension = Path(filename).suffix.lower()
    return extension in allowed_types


def is_valid_file_size(
    size_bytes: int,
    max_size: int = DEFAULT_MAX_FILE_SIZE
) -> bool:
    """
    Check if a file size is within the allowed limit.

    Args:
        size_bytes: File size in bytes
        max_size: Maximum allowed size in bytes (default 100MB)

    Returns:
        True if size is valid, False otherwise
    """
    return 0 < size_bytes <= max_size


def detect_file_type(content: bytes) -> Optional[str]:
    """
    Detect file type based on magic bytes (file signature).

    Args:
        content: File content as bytes

    Returns:
        Detected file type string or None if unknown
    """
    if len(content) < 8:
        return None

    # Common file signatures
    signatures = {
        b'\x50\x4B\x03\x04': 'xlsx',  # ZIP-based (xlsx, docx, etc.)
        b'\xD0\xCF\x11\xE0': 'xls',   # OLE2 (xls, doc, etc.)
        b'%PDF': 'pdf',
        b'\x89PNG': 'png',
        b'\xFF\xD8\xFF': 'jpeg',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
    }

    for signature, file_type in signatures.items():
        if content.startswith(signature):
            return file_type

    # Additional checks
    if content[:2] == b'PK':
        return 'xlsx'

    return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for use
    """
    if not filename:
        return "unnamed_file"

    # Remove path components
    filename = Path(filename).name

    # Remove dangerous characters
    dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*', "'", ';', '&']
    for char in dangerous_chars:
        filename = filename.replace(char, '')

    # Remove control characters
    filename = ''.join(c for c in filename if ord(c) >= 32)

    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:max_length - len(ext)] + ext

    # Ensure we have a valid filename
    if not filename or filename.isspace():
        return "unnamed_file"

    return filename.strip()


# =============================================================================
# Bank Account Validation
# =============================================================================

def is_valid_account_number(account_number: str) -> bool:
    """
    Validate a bank account number.

    Accepts:
    - Numeric strings (10-20 digits)
    - Numbers with dashes (e.g., 1234-5678-9012)

    Args:
        account_number: Account number string to validate

    Returns:
        True if valid, False otherwise
    """
    if not account_number:
        return False

    # Remove dashes and spaces
    cleaned = account_number.replace('-', '').replace(' ', '')

    # Must be all digits
    if not cleaned.isdigit():
        return False

    # Must be between 8 and 20 digits (most bank accounts)
    if not (8 <= len(cleaned) <= 20):
        return False

    return True


def format_account_number(account_number: str, mask: bool = False) -> str:
    """
    Format an account number for display.

    Args:
        account_number: Raw account number
        mask: If True, mask middle digits for security

    Returns:
        Formatted account number string
    """
    cleaned = account_number.replace('-', '').replace(' ', '')

    if mask and len(cleaned) > 8:
        # Show first 4 and last 4 digits
        return f"{cleaned[:4]}{'*' * (len(cleaned) - 8)}{cleaned[-4:]}"

    # Format with dashes every 4 digits
    return '-'.join([cleaned[i:i+4] for i in range(0, len(cleaned), 4)])


# =============================================================================
# Amount/Currency Parsing
# =============================================================================

def parse_amount(value: str) -> Optional[float]:
    """
    Parse a currency amount string to float.

    Handles:
    - Plain numbers: "1000000"
    - Thousands separators: "1,000,000"
    - Decimals: "1000.50"
    - Negative values: "-500000" or "(500000)"
    - Currency symbols: "VND 1,000,000" or "1,000,000 VND"

    Args:
        value: String value to parse

    Returns:
        Parsed float value or None if invalid
    """
    if not value or not isinstance(value, str):
        return None

    original = value
    value = value.strip()

    if not value:
        return None

    # Check for accounting notation (negative in parentheses)
    is_negative = False
    if value.startswith('(') and value.endswith(')'):
        is_negative = True
        value = value[1:-1]
    elif value.startswith('-'):
        is_negative = True
        value = value[1:]

    # Remove currency symbols and common prefixes/suffixes
    currency_patterns = [
        r'VND\s*', r'\s*VND',
        r'USD\s*', r'\s*USD',
        r'EUR\s*', r'\s*EUR',
        r'\$\s*', r'\s*\$',
        r'đ\s*', r'\s*đ',
    ]

    for pattern in currency_patterns:
        value = re.sub(pattern, '', value, flags=re.IGNORECASE)

    # Remove thousand separators (commas)
    value = value.replace(',', '')

    # Remove spaces
    value = value.replace(' ', '')

    try:
        result = float(value)
        return -result if is_negative else result
    except (ValueError, TypeError):
        return None


def format_amount(
    value: float,
    currency: str = "VND",
    decimal_places: int = 0
) -> str:
    """
    Format a numeric amount for display.

    Args:
        value: Numeric value
        currency: Currency code for display
        decimal_places: Number of decimal places (default 0 for VND)

    Returns:
        Formatted amount string
    """
    if value is None:
        return "N/A"

    # Format with thousand separators
    if decimal_places == 0:
        formatted = f"{int(value):,}"
    else:
        formatted = f"{value:,.{decimal_places}f}"

    return f"{formatted} {currency}"


# =============================================================================
# General Input Validation
# =============================================================================

def is_valid_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid format, False otherwise
    """
    if not email:
        return False

    # Simple regex for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """
    Validate a Vietnamese phone number format.

    Args:
        phone: Phone number to validate

    Returns:
        True if valid format, False otherwise
    """
    if not phone:
        return False

    # Remove common formatting
    cleaned = re.sub(r'[\s\-\.\(\)]', '', phone)

    # Vietnamese phone patterns
    patterns = [
        r'^(\+84|84|0)[3|5|7|8|9][0-9]{8}$',  # Mobile
        r'^(\+84|84|0)[2][0-9]{9}$',           # Landline
    ]

    return any(re.match(p, cleaned) for p in patterns)


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize a string input.

    Args:
        value: Input string
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not value:
        return ""

    # Remove control characters except newlines and tabs
    sanitized = ''.join(
        c for c in value
        if ord(c) >= 32 or c in '\n\t\r'
    )

    # Trim whitespace
    sanitized = sanitized.strip()

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized
