"""
Datetime utilities for consistent timezone handling.

IMPORTANT: Always use these functions instead of datetime.utcnow() to ensure
timezone-aware datetimes are used consistently throughout the application.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_now() -> datetime:
    """
    Get the current UTC time as a timezone-aware datetime.

    Use this instead of datetime.utcnow() which returns naive datetime.

    Returns:
        Timezone-aware datetime in UTC

    Example:
        >>> from app.shared.utils.datetime_utils import utc_now
        >>> now = utc_now()
        >>> now.tzinfo  # datetime.timezone.utc
    """
    return datetime.now(timezone.utc)


def utc_now_str(fmt: str = "%Y%m%d%H%M%S") -> str:
    """
    Get the current UTC time as a formatted string.

    Args:
        fmt: strftime format string (default: YYYYMMDDHHmmss)

    Returns:
        Formatted datetime string

    Example:
        >>> utc_now_str()  # '20260127103045'
        >>> utc_now_str("%Y-%m-%d")  # '2026-01-27'
    """
    return utc_now().strftime(fmt)


def days_ago(days: int) -> datetime:
    """
    Get a timezone-aware datetime for N days ago.

    Args:
        days: Number of days to subtract from now

    Returns:
        Timezone-aware datetime in UTC

    Example:
        >>> cutoff = days_ago(7)  # 7 days ago
    """
    return utc_now() - timedelta(days=days)


def hours_ago(hours: int) -> datetime:
    """
    Get a timezone-aware datetime for N hours ago.

    Args:
        hours: Number of hours to subtract from now

    Returns:
        Timezone-aware datetime in UTC
    """
    return utc_now() - timedelta(hours=hours)


def minutes_ago(minutes: int) -> datetime:
    """
    Get a timezone-aware datetime for N minutes ago.

    Args:
        minutes: Number of minutes to subtract from now

    Returns:
        Timezone-aware datetime in UTC
    """
    return utc_now() - timedelta(minutes=minutes)


def is_expired(dt: datetime, max_age_minutes: int) -> bool:
    """
    Check if a datetime is older than the specified age.

    Args:
        dt: Datetime to check (can be naive or aware)
        max_age_minutes: Maximum age in minutes

    Returns:
        True if the datetime is older than max_age_minutes
    """
    if dt.tzinfo is None:
        # Treat naive datetime as UTC
        dt = dt.replace(tzinfo=timezone.utc)

    expiry_time = dt + timedelta(minutes=max_age_minutes)
    return utc_now() > expiry_time


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure a datetime is timezone-aware UTC.

    Args:
        dt: Datetime to convert (can be naive or aware)

    Returns:
        Timezone-aware datetime in UTC, or None if input is None
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Assume naive datetime is already UTC
        return dt.replace(tzinfo=timezone.utc)

    # Convert to UTC if in different timezone
    return dt.astimezone(timezone.utc)
