# app/core/config.py
"""
Application configuration management.

This module provides backward compatibility while transitioning to the new unified config system.
New code should use the unified config system in unified_config.py.
"""

from functools import lru_cache
from typing import Dict, Any

# Import the new unified configuration system
from .unified_config import (
    get_unified_config,
    get_settings as get_unified_settings,
    get_analysis_config as get_unified_analysis_config,
    load_config_from_env
)

# Backward compatibility exports
from .unified_config import UnifiedConfig

# For backward compatibility, we keep the old function names
@lru_cache()
def get_settings():
    """
    Get application settings (backward compatible).

    Returns the app section of the unified configuration for compatibility
    with existing code that expects the old Settings dataclass.
    """
    return get_unified_settings()

@lru_cache()
def get_analysis_config() -> Dict[str, Any]:
    """
    Get analysis configuration dictionary (backward compatible).

    Returns a dictionary compatible with the old DEFAULT_CONFIG format.
    """
    return get_unified_analysis_config()

# Legacy class for backward compatibility
class Settings:
    """
    Legacy Settings class for backward compatibility.

    This class is deprecated. New code should use the unified config system.
    """

    def __init__(self):
        config = get_unified_config()

        # Map unified config to legacy attributes
        self.app_name = config.app.app_name
        self.app_version = config.app.app_version
        self.debug = config.app.debug
        self.cors_origins = config.app.cors_origins
        self.cors_methods = config.app.cors_methods
        self.cors_headers = config.app.cors_headers
        self.max_file_size = config.file_processing.max_file_size
        self.max_files_per_request = config.file_processing.max_files_per_request
        self.allowed_file_extensions = config.file_processing.allowed_file_extensions
        self.default_months_to_analyze = config.revenue_analysis.months_to_analyze
        self.revenue_change_threshold_vnd = config.revenue_analysis.revenue_change_threshold_vnd
        self.revenue_entity_threshold_vnd = config.revenue_analysis.revenue_entity_threshold_vnd
        self.llm_model = config.ai_analysis.llm_model
        self.enable_ai_analysis = config.ai_analysis.enable_ai_analysis
        self.session_timeout_minutes = config.security.session_timeout_minutes
        self.max_concurrent_sessions = config.security.max_concurrent_sessions

# Initialize configuration on module import
try:
    _ = load_config_from_env()
except Exception as e:
    import logging
    logging.warning(f"Failed to load configuration: {e}")

# Export new unified config functions for new code
__all__ = [
    "get_settings",
    "get_analysis_config",
    "get_unified_config",
    "UnifiedConfig",
    "Settings"  # For backward compatibility
]