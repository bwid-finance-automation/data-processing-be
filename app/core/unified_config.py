# app/core/unified_config.py
"""Unified configuration management system with validation."""

import os
from typing import List, Dict, Any, Optional
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

class RevenueAnalysisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_REVENUE_ANALYSIS__")
    """Revenue analysis configuration section."""

    # Revenue account analysis
    revenue_change_threshold_vnd: float = Field(
        default=1_000_000,
        description="VND threshold for significant revenue changes",
        ge=0,
        le=1e12
    )
    revenue_entity_threshold_vnd: float = Field(
        default=100_000,
        description="VND threshold for tracking entity revenue changes",
        ge=0,
        le=1e12
    )
    revenue_account_prefixes: List[str] = Field(
        default=["511"],
        description="Account code prefixes for revenue accounts"
    )

    # COGS analysis
    cogs_change_threshold_vnd: float = Field(
        default=500_000,
        description="VND threshold for significant COGS changes",
        ge=0,
        le=1e12
    )
    cogs_entity_threshold_vnd: float = Field(
        default=50_000,
        description="VND threshold for tracking entity COGS changes",
        ge=0,
        le=1e12
    )
    cogs_account_prefixes: List[str] = Field(
        default=["632"],
        description="Account code prefixes for COGS accounts"
    )

    # SG&A analysis
    sga_change_threshold_vnd: float = Field(
        default=500_000,
        description="VND threshold for significant SG&A changes",
        ge=0,
        le=1e12
    )
    sga_entity_threshold_vnd: float = Field(
        default=50_000,
        description="VND threshold for tracking entity SG&A changes",
        ge=0,
        le=1e12
    )
    sga_641_account_prefixes: List[str] = Field(
        default=["641"],
        description="Account code prefixes for SG&A 641 accounts"
    )
    sga_642_account_prefixes: List[str] = Field(
        default=["642"],
        description="Account code prefixes for SG&A 642 accounts"
    )

    # Risk assessment thresholds
    gross_margin_change_threshold_pct: float = Field(
        default=1.0,
        description="Percentage threshold for gross margin change risk",
        ge=0,
        le=100
    )
    high_gross_margin_risk_threshold_pct: float = Field(
        default=-2.0,
        description="Percentage threshold for high gross margin risk",
        ge=-100,
        le=100
    )
    sga_ratio_change_threshold_pct: float = Field(
        default=2.0,
        description="Percentage threshold for SG&A ratio change risk",
        ge=0,
        le=100
    )
    high_sga_ratio_threshold_pct: float = Field(
        default=3.0,
        description="Percentage threshold for high SG&A ratio risk",
        ge=0,
        le=100
    )
    revenue_pct_change_risk_threshold: float = Field(
        default=5.0,
        description="Percentage threshold for revenue change risk",
        ge=0,
        le=100
    )
    high_revenue_pct_change_threshold: float = Field(
        default=20.0,
        description="Percentage threshold for high revenue change risk",
        ge=0,
        le=100
    )

    # Analysis parameters
    months_to_analyze: int = Field(
        default=8,
        description="Number of months to analyze",
        ge=1,
        le=120
    )
    top_entity_impacts: int = Field(
        default=5,
        description="Number of top entity impacts to show",
        ge=1,
        le=50
    )
    lookback_periods: int = Field(
        default=10,
        description="Number of periods to look back for account detection",
        ge=1,
        le=60
    )

class ExcelProcessingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_EXCEL_PROCESSING__")
    """Excel file processing configuration section."""

    max_sheet_name_length: int = Field(
        default=31,
        description="Maximum length for Excel sheet names",
        ge=1,
        le=255
    )
    header_scan_rows: int = Field(
        default=40,
        description="Number of rows to scan for headers",
        ge=1,
        le=1000
    )
    data_row_offset: int = Field(
        default=2,
        description="Row offset for data after headers",
        ge=0,
        le=100
    )
    account_code_min_digits: int = Field(
        default=4,
        description="Minimum digits required for account codes",
        ge=1,
        le=20
    )

    # Progress milestones for UI feedback
    progress_milestones: Dict[str, int] = Field(
        default={
            "start": 10,
            "load": 15,
            "config": 20,
            "ai_thresholds": 25,
            "analysis_start": 30,
            "analysis_complete": 85,
            "storage": 90,
            "finalize": 95,
            "complete": 100
        },
        description="Progress milestone percentages"
    )

class CoreAnalysisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_CORE_ANALYSIS__")
    """Core financial analysis configuration section."""

    materiality_vnd: float = Field(
        default=1_000_000_000,
        description="Absolute VND change threshold for materiality",
        ge=0,
        le=1e15
    )
    recurring_pct_threshold: float = Field(
        default=0.05,
        description="Percentage threshold for recurring P/L accounts",
        ge=0,
        le=1
    )
    revenue_opex_pct_threshold: float = Field(
        default=0.10,
        description="Percentage threshold for revenue/opex accounts",
        ge=0,
        le=1
    )
    bs_pct_threshold: float = Field(
        default=0.05,
        description="Percentage threshold for balance sheet changes",
        ge=0,
        le=1
    )
    recurring_code_prefixes: List[str] = Field(
        default=["6321", "635", "515"],
        description="Account code prefixes for recurring revenue/costs"
    )
    min_trend_periods: int = Field(
        default=3,
        description="Minimum periods required for trend analysis",
        ge=1,
        le=120
    )
    gm_drop_threshold_pct: float = Field(
        default=0.01,
        description="Gross margin drop threshold (absolute percentage points)",
        ge=0,
        le=1
    )
    dep_pct_only_prefixes: List[str] = Field(
        default=["217", "632"],
        description="Account prefixes for depreciation percentage-only analysis"
    )
    customer_column_hints: List[str] = Field(
        default=[
            "customer", "khách", "khach", "client", "buyer", "entity",
            "company", "subsidiary", "parent company", "bwid", "vc1", "vc2", "vc3", "logistics"
        ],
        description="Keywords to identify customer columns"
    )

class AIAnalysisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_AI_ANALYSIS__")
    """AI analysis configuration section."""

    use_llm_analysis: bool = Field(
        default=False,
        description="Whether to enable AI-powered analysis"
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model to use for AI analysis"
    )
    enable_ai_analysis: bool = Field(
        default=True,
        description="Global AI analysis feature flag"
    )
    max_ai_retries: int = Field(
        default=3,
        description="Maximum retries for AI analysis calls",
        ge=1,
        le=10
    )
    ai_timeout_seconds: int = Field(
        default=300,
        description="Timeout for AI analysis in seconds",
        ge=30,
        le=3600
    )

class FileProcessingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_FILE_PROCESSING__")
    """File upload and processing configuration section."""

    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum file size in bytes",
        ge=1024,  # 1KB minimum
        le=1024 * 1024 * 1024  # 1GB maximum
    )
    allowed_file_extensions: List[str] = Field(
        default=[".xlsx", ".xls"],
        description="Allowed file extensions for uploads"
    )
    max_files_per_request: int = Field(
        default=50,
        description="Maximum number of files per upload request",
        ge=1,
        le=100
    )
    required_sheets: List[str] = Field(
        default=["BS Breakdown", "PL Breakdown"],
        description="Required sheet names in Excel files"
    )

class SecurityConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_SECURITY__")
    """Security and session management configuration section."""

    session_timeout_minutes: int = Field(
        default=60,
        description="Session timeout in minutes",
        ge=1,
        le=1440  # 24 hours
    )
    max_concurrent_sessions: int = Field(
        default=10,
        description="Maximum concurrent analysis sessions",
        ge=1,
        le=1000
    )
    rate_limit_requests_per_minute: int = Field(
        default=100,
        description="Rate limit: requests per minute per IP",
        ge=1,
        le=10000
    )
    rate_limit_window_minutes: int = Field(
        default=1,
        description="Rate limiting window in minutes",
        ge=1,
        le=60
    )

class ApplicationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_APP__")
    """Main application configuration section."""

    # Application metadata
    app_name: str = Field(
        default="Variance Analysis Tool API",
        description="Application name"
    )
    app_version: str = Field(
        default="2.0.0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # CORS configuration
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_methods: List[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="logs/variance_analysis.log",
        description="Log file path"
    )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()

class DataProcessingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VARIANCE_DATA_PROCESSING__")
    """Data processing constants and defaults."""

    year_range: List[str] = Field(
        default=["2024", "2025", "2026", "2027", "2028", "2029", "2030"],
        description="Valid year suffixes for period detection"
    )
    trend_window_max: int = Field(
        default=5,
        description="Maximum trend window periods",
        ge=1,
        le=50
    )
    zero_division_replacement: float = Field(
        default=0.0,
        description="Value to use when dividing by zero"
    )
    numeric_fill_value: float = Field(
        default=0.0,
        description="Fill value for numeric columns"
    )
    percentage_multiplier: float = Field(
        default=100.0,
        description="Multiplier to convert decimals to percentages",
        gt=0
    )

class UnifiedConfig(BaseSettings):
    """
    Unified configuration system that consolidates all application settings.

    This replaces the fragmented configuration scattered across multiple files
    and provides validation, environment variable support, and clear documentation.
    """

    # Configuration sections
    app: ApplicationConfig = Field(default_factory=ApplicationConfig)
    revenue_analysis: RevenueAnalysisConfig = Field(default_factory=RevenueAnalysisConfig)
    excel_processing: ExcelProcessingConfig = Field(default_factory=ExcelProcessingConfig)
    core_analysis: CoreAnalysisConfig = Field(default_factory=CoreAnalysisConfig)
    ai_analysis: AIAnalysisConfig = Field(default_factory=AIAnalysisConfig)
    file_processing: FileProcessingConfig = Field(default_factory=FileProcessingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)

    model_config = SettingsConfigDict(
        env_prefix="VARIANCE_",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    @model_validator(mode='before')
    @classmethod
    def validate_config_consistency(cls, values):
        """Validate configuration consistency across sections."""
        if isinstance(values, dict):
            app = values.get('app')
            ai_analysis = values.get('ai_analysis')

            # Ensure AI settings are consistent
            if app and ai_analysis:
                if hasattr(app, 'debug') and app.debug:
                    # In debug mode, enable more verbose AI logging
                    pass

        return values

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert unified config to legacy dictionary format for backward compatibility.

        This method ensures existing code that expects the old DEFAULT_CONFIG
        dictionary format continues to work without modification.
        """
        legacy_config = {}

        # Core analysis settings
        legacy_config.update({
            "materiality_vnd": self.core_analysis.materiality_vnd,
            "recurring_pct_threshold": self.core_analysis.recurring_pct_threshold,
            "revenue_opex_pct_threshold": self.core_analysis.revenue_opex_pct_threshold,
            "bs_pct_threshold": self.core_analysis.bs_pct_threshold,
            "recurring_code_prefixes": self.core_analysis.recurring_code_prefixes,
            "min_trend_periods": self.core_analysis.min_trend_periods,
            "gm_drop_threshold_pct": self.core_analysis.gm_drop_threshold_pct,
            "dep_pct_only_prefixes": self.core_analysis.dep_pct_only_prefixes,
            "customer_column_hints": self.core_analysis.customer_column_hints,
        })

        # Revenue analysis settings
        legacy_config.update({
            "revenue_change_threshold_vnd": self.revenue_analysis.revenue_change_threshold_vnd,
            "revenue_entity_threshold_vnd": self.revenue_analysis.revenue_entity_threshold_vnd,
            "revenue_account_prefixes": self.revenue_analysis.revenue_account_prefixes,
            "cogs_change_threshold_vnd": self.revenue_analysis.cogs_change_threshold_vnd,
            "cogs_entity_threshold_vnd": self.revenue_analysis.cogs_entity_threshold_vnd,
            "cogs_account_prefixes": self.revenue_analysis.cogs_account_prefixes,
            "sga_change_threshold_vnd": self.revenue_analysis.sga_change_threshold_vnd,
            "sga_entity_threshold_vnd": self.revenue_analysis.sga_entity_threshold_vnd,
            "sga_641_account_prefixes": self.revenue_analysis.sga_641_account_prefixes,
            "sga_642_account_prefixes": self.revenue_analysis.sga_642_account_prefixes,
            "gross_margin_change_threshold_pct": self.revenue_analysis.gross_margin_change_threshold_pct,
            "high_gross_margin_risk_threshold_pct": self.revenue_analysis.high_gross_margin_risk_threshold_pct,
            "sga_ratio_change_threshold_pct": self.revenue_analysis.sga_ratio_change_threshold_pct,
            "high_sga_ratio_threshold_pct": self.revenue_analysis.high_sga_ratio_threshold_pct,
            "revenue_pct_change_risk_threshold": self.revenue_analysis.revenue_pct_change_risk_threshold,
            "high_revenue_pct_change_threshold": self.revenue_analysis.high_revenue_pct_change_threshold,
            "months_to_analyze": self.revenue_analysis.months_to_analyze,
            "top_entity_impacts": self.revenue_analysis.top_entity_impacts,
            "lookback_periods": self.revenue_analysis.lookback_periods,
        })

        # Excel processing settings
        legacy_config.update({
            "max_sheet_name_length": self.excel_processing.max_sheet_name_length,
            "header_scan_rows": self.excel_processing.header_scan_rows,
            "data_row_offset": self.excel_processing.data_row_offset,
            "account_code_min_digits": self.excel_processing.account_code_min_digits,
            "progress_milestones": self.excel_processing.progress_milestones,
        })

        # AI analysis settings
        legacy_config.update({
            "use_llm_analysis": self.ai_analysis.use_llm_analysis,
            "llm_model": self.ai_analysis.llm_model,
        })

        # Data processing settings
        legacy_config.update({
            "year_range": self.data_processing.year_range,
            "trend_window_max": self.data_processing.trend_window_max,
            "zero_division_replacement": self.data_processing.zero_division_replacement,
            "numeric_fill_value": self.data_processing.numeric_fill_value,
            "percentage_multiplier": self.data_processing.percentage_multiplier,
        })

        # File processing constants (for backward compatibility with processing_service.py)
        legacy_config.update({
            "bytes_per_kb": 1024,
            "progress_file_range": 50,
            "progress_base_start": 30,
            "file_progress_offset": {
                "extract": 2,
                "analysis": 5,
                "complete": 5
            },
        })

        # Legacy accounting thresholds (for backward compatibility with analysis algorithms)
        legacy_config.update({
            "gross_margin_pct_delta": 0.01,    # 1% point change m/m
            "depr_pct_delta": 0.10,            # 10% change m/m for 217*, 632*, 214
            "cogs_ratio_delta": 0.02,          # 2% points drift vs hist
            "sga_pct_of_rev_delta": 0.10,      # +10% vs hist % of revenue
            "fin_swing_pct": 0.50,             # >50% swings
            "bs_pl_dep_diff_pct": 0.05,        # 5% mismatch between 214/217 Δ and 632 dep expense
        })

        return legacy_config

# Global configuration instance
@lru_cache()
def get_unified_config() -> UnifiedConfig:
    """Get the unified configuration instance with caching."""
    try:
        config = UnifiedConfig()
        logger.info("Unified configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load unified configuration: {e}", exc_info=True)
        raise

# Backward compatibility functions
@lru_cache()
def get_settings():
    """Backward compatible settings getter."""
    config = get_unified_config()
    return config.app

@lru_cache()
def get_analysis_config() -> dict:
    """Backward compatible analysis config getter."""
    config = get_unified_config()
    return config.to_legacy_dict()

# Environment variable loading with validation
def load_config_from_env():
    """Load and validate configuration from environment variables."""
    logger.info("Loading configuration from environment variables...")

    # List of environment variables to check
    env_vars = [
        "VARIANCE_APP__DEBUG",
        "VARIANCE_APP__LOG_LEVEL",
        "VARIANCE_AI_ANALYSIS__LLM_MODEL",
        "VARIANCE_AI_ANALYSIS__USE_LLM_ANALYSIS",
        "VARIANCE_FILE_PROCESSING__MAX_FILE_SIZE",
        "VARIANCE_SECURITY__SESSION_TIMEOUT_MINUTES",
        "VARIANCE_CORE_ANALYSIS__MATERIALITY_VND",
    ]

    found_vars = []
    for var in env_vars:
        if os.getenv(var):
            found_vars.append(var)

    if found_vars:
        logger.info(f"Found {len(found_vars)} environment configuration variables")
    else:
        logger.info("No environment configuration variables found, using defaults")

    return get_unified_config()