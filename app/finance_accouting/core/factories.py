# app/core/factories.py
"""
Factory patterns for creating analysis services and configurations.

This module implements the Factory design pattern to create appropriate
service instances based on analysis type, configuration, and runtime conditions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, Protocol
from enum import Enum

from ..models.analysis import AnalysisType, SessionStatus
from ..core.unified_config import get_unified_config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Service factory protocols

class IAnalysisServiceFactory(Protocol):
    """Protocol for analysis service factories."""

    def create_analysis_service(self, analysis_type: AnalysisType, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create an analysis service instance."""
        ...

    def get_supported_types(self) -> list[AnalysisType]:
        """Get list of supported analysis types."""
        ...

class IConfigurationFactory(Protocol):
    """Protocol for configuration factories."""

    def create_configuration(self, config_type: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a configuration instance."""
        ...

    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        ...

# Base factory class

class BaseFactory(ABC):
    """Base factory class with common functionality."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_unified_config()

    def log_creation(self, item_type: str, item_name: str, **kwargs):
        """Log factory creation with context."""
        self.logger.info(
            f"Factory created {item_type}: {item_name}",
            extra={
                "factory": self.__class__.__name__,
                "item_type": item_type,
                "item_name": item_name,
                **kwargs
            }
        )

# Analysis service factory

class AnalysisServiceFactory(BaseFactory):
    """
    Factory for creating analysis service instances.

    This factory creates appropriate analysis services based on the analysis type
    and configuration, with support for dependency injection and service composition.
    """

    def __init__(self, container=None):
        super().__init__()
        self.container = container
        self._service_registry: Dict[AnalysisType, Type] = {}
        self._service_instances: Dict[str, Any] = {}  # Cache for singleton services

        # Register default service types
        self._register_default_services()

    def _register_default_services(self):
        """Register default analysis service types."""
        try:
            # Import services dynamically to avoid circular imports
            from ..services.analysis_service import AnalysisService
            from ..services.file_processing_service import FileProcessingService
            from ..services.session_service import SessionService

            # Register core services
            self._service_registry[AnalysisType.COMPREHENSIVE] = AnalysisService
            self._service_registry[AnalysisType.REVENUE_VARIANCE] = self._create_revenue_analysis_service
            self._service_registry[AnalysisType.AI_POWERED] = self._create_ai_analysis_service

            self.logger.info(f"Registered {len(self._service_registry)} default analysis services")

        except ImportError as e:
            self.logger.error(f"Failed to register default services: {e}")
            raise

    def create_analysis_service(self, analysis_type: AnalysisType, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an analysis service instance.

        Args:
            analysis_type: Type of analysis service to create
            config: Optional configuration overrides

        Returns:
            Analysis service instance

        Raises:
            ValueError: If analysis type is not supported
        """
        try:
            if analysis_type not in self._service_registry:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            # Check for cached instance (for singleton services)
            cache_key = f"{analysis_type}_{hash(str(config))}"
            if cache_key in self._service_instances:
                self.log_creation("cached_analysis_service", analysis_type.value)
                return self._service_instances[cache_key]

            # Create new instance
            service_creator = self._service_registry[analysis_type]

            if callable(service_creator):
                # If it's a function/method, call it with config
                service = service_creator(config)
            else:
                # If it's a class, instantiate it
                if self.container:
                    # Use dependency injection if container is available
                    service = self.container.resolve(service_creator)
                else:
                    # Fallback to direct instantiation
                    service = service_creator()

            # Configure the service if config is provided
            if config and hasattr(service, 'update_configuration'):
                service.update_configuration(config)

            # Cache singleton services
            if self._is_singleton_service(analysis_type):
                self._service_instances[cache_key] = service

            self.log_creation(
                "analysis_service",
                analysis_type.value,
                has_config=config is not None,
                is_cached=False
            )

            return service

        except Exception as e:
            self.logger.error(f"Failed to create analysis service: {e}", exc_info=True)
            raise

    def _create_revenue_analysis_service(self, config: Optional[Dict[str, Any]] = None):
        """Create specialized revenue analysis service."""
        try:
            from ..services.analysis_service import AnalysisService

            # Create base service
            service = AnalysisService() if not self.container else self.container.resolve(AnalysisService)

            # Configure for revenue analysis
            revenue_config = self.config.revenue_analysis.model_dump()
            if config:
                revenue_config.update(config)

            service.set_analysis_mode("revenue_variance")
            return service

        except Exception as e:
            self.logger.error(f"Failed to create revenue analysis service: {e}")
            raise

    def _create_ai_analysis_service(self, config: Optional[Dict[str, Any]] = None):
        """Create AI-powered analysis service."""
        try:
            from ..services.analysis_service import AnalysisService

            # Check if AI analysis is enabled
            if not self.config.ai_analysis.enable_ai_analysis:
                raise ValueError("AI analysis is disabled in configuration")

            # Create base service
            service = AnalysisService() if not self.container else self.container.resolve(AnalysisService)

            # Configure for AI analysis
            ai_config = self.config.ai_analysis.model_dump()
            if config:
                ai_config.update(config)

            service.set_analysis_mode("ai_powered")
            service.configure_ai_settings(ai_config)
            return service

        except Exception as e:
            self.logger.error(f"Failed to create AI analysis service: {e}")
            raise

    def _is_singleton_service(self, analysis_type: AnalysisType) -> bool:
        """Check if a service type should be treated as singleton."""
        # AI services are typically singleton due to model loading overhead
        singleton_types = {AnalysisType.AI_POWERED}
        return analysis_type in singleton_types

    def register_service_type(self, analysis_type: AnalysisType, service_class: Type):
        """Register a custom service type."""
        self._service_registry[analysis_type] = service_class
        self.logger.info(f"Registered custom service type: {analysis_type} -> {service_class.__name__}")

    def get_supported_types(self) -> list[AnalysisType]:
        """Get list of supported analysis types."""
        return list(self._service_registry.keys())

    def clear_cache(self):
        """Clear cached service instances."""
        self._service_instances.clear()
        self.logger.info("Cleared service instance cache")

# Configuration factory

class ConfigurationFactory(BaseFactory):
    """
    Factory for creating and validating configurations.

    This factory creates appropriate configuration objects based on analysis type,
    environment, and user overrides with comprehensive validation.
    """

    def __init__(self):
        super().__init__()
        self._config_builders: Dict[str, callable] = {}
        self._config_validators: Dict[str, callable] = {}

        # Register default configuration builders
        self._register_default_builders()

    def _register_default_builders(self):
        """Register default configuration builders."""
        self._config_builders.update({
            "analysis": self._build_analysis_config,
            "revenue": self._build_revenue_config,
            "ai": self._build_ai_config,
            "file_processing": self._build_file_processing_config,
            "security": self._build_security_config,
            "development": self._build_development_config,
            "production": self._build_production_config
        })

        self._config_validators.update({
            "analysis": self._validate_analysis_config,
            "revenue": self._validate_revenue_config,
            "ai": self._validate_ai_config,
            "file_processing": self._validate_file_processing_config,
            "security": self._validate_security_config
        })

    def create_configuration(self, config_type: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a configuration instance.

        Args:
            config_type: Type of configuration to create
            overrides: Optional configuration overrides

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If configuration type is not supported
        """
        try:
            if config_type not in self._config_builders:
                raise ValueError(f"Unsupported configuration type: {config_type}")

            # Build base configuration
            builder = self._config_builders[config_type]
            config = builder()

            # Apply overrides
            if overrides:
                config = self._merge_configurations(config, overrides)

            # Validate configuration
            validation_result = self.validate_configuration(config, config_type)
            if not validation_result["is_valid"]:
                raise ValueError(f"Configuration validation failed: {validation_result['errors']}")

            self.log_creation(
                "configuration",
                config_type,
                has_overrides=overrides is not None,
                validation_passed=validation_result["is_valid"]
            )

            return config

        except Exception as e:
            self.logger.error(f"Failed to create configuration: {e}", exc_info=True)
            raise

    def validate_configuration(self, config: Dict[str, Any], config_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration and return validation results.

        Args:
            config: Configuration to validate
            config_type: Optional type-specific validation

        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "recommendations": []
            }

            # Perform general validation
            general_validation = self._validate_general_config(config)
            validation_results["errors"].extend(general_validation.get("errors", []))
            validation_results["warnings"].extend(general_validation.get("warnings", []))

            # Perform type-specific validation
            if config_type and config_type in self._config_validators:
                validator = self._config_validators[config_type]
                type_validation = validator(config)
                validation_results["errors"].extend(type_validation.get("errors", []))
                validation_results["warnings"].extend(type_validation.get("warnings", []))

            # Set overall validity
            validation_results["is_valid"] = len(validation_results["errors"]) == 0

            return validation_results

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": []
            }

    def _build_analysis_config(self) -> Dict[str, Any]:
        """Build analysis configuration."""
        return self.config.to_legacy_dict()

    def _build_revenue_config(self) -> Dict[str, Any]:
        """Build revenue analysis configuration."""
        config = self.config.revenue_analysis.model_dump()
        config.update(self.config.core_analysis.model_dump())
        return config

    def _build_ai_config(self) -> Dict[str, Any]:
        """Build AI analysis configuration."""
        if not self.config.ai_analysis.enable_ai_analysis:
            raise ValueError("AI analysis is disabled")
        return self.config.ai_analysis.model_dump()

    def _build_file_processing_config(self) -> Dict[str, Any]:
        """Build file processing configuration."""
        return self.config.file_processing.model_dump()

    def _build_security_config(self) -> Dict[str, Any]:
        """Build security configuration."""
        return self.config.security.model_dump()

    def _build_development_config(self) -> Dict[str, Any]:
        """Build development environment configuration."""
        config = self.config.to_legacy_dict()

        # Development-specific overrides
        config.update({
            "debug": True,
            "log_level": "DEBUG",
            "ai_analysis_enabled": True,
            "session_timeout_minutes": 240,  # 4 hours for development
            "rate_limit_requests_per_minute": 1000,  # Relaxed for development
            "max_file_size": 50 * 1024 * 1024,  # 50MB for development
        })

        return config

    def _build_production_config(self) -> Dict[str, Any]:
        """Build production environment configuration."""
        config = self.config.to_legacy_dict()

        # Production-specific overrides
        config.update({
            "debug": False,
            "log_level": "INFO",
            "session_timeout_minutes": 60,  # 1 hour for production
            "rate_limit_requests_per_minute": 100,  # Strict for production
            "max_file_size": 100 * 1024 * 1024,  # 100MB for production
        })

        return config

    def _validate_general_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general configuration validation."""
        errors = []
        warnings = []

        # Required fields validation
        required_fields = ["materiality_vnd", "recurring_pct_threshold"]
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"Required field missing: {field}")

        # Value range validation
        if "materiality_vnd" in config and config["materiality_vnd"] <= 0:
            errors.append("materiality_vnd must be positive")

        if "session_timeout_minutes" in config and config["session_timeout_minutes"] > 1440:
            warnings.append("session_timeout_minutes exceeds 24 hours")

        return {"errors": errors, "warnings": warnings}

    def _validate_analysis_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis-specific configuration."""
        errors = []
        warnings = []

        # Analysis-specific validation
        if config.get("min_trend_periods", 0) < 1:
            errors.append("min_trend_periods must be at least 1")

        if config.get("gm_drop_threshold_pct", 0) < 0:
            errors.append("gm_drop_threshold_pct cannot be negative")

        return {"errors": errors, "warnings": warnings}

    def _validate_revenue_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate revenue analysis configuration."""
        errors = []
        warnings = []

        # Revenue-specific validation
        if config.get("months_to_analyze", 0) < 1:
            errors.append("months_to_analyze must be at least 1")

        if config.get("revenue_change_threshold_vnd", 0) <= 0:
            errors.append("revenue_change_threshold_vnd must be positive")

        return {"errors": errors, "warnings": warnings}

    def _validate_ai_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AI analysis configuration."""
        errors = []
        warnings = []

        # AI-specific validation
        if not config.get("llm_model"):
            errors.append("llm_model is required for AI analysis")

        if config.get("ai_timeout_seconds", 0) < 30:
            warnings.append("ai_timeout_seconds is very low, may cause timeouts")

        return {"errors": errors, "warnings": warnings}

    def _validate_file_processing_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file processing configuration."""
        errors = []
        warnings = []

        # File processing validation
        max_size = config.get("max_file_size", 0)
        if max_size > 1024 * 1024 * 1024:  # 1GB
            warnings.append("max_file_size is very large, may impact performance")

        return {"errors": errors, "warnings": warnings}

    def _validate_security_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security configuration."""
        errors = []
        warnings = []

        # Security validation
        if config.get("rate_limit_requests_per_minute", 0) > 10000:
            warnings.append("rate_limit_requests_per_minute is very high")

        return {"errors": errors, "warnings": warnings}

    def _merge_configurations(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration overrides with base configuration."""
        merged = base.copy()

        for key, value in overrides.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configurations(merged[key], value)
            else:
                # Direct override
                merged[key] = value

        return merged

    def register_config_builder(self, config_type: str, builder: callable):
        """Register a custom configuration builder."""
        self._config_builders[config_type] = builder
        self.logger.info(f"Registered custom config builder: {config_type}")

    def register_config_validator(self, config_type: str, validator: callable):
        """Register a custom configuration validator."""
        self._config_validators[config_type] = validator
        self.logger.info(f"Registered custom config validator: {config_type}")

# Service locator pattern for global factory access

class ServiceLocator:
    """Service locator for accessing factories globally."""

    _analysis_factory: Optional[AnalysisServiceFactory] = None
    _configuration_factory: Optional[ConfigurationFactory] = None

    @classmethod
    def get_analysis_factory(cls, container=None) -> AnalysisServiceFactory:
        """Get the analysis service factory instance."""
        if cls._analysis_factory is None:
            cls._analysis_factory = AnalysisServiceFactory(container)
        return cls._analysis_factory

    @classmethod
    def get_configuration_factory(cls) -> ConfigurationFactory:
        """Get the configuration factory instance."""
        if cls._configuration_factory is None:
            cls._configuration_factory = ConfigurationFactory()
        return cls._configuration_factory

    @classmethod
    def reset(cls):
        """Reset all factory instances (useful for testing)."""
        cls._analysis_factory = None
        cls._configuration_factory = None

# Convenience functions

def create_analysis_service(analysis_type: AnalysisType, config: Optional[Dict[str, Any]] = None, container=None):
    """Convenience function to create analysis service."""
    factory = ServiceLocator.get_analysis_factory(container)
    return factory.create_analysis_service(analysis_type, config)

def create_configuration(config_type: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to create configuration."""
    factory = ServiceLocator.get_configuration_factory()
    return factory.create_configuration(config_type, overrides)