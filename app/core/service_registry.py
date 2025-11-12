# app/core/service_registry.py
"""
Service registry and dependency injection configuration.

This module configures the dependency injection container with all application
services, ensuring proper service lifetimes and dependency resolution.
"""

from typing import Dict, Any

from .container import Container, get_container
from .factories import AnalysisServiceFactory, ConfigurationFactory, ServiceLocator
from .unified_config import get_unified_config
from app.shared.utils.logging_config import get_logger
from app.application.finance.use_cases.interfaces import (
    IFileProcessingService, IAnalysisService, ISessionService,
    IHealthService, ILoggingService, IConfigurationService, IDebugService
)

logger = get_logger(__name__)

def register_core_services(container: Container) -> None:
    """Register core application services."""
    try:
        # Configuration services
        container.register_singleton(ConfigurationFactory)
        container.register_factory(
            IConfigurationService,
            lambda: _create_configuration_service(),
            lifetime="singleton"
        )

        # File processing services
        from app.application.finance.use_cases.file_processing_service import FileProcessingService
        container.register_singleton(IFileProcessingService, FileProcessingService)

        # Session management services
        from app.application.finance.use_cases.session_service import SessionService
        container.register_singleton(ISessionService, SessionService)

        # Analysis services
        from app.application.finance.use_cases.analysis_service import AnalysisService
        container.register_singleton(IAnalysisService, AnalysisService)

        # Factory services
        container.register_factory(
            AnalysisServiceFactory,
            lambda: AnalysisServiceFactory(container),
            lifetime="singleton"
        )

        logger.info("Registered core services successfully")

    except Exception as e:
        logger.error(f"Failed to register core services: {e}", exc_info=True)
        raise

def register_utility_services(container: Container) -> None:
    """Register utility and support services."""
    try:
        # Health monitoring services
        container.register_factory(
            IHealthService,
            lambda: _create_health_service(),
            lifetime="singleton"
        )

        # Logging services
        container.register_factory(
            ILoggingService,
            lambda: _create_logging_service(),
            lifetime="singleton"
        )

        # Debug services (only in debug mode)
        config = get_unified_config()
        if config.app.debug:
            container.register_factory(
                IDebugService,
                lambda: _create_debug_service(),
                lifetime="singleton"
            )

        logger.info("Registered utility services successfully")

    except Exception as e:
        logger.error(f"Failed to register utility services: {e}", exc_info=True)
        raise

def register_analysis_services(container: Container) -> None:
    """Register analysis-specific services."""
    try:
        config = get_unified_config()

        # Revenue analysis services
        from app.application.finance.use_cases.revenue_analysis_service import RevenueAnalysisService
        from app.application.finance.use_cases.interfaces import IRevenueAnalysisService
        container.register_singleton(IRevenueAnalysisService, RevenueAnalysisService)

        # AI analysis services (if enabled)
        if config.ai_analysis.enable_ai_analysis:
            from app.application.finance.use_cases.ai_analysis_service import AIAnalysisService
            from app.application.finance.use_cases.interfaces import IAIAnalysisService
            container.register_singleton(IAIAnalysisService, AIAnalysisService)

        logger.info("Registered analysis services successfully")

    except ImportError as e:
        logger.warning(f"Some analysis services not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register analysis services: {e}", exc_info=True)
        raise

def configure_service_container() -> Container:
    """
    Configure and return the main application service container.

    Returns:
        Configured Container instance with all services registered
    """
    try:
        logger.info("Configuring service container...")

        # Create container
        container = Container()

        # Register service categories
        register_core_services(container)
        register_utility_services(container)
        register_analysis_services(container)

        # Configure service locator
        ServiceLocator._analysis_factory = container.resolve(AnalysisServiceFactory)
        ServiceLocator._configuration_factory = container.resolve(ConfigurationFactory)

        # Log container information
        services = container.list_services()
        logger.info(f"Service container configured with {len(services)} services")

        import logging
        if logger.isEnabledFor(logging.DEBUG):
            for service_name, service_info in services.items():
                logger.debug(f"Registered service: {service_name} ({service_info['lifetime']})")

        return container

    except Exception as e:
        logger.error(f"Failed to configure service container: {e}", exc_info=True)
        raise

def _create_configuration_service():
    """Create configuration service instance."""
    from app.application.finance.use_cases.configuration_service import ConfigurationService
    return ConfigurationService()

def _create_health_service():
    """Create health service instance."""
    from app.application.finance.use_cases.health_service import HealthService
    return HealthService()

def _create_logging_service():
    """Create logging service instance."""
    from app.application.finance.use_cases.logging_service import LoggingService
    return LoggingService()

def _create_debug_service():
    """Create debug service instance."""
    from app.application.finance.use_cases.debug_service import DebugService
    return DebugService()

# Service validation

def validate_service_configuration(container: Container) -> Dict[str, Any]:
    """
    Validate that all required services are properly configured.

    Args:
        container: Container to validate

    Returns:
        Validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "services_validated": 0
    }

    try:
        # Required services for basic operation
        required_services = [
            IFileProcessingService,
            ISessionService,
            IAnalysisService,
            ConfigurationFactory
        ]

        # Validate required services
        for service_type in required_services:
            try:
                service = container.resolve(service_type)
                validation_results["services_validated"] += 1

                # Check if service has required methods
                if hasattr(service, 'get_service_name'):
                    logger.debug(f"Validated service: {service.get_service_name()}")

            except Exception as e:
                validation_results["errors"].append(f"Failed to resolve {service_type.__name__}: {e}")
                validation_results["is_valid"] = False

        # Optional services (warnings if missing)
        optional_services = [IHealthService, ILoggingService]
        for service_type in optional_services:
            try:
                container.resolve(service_type)
                validation_results["services_validated"] += 1
            except Exception as e:
                validation_results["warnings"].append(f"Optional service not available: {service_type.__name__}")

        # Validate service dependencies
        dependency_validation = _validate_service_dependencies(container)
        validation_results["errors"].extend(dependency_validation.get("errors", []))
        validation_results["warnings"].extend(dependency_validation.get("warnings", []))

        if validation_results["errors"]:
            validation_results["is_valid"] = False

        logger.info(
            f"Service validation completed: {validation_results['services_validated']} services validated, "
            f"{len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
        )

        return validation_results

    except Exception as e:
        logger.error(f"Service validation failed: {e}", exc_info=True)
        return {
            "is_valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": [],
            "services_validated": 0
        }

def _validate_service_dependencies(container: Container) -> Dict[str, Any]:
    """Validate service dependencies are properly resolved."""
    errors = []
    warnings = []

    try:
        # Check circular dependencies
        services = container.list_services()
        for service_name, service_info in services.items():
            dependencies = service_info.get("dependencies", [])
            if len(dependencies) > 5:  # Arbitrary threshold
                warnings.append(f"Service {service_name} has many dependencies: {len(dependencies)}")

        # Check for potential issues
        config = get_unified_config()

        # AI services validation
        if config.ai_analysis.enable_ai_analysis:
            try:
                container.resolve(IAIAnalysisService)
            except Exception:
                warnings.append("AI analysis enabled but IAIAnalysisService not available")

    except Exception as e:
        errors.append(f"Dependency validation error: {str(e)}")

    return {"errors": errors, "warnings": warnings}

# Global container management

_global_container: Container = None

def get_application_container() -> Container:
    """Get the global application container."""
    global _global_container
    if _global_container is None:
        _global_container = configure_service_container()
    return _global_container

def reset_application_container():
    """Reset the global application container (useful for testing)."""
    global _global_container
    _global_container = None
    ServiceLocator.reset()

# FastAPI dependency integration

def get_file_processing_service() -> IFileProcessingService:
    """FastAPI dependency for file processing service."""
    container = get_application_container()
    return container.resolve(IFileProcessingService)

def get_session_service() -> ISessionService:
    """FastAPI dependency for session service."""
    container = get_application_container()
    return container.resolve(ISessionService)

def get_analysis_service() -> IAnalysisService:
    """FastAPI dependency for analysis service."""
    container = get_application_container()
    return container.resolve(IAnalysisService)

def get_health_service() -> IHealthService:
    """FastAPI dependency for health service."""
    container = get_application_container()
    return container.resolve(IHealthService)

def get_configuration_service() -> IConfigurationService:
    """FastAPI dependency for configuration service."""
    container = get_application_container()
    return container.resolve(IConfigurationService)