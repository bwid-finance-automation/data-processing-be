# app/core/container.py
"""
Dependency injection container for managing service lifecycles and dependencies.

This module provides a lightweight dependency injection system that:
- Manages service instantiation and lifecycle
- Enables easy testing through dependency injection
- Provides clear service interfaces and abstractions
- Supports singleton and transient service lifetimes
"""

import inspect
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Dict, Any, Optional, Callable, get_type_hints
from functools import wraps

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class ServiceLifetime:
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

class ServiceDescriptor:
    """Describes how a service should be created and managed."""

    def __init__(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        instance: Optional[T] = None,
        lifetime: str = ServiceLifetime.SINGLETON
    ):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.dependencies: Dict[str, Type] = {}

        # Analyze constructor dependencies
        if self.implementation_type and not self.factory and not self.instance:
            self._analyze_dependencies()

    def _analyze_dependencies(self):
        """Analyze constructor dependencies from type hints."""
        try:
            init_signature = inspect.signature(self.implementation_type.__init__)
            type_hints = get_type_hints(self.implementation_type.__init__)

            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue

                if param_name in type_hints:
                    self.dependencies[param_name] = type_hints[param_name]

        except Exception as e:
            logger.warning(f"Could not analyze dependencies for {self.implementation_type}: {e}")

class Container:
    """
    Lightweight dependency injection container.

    Provides service registration, resolution, and lifecycle management.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._resolving: set = set()  # Prevent circular dependencies

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'Container':
        """Register a service as singleton (one instance per container lifetime)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered singleton service: {service_type.__name__}")
        return self

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'Container':
        """Register a service as transient (new instance every time)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.TRANSIENT
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered transient service: {service_type.__name__}")
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> 'Container':
        """Register a specific instance for a service type."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[service_type] = descriptor
        self._instances[service_type] = instance
        logger.debug(f"Registered instance for service: {service_type.__name__}")
        return self

    def register_factory(self, service_type: Type[T], factory: Callable[[], T], lifetime: str = ServiceLifetime.SINGLETON) -> 'Container':
        """Register a factory function for creating service instances."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered factory for service: {service_type.__name__}")
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance by type."""
        if service_type in self._resolving:
            raise ValueError(f"Circular dependency detected for service: {service_type.__name__}")

        self._resolving.add(service_type)
        try:
            return self._resolve_internal(service_type)
        finally:
            self._resolving.discard(service_type)

    def _resolve_internal(self, service_type: Type[T]) -> T:
        """Internal method to resolve service instances."""
        if service_type not in self._services:
            raise ValueError(f"Service not registered: {service_type.__name__}")

        descriptor = self._services[service_type]

        # Return existing instance for singletons
        if descriptor.lifetime == ServiceLifetime.SINGLETON and service_type in self._instances:
            return self._instances[service_type]

        # Create new instance
        instance = self._create_instance(descriptor)

        # Cache singleton instances
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            self._instances[service_type] = instance

        return instance

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new service instance."""
        if descriptor.instance is not None:
            return descriptor.instance

        if descriptor.factory is not None:
            logger.debug(f"Creating instance via factory: {descriptor.service_type.__name__}")
            return descriptor.factory()

        # Create instance with dependency injection
        implementation_type = descriptor.implementation_type
        dependencies = {}

        # Resolve constructor dependencies
        for param_name, param_type in descriptor.dependencies.items():
            try:
                dependencies[param_name] = self.resolve(param_type)
            except Exception as e:
                logger.warning(f"Could not resolve dependency {param_name} for {implementation_type.__name__}: {e}")

        logger.debug(f"Creating instance: {implementation_type.__name__} with dependencies: {list(dependencies.keys())}")
        return implementation_type(**dependencies)

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services

    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get information about a registered service."""
        if service_type not in self._services:
            return None

        descriptor = self._services[service_type]
        return {
            "service_type": descriptor.service_type.__name__,
            "implementation_type": descriptor.implementation_type.__name__ if descriptor.implementation_type else None,
            "lifetime": descriptor.lifetime,
            "has_factory": descriptor.factory is not None,
            "has_instance": descriptor.instance is not None,
            "dependencies": list(descriptor.dependencies.keys()),
            "is_instantiated": service_type in self._instances
        }

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services with their information."""
        return {
            service_type.__name__: self.get_service_info(service_type)
            for service_type in self._services.keys()
        }

def inject(*dependencies: Type) -> Callable:
    """
    Decorator for automatic dependency injection in functions.

    Usage:
        @inject(SomeService, AnotherService)
        def my_function(service1: SomeService, service2: AnotherService):
            # Use services
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get container from request state or global container
            container = getattr(args[0], 'container', None) if args else None
            if not container:
                container = get_container()

            # Resolve dependencies
            for i, dep_type in enumerate(dependencies):
                if len(args) <= i + 1:  # +1 to account for self/request parameter
                    resolved = container.resolve(dep_type)
                    args = args + (resolved,)

            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get container from request state or global container
            container = getattr(args[0], 'container', None) if args else None
            if not container:
                container = get_container()

            # Resolve dependencies
            for i, dep_type in enumerate(dependencies):
                if len(args) <= i + 1:  # +1 to account for self/request parameter
                    resolved = container.resolve(dep_type)
                    args = args + (resolved,)

            return func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator

# Global container instance
_global_container: Optional[Container] = None

def get_container() -> Container:
    """Get the global container instance."""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container

def set_container(container: Container) -> None:
    """Set the global container instance."""
    global _global_container
    _global_container = container

def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    _global_container = None