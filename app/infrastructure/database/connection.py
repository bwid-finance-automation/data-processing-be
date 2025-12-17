"""Database connection management with async support."""

from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.unified_config import get_unified_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_async_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine

    if _engine is None:
        config = get_unified_config()
        db_config = config.database

        # Log connection mode
        if db_config.is_cloud:
            logger.info("Creating async database engine (Cloud mode - DATABASE_URL)")
        else:
            logger.info(f"Creating async database engine (Local mode - {db_config.host}:{db_config.port}/{db_config.name})")

        # Build engine kwargs
        engine_kwargs = {
            "echo": db_config.echo,
            "pool_pre_ping": True,  # Verify connections before use
            "pool_recycle": 3600,  # Recycle connections after 1 hour
        }

        # Add pool settings (not applicable for NullPool in serverless)
        if db_config.pool_size > 0:
            engine_kwargs["pool_size"] = db_config.pool_size
            engine_kwargs["max_overflow"] = db_config.max_overflow

        # Add SSL settings for cloud deployments
        if db_config.ssl_mode == "require":
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            engine_kwargs["connect_args"] = {"ssl": ssl_context}
            logger.info("SSL enabled for database connection")

        _engine = create_async_engine(
            db_config.async_url,
            **engine_kwargs
        )

        logger.info("Async database engine created successfully")

    return _engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _async_session_factory

    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        logger.info("Async session factory created")

    return _async_session_factory


async def init_db() -> None:
    """Initialize database connection and verify connectivity.

    Note: Table creation is handled by Alembic migrations.
    This function only initializes the connection pool.
    """
    # Import all models to register them with Base
    from app.infrastructure.database import models  # noqa: F401

    engine = get_async_engine()

    logger.info("Initializing database connection...")

    # Just verify we can connect - don't create tables
    # Table creation is handled by Alembic migrations
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    logger.info("Database connection initialized successfully")


async def close_db() -> None:
    """Close database connections."""
    global _engine, _async_session_factory

    if _engine is not None:
        logger.info("Closing database connections...")
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_context():
    """Async context manager for database sessions."""
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
