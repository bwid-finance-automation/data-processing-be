"""Database session management for FastAPI dependency injection."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.connection import get_async_session_factory
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncSession:
    """
    Get a database session directly (not as a generator).

    Note: Caller is responsible for committing/closing the session.

    Usage:
        session = await get_db_session()
        try:
            # do work
            await session.commit()
        finally:
            await session.close()
    """
    session_factory = get_async_session_factory()
    return session_factory()
