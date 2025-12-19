# app/core/dependencies.py
"""Dependency injection for FastAPI."""

from functools import lru_cache
from typing import Optional, AsyncGenerator, TYPE_CHECKING

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings, Settings

# Security scheme (optional - for future authentication)
security = HTTPBearer(auto_error=False)


# =============================================================================
# Database Dependencies
# =============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    from app.infrastructure.database.session import get_db as db_session_generator
    async for session in db_session_generator():
        yield session


# Repository dependencies - lazy import to avoid circular imports
def get_bank_statement_repository(db: AsyncSession = Depends(get_db)):
    """Get BankStatementRepository instance."""
    from app.infrastructure.persistence.repositories import BankStatementRepository
    return BankStatementRepository(db)


def get_bank_transaction_repository(db: AsyncSession = Depends(get_db)):
    """Get BankTransactionRepository instance."""
    from app.infrastructure.persistence.repositories import BankTransactionRepository
    return BankTransactionRepository(db)


def get_bank_balance_repository(db: AsyncSession = Depends(get_db)):
    """Get BankBalanceRepository instance."""
    from app.infrastructure.persistence.repositories import BankBalanceRepository
    return BankBalanceRepository(db)


def get_contract_repository(db: AsyncSession = Depends(get_db)):
    """Get ContractRepository instance."""
    from app.infrastructure.persistence.repositories import ContractRepository
    return ContractRepository(db)


def get_analysis_session_repository(db: AsyncSession = Depends(get_db)):
    """Get AnalysisSessionRepository instance."""
    from app.infrastructure.persistence.repositories import AnalysisSessionRepository
    return AnalysisSessionRepository(db)


def get_file_upload_repository(db: AsyncSession = Depends(get_db)):
    """Get FileUploadRepository instance."""
    from app.infrastructure.persistence.repositories import FileUploadRepository
    return FileUploadRepository(db)


def get_bank_statement_db_service(db: AsyncSession = Depends(get_db)):
    """Get BankStatementDbService instance."""
    from app.application.finance.bank_statement_parser.bank_statement_db_service import BankStatementDbService
    return BankStatementDbService(db)


def get_project_service(db: AsyncSession = Depends(get_db)):
    """Get ProjectService instance."""
    from app.application.project.project_service import ProjectService
    return ProjectService(db)


def get_project_repository(db: AsyncSession = Depends(get_db)):
    """Get ProjectRepository instance."""
    from app.infrastructure.persistence.repositories import ProjectRepository
    return ProjectRepository(db)


def get_project_case_repository(db: AsyncSession = Depends(get_db)):
    """Get ProjectCaseRepository instance."""
    from app.infrastructure.persistence.repositories import ProjectCaseRepository
    return ProjectCaseRepository(db)


def get_ai_usage_repository(db: AsyncSession = Depends(get_db)):
    """Get AIUsageRepository instance."""
    from app.infrastructure.persistence.repositories import AIUsageRepository
    return AIUsageRepository(db)


def get_analysis_service():
    """Get analysis service singleton."""
    # Lazy import to avoid circular dependency
    try:
        from app.application.finance.use_cases.analysis_service import analysis_service
        return analysis_service
    except ImportError:
        # Fallback - service not available
        return None

def get_current_session(session_id: str, service = Depends(get_analysis_service)):
    """Dependency to get and validate current session."""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )
    return session

async def validate_file_upload(file_content: bytes, filename: str, settings: Settings = Depends(get_settings)):
    """Validate uploaded file."""
    # Check file size
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )

    # Check file extension
    if filename:
        extension = "." + filename.split(".")[-1].lower()
        if extension not in settings.allowed_file_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_extensions)}"
            )

    return True