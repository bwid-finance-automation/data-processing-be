# app/core/dependencies.py
"""Dependency injection for FastAPI."""

from functools import lru_cache
from typing import Optional, AsyncGenerator, TYPE_CHECKING

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings, Settings

# Security scheme for JWT Bearer token
security = HTTPBearer(auto_error=False)
security_required = HTTPBearer(auto_error=True)


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


# =============================================================================
# Authentication Dependencies
# =============================================================================

def get_auth_service(db: AsyncSession = Depends(get_db)):
    """Get AuthService instance."""
    from app.application.auth.auth_service import AuthService
    return AuthService(db)


def get_user_repository(db: AsyncSession = Depends(get_db)):
    """Get UserRepository instance."""
    from app.infrastructure.persistence.repositories import UserRepository
    return UserRepository(db)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_required),
    db: AsyncSession = Depends(get_db),
):
    """
    FastAPI dependency that extracts and validates the current user from JWT token.

    Usage:
        @router.get("/profile")
        async def get_profile(current_user: UserModel = Depends(get_current_user)):
            return current_user

    Raises:
        HTTPException 401: If token is missing or invalid
    """
    from app.application.auth.token_service import get_token_service
    from app.infrastructure.persistence.repositories import UserRepository

    token_service = get_token_service()
    user_repo = UserRepository(db)

    # Verify token
    payload = token_service.verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user_id = int(payload.get("sub", 0))
    user = await user_repo.get(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Optional authentication - returns user if authenticated, None otherwise.

    Usage for routes that work both authenticated and unauthenticated:
        @router.get("/items")
        async def get_items(user: Optional[UserModel] = Depends(get_current_user_optional)):
            if user:
                # Show personalized items
            else:
                # Show public items
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.delete("/users/{user_id}")
        async def delete_user(
            user_id: int,
            current_user: UserModel = Depends(require_role("admin"))
        ):
            ...
    """
    async def role_checker(
        credentials: HTTPAuthorizationCredentials = Depends(security_required),
        db: AsyncSession = Depends(get_db),
    ):
        user = await get_current_user(credentials, db)

        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role}",
            )

        return user

    return role_checker