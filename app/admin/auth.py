"""SQLAdmin authentication backend.

Reuses existing verify_password() and user query infrastructure
to restrict admin dashboard access to users with role="admin".
"""

from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from sqlalchemy import select

from app.application.auth.auth_service import verify_password
from app.infrastructure.database.connection import get_async_session_factory
from app.infrastructure.database.models.user import UserModel

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class AdminAuth(AuthenticationBackend):
    """Authentication backend for SQLAdmin dashboard."""

    async def login(self, request: Request) -> bool:
        """Handle admin login form submission."""
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")

        # Query active local user by username.
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(UserModel).where(
                    UserModel.username == username,
                    UserModel.is_active.is_(True),
                    UserModel.is_deleted.is_(False),
                )
            )
            user = result.scalar_one_or_none()

        if not user or not user.password_hash:
            logger.warning(f"Admin login failed: user '{username}' not found or has no password")
            return False

        # Verify password
        if not verify_password(password, user.password_hash):
            logger.warning(f"Admin login failed: invalid password for user '{username}'")
            return False

        # Only allow admin role
        if user.role != "admin":
            logger.warning(f"Admin login failed: user '{username}' has role '{user.role}', not 'admin'")
            return False

        # Store user info in session.
        request.session.update({
            "admin_user_id": user.id,
            "admin_username": user.username,
        })
        return True

    async def logout(self, request: Request) -> bool:
        """Handle admin logout."""
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        """Check if the current request is authenticated."""
        admin_user_id = request.session.get("admin_user_id")
        if not admin_user_id:
            return False

        # Re-check user on each request to enforce current role and status.
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(UserModel).where(
                    UserModel.id == admin_user_id,
                    UserModel.role == "admin",
                    UserModel.is_active.is_(True),
                    UserModel.is_deleted.is_(False),
                )
            )
            user = result.scalar_one_or_none()

        if user is None:
            request.session.clear()
            return False

        return True
