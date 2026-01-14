"""Authentication service orchestrating login, logout, and token management."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_config import get_auth_config
from app.infrastructure.database.models.user import UserModel, UserSessionModel
from app.infrastructure.persistence.repositories.user_repository import (
    UserRepository,
    UserSessionRepository,
)
from app.infrastructure.external.google_oauth import (
    GoogleOAuthClient,
    GoogleUserInfo,
    get_google_oauth_client,
)
from app.application.auth.token_service import TokenService, get_token_service
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AuthTokens:
    """Authentication tokens response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 0


@dataclass
class AuthResult:
    """Result of authentication operation."""

    user: UserModel
    tokens: AuthTokens
    is_new_user: bool = False


class AuthenticationError(Exception):
    """Authentication-related errors."""

    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)
        self.session_repo = UserSessionRepository(db)
        self.token_service = get_token_service()
        self.oauth_client = get_google_oauth_client()
        self.config = get_auth_config()

    def get_google_auth_url(self, state: Optional[str] = None) -> Tuple[str, str]:
        """
        Get Google OAuth authorization URL.

        Args:
            state: Optional CSRF state token

        Returns:
            Tuple of (authorization_url, state)
        """
        return self.oauth_client.get_authorization_url(state)

    async def authenticate_with_google(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthResult:
        """
        Authenticate user with Google OAuth code.

        Args:
            code: Authorization code from Google
            redirect_uri: Override redirect URI
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            AuthResult with user and tokens

        Raises:
            AuthenticationError: If authentication fails
        """
        # Exchange code for Google tokens
        try:
            google_tokens = await self.oauth_client.exchange_code_for_tokens(
                code, redirect_uri
            )
        except Exception as e:
            logger.error(f"Failed to exchange code: {e}")
            raise AuthenticationError(
                "Failed to authenticate with Google",
                "GOOGLE_AUTH_FAILED",
            )

        # Get user info from Google
        access_token = google_tokens.get("access_token")
        if not access_token:
            raise AuthenticationError(
                "No access token received from Google",
                "GOOGLE_TOKEN_MISSING",
            )

        try:
            user_info = await self.oauth_client.get_user_info(access_token)
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise AuthenticationError(
                "Failed to get user information from Google",
                "GOOGLE_USERINFO_FAILED",
            )

        # Check email domain restriction
        if not self.oauth_client.is_email_allowed(user_info.email):
            raise AuthenticationError(
                f"Email domain not allowed: {user_info.email}",
                "EMAIL_DOMAIN_NOT_ALLOWED",
            )

        # Find or create user
        user = await self.user_repo.get_by_google_id(user_info.google_id)
        is_new_user = False

        if user:
            # Update existing user info
            user = await self._update_user_from_google(user, user_info)
        else:
            # Create new user
            user = await self._create_user_from_google(user_info)
            is_new_user = True

        # Check if user is active
        if not user.is_active:
            raise AuthenticationError(
                "Your account has been deactivated",
                "ACCOUNT_DEACTIVATED",
            )

        # Update last login
        await self.user_repo.update_last_login(user.id, ip_address)

        # Manage sessions (enforce max sessions limit)
        await self._enforce_session_limit(user.id)

        # Create session and tokens
        tokens = await self._create_session(user, ip_address, user_agent)

        await self.db.commit()

        return AuthResult(
            user=user,
            tokens=tokens,
            is_new_user=is_new_user,
        )

    async def refresh_tokens(
        self,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthResult:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Current refresh token
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            AuthResult with new tokens

        Raises:
            AuthenticationError: If refresh fails
        """
        # Find all active sessions and verify token against each
        # This is because we store hashed tokens
        sessions = await self.session_repo.get_all(limit=1000)

        valid_session: Optional[UserSessionModel] = None
        for session in sessions:
            if session.is_revoked:
                continue
            if session.expires_at < datetime.now(timezone.utc):
                continue
            if self.token_service.verify_refresh_token(
                refresh_token, session.refresh_token_hash
            ):
                valid_session = session
                break

        if not valid_session:
            raise AuthenticationError(
                "Invalid or expired refresh token",
                "INVALID_REFRESH_TOKEN",
            )

        # Get user
        user = await self.user_repo.get(valid_session.user_id)
        if not user or not user.is_active:
            raise AuthenticationError(
                "User not found or deactivated",
                "USER_NOT_FOUND",
            )

        # Revoke old session
        await self.session_repo.revoke_session(valid_session.id)

        # Create new session
        tokens = await self._create_session(user, ip_address, user_agent)

        await self.db.commit()

        return AuthResult(user=user, tokens=tokens)

    async def logout(
        self,
        user_id: int,
        refresh_token: Optional[str] = None,
        logout_all: bool = False,
    ) -> bool:
        """
        Logout user by revoking sessions.

        Args:
            user_id: User's database ID
            refresh_token: Specific token to revoke (optional)
            logout_all: If True, revoke all sessions

        Returns:
            True if logout successful
        """
        if logout_all:
            await self.session_repo.revoke_all_user_sessions(user_id)
        elif refresh_token:
            # Find and revoke specific session
            sessions = await self.session_repo.get_active_sessions(user_id)
            for session in sessions:
                if self.token_service.verify_refresh_token(
                    refresh_token, session.refresh_token_hash
                ):
                    await self.session_repo.revoke_session(session.id)
                    break

        await self.db.commit()
        return True

    async def get_user_by_id(self, user_id: int) -> Optional[UserModel]:
        """Get user by ID."""
        return await self.user_repo.get(user_id)

    async def get_user_sessions(self, user_id: int) -> list:
        """Get active sessions for a user."""
        return await self.session_repo.get_active_sessions(user_id)

    async def revoke_session(self, user_id: int, session_id: int) -> bool:
        """Revoke a specific session."""
        session = await self.session_repo.get(session_id)
        if session and session.user_id == user_id:
            await self.session_repo.revoke_session(session_id)
            await self.db.commit()
            return True
        return False

    # Private helper methods

    async def _create_user_from_google(self, user_info: GoogleUserInfo) -> UserModel:
        """Create new user from Google user info."""
        user = UserModel(
            google_id=user_info.google_id,
            email=user_info.email,
            email_verified=user_info.email_verified,
            full_name=user_info.full_name,
            given_name=user_info.given_name,
            family_name=user_info.family_name,
            picture_url=user_info.picture_url,
            locale=user_info.locale,
            role="user",
            is_active=True,
        )
        return await self.user_repo.create(user)

    async def _update_user_from_google(
        self, user: UserModel, user_info: GoogleUserInfo
    ) -> UserModel:
        """Update existing user with latest Google info."""
        user.email = user_info.email
        user.email_verified = user_info.email_verified
        user.full_name = user_info.full_name
        user.given_name = user_info.given_name
        user.family_name = user_info.family_name
        user.picture_url = user_info.picture_url
        user.locale = user_info.locale

        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def _create_session(
        self,
        user: UserModel,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthTokens:
        """Create new session and generate tokens."""
        # Generate tokens
        access_token, _ = self.token_service.create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )

        refresh_token, refresh_hash, expires_at = (
            self.token_service.create_refresh_token()
        )

        # Create session record
        session = UserSessionModel(
            user_id=user.id,
            refresh_token_hash=refresh_hash,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
        )
        await self.session_repo.create(session)

        return AuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.token_service.get_token_expiry_seconds(),
        )

    async def _enforce_session_limit(self, user_id: int) -> None:
        """Enforce maximum sessions per user limit."""
        active_count = await self.session_repo.count_active_sessions(user_id)

        if active_count >= self.config.max_sessions_per_user:
            # Revoke oldest sessions to make room
            sessions = await self.session_repo.get_active_sessions(user_id)
            sessions_to_revoke = active_count - self.config.max_sessions_per_user + 1

            for session in sessions[-sessions_to_revoke:]:
                await self.session_repo.revoke_session(session.id)

            logger.info(
                f"Revoked {sessions_to_revoke} sessions for user {user_id} "
                f"(limit: {self.config.max_sessions_per_user})"
            )
