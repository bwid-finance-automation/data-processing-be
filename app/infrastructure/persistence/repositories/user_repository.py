"""Repository for user and session data access."""

from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.user import UserModel, UserSessionModel


class UserRepository(BaseRepository[UserModel]):
    """Repository for user operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(UserModel, session)

    async def get_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email address."""
        result = await self.session.execute(
            select(UserModel)
            .where(UserModel.email == email)
            .where(UserModel.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_by_google_id(self, google_id: str) -> Optional[UserModel]:
        """Get user by Google OAuth ID."""
        result = await self.session.execute(
            select(UserModel)
            .where(UserModel.google_id == google_id)
            .where(UserModel.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_by_uuid(self, uuid: UUID) -> Optional[UserModel]:
        """Get user by UUID."""
        result = await self.session.execute(
            select(UserModel)
            .where(UserModel.uuid == uuid)
            .where(UserModel.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_with_sessions(self, user_id: int) -> Optional[UserModel]:
        """Get user with their active sessions."""
        result = await self.session.execute(
            select(UserModel)
            .options(selectinload(UserModel.sessions))
            .where(UserModel.id == user_id)
            .where(UserModel.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserModel]:
        """Get all active (non-deleted) users."""
        result = await self.session.execute(
            select(UserModel)
            .where(UserModel.is_deleted == False)
            .where(UserModel.is_active == True)
            .order_by(UserModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_last_login(
        self,
        user_id: int,
        ip_address: Optional[str] = None,
    ) -> Optional[UserModel]:
        """Update user's last login timestamp and increment login count."""
        user = await self.get(user_id)
        if user:
            user.last_login_at = datetime.now(timezone.utc)
            user.last_login_ip = ip_address
            user.login_count += 1
            await self.session.flush()
            await self.session.refresh(user)
        return user

    async def count_active_users(self) -> int:
        """Count total active users."""
        result = await self.session.execute(
            select(func.count())
            .select_from(UserModel)
            .where(UserModel.is_deleted == False)
            .where(UserModel.is_active == True)
        )
        return result.scalar_one()


class UserSessionRepository(BaseRepository[UserSessionModel]):
    """Repository for user session operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(UserSessionModel, session)

    async def get_by_refresh_token_hash(
        self,
        token_hash: str,
    ) -> Optional[UserSessionModel]:
        """Get session by refresh token hash."""
        result = await self.session.execute(
            select(UserSessionModel)
            .where(UserSessionModel.refresh_token_hash == token_hash)
            .where(UserSessionModel.is_revoked == False)
        )
        return result.scalar_one_or_none()

    async def get_user_sessions(
        self,
        user_id: int,
        include_revoked: bool = False,
    ) -> List[UserSessionModel]:
        """Get all sessions for a user."""
        query = select(UserSessionModel).where(
            UserSessionModel.user_id == user_id
        )

        if not include_revoked:
            query = query.where(UserSessionModel.is_revoked == False)

        query = query.order_by(UserSessionModel.created_at.desc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_active_sessions(
        self,
        user_id: int,
    ) -> List[UserSessionModel]:
        """Get active (non-expired, non-revoked) sessions for a user."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(UserSessionModel)
            .where(UserSessionModel.user_id == user_id)
            .where(UserSessionModel.is_revoked == False)
            .where(UserSessionModel.expires_at > now)
            .order_by(UserSessionModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_active_sessions(self, user_id: int) -> int:
        """Count active sessions for a user."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(func.count())
            .select_from(UserSessionModel)
            .where(UserSessionModel.user_id == user_id)
            .where(UserSessionModel.is_revoked == False)
            .where(UserSessionModel.expires_at > now)
        )
        return result.scalar_one()

    async def revoke_session(
        self,
        session_id: int,
    ) -> Optional[UserSessionModel]:
        """Revoke a specific session."""
        session = await self.get(session_id)
        if session:
            session.is_revoked = True
            session.revoked_at = datetime.now(timezone.utc)
            await self.session.flush()
            await self.session.refresh(session)
        return session

    async def revoke_all_user_sessions(
        self,
        user_id: int,
        except_session_id: Optional[int] = None,
    ) -> int:
        """Revoke all sessions for a user, optionally keeping one."""
        sessions = await self.get_active_sessions(user_id)
        count = 0

        for session in sessions:
            if except_session_id and session.id == except_session_id:
                continue
            session.is_revoked = True
            session.revoked_at = datetime.now(timezone.utc)
            count += 1

        await self.session.flush()
        return count

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions (cleanup job)."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(UserSessionModel)
            .where(UserSessionModel.expires_at < now)
        )
        sessions = list(result.scalars().all())

        for session in sessions:
            await self.session.delete(session)

        await self.session.flush()
        return len(sessions)
