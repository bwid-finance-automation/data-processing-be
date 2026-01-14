"""User database models for authentication (Google OAuth + Local accounts)."""

from datetime import datetime
from typing import Optional, List
import uuid as uuid_lib

from sqlalchemy import String, Boolean, Integer, ForeignKey, Index, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin, SoftDeleteMixin


class UserModel(Base, TimestampMixin, SoftDeleteMixin):
    """Model for authenticated users (Google OAuth or local accounts)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )

    # Local account credentials (for admin accounts)
    username: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Google OAuth info (nullable for local accounts)
    google_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Profile info
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    given_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    family_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    picture_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    locale: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Authorization
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Login tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_login_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Metadata (for future extensibility)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    sessions: Mapped[List["UserSessionModel"]] = relationship(
        "UserSessionModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_users_uuid", "uuid"),
        Index("ix_users_username", "username"),
        Index("ix_users_google_id", "google_id"),
        Index("ix_users_email", "email"),
        Index("ix_users_is_active", "is_active"),
        Index("ix_users_role", "role"),
        Index("ix_users_is_deleted", "is_deleted"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class UserSessionModel(Base, TimestampMixin):
    """Model for tracking user sessions (refresh tokens)."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid_lib.uuid4,
        unique=True,
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Session info
    refresh_token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    device_info: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Validity
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationship
    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="sessions",
    )

    __table_args__ = (
        Index("ix_user_sessions_uuid", "uuid"),
        Index("ix_user_sessions_user_id", "user_id"),
        Index("ix_user_sessions_refresh_token_hash", "refresh_token_hash"),
        Index("ix_user_sessions_expires_at", "expires_at"),
        Index("ix_user_sessions_is_revoked", "is_revoked"),
    )

    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, revoked={self.is_revoked})>"

    @property
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return (
            not self.is_revoked
            and self.expires_at > datetime.now(self.expires_at.tzinfo)
        )

    def revoke(self) -> None:
        """Revoke this session."""
        self.is_revoked = True
        self.revoked_at = datetime.now(self.expires_at.tzinfo)
