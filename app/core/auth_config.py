"""Authentication configuration for Google OAuth and JWT."""

from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthConfig(BaseSettings):
    """Authentication configuration section."""

    model_config = SettingsConfigDict(
        env_prefix="AUTH__",
        case_sensitive=False,
    )

    # Google OAuth Settings
    google_client_id: str = Field(
        default="",
        description="Google OAuth Client ID from Google Cloud Console"
    )
    google_client_secret: str = Field(
        default="",
        description="Google OAuth Client Secret"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:5173/auth/callback",
        description="OAuth callback URL (frontend)"
    )

    # JWT Settings
    jwt_secret_key: str = Field(
        default="change-this-secret-key-in-production-min-32-chars",
        description="Secret key for JWT signing (min 32 characters)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=15,
        description="Access token expiry in minutes",
        ge=5,
        le=60
    )
    refresh_token_expire_days: int = Field(
        default=30,
        description="Refresh token expiry in days",
        ge=1,
        le=90
    )

    # Domain Restriction (optional - empty list allows all domains)
    allowed_email_domains: List[str] = Field(
        default=[],
        description="Allowed email domains (empty = all allowed)"
    )

    # Session Management
    max_sessions_per_user: int = Field(
        default=5,
        description="Maximum concurrent sessions per user",
        ge=1,
        le=20
    )

    @property
    def is_configured(self) -> bool:
        """Check if Google OAuth is properly configured."""
        return bool(self.google_client_id and self.google_client_secret)


@lru_cache()
def get_auth_config() -> AuthConfig:
    """Get cached authentication configuration."""
    return AuthConfig()
