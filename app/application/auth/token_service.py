"""JWT token generation and validation service."""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any

from jose import jwt, JWTError
from passlib.hash import bcrypt

from app.core.auth_config import get_auth_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class TokenService:
    """Service for JWT token operations."""

    def __init__(self):
        self.config = get_auth_config()

    def create_access_token(
        self,
        user_id: int,
        email: str,
        role: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, datetime]:
        """
        Create JWT access token.

        Args:
            user_id: User's database ID
            email: User's email
            role: User's role
            additional_claims: Optional additional JWT claims

        Returns:
            Tuple of (token_string, expiration_datetime)
        """
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self.config.access_token_expire_minutes
        )

        payload = {
            "sub": str(user_id),
            "email": email,
            "role": role,
            "type": "access",
            "exp": expires_at,
            "iat": datetime.now(timezone.utc),
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
        )

        return token, expires_at

    def create_refresh_token(self) -> Tuple[str, str, datetime]:
        """
        Create refresh token.

        Returns:
            Tuple of (token_string, token_hash, expiration_datetime)
        """
        token = secrets.token_urlsafe(64)
        token_hash = bcrypt.hash(token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.config.refresh_token_expire_days
        )

        return token, token_hash, expires_at

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode access token.

        Args:
            token: JWT access token string

        Returns:
            Decoded payload dict or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )

            # Verify token type
            if payload.get("type") != "access":
                logger.warning("Token type mismatch: expected 'access'")
                return None

            return payload

        except JWTError as e:
            logger.debug(f"JWT verification failed: {e}")
            return None

    def verify_refresh_token(self, token: str, token_hash: str) -> bool:
        """
        Verify refresh token against stored hash.

        Args:
            token: Raw refresh token
            token_hash: Stored bcrypt hash

        Returns:
            True if token matches hash
        """
        try:
            return bcrypt.verify(token, token_hash)
        except Exception as e:
            logger.debug(f"Refresh token verification failed: {e}")
            return False

    def get_token_expiry_seconds(self) -> int:
        """Get access token expiry in seconds."""
        return self.config.access_token_expire_minutes * 60

    def decode_token_unverified(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for debugging/logging).

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None
        """
        try:
            return jwt.get_unverified_claims(token)
        except Exception:
            return None


# Singleton instance
_token_service: Optional[TokenService] = None


def get_token_service() -> TokenService:
    """Get or create token service singleton."""
    global _token_service
    if _token_service is None:
        _token_service = TokenService()
    return _token_service
