"""Google OAuth client for authentication."""

import secrets
from typing import Optional, Dict, Any
from dataclasses import dataclass

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client

from app.core.auth_config import get_auth_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Google OAuth endpoints
GOOGLE_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


@dataclass
class GoogleUserInfo:
    """Google user information from OAuth."""

    google_id: str
    email: str
    email_verified: bool
    full_name: str
    given_name: Optional[str]
    family_name: Optional[str]
    picture_url: Optional[str]
    locale: Optional[str]


class GoogleOAuthClient:
    """Client for Google OAuth 2.0 authentication."""

    def __init__(self):
        self.config = get_auth_config()

    def _create_oauth_client(self) -> AsyncOAuth2Client:
        """Create an OAuth2 client instance."""
        return AsyncOAuth2Client(
            client_id=self.config.google_client_id,
            client_secret=self.config.google_client_secret,
            redirect_uri=self.config.google_redirect_uri,
        )

    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Generate Google OAuth authorization URL.

        Args:
            state: Optional CSRF state token (generated if not provided)

        Returns:
            Tuple of (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        client = self._create_oauth_client()
        url, _ = client.create_authorization_url(
            GOOGLE_AUTHORIZATION_URL,
            state=state,
            scope="openid email profile",
            access_type="offline",
            prompt="consent",
        )

        return url, state

    async def exchange_code_for_tokens(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from Google
            redirect_uri: Override redirect URI (must match original)

        Returns:
            Token response containing access_token, refresh_token, etc.
        """
        client = self._create_oauth_client()

        if redirect_uri:
            client.redirect_uri = redirect_uri

        try:
            token = await client.fetch_token(
                GOOGLE_TOKEN_URL,
                code=code,
                grant_type="authorization_code",
            )
            return dict(token)
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            raise

    async def get_user_info(self, access_token: str) -> GoogleUserInfo:
        """
        Get user information from Google using access token.

        Args:
            access_token: Google OAuth access token

        Returns:
            GoogleUserInfo dataclass with user details
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            data = response.json()

        return GoogleUserInfo(
            google_id=data.get("sub", ""),
            email=data.get("email", ""),
            email_verified=data.get("email_verified", False),
            full_name=data.get("name", ""),
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            picture_url=data.get("picture"),
            locale=data.get("locale"),
        )

    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a Google OAuth token.

        Args:
            token: Access or refresh token to revoke

        Returns:
            True if revocation was successful
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    GOOGLE_REVOKE_URL,
                    data={"token": token},
                )
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Failed to revoke Google token: {e}")
                return False

    def is_email_allowed(self, email: str) -> bool:
        """
        Check if email domain is allowed.

        Args:
            email: Email address to check

        Returns:
            True if email domain is allowed or no restrictions set
        """
        if not self.config.allowed_email_domains:
            return True

        domain = email.split("@")[-1].lower()
        allowed_domains = [d.lower() for d in self.config.allowed_email_domains]

        return domain in allowed_domains


# Singleton instance
_google_oauth_client: Optional[GoogleOAuthClient] = None


def get_google_oauth_client() -> GoogleOAuthClient:
    """Get or create Google OAuth client singleton."""
    global _google_oauth_client
    if _google_oauth_client is None:
        _google_oauth_client = GoogleOAuthClient()
    return _google_oauth_client
