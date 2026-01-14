"""Authentication API endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_auth_service, get_current_user
from app.core.auth_config import get_auth_config
from app.application.auth.auth_service import AuthService, AuthenticationError
from app.infrastructure.database.models.user import UserModel
from app.presentation.schemas.auth_schemas import (
    GoogleCallbackRequest,
    RefreshTokenRequest,
    LogoutRequest,
    UserResponse,
    TokenResponse,
    GoogleAuthUrlResponse,
    SessionResponse,
    SessionListResponse,
    MessageResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def get_user_agent(request: Request) -> Optional[str]:
    """Extract user agent from request."""
    return request.headers.get("User-Agent")


@router.get("/google/url", response_model=GoogleAuthUrlResponse)
async def get_google_auth_url(
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Get Google OAuth authorization URL.

    Returns the URL to redirect the user to for Google authentication.
    """
    config = get_auth_config()
    if not config.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured",
        )

    url, state = auth_service.get_google_auth_url()

    return GoogleAuthUrlResponse(
        authorization_url=url,
        state=state,
    )


@router.post("/google/callback", response_model=TokenResponse)
async def google_callback(
    request_body: GoogleCallbackRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Handle Google OAuth callback.

    Exchange the authorization code for tokens and create/update user.
    """
    try:
        result = await auth_service.authenticate_with_google(
            code=request_body.code,
            redirect_uri=request_body.redirect_uri,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
        )

        return TokenResponse(
            access_token=result.tokens.access_token,
            refresh_token=result.tokens.refresh_token,
            token_type=result.tokens.token_type,
            expires_in=result.tokens.expires_in,
            user=UserResponse.model_validate(result.user),
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request_body: RefreshTokenRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Refresh access token using refresh token.

    Returns a new token pair.
    """
    try:
        result = await auth_service.refresh_tokens(
            refresh_token=request_body.refresh_token,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
        )

        return TokenResponse(
            access_token=result.tokens.access_token,
            refresh_token=result.tokens.refresh_token,
            token_type=result.tokens.token_type,
            expires_in=result.tokens.expires_in,
            user=UserResponse.model_validate(result.user),
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
        )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request_body: LogoutRequest,
    current_user: UserModel = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Logout current user.

    Optionally revoke specific token or all sessions.
    """
    await auth_service.logout(
        user_id=current_user.id,
        refresh_token=request_body.refresh_token,
        logout_all=request_body.logout_all,
    )

    message = "Logged out from all devices" if request_body.logout_all else "Logged out successfully"
    return MessageResponse(message=message)


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserModel = Depends(get_current_user),
):
    """
    Get current authenticated user's profile.
    """
    return UserResponse.model_validate(current_user)


@router.get("/sessions", response_model=SessionListResponse)
async def get_user_sessions(
    current_user: UserModel = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Get all active sessions for current user.
    """
    sessions = await auth_service.get_user_sessions(current_user.id)

    session_responses = [
        SessionResponse(
            uuid=s.uuid,
            device_info=s.device_info,
            ip_address=s.ip_address,
            user_agent=s.user_agent,
            created_at=s.created_at,
            expires_at=s.expires_at,
            is_current=False,  # Could be enhanced to mark current session
        )
        for s in sessions
    ]

    return SessionListResponse(
        sessions=session_responses,
        total=len(session_responses),
    )


@router.delete("/sessions/{session_uuid}", response_model=MessageResponse)
async def revoke_session(
    session_uuid: UUID,
    current_user: UserModel = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke a specific session by UUID.
    """
    from app.infrastructure.persistence.repositories import UserSessionRepository

    session_repo = UserSessionRepository(db)

    # Find session by UUID
    sessions = await auth_service.get_user_sessions(current_user.id)
    target_session = next(
        (s for s in sessions if s.uuid == session_uuid),
        None,
    )

    if not target_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    await auth_service.revoke_session(current_user.id, target_session.id)

    return MessageResponse(message="Session revoked successfully")


@router.delete("/sessions", response_model=MessageResponse)
async def revoke_all_sessions(
    current_user: UserModel = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Revoke all sessions for current user (logout everywhere).
    """
    await auth_service.logout(
        user_id=current_user.id,
        logout_all=True,
    )

    return MessageResponse(message="All sessions revoked successfully")


@router.get("/config")
async def get_auth_config_status():
    """
    Get authentication configuration status (public endpoint).
    """
    config = get_auth_config()

    return {
        "google_oauth_configured": config.is_configured,
        "domain_restriction_enabled": len(config.allowed_email_domains) > 0,
        "allowed_domains": config.allowed_email_domains if config.allowed_email_domains else None,
    }
