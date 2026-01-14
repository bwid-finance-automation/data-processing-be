"""Request and response schemas for authentication endpoints."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, ConfigDict


# ==================== Request Schemas ====================


class GoogleCallbackRequest(BaseModel):
    """Request for Google OAuth callback."""

    code: str = Field(..., description="Authorization code from Google")
    redirect_uri: Optional[str] = Field(
        None, description="Redirect URI used in authorization (optional override)"
    )


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token."""

    refresh_token: str = Field(..., description="Refresh token")


class LogoutRequest(BaseModel):
    """Request to logout."""

    refresh_token: Optional[str] = Field(
        None, description="Specific refresh token to revoke"
    )
    logout_all: bool = Field(
        False, description="If true, revoke all sessions"
    )


class LoginRequest(BaseModel):
    """Request for username/password login."""

    username: str = Field(..., min_length=1, description="Username or email")
    password: str = Field(..., min_length=1, description="Password")


# ==================== Response Schemas ====================


class UserResponse(BaseModel):
    """User profile response."""

    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    email: str
    email_verified: bool
    full_name: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture_url: Optional[str] = None
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    login_count: int = 0
    created_at: datetime


class TokenResponse(BaseModel):
    """Token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = Field(..., description="Access token expiry in seconds")
    user: UserResponse


class GoogleAuthUrlResponse(BaseModel):
    """Google OAuth URL response."""

    authorization_url: str
    state: str = Field(..., description="CSRF protection state token")


class SessionResponse(BaseModel):
    """User session response."""

    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    is_current: bool = False


class SessionListResponse(BaseModel):
    """List of user sessions."""

    sessions: List[SessionResponse]
    total: int


class AuthStatusResponse(BaseModel):
    """Authentication status response."""

    is_authenticated: bool
    user: Optional[UserResponse] = None


class MessageResponse(BaseModel):
    """Simple message response."""

    message: str
    success: bool = True


# ==================== Admin Schemas ====================


class UserListResponse(BaseModel):
    """List of users for admin."""

    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class UpdateUserRoleRequest(BaseModel):
    """Request to update user role."""

    role: str = Field(..., pattern="^(user|admin)$", description="New role: 'user' or 'admin'")


class UpdateUserStatusRequest(BaseModel):
    """Request to update user active status."""

    is_active: bool = Field(..., description="Whether the user account is active")


class AdminUserResponse(UserResponse):
    """Extended user response for admin with additional fields."""

    google_id: Optional[str] = None
    last_login_ip: Optional[str] = None
    is_deleted: bool = False
