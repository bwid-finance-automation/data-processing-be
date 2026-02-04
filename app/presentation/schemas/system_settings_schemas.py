"""Request and response schemas for system settings endpoints."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ==================== Feature Toggle Schemas ====================


class FeatureConfig(BaseModel):
    """Configuration for a single feature."""

    enabled: bool = Field(True, description="Whether the feature is enabled")
    disabledMessage: str = Field(
        "Tính năng này đang tạm ngưng. Vui lòng liên hệ Admin.",
        description="Message to display when feature is disabled",
    )


class FeatureTogglesRequest(BaseModel):
    """Request to update feature toggles."""

    bankStatementOcr: Optional[FeatureConfig] = None
    contractOcr: Optional[FeatureConfig] = None
    varianceAnalysis: Optional[FeatureConfig] = None
    glaAnalysis: Optional[FeatureConfig] = None
    excelComparison: Optional[FeatureConfig] = None
    utilityBilling: Optional[FeatureConfig] = None
    cashReport: Optional[FeatureConfig] = None
    ntmEbitdaAnalysis: Optional[FeatureConfig] = None


class FeatureTogglesResponse(BaseModel):
    """Response containing all feature toggles."""

    bankStatementOcr: FeatureConfig
    contractOcr: FeatureConfig
    varianceAnalysis: FeatureConfig
    glaAnalysis: FeatureConfig
    excelComparison: FeatureConfig
    utilityBilling: FeatureConfig
    cashReport: FeatureConfig
    ntmEbitdaAnalysis: FeatureConfig


# ==================== General Settings Schemas ====================


class SystemSettingResponse(BaseModel):
    """Response for a single system setting."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    key: str
    value: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SystemSettingsListResponse(BaseModel):
    """Response containing all system settings."""

    settings: list[SystemSettingResponse]


class SetSettingRequest(BaseModel):
    """Request to set a system setting."""

    key: str = Field(..., min_length=1, max_length=100, description="Setting key")
    value: Dict[str, Any] = Field(..., description="Setting value as JSON object")
    description: Optional[str] = Field(None, description="Description of the setting")


class FeatureStatusResponse(BaseModel):
    """Response for checking a single feature's status."""

    feature: str
    enabled: bool
    disabledMessage: Optional[str] = None


class MessageResponse(BaseModel):
    """Simple message response."""

    message: str
    success: bool = True
