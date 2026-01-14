"""System settings API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_user, require_role
from app.infrastructure.database.models.user import UserModel
from app.infrastructure.persistence.repositories import SystemSettingsRepository
from app.presentation.schemas.system_settings_schemas import (
    FeatureTogglesRequest,
    FeatureTogglesResponse,
    FeatureConfig,
    FeatureStatusResponse,
    MessageResponse,
)

router = APIRouter(prefix="/settings", tags=["System Settings"])


# ==================== Public Endpoints ====================


@router.get("/features", response_model=FeatureTogglesResponse)
async def get_feature_toggles(
    db: AsyncSession = Depends(get_db),
):
    """
    Get all feature toggles (public endpoint).

    Returns the current enabled/disabled status of all features.
    Used by frontend to check which features are available.
    """
    repo = SystemSettingsRepository(db)
    features = await repo.get_feature_toggles()

    return FeatureTogglesResponse(
        bankStatementOcr=FeatureConfig(**features.get("bankStatementOcr", {"enabled": True, "disabledMessage": ""})),
        contractOcr=FeatureConfig(**features.get("contractOcr", {"enabled": True, "disabledMessage": ""})),
        varianceAnalysis=FeatureConfig(**features.get("varianceAnalysis", {"enabled": True, "disabledMessage": ""})),
        glaAnalysis=FeatureConfig(**features.get("glaAnalysis", {"enabled": True, "disabledMessage": ""})),
        excelComparison=FeatureConfig(**features.get("excelComparison", {"enabled": True, "disabledMessage": ""})),
        utilityBilling=FeatureConfig(**features.get("utilityBilling", {"enabled": True, "disabledMessage": ""})),
    )


@router.get("/features/{feature_key}", response_model=FeatureStatusResponse)
async def get_feature_status(
    feature_key: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Check if a specific feature is enabled.

    Returns the feature status and disabled message if applicable.
    """
    valid_features = [
        "bankStatementOcr",
        "contractOcr",
        "varianceAnalysis",
        "glaAnalysis",
        "excelComparison",
        "utilityBilling",
    ]

    if feature_key not in valid_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feature key. Valid keys: {', '.join(valid_features)}",
        )

    repo = SystemSettingsRepository(db)
    is_enabled = await repo.is_feature_enabled(feature_key)
    disabled_message = await repo.get_feature_disabled_message(feature_key)

    return FeatureStatusResponse(
        feature=feature_key,
        enabled=is_enabled,
        disabledMessage=disabled_message,
    )


# ==================== Admin Endpoints ====================


@router.put("/features", response_model=FeatureTogglesResponse, tags=["Admin"])
async def update_feature_toggles(
    request_body: FeatureTogglesRequest,
    current_user: UserModel = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update feature toggles (Admin only).

    Allows admin to enable/disable features and set disabled messages.
    Only provided features will be updated; others remain unchanged.
    """
    repo = SystemSettingsRepository(db)

    # Get current features
    current_features = await repo.get_feature_toggles()

    # Update with provided values
    update_dict = request_body.model_dump(exclude_none=True)
    for key, value in update_dict.items():
        current_features[key] = value

    # Save updated features
    await repo.set_feature_toggles(current_features)
    await db.commit()

    # Return updated features
    return FeatureTogglesResponse(
        bankStatementOcr=FeatureConfig(**current_features.get("bankStatementOcr", {"enabled": True, "disabledMessage": ""})),
        contractOcr=FeatureConfig(**current_features.get("contractOcr", {"enabled": True, "disabledMessage": ""})),
        varianceAnalysis=FeatureConfig(**current_features.get("varianceAnalysis", {"enabled": True, "disabledMessage": ""})),
        glaAnalysis=FeatureConfig(**current_features.get("glaAnalysis", {"enabled": True, "disabledMessage": ""})),
        excelComparison=FeatureConfig(**current_features.get("excelComparison", {"enabled": True, "disabledMessage": ""})),
        utilityBilling=FeatureConfig(**current_features.get("utilityBilling", {"enabled": True, "disabledMessage": ""})),
    )


@router.patch("/features/{feature_key}", response_model=FeatureStatusResponse, tags=["Admin"])
async def update_single_feature(
    feature_key: str,
    enabled: bool,
    disabled_message: str = None,
    current_user: UserModel = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a single feature's status (Admin only).
    """
    valid_features = [
        "bankStatementOcr",
        "contractOcr",
        "varianceAnalysis",
        "glaAnalysis",
        "excelComparison",
        "utilityBilling",
    ]

    if feature_key not in valid_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feature key. Valid keys: {', '.join(valid_features)}",
        )

    repo = SystemSettingsRepository(db)
    current_features = await repo.get_feature_toggles()

    # Update the specific feature
    feature_config = current_features.get(feature_key, {"enabled": True, "disabledMessage": ""})
    feature_config["enabled"] = enabled
    if disabled_message is not None:
        feature_config["disabledMessage"] = disabled_message

    current_features[feature_key] = feature_config

    await repo.set_feature_toggles(current_features)
    await db.commit()

    return FeatureStatusResponse(
        feature=feature_key,
        enabled=enabled,
        disabledMessage=feature_config.get("disabledMessage") if not enabled else None,
    )
