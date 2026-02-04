"""System settings repository implementation."""

from typing import Optional, Dict, Any, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.database.models.system_settings import SystemSettingsModel


class SystemSettingsRepository(BaseRepository[SystemSettingsModel]):
    """Repository for system settings operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(SystemSettingsModel, session)

    async def get_by_key(self, key: str) -> Optional[SystemSettingsModel]:
        """Get a setting by key."""
        result = await self.session.execute(
            select(SystemSettingsModel).where(SystemSettingsModel.key == key)
        )
        return result.scalar_one_or_none()

    async def get_value(self, key: str, default: Any = None) -> Any:
        """Get the value of a setting by key, returning default if not found."""
        setting = await self.get_by_key(key)
        if setting is None:
            return default
        return setting.value

    async def set_value(
        self,
        key: str,
        value: Dict[str, Any],
        description: Optional[str] = None,
    ) -> SystemSettingsModel:
        """Set or update a setting value."""
        existing = await self.get_by_key(key)

        if existing:
            existing.value = value
            if description is not None:
                existing.description = description
            await self.session.flush()
            await self.session.refresh(existing)
            return existing
        else:
            new_setting = SystemSettingsModel(
                key=key,
                value=value,
                description=description,
            )
            self.session.add(new_setting)
            await self.session.flush()
            await self.session.refresh(new_setting)
            return new_setting

    async def get_all_settings(self) -> List[SystemSettingsModel]:
        """Get all system settings."""
        result = await self.session.execute(
            select(SystemSettingsModel).order_by(SystemSettingsModel.key)
        )
        return list(result.scalars().all())

    async def get_feature_toggles(self) -> Dict[str, Any]:
        """Get feature toggles setting or return default."""
        default_features = {
            "bankStatementOcr": {
                "enabled": True,
                "disabledMessage": "Bank Statement OCR is temporarily unavailable. Please contact Admin for assistance.",
            },
            "contractOcr": {
                "enabled": True,
                "disabledMessage": "Contract OCR is temporarily unavailable. Please contact Admin for assistance.",
            },
            "varianceAnalysis": {
                "enabled": True,
                "disabledMessage": "Variance Analysis is temporarily unavailable. Please contact Admin for assistance.",
            },
            "glaAnalysis": {
                "enabled": True,
                "disabledMessage": "GLA Analysis is temporarily unavailable. Please contact Admin for assistance.",
            },
            "excelComparison": {
                "enabled": True,
                "disabledMessage": "Excel Comparison is temporarily unavailable. Please contact Admin for assistance.",
            },
            "utilityBilling": {
                "enabled": True,
                "disabledMessage": "Utility Billing is temporarily unavailable. Please contact Admin for assistance.",
            },
            "cashReport": {
                "enabled": True,
                "disabledMessage": "Cash Report is temporarily unavailable. Please contact Admin for assistance.",
            },
            "ntmEbitdaAnalysis": {
                "enabled": True,
                "disabledMessage": "NTM EBITDA Analysis is temporarily unavailable. Please contact Admin for assistance.",
            },
        }

        return await self.get_value("feature_toggles", default_features)

    async def set_feature_toggles(self, features: Dict[str, Any]) -> SystemSettingsModel:
        """Set feature toggles setting."""
        return await self.set_value(
            key="feature_toggles",
            value=features,
            description="Feature toggles for enabling/disabling application features",
        )

    async def is_feature_enabled(self, feature_key: str) -> bool:
        """Check if a specific feature is enabled."""
        features = await self.get_feature_toggles()
        feature = features.get(feature_key, {})
        return feature.get("enabled", True)

    async def get_feature_disabled_message(self, feature_key: str) -> Optional[str]:
        """Get the disabled message for a feature."""
        features = await self.get_feature_toggles()
        feature = features.get(feature_key, {})
        if not feature.get("enabled", True):
            return feature.get("disabledMessage", "This feature is temporarily unavailable. Please contact Admin for assistance.")
        return None
