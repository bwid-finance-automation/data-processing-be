"""Custom SQLAdmin view for editing feature toggles with form controls."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from sqladmin import BaseView, expose

from app.infrastructure.database.connection import get_async_session_factory
from app.infrastructure.persistence.repositories.system_settings_repository import (
    SystemSettingsRepository,
)


class FeatureTogglesAdminView(BaseView):
    """UI-first admin page for toggling feature flags."""

    name = "Feature Toggles UI"
    icon = "fa-solid fa-sliders"
    category = "Feature Toggles"

    FEATURE_LABELS = {
        "bankStatementOcr": "Bank Statement OCR",
        "contractOcr": "Contract OCR",
        "varianceAnalysis": "Variance Analysis",
        "glaAnalysis": "GLA Analysis",
        "excelComparison": "Excel Comparison",
        "utilityBilling": "Utility Billing",
        "cashReport": "Cash Report",
        "ntmEbitdaAnalysis": "NTM EBITDA Analysis",
    }

    @expose(
        "/feature-toggles",
        methods=["GET", "POST"],
        identity="feature_toggles_ui",
    )
    async def feature_toggles(self, request: Request) -> Response:
        """Render and process form for feature toggles."""
        session_factory = get_async_session_factory()

        async with session_factory() as session:
            repo = SystemSettingsRepository(session)
            current_features = await repo.get_feature_toggles()

            if request.method == "POST":
                form = await request.form()
                updated_features = {}

                for feature_key in self.FEATURE_LABELS:
                    enabled = form.get(f"{feature_key}_enabled") == "on"
                    disabled_message = str(form.get(f"{feature_key}_message", "")).strip()
                    updated_features[feature_key] = {
                        "enabled": enabled,
                        "disabledMessage": disabled_message,
                    }

                # Preserve any extra feature keys if they exist.
                for key, value in current_features.items():
                    if key not in updated_features:
                        updated_features[key] = value

                await repo.set_feature_toggles(updated_features)
                await session.commit()

                return RedirectResponse(
                    request.url_for("admin:feature_toggles_ui").include_query_params(saved=1),
                    status_code=302,
                )

        feature_items = []
        for feature_key, feature_label in self.FEATURE_LABELS.items():
            config = current_features.get(feature_key, {})
            feature_items.append(
                {
                    "key": feature_key,
                    "label": feature_label,
                    "enabled": bool(config.get("enabled", True)),
                    "disabled_message": str(config.get("disabledMessage", "")),
                }
            )

        context = {
            "title": "Feature Toggles UI",
            "subtitle": "Enable/disable features without editing raw JSON",
            "feature_items": feature_items,
            "saved": request.query_params.get("saved") == "1",
        }
        return await self.templates.TemplateResponse(
            request,
            "admin/feature_toggles.html",
            context,
        )
