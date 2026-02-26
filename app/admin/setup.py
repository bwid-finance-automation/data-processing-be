"""Setup function to mount SQLAdmin onto the FastAPI application."""

import os
from pathlib import Path
from fastapi import FastAPI
from sqladmin import Admin

from app.infrastructure.database.connection import get_async_engine
from app.admin.auth import AdminAuth
from app.admin.feature_toggles import FeatureTogglesAdminView
from app.admin.views import (
    UserAdmin,
    UserSessionAdmin,
    SystemSettingsAdmin,
    BankStatementAdmin,
    BankTransactionAdmin,
    BankBalanceAdmin,
    ContractAdmin,
    AIUsageAdmin,
    CashReportSessionAdmin,
    CashReportUploadedFileAdmin,
)


def setup_admin(app: FastAPI) -> Admin:
    """Mount SQLAdmin dashboard onto the FastAPI app.

    Args:
        app: The FastAPI application instance.

    Returns:
        The configured Admin instance.
    """
    engine = get_async_engine()

    # Authentication backend with secret key from environment
    secret_key = os.environ.get("ADMIN_SECRET_KEY", os.environ.get("SECRET_KEY", "change-me-in-production"))
    authentication_backend = AdminAuth(secret_key=secret_key)
    templates_dir = str(Path(__file__).resolve().parents[2] / "templates")

    admin = Admin(
        app,
        engine,
        authentication_backend=authentication_backend,
        title="BWID Automation Admin",
        base_url="/admin",
        templates_dir=templates_dir,
    )

    # Register views in business-flow order for a coherent sidebar.
    admin.add_view(UserAdmin)
    admin.add_view(UserSessionAdmin)

    admin.add_view(FeatureTogglesAdminView)
    admin.add_view(SystemSettingsAdmin)  # Hidden from menu; kept for advanced direct access.

    admin.add_view(AIUsageAdmin)

    admin.add_view(BankStatementAdmin)
    admin.add_view(BankTransactionAdmin)
    admin.add_view(BankBalanceAdmin)

    admin.add_view(CashReportSessionAdmin)
    admin.add_view(CashReportUploadedFileAdmin)

    admin.add_view(ContractAdmin)

    return admin
