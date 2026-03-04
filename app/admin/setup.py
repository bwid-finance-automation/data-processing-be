"""Setup function to mount SQLAdmin onto the FastAPI application."""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqladmin import Admin
from sqlalchemy import case, func, select
from starlette.requests import Request

from app.infrastructure.database.connection import (
    get_async_engine,
    get_async_session_factory,
)
from app.infrastructure.database.models.ai_usage import AIUsageModel
from app.infrastructure.database.models.bank_statement import (
    BankStatementModel,
    BankTransactionModel,
)
from app.infrastructure.database.models.cash_report_session import (
    CashReportSessionModel,
    CashReportSessionStatus,
    CashReportUploadedFileModel,
)
from app.infrastructure.database.models.contract import ContractModel
from app.infrastructure.database.models.user import UserModel, UserSessionModel
from app.shared.utils.logging_config import get_logger
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


logger = get_logger(__name__)


def _int_or_zero(value: int | None) -> int:
    """Return integer value with `None` normalized to 0."""
    return int(value or 0)


def _float_or_zero(value: float | None) -> float:
    """Return float value with `None` normalized to 0.0."""
    return float(value or 0.0)


def _naive_utc(value: datetime) -> datetime:
    """Normalize datetime to naive UTC for TIMESTAMP WITHOUT TIME ZONE columns."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _register_dashboard_stats_route(app: FastAPI, authentication_backend: AdminAuth) -> None:
    """Register dashboard metrics endpoint used by the custom admin landing page."""
    route_path = "/api/admin/dashboard-stats"
    if any(getattr(route, "path", None) == route_path for route in app.router.routes):
        return

    @app.get(route_path, include_in_schema=False)
    async def admin_dashboard_stats(request: Request) -> JSONResponse:
        """Return aggregated metrics for SQLAdmin dashboard widgets and charts."""
        if not await authentication_backend.authenticate(request):
            raise HTTPException(status_code=401, detail="Admin authentication required")

        now_tz = datetime.now(timezone.utc)
        now_naive = _naive_utc(now_tz)
        window_30d_tz = now_tz - timedelta(days=30)
        window_30d_naive = _naive_utc(now_tz - timedelta(days=30))
        daily_points = 14
        daily_start_date = (now_naive - timedelta(days=daily_points - 1)).date()
        daily_start_dt = datetime(
            daily_start_date.year,
            daily_start_date.month,
            daily_start_date.day,
        )

        session_factory = get_async_session_factory()
        try:
            async with session_factory() as session:
                total_users = _int_or_zero(
                    await session.scalar(
                        select(func.count(UserModel.id)).where(UserModel.is_deleted.is_(False))
                    )
                )
                active_users = _int_or_zero(
                    await session.scalar(
                        select(func.count(UserModel.id)).where(
                            UserModel.is_deleted.is_(False),
                            UserModel.is_active.is_(True),
                        )
                    )
                )
                new_users_30d = _int_or_zero(
                    await session.scalar(
                        select(func.count(UserModel.id)).where(
                            UserModel.is_deleted.is_(False),
                            UserModel.created_at >= window_30d_tz,
                        )
                    )
                )
                active_sessions = _int_or_zero(
                    await session.scalar(
                        select(func.count(UserSessionModel.id)).where(
                            UserSessionModel.is_revoked.is_(False),
                            UserSessionModel.expires_at > now_tz,
                        )
                    )
                )

                ai_requests_30d = _int_or_zero(
                    await session.scalar(
                        select(func.count(AIUsageModel.id)).where(
                            AIUsageModel.requested_at >= window_30d_naive
                        )
                    )
                )
                ai_success_30d = _int_or_zero(
                    await session.scalar(
                        select(
                            func.coalesce(
                                func.sum(
                                    case((AIUsageModel.success.is_(True), 1), else_=0)
                                ),
                                0,
                            )
                        ).where(AIUsageModel.requested_at >= window_30d_naive)
                    )
                )
                ai_tokens_30d = _int_or_zero(
                    await session.scalar(
                        select(func.coalesce(func.sum(AIUsageModel.total_tokens), 0)).where(
                            AIUsageModel.requested_at >= window_30d_naive
                        )
                    )
                )
                ai_cost_30d = _float_or_zero(
                    await session.scalar(
                        select(
                            func.coalesce(func.sum(AIUsageModel.estimated_cost_usd), 0.0)
                        ).where(AIUsageModel.requested_at >= window_30d_naive)
                    )
                )

                bank_statements_total = _int_or_zero(
                    await session.scalar(select(func.count(BankStatementModel.id)))
                )
                bank_transactions_total = _int_or_zero(
                    await session.scalar(select(func.count(BankTransactionModel.id)))
                )
                contracts_total = _int_or_zero(
                    await session.scalar(select(func.count(ContractModel.id)))
                )
                cash_report_sessions_total = _int_or_zero(
                    await session.scalar(select(func.count(CashReportSessionModel.id)))
                )
                cash_report_files_total = _int_or_zero(
                    await session.scalar(select(func.count(CashReportUploadedFileModel.id)))
                )
                cash_sessions_running = _int_or_zero(
                    await session.scalar(
                        select(func.count(CashReportSessionModel.id)).where(
                            CashReportSessionModel.status.in_(
                                [
                                    CashReportSessionStatus.ACTIVE,
                                    CashReportSessionStatus.PROCESSING,
                                ]
                            )
                        )
                    )
                )

                daily_rows = (
                    await session.execute(
                        select(
                            func.date(AIUsageModel.requested_at).label("day"),
                            func.count(AIUsageModel.id).label("total_requests"),
                            func.coalesce(
                                func.sum(
                                    case((AIUsageModel.success.is_(True), 1), else_=0)
                                ),
                                0,
                            ).label("successful_requests"),
                            func.coalesce(
                                func.sum(
                                    case((AIUsageModel.success.is_(False), 1), else_=0)
                                ),
                                0,
                            ).label("failed_requests"),
                        )
                        .where(AIUsageModel.requested_at >= daily_start_dt)
                        .group_by(func.date(AIUsageModel.requested_at))
                        .order_by(func.date(AIUsageModel.requested_at))
                    )
                ).all()

                provider_rows = (
                    await session.execute(
                        select(
                            AIUsageModel.provider.label("provider"),
                            func.count(AIUsageModel.id).label("requests"),
                        )
                        .where(AIUsageModel.requested_at >= window_30d_naive)
                        .group_by(AIUsageModel.provider)
                        .order_by(func.count(AIUsageModel.id).desc())
                        .limit(6)
                    )
                ).all()

                module_totals = [
                    {"label": "Users", "value": total_users},
                    {"label": "Active Sessions", "value": active_sessions},
                    {"label": "Bank Statements", "value": bank_statements_total},
                    {"label": "Transactions", "value": bank_transactions_total},
                    {"label": "Contracts", "value": contracts_total},
                    {"label": "Cash Sessions", "value": cash_report_sessions_total},
                ]

                daily_lookup: dict[date, dict[str, int]] = {}
                for row in daily_rows:
                    day_key: date
                    if isinstance(row.day, datetime):
                        day_key = row.day.date()
                    elif isinstance(row.day, str):
                        day_key = date.fromisoformat(row.day)
                    else:
                        day_key = row.day
                    daily_lookup[day_key] = {
                        "total_requests": _int_or_zero(row.total_requests),
                        "successful_requests": _int_or_zero(row.successful_requests),
                        "failed_requests": _int_or_zero(row.failed_requests),
                    }

                daily_series = []
                for offset in range(daily_points):
                    day = daily_start_date + timedelta(days=offset)
                    record = daily_lookup.get(
                        day,
                        {
                            "total_requests": 0,
                            "successful_requests": 0,
                            "failed_requests": 0,
                        },
                    )
                    daily_series.append(
                        {
                            "date": day.isoformat(),
                            **record,
                        }
                    )

                provider_breakdown = [
                    {
                        "provider": str(row.provider or "unknown").upper(),
                        "requests": _int_or_zero(row.requests),
                    }
                    for row in provider_rows
                ]

                success_rate = (
                    (ai_success_30d / ai_requests_30d) * 100.0
                    if ai_requests_30d > 0
                    else 0.0
                )

                payload = {
                    "generated_at": now_tz.isoformat(),
                    "summary": {
                        "users_total": total_users,
                        "users_active": active_users,
                        "users_new_30d": new_users_30d,
                        "active_sessions": active_sessions,
                        "ai_requests_30d": ai_requests_30d,
                        "ai_tokens_30d": ai_tokens_30d,
                        "ai_cost_30d": round(ai_cost_30d, 6),
                        "ai_success_rate_30d": round(success_rate, 2),
                        "bank_statements_total": bank_statements_total,
                        "bank_transactions_total": bank_transactions_total,
                        "contracts_total": contracts_total,
                        "cash_report_sessions_total": cash_report_sessions_total,
                        "cash_report_files_total": cash_report_files_total,
                        "cash_sessions_running": cash_sessions_running,
                    },
                    "charts": {
                        "daily_ai_usage": daily_series,
                        "provider_breakdown": provider_breakdown,
                        "module_totals": module_totals,
                    },
                }
                return JSONResponse(payload)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Unable to load admin dashboard stats")
            raise HTTPException(status_code=500, detail="Unable to load dashboard stats")


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

    _register_dashboard_stats_route(app, authentication_backend)

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
