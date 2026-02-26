"""SQLAdmin ModelAdmin views for all existing database models.

Each view configures column visibility, permissions (can_create, can_edit, can_delete),
search, sort, column filters, and display customization for the admin dashboard.
"""

from datetime import datetime, timezone

from sqlalchemy import update
from starlette.requests import Request
from starlette.responses import RedirectResponse
from sqladmin import ModelView, action
from sqladmin.filters import (
    BooleanFilter,
    AllUniqueStringValuesFilter,
    OperationColumnFilter,
)

from app.infrastructure.database.models.user import UserModel, UserSessionModel
from app.infrastructure.database.models.system_settings import SystemSettingsModel
from app.infrastructure.database.models.bank_statement import (
    BankStatementModel,
    BankTransactionModel,
    BankBalanceModel,
)
from app.infrastructure.database.models.contract import ContractModel
from app.infrastructure.database.models.ai_usage import AIUsageModel
from app.infrastructure.database.models.cash_report_session import (
    CashReportSessionModel,
    CashReportUploadedFileModel,
)


# ==================== 1. User Management ====================

class UserAdmin(ModelView, model=UserModel):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-user"
    category = "User Management"

    can_create = False
    can_delete = False
    can_edit = True
    can_export = True

    column_list = [
        UserModel.id,
        UserModel.username,
        UserModel.email,
        UserModel.full_name,
        UserModel.role,
        UserModel.is_active,
        UserModel.login_count,
        UserModel.last_login_at,
        UserModel.last_login_ip,
        UserModel.created_at,
    ]

    column_details_exclude_list = [
        UserModel.password_hash,
    ]

    form_columns = [
        "role",
        "is_active",
    ]

    column_searchable_list = [
        UserModel.username,
        UserModel.email,
        UserModel.full_name,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(UserModel.role, title="Role"),
        BooleanFilter(UserModel.is_active, title="Active"),
        BooleanFilter(UserModel.is_deleted, title="Deleted"),
        OperationColumnFilter(UserModel.login_count, title="Login Count"),
    ]

    column_sortable_list = [
        UserModel.id,
        UserModel.username,
        UserModel.email,
        UserModel.role,
        UserModel.is_active,
        UserModel.login_count,
        UserModel.last_login_at,
        UserModel.created_at,
    ]

    column_default_sort = [(UserModel.id, True)]

    column_labels = {
        UserModel.last_login_at: "Last Login",
        UserModel.last_login_ip: "Last IP",
        UserModel.login_count: "Logins",
        UserModel.is_active: "Active",
        UserModel.created_at: "Registered",
    }

    @action(
        name="revoke_user_sessions",
        label="Revoke Sessions",
        confirmation_message="Revoke all active sessions for selected users?",
    )
    async def revoke_user_sessions(self, request: Request):
        """Revoke all active sessions for selected users."""
        params = request.query_params.get("pks", "")
        user_ids = [int(pk) for pk in params.split(",") if pk.isdigit()]
        if not user_ids:
            return RedirectResponse(
                request.headers.get(
                    "referer",
                    str(request.url_for("admin:list", identity=self.identity)),
                ),
                status_code=302,
            )

        now = datetime.now(timezone.utc)

        if self.is_async:
            async with self.session_maker() as session:
                await session.execute(
                    update(UserSessionModel)
                    .where(
                        UserSessionModel.user_id.in_(user_ids),
                        UserSessionModel.is_revoked.is_(False),
                    )
                    .values(is_revoked=True, revoked_at=now)
                )
                await session.commit()
        else:
            with self.session_maker() as session:
                session.execute(
                    update(UserSessionModel)
                    .where(
                        UserSessionModel.user_id.in_(user_ids),
                        UserSessionModel.is_revoked.is_(False),
                    )
                    .values(is_revoked=True, revoked_at=now)
                )
                session.commit()

        return RedirectResponse(
            request.headers.get(
                "referer",
                str(request.url_for("admin:list", identity=self.identity)),
            ),
            status_code=302,
        )


class UserSessionAdmin(ModelView, model=UserSessionModel):
    name = "User Session"
    name_plural = "User Sessions"
    icon = "fa-solid fa-key"
    category = "User Management"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True

    column_list = [
        UserSessionModel.id,
        UserSessionModel.user_id,
        UserSessionModel.ip_address,
        UserSessionModel.user_agent,
        UserSessionModel.is_revoked,
        UserSessionModel.revoked_at,
        UserSessionModel.expires_at,
        UserSessionModel.created_at,
    ]

    column_searchable_list = [
        UserSessionModel.ip_address,
    ]

    column_filters = [
        BooleanFilter(UserSessionModel.is_revoked, title="Revoked"),
        OperationColumnFilter(UserSessionModel.user_id, title="User ID"),
    ]

    column_sortable_list = [
        UserSessionModel.id,
        UserSessionModel.user_id,
        UserSessionModel.is_revoked,
        UserSessionModel.revoked_at,
        UserSessionModel.expires_at,
        UserSessionModel.created_at,
    ]

    column_default_sort = [(UserSessionModel.id, True)]

    column_labels = {
        UserSessionModel.is_revoked: "Revoked",
        UserSessionModel.revoked_at: "Revoked At",
        UserSessionModel.expires_at: "Expires",
        UserSessionModel.user_agent: "Browser/Agent",
    }

    @action(
        name="revoke_sessions",
        label="Revoke Selected Sessions",
        confirmation_message="Revoke selected sessions now?",
    )
    async def revoke_sessions(self, request: Request):
        """Revoke selected sessions from list/details action."""
        params = request.query_params.get("pks", "")
        session_ids = [int(pk) for pk in params.split(",") if pk.isdigit()]
        if not session_ids:
            return RedirectResponse(
                request.headers.get(
                    "referer",
                    str(request.url_for("admin:list", identity=self.identity)),
                ),
                status_code=302,
            )

        now = datetime.now(timezone.utc)
        if self.is_async:
            async with self.session_maker() as session:
                await session.execute(
                    update(UserSessionModel)
                    .where(UserSessionModel.id.in_(session_ids))
                    .values(is_revoked=True, revoked_at=now)
                )
                await session.commit()
        else:
            with self.session_maker() as session:
                session.execute(
                    update(UserSessionModel)
                    .where(UserSessionModel.id.in_(session_ids))
                    .values(is_revoked=True, revoked_at=now)
                )
                session.commit()

        return RedirectResponse(
            request.headers.get(
                "referer",
                str(request.url_for("admin:list", identity=self.identity)),
            ),
            status_code=302,
        )


# ==================== 2. Feature Toggles ====================

class SystemSettingsAdmin(ModelView, model=SystemSettingsModel):
    name = "Raw Feature Settings"
    name_plural = "Raw Feature Settings"
    icon = "fa-solid fa-code"
    category = "Feature Toggles"

    can_create = False
    can_delete = False
    can_edit = False
    can_export = True

    def is_visible(self, request: Request) -> bool:
        """Hide raw JSON settings from sidebar to keep business UI clean."""
        return False

    column_list = [
        SystemSettingsModel.id,
        SystemSettingsModel.key,
        SystemSettingsModel.value,
        SystemSettingsModel.description,
        SystemSettingsModel.updated_at,
    ]

    column_searchable_list = [
        SystemSettingsModel.key,
        SystemSettingsModel.description,
    ]

    column_filters = [
        OperationColumnFilter(SystemSettingsModel.key, title="Feature Key"),
    ]

    column_sortable_list = [
        SystemSettingsModel.id,
        SystemSettingsModel.key,
        SystemSettingsModel.updated_at,
    ]

    form_columns = [
        "value",
        "description",
    ]

    form_widget_args = {
        "key": {"readonly": True},
    }

    column_labels = {
        SystemSettingsModel.key: "Feature Key",
        SystemSettingsModel.value: "Config (JSON)",
        SystemSettingsModel.updated_at: "Last Modified",
    }

    form_args = {
        "key": {"description": "e.g. feature:bank_statement_ocr, feature:contract_ocr"},
        "value": {"description": 'JSON object, e.g. {"enabled": true, "disabled_message": "Under maintenance"}'},
        "description": {"description": "Human-readable description of this setting"},
    }


# ==================== 3. AI Usage Monitoring ====================

class AIUsageAdmin(ModelView, model=AIUsageModel):
    name = "AI Usage Log"
    name_plural = "AI Usage Logs"
    icon = "fa-solid fa-robot"
    category = "AI Usage Monitoring"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True
    page_size = 50

    column_list = [
        AIUsageModel.id,
        AIUsageModel.provider,
        AIUsageModel.model_name,
        AIUsageModel.task_type,
        AIUsageModel.file_name,
        AIUsageModel.input_tokens,
        AIUsageModel.output_tokens,
        AIUsageModel.total_tokens,
        AIUsageModel.estimated_cost_usd,
        AIUsageModel.processing_time_ms,
        AIUsageModel.success,
        AIUsageModel.requested_at,
    ]

    column_searchable_list = [
        AIUsageModel.provider,
        AIUsageModel.model_name,
        AIUsageModel.task_type,
        AIUsageModel.file_name,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(AIUsageModel.provider, title="Provider"),
        AllUniqueStringValuesFilter(AIUsageModel.model_name, title="Model"),
        AllUniqueStringValuesFilter(AIUsageModel.task_type, title="Task Type"),
        BooleanFilter(AIUsageModel.success, title="Success"),
        OperationColumnFilter(AIUsageModel.requested_at, title="Requested At"),
        OperationColumnFilter(AIUsageModel.total_tokens, title="Total Tokens"),
        OperationColumnFilter(AIUsageModel.estimated_cost_usd, title="Cost (USD)"),
    ]

    column_sortable_list = [
        AIUsageModel.id,
        AIUsageModel.provider,
        AIUsageModel.model_name,
        AIUsageModel.total_tokens,
        AIUsageModel.estimated_cost_usd,
        AIUsageModel.processing_time_ms,
        AIUsageModel.success,
        AIUsageModel.requested_at,
    ]

    column_default_sort = [(AIUsageModel.id, True)]

    column_labels = {
        AIUsageModel.provider: "Provider",
        AIUsageModel.model_name: "Model",
        AIUsageModel.task_type: "Task",
        AIUsageModel.input_tokens: "In Tokens",
        AIUsageModel.output_tokens: "Out Tokens",
        AIUsageModel.total_tokens: "Total Tokens",
        AIUsageModel.estimated_cost_usd: "Cost (USD)",
        AIUsageModel.processing_time_ms: "Time (ms)",
        AIUsageModel.success: "OK",
        AIUsageModel.requested_at: "Date",
    }


# ==================== 5. Bank Statement Monitoring ====================

class BankStatementAdmin(ModelView, model=BankStatementModel):
    name = "Bank Statement"
    name_plural = "Bank Statements"
    icon = "fa-solid fa-file-invoice"
    category = "Bank Statement Monitoring"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True

    column_list = [
        BankStatementModel.id,
        BankStatementModel.bank_name,
        BankStatementModel.file_name,
        BankStatementModel.session_id,
        BankStatementModel.uploaded_at,
        BankStatementModel.processed_at,
    ]

    column_searchable_list = [
        BankStatementModel.bank_name,
        BankStatementModel.file_name,
        BankStatementModel.session_id,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(BankStatementModel.bank_name, title="Bank"),
    ]

    column_sortable_list = [
        BankStatementModel.id,
        BankStatementModel.bank_name,
        BankStatementModel.uploaded_at,
    ]

    column_default_sort = [(BankStatementModel.id, True)]

    column_labels = {
        BankStatementModel.bank_name: "Bank",
        BankStatementModel.file_name: "File",
        BankStatementModel.uploaded_at: "Uploaded",
        BankStatementModel.processed_at: "Processed",
    }


class BankTransactionAdmin(ModelView, model=BankTransactionModel):
    name = "Bank Transaction"
    name_plural = "Bank Transactions"
    icon = "fa-solid fa-money-bill-transfer"
    category = "Bank Statement Monitoring"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True
    page_size = 50

    column_list = [
        BankTransactionModel.id,
        BankTransactionModel.bank_name,
        BankTransactionModel.acc_no,
        BankTransactionModel.transaction_date,
        BankTransactionModel.description,
        BankTransactionModel.debit,
        BankTransactionModel.credit,
        BankTransactionModel.currency,
        BankTransactionModel.beneficiary_acc_name,
    ]

    column_searchable_list = [
        BankTransactionModel.acc_no,
        BankTransactionModel.description,
        BankTransactionModel.bank_name,
        BankTransactionModel.beneficiary_acc_name,
        BankTransactionModel.beneficiary_acc_no,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(BankTransactionModel.bank_name, title="Bank"),
        AllUniqueStringValuesFilter(BankTransactionModel.currency, title="Currency"),
        OperationColumnFilter(BankTransactionModel.acc_no, title="Account"),
        OperationColumnFilter(BankTransactionModel.statement_id, title="Statement ID"),
    ]

    column_sortable_list = [
        BankTransactionModel.id,
        BankTransactionModel.transaction_date,
        BankTransactionModel.bank_name,
        BankTransactionModel.debit,
        BankTransactionModel.credit,
    ]

    column_default_sort = [(BankTransactionModel.id, True)]

    column_labels = {
        BankTransactionModel.bank_name: "Bank",
        BankTransactionModel.acc_no: "Account",
        BankTransactionModel.transaction_date: "Date",
        BankTransactionModel.beneficiary_acc_name: "Beneficiary",
    }


class BankBalanceAdmin(ModelView, model=BankBalanceModel):
    name = "Bank Balance"
    name_plural = "Bank Balances"
    icon = "fa-solid fa-wallet"
    category = "Bank Statement Monitoring"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True

    column_list = [
        BankBalanceModel.id,
        BankBalanceModel.bank_name,
        BankBalanceModel.acc_no,
        BankBalanceModel.currency,
        BankBalanceModel.opening_balance,
        BankBalanceModel.closing_balance,
    ]

    column_searchable_list = [
        BankBalanceModel.acc_no,
        BankBalanceModel.bank_name,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(BankBalanceModel.bank_name, title="Bank"),
        AllUniqueStringValuesFilter(BankBalanceModel.currency, title="Currency"),
    ]

    column_sortable_list = [
        BankBalanceModel.id,
        BankBalanceModel.bank_name,
        BankBalanceModel.acc_no,
        BankBalanceModel.opening_balance,
        BankBalanceModel.closing_balance,
    ]

    column_labels = {
        BankBalanceModel.bank_name: "Bank",
        BankBalanceModel.acc_no: "Account",
        BankBalanceModel.opening_balance: "Opening",
        BankBalanceModel.closing_balance: "Closing",
    }


# ==================== 6. Cash Report Management ====================

class CashReportSessionAdmin(ModelView, model=CashReportSessionModel):
    name = "Cash Report Session"
    name_plural = "Cash Report Sessions"
    icon = "fa-solid fa-receipt"
    category = "Cash Report Management"

    can_create = False
    can_edit = False
    can_delete = True
    can_export = True

    column_list = [
        CashReportSessionModel.id,
        CashReportSessionModel.session_id,
        CashReportSessionModel.status,
        CashReportSessionModel.period_name,
        CashReportSessionModel.opening_date,
        CashReportSessionModel.ending_date,
        CashReportSessionModel.total_files_uploaded,
        CashReportSessionModel.total_transactions,
        CashReportSessionModel.user_id,
        CashReportSessionModel.created_at,
    ]

    column_searchable_list = [
        CashReportSessionModel.session_id,
        CashReportSessionModel.period_name,
    ]

    column_filters = [
        OperationColumnFilter(CashReportSessionModel.user_id, title="User ID"),
    ]

    column_sortable_list = [
        CashReportSessionModel.id,
        CashReportSessionModel.status,
        CashReportSessionModel.opening_date,
        CashReportSessionModel.ending_date,
        CashReportSessionModel.total_files_uploaded,
        CashReportSessionModel.total_transactions,
        CashReportSessionModel.created_at,
    ]

    column_default_sort = [(CashReportSessionModel.id, True)]

    column_labels = {
        CashReportSessionModel.session_id: "Session",
        CashReportSessionModel.period_name: "Period",
        CashReportSessionModel.opening_date: "From",
        CashReportSessionModel.ending_date: "To",
        CashReportSessionModel.total_files_uploaded: "Files",
        CashReportSessionModel.total_transactions: "Transactions",
    }


class CashReportUploadedFileAdmin(ModelView, model=CashReportUploadedFileModel):
    name = "Cash Report File"
    name_plural = "Cash Report Files"
    icon = "fa-solid fa-file-arrow-up"
    category = "Cash Report Management"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True

    column_list = [
        CashReportUploadedFileModel.id,
        CashReportUploadedFileModel.session_id,
        CashReportUploadedFileModel.original_filename,
        CashReportUploadedFileModel.file_size,
        CashReportUploadedFileModel.transactions_count,
        CashReportUploadedFileModel.transactions_added,
        CashReportUploadedFileModel.transactions_skipped,
        CashReportUploadedFileModel.processed_at,
        CashReportUploadedFileModel.error_message,
    ]

    column_searchable_list = [
        CashReportUploadedFileModel.original_filename,
        CashReportUploadedFileModel.error_message,
    ]

    column_filters = [
        OperationColumnFilter(CashReportUploadedFileModel.session_id, title="Session ID"),
    ]

    column_sortable_list = [
        CashReportUploadedFileModel.id,
        CashReportUploadedFileModel.session_id,
        CashReportUploadedFileModel.file_size,
        CashReportUploadedFileModel.transactions_count,
        CashReportUploadedFileModel.processed_at,
    ]

    column_default_sort = [(CashReportUploadedFileModel.id, True)]

    column_labels = {
        CashReportUploadedFileModel.original_filename: "Filename",
        CashReportUploadedFileModel.file_size: "Size",
        CashReportUploadedFileModel.transactions_count: "Total Txns",
        CashReportUploadedFileModel.transactions_added: "Added",
        CashReportUploadedFileModel.transactions_skipped: "Skipped",
        CashReportUploadedFileModel.error_message: "Error",
    }


# ==================== 7. Contract Monitoring ====================

class ContractAdmin(ModelView, model=ContractModel):
    name = "Contract"
    name_plural = "Contracts"
    icon = "fa-solid fa-file-contract"
    category = "Contract Monitoring"

    can_create = False
    can_edit = False
    can_delete = False
    can_export = True

    column_list = [
        ContractModel.id,
        ContractModel.contract_number,
        ContractModel.contract_title,
        ContractModel.tenant,
        ContractModel.customer_name,
        ContractModel.effective_date,
        ContractModel.expiration_date,
        ContractModel.status,
        ContractModel.created_at,
    ]

    column_searchable_list = [
        ContractModel.contract_number,
        ContractModel.contract_title,
        ContractModel.tenant,
        ContractModel.customer_name,
    ]

    column_filters = [
        AllUniqueStringValuesFilter(ContractModel.tenant, title="Tenant"),
        AllUniqueStringValuesFilter(ContractModel.status, title="Status"),
        OperationColumnFilter(ContractModel.customer_name, title="Customer"),
    ]

    column_sortable_list = [
        ContractModel.id,
        ContractModel.contract_number,
        ContractModel.tenant,
        ContractModel.effective_date,
        ContractModel.expiration_date,
        ContractModel.created_at,
    ]

    column_default_sort = [(ContractModel.id, True)]

    column_labels = {
        ContractModel.contract_number: "Contract #",
        ContractModel.contract_title: "Title",
        ContractModel.effective_date: "Effective",
        ContractModel.expiration_date: "Expires",
    }
