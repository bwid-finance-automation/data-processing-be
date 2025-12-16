"""Database models package - Import all models here for Alembic discovery."""

from app.infrastructure.database.models.bank_statement import (
    BankStatementModel,
    BankTransactionModel,
    BankBalanceModel,
)
from app.infrastructure.database.models.contract import (
    ContractModel,
    ContractPartyModel,
    ContractRatePeriodModel,
    ContractUnitModel,
)
from app.infrastructure.database.models.gla import (
    GLARecordModel,
    GLAProjectModel,
    GLATenantModel,
)
from app.infrastructure.database.models.analysis_session import (
    AnalysisSessionModel,
    AnalysisResultModel,
)
from app.infrastructure.database.models.file_upload import FileUploadModel

__all__ = [
    # Bank Statement models
    "BankStatementModel",
    "BankTransactionModel",
    "BankBalanceModel",
    # Contract models
    "ContractModel",
    "ContractPartyModel",
    "ContractRatePeriodModel",
    "ContractUnitModel",
    # GLA models
    "GLARecordModel",
    "GLAProjectModel",
    "GLATenantModel",
    # Analysis models
    "AnalysisSessionModel",
    "AnalysisResultModel",
    # File models
    "FileUploadModel",
]
