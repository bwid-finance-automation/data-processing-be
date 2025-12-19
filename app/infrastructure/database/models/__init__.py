"""Database models package - Import all models here for Alembic discovery."""

from app.infrastructure.database.models.project import (
    ProjectModel,
    ProjectCaseModel,
    CaseType,
)
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
from app.infrastructure.database.models.ai_usage import (
    AIUsageModel,
    AIProvider,
    AITaskType,
)

__all__ = [
    # Project models
    "ProjectModel",
    "ProjectCaseModel",
    "CaseType",
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
    # AI Usage models
    "AIUsageModel",
    "AIProvider",
    "AITaskType",
]
