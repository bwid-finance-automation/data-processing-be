"""Repository implementations for database access."""

from app.infrastructure.persistence.repositories.base import BaseRepository
from app.infrastructure.persistence.repositories.project_repository import (
    ProjectRepository,
    ProjectCaseRepository,
)
from app.infrastructure.persistence.repositories.bank_statement_repository import (
    BankStatementRepository,
    BankTransactionRepository,
    BankBalanceRepository,
)
from app.infrastructure.persistence.repositories.contract_repository import (
    ContractRepository,
    ContractPartyRepository,
    ContractRatePeriodRepository,
    ContractUnitRepository,
)
from app.infrastructure.persistence.repositories.analysis_session_repository import (
    AnalysisSessionRepository,
    AnalysisResultRepository,
)
from app.infrastructure.persistence.repositories.file_upload_repository import (
    FileUploadRepository,
)
from app.infrastructure.persistence.repositories.ai_usage_repository import (
    AIUsageRepository,
)
from app.infrastructure.persistence.repositories.user_repository import (
    UserRepository,
    UserSessionRepository,
)
from app.infrastructure.persistence.repositories.system_settings_repository import (
    SystemSettingsRepository,
)

__all__ = [
    # Base
    "BaseRepository",
    # Project
    "ProjectRepository",
    "ProjectCaseRepository",
    # Bank Statement
    "BankStatementRepository",
    "BankTransactionRepository",
    "BankBalanceRepository",
    # Contract
    "ContractRepository",
    "ContractPartyRepository",
    "ContractRatePeriodRepository",
    "ContractUnitRepository",
    # Analysis
    "AnalysisSessionRepository",
    "AnalysisResultRepository",
    # File
    "FileUploadRepository",
    # AI Usage
    "AIUsageRepository",
    # User/Auth
    "UserRepository",
    "UserSessionRepository",
    # System Settings
    "SystemSettingsRepository",
]
