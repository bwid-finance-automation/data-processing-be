"""Repository implementations for database access."""

from app.infrastructure.persistence.repositories.base import BaseRepository
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

__all__ = [
    # Base
    "BaseRepository",
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
]
