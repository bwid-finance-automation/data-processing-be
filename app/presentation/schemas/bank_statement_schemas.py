"""Pydantic schemas for bank statement parsing API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import date, datetime


class BankTransactionResponse(BaseModel):
    """Response schema for a single transaction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bank_name: str
    acc_no: str
    debit: Optional[float] = None
    credit: Optional[float] = None
    date: Optional[str] = None  # Changed to str to avoid Pydantic validation issues
    description: str = ""
    currency: str = "VND"
    transaction_id: str = ""
    beneficiary_bank: str = ""
    beneficiary_acc_no: str = ""
    beneficiary_acc_name: str = ""


class BankBalanceResponse(BaseModel):
    """Response schema for balance information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bank_name: str
    acc_no: str
    currency: str = "VND"
    opening_balance: float = 0.0
    closing_balance: float = 0.0


class BankStatementResponse(BaseModel):
    """Response schema for a single bank statement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bank_name: str
    file_name: str
    balance: Optional[BankBalanceResponse] = None
    transactions: List[BankTransactionResponse] = Field(default_factory=list)
    transaction_count: int = Field(0, description="Number of transactions parsed")


class ParseBankStatementsResponse(BaseModel):
    """Response schema for batch bank statement parsing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "statements": [
                    {
                        "bank_name": "ACB",
                        "file_name": "ACB_Jan2025.xlsx",
                        "balance": {
                            "bank_name": "ACB",
                            "acc_no": "123456789",
                            "currency": "VND",
                            "opening_balance": 10000000.0,
                            "closing_balance": 15000000.0
                        },
                        "transactions": [],
                        "transaction_count": 25
                    }
                ],
                "summary": {
                    "total_files": 5,
                    "successful": 4,
                    "failed": 1,
                    "failed_files": [
                        {"file_name": "unknown_bank.xlsx", "error": "Bank not recognized"}
                    ],
                    "total_transactions": 125,
                    "total_balances": 4
                },
                "download_url": "/api/finance/bank-statements/download/abc123"
            }
        }
    )

    statements: List[BankStatementResponse] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    download_url: Optional[str] = Field(None, description="URL to download Excel output")


class SupportedBanksResponse(BaseModel):
    """Response schema for supported banks list."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "banks": ["ACB", "OCB", "Vietcombank"],
                "count": 3
            }
        }
    )

    banks: List[str] = Field(default_factory=list)
    count: int = 0
