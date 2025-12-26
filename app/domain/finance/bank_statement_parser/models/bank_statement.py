"""Domain model for bank statement."""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field

from .bank_transaction import BankTransaction


class BankBalance(BaseModel):
    """Represents bank account balance information."""

    bank_name: str = Field(..., description="Bank name")
    acc_no: str = Field(..., description="Account number")
    currency: str = Field("VND", description="Currency")
    opening_balance: float = Field(0.0, description="Opening balance")
    closing_balance: float = Field(0.0, description="Closing balance")
    statement_date: Optional[date] = Field(None, description="Statement period end date (from file)")

    class Config:
        json_schema_extra = {
            "example": {
                "bank_name": "ACB",
                "acc_no": "123456789",
                "currency": "VND",
                "opening_balance": 10000000.0,
                "closing_balance": 15000000.0
            }
        }


class BankStatement(BaseModel):
    """Represents a complete bank statement with transactions and balances."""

    bank_name: str = Field(..., description="Detected bank name")
    file_name: str = Field(..., description="Original file name")
    balance: Optional[BankBalance] = Field(None, description="Balance information")
    transactions: List[BankTransaction] = Field(default_factory=list, description="List of transactions")

    class Config:
        json_schema_extra = {
            "example": {
                "bank_name": "ACB",
                "file_name": "ACB_statement_Jan2025.xlsx",
                "balance": {
                    "bank_name": "ACB",
                    "acc_no": "123456789",
                    "currency": "VND",
                    "opening_balance": 10000000.0,
                    "closing_balance": 15000000.0
                },
                "transactions": []
            }
        }
