"""Domain model for bank transaction."""

from datetime import date as date_type
from typing import Optional
from pydantic import BaseModel, Field


class BankTransaction(BaseModel):
    """Represents a single bank transaction (standardized format)."""

    bank_name: str = Field(..., description="Bank name (e.g., ACB, OCB)")
    acc_no: str = Field(..., description="Account number")
    debit: Optional[float] = Field(None, description="Debit amount (money out)")
    credit: Optional[float] = Field(None, description="Credit amount (money in)")
    date: Optional[date_type] = Field(None, description="Transaction date")
    description: str = Field("", description="Transaction description")
    currency: str = Field("VND", description="Currency (VND, USD, etc.)")
    transaction_id: str = Field("", description="Transaction ID")
    beneficiary_bank: str = Field("", description="Beneficiary bank name")
    beneficiary_acc_no: str = Field("", description="Beneficiary account number")
    beneficiary_acc_name: str = Field("", description="Beneficiary account name")

    class Config:
        json_schema_extra = {
            "example": {
                "bank_name": "ACB",
                "acc_no": "123456789",
                "debit": 1000000.0,
                "credit": None,
                "date": "2025-01-15",
                "description": "Transfer to supplier",
                "currency": "VND",
                "transaction_id": "TX001",
                "beneficiary_bank": "VCB",
                "beneficiary_acc_no": "987654321",
                "beneficiary_acc_name": "Supplier Co Ltd"
            }
        }
