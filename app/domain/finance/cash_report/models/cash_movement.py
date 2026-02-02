"""
Cash Movement model - represents a single transaction in the Movement sheet.
"""
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class MovementType(str, Enum):
    """Type of cash movement - Debit (Cash in) or Credit (Cash out)"""
    DEBIT = "debit"      # Nợ - Cash in
    CREDIT = "credit"    # Có - Cash out


class TransactionNature(str, Enum):
    """Nature/Category of transaction based on key payment mapping"""
    # Cash In (Receipt)
    RECEIPT_FROM_TENANTS = "Receipt from tenants"
    REFUND_LAND_DEAL_DEPOSIT = "Refund land/deal deposit payment"
    CASH_FROM_WP = "Cash from WP"
    INTERNAL_CONTRIBUTION = "Internal Contribution"
    INTERNAL_TRANSFER = "Internal transfer"
    OTHER_RECEIPTS = "Other receipts"
    CONTRIBUTION = "Contribution"
    LOAN_RECEIPTS = "Loan receipts"
    CORPORATE_LOAN_DRAWDOWN = "Corporate Loan drawdown"
    VAT_REFUND = "VAT refund"
    CASH_RECEIVED_FROM_ACQUISITION = "Cash received from acquisition"
    LOAN_DRAWDOWN = "Loan drawdown"
    DIVIDEND_RECEIPT_INSIDE_GROUP = "Dividend receipt (inside group)"
    MANAGEMENT_FEE_FROM_SUBSIDIARIES = "Management fee from subsidiaries"
    REFINANCING = "Refinancing"
    LOAN_REPAYMENT_RECEIPT = "Loan repayment"

    # Cash Out (Payment)
    OPERATING_EXPENSE = "Operating expense"
    CONSTRUCTION_EXPENSE = "Construction expense"
    LAND_ACQUISITION = "Land acquisition"
    DEAL_PAYMENT = "Deal payment"
    INTERNAL_TRANSFER_OUT = "Internal transfer"
    LOAN_REPAYMENT = "Loan repayment"
    LOAN_INTEREST = "Loan interest"
    PAYMENT_FOR_ACQUISITION = "Payment for acquisition"
    DIVIDEND_PAID_INSIDE_GROUP = "Dividend paid (inside group)"

    # Special
    INTERNAL_TRANSFER_IN = "Internal transfer in"
    INTERNAL_TRANSFER_OUT_EXPLICIT = "Internal transfer out"
    INTERNAL_CONTRIBUTION_IN = "Internal contribution in"
    INTERNAL_CONTRIBUTION_OUT = "Internal contribution out"

    UNKNOWN = "Unknown"


class CashMovement(BaseModel):
    """
    Represents a single cash movement/transaction.
    Maps to a row in the Movement sheet.
    """
    # Source information
    source: str = Field(description="Source of data: NS (NetSuite), Manual, etc.")
    bank_code: str = Field(description="Bank code: BIDV, VTB, VCB, etc.")
    account_number: str = Field(description="Bank account number")

    # Transaction details
    transaction_date: date = Field(description="Transaction date")
    description: str = Field(description="Bank description/memo")

    # Amount - only one should be filled
    debit_amount: Optional[Decimal] = Field(default=None, description="Debit amount (Cash in - Nợ)")
    credit_amount: Optional[Decimal] = Field(default=None, description="Credit amount (Cash out - Có)")

    # Classification
    nature: Optional[TransactionNature] = Field(default=None, description="Transaction nature/category")
    key_payment: Optional[str] = Field(default=None, description="Key payment type")

    # Entity information
    entity: Optional[str] = Field(default=None, description="Entity name (e.g., BWID BINH DUONG)")
    grouping: Optional[str] = Field(default=None, description="Grouping: BWID JSC, VC3, Subsidiaries")

    # Additional metadata
    currency: str = Field(default="VND", description="Currency: VND or USD")
    account_type: str = Field(default="Current Account", description="Account type")
    period: Optional[str] = Field(default=None, description="Report period (e.g., W1-2Oct25)")

    # For saving account transactions
    related_saving_account: Optional[str] = Field(default=None, description="Related saving account number")
    is_counter_entry: bool = Field(default=False, description="Is this a counter entry for saving account")

    @field_validator('transaction_date', mode='before')
    @classmethod
    def parse_date(cls, v):
        if v is None:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            # Try multiple formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%Y %H:%M:%S']:
                try:
                    return datetime.strptime(v, fmt).date()
                except ValueError:
                    continue
        return v

    @property
    def net_amount(self) -> Decimal:
        """Calculate net amount (positive for cash in, negative for cash out)"""
        debit = self.debit_amount or Decimal(0)
        credit = self.credit_amount or Decimal(0)
        return debit - credit

    @property
    def movement_type(self) -> MovementType:
        """Determine if this is a debit (cash in) or credit (cash out)"""
        if self.debit_amount and self.debit_amount > 0:
            return MovementType.DEBIT
        return MovementType.CREDIT

    @property
    def amount(self) -> Decimal:
        """Get the absolute amount"""
        return self.debit_amount or self.credit_amount or Decimal(0)

    def is_internal_transfer(self) -> bool:
        """Check if this is an internal transfer transaction"""
        if self.nature:
            return "internal transfer" in self.nature.value.lower()
        return False

    def is_saving_related(self) -> bool:
        """Check if this transaction is related to saving accounts"""
        keywords = ['tiền gửi', 'tien gui', 'saving', 'deposit', 'hdtk', 'hđtk']
        desc_lower = self.description.lower()
        return any(kw in desc_lower for kw in keywords)

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat() if v else None,
        }
