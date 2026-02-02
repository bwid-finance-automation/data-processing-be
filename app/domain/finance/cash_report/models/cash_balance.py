"""
Cash Balance model - represents account balance in Cash Balance sheet.
"""
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, computed_field


class AccountType(str, Enum):
    """Type of bank account"""
    CURRENT_ACCOUNT = "Current Account"
    SAVING_ACCOUNT = "Saving Account"
    CAPITAL_ACCOUNT = "Capital Account"


class CashBalance(BaseModel):
    """
    Represents cash balance for a single bank account.
    Maps to a row in the Cash Balance sheet.
    """
    # Entity information
    entity: str = Field(description="Entity name (e.g., BWID BINH DUONG)")
    grouping: str = Field(default="Subsidiaries", description="Grouping: BWID JSC, VC3, Subsidiaries")

    # Bank information
    bank_branch: str = Field(description="Bank branch (e.g., VIETCOMBANK - BA DINH)")
    bank_name: str = Field(description="Short bank name (e.g., VCB)")
    account_number: str = Field(description="Bank account number")
    account_type: AccountType = Field(default=AccountType.CURRENT_ACCOUNT)
    currency: str = Field(default="VND")

    # Balance information
    opening_balance_vnd: Decimal = Field(default=Decimal(0), description="Opening balance in VND")
    opening_balance_usd: Decimal = Field(default=Decimal(0), description="Opening balance in USD")
    closing_balance_vnd: Decimal = Field(default=Decimal(0), description="Closing balance in VND")
    closing_balance_usd: Decimal = Field(default=Decimal(0), description="Closing balance in USD")

    # Calculated from movements
    total_debit_vnd: Decimal = Field(default=Decimal(0), description="Total debit (cash in) in VND")
    total_credit_vnd: Decimal = Field(default=Decimal(0), description="Total credit (cash out) in VND")
    total_debit_usd: Decimal = Field(default=Decimal(0), description="Total debit (cash in) in USD")
    total_credit_usd: Decimal = Field(default=Decimal(0), description="Total credit (cash out) in USD")

    # Reconciliation
    bank_statement_balance: Optional[Decimal] = Field(default=None, description="Balance from bank statement")
    is_reconciled: bool = Field(default=False, description="Whether balance is reconciled")
    reconciliation_diff: Optional[Decimal] = Field(default=None, description="Difference if not reconciled")

    @computed_field
    @property
    def calculated_closing_vnd(self) -> Decimal:
        """Calculate closing balance from opening + movements"""
        return self.opening_balance_vnd + self.total_debit_vnd - self.total_credit_vnd

    @computed_field
    @property
    def calculated_closing_usd(self) -> Decimal:
        """Calculate closing balance from opening + movements"""
        return self.opening_balance_usd + self.total_debit_usd - self.total_credit_usd

    @computed_field
    @property
    def net_movement_vnd(self) -> Decimal:
        """Net cash movement in VND"""
        return self.total_debit_vnd - self.total_credit_vnd

    @computed_field
    @property
    def net_movement_usd(self) -> Decimal:
        """Net cash movement in USD"""
        return self.total_debit_usd - self.total_credit_usd

    def reconcile(self, bank_balance: Decimal) -> bool:
        """
        Reconcile calculated balance with actual bank statement balance.
        Returns True if reconciled (difference is 0 or negligible).
        """
        self.bank_statement_balance = bank_balance

        if self.currency == "VND":
            diff = abs(self.calculated_closing_vnd - bank_balance)
        else:
            diff = abs(self.calculated_closing_usd - bank_balance)

        self.reconciliation_diff = diff
        # Allow small difference due to rounding (1 VND or 0.01 USD)
        tolerance = Decimal(1) if self.currency == "VND" else Decimal("0.01")
        self.is_reconciled = diff <= tolerance

        return self.is_reconciled

    def update_from_movements(self, movements: list) -> None:
        """Update totals from list of CashMovement objects"""
        self.total_debit_vnd = Decimal(0)
        self.total_credit_vnd = Decimal(0)
        self.total_debit_usd = Decimal(0)
        self.total_credit_usd = Decimal(0)

        for mov in movements:
            if mov.account_number != self.account_number:
                continue

            if mov.currency == "VND":
                self.total_debit_vnd += mov.debit_amount or Decimal(0)
                self.total_credit_vnd += mov.credit_amount or Decimal(0)
            else:
                self.total_debit_usd += mov.debit_amount or Decimal(0)
                self.total_credit_usd += mov.credit_amount or Decimal(0)

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
        }
