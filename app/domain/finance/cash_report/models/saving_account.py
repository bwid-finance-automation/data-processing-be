"""
Saving Account model - represents a term deposit account.
"""
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, computed_field


class SavingAccountStatus(str, Enum):
    """Status of saving account"""
    ACTIVE = "Active"
    MATURED = "Matured"
    CLOSED = "Closed"
    AUTO_ROLLED = "Auto-rolled"


class SavingAccount(BaseModel):
    """
    Represents a saving/term deposit account.
    Maps to a row in the Saving Account sheet.
    """
    # Account identification
    account_number: str = Field(description="Saving account number")
    entity_code: str = Field(description="Entity code (e.g., WJ2, SP5A)")
    entity_name: str = Field(description="Entity name")

    # Bank information
    bank_branch: str = Field(description="Bank branch (e.g., VIETINBANK - TIEN SON)")
    bank_name: str = Field(description="Short bank name")

    # Account details
    currency: str = Field(default="VND")
    grouping: str = Field(default="Subsidiaries", description="BWID JSC, VC3, Subsidiaries")

    # Term information
    opening_date: date = Field(description="Date account was opened")
    maturity_date: date = Field(description="Maturity date")
    term_months: int = Field(description="Term in months (1M, 3M, 6M, 12M)")
    interest_rate: Decimal = Field(description="Annual interest rate (%)")
    previous_interest_rate: Optional[Decimal] = Field(default=None, description="Previous interest rate before auto-roll")

    # Balance
    principal_amount: Decimal = Field(description="Principal/deposit amount")
    accrued_interest: Decimal = Field(default=Decimal(0), description="Accrued interest")
    current_balance: Decimal = Field(default=Decimal(0), description="Current balance including interest")

    # Status
    status: SavingAccountStatus = Field(default=SavingAccountStatus.ACTIVE)
    is_auto_roll: bool = Field(default=False, description="Whether account auto-rolls on maturity")

    # Linked current account
    linked_current_account: Optional[str] = Field(default=None, description="Current account for transfers")

    # Review flag for newly detected accounts with incomplete data
    needs_review: bool = Field(default=False, description="Account needs manual review (missing data)")
    review_notes: Optional[str] = Field(default=None, description="Notes about what needs review")

    @computed_field
    @property
    def days_to_maturity(self) -> int:
        """Calculate days until maturity"""
        today = date.today()
        if self.maturity_date < today:
            return 0
        return (self.maturity_date - today).days

    @computed_field
    @property
    def is_matured(self) -> bool:
        """Check if account has matured"""
        return self.maturity_date <= date.today()

    @computed_field
    @property
    def total_balance(self) -> Decimal:
        """Total balance = principal + accrued interest"""
        return self.principal_amount + self.accrued_interest

    @computed_field
    @property
    def interest_rate_drop(self) -> Decimal:
        """Calculate interest rate drop from previous rate (positive = drop)"""
        if self.previous_interest_rate is None:
            return Decimal(0)
        return self.previous_interest_rate - self.interest_rate

    @computed_field
    @property
    def has_significant_rate_drop(self) -> bool:
        """Check if interest rate dropped by ≥2%"""
        return self.interest_rate_drop >= Decimal(2)

    def calculate_interest(self, as_of_date: Optional[date] = None) -> Decimal:
        """Calculate accrued interest up to a given date"""
        if as_of_date is None:
            as_of_date = date.today()

        if as_of_date < self.opening_date:
            return Decimal(0)

        # Use maturity date if as_of_date is after maturity
        end_date = min(as_of_date, self.maturity_date)
        days = (end_date - self.opening_date).days

        # Simple interest calculation: Principal * Rate * Days / 365
        interest = self.principal_amount * (self.interest_rate / 100) * Decimal(days) / Decimal(365)
        return interest.quantize(Decimal('0.01'))

    def close(self, closing_date: date) -> Decimal:
        """
        Close the saving account and return total amount (principal + interest).
        """
        self.accrued_interest = self.calculate_interest(closing_date)
        self.current_balance = Decimal(0)
        self.status = SavingAccountStatus.CLOSED
        return self.principal_amount + self.accrued_interest

    def partial_withdraw(self, amount: Decimal, withdraw_date: date) -> Decimal:
        """
        Partial withdrawal from saving account.
        Returns the withdrawn amount.
        Note: This may affect interest rate (usually reverts to demand rate).
        """
        if amount > self.principal_amount:
            raise ValueError("Withdrawal amount exceeds principal")

        self.principal_amount -= amount
        # Recalculate interest (usually at lower demand rate)
        # For simplicity, we just update the balance
        self.current_balance = self.principal_amount

        if self.principal_amount == Decimal(0):
            self.status = SavingAccountStatus.CLOSED

        return amount

    def auto_roll(self, new_maturity_date: date, new_interest_rate: Optional[Decimal] = None) -> None:
        """
        Auto-roll the saving account on maturity.
        Interest is added to principal.
        """
        # Add accrued interest to principal
        self.accrued_interest = self.calculate_interest(self.maturity_date)
        self.principal_amount += self.accrued_interest
        self.accrued_interest = Decimal(0)

        # Update term
        self.opening_date = self.maturity_date
        self.maturity_date = new_maturity_date

        if new_interest_rate is not None:
            self.interest_rate = new_interest_rate

        self.status = SavingAccountStatus.AUTO_ROLLED

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat() if v else None,
        }
