"""
Cash Report model - the main report container.
"""
from datetime import date
from decimal import Decimal
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, computed_field

from .cash_movement import CashMovement
from .cash_balance import CashBalance
from .saving_account import SavingAccount


class EntitySummary(BaseModel):
    """Summary for a single entity grouping (BWID JSC, VC3, Subsidiaries)"""
    grouping: str = Field(description="Entity grouping name")

    # Cash in VND
    cash_in_vnd: Decimal = Field(default=Decimal(0), description="Cash in original VND")
    cash_in_vnd_equivalent: Decimal = Field(default=Decimal(0), description="VND cash in USDmn equivalent")

    # Cash in USD
    cash_in_usd: Decimal = Field(default=Decimal(0), description="Cash in original USD")

    # Total
    total_cash_usd_equivalent: Decimal = Field(default=Decimal(0), description="Total cash in USDmn equivalent")

    # Movement summary
    total_receipt: Decimal = Field(default=Decimal(0), description="Total cash in")
    total_payment: Decimal = Field(default=Decimal(0), description="Total cash out")
    net_movement: Decimal = Field(default=Decimal(0), description="Net cash movement")

    # Opening/Closing
    opening_balance: Decimal = Field(default=Decimal(0))
    closing_balance: Decimal = Field(default=Decimal(0))


class BankSummary(BaseModel):
    """Summary for a single bank across all entities"""
    bank_name: str
    bank_branch: str

    bwid_jsc_amount: Decimal = Field(default=Decimal(0))
    vc3_amount: Decimal = Field(default=Decimal(0))
    subsidiaries_amount: Decimal = Field(default=Decimal(0))
    total_amount: Decimal = Field(default=Decimal(0))
    percentage: Decimal = Field(default=Decimal(0))


class CashMovementSummary(BaseModel):
    """Summary of cash movements by category"""
    # Cash In categories
    ordinary_receipt: Decimal = Field(default=Decimal(0))
    receipt_from_tenants: Decimal = Field(default=Decimal(0))
    refund_land_deal_deposit: Decimal = Field(default=Decimal(0))
    refinancing: Decimal = Field(default=Decimal(0))
    loan_drawdown: Decimal = Field(default=Decimal(0))
    vat_refund: Decimal = Field(default=Decimal(0))
    corporate_loan_drawdown: Decimal = Field(default=Decimal(0))
    loan_receipts: Decimal = Field(default=Decimal(0))
    contribution: Decimal = Field(default=Decimal(0))
    other_receipts: Decimal = Field(default=Decimal(0))

    # Internal transactions - In
    internal_transfer_in: Decimal = Field(default=Decimal(0))
    internal_contribution_in: Decimal = Field(default=Decimal(0))
    dividend_receipt_inside_group: Decimal = Field(default=Decimal(0))
    cash_received_from_acquisition: Decimal = Field(default=Decimal(0))

    total_cash_in: Decimal = Field(default=Decimal(0))

    # Cash Out categories
    ordinary_payment: Decimal = Field(default=Decimal(0))
    land_acquisition: Decimal = Field(default=Decimal(0))
    deal_payment: Decimal = Field(default=Decimal(0))
    construction_expense: Decimal = Field(default=Decimal(0))
    operating_expense: Decimal = Field(default=Decimal(0))
    loan_repayment: Decimal = Field(default=Decimal(0))
    loan_interest: Decimal = Field(default=Decimal(0))

    # Internal transactions - Out
    internal_transfer_out: Decimal = Field(default=Decimal(0))
    internal_contribution_out: Decimal = Field(default=Decimal(0))
    dividend_paid_inside_group: Decimal = Field(default=Decimal(0))
    payment_for_acquisition: Decimal = Field(default=Decimal(0))

    total_cash_out: Decimal = Field(default=Decimal(0))

    @computed_field
    @property
    def net_cash_flow(self) -> Decimal:
        return self.total_cash_in - self.total_cash_out


class CashReportSummary(BaseModel):
    """Overall summary of the cash report"""
    # Report metadata
    report_date: date = Field(description="Report date (ending date)")
    period: str = Field(description="Report period (e.g., W1-2Oct25)")
    opening_date: date
    ending_date: date
    fx_rate: Decimal = Field(description="VND/USD exchange rate")

    # Total cash position
    total_cash_usd_equivalent: Decimal = Field(default=Decimal(0))
    total_cash_vnd: Decimal = Field(default=Decimal(0))
    total_cash_usd: Decimal = Field(default=Decimal(0))

    # Change from prior period
    prior_period_cash: Decimal = Field(default=Decimal(0))
    cash_change_amount: Decimal = Field(default=Decimal(0))
    cash_change_percentage: Decimal = Field(default=Decimal(0))

    # Entity summaries
    bwid_jsc: EntitySummary = Field(default_factory=lambda: EntitySummary(grouping="BWID JSC"))
    vc3: EntitySummary = Field(default_factory=lambda: EntitySummary(grouping="VC3"))
    subsidiaries: EntitySummary = Field(default_factory=lambda: EntitySummary(grouping="Subsidiaries"))

    # Bank summaries
    bank_summaries: List[BankSummary] = Field(default_factory=list)

    # Movement summary
    movement_summary: CashMovementSummary = Field(default_factory=CashMovementSummary)

    # Validation
    internal_transfer_net: Decimal = Field(default=Decimal(0), description="Should be 0")
    internal_contribution_net: Decimal = Field(default=Decimal(0), description="Should be 0")
    is_balanced: bool = Field(default=False, description="All validations pass")


class CashReport(BaseModel):
    """
    Complete Cash Report containing all data.
    This is the main output of the cash report generation process.
    """
    # Metadata
    report_id: Optional[str] = Field(default=None, description="Unique report identifier")
    created_at: Optional[date] = Field(default=None)
    created_by: Optional[str] = Field(default=None)

    # Summary
    summary: CashReportSummary

    # Detail data
    movements: List[CashMovement] = Field(default_factory=list, description="All cash movements")
    cash_balances: List[CashBalance] = Field(default_factory=list, description="All account balances")
    saving_accounts: List[SavingAccount] = Field(default_factory=list, description="All saving accounts")

    # Prior period data (for comparison)
    prior_cash_balances: List[CashBalance] = Field(default_factory=list)

    # Validation results
    reconciliation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def total_movements(self) -> int:
        return len(self.movements)

    @computed_field
    @property
    def total_accounts(self) -> int:
        return len(self.cash_balances)

    @computed_field
    @property
    def total_saving_accounts(self) -> int:
        return len(self.saving_accounts)

    def get_movements_by_account(self, account_number: str) -> List[CashMovement]:
        """Get all movements for a specific account"""
        return [m for m in self.movements if m.account_number == account_number]

    def get_movements_by_entity(self, grouping: str) -> List[CashMovement]:
        """Get all movements for an entity grouping"""
        return [m for m in self.movements if m.grouping == grouping]

    def get_movements_by_nature(self, nature: str) -> List[CashMovement]:
        """Get all movements by nature/category"""
        return [m for m in self.movements if m.nature and m.nature.value == nature]

    def get_balance_by_account(self, account_number: str) -> Optional[CashBalance]:
        """Get balance for a specific account"""
        for bal in self.cash_balances:
            if bal.account_number == account_number:
                return bal
        return None

    def get_saving_accounts_by_status(self, status: str) -> List[SavingAccount]:
        """Get saving accounts by status"""
        return [sa for sa in self.saving_accounts if sa.status.value == status]

    def validate(self) -> bool:
        """
        Validate the report:
        1. Internal transfer in/out should net to 0
        2. Internal contribution in/out should net to 0
        3. All accounts should be reconciled
        """
        self.reconciliation_errors = []
        self.warnings = []

        # Check internal transfer balance
        summary = self.summary.movement_summary
        if summary.internal_transfer_in != summary.internal_transfer_out:
            diff = summary.internal_transfer_in - summary.internal_transfer_out
            self.reconciliation_errors.append(
                f"Internal transfer not balanced: in={summary.internal_transfer_in}, "
                f"out={summary.internal_transfer_out}, diff={diff}"
            )

        # Check internal contribution balance
        if summary.internal_contribution_in != summary.internal_contribution_out:
            diff = summary.internal_contribution_in - summary.internal_contribution_out
            self.reconciliation_errors.append(
                f"Internal contribution not balanced: in={summary.internal_contribution_in}, "
                f"out={summary.internal_contribution_out}, diff={diff}"
            )

        # Check account reconciliation
        for balance in self.cash_balances:
            if not balance.is_reconciled and balance.bank_statement_balance is not None:
                self.warnings.append(
                    f"Account {balance.account_number} not reconciled: "
                    f"calculated={balance.calculated_closing_vnd}, "
                    f"bank={balance.bank_statement_balance}, "
                    f"diff={balance.reconciliation_diff}"
                )

        self.summary.is_balanced = len(self.reconciliation_errors) == 0
        return self.summary.is_balanced

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v),
            date: lambda v: v.isoformat() if v else None,
        }
