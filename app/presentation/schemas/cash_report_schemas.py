"""
Cash Report API Schemas - Pydantic models for request/response.
"""
from datetime import date
from decimal import Decimal
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class CashMovementResponse(BaseModel):
    """Response schema for a single cash movement"""
    source: str
    bank_code: str
    account_number: str
    transaction_date: date
    description: str
    debit_amount: Optional[float] = None
    credit_amount: Optional[float] = None
    net_amount: float
    nature: Optional[str] = None
    key_payment: Optional[str] = None
    entity: Optional[str] = None
    grouping: Optional[str] = None
    currency: str = "VND"
    account_type: str = "Current Account"
    period: Optional[str] = None


class CashBalanceResponse(BaseModel):
    """Response schema for a cash balance"""
    entity: str
    bank_branch: str
    bank_name: str
    account_number: str
    account_type: str
    currency: str
    opening_balance_vnd: float
    opening_balance_usd: float
    closing_balance_vnd: float
    closing_balance_usd: float
    calculated_closing_vnd: float
    is_reconciled: bool
    reconciliation_diff: Optional[float] = None
    grouping: str


class SavingAccountResponse(BaseModel):
    """Response schema for a saving account"""
    account_number: str
    entity_code: str
    entity_name: str
    bank_branch: str
    bank_name: str
    currency: str
    grouping: str
    opening_date: date
    maturity_date: date
    term_months: int
    interest_rate: float
    principal_amount: float
    accrued_interest: float
    total_balance: float
    status: str
    days_to_maturity: int
    is_matured: bool


class EntitySummaryResponse(BaseModel):
    """Summary for an entity grouping"""
    grouping: str
    cash_in_vnd: float
    cash_in_vnd_equivalent: float
    cash_in_usd: float
    total_cash_usd_equivalent: float
    total_receipt: float
    total_payment: float
    net_movement: float
    opening_balance: float
    closing_balance: float


class BankSummaryResponse(BaseModel):
    """Summary for a bank"""
    bank_name: str
    bank_branch: str
    bwid_jsc_amount: float
    vc3_amount: float
    subsidiaries_amount: float
    total_amount: float
    percentage: float


class CashMovementSummaryResponse(BaseModel):
    """Summary of cash movements by category"""
    ordinary_receipt: float
    receipt_from_tenants: float
    refund_land_deal_deposit: float
    loan_drawdown: float
    vat_refund: float
    other_receipts: float
    internal_transfer_in: float
    internal_contribution_in: float
    total_cash_in: float

    ordinary_payment: float
    land_acquisition: float
    construction_expense: float
    operating_expense: float
    loan_repayment: float
    loan_interest: float
    internal_transfer_out: float
    internal_contribution_out: float
    total_cash_out: float

    net_cash_flow: float


class CashReportSummaryResponse(BaseModel):
    """Summary section of cash report"""
    report_date: date
    period: str
    opening_date: date
    ending_date: date
    fx_rate: float

    total_cash_usd_equivalent: float
    total_cash_vnd: float
    total_cash_usd: float

    prior_period_cash: float
    cash_change_amount: float
    cash_change_percentage: float

    bwid_jsc: EntitySummaryResponse
    vc3: EntitySummaryResponse
    subsidiaries: EntitySummaryResponse

    bank_summaries: List[BankSummaryResponse]
    movement_summary: CashMovementSummaryResponse

    internal_transfer_net: float
    internal_contribution_net: float
    is_balanced: bool


class CashReportResponse(BaseModel):
    """Complete cash report response"""
    report_id: str
    created_at: Optional[date] = None

    summary: CashReportSummaryResponse

    total_movements: int
    total_accounts: int
    total_saving_accounts: int

    reconciliation_errors: List[str] = []
    warnings: List[str] = []


class GenerateCashReportRequest(BaseModel):
    """Request schema for generating cash report"""
    fx_rate: float = Field(default=26175, description="VND/USD exchange rate")
    report_period: str = Field(default="", description="Report period (e.g., W1-2Oct25)")
    opening_date: Optional[date] = Field(default=None, description="Report start date")
    ending_date: Optional[date] = Field(default=None, description="Report end date")


class GenerateCashReportResponse(BaseModel):
    """Response schema for cash report generation"""
    success: bool
    report_id: Optional[str] = None
    download_url: Optional[str] = None

    summary: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []

    # Statistics
    total_movements: int = 0
    total_accounts: int = 0
    total_saving_accounts: int = 0
    is_balanced: bool = False


class ValidateTemplateResponse(BaseModel):
    """Response schema for template validation"""
    is_valid: bool
    sheet_count: int = 0
    movements_count: int = 0
    balances_count: int = 0
    saving_accounts_count: int = 0
    issues: List[str] = []


class ClassifyMovementRequest(BaseModel):
    """Request to classify a single movement"""
    description: str
    is_debit: bool = True


class ClassifyMovementResponse(BaseModel):
    """Response for movement classification"""
    key_payment: str
    category: str
    nature: str
    confidence: float = 1.0
    suggestions: List[Dict[str, Any]] = []


class ReconciliationResultResponse(BaseModel):
    """Response for reconciliation result"""
    account_number: str
    bank_name: str
    currency: str
    opening_balance: float
    total_debit: float
    total_credit: float
    calculated_closing: float
    bank_statement_balance: Optional[float] = None
    difference: float
    is_reconciled: bool
    issues: List[str] = []


class CashReportListItem(BaseModel):
    """Item in cash report list"""
    report_id: str
    period: str
    created_at: date
    total_cash_usd_mn: float
    is_balanced: bool
    status: str = "completed"


class CashReportListResponse(BaseModel):
    """Response for listing cash reports"""
    reports: List[CashReportListItem]
    total: int
    page: int = 1
    page_size: int = 20
