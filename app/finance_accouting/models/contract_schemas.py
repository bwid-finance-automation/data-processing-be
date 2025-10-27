"""Data models for contract OCR information extraction."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Party(BaseModel):
    """Represents a party in the contract."""
    name: str
    role: str  # e.g., "Buyer", "Seller", "Contractor", "Client"
    address: Optional[str] = None


class RatePeriod(BaseModel):
    """Represents a rate period with start date, end date, and rates."""
    start_date: Optional[str] = Field(None, description="Period start date (MM-DD-YYYY)")
    end_date: Optional[str] = Field(None, description="Period end date (MM-DD-YYYY)")
    monthly_rate_per_sqm: Optional[str] = Field(None, description="Monthly rate per sqm for this period")
    total_monthly_rate: Optional[str] = Field(None, description="Total monthly rate for this period")
    num_months: Optional[str] = Field(None, description="Number of months in this billing period (as string)")
    foc_from: Optional[str] = Field(None, description="FOC (Free of Charge) start date within this period (MM-DD-YYYY)")
    foc_to: Optional[str] = Field(None, description="FOC (Free of Charge) end date within this period (MM-DD-YYYY)")
    foc_num_months: Optional[str] = Field(None, description="Number of FOC months (as string)")
    service_charge_rate_per_sqm: Optional[str] = Field(None, description="Service charge rate per sqm per month for this period (raw rate, not calculated total)")


class ContractInfo(BaseModel):
    """Extracted contract information."""
    # General fields
    contract_title: Optional[str] = Field(None, description="Title or name of the contract")
    contract_type: Optional[str] = Field(None, description="Type of contract (e.g., Service Agreement, NDA, Lease)")
    parties_involved: List[Party] = Field(default_factory=list, description="All parties involved in the contract")
    effective_date: Optional[str] = Field(None, description="Date when the contract becomes effective")
    expiration_date: Optional[str] = Field(None, description="Date when the contract expires or terminates")
    contract_value: Optional[str] = Field(None, description="Total monetary value of the contract")
    payment_terms: Optional[str] = Field(None, description="Payment schedule and terms")
    key_obligations: List[str] = Field(default_factory=list, description="Main obligations of each party")
    termination_clauses: Optional[str] = Field(None, description="Conditions under which contract can be terminated")
    governing_law: Optional[str] = Field(None, description="Jurisdiction/law governing the contract")
    signatures_present: bool = Field(False, description="Whether signatures are present on the document")
    special_conditions: List[str] = Field(default_factory=list, description="Any special conditions or clauses")

    # Lease-specific fields
    internal_id: Optional[str] = Field(None, description="Internal ID")
    id: Optional[str] = Field(None, description="ID")
    historical: Optional[bool] = Field(None, description="Historical flag")
    master_record_id: Optional[str] = Field(None, description="Master Record ID")
    plc_id: Optional[str] = Field(None, description="PLC ID")
    unit_for_lease: Optional[str] = Field(None, description="Unit For Lease")
    type: Optional[str] = Field(None, description="Type")
    start_date: Optional[str] = Field(None, description="Start Date (deprecated - use rate_periods)")
    end_date: Optional[str] = Field(None, description="End Date (deprecated - use rate_periods)")
    tenant: Optional[str] = Field(None, description="Tenant name")
    monthly_rate_per_sqm: Optional[str] = Field(None, description="Monthly Rate per Sqm (deprecated - use rate_periods)")
    gla_for_lease: Optional[str] = Field(None, description="GLA for Lease (Gross Leasable Area)")
    total_monthly_rate: Optional[str] = Field(None, description="Total Monthly Rate (deprecated - use rate_periods)")
    rate_periods: List[RatePeriod] = Field(default_factory=list, description="Array of rate periods with dates and rates")
    months: Optional[int] = Field(None, description="Number of months")
    total_rate: Optional[str] = Field(None, description="Total Rate")
    status: Optional[str] = Field(None, description="Status")
    historical_journal_entry: Optional[str] = Field(None, description="Historical Journal Entry")
    amortization_journal_entry: Optional[str] = Field(None, description="Amortization Journal Entry")
    related_billing_schedule: Optional[str] = Field(None, description="Related Billing Schedule")
    subsidiary: Optional[str] = Field(None, description="Subsidiary")
    ccs_product_type: Optional[str] = Field(None, description="CCS Product Type")
    bwid_project: Optional[str] = Field(None, description="BWID Project")
    phase: Optional[str] = Field(None, description="Phase")

    # New Vietnamese contract fields (multilingual support: Vietnamese/English/Chinese)
    customer_name: Optional[str] = Field(None, description="Customer/Tenant name from 'bên đi thuê' section")
    contract_number: Optional[str] = Field(None, description="Contract number (Số hợp đồng / Contract No / 合同编号)")
    contract_date: Optional[str] = Field(None, description="Contract signing date (Ngày hợp đồng / Contract Date) - YYYY-MM-DD format")
    payment_terms_details: Optional[str] = Field(None, description="Full payment terms text (Thời hạn thanh toán tiền thuê)")
    deposit_amount: Optional[str] = Field(None, description="Total deposit amount only (Tiền đặt cọc / Deposit), no installment breakdown")
    handover_date: Optional[str] = Field(None, description="Property handover date (Ngày bàn giao / Handover Date) - YYYY-MM-DD format")
    gfa: Optional[str] = Field(None, description="Gross Floor Area - leasable construction floor area in sqm (different from GLA)")
    service_charge_rate: Optional[str] = Field(None, description="Service charge rate per sqm per month (Phí dịch vụ)")
    service_charge_applies_to: Optional[str] = Field(None, description="When service charge applies: 'rent_free_only', 'all_periods', or 'not_applicable'")
    service_charge_total: Optional[str] = Field(None, description="Calculated total service charge (computed in post-processing)")

    # OCR metadata
    raw_text: Optional[str] = Field(None, description="Raw OCR text extracted from the document")
    confidence_score: Optional[float] = Field(None, description="Confidence score of the extraction")


class TokenUsage(BaseModel):
    """Token usage statistics for API calls."""
    prompt_tokens: int = Field(0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the completion")
    total_tokens: int = Field(0, description="Total tokens used (prompt + completion)")


class CostEstimate(BaseModel):
    """Cost estimate for API usage."""
    input_cost: float = Field(0.0, description="Cost for input/prompt tokens in USD")
    output_cost: float = Field(0.0, description="Cost for output/completion tokens in USD")
    total_cost: float = Field(0.0, description="Total estimated cost in USD")
    model: Optional[str] = Field(None, description="Model used for pricing")
    currency: str = Field("USD", description="Currency of the cost estimate")


class ContractExtractionResult(BaseModel):
    """Result of the contract extraction process."""
    success: bool
    data: Optional[ContractInfo] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    source_file: Optional[str] = None
    token_usage: Optional[TokenUsage] = Field(None, description="Token usage statistics for this extraction")
    cost_estimate: Optional[CostEstimate] = Field(None, description="Estimated cost for this extraction")


class BatchContractResult(BaseModel):
    """Result of batch contract processing."""
    success: bool
    total_files: int
    successful: int
    failed: int
    results: List[ContractExtractionResult]
    total_token_usage: Optional[TokenUsage] = Field(None, description="Aggregated token usage across all contracts")
    total_cost_estimate: Optional[CostEstimate] = Field(None, description="Aggregated cost estimate across all contracts")


class SupportedFormatsResponse(BaseModel):
    """Supported file formats response."""
    formats: List[str]
    description: str


class UnitBreakdownInfo(BaseModel):
    """Information about a single unit from breakdown file."""
    customer_code: Optional[str] = None
    customer_name: Optional[str] = None
    tax_rate: Optional[float] = None
    unit: str
    gfa: float


class UnitBreakdownSummary(BaseModel):
    """Summary of unit breakdown processing."""
    success: bool
    total_units: int
    total_gfa: float
    units: List[UnitBreakdownInfo]
    error: Optional[str] = None


class ContractWithUnitsResult(BaseModel):
    """Result of processing contract with unit breakdown."""
    success: bool
    base_contract: Optional[ContractExtractionResult] = Field(None, description="Base contract extraction from PDF")
    unit_breakdown: Optional[UnitBreakdownSummary] = Field(None, description="Unit breakdown from Excel")
    unit_contracts: List[ContractInfo] = Field(default_factory=list, description="Individual contracts for each unit")
    total_units: int = Field(0, description="Total number of unit-specific contracts created")
    gfa_validation: Optional[dict] = Field(None, description="GFA validation result")
    error: Optional[str] = None
