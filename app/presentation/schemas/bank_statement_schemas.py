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
    statement_date: Optional[date] = Field(None, description="Statement period end date (from file)")


class BankStatementResponse(BaseModel):
    """Response schema for a single bank statement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bank_name: str
    file_name: str
    balance: Optional[BankBalanceResponse] = None
    transactions: List[BankTransactionResponse] = Field(default_factory=list)
    transaction_count: int = Field(0, description="Number of transactions parsed")


class AIUsageFileMetrics(BaseModel):
    """Usage metrics for a single file processed by AI."""
    file_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class AIUsageMetrics(BaseModel):
    """Aggregated AI API usage metrics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_processing_time_ms: float = 0.0
    total_processing_time_seconds: float = 0.0
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    model_name: str = ""
    file_metrics: List[AIUsageFileMetrics] = Field(default_factory=list)


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
                            "closing_balance": 15000000.0,
                            "statement_date": "2025-01-31"
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
                "download_url": "/api/finance/bank-statements/download/abc123",
                "session_id": "abc123",
                "ai_usage": {
                    "total_input_tokens": 5000,
                    "total_output_tokens": 2000,
                    "total_tokens": 7000,
                    "total_processing_time_seconds": 3.5,
                    "model_name": "gemini-2.0-flash"
                }
            }
        }
    )

    statements: List[BankStatementResponse] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    download_url: Optional[str] = Field(None, description="URL to download Excel output")
    session_id: Optional[str] = Field(None, description="Session ID for retrieving uploaded files")
    ai_usage: Optional[AIUsageMetrics] = Field(None, description="AI API usage metrics (only for PDF parsing)")


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


# ========== SharePoint Data Schemas ==========

class SharePointBalanceItem(BaseModel):
    """SharePoint List item for BSM_Balances."""

    ExternalID: str = Field(..., description="External ID format: {ddMMyy}_{sequence:04d}")
    StatementName: str = Field(..., description="BS/{BankCode}/{Currency}-{AccNo}/{YYYYMMDD}")
    BankAccountNumber: str = Field(..., description="Account number")
    BankCode: str = Field(..., description="Bank code/name")
    OpeningBalance: float = Field(0.0, description="Opening balance")
    ClosingBalance: float = Field(0.0, description="Closing balance")
    TotalDebit: float = Field(0.0, description="Sum of all debits")
    TotalCredit: float = Field(0.0, description="Sum of all credits")
    Currency: str = Field("VND", description="Currency code")
    StatementDate: str = Field(..., description="Statement date in YYYY-MM-DD format")


class SharePointTransactionItem(BaseModel):
    """SharePoint List item for BSM_Transactions."""

    ExternalID: str = Field(..., description="External ID format: line_{ddMMyy}_{sequence:04d}")
    BalanceExternalID: str = Field(..., description="Link to parent balance ExternalID")
    BankStatementDaily: str = Field(..., description="BS/{BankCode}/{Currency}-{AccNo}/{YYYYMMDD}")
    LineName: str = Field(..., description="{BankStatementDaily}/{sequence}")
    BankCode: str = Field(..., description="Bank code/name")
    BankAccountNumber: str = Field(..., description="Account number")
    TransID: Optional[str] = Field(None, description="Transaction ID")
    TransDate: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    Description: str = Field("", description="Transaction description")
    Currency: str = Field("VND", description="Currency code")
    Debit: float = Field(0.0, description="Debit amount")
    Credit: float = Field(0.0, description="Credit amount")
    Amount: float = Field(0.0, description="Absolute amount (debit or credit)")
    TransType: Optional[str] = Field(None, description="D for debit, C for credit")
    Balance: Optional[float] = Field(None, description="Running balance (not used)")
    Partner: Optional[str] = Field(None, description="Beneficiary account name")
    PartnerAccount: Optional[str] = Field(None, description="Beneficiary account number")
    PartnerBankID: Optional[str] = Field(None, description="Beneficiary bank")


class SharePointData(BaseModel):
    """Container for SharePoint Lists data."""

    balances: List[SharePointBalanceItem] = Field(default_factory=list, description="Items for BSM_Balances list")
    transactions: List[SharePointTransactionItem] = Field(default_factory=list, description="Items for BSM_Transactions list")


# ========== Power Automate Schemas ==========

class PowerAutomateFileInput(BaseModel):
    """Single file input for Power Automate - supports both formats."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields from Power Automate

    # Support both 'name' (from Power Automate) and 'file_name'
    name: Optional[str] = Field(default=None, description="Name of the file (Power Automate format)")
    file_name: Optional[str] = Field(default=None, description="Name of the file (alternative)")

    # Support both 'contentBytes' and 'file_content_base64'
    contentBytes: Optional[str] = Field(None, description="Base64 file content (Power Automate format)")
    file_content_base64: Optional[str] = Field(None, description="Base64 file content (alternative)")

    # OCR text input (from AI Builder)
    ocr_text: Optional[str] = Field(None, description="OCR extracted text from AI Builder. If provided, contentBytes is ignored")
    bank_code: Optional[str] = Field(None, description="Force specific bank parser (VIB, ACB, etc.). If null, auto-detect from text")

    # PDF password (for encrypted PDFs)
    password: Optional[str] = Field(None, description="Password for encrypted PDF files. Only needed if PDF is password-protected")

    # Optional fields from Power Automate
    url: Optional[str] = Field(None, description="OneDrive URL (not used for processing)")
    source: Optional[str] = Field(None, description="Source type (ZIP/PDF)")

    def get_file_name(self) -> str:
        """Get file name from either field."""
        return self.name or self.file_name or "unknown"

    def get_content_base64(self) -> Optional[str]:
        """Get base64 content from either field."""
        return self.contentBytes or self.file_content_base64

    def has_ocr_text(self) -> bool:
        """Check if OCR text is provided and not empty."""
        return self.ocr_text is not None and self.ocr_text.strip() != ""

    def get_password(self) -> Optional[str]:
        """Get password for encrypted PDF."""
        return self.password


class PowerAutomateParseRequest(BaseModel):
    """Request schema for Power Automate bank statement parsing."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "files": [
                    {
                        "name": "ACB_Jan2025.xlsx",
                        "contentBytes": "UEsDBBQAAAAI..."
                    }
                ]
            }
        }
    )

    files: List[PowerAutomateFileInput] = Field(
        ...,
        description="List of files to parse. Can be .xlsx, .xls, .pdf"
    )


class PowerAutomateParseResponse(BaseModel):
    """Response schema for Power Automate bank statement parsing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Processed successfully",
                "summary": {
                    "total_files": 2,
                    "successful": 2,
                    "failed": 0,
                    "total_transactions": 50,
                    "total_balances": 2
                },
                "excel_base64": "UEsDBBQAAAAI...",
                "excel_filename": "NetSuite_Export_abc123.xlsx",
                "sharepoint_data": {
                    "balances": [
                        {
                            "ExternalID": "070126_0001",
                            "StatementName": "BS/ACB/VND-123456789/20260107",
                            "BankAccountNumber": "123456789",
                            "BankCode": "ACB",
                            "OpeningBalance": 10000000.0,
                            "ClosingBalance": 15000000.0,
                            "TotalDebit": 2000000.0,
                            "TotalCredit": 7000000.0,
                            "Currency": "VND",
                            "StatementDate": "2026-01-07"
                        }
                    ],
                    "transactions": [
                        {
                            "ExternalID": "line_070126_0001",
                            "BalanceExternalID": "070126_0001",
                            "BankStatementDaily": "BS/ACB/VND-123456789/20260107",
                            "LineName": "BS/ACB/VND-123456789/20260107/0001",
                            "BankCode": "ACB",
                            "BankAccountNumber": "123456789",
                            "TransID": "FT12345",
                            "TransDate": "2026-01-07",
                            "Description": "Transfer from ABC Corp",
                            "Currency": "VND",
                            "Debit": 0.0,
                            "Credit": 5000000.0,
                            "Amount": 5000000.0,
                            "TransType": "C",
                            "Balance": None,
                            "Partner": "ABC Corp",
                            "PartnerAccount": "987654321",
                            "PartnerBankID": "VCB"
                        }
                    ]
                }
            }
        }
    )

    success: bool = Field(default=True)
    message: str = Field(default="")
    summary: Dict[str, Any] = Field(default_factory=dict)
    excel_base64: Optional[str] = Field(None, description="Base64 encoded Excel file with 2 sheets (Balance + Details)")
    excel_filename: Optional[str] = Field(default=None, description="Excel filename")
    sharepoint_data: Optional[SharePointData] = Field(None, description="SharePoint Lists data for Power Automate")
