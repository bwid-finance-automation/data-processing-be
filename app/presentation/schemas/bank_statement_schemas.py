"""Pydantic schemas for bank statement parsing API."""

from typing import List, Optional, Dict, Any, Literal
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
                ],
                "return_excel_base64": True
            }
        }
    )

    files: List[PowerAutomateFileInput] = Field(
        ...,
        description="List of files to parse. Can be .xlsx, .xls, .pdf"
    )
    return_excel_base64: bool = Field(
        default=True,
        description="If true, return Excel output as base64 in response"
    )
    output_format: Literal["excel", "netsuite_csv", "both"] = Field(
        default="excel",
        description="Output format: 'excel' for Excel only, 'netsuite_csv' for NetSuite CSV files (Balance + Details), 'both' for all formats"
    )


class PowerAutomateParseResponse(BaseModel):
    """Response schema for Power Automate bank statement parsing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "success": True,
                "summary": {
                    "total_files": 2,
                    "successful": 2,
                    "failed": 0,
                    "total_transactions": 50,
                    "total_balances": 2
                },
                "statements": [],
                "excel_base64": "UEsDBBQAAAAI...",
                "excel_filename": "bank_statements_output.xlsx",
                "download_url": "/api/finance/bank-statements/download/abc123"
            }
        }
    )

    success: bool = Field(default=True)
    message: str = Field(default="")
    summary: Dict[str, Any] = Field(default_factory=dict)
    statements: List[BankStatementResponse] = Field(default_factory=list)
    excel_base64: Optional[str] = Field(None, description="Base64 encoded Excel output file")
    excel_filename: Optional[str] = Field(default="bank_statements_output.xlsx")
    csv_balance_base64: Optional[str] = Field(None, description="Base64 encoded NetSuite Balance CSV file")
    csv_balance_filename: Optional[str] = Field(None, description="NetSuite Balance CSV filename")
    csv_details_base64: Optional[str] = Field(None, description="Base64 encoded NetSuite Details CSV file")
    csv_details_filename: Optional[str] = Field(None, description="NetSuite Details CSV filename")
    download_url: Optional[str] = Field(None, description="URL to download Excel output")
