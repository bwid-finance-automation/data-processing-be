"""API router for bank statement parsing."""

from typing import List, Tuple, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from fastapi.responses import Response
import uuid
import math
import base64
import re
import zipfile
from io import BytesIO
import pandas as pd  # Added pandas for HTML handling

from app.presentation.schemas.bank_statement_schemas import (
    ParseBankStatementsResponse,
    BankStatementResponse,
    BankBalanceResponse,
    BankTransactionResponse,
    SupportedBanksResponse,
    PowerAutomateParseRequest,
    PowerAutomateParseResponse,
    SharePointBalanceItem,
    SharePointTransactionItem,
    SharePointData
)
from app.application.finance.bank_statement_parser.parse_bank_statements import ParseBankStatementsUseCase
from app.application.finance.bank_statement_parser.bank_parsers.parser_factory import ParserFactory
from app.application.finance.bank_statement_parser.bank_statement_db_service import BankStatementDbService
from app.application.finance.bank_statement_parser.gemini_ocr_service import GeminiOCRService
from app.core.dependencies import get_bank_statement_db_service, get_ai_usage_repository
from app.infrastructure.persistence.repositories import AIUsageRepository
from app.shared.utils.logging_config import get_logger
from uuid import UUID as PyUUID

logger = get_logger(__name__)

router = APIRouter(prefix="/bank-statements", tags=["Finance - Bank Statement Parser"])

# In-memory storage for downloaded files (session-based)
_file_storage = {}


def _generate_export_filename(transactions: list, balances: list) -> str:
    """
    Generate export filename:
    - Single bank: {BANK_NAME}_statement_{timestamp}.xlsx (e.g., VCB_statement_20260112_165835.xlsx)
    - Multiple banks: bank_statement_{timestamp}.xlsx
    """
    from datetime import datetime

    # Collect unique bank names from transactions and balances
    bank_names = set()
    for tx in transactions:
        if hasattr(tx, 'bank_name') and tx.bank_name:
            bank_names.add(tx.bank_name.upper())
    for bal in balances:
        if hasattr(bal, 'bank_name') and bal.bank_name:
            bank_names.add(bal.bank_name.upper())

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Single bank: use bank name, multiple banks: use generic name
    if len(bank_names) == 1:
        bank_code = list(bank_names)[0]
        return f"{bank_code}_statement_{timestamp}.xlsx"
    else:
        return f"bank_statement_{timestamp}.xlsx"


def _format_for_sharepoint(transactions: list, balances: list) -> SharePointData:
    """
    Convert parsed transactions and balances to SharePoint Lists format.

    Args:
        transactions: List of BankTransaction objects
        balances: List of BankBalance objects

    Returns:
        SharePointData containing formatted balances and transactions for SharePoint Lists
    """
    from datetime import datetime

    def safe_float(value, default=0.0) -> float:
        """Convert value to float, handling NaN and None."""
        if value is None:
            return default
        try:
            f = float(value)
            return default if math.isnan(f) else f
        except (ValueError, TypeError):
            return default

    def safe_str(value, default="") -> str:
        """Convert value to string, handling None."""
        if value is None:
            return default
        return str(value).strip() if str(value).strip() else default

    def get_date_str(value) -> str:
        """Get date as YYYY-MM-DD string."""
        if value is None:
            return datetime.now().strftime("%Y-%m-%d")
        if hasattr(value, 'strftime'):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10] if len(str(value)) >= 10 else str(value)

    def get_date_short(value) -> str:
        """Get date as ddMMyy string for External ID."""
        if value is None:
            return datetime.now().strftime("%d%m%y")
        if hasattr(value, 'strftime'):
            return value.strftime("%d%m%y")
        # Try to parse YYYY-MM-DD format
        try:
            date_str = str(value)[:10]
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 3:
                    return f"{parts[2]}{parts[1]}{parts[0][2:]}"
        except Exception:
            pass
        return datetime.now().strftime("%d%m%y")

    def get_date_yyyymmdd(value) -> str:
        """Get date as YYYYMMDD string for StatementName."""
        if value is None:
            return datetime.now().strftime("%Y%m%d")
        if hasattr(value, 'strftime'):
            return value.strftime("%Y%m%d")
        # Try to parse YYYY-MM-DD format
        try:
            date_str = str(value)[:10]
            if "-" in date_str:
                return date_str.replace("-", "")
        except Exception:
            pass
        return datetime.now().strftime("%Y%m%d")

    sp_balances: list[SharePointBalanceItem] = []
    sp_transactions: list[SharePointTransactionItem] = []

    # Build a map of balance keys for linking transactions
    # Key format: {bank_code}_{acc_no}_{currency}
    balance_map: dict[str, tuple[str, str]] = {}  # key -> (balance_external_id, statement_name)

    # Process balances
    for idx, bal in enumerate(balances, start=1):
        bank_code = safe_str(getattr(bal, 'bank_name', ''), 'UNKNOWN')
        acc_no = safe_str(getattr(bal, 'acc_no', ''), '')
        currency = safe_str(getattr(bal, 'currency', ''), 'VND')
        statement_date = getattr(bal, 'statement_date', None)

        # Generate IDs
        date_short = get_date_short(statement_date)
        date_yyyymmdd = get_date_yyyymmdd(statement_date)
        external_id = f"{date_short}_{idx:04d}"
        statement_name = f"BS/{bank_code}/{currency}-{acc_no}/{date_yyyymmdd}"

        # Calculate totals from transactions for this balance
        balance_key = f"{bank_code}_{acc_no}_{currency}"
        total_debit = 0.0
        total_credit = 0.0
        for tx in transactions:
            tx_bank = safe_str(getattr(tx, 'bank_name', ''), '')
            tx_acc = safe_str(getattr(tx, 'acc_no', ''), '')
            tx_currency = safe_str(getattr(tx, 'currency', ''), 'VND')
            tx_key = f"{tx_bank}_{tx_acc}_{tx_currency}"
            if tx_key == balance_key:
                total_debit += safe_float(getattr(tx, 'debit', 0))
                total_credit += safe_float(getattr(tx, 'credit', 0))

        # Store mapping for transaction linking
        balance_map[balance_key] = (external_id, statement_name)

        sp_balances.append(SharePointBalanceItem(
            ExternalID=external_id,
            StatementName=statement_name,
            BankAccountNumber=acc_no,
            BankCode=bank_code,
            OpeningBalance=safe_float(getattr(bal, 'opening_balance', 0)),
            ClosingBalance=safe_float(getattr(bal, 'closing_balance', 0)),
            TotalDebit=total_debit,
            TotalCredit=total_credit,
            Currency=currency,
            StatementDate=get_date_str(statement_date)
        ))

    # Process transactions
    for idx, tx in enumerate(transactions, start=1):
        bank_code = safe_str(getattr(tx, 'bank_name', ''), 'UNKNOWN')
        acc_no = safe_str(getattr(tx, 'acc_no', ''), '')
        currency = safe_str(getattr(tx, 'currency', ''), 'VND')
        tx_date = getattr(tx, 'date', None)

        # Find parent balance
        tx_key = f"{bank_code}_{acc_no}_{currency}"
        balance_external_id, bank_statement_daily = balance_map.get(
            tx_key,
            ("", f"BS/{bank_code}/{currency}-{acc_no}/{get_date_yyyymmdd(tx_date)}")
        )

        # Generate IDs
        date_short = get_date_short(tx_date)
        external_id = f"line_{date_short}_{idx:04d}"
        line_name = f"{bank_statement_daily}/{idx:04d}"

        # Determine amounts and type
        debit = safe_float(getattr(tx, 'debit', 0))
        credit = safe_float(getattr(tx, 'credit', 0))
        amount = abs(debit) if debit > 0 else abs(credit)
        trans_type = "D" if debit > 0 else ("C" if credit > 0 else None)

        sp_transactions.append(SharePointTransactionItem(
            ExternalID=external_id,
            BalanceExternalID=balance_external_id,
            BankStatementDaily=bank_statement_daily,
            LineName=line_name,
            BankCode=bank_code,
            BankAccountNumber=acc_no,
            TransID=safe_str(getattr(tx, 'transaction_id', ''), None) or None,
            TransDate=get_date_str(tx_date),
            Description=safe_str(getattr(tx, 'description', ''), ''),
            Currency=currency,
            Debit=debit,
            Credit=credit,
            Amount=amount,
            TransType=trans_type,
            Balance=None,  # Not used
            Partner=safe_str(getattr(tx, 'beneficiary_acc_name', ''), None) or None,
            PartnerAccount=safe_str(getattr(tx, 'beneficiary_acc_no', ''), None) or None,
            PartnerBankID=safe_str(getattr(tx, 'beneficiary_bank', ''), None) or None
        ))

    return SharePointData(balances=sp_balances, transactions=sp_transactions)


def _check_zip_encrypted(zip_bytes: bytes) -> bool:
    """Check if a ZIP file is password-protected."""
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
            for file_info in zf.infolist():
                if file_info.flag_bits & 0x1:  # Encrypted flag
                    return True
                # Try to read a small file to check for encryption
                if not file_info.is_dir():
                    try:
                        zf.read(file_info.filename)
                        return False  # Successfully read without password
                    except RuntimeError as e:
                        if "encrypted" in str(e).lower() or "password" in str(e).lower():
                            return True
                        raise
    except Exception:
        pass
    return False


def _extract_files_from_zip(
    zip_bytes: bytes,
    zip_filename: str,
    password: Optional[str] = None
) -> Tuple[List[Tuple[str, bytes]], List[Tuple[str, bytes]], List[dict], bool]:
    """
    Extract Excel and PDF files from a ZIP archive.

    Args:
        zip_bytes: The ZIP file content as bytes
        zip_filename: Original ZIP filename for error reporting
        password: Optional password for encrypted ZIP files

    Returns:
        Tuple of:
        - excel_files: List of (filename, content) tuples for Excel files
        - pdf_files: List of (filename, content) tuples for PDF files
        - errors: List of error dicts for unsupported files
        - needs_password: True if ZIP is encrypted and no/wrong password provided
    """
    excel_files = []
    pdf_files = []
    errors = []
    needs_password = False

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
            # Set password if provided
            pwd_bytes = password.encode('utf-8') if password else None

            for file_info in zf.infolist():
                # Skip directories
                if file_info.is_dir():
                    continue

                # Skip hidden files and macOS resource forks
                filename = file_info.filename
                basename = filename.split('/')[-1]
                if basename.startswith('.') or basename.startswith('__MACOSX'):
                    continue

                # Read file content
                try:
                    content = zf.read(filename, pwd=pwd_bytes)
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "encrypted" in error_msg or "password" in error_msg or "bad password" in error_msg:
                        needs_password = True
                        if password:
                            errors.append({
                                "file_name": zip_filename,
                                "error": "Wrong password for encrypted ZIP file"
                            })
                        else:
                            errors.append({
                                "file_name": zip_filename,
                                "error": "ZIP file is encrypted. Please provide password."
                            })
                        return excel_files, pdf_files, errors, needs_password
                    errors.append({
                        "file_name": f"{zip_filename}/{filename}",
                        "error": f"Failed to read from ZIP: {str(e)}"
                    })
                    continue
                except Exception as e:
                    errors.append({
                        "file_name": f"{zip_filename}/{filename}",
                        "error": f"Failed to read from ZIP: {str(e)}"
                    })
                    continue

                # Categorize by extension
                lower_name = filename.lower()
                if lower_name.endswith(('.xlsx', '.xls')):
                    excel_files.append((basename, content))
                elif lower_name.endswith('.pdf'):
                    pdf_files.append((basename, content))
                else:
                    errors.append({
                        "file_name": f"{zip_filename}/{filename}",
                        "error": "Unsupported file type. Only .xlsx, .xls, .pdf are supported inside ZIP."
                    })

    except zipfile.BadZipFile:
        errors.append({
            "file_name": zip_filename,
            "error": "Invalid ZIP file or corrupted archive"
        })
    except Exception as e:
        errors.append({
            "file_name": zip_filename,
            "error": f"Failed to process ZIP: {str(e)}"
        })

    return excel_files, pdf_files, errors, needs_password


@router.get("/", summary="Bank Statement Parser Info")
def get_info():
    """Get information about the Bank Statement Parser API."""
    return {
        "name": "Bank Statement Parser",
        "version": "1.0.0",
        "description": "Automatically detect and parse bank statements from multiple banks",
        "features": [
            "Auto-detect bank from Excel files",
            "PDF OCR with Gemini 2.0 Flash",
            "Parse transactions (Date, Amount, Description, etc.)",
            "Extract opening/closing balances",
            "Batch processing (multiple files)",
            "Standardized Excel output",
            "Support for Vietnamese banks"
        ],
        "supported_banks": ParserFactory.get_supported_banks(),
        "endpoints": {
            "parse": "POST /bank-statements/parse - Parse bank statements from Excel (batch)",
            "parse_pdf": "POST /bank-statements/parse-pdf - Parse bank statements from PDF using Gemini OCR",
            "download": "GET /bank-statements/download/{session_id} - Download Excel output",
            "supported_banks": "GET /bank-statements/supported-banks - Get list of supported banks"
        }
    }


@router.get("/health", summary="Health Check")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bank_statement_parser",
        "supported_banks": ParserFactory.get_supported_banks()
    }


@router.get("/supported-banks", response_model=SupportedBanksResponse, summary="Get Supported Banks")
def get_supported_banks():
    """Get list of supported banks."""
    banks = ParserFactory.get_supported_banks()
    return SupportedBanksResponse(
        banks=banks,
        count=len(banks)
    )


def _check_pdf_encrypted_in_memory(pdf_bytes: bytes) -> bool:
    """Check if a PDF is password-protected using pikepdf."""
    try:
        import pikepdf
        from io import BytesIO
        input_stream = BytesIO(pdf_bytes)
        pdf = pikepdf.open(input_stream)
        pdf.close()
        return False  # Can open without password = not encrypted
    except Exception as e:
        error_msg = str(e).lower()
        if "password" in error_msg or "encrypted" in error_msg:
            return True  # Needs password = encrypted
        return False  # Other error, assume not encrypted


def _analyze_zip_contents(
    zip_bytes: bytes,
    zip_filename: str,
    zip_password: Optional[str] = None
) -> dict:
    """
    Analyze contents of a ZIP file without extracting fully.
    Returns list of files with their encryption status.
    """
    result = {
        "zip_filename": zip_filename,
        "zip_encrypted": False,
        "zip_password_correct": True,
        "files": [],
        "total_files": 0,
        "pdf_count": 0,
        "excel_count": 0,
        "encrypted_pdf_count": 0,
        "error": None
    }

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
            pwd_bytes = zip_password.encode('utf-8') if zip_password else None

            for file_info in zf.infolist():
                # Skip directories and hidden files
                if file_info.is_dir():
                    continue
                filename = file_info.filename
                basename = filename.split('/')[-1]
                if basename.startswith('.') or basename.startswith('__MACOSX'):
                    continue

                lower_name = filename.lower()

                # Skip unsupported file types
                if not (lower_name.endswith('.pdf') or lower_name.endswith(('.xlsx', '.xls'))):
                    continue

                file_entry = {
                    "filename": basename,
                    "full_path": filename,
                    "file_type": "pdf" if lower_name.endswith('.pdf') else "excel",
                    "size": file_info.file_size,
                    "is_encrypted": False,
                    "password_required": False,
                    "error": None
                }

                result["total_files"] += 1

                # Check if ZIP itself is encrypted
                if file_info.flag_bits & 0x1:
                    result["zip_encrypted"] = True
                    if not zip_password:
                        file_entry["error"] = "ZIP file is encrypted, password required"
                        file_entry["password_required"] = True
                        result["zip_password_correct"] = False
                        result["files"].append(file_entry)
                        continue

                # Try to read the file
                try:
                    content = zf.read(filename, pwd=pwd_bytes)

                    # For PDF files, check if they are password-protected
                    if lower_name.endswith('.pdf'):
                        result["pdf_count"] += 1
                        is_encrypted = _check_pdf_encrypted_in_memory(content)
                        file_entry["is_encrypted"] = is_encrypted
                        if is_encrypted:
                            result["encrypted_pdf_count"] += 1
                            file_entry["password_required"] = True
                    else:
                        result["excel_count"] += 1

                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "encrypted" in error_msg or "password" in error_msg or "bad password" in error_msg:
                        result["zip_encrypted"] = True
                        result["zip_password_correct"] = False
                        file_entry["error"] = "ZIP password required or incorrect"
                        file_entry["password_required"] = True
                    else:
                        file_entry["error"] = f"Failed to read: {str(e)}"
                except Exception as e:
                    file_entry["error"] = f"Failed to read: {str(e)}"

                result["files"].append(file_entry)

    except zipfile.BadZipFile:
        result["error"] = "Invalid ZIP file or corrupted archive"
    except Exception as e:
        result["error"] = f"Failed to analyze ZIP: {str(e)}"

    return result


@router.post("/analyze-zip", summary="Analyze ZIP Contents")
async def analyze_zip_contents(
    file: UploadFile = File(..., description="ZIP file to analyze"),
    password: Optional[str] = Form(None, description="Password for encrypted ZIP file"),
):
    """
    Analyze the contents of a ZIP file before parsing.

    Returns a list of files inside the ZIP with:
    - File names and types (PDF/Excel)
    - Which PDF files are password-protected
    - Total counts

    Use this endpoint to determine which files need passwords before calling parse-pdf.
    """
    try:
        content = await file.read()
        result = _analyze_zip_contents(content, file.filename, password)

        if result["error"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing ZIP contents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze ZIP: {str(e)}")


@router.post("/verify-zip-password", summary="Verify ZIP Password")
async def verify_zip_password(
    file: UploadFile = File(..., description="ZIP file to verify password"),
    password: str = Form(..., description="Password to verify"),
):
    """
    Verify if the provided password is correct for an encrypted ZIP file.
    Returns {"valid": true} if password is correct, {"valid": false} otherwise.
    """
    try:
        content = await file.read()

        # Check if ZIP is encrypted first
        if not _check_zip_encrypted(content):
            return {"valid": True, "message": "ZIP file is not encrypted"}

        # Try to extract with the password
        try:
            with zipfile.ZipFile(BytesIO(content), 'r') as zf:
                pwd_bytes = password.encode('utf-8') if password else None

                # Try to read the first non-directory file
                for file_info in zf.infolist():
                    if not file_info.is_dir():
                        try:
                            zf.read(file_info.filename, pwd=pwd_bytes)
                            return {"valid": True, "message": "Password is correct"}
                        except RuntimeError as e:
                            error_msg = str(e).lower()
                            if "bad password" in error_msg or "password" in error_msg or "encrypted" in error_msg:
                                return {"valid": False, "message": "Incorrect password"}
                            raise

                return {"valid": True, "message": "ZIP file is empty or contains only directories"}

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying ZIP password: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify ZIP password: {str(e)}")


@router.post("/parse", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements (Batch)")
async def parse_bank_statements(
    files: List[UploadFile] = File(..., description="Bank statement files (.xlsx, .xls, .zip containing Excel/PDF)"),
    zip_passwords: Optional[str] = Form(None, description="Comma-separated passwords for encrypted ZIP files (e.g., 'pass1,,pass3'). Use empty string for non-encrypted ZIPs."),
    zip_pdf_passwords: Optional[str] = Form(None, description="JSON object mapping PDF filenames inside ZIP to passwords (e.g., '{\"file1.pdf\": \"pass1\", \"file2.pdf\": \"pass2\"}'). Use this for password-protected PDFs inside ZIP files."),
    project_uuid: Optional[str] = Form(None, description="Project UUID to save statements to (optional)"),
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Parse multiple bank statement files in batch.

    Supports:
    - Excel files (.xlsx, .xls)
    - ZIP files containing Excel (.xlsx, .xls) and/or PDF (.pdf) files (including password-protected ZIPs)

    When a ZIP file is uploaded, all Excel and PDF files inside will be extracted
    and processed automatically. Excel files are parsed directly, PDF files are
    processed using Gemini OCR.

    For password-protected PDFs inside ZIP:
    1. First call /analyze-zip to see which PDFs need passwords
    2. Pass the passwords in zip_pdf_passwords as JSON: {"filename.pdf": "password"}
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Parse zip_passwords if provided
        zip_password_list = []
        if zip_passwords:
            zip_password_list = [pwd.strip() if pwd.strip() else None for pwd in zip_passwords.split(",")]

        # Parse zip_pdf_passwords JSON if provided (passwords for PDFs inside ZIP files)
        zip_pdf_password_map = {}
        if zip_pdf_passwords:
            try:
                import json
                zip_pdf_password_map = json.loads(zip_pdf_passwords)
                logger.info(f"Loaded {len(zip_pdf_password_map)} PDF passwords for ZIP contents")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse zip_pdf_passwords JSON: {e}")
                # Don't fail, just ignore invalid JSON

        # Separate files by type
        excel_files: List[Tuple[str, bytes]] = []
        pdf_files: List[Tuple[str, bytes, Optional[str], Optional[str]]] = []  # (filename, content, bank_code, password)
        zip_errors: List[dict] = []
        original_uploads: List[Tuple[str, bytes]] = []  # For saving to DB
        zip_extraction_info: dict = {}  # Track extracted file counts per ZIP
        zip_index = 0  # Track index for zip password mapping

        for file in files:
            filename = file.filename
            content = await file.read()
            original_uploads.append((filename, content))

            lower_name = filename.lower()

            if lower_name.endswith('.zip'):
                # Get password for this ZIP file
                zip_password = zip_password_list[zip_index] if zip_index < len(zip_password_list) else None
                zip_index += 1

                # Extract files from ZIP
                extracted_excel, extracted_pdf, errors, needs_password = _extract_files_from_zip(content, filename, zip_password)
                excel_files.extend(extracted_excel)
                # Add extracted PDFs with passwords from zip_pdf_password_map if available
                for name, data in extracted_pdf:
                    pdf_password = zip_pdf_password_map.get(name)  # Get password by filename
                    pdf_files.append((name, data, None, pdf_password))
                zip_errors.extend(errors)

                # Track extraction info for this ZIP
                zip_extraction_info[filename] = {
                    "excel_count": len(extracted_excel),
                    "pdf_count": len(extracted_pdf),
                    "excel_files": [f[0] for f in extracted_excel],
                    "pdf_files": [f[0] for f in extracted_pdf]
                }
                logger.info(f"Extracted from {filename}: {len(extracted_excel)} Excel, {len(extracted_pdf)} PDF files")

            elif lower_name.endswith(('.xlsx', '.xls')):
                excel_files.append((filename, content))

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {filename}. Supported: .xlsx, .xls, .zip"
                )

        # Check if we have any files to process
        if not excel_files and not pdf_files:
            if zip_errors:
                return ParseBankStatementsResponse(
                    statements=[],
                    summary={
                        "total_files": len(files),
                        "successful": 0,
                        "failed": len(zip_errors),
                        "failed_files": zip_errors,
                        "total_transactions": 0,
                        "total_balances": 0
                    },
                    download_url="",
                    session_id=""
                )
            raise HTTPException(status_code=400, detail="No valid files found to process")

        # Parse using use case
        use_case = ParseBankStatementsUseCase()

        # Initialize combined result
        combined_result = {
            "statements": [],
            "all_transactions": [],
            "all_balances": [],
            "summary": {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "failed_files": [],
                "total_transactions": 0,
                "total_balances": 0
            },
            "ai_usage": None
        }

        # Process Excel files
        if excel_files:
            excel_result = use_case.execute(excel_files)
            combined_result["statements"].extend(excel_result["statements"])
            combined_result["all_transactions"].extend(excel_result["all_transactions"])
            combined_result["all_balances"].extend(excel_result["all_balances"])
            combined_result["summary"]["total_files"] += excel_result["summary"]["total_files"]
            combined_result["summary"]["successful"] += excel_result["summary"]["successful"]
            combined_result["summary"]["failed"] += excel_result["summary"]["failed"]
            combined_result["summary"]["failed_files"].extend(excel_result["summary"]["failed_files"])

        # Process PDF files (from ZIP)
        if pdf_files:
            pdf_result = use_case.execute_from_pdf(pdf_files)
            combined_result["statements"].extend(pdf_result["statements"])
            combined_result["all_transactions"].extend(pdf_result["all_transactions"])
            combined_result["all_balances"].extend(pdf_result["all_balances"])
            combined_result["summary"]["total_files"] += pdf_result["summary"]["total_files"]
            combined_result["summary"]["successful"] += pdf_result["summary"]["successful"]
            combined_result["summary"]["failed"] += pdf_result["summary"]["failed"]
            combined_result["summary"]["failed_files"].extend(pdf_result["summary"]["failed_files"])
            combined_result["ai_usage"] = pdf_result.get("ai_usage")

        # Add ZIP extraction errors
        if zip_errors:
            combined_result["summary"]["failed"] += len(zip_errors)
            combined_result["summary"]["failed_files"].extend(zip_errors)

        # Update totals
        combined_result["summary"]["total_transactions"] = len(combined_result["all_transactions"])
        combined_result["summary"]["total_balances"] = len(combined_result["all_balances"])

        result = combined_result

        # Generate Excel output (ERP Template format)
        excel_bytes = use_case.export_to_erp_template_excel(
            result["all_transactions"],
            result["all_balances"]
        )

        # Store for download
        session_id = str(uuid.uuid4())
        export_filename = _generate_export_filename(result["all_transactions"], result["all_balances"])
        _file_storage[session_id] = {
            "content": excel_bytes,
            "filename": export_filename
        }

        # Save to database
        try:
            # Parse project_uuid if provided
            parsed_project_uuid = None
            if project_uuid:
                try:
                    parsed_project_uuid = PyUUID(project_uuid)
                except ValueError:
                    logger.warning(f"Invalid project_uuid format: {project_uuid}")

            # Save file upload records (with file content for download later)
            for filename, content in original_uploads:
                # Determine content type
                lower_name = filename.lower()
                if lower_name.endswith('.xlsx'):
                    content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_metadata = {"source": "parse-excel", "project_uuid": project_uuid}
                elif lower_name.endswith('.xls'):
                    content_type = "application/vnd.ms-excel"
                    file_metadata = {"source": "parse-excel", "project_uuid": project_uuid}
                elif lower_name.endswith('.zip'):
                    content_type = "application/zip"
                    # Include extraction info for ZIP files
                    extraction_info = zip_extraction_info.get(filename, {})
                    file_metadata = {
                        "source": "parse-zip",
                        "project_uuid": project_uuid,
                        "extracted_excel_count": extraction_info.get("excel_count", 0),
                        "extracted_pdf_count": extraction_info.get("pdf_count", 0),
                        "extracted_excel_files": extraction_info.get("excel_files", []),
                        "extracted_pdf_files": extraction_info.get("pdf_files", [])
                    }
                else:
                    content_type = "application/octet-stream"
                    file_metadata = {"source": "parse-excel", "project_uuid": project_uuid}

                await db_service.save_file_upload(
                    filename=filename,
                    file_size=len(content),
                    file_content=content,  # Save actual file to disk
                    file_type="bank_statement",
                    session_id=session_id,
                    content_type=content_type,
                    processing_status="completed",
                    metadata=file_metadata,
                )

            # Save parsed statements to database (with project linkage if provided)
            await db_service.save_statements_batch(
                result["statements"],
                session_id,
                project_uuid=parsed_project_uuid
            )

            # Save Excel output to disk for later download from history
            await db_service.save_excel_output(session_id, excel_bytes)

            await db_service.db.commit()
            logger.info(f"Saved {len(result['statements'])} statements to database (session: {session_id}, project: {project_uuid})")
        except Exception as db_error:
            logger.error(f"Failed to save to database: {db_error}")
            # Don't fail the request, just log the error - parsing was successful

        # Log AI usage if PDF files were processed from ZIP
        ai_usage = None
        if result.get("ai_usage") and pdf_files:
            from app.presentation.schemas.bank_statement_schemas import AIUsageMetrics, AIUsageFileMetrics
            from app.infrastructure.database.models.ai_usage import AIUsageModel
            from datetime import datetime

            ai_usage_data = result["ai_usage"]
            ai_usage = AIUsageMetrics(
                total_input_tokens=ai_usage_data.get("total_input_tokens", 0),
                total_output_tokens=ai_usage_data.get("total_output_tokens", 0),
                total_tokens=ai_usage_data.get("total_tokens", 0),
                total_processing_time_ms=ai_usage_data.get("total_processing_time_ms", 0),
                total_processing_time_seconds=ai_usage_data.get("total_processing_time_seconds", 0),
                files_processed=ai_usage_data.get("files_processed", 0),
                files_successful=ai_usage_data.get("files_successful", 0),
                files_failed=ai_usage_data.get("files_failed", 0),
                model_name=ai_usage_data.get("model_name", ""),
                file_metrics=[
                    AIUsageFileMetrics(**fm) for fm in ai_usage_data.get("file_metrics", [])
                ]
            )

            # Save AI usage to database
            try:
                input_cost = ai_usage_data.get("total_input_tokens", 0) * 0.000000075
                output_cost = ai_usage_data.get("total_output_tokens", 0) * 0.0000003
                estimated_cost = input_cost + output_cost

                ai_usage_log = AIUsageModel(
                    project_id=None,
                    case_id=None,
                    session_id=session_id,
                    provider="gemini",
                    model_name=ai_usage_data.get("model_name", "gemini-2.0-flash"),
                    task_type="ocr",
                    task_description="Bank statement PDF OCR parsing (from ZIP)",
                    file_name=", ".join([f[0] for f in pdf_files]),
                    file_count=ai_usage_data.get("files_processed", len(pdf_files)),
                    input_tokens=ai_usage_data.get("total_input_tokens", 0),
                    output_tokens=ai_usage_data.get("total_output_tokens", 0),
                    total_tokens=ai_usage_data.get("total_tokens", 0),
                    processing_time_ms=ai_usage_data.get("total_processing_time_ms", 0),
                    estimated_cost_usd=estimated_cost,
                    success=ai_usage_data.get("files_failed", 0) == 0,
                    error_message=None,
                    metadata_json={
                        "files_successful": ai_usage_data.get("files_successful", 0),
                        "files_failed": ai_usage_data.get("files_failed", 0),
                        "source": "zip_extraction",
                    },
                    requested_at=datetime.utcnow(),
                )
                await ai_usage_repo.create(ai_usage_log)
                await ai_usage_repo.session.commit()
                logger.info(f"Saved AI usage log for session: {session_id}")
            except Exception as ai_log_error:
                logger.error(f"Failed to save AI usage log: {ai_log_error}")

        # Convert to response schema
        statements_response = []
        for stmt in result["statements"]:
            # Convert balance
            balance_response = None
            if stmt.balance:
                # Handle NaN values in balance
                opening_bal = stmt.balance.opening_balance
                closing_bal = stmt.balance.closing_balance
                if isinstance(opening_bal, float) and math.isnan(opening_bal):
                    opening_bal = 0.0
                if isinstance(closing_bal, float) and math.isnan(closing_bal):
                    closing_bal = 0.0

                balance_response = BankBalanceResponse(
                    bank_name=stmt.balance.bank_name,
                    acc_no=stmt.balance.acc_no,
                    currency=stmt.balance.currency,
                    opening_balance=opening_bal,
                    closing_balance=closing_bal,
                    statement_date=stmt.balance.statement_date
                )

            # Convert transactions
            transactions_response = [
                BankTransactionResponse(
                    bank_name=tx.bank_name,
                    acc_no=tx.acc_no,
                    debit=None if (tx.debit is None or (isinstance(tx.debit, float) and math.isnan(tx.debit))) else tx.debit,
                    credit=None if (tx.credit is None or (isinstance(tx.credit, float) and math.isnan(tx.credit))) else tx.credit,
                    date=tx.date.isoformat() if tx.date else None,  # Convert date to ISO string
                    description=tx.description,
                    currency=tx.currency,
                    transaction_id=tx.transaction_id,
                    beneficiary_bank=tx.beneficiary_bank,
                    beneficiary_acc_no=tx.beneficiary_acc_no,
                    beneficiary_acc_name=tx.beneficiary_acc_name
                )
                for tx in stmt.transactions
            ]

            statements_response.append(BankStatementResponse(
                bank_name=stmt.bank_name,
                file_name=stmt.file_name,
                balance=balance_response,
                transactions=transactions_response,
                transaction_count=len(stmt.transactions)
            ))

        return ParseBankStatementsResponse(
            statements=statements_response,
            summary=result["summary"],
            download_url=f"/api/finance/bank-statements/download/{session_id}",
            session_id=session_id,
            ai_usage=ai_usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse bank statements: {str(e)}")


@router.post("/parse-pdf", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements from PDF (Gemini OCR)")
async def parse_bank_statements_pdf(
    files: List[UploadFile] = File(..., description="Bank statement PDF files (.pdf, .zip containing PDFs)"),
    bank_codes: Optional[str] = Form(None, description="Comma-separated bank codes for each file (e.g., 'VIB,ACB,VCB'). Leave empty for auto-detection."),
    passwords: Optional[str] = Form(None, description="Comma-separated passwords for encrypted PDF files (e.g., 'pass1,,pass3'). Use empty string for non-encrypted PDFs."),
    zip_passwords: Optional[str] = Form(None, description="Comma-separated passwords for encrypted ZIP files (e.g., 'pass1,,pass3'). Use empty string for non-encrypted ZIPs."),
    zip_pdf_passwords: Optional[str] = Form(None, description="JSON object mapping PDF filenames inside ZIP to passwords (e.g., '{\"file1.pdf\": \"pass1\", \"file2.pdf\": \"pass2\"}'). Use this for password-protected PDFs inside ZIP files."),
    project_uuid: Optional[str] = Form(None, description="Project UUID to save statements to (optional)"),
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Parse multiple bank statement PDF files using Gemini Flash OCR.

    Supports:
    - PDF files (.pdf)
    - ZIP files containing PDF files (.zip) (including password-protected ZIPs)

    When a ZIP file is uploaded, all PDF files inside will be extracted and processed.

    For password-protected PDFs inside ZIP:
    1. First call /analyze-zip to see which PDFs need passwords
    2. Pass the passwords in zip_pdf_passwords as JSON: {"filename.pdf": "password"}
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Parse bank_codes if provided
        bank_code_list = []
        if bank_codes:
            bank_code_list = [code.strip() if code.strip() else None for code in bank_codes.split(",")]

        # Parse passwords if provided (for PDF files)
        password_list = []
        if passwords:
            password_list = [pwd.strip() if pwd.strip() else None for pwd in passwords.split(",")]

        # Parse zip_passwords if provided
        zip_password_list = []
        if zip_passwords:
            zip_password_list = [pwd.strip() if pwd.strip() else None for pwd in zip_passwords.split(",")]

        # Parse zip_pdf_passwords JSON if provided (passwords for PDFs inside ZIP files)
        zip_pdf_password_map = {}
        if zip_pdf_passwords:
            try:
                import json
                zip_pdf_password_map = json.loads(zip_pdf_passwords)
                logger.info(f"Loaded {len(zip_pdf_password_map)} PDF passwords for ZIP contents")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse zip_pdf_passwords JSON: {e}")
                # Don't fail, just ignore invalid JSON

        # Read files into memory
        pdf_inputs = []
        zip_errors = []
        original_uploads: List[Tuple[str, bytes]] = []  # For saving to DB
        zip_extraction_info: dict = {}  # Track extracted file counts per ZIP
        pdf_index = 0  # Track index for bank_code/password mapping
        zip_index = 0  # Track index for zip password mapping

        for file in files:
            filename = file.filename
            content = await file.read()
            original_uploads.append((filename, content))
            lower_name = filename.lower()

            if lower_name.endswith('.zip'):
                # Get password for this ZIP file
                zip_password = zip_password_list[zip_index] if zip_index < len(zip_password_list) else None
                zip_index += 1

                # Extract PDFs from ZIP
                extracted_excel, extracted_pdfs, errors, needs_password = _extract_files_from_zip(content, filename, zip_password)
                # Add extracted PDFs with passwords from zip_pdf_password_map if available
                for name, data in extracted_pdfs:
                    pdf_password = zip_pdf_password_map.get(name)  # Get password by filename
                    pdf_inputs.append((name, data, None, pdf_password))
                zip_errors.extend(errors)

                # Track extraction info for this ZIP (for PDF endpoint, we only process PDFs)
                zip_extraction_info[filename] = {
                    "excel_count": len(extracted_excel),
                    "pdf_count": len(extracted_pdfs),
                    "excel_files": [f[0] for f in extracted_excel],
                    "pdf_files": [f[0] for f in extracted_pdfs]
                }
                logger.info(f"Extracted {len(extracted_pdfs)} PDF files from {filename}")

            elif lower_name.endswith('.pdf'):
                bank_code = bank_code_list[pdf_index] if pdf_index < len(bank_code_list) else None
                password = password_list[pdf_index] if pdf_index < len(password_list) else None
                pdf_inputs.append((filename, content, bank_code, password))
                pdf_index += 1

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {filename}. Supported: .pdf, .zip"
                )

        # Check if we have any PDFs to process
        if not pdf_inputs:
            if zip_errors:
                return ParseBankStatementsResponse(
                    statements=[],
                    summary={
                        "total_files": len(files),
                        "successful": 0,
                        "failed": len(zip_errors),
                        "failed_files": zip_errors,
                        "total_transactions": 0,
                        "total_balances": 0
                    },
                    download_url="",
                    session_id=""
                )
            raise HTTPException(status_code=400, detail="No PDF files found to process")

        # Parse using use case
        use_case = ParseBankStatementsUseCase()
        result = use_case.execute_from_pdf(pdf_inputs)

        # Add ZIP extraction errors to result
        if zip_errors:
            result["summary"]["failed"] += len(zip_errors)
            result["summary"]["failed_files"].extend(zip_errors)

        # Generate Excel output (ERP Template format)
        excel_bytes = use_case.export_to_erp_template_excel(
            result["all_transactions"],
            result["all_balances"]
        )

        # Store for download
        session_id = str(uuid.uuid4())
        export_filename = _generate_export_filename(result["all_transactions"], result["all_balances"])
        _file_storage[session_id] = {
            "content": excel_bytes,
            "filename": export_filename
        }

        # Save to database
        try:
            # Parse project_uuid if provided
            parsed_project_uuid = None
            if project_uuid:
                try:
                    parsed_project_uuid = PyUUID(project_uuid)
                except ValueError:
                    logger.warning(f"Invalid project_uuid format: {project_uuid}")

            # Save file upload records (with file content for download later)
            for filename, content in original_uploads:
                # Determine content type
                lower_name = filename.lower()
                if lower_name.endswith('.pdf'):
                    content_type = "application/pdf"
                    file_metadata = {"source": "parse-pdf", "project_uuid": project_uuid}
                elif lower_name.endswith('.zip'):
                    content_type = "application/zip"
                    # Include extraction info for ZIP files
                    extraction_info = zip_extraction_info.get(filename, {})
                    file_metadata = {
                        "source": "parse-zip",
                        "project_uuid": project_uuid,
                        "extracted_excel_count": extraction_info.get("excel_count", 0),
                        "extracted_pdf_count": extraction_info.get("pdf_count", 0),
                        "extracted_excel_files": extraction_info.get("excel_files", []),
                        "extracted_pdf_files": extraction_info.get("pdf_files", [])
                    }
                else:
                    content_type = "application/octet-stream"
                    file_metadata = {"source": "parse-pdf", "project_uuid": project_uuid}

                await db_service.save_file_upload(
                    filename=filename,
                    file_size=len(content),
                    file_content=content,  # Save actual file to disk
                    file_type="bank_statement",
                    session_id=session_id,
                    content_type=content_type,
                    processing_status="completed",
                    metadata=file_metadata,
                )

            # Save parsed statements to database (with project linkage if provided)
            await db_service.save_statements_batch(
                result["statements"],
                session_id,
                project_uuid=parsed_project_uuid
            )

            # Save Excel output to disk for later download from history
            await db_service.save_excel_output(session_id, excel_bytes)

            await db_service.db.commit()
            logger.info(f"Saved {len(result['statements'])} statements to database (session: {session_id}, project: {project_uuid})")
        except Exception as db_error:
            logger.error(f"Failed to save to database: {db_error}")
            # Don't fail the request, just log the error - parsing was successful

        # Convert to response schema
        statements_response = []
        for stmt in result["statements"]:
            # Convert balance
            balance_response = None
            if stmt.balance:
                # Handle NaN values in balance
                opening_bal = stmt.balance.opening_balance
                closing_bal = stmt.balance.closing_balance
                if isinstance(opening_bal, float) and math.isnan(opening_bal):
                    opening_bal = 0.0
                if isinstance(closing_bal, float) and math.isnan(closing_bal):
                    closing_bal = 0.0

                balance_response = BankBalanceResponse(
                    bank_name=stmt.balance.bank_name,
                    acc_no=stmt.balance.acc_no,
                    currency=stmt.balance.currency,
                    opening_balance=opening_bal,
                    closing_balance=closing_bal,
                    statement_date=stmt.balance.statement_date
                )

            # Convert transactions
            transactions_response = [
                BankTransactionResponse(
                    bank_name=tx.bank_name,
                    acc_no=tx.acc_no,
                    debit=None if (tx.debit is None or (isinstance(tx.debit, float) and math.isnan(tx.debit))) else tx.debit,
                    credit=None if (tx.credit is None or (isinstance(tx.credit, float) and math.isnan(tx.credit))) else tx.credit,
                    date=tx.date.isoformat() if tx.date else None,
                    description=tx.description,
                    currency=tx.currency,
                    transaction_id=tx.transaction_id,
                    beneficiary_bank=tx.beneficiary_bank,
                    beneficiary_acc_no=tx.beneficiary_acc_no,
                    beneficiary_acc_name=tx.beneficiary_acc_name
                )
                for tx in stmt.transactions
            ]

            statements_response.append(BankStatementResponse(
                bank_name=stmt.bank_name,
                file_name=stmt.file_name,
                balance=balance_response,
                transactions=transactions_response,
                transaction_count=len(stmt.transactions)
            ))

        # Build AI usage metrics if available (PDF parsing only)
        ai_usage = None
        if "ai_usage" in result and result["ai_usage"]:
            from app.presentation.schemas.bank_statement_schemas import AIUsageMetrics, AIUsageFileMetrics
            from app.infrastructure.database.models.ai_usage import AIUsageModel
            from datetime import datetime

            ai_usage_data = result["ai_usage"]
            ai_usage = AIUsageMetrics(
                total_input_tokens=ai_usage_data.get("total_input_tokens", 0),
                total_output_tokens=ai_usage_data.get("total_output_tokens", 0),
                total_tokens=ai_usage_data.get("total_tokens", 0),
                total_processing_time_ms=ai_usage_data.get("total_processing_time_ms", 0),
                total_processing_time_seconds=ai_usage_data.get("total_processing_time_seconds", 0),
                files_processed=ai_usage_data.get("files_processed", 0),
                files_successful=ai_usage_data.get("files_successful", 0),
                files_failed=ai_usage_data.get("files_failed", 0),
                model_name=ai_usage_data.get("model_name", ""),
                file_metrics=[
                    AIUsageFileMetrics(**fm) for fm in ai_usage_data.get("file_metrics", [])
                ]
            )

            # Save AI usage to database
            try:
                # Calculate estimated cost (Gemini Flash pricing)
                input_cost = ai_usage_data.get("total_input_tokens", 0) * 0.000000075  # $0.075/1M tokens
                output_cost = ai_usage_data.get("total_output_tokens", 0) * 0.0000003  # $0.30/1M tokens
                estimated_cost = input_cost + output_cost

                ai_usage_log = AIUsageModel(
                    project_id=None,  # Will be set if we have project info
                    case_id=None,
                    session_id=session_id,
                    provider="gemini",
                    model_name=ai_usage_data.get("model_name", "gemini-2.0-flash"),
                    task_type="ocr",
                    task_description="Bank statement PDF OCR parsing",
                    file_name=", ".join([f.filename for f in files]),
                    file_count=ai_usage_data.get("files_processed", len(files)),
                    input_tokens=ai_usage_data.get("total_input_tokens", 0),
                    output_tokens=ai_usage_data.get("total_output_tokens", 0),
                    total_tokens=ai_usage_data.get("total_tokens", 0),
                    processing_time_ms=ai_usage_data.get("total_processing_time_ms", 0),
                    estimated_cost_usd=estimated_cost,
                    success=ai_usage_data.get("files_failed", 0) == 0,
                    error_message=None,
                    metadata_json={
                        "files_successful": ai_usage_data.get("files_successful", 0),
                        "files_failed": ai_usage_data.get("files_failed", 0),
                        "file_metrics": ai_usage_data.get("file_metrics", []),
                    },
                    requested_at=datetime.utcnow(),
                )
                await ai_usage_repo.create(ai_usage_log)
                await ai_usage_repo.session.commit()
                logger.info(f"Saved AI usage log for session: {session_id}")
            except Exception as ai_log_error:
                logger.error(f"Failed to save AI usage log: {ai_log_error}")
                # Don't fail the request

        return ParseBankStatementsResponse(
            statements=statements_response,
            summary=result["summary"],
            download_url=f"/api/finance/bank-statements/download/{session_id}",
            session_id=session_id,
            ai_usage=ai_usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF bank statements: {str(e)}")


@router.get("/download/{session_id}", summary="Download Excel Output")
def download_excel(session_id: str):
    if session_id not in _file_storage:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    file_data = _file_storage[session_id]
    logger.info(f"Download filename: {file_data['filename']}")

    return Response(
        content=file_data["content"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={file_data['filename']}"
        }
    )


@router.get("/download-history/{session_id}", summary="Download Excel from History")
async def download_excel_from_history(
    session_id: str,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    # ... (giữ nguyên code cũ)
    from app.domain.finance.bank_statement_parser.models.bank_transaction import BankTransaction
    from app.domain.finance.bank_statement_parser.models.bank_statement import BankBalance

    try:
        # First, try to get cached Excel from disk
        excel_bytes = await db_service.get_excel_output(session_id)

        if excel_bytes:
            logger.info(f"Serving cached Excel for session: {session_id}")
            # Get bank names from database for filename
            statements = await db_service.get_statements_by_session(session_id)
            bank_names = set(stmt.bank_name.upper() for stmt in statements if stmt.bank_name)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Single bank: use bank name, multiple banks: use generic name
            if len(bank_names) == 1:
                filename = f"{list(bank_names)[0]}_statement_{timestamp}.xlsx"
            else:
                filename = f"bank_statement_{timestamp}.xlsx"
            return Response(
                content=excel_bytes,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )

        # Fallback: regenerate from database
        logger.info(f"Excel not cached, regenerating from database for session: {session_id}")

        # Get statements from database
        statements = await db_service.get_statements_by_session(session_id)

        if not statements:
            raise HTTPException(status_code=404, detail="Session not found in history")

        # Convert database models to domain models
        all_transactions = []
        all_balances = []

        for stmt in statements:
            # Convert transactions
            for tx in stmt.transactions:
                all_transactions.append(BankTransaction(
                    bank_name=stmt.bank_name,
                    acc_no=tx.acc_no or "",
                    debit=float(tx.debit) if tx.debit else None,
                    credit=float(tx.credit) if tx.credit else None,
                    date=tx.transaction_date,
                    description=tx.description or "",
                    currency=tx.currency or "VND",
                    transaction_id=tx.transaction_id or "",
                    beneficiary_bank=tx.beneficiary_bank or "",
                    beneficiary_acc_no=tx.beneficiary_acc_no or "",
                    beneficiary_acc_name=tx.beneficiary_acc_name or "",
                ))

            # Convert balances
            for bal in stmt.balances:
                all_balances.append(BankBalance(
                    bank_name=stmt.bank_name,
                    acc_no=bal.acc_no or "",
                    currency=bal.currency or "VND",
                    opening_balance=float(bal.opening_balance) if bal.opening_balance else 0.0,
                    closing_balance=float(bal.closing_balance) if bal.closing_balance else 0.0,
                ))

        if not all_transactions and not all_balances:
            raise HTTPException(status_code=404, detail="No data found for this session")

        # Generate Excel using the use case
        use_case = ParseBankStatementsUseCase()
        excel_bytes = use_case.export_to_erp_template_excel(all_transactions, all_balances)

        # Save to disk for future requests
        await db_service.save_excel_output(session_id, excel_bytes)

        # Generate filename
        filename = _generate_export_filename(all_transactions, all_balances)

        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Excel from history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Excel: {str(e)}")


# ========== File History & Storage Endpoints ==========

@router.get("/storage/stats", summary="Get Storage Statistics")
async def get_storage_stats(
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """Get storage statistics for uploaded files."""
    try:
        stats = await db_service.get_storage_stats()
        return {
            "status": "ok",
            "retention_days": 30,
            **stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")


@router.post("/storage/cleanup", summary="Cleanup Old Files")
async def cleanup_old_files(
    retention_days: int = 7,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """Manually trigger cleanup of old uploaded files (default: 7 days)."""
    try:
        stats = await db_service.cleanup_old_files(retention_days)
        return {
            "status": "ok",
            "message": f"Cleanup completed. {stats['files_deleted']} files deleted.",
            **stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/uploaded-files/{session_id}", summary="List Uploaded Files by Session")
async def list_uploaded_files_by_session(
    session_id: str,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """List all uploaded files for a specific session."""
    try:
        files = await db_service.get_files_by_session(session_id)
        return {
            "session_id": session_id,
            "files": [
                {
                    "id": f.id,
                    "original_filename": f.original_filename,
                    "file_size": f.file_size,
                    "file_type": f.file_type,
                    "content_type": f.content_type,
                    "processing_status": f.processing_status,
                    "uploaded_at": f.created_at.isoformat() if f.created_at else None,
                    "download_url": f"/api/finance/bank-statements/uploaded-file/{f.id}" if f.file_path else None,
                }
                for f in files
            ],
            "count": len(files),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/session/{session_id}", summary="Get Session Details")
async def get_session_details(
    session_id: str,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """
    Get detailed information about a parsing session.

    Returns:
    - Session metadata
    - Uploaded files with extraction info
    - Parsed statements with transactions and balances
    """
    try:
        # Get uploaded files
        files = await db_service.get_files_by_session(session_id)
        if not files:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get parsed statements
        statements = await db_service.get_statements_by_session(session_id)

        # Calculate totals
        total_transactions = sum(len(s.transactions) for s in statements)
        total_balances = sum(len(s.balances) for s in statements)
        banks = list(set(s.bank_name for s in statements if s.bank_name))

        # Get earliest and latest processed_at
        processed_dates = [s.processed_at for s in statements if s.processed_at]
        processed_at = max(processed_dates) if processed_dates else None

        # Calculate file type breakdown
        zip_files = []
        excel_files = []
        pdf_files = []

        for f in files:
            file_info = {
                "id": f.id,
                "file_name": f.original_filename,
                "file_size": f.file_size,
                "content_type": f.content_type,
                "uploaded_at": f.created_at.isoformat() if f.created_at else None,
                "download_url": f"/api/finance/bank-statements/uploaded-file/{f.id}" if f.file_path else None,
                "metadata": f.metadata_json or {},
            }

            lower_name = (f.original_filename or "").lower()
            if lower_name.endswith('.zip'):
                zip_files.append(file_info)
            elif lower_name.endswith('.pdf'):
                pdf_files.append(file_info)
            elif lower_name.endswith(('.xlsx', '.xls')):
                excel_files.append(file_info)

        # Calculate extracted counts from ZIP metadata
        extracted_excel_count = 0
        extracted_pdf_count = 0
        extracted_excel_files = []
        extracted_pdf_files = []

        for zf in zip_files:
            meta = zf.get("metadata", {})
            extracted_excel_count += meta.get("extracted_excel_count", 0)
            extracted_pdf_count += meta.get("extracted_pdf_count", 0)
            extracted_excel_files.extend(meta.get("extracted_excel_files", []))
            extracted_pdf_files.extend(meta.get("extracted_pdf_files", []))

        # Format statements for response
        statements_data = []
        for stmt in statements:
            # Get balance info
            balance_data = None
            if stmt.balances:
                bal = stmt.balances[0]  # Usually one balance per statement
                balance_data = {
                    "acc_no": bal.acc_no,
                    "currency": bal.currency,
                    "opening_balance": float(bal.opening_balance) if bal.opening_balance else 0,
                    "closing_balance": float(bal.closing_balance) if bal.closing_balance else 0,
                }

            # Format transactions
            transactions_data = []
            for tx in stmt.transactions:
                transactions_data.append({
                    "id": tx.id,
                    "date": tx.transaction_date.isoformat() if tx.transaction_date else None,
                    "description": tx.description,
                    "debit": float(tx.debit) if tx.debit else None,
                    "credit": float(tx.credit) if tx.credit else None,
                    "currency": tx.currency,
                    "acc_no": tx.acc_no,
                    "transaction_id": tx.transaction_id,
                    "beneficiary_bank": tx.beneficiary_bank,
                    "beneficiary_acc_no": tx.beneficiary_acc_no,
                    "beneficiary_acc_name": tx.beneficiary_acc_name,
                })

            statements_data.append({
                "uuid": str(stmt.uuid),
                "bank_name": stmt.bank_name,
                "file_name": stmt.file_name,
                "processed_at": stmt.processed_at.isoformat() if stmt.processed_at else None,
                "transaction_count": len(stmt.transactions),
                "balance": balance_data,
                "transactions": transactions_data,
            })

        return {
            "session_id": session_id,
            "processed_at": processed_at.isoformat() if processed_at else None,
            "banks": banks,
            "total_files": len(files),
            "total_statements": len(statements),
            "total_transactions": total_transactions,
            "total_balances": total_balances,
            "download_url": f"/api/finance/bank-statements/download-history/{session_id}",

            # File breakdown
            "uploaded_files": {
                "zip_files": zip_files,
                "excel_files": excel_files,
                "pdf_files": pdf_files,
                "zip_count": len(zip_files),
                "excel_count": len(excel_files),
                "pdf_count": len(pdf_files),
            },

            # Extracted from ZIP
            "extracted_from_zip": {
                "excel_count": extracted_excel_count,
                "pdf_count": extracted_pdf_count,
                "excel_files": extracted_excel_files,
                "pdf_files": extracted_pdf_files,
            },

            # Parsed statements
            "statements": statements_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session details: {str(e)}")


@router.get("/uploaded-file/{file_id}", summary="Download Uploaded File")
async def download_uploaded_file(
    file_id: int,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """Download a previously uploaded file by ID."""
    try:
        result = await db_service.get_file_content(file_id)
        if not result:
            raise HTTPException(status_code=404, detail="File not found or no longer available")

        content, filename, content_type = result

        return Response(
            content=content,
            media_type=content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\""
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


# ========== Power Automate Endpoint ==========

@router.post(
    "/parse-power-automate",
    response_model=PowerAutomateParseResponse,
    summary="Parse Bank Statements (Power Automate)",
    tags=["Finance - Bank Statement Parser", "Power Automate"]
)
async def parse_bank_statements_power_automate(request: PowerAutomateParseRequest):
    """
    Parse bank statement files from Power Automate.

    **Designed for Power Automate HTTP action with JSON body.**

    **Input:**
    - files: Array of objects with file_name and file_content_base64
    - Supported formats: .xlsx, .xls, .pdf
    - return_excel_base64: If true, returns Excel output as base64 string

    **Power Automate Usage:**
    1. Use "Get attachment" action to get email attachments
    2. Use base64() expression to encode file content
    3. Send HTTP POST to this endpoint with JSON body
    4. Use the excel_base64 from response to create file in SharePoint/OneDrive

    **Example Request Body:**
    ```json
    {
        "files": [
            {
                "file_name": "ACB_Statement.xlsx",
                "file_content_base64": "UEsDBBQAAAAI..."
            }
        ],
        "return_excel_base64": true
    }
    ```
    """
    try:
        # Validate request
        if not request.files:
            return PowerAutomateParseResponse(
                success=False,
                message="No files provided",
                summary={"total_files": 0, "successful": 0, "failed": 0}
            )

        # Initialize lists for different input types
        file_data: List[Tuple[str, bytes]] = []  # For Excel/PDF files
        text_data: List[Tuple[str, str, str]] = []  # For OCR text (file_name, ocr_text, bank_code)
        json_data: List[Tuple[str, str, str]] = []  # For JSON from Gemini (file_name, json_content, bank_code)
        decode_errors = []

        for file_input in request.files:
            file_name = file_input.get_file_name()

            # Check if OCR text is provided (prioritize over file content)
            if file_input.has_ocr_text():
                # Process as OCR text input
                text_data.append((
                    file_name,
                    file_input.ocr_text,
                    file_input.bank_code  # Can be None for auto-detection
                ))
                continue

            # Otherwise, process as file content
            content_base64 = file_input.get_content_base64()

            # Check if content is provided
            if not content_base64:
                decode_errors.append({
                    "file_name": file_name,
                    "error": "No content provided. Include 'ocr_text' or 'contentBytes'."
                })
                continue

            # Validate file extension for binary files
            if not file_name.lower().endswith(('.xlsx', '.xls', '.pdf', '.txt')):
                decode_errors.append({
                    "file_name": file_name,
                    "error": "Unsupported file type. Only .xlsx, .xls, .pdf, .txt are supported."
                })
                continue

            try:
                # Decode base64 content
                file_bytes = base64.b64decode(content_base64)

                # Handle .txt files - try JSON first, then fallback to OCR text
                if file_name.lower().endswith('.txt'):
                    try:
                        txt_content = file_bytes.decode('utf-8')

                        # Try to detect if content is JSON (from Gemini structured output)
                        clean_content = txt_content.strip()
                        # Remove markdown code blocks if present
                        if clean_content.startswith("```"):
                            clean_content = re.sub(r'^```(?:json)?\s*', '', clean_content)
                            clean_content = re.sub(r'\s*```$', '', clean_content)

                        # Check if it looks like JSON
                        is_json = clean_content.startswith('{') or clean_content.startswith('[')

                        if is_json:
                            # Will be processed by execute_from_json
                            json_data.append((
                                file_name,
                                txt_content,
                                file_input.bank_code
                            ))
                            logger.info(f"Processing {file_name} as JSON input from Gemini ({len(txt_content)} characters)")
                        else:
                            # Fallback to OCR text parsing
                            text_data.append((
                                file_name,
                                txt_content,
                                file_input.bank_code
                            ))
                            logger.info(f"Processing {file_name} as OCR text input ({len(txt_content)} characters)")

                        continue  # Skip file_data processing
                    except UnicodeDecodeError:
                        decode_errors.append({
                            "file_name": file_name,
                            "error": "Failed to decode .txt file as UTF-8 text"
                        })
                        continue

                # =========================================================
                # FIX: Handle "Fake Excel" (HTML) files
                # VCB parser already handles HTML natively, so don't convert VCB HTML files
                # =========================================================
                try:
                    # Check signature for HTML content
                    prefix = file_bytes[:100].lower()
                    if b"<html" in prefix or b"<!doctype html" in prefix:
                        # Check if this is a VCB HTML file (VCB parser handles HTML natively)
                        content_preview = file_bytes.decode('utf-8', errors='ignore')
                        is_vcb_html = "SAO KÊ TÀI KHOẢN" in content_preview and "VIETCOMBANK" in content_preview.upper()

                        if is_vcb_html:
                            logger.info(f"Detected VCB HTML file: {file_name}. Passing directly to VCB parser (no conversion needed).")
                            # Don't convert - VCB parser handles HTML natively
                        else:
                            logger.info(f"Detected HTML-based Excel file: {file_name}. Attempting conversion...")

                            # Read HTML tables using pandas
                            dfs = pd.read_html(BytesIO(file_bytes))

                            if dfs:
                                # Convert the first table found to a clean Excel binary
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    # Write without header/index to preserve exact layout for parser
                                    dfs[0].to_excel(writer, index=False, header=False)

                                file_bytes = output.getvalue()
                                logger.info(f"Successfully converted {file_name} from HTML to Excel binary.")
                except Exception as html_err:
                    logger.warning(f"Failed to convert HTML-Excel for {file_name}, using original bytes. Error: {html_err}")
                # =========================================================

                file_data.append((file_name, file_bytes))
            except Exception as e:
                decode_errors.append({
                    "file_name": file_name,
                    "error": f"Failed to decode base64 or convert file: {str(e)}"
                })

        # Check if we have any data to process
        if not file_data and not text_data and not json_data:
            return PowerAutomateParseResponse(
                success=False,
                message="No valid files to process",
                summary={
                    "total_files": len(request.files),
                    "successful": 0,
                    "failed": len(request.files),
                    "failed_files": decode_errors
                }
            )

        # Parse using use case
        use_case = ParseBankStatementsUseCase()

        # Initialize combined result
        result = {
            "statements": [],
            "all_transactions": [],
            "all_balances": [],
            "summary": {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "failed_files": [],
                "total_transactions": 0,
                "total_balances": 0
            }
        }

        # Process JSON inputs from Gemini (highest accuracy - visual table understanding)
        if json_data:
            json_result = use_case.execute_from_json(json_data)
            result["statements"].extend(json_result["statements"])
            result["all_transactions"].extend(json_result["all_transactions"])
            result["all_balances"].extend(json_result["all_balances"])
            result["summary"]["total_files"] += json_result["summary"]["total_files"]
            result["summary"]["successful"] += json_result["summary"]["successful"]
            result["summary"]["failed"] += json_result["summary"]["failed"]
            result["summary"]["failed_files"].extend(json_result["summary"]["failed_files"])

        # Process OCR text inputs (fallback for non-JSON text)
        if text_data:
            text_result = use_case.execute_from_text(text_data)
            result["statements"].extend(text_result["statements"])
            result["all_transactions"].extend(text_result["all_transactions"])
            result["all_balances"].extend(text_result["all_balances"])
            result["summary"]["total_files"] += text_result["summary"]["total_files"]
            result["summary"]["successful"] += text_result["summary"]["successful"]
            result["summary"]["failed"] += text_result["summary"]["failed"]
            result["summary"]["failed_files"].extend(text_result["summary"]["failed_files"])

        # Process Excel/PDF file inputs (existing logic)
        if file_data:
            file_result = use_case.execute(file_data)
            result["statements"].extend(file_result["statements"])
            result["all_transactions"].extend(file_result["all_transactions"])
            result["all_balances"].extend(file_result["all_balances"])
            result["summary"]["total_files"] += file_result["summary"]["total_files"]
            result["summary"]["successful"] += file_result["summary"]["successful"]
            result["summary"]["failed"] += file_result["summary"]["failed"]
            result["summary"]["failed_files"].extend(file_result["summary"]["failed_files"])

        # Update totals
        result["summary"]["total_transactions"] = len(result["all_transactions"])
        result["summary"]["total_balances"] = len(result["all_balances"])
        result["summary"]["total_accounts"] = len(result["all_balances"])  # Alias for UI display

        # Add decode errors to failed files
        if decode_errors:
            result["summary"]["failed"] += len(decode_errors)
            result["summary"]["failed_files"].extend(decode_errors)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Generate Excel file with 2 sheets (Balance + Details)
        excel_bytes = use_case.export_to_netsuite_excel(
            result["all_transactions"],
            result["all_balances"]
        )
        excel_filename = _generate_export_filename(result["all_transactions"], result["all_balances"])
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')

        # Format data for SharePoint Lists
        sharepoint_data = _format_for_sharepoint(
            result["all_transactions"],
            result["all_balances"]
        )

        # Prepare response
        response = PowerAutomateParseResponse(
            success=result["summary"]["successful"] > 0,
            message=f"Processed {result['summary']['successful']} of {result['summary']['total_files']} files successfully",
            summary=result["summary"],
            excel_base64=excel_base64,
            excel_filename=excel_filename,
            sharepoint_data=sharepoint_data
        )

        return response

    except Exception as e:
        return PowerAutomateParseResponse(
            success=False,
            message=f"Failed to parse bank statements: {str(e)}",
            summary={"total_files": len(request.files), "successful": 0, "failed": len(request.files)}
        )