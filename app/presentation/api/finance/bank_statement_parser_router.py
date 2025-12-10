"""API router for bank statement parsing."""

from typing import List, Tuple, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
import uuid
import math
import base64
from io import BytesIO

from app.presentation.schemas.bank_statement_schemas import (
    ParseBankStatementsResponse,
    BankStatementResponse,
    BankBalanceResponse,
    BankTransactionResponse,
    SupportedBanksResponse,
    PowerAutomateParseRequest,
    PowerAutomateParseResponse
)
from app.application.finance.bank_statement_parser.parse_bank_statements import ParseBankStatementsUseCase
from app.application.finance.bank_statement_parser.bank_parsers.parser_factory import ParserFactory

router = APIRouter(prefix="/bank-statements", tags=["Finance - Bank Statement Parser"])

# In-memory storage for downloaded files (session-based)
_file_storage = {}


@router.get("/", summary="Bank Statement Parser Info")
def get_info():
    """Get information about the Bank Statement Parser API."""
    return {
        "name": "Bank Statement Parser",
        "version": "1.0.0",
        "description": "Automatically detect and parse bank statements from multiple banks",
        "features": [
            "Auto-detect bank from Excel files",
            "PDF OCR with Gemini 2.5 Flash",
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


@router.post("/parse", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements (Batch)")
async def parse_bank_statements(
    files: List[UploadFile] = File(..., description="Bank statement Excel files (.xlsx, .xls)")
):
    """
    Parse multiple bank statement files in batch.

    **Features:**
    - Auto-detect bank from file content
    - Parse transactions and balances
    - Generate standardized Excel output

    **Supported Banks:**
    - ACB (Asia Commercial Bank)
    - More banks coming soon...

    **Input:**
    - Multiple Excel files (.xlsx, .xls)

    **Output:**
    - Parsed statements with transactions and balances
    - Summary of processing results
    - Download URL for Excel output
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Read files into memory
        file_data = []
        for file in files:
            # Validate file extension
            if not file.filename.lower().endswith(('.xlsx', '.xls')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Only .xlsx and .xls files are supported."
                )

            content = await file.read()
            file_data.append((file.filename, content))

        # Parse using use case
        use_case = ParseBankStatementsUseCase()
        result = use_case.execute(file_data)

        # Generate Excel output
        excel_bytes = use_case.export_to_excel(
            result["all_transactions"],
            result["all_balances"]
        )

        # Store for download
        session_id = str(uuid.uuid4())
        _file_storage[session_id] = {
            "content": excel_bytes,
            "filename": f"bank_statements_{session_id}.xlsx"
        }

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
                    closing_balance=closing_bal
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
            download_url=f"/api/finance/bank-statements/download/{session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse bank statements: {str(e)}")


@router.post("/parse-pdf", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements from PDF (Gemini OCR)")
async def parse_bank_statements_pdf(
    files: List[UploadFile] = File(..., description="Bank statement PDF files (.pdf)"),
    bank_codes: Optional[str] = Form(None, description="Comma-separated bank codes for each file (e.g., 'VIB,ACB,VCB'). Leave empty for auto-detection.")
):
    """
    Parse multiple bank statement PDF files using Gemini Flash OCR.

    **Features:**
    - Direct PDF OCR using Gemini 2.5 Flash
    - Auto-detect bank from OCR text
    - Parse transactions and balances
    - Generate standardized Excel output

    **Supported Banks:**
    - VIB (Vietnam International Bank)
    - ACB (Asia Commercial Bank)
    - VCB (Vietcombank)
    - BIDV (Bank for Investment and Development)
    - And more...

    **Input:**
    - Multiple PDF files (.pdf)
    - Optional: bank_codes (comma-separated, same order as files)

    **Output:**
    - Parsed statements with transactions and balances
    - Summary of processing results
    - Download URL for Excel output

    **Note:** Requires GEMINI_API_KEY to be configured in .env
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Parse bank_codes if provided
        bank_code_list = []
        if bank_codes:
            bank_code_list = [code.strip() if code.strip() else None for code in bank_codes.split(",")]

        # Read files into memory
        pdf_inputs = []
        for i, file in enumerate(files):
            # Validate file extension
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Only .pdf files are supported for this endpoint."
                )

            content = await file.read()
            bank_code = bank_code_list[i] if i < len(bank_code_list) else None
            pdf_inputs.append((file.filename, content, bank_code))

        # Parse using use case
        use_case = ParseBankStatementsUseCase()
        result = use_case.execute_from_pdf(pdf_inputs)

        # Generate Excel output
        excel_bytes = use_case.export_to_excel(
            result["all_transactions"],
            result["all_balances"]
        )

        # Store for download
        session_id = str(uuid.uuid4())
        _file_storage[session_id] = {
            "content": excel_bytes,
            "filename": f"bank_statements_pdf_{session_id}.xlsx"
        }

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
                    closing_balance=closing_bal
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

        return ParseBankStatementsResponse(
            statements=statements_response,
            summary=result["summary"],
            download_url=f"/api/finance/bank-statements/download/{session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF bank statements: {str(e)}")


@router.get("/download/{session_id}", summary="Download Excel Output")
def download_excel(session_id: str):
    """
    Download the Excel output file for a parsing session.

    **Parameters:**
    - session_id: Session ID from parse response

    **Returns:**
    - Excel file with:
      - Transactions sheet (all transactions)
      - Balances sheet (all balances)
      - Summary sheet (by bank and account)
    """
    if session_id not in _file_storage:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    file_data = _file_storage[session_id]

    return Response(
        content=file_data["content"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={file_data['filename']}"
        }
    )


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
            if not file_name.lower().endswith(('.xlsx', '.xls', '.pdf')):
                decode_errors.append({
                    "file_name": file_name,
                    "error": "Unsupported file type. Only .xlsx, .xls, .pdf are supported."
                })
                continue

            try:
                # Decode base64 content
                file_bytes = base64.b64decode(content_base64)
                file_data.append((file_name, file_bytes))
            except Exception as e:
                decode_errors.append({
                    "file_name": file_name,
                    "error": f"Failed to decode base64: {str(e)}"
                })

        # Check if we have any data to process
        if not file_data and not text_data:
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

        # Process OCR text inputs
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

        # Add decode errors to failed files
        if decode_errors:
            result["summary"]["failed"] += len(decode_errors)
            result["summary"]["failed_files"].extend(decode_errors)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Initialize output variables
        excel_base64 = None
        excel_filename = None
        csv_balance_base64 = None
        csv_balance_filename = None
        csv_details_base64 = None
        csv_details_filename = None
        excel_bytes = None

        # Get output_format (default: "excel")
        output_format = request.output_format

        # Generate Excel if requested
        if output_format in ("excel", "both"):
            excel_bytes = use_case.export_to_excel(
                result["all_transactions"],
                result["all_balances"]
            )
            excel_filename = f"bank_statements_{session_id}.xlsx"
            if request.return_excel_base64:
                excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')

        # Generate NetSuite CSVs if requested
        if output_format in ("netsuite_csv", "both"):
            # Generate Balance CSV (returns bytes and external_ids mapping)
            balance_bytes, balance_external_ids = use_case.export_to_netsuite_balance_csv(
                result["all_transactions"],
                result["all_balances"]
            )
            csv_balance_filename = f"netsuite_balance_{session_id}.csv"
            csv_balance_base64 = base64.b64encode(balance_bytes).decode('utf-8')

            # Generate Details CSV (uses balance_external_ids for linking)
            details_bytes = use_case.export_to_netsuite_details_csv(
                result["all_transactions"],
                balance_external_ids
            )
            csv_details_filename = f"netsuite_details_{session_id}.csv"
            csv_details_base64 = base64.b64encode(details_bytes).decode('utf-8')

        # Store Excel for download (for backward compatibility)
        if excel_bytes:
            _file_storage[session_id] = {
                "content": excel_bytes,
                "filename": excel_filename
            }

        # Convert to response schema
        statements_response = []
        for stmt in result["statements"]:
            # Convert balance
            balance_response = None
            if stmt.balance:
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
                    closing_balance=closing_bal
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

        # Prepare response
        response = PowerAutomateParseResponse(
            success=result["summary"]["successful"] > 0,
            message=f"Processed {result['summary']['successful']} of {result['summary']['total_files']} files successfully",
            summary=result["summary"],
            statements=statements_response,
            excel_base64=excel_base64,
            excel_filename=excel_filename,
            csv_balance_base64=csv_balance_base64,
            csv_balance_filename=csv_balance_filename,
            csv_details_base64=csv_details_base64,
            csv_details_filename=csv_details_filename,
            download_url=f"/api/finance/bank-statements/download/{session_id}" if excel_bytes else None
        )

        return response

    except Exception as e:
        return PowerAutomateParseResponse(
            success=False,
            message=f"Failed to parse bank statements: {str(e)}",
            summary={"total_files": len(request.files), "successful": 0, "failed": len(request.files)}
        )
