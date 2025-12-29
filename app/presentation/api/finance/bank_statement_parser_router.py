"""API router for bank statement parsing."""

from typing import List, Tuple, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from fastapi.responses import Response
import uuid
import math
import base64
import re
from io import BytesIO
import pandas as pd  # Added pandas for HTML handling

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
from app.application.finance.bank_statement_parser.bank_statement_db_service import BankStatementDbService
from app.core.dependencies import get_bank_statement_db_service, get_ai_usage_repository
from app.infrastructure.persistence.repositories import AIUsageRepository
from app.shared.utils.logging_config import get_logger
from uuid import UUID as PyUUID

logger = get_logger(__name__)

router = APIRouter(prefix="/bank-statements", tags=["Finance - Bank Statement Parser"])

# In-memory storage for downloaded files (session-based)
_file_storage = {}


def _generate_export_filename(transactions: list, balances: list) -> str:
    """Generate export filename with format: {bank_codes}_statement_{timestamp}.xlsx"""
    from datetime import datetime

    # Collect unique bank names from transactions and balances
    bank_names = set()
    for tx in transactions:
        if hasattr(tx, 'bank_name') and tx.bank_name:
            bank_names.add(tx.bank_name.upper())
    for bal in balances:
        if hasattr(bal, 'bank_name') and bal.bank_name:
            bank_names.add(bal.bank_name.upper())

    # Sort and join bank names (limit to first 3 if too many)
    sorted_banks = sorted(bank_names)
    if len(sorted_banks) > 3:
        bank_code = "_".join(sorted_banks[:3]) + "_etc"
    elif sorted_banks:
        bank_code = "_".join(sorted_banks)
    else:
        bank_code = "UNKNOWN"

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{bank_code}_statement_{timestamp}.xlsx"


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


@router.post("/parse", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements (Batch)")
async def parse_bank_statements(
    files: List[UploadFile] = File(..., description="Bank statement Excel files (.xlsx, .xls)"),
    project_uuid: Optional[str] = Form(None, description="Project UUID to save statements to (optional)"),
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """
    Parse multiple bank statement files in batch.
    ... (giữ nguyên docstring)
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
            for i, (filename, content) in enumerate(file_data):
                await db_service.save_file_upload(
                    filename=filename,
                    file_size=len(content),
                    file_content=content,  # Save actual file to disk
                    file_type="bank_statement",
                    session_id=session_id,
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if filename.endswith('.xlsx') else "application/vnd.ms-excel",
                    processing_status="completed",
                    metadata={"source": "parse-excel", "project_uuid": project_uuid},
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
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse bank statements: {str(e)}")


@router.post("/parse-pdf", response_model=ParseBankStatementsResponse, summary="Parse Bank Statements from PDF (Gemini OCR)")
async def parse_bank_statements_pdf(
    files: List[UploadFile] = File(..., description="Bank statement PDF files (.pdf)"),
    bank_codes: Optional[str] = Form(None, description="Comma-separated bank codes for each file (e.g., 'VIB,ACB,VCB'). Leave empty for auto-detection."),
    passwords: Optional[str] = Form(None, description="Comma-separated passwords for encrypted PDF files (e.g., 'pass1,,pass3'). Use empty string for non-encrypted PDFs."),
    project_uuid: Optional[str] = Form(None, description="Project UUID to save statements to (optional)"),
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Parse multiple bank statement PDF files using Gemini Flash OCR.
    ... (giữ nguyên docstring)
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Parse bank_codes if provided
        bank_code_list = []
        if bank_codes:
            bank_code_list = [code.strip() if code.strip() else None for code in bank_codes.split(",")]

        # Parse passwords if provided
        password_list = []
        if passwords:
            password_list = [pwd.strip() if pwd.strip() else None for pwd in passwords.split(",")]

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
            password = password_list[i] if i < len(password_list) else None
            pdf_inputs.append((file.filename, content, bank_code, password))

        # Parse using use case
        use_case = ParseBankStatementsUseCase()
        result = use_case.execute_from_pdf(pdf_inputs)

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
            for i, (filename, content, bank_code, password) in enumerate(pdf_inputs):
                await db_service.save_file_upload(
                    filename=filename,
                    file_size=len(content),
                    file_content=content,  # Save actual file to disk
                    file_type="bank_statement",
                    session_id=session_id,
                    content_type="application/pdf",
                    processing_status="completed",
                    metadata={"source": "parse-pdf", "bank_code": bank_code, "project_uuid": project_uuid},
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
            bank_names = sorted(set(stmt.bank_name.upper() for stmt in statements if stmt.bank_name))
            if len(bank_names) > 3:
                bank_code = "_".join(bank_names[:3]) + "_etc"
            elif bank_names:
                bank_code = "_".join(bank_names)
            else:
                bank_code = "UNKNOWN"
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{bank_code}_statement_{timestamp}.xlsx"
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
    retention_days: int = 30,
    db_service: BankStatementDbService = Depends(get_bank_statement_db_service),
):
    """Manually trigger cleanup of old uploaded files."""
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

        # Prepare response
        response = PowerAutomateParseResponse(
            success=result["summary"]["successful"] > 0,
            message=f"Processed {result['summary']['successful']} of {result['summary']['total_files']} files successfully",
            summary=result["summary"],
            excel_base64=excel_base64,
            excel_filename=excel_filename
        )

        return response

    except Exception as e:
        return PowerAutomateParseResponse(
            success=False,
            message=f"Failed to parse bank statements: {str(e)}",
            summary={"total_files": len(request.files), "successful": 0, "failed": len(request.files)}
        )