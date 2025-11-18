"""API router for bank statement parsing."""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import uuid
import math

from app.presentation.schemas.bank_statement_schemas import (
    ParseBankStatementsResponse,
    BankStatementResponse,
    BankBalanceResponse,
    BankTransactionResponse,
    SupportedBanksResponse
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
            "Parse transactions (Date, Amount, Description, etc.)",
            "Extract opening/closing balances",
            "Batch processing (multiple files)",
            "Standardized Excel output",
            "Support for Vietnamese banks"
        ],
        "supported_banks": ParserFactory.get_supported_banks(),
        "endpoints": {
            "parse": "POST /bank-statements/parse - Parse bank statements (batch)",
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
