"""
Cash Report Router - API endpoints for cash report automation.
Uses master template approach with COM automation for Excel.
"""
import asyncio
import json
import queue
import time
from datetime import date
from decimal import Decimal
from typing import List
from io import BytesIO

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import StreamingResponse

from app.application.finance.cash_report import CashReportService
from app.application.finance.cash_report.progress_store import progress_store, ProgressEvent
from app.infrastructure.database.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/cash-report",
    tags=["Finance - Cash Report"],
)


@router.get("/")
async def get_info():
    """Get information about the Cash Report API"""
    return {
        "service": "Cash Report Automation",
        "version": "2.0.0",
        "description": "Automates biweekly cash report generation using master template",
        "features": [
            "Session-based workflow with master template",
            "Upload parsed bank statements",
            "AI-powered transaction classification",
            "Settlement automation (tất toán)",
            "Preserves Excel formulas and formatting",
            "Windows COM automation for full Excel compatibility",
        ],
        "endpoints": {
            "POST /init-session": "Initialize a new session",
            "POST /upload-statements/{session_id}": "Upload bank statements",
            "POST /run-settlement/{session_id}": "Run settlement automation",
            "GET /session/{session_id}": "Get session status",
            "GET /download/{session_id}": "Download result",
            "POST /reset/{session_id}": "Reset session",
            "DELETE /session/{session_id}": "Delete session",
        },
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "cash-report"}


@router.post("/init-session")
async def init_session(
    opening_date: date = Form(..., description="Report period start date"),
    ending_date: date = Form(..., description="Report period end date"),
    fx_rate: float = Form(default=26175, description="VND/USD exchange rate"),
    period_name: str = Form(default="", description="Period name (e.g., W3-4Jan26)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Initialize or get existing cash report automation session.

    Only ONE session is allowed at a time. If an active session exists,
    returns that session instead of creating a new one.

    Returns a session_id to use for subsequent operations.
    """
    try:
        service = CashReportService(db_session=db)
        result = await service.get_or_create_session(
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=Decimal(str(fx_rate)),
            period_name=period_name,
        )

        is_existing = result.get("is_existing", False)
        return {
            "success": True,
            "session_id": result["session_id"],
            "config": result["config"],
            "is_existing": is_existing,
            "movement_rows": result.get("movement_rows", 0),
            "message": "Using existing session" if is_existing else "Session initialized successfully",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Master template not found: {e}")
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-statements/{session_id}")
async def upload_bank_statements(
    session_id: str,
    files: List[UploadFile] = File(..., description="Parsed bank statement Excel files"),
    filter_by_date: bool = Form(default=True, description="Filter transactions by session date range"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload parsed bank statement files to a session.

    This endpoint:
    1. Reads transactions from uploaded Excel files (output from bank statement parser)
    2. Filters by date range if enabled
    3. Classifies transactions using AI
    4. Appends to the Movement sheet (accumulative)

    Can be called multiple times to add more data.
    """
    try:
        # Read file contents
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename, content))

        # Create progress queue for SSE streaming
        progress_store.create(session_id)

        def on_progress(event: ProgressEvent):
            progress_store.emit(session_id, event)

        service = CashReportService(db_session=db)
        result = await service.upload_bank_statements(
            session_id=session_id,
            files=file_data,
            filter_by_date=filter_by_date,
            progress_callback=on_progress,
        )

        # Delay cleanup so SSE generator has time to read final events from queue
        await asyncio.sleep(1)
        progress_store.cleanup(session_id)

        return {
            "success": True,
            **result,
        }

    except ValueError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Emit error event before cleanup
        progress_store.emit(session_id, ProgressEvent(
            event_type="error", step="error",
            message=str(e), percentage=0,
        ))
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        logger.error(f"Error uploading statements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upload-progress/{session_id}")
async def stream_upload_progress(session_id: str):
    """
    Stream upload progress events using Server-Sent Events (SSE).

    Connect to this endpoint BEFORE firing the upload POST request.
    Events stream in real-time as the backend processes files.
    """

    def generate():
        # Wait for queue to be created (POST might not have started yet)
        max_wait = 10  # seconds
        waited = 0.0
        while progress_store.get(session_id) is None and waited < max_wait:
            time.sleep(0.2)
            waited += 0.2
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Waiting for upload to start...'})}\n\n"

        progress_queue = progress_store.get(session_id)
        if not progress_queue:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Upload session not found'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to progress stream'})}\n\n"

        last_heartbeat = time.time()
        last_real_event = time.time()  # Track last real (non-heartbeat) event

        while True:
            try:
                message = progress_queue.get(timeout=0.1)
                last_heartbeat = time.time()
                last_real_event = time.time()

                # message is already a JSON string from ProgressEvent.to_json()
                event_data = json.loads(message)
                yield f"data: {message}\n\n"

                # End stream on complete or error
                if event_data.get("type") in ("complete", "error"):
                    break

            except queue.Empty:
                current_time = time.time()
                if current_time - last_heartbeat >= 2.0:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = current_time

                # Safety timeout: 5 minutes without any real event
                if current_time - last_real_event >= 300:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Progress stream timed out'})}\n\n"
                    break

            except Exception as e:
                logger.error(f"Error in progress stream: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run-settlement/{session_id}")
async def run_settlement(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Run settlement (tất toán) automation on Movement data.

    This detects saving account settlement transactions and creates counter entries:
    - Detects "tất toán" (close/settle) transactions
    - Creates counter entries for internal transfers
    - Appends counter entries to Movement sheet

    Returns summary of counter entries created.
    """
    try:
        service = CashReportService(db_session=db)
        result = await service.run_settlement_automation(session_id)

        return {
            "success": True,
            **result,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error running settlement automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the status and statistics of a session.

    Returns:
    - Session configuration
    - Number of transactions in Movement sheet
    - List of uploaded files
    - File size
    """
    try:
        service = CashReportService(db_session=db)
        result = await service.get_session_status(session_id)

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{session_id}")
async def download_session_result(session_id: str):
    """
    Download the working Excel file for a session.

    Returns the Excel file with all uploaded transactions in the Movement sheet.
    """
    try:
        service = CashReportService()
        file_path = service.get_working_file_path(session_id)

        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Session file not found")

        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()

        return StreamingResponse(
            BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=Cash_Report_{session_id[:8]}.xlsx"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading session file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{session_id}")
async def reset_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Reset a session to clean state.

    This clears all uploaded data but preserves the session configuration.
    """
    try:
        service = CashReportService(db_session=db)
        result = await service.reset_session(session_id)

        return {
            "success": True,
            **result,
            "message": "Session reset successfully",
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a session and its working files.
    """
    try:
        service = CashReportService(db_session=db)
        deleted = await service.delete_session(session_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "message": "Session deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
):
    """
    List all active automation sessions.
    """
    try:
        service = CashReportService(db_session=db)
        sessions = await service.list_sessions()

        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions),
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview/{session_id}")
async def preview_movement_data(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of rows to preview"),
):
    """
    Preview Movement sheet data for a session.

    Useful for checking uploaded data before downloading.
    """
    try:
        service = CashReportService()
        data = service.get_data_preview(session_id, limit)

        return {
            "success": True,
            "session_id": session_id,
            "preview_rows": data,
            "rows_shown": len(data),
        }

    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
