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
from typing import List, Optional
from io import BytesIO

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import StreamingResponse

from app.application.finance.cash_report import CashReportService
from app.application.finance.cash_report.progress_store import progress_store, ProgressEvent
from app.infrastructure.database.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user, get_ai_usage_repository
from app.infrastructure.database.models.user import UserModel
from app.infrastructure.persistence.repositories.ai_usage_repository import AIUsageRepository
from app.shared.utils.logging_config import get_logger
from app.shared.utils.ai_usage_tracker import log_ai_usage

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
            "Hybrid classification: rule-based keywords first, AI for leftovers",
            "Review/confirm classifications before writing to Excel",
            "Settlement automation (tất toán) - close saving accounts",
            "Open-new automation (mở mới) - open saving accounts",
            "Preserves Excel formulas and formatting",
        ],
        "endpoints": {
            "POST /init-session": "Initialize a new session",
            "POST /upload-statements/{session_id}": "Upload + classify + write (auto-confirm)",
            "POST /upload-preview/{session_id}": "Upload + classify + preview (review before writing)",
            "POST /confirm-upload/{session_id}": "Confirm and write pending classifications to Excel",
            "POST /run-settlement/{session_id}": "Run settlement automation (tất toán)",
            "POST /run-open-new/{session_id}": "Run open-new automation (mở mới)",
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
    current_user: UserModel = Depends(get_current_user),
):
    """
    Initialize or get existing cash report automation session.

    Only ONE session is allowed per user. If an active session exists,
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
            user_id=current_user.id,
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
    current_user: UserModel = Depends(get_current_user),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
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
            user_id=current_user.id,
        )

        # Log AI usage for Gemini classification
        ai_usage = result.get("ai_usage")
        logger.info(f"AI usage data from classifier: {ai_usage}")
        if ai_usage and ai_usage.get("total_tokens", 0) > 0:
            await log_ai_usage(
                ai_usage_repo,
                provider=ai_usage.get("provider", "gemini"),
                model_name=ai_usage.get("model", "gemini-2.0-flash"),
                task_type="classification",
                input_tokens=ai_usage.get("input_tokens", 0),
                output_tokens=ai_usage.get("output_tokens", 0),
                processing_time_ms=ai_usage.get("processing_time_ms", 0),
                task_description="Cash report transaction classification",
                file_name=", ".join([f.filename for f in files]),
                file_count=len(files),
                session_id=session_id,
                user_id=current_user.id,
            )

        # Delay cleanup so SSE generator has time to read final events from queue
        await asyncio.sleep(1)
        progress_store.cleanup(session_id)

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        raise HTTPException(status_code=403, detail=str(e))
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


@router.post("/upload-preview/{session_id}")
async def upload_and_preview(
    session_id: str,
    files: List[UploadFile] = File(..., description="Parsed bank statement Excel files"),
    filter_by_date: bool = Form(default=True, description="Filter transactions by session date range"),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Upload files, classify transactions (rule-based + AI), and return preview for review.
    Does NOT write to Excel yet. Call POST /confirm-upload/{session_id} to confirm and write.

    Classification flow:
    1. Rule-based keyword matching (fast, deterministic)
    2. AI (Gemini) only for transactions that rule-based couldn't classify
    3. Returns preview with each transaction's nature and classification source (rule/ai)
    """
    try:
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename, content))

        progress_store.create(session_id)

        def on_progress(event: ProgressEvent):
            progress_store.emit(session_id, event)

        service = CashReportService(db_session=db)
        result = await service.upload_and_preview(
            session_id=session_id,
            files=file_data,
            filter_by_date=filter_by_date,
            progress_callback=on_progress,
            user_id=current_user.id,
        )

        # Log AI usage
        ai_usage = result.get("ai_usage")
        if ai_usage and ai_usage.get("total_tokens", 0) > 0:
            await log_ai_usage(
                ai_usage_repo,
                provider=ai_usage.get("provider", "gemini"),
                model_name=ai_usage.get("model", "gemini-2.0-flash"),
                task_type="classification",
                input_tokens=ai_usage.get("input_tokens", 0),
                output_tokens=ai_usage.get("output_tokens", 0),
                processing_time_ms=ai_usage.get("processing_time_ms", 0),
                task_description="Cash report transaction classification (preview)",
                file_name=", ".join([f.filename for f in files]),
                file_count=len(files),
                session_id=session_id,
                user_id=current_user.id,
            )

        await asyncio.sleep(1)
        progress_store.cleanup(session_id)

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        progress_store.emit(session_id, ProgressEvent(
            event_type="error", step="error",
            message=str(e), percentage=0,
        ))
        await asyncio.sleep(0.5)
        progress_store.cleanup(session_id)
        logger.error(f"Error in upload preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confirm-upload/{session_id}")
async def confirm_upload(
    session_id: str,
    modifications: Optional[str] = Form(default=None, description='JSON array of {index, nature} to override classifications'),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Confirm and write pending classifications to Excel.

    Call this after reviewing the preview from POST /upload-preview/{session_id}.
    Optionally pass modifications to override specific transaction classifications.

    Args:
        modifications: JSON string like [{"index": 0, "nature": "Operating expense"}, ...]
    """
    try:
        mods = None
        if modifications:
            mods = json.loads(modifications)

        service = CashReportService(db_session=db)
        result = await service.confirm_classifications(
            session_id=session_id,
            modifications=mods,
            user_id=current_user.id,
        )

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid modifications JSON format")
    except Exception as e:
        logger.error(f"Error confirming upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-settlement/{session_id}")
async def run_settlement(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Run settlement (tất toán) automation on Movement data.

    This detects saving account settlement transactions and creates counter entries:
    - Detects "tất toán" (close/settle) transactions
    - Creates counter entries for internal transfers
    - Appends counter entries to Movement sheet

    Returns summary of counter entries created.
    Connect to GET /settlement-progress/{session_id} BEFORE calling this for real-time progress.
    """
    try:
        # Create progress queue for SSE streaming
        progress_store.create(f"settlement-{session_id}")

        def on_progress(event: ProgressEvent):
            progress_store.emit(f"settlement-{session_id}", event)

        service = CashReportService(db_session=db)
        result = await service.run_settlement_automation(
            session_id,
            user_id=current_user.id,
            progress_callback=on_progress,
        )

        # Delay cleanup so SSE generator has time to read final events
        await asyncio.sleep(1)
        progress_store.cleanup(f"settlement-{session_id}")

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"settlement-{session_id}")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"settlement-{session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        progress_store.emit(f"settlement-{session_id}", ProgressEvent(
            event_type="error", step="error",
            message=str(e), percentage=0,
        ))
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"settlement-{session_id}")
        logger.error(f"Error running settlement automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settlement-progress/{session_id}")
async def stream_settlement_progress(session_id: str):
    """
    Stream settlement progress events using Server-Sent Events (SSE).

    Connect to this endpoint BEFORE firing the settlement POST request.
    Events stream in real-time as the backend processes settlement steps.
    """

    def generate():
        queue_key = f"settlement-{session_id}"
        max_wait = 10
        waited = 0.0
        while progress_store.get(queue_key) is None and waited < max_wait:
            time.sleep(0.2)
            waited += 0.2
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Waiting for settlement to start...'})}\n\n"

        progress_queue = progress_store.get(queue_key)
        if not progress_queue:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Settlement session not found'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to settlement progress stream'})}\n\n"

        last_heartbeat = time.time()
        last_real_event = time.time()

        while True:
            try:
                message = progress_queue.get(timeout=0.1)
                last_heartbeat = time.time()
                last_real_event = time.time()

                event_data = json.loads(message)
                yield f"data: {message}\n\n"

                if event_data.get("type") in ("complete", "error"):
                    break

            except queue.Empty:
                current_time = time.time()
                if current_time - last_heartbeat >= 2.0:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = current_time

                if current_time - last_real_event >= 300:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Settlement progress stream timed out'})}\n\n"
                    break

            except Exception as e:
                logger.error(f"Error in settlement progress stream: {e}")
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


@router.post("/run-open-new/{session_id}")
async def run_open_new(
    session_id: str,
    lookup_files: List[UploadFile] = File(default=[], description="Optional lookup files (VTB Saving style .xls/.xlsx) for account matching"),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Run "mở mới" (open new saving account) automation on Movement data.

    This detects transactions that open/deposit into saving accounts and creates counter entries:
    - Detects GROUP B patterns (Gửi tiền, Mở HDTG, etc.) with Nature = "Internal transfer out"
    - Extracts saving account from description OR lookups from uploaded file(s)
    - Creates counter entries with Nature = "Internal transfer in"
    - Appends counter entries to Movement sheet

    Optionally upload one or more lookup files (VTB Saving style) to match accounts when
    description doesn't contain the saving account number.

    Returns summary of counter entries created.
    Connect to GET /open-new-progress/{session_id} BEFORE calling this for real-time progress.
    """
    try:
        # Read lookup files if provided
        lookup_contents = []
        for lf in lookup_files:
            if lf.filename:  # Skip empty file entries
                content = await lf.read()
                if content:
                    lookup_contents.append(content)

        # Create progress queue for SSE streaming
        progress_store.create(f"open-new-{session_id}")

        def on_progress(event: ProgressEvent):
            progress_store.emit(f"open-new-{session_id}", event)

        service = CashReportService(db_session=db)
        result = await service.run_open_new_automation(
            session_id,
            lookup_file_contents=lookup_contents or None,
            user_id=current_user.id,
            progress_callback=on_progress,
        )

        # Delay cleanup so SSE generator has time to read final events
        await asyncio.sleep(1)
        progress_store.cleanup(f"open-new-{session_id}")

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"open-new-{session_id}")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"open-new-{session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        progress_store.emit(f"open-new-{session_id}", ProgressEvent(
            event_type="error", step="error",
            message=str(e), percentage=0,
        ))
        await asyncio.sleep(0.5)
        progress_store.cleanup(f"open-new-{session_id}")
        logger.error(f"Error running open-new automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/open-new-progress/{session_id}")
async def stream_open_new_progress(session_id: str):
    """
    Stream open-new progress events using Server-Sent Events (SSE).

    Connect to this endpoint BEFORE firing the open-new POST request.
    Events stream in real-time as the backend processes open-new steps.
    """

    def generate():
        queue_key = f"open-new-{session_id}"
        max_wait = 10
        waited = 0.0
        while progress_store.get(queue_key) is None and waited < max_wait:
            time.sleep(0.2)
            waited += 0.2
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Waiting for open-new to start...'})}\n\n"

        progress_queue = progress_store.get(queue_key)
        if not progress_queue:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Open-new session not found'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to open-new progress stream'})}\n\n"

        last_heartbeat = time.time()
        last_real_event = time.time()

        while True:
            try:
                message = progress_queue.get(timeout=0.1)
                last_heartbeat = time.time()
                last_real_event = time.time()

                event_data = json.loads(message)
                yield f"data: {message}\n\n"

                if event_data.get("type") in ("complete", "error"):
                    break

            except queue.Empty:
                current_time = time.time()
                if current_time - last_heartbeat >= 2.0:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = current_time

                if current_time - last_real_event >= 300:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Open-new progress stream timed out'})}\n\n"
                    break

            except Exception as e:
                logger.error(f"Error in open-new progress stream: {e}")
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


@router.get("/session/{session_id}")
async def get_session_status(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
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
        result = await service.get_session_status(session_id, user_id=current_user.id)

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            **result,
        }

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{session_id}")
async def download_session_result(
    session_id: str,
    step: Optional[str] = Query(default=None, description="Step name: settlement, open_new. If omitted returns latest."),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Download the working Excel file for a session.

    Pass ``?step=settlement`` or ``?step=open_new`` to download the snapshot
    from that specific step rather than the latest version.
    """
    try:
        service = CashReportService(db_session=db)
        file_path = await service.get_working_file_path(
            session_id, user_id=current_user.id, step=step,
        )

        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Session file not found")

        # Read and sanitize: clean all worksheet XMLs to prevent
        # "Removed Records" errors regardless of which code path produced the file
        import zipfile, io as _io
        from app.application.finance.cash_report.openpyxl_handler import OpenpyxlHandler

        with open(file_path, 'rb') as f:
            raw = f.read()

        buf = _io.BytesIO()
        with zipfile.ZipFile(_io.BytesIO(raw), "r") as src:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
                for entry in src.namelist():
                    data = src.read(entry)
                    if entry.startswith("xl/worksheets/") and entry.endswith(".xml"):
                        data = OpenpyxlHandler._sanitize_worksheet_xml_for_download(data)
                    dst.writestr(entry, data)
        content = buf.getvalue()

        suffix = f"_{step}" if step else ""
        return StreamingResponse(
            BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=Cash_Report_{session_id[:8]}{suffix}.xlsx"
            }
        )

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading session file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{session_id}")
async def reset_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Reset a session to clean state.

    This clears all uploaded data but preserves the session configuration.
    """
    try:
        service = CashReportService(db_session=db)
        result = await service.reset_session(session_id, user_id=current_user.id)

        return {
            "success": True,
            **result,
            "message": "Session reset successfully",
        }

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Delete a session and its working files.
    """
    try:
        service = CashReportService(db_session=db)
        deleted = await service.delete_session(session_id, user_id=current_user.id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "message": "Session deleted successfully",
        }

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    List active automation sessions for the current user.
    """
    try:
        service = CashReportService(db_session=db)
        sessions = await service.list_sessions(user_id=current_user.id)

        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions),
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-test")
async def run_test_automation(
    lookup_files: List[UploadFile] = File(default=[], description="Optional lookup files for open-new account matching"),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Run settlement + open-new automation using the test template (cash_report_for_test.xlsx).
    No AI calls needed — natures are pre-classified.

    Optionally upload lookup files (VTB Saving style) for open-new account matching,
    same as the production open-new endpoint.

    Creates a temporary test session, runs both automations, returns combined results.
    The result file can be downloaded via GET /download-test/{test_session_id}.
    """
    try:
        # Read lookup files if provided (same as production open-new endpoint)
        lookup_contents = []
        for lf in lookup_files:
            if lf.filename:
                content = await lf.read()
                if content:
                    lookup_contents.append(content)

        progress_store.create("test-automation")

        def on_progress(event: ProgressEvent):
            progress_store.emit("test-automation", event)

        service = CashReportService(db_session=None)
        result = await service.run_test_automation(
            lookup_file_contents=lookup_contents or None,
            progress_callback=on_progress,
        )

        await asyncio.sleep(1)
        progress_store.cleanup("test-automation")

        return {
            "success": True,
            **result,
        }

    except FileNotFoundError as e:
        await asyncio.sleep(0.5)
        progress_store.cleanup("test-automation")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        progress_store.emit("test-automation", ProgressEvent(
            event_type="error", step="error",
            message=str(e), percentage=0,
        ))
        await asyncio.sleep(0.5)
        progress_store.cleanup("test-automation")
        logger.error(f"Error running test automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-progress")
async def stream_test_progress():
    """Stream test automation progress events using SSE."""

    def generate():
        queue_key = "test-automation"
        max_wait = 10
        waited = 0.0
        while progress_store.get(queue_key) is None and waited < max_wait:
            time.sleep(0.2)
            waited += 0.2
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Waiting for test to start...'})}\n\n"

        progress_queue = progress_store.get(queue_key)
        if not progress_queue:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Test session not found'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to test progress stream'})}\n\n"

        last_heartbeat = time.time()
        last_real_event = time.time()

        while True:
            try:
                message = progress_queue.get(timeout=0.1)
                last_heartbeat = time.time()
                last_real_event = time.time()

                event_data = json.loads(message)
                yield f"data: {message}\n\n"

                if event_data.get("type") in ("complete", "error"):
                    break

            except queue.Empty:
                current_time = time.time()
                if current_time - last_heartbeat >= 2.0:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = current_time

                if current_time - last_real_event >= 300:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Test progress timed out'})}\n\n"
                    break

            except Exception as e:
                logger.error(f"Error in test progress stream: {e}")
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


@router.get("/download-test/{test_session_id}")
async def download_test_result(
    test_session_id: str,
    current_user: UserModel = Depends(get_current_user),
):
    """Download the test automation result file."""
    try:
        service = CashReportService(db_session=None)
        file_path = await service.get_working_file_path(test_session_id, user_id=None)

        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Test session file not found")

        # Sanitize worksheet XML before download
        import zipfile, io as _io
        from app.application.finance.cash_report.openpyxl_handler import OpenpyxlHandler

        with open(file_path, 'rb') as f:
            raw = f.read()

        buf = _io.BytesIO()
        with zipfile.ZipFile(_io.BytesIO(raw), "r") as src:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
                for entry in src.namelist():
                    data = src.read(entry)
                    if entry.startswith("xl/worksheets/") and entry.endswith(".xml"):
                        data = OpenpyxlHandler._sanitize_worksheet_xml_for_download(data)
                    dst.writestr(entry, data)
        content = buf.getvalue()

        return StreamingResponse(
            BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=Cash_Report_Test_{test_session_id}.xlsx"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading test result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview/{session_id}")
async def preview_movement_data(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of rows to preview"),
    db: AsyncSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Preview Movement sheet data for a session.

    Useful for checking uploaded data before downloading.
    """
    try:
        service = CashReportService(db_session=db)
        data = await service.get_data_preview(session_id, limit, user_id=current_user.id)

        return {
            "success": True,
            "session_id": session_id,
            "preview_rows": data,
            "rows_shown": len(data),
        }

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats(
    current_user: UserModel = Depends(get_current_user),
):
    """
    Cache stats endpoint kept for backward compatibility.
    Cash Report no longer uses Redis caching.
    """
    return {
        "success": True,
        "connected": False,
        "message": "Redis cache is disabled for Cash Report module.",
    }
