# app/presentation/api/finance/analysis_router.py
"""Analysis endpoints with enhanced validation."""

import io
import json
import queue
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from app.presentation.schemas.analysis import (
    AnalysisSession, AnalysisConfigRequest, RevenueVarianceAnalysisResponse,
    DebugFilesResponse, ErrorResponse
)
from app.application.finance.variance_analysis.analyze_variance import analysis_service
from app.shared.utils.helpers import build_config_overrides
from app.shared.utils.file_validation import validate_file_list, FileValidator
from app.shared.utils.input_sanitization import validate_analysis_parameters, sanitize_session_id
from app.core.config import get_settings, Settings
from app.core.unified_config import get_unified_config
from app.core.exceptions import FileProcessingError, ValidationError, SessionError
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["analysis"])

@router.post("/process")
async def process_python_analysis(
    request: Request,
    excel_files: List[UploadFile] = File(..., description="BS/PL Breakdown Excel files"),
    loan_interest_file: Optional[UploadFile] = File(None, description="Optional: ERP Loan Interest Rate file for enhanced A2 analysis")
):
    """Process Excel files using Python-based 22-rule variance analysis.

    Upload one or more BS/PL Breakdown Excel files for variance analysis.
    Optionally upload an ERP Loan Interest Rate file (from Save Search) to enable
    enhanced Rule A2 analysis with interest rate lookups.

    Args:
        excel_files: List of BS/PL Breakdown Excel files (required)
        loan_interest_file: Optional ERP Loan Interest Rate file for enhanced A2

    Returns:
        Excel file with variance flags and raw data sheets
    """
    logger.info(f"Processing {len(excel_files)} files for Python variance analysis")
    if loan_interest_file:
        logger.info(f"Loan interest file provided: {loan_interest_file.filename}")

    try:
        # Get unified config to access file_processing settings
        config = get_unified_config()

        # 1. Validate uploaded BS/PL files (use configured max files per request)
        validation_results = await validate_file_list(
            excel_files,
            max_files=config.file_processing.max_files_per_request
        )

        # Check if any files failed validation
        failed_files = [result for result in validation_results if not result.get('is_valid', False)]
        if failed_files:
            error_details = []
            for failed in failed_files:
                error_details.extend(failed.get('errors', []))

            raise FileProcessingError(
                f"File validation failed for {len(failed_files)} file(s)",
                details="; ".join(error_details)
            )

        # 2. Validate loan interest file if provided
        if loan_interest_file:
            loan_validation = await validate_file_list([loan_interest_file], max_files=1)
            if loan_validation and not loan_validation[0].get('is_valid', False):
                raise FileProcessingError(
                    "Loan interest file validation failed",
                    details="; ".join(loan_validation[0].get('errors', []))
                )

        # 3. Process files using 22-rule pipeline with optional loan data
        xlsx_bytes = await analysis_service.process_python_analysis(
            excel_files=excel_files,
            loan_interest_file=loan_interest_file
        )

        logger.info("Python variance analysis completed successfully")
        return StreamingResponse(
            iter([xlsx_bytes]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="variance_analysis_output.xlsx"'}
        )

    except (FileProcessingError, ValidationError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Python analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during analysis. Please try again."
        )

@router.post("/start-analysis", response_model=AnalysisSession)
async def start_ai_analysis(
    request: Request,
    excel_files: List[UploadFile] = File(...)
):
    """Start AI-powered analysis with file validation."""
    logger.info(f"Starting AI analysis for {len(excel_files)} files")

    try:
        # 1. Validate uploaded files
        validation_results = await validate_file_list(excel_files, max_files=5)  # Lower limit for AI

        # Check if any files failed validation
        failed_files = [result for result in validation_results if not result.get('is_valid', False)]
        if failed_files:
            error_details = []
            for failed in failed_files:
                error_details.extend(failed.get('errors', []))

            raise FileProcessingError(
                f"File validation failed for {len(failed_files)} file(s)",
                details="; ".join(error_details)
            )

        # 2. Start AI analysis
        session = await analysis_service.start_ai_analysis(excel_files)

        logger.info(f"AI analysis session started: {session.session_id}")
        return session

    except (FileProcessingError, ValidationError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Failed to start AI analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to start AI analysis. Please check your files and try again."
        )

@router.get("/logs/{session_id}")
async def stream_logs(
    session_id: str,
    request: Request
):
    """Stream analysis logs using Server-Sent Events with validation."""
    # Validate session ID
    clean_session_id = sanitize_session_id(session_id)

    # Check if session exists
    session = analysis_service.get_session(clean_session_id)
    if not session:
        raise SessionError(
            f"Session {clean_session_id} not found",
            session_id=clean_session_id
        )

    logger.info(f"[SSE] Streaming logs for session: {clean_session_id}")
    logger.info(f"[SSE] Available log_streams: {list(analysis_service.log_streams.keys())}")

    def generate():
        logger.info(f"[SSE Generator] Starting for session: {clean_session_id}")
        try:
            if clean_session_id not in analysis_service.log_streams:
                logger.error(f"[SSE Generator] Session {clean_session_id} not found in log_streams!")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
                return

            log_queue = analysis_service.log_streams[clean_session_id]
            logger.info(f"[SSE Generator] Log queue obtained, sending connection message")
            yield f"data: {json.dumps({'type': 'log', 'message': '📡 SSE connection established'})}\n\n"
            logger.info(f"[SSE Generator] Connection message yielded, entering main loop")

            message_count = 0
            import time
            last_heartbeat = time.time()

            while True:
                try:
                    # Use very short timeout (0.1s) and track heartbeats separately
                    message = log_queue.get(timeout=0.1)
                    message_count += 1
                    last_heartbeat = time.time()  # Reset heartbeat timer on message
                    logger.info(f"[SSE Generator] Message #{message_count}: {message[:80]}...")

                    if message == "__ANALYSIS_COMPLETE__":
                        yield f"data: {json.dumps({'type': 'complete', 'message': 'Analysis completed successfully'})}\n\n"
                        break
                    elif message.startswith("__ERROR__"):
                        error_msg = message[9:]
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                        break
                    elif message.startswith("__PROGRESS__"):
                        parts = message.split("__")
                        if len(parts) >= 4:
                            try:
                                percentage = int(parts[2])
                                progress_msg = parts[3]
                                yield f"data: {json.dumps({'type': 'progress', 'percentage': percentage, 'message': progress_msg})}\n\n"
                            except ValueError as ve:
                                logger.error(f"Invalid progress format: {message} - {ve}")
                                yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"

                except queue.Empty:
                    # Send heartbeat every 2 seconds to keep connection alive
                    current_time = time.time()
                    if current_time - last_heartbeat >= 2.0:
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                        last_heartbeat = current_time
                    continue
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Fatal error in SSE generator: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Fatal stream error: {str(e)}'})}\n\n"
        finally:
            # Don't cleanup immediately - polling may still need access to log history
            # Session cleanup will happen via cleanup_old_sessions after expiration
            logger.info(f"SSE stream closed for session: {clean_session_id}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        }
    )

@router.post("/analyze-revenue-variance", response_model=RevenueVarianceAnalysisResponse)
async def analyze_revenue_variance(
    excel_file: UploadFile = File(...)
):
    """Perform comprehensive revenue variance analysis with net effect breakdown."""
    return await analysis_service.analyze_revenue_variance(excel_file)

@router.get("/status/{session_id}")
async def get_analysis_status(session_id: str):
    """Check if analysis is complete and file is ready for download. Also returns recent logs and progress."""
    clean_session_id = sanitize_session_id(session_id)

    session = analysis_service.get_session(clean_session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{clean_session_id}' not found")

    # Check if main result file exists
    main_file_key = f"{clean_session_id}_main_result"
    file_exists = analysis_service.get_file(main_file_key) is not None

    # Get progress from separate tracking (doesn't disrupt queue)
    current_progress = 0
    progress_message = ""
    progress_data = analysis_service.session_progress.get(clean_session_id)
    if progress_data:
        current_progress = progress_data.get("percentage", 0)
        progress_message = progress_data.get("message", "")

    # Get recent logs from history buffer (doesn't disrupt queue)
    recent_logs = []
    log_capture = analysis_service.log_captures.get(clean_session_id)
    if log_capture:
        recent_logs = log_capture.get_recent_logs(count=10)

    return {
        "session_id": clean_session_id,
        "status": session.status,
        "file_ready": file_exists,
        "progress": current_progress,
        "progress_message": progress_message,
        "recent_logs": recent_logs,
        "created_at": session.created_at.isoformat() if hasattr(session.created_at, 'isoformat') else str(session.created_at)
    }

@router.get("/download/{session_id}")
async def download_main_result(session_id: str):
    """Download the main analysis result."""
    main_file_key = f"{session_id}_main_result"
    file_data = analysis_service.get_file(main_file_key)

    if not file_data:
        raise HTTPException(status_code=404, detail=f"Main result for session '{session_id}' not found")

    filename, file_bytes = file_data

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/debug/{file_key}")
async def download_debug_file(file_key: str):
    """Download a debug file by key."""
    file_data = analysis_service.get_file(file_key)

    if not file_data:
        raise HTTPException(status_code=404, detail=f"Debug file '{file_key}' not found")

    filename, file_bytes = file_data

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/debug/list/{session_id}", response_model=DebugFilesResponse)
async def list_debug_files(session_id: str):
    """List all debug files for a session."""
    files = analysis_service.get_debug_files(session_id)
    return DebugFilesResponse(session_id=session_id, files=files)

@router.post("/compare")
async def compare_excel_files(
    request: Request,
    excel_files: List[UploadFile] = File(..., description="Excel files to compare"),
    settings: Settings = Depends(get_settings)
):
    """Compare Excel files - alias for /api/process endpoint."""
    try:
        logger.info(f"Comparing {len(excel_files)} files")
        logger.info(f"Request content-type: {request.headers.get('content-type')}")
        logger.info(f"Files received: {[f.filename for f in excel_files]}")

        # Validate that files are provided
        if not excel_files or len(excel_files) == 0:
            raise FileProcessingError(
                "No files provided for comparison",
                details="The 'excel_files' field is required and must contain at least one file"
            )
    except Exception as e:
        logger.error(f"Error in compare endpoint: {str(e)}", exc_info=True)
        raise

    return await process_python_analysis(
        request=request,
        excel_files=excel_files,
        settings=settings
    )

