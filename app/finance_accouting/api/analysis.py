# app/api/analysis.py
"""Analysis endpoints with enhanced validation."""

import io
import json
import queue
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from ..models.analysis import (
    AnalysisSession, AnalysisConfigRequest, RevenueVarianceAnalysisResponse,
    DebugFilesResponse, ErrorResponse
)
from ..services.analysis_service import analysis_service
from ..utils.helpers import build_config_overrides
from ..utils.file_validation import validate_file_list, FileValidator
from ..utils.input_sanitization import validate_analysis_parameters, sanitize_session_id
from ..core.config import get_settings, Settings
from ..core.exceptions import FileProcessingError, ValidationError, SessionError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

@router.post("/process")
async def process_python_analysis(
    request: Request,
    excel_files: List[UploadFile] = File(...),
    mapping_file: Optional[UploadFile] = File(None),
    materiality_vnd: Optional[float] = Form(None),
    recurring_pct_threshold: Optional[float] = Form(None),
    revenue_opex_pct_threshold: Optional[float] = Form(None),
    bs_pct_threshold: Optional[float] = Form(None),
    recurring_code_prefixes: Optional[str] = Form(None),
    min_trend_periods: Optional[int] = Form(None),
    gm_drop_threshold_pct: Optional[float] = Form(None),
    dep_pct_only_prefixes: Optional[str] = Form(None),
    customer_column_hints: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings)
):
    """Process Excel files using Python-based analysis with comprehensive validation."""
    logger.info(f"Processing {len(excel_files)} files for Python analysis")

    try:
        # 1. Validate uploaded files
        validation_results = await validate_file_list(excel_files, max_files=10)

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

        # 2. Validate and sanitize analysis parameters
        params = {
            'materiality_vnd': materiality_vnd,
            'recurring_pct_threshold': recurring_pct_threshold,
            'revenue_opex_pct_threshold': revenue_opex_pct_threshold,
            'bs_pct_threshold': bs_pct_threshold,
            'recurring_code_prefixes': recurring_code_prefixes,
            'min_trend_periods': min_trend_periods,
            'gm_drop_threshold_pct': gm_drop_threshold_pct,
            'dep_pct_only_prefixes': dep_pct_only_prefixes,
            'customer_column_hints': customer_column_hints,
        }

        validated_params = validate_analysis_parameters(params)

        # Build configuration overrides
        config_overrides = build_config_overrides(
            materiality_vnd=validated_params.materiality_vnd,
            recurring_pct_threshold=validated_params.recurring_pct_threshold,
            revenue_opex_pct_threshold=validated_params.revenue_opex_pct_threshold,
            bs_pct_threshold=validated_params.bs_pct_threshold,
            recurring_code_prefixes=validated_params.recurring_code_prefixes,
            min_trend_periods=validated_params.min_trend_periods,
            gm_drop_threshold_pct=validated_params.gm_drop_threshold_pct,
            dep_pct_only_prefixes=validated_params.dep_pct_only_prefixes,
            customer_column_hints=validated_params.customer_column_hints,
        )

        # 3. Process files
        xlsx_bytes = await analysis_service.process_python_analysis(
            excel_files=excel_files,
            mapping_file=mapping_file,
            config_overrides=config_overrides
        )

        logger.info("Python analysis completed successfully")
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

    logger.info(f"Streaming logs for session: {clean_session_id}")

    def generate():
        if clean_session_id not in analysis_service.log_streams:
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
            return

        log_queue = analysis_service.log_streams[session_id]
        yield f"data: {json.dumps({'type': 'log', 'message': '📡 SSE connection established'})}\n\n"

        while True:
            try:
                message = log_queue.get(timeout=1)

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
                        except ValueError:
                            yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"

            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                continue

        # Cleanup
        if session_id in analysis_service.log_streams:
            del analysis_service.log_streams[session_id]

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

