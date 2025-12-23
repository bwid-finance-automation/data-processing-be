"""
API Routes for Utility Billing Application with Session Management
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response, Form, Depends
from fastapi.responses import FileResponse
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.presentation.schemas.billing_schemas import (
    FileUploadResponse, ProcessingRequest, ProcessingResponse,
    FileInfo, SystemStatus, MasterDataStatus, SessionResponse,
    ProcessingStats, ValidationIssue
)
from app.application.finance.utility_billing.generate_billing import BillingProcessor
from app.application.project.project_service import ProjectService
from app.core.dependencies import get_db
from app.infrastructure.database.models.analysis_session import AnalysisSessionModel
from app.shared.utils.file_utils import (
    save_upload_file, validate_file_extension,
    get_files_in_directory, get_file_size, delete_file
)
from app.core.config import get_settings
from app.core.session import SessionManager
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])


async def save_billing_to_project(
    db: AsyncSession,
    project_uuid: str,
    session_id: str,
    result: dict
) -> None:
    """
    Save utility billing session to project.

    Args:
        db: Database session
        project_uuid: Project UUID string
        session_id: Billing session ID
        result: Processing result
    """
    try:
        project_service = ProjectService(db)
        case = await project_service.get_or_create_case(UUID(project_uuid), "utility_billing")

        if case:
            stats = result.get('stats', {})

            # Create AnalysisSessionModel record for billing
            analysis_session = AnalysisSessionModel(
                session_id=session_id,
                case_id=case.id,
                status="COMPLETED" if result.get('success') else "FAILED",
                analysis_type="UTILITY_BILLING",
                files_count=stats.get('total_records', 0),
                progress_percentage=100,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                processing_details={
                    "total_invoices": stats.get('total_invoices', 0),
                    "total_line_items": stats.get('total_line_items', 0),
                    "validation_issues_count": stats.get('validation_issues_count', 0),
                    "output_file": result.get('output_file'),
                },
            )
            db.add(analysis_session)

            # Update case file count
            await project_service.increment_case_file_count(case.id)
            await db.commit()

            logger.info(f"Saved utility billing to project {project_uuid}, case {case.id}")
    except Exception as e:
        logger.error(f"Failed to save utility billing to project: {e}")
        await db.rollback()


# === SESSION MANAGEMENT ===

def get_session_id(session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> str:
    """Get and validate session ID from header"""
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required. Call /session/create first.")

    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Update last activity
    SessionManager.update_activity(session_id)
    return session_id


@router.post("/session/create", response_model=SessionResponse)
async def create_session(
    project_uuid: Optional[str] = Form(None, description="Project UUID to associate with session (optional)"),
):
    """Create a new session for file uploads and processing.
    Optionally associate with a project for tracking."""
    session_id = SessionManager.create_session()

    # Store project association if provided
    if project_uuid:
        SessionManager.set_project_uuid(session_id, project_uuid)

    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
        expires_in_minutes=SessionManager.SESSION_TIMEOUT
    )


@router.delete("/session/cleanup")
async def cleanup_session(session_id: str = Header(..., alias="X-Session-ID")):
    """Manually cleanup a session and delete all its files"""
    SessionManager.cleanup_session(session_id)
    return {"message": "Session cleaned up successfully"}


# === FILE UPLOAD ENDPOINTS ===

@router.post("/upload/input", response_model=FileUploadResponse)
async def upload_input_file(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID")
):
    """Upload a CS input file (water/electricity readings)"""
    settings = get_settings()

    # Validate session
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    allowed_extensions = ['.xlsx', '.xls']
    if not validate_file_extension(file.filename, allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Check file size
    content = await file.read()
    max_file_size = 100 * 1024 * 1024  # 100MB
    if len(content) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_file_size / 1024 / 1024}MB"
        )

    # Reset file pointer
    await file.seek(0)

    # Save file to session directory
    input_dir = SessionManager.get_input_dir(session_id)
    destination = input_dir / file.filename
    await save_upload_file(file, destination)

    SessionManager.update_activity(session_id)

    return FileUploadResponse(
        filename=file.filename,
        file_type="input",
        size=len(content),
        uploaded_at=datetime.now(),
        message="Input file uploaded successfully"
    )


@router.post("/upload/master-data", response_model=FileUploadResponse)
async def upload_master_data_file(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID")
):
    """Upload a master data file (Customers, Units, Config)"""

    # Validate session
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    allowed_extensions = ['.xlsx', '.xls']
    if not validate_file_extension(file.filename, allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Check file size
    content = await file.read()
    max_file_size = 100 * 1024 * 1024  # 100MB
    if len(content) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_file_size / 1024 / 1024}MB"
        )

    # Reset file pointer
    await file.seek(0)

    # Save file to session directory
    master_data_dir = SessionManager.get_master_data_dir(session_id)
    destination = master_data_dir / file.filename
    await save_upload_file(file, destination)

    SessionManager.update_activity(session_id)

    return FileUploadResponse(
        filename=file.filename,
        file_type="master_data",
        size=len(content),
        uploaded_at=datetime.now(),
        message="Master data file uploaded successfully"
    )


# === FILE MANAGEMENT ENDPOINTS ===

@router.get("/files/input", response_model=List[FileInfo])
async def list_input_files(session_id: str = Header(..., alias="X-Session-ID")):
    """List all input files for this session"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    input_dir = SessionManager.get_input_dir(session_id)
    allowed_extensions = ['.xlsx', '.xls']
    files = get_files_in_directory(input_dir, allowed_extensions)

    SessionManager.update_activity(session_id)

    return [
        FileInfo(
            filename=f.name,
            size=get_file_size(f),
            uploaded_at=datetime.fromtimestamp(f.stat().st_mtime),
            file_type="input"
        )
        for f in files
    ]


@router.get("/files/master-data", response_model=List[FileInfo])
async def list_master_data_files(session_id: str = Header(..., alias="X-Session-ID")):
    """List all master data files for this session"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    master_data_dir = SessionManager.get_master_data_dir(session_id)
    allowed_extensions = ['.xlsx', '.xls']
    files = get_files_in_directory(master_data_dir, allowed_extensions)

    SessionManager.update_activity(session_id)

    return [
        FileInfo(
            filename=f.name,
            size=get_file_size(f),
            uploaded_at=datetime.fromtimestamp(f.stat().st_mtime),
            file_type="master_data"
        )
        for f in files
    ]


@router.get("/files/output", response_model=List[FileInfo])
async def list_output_files(session_id: str = Header(..., alias="X-Session-ID")):
    """List all generated output files for this session (CSV + validation reports)"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    output_dir = SessionManager.get_output_dir(session_id)

    # Get all output files (CSV + Excel validation reports)
    files = get_files_in_directory(output_dir, ['.csv', '.xlsx'])

    SessionManager.update_activity(session_id)

    return [
        FileInfo(
            filename=f.name,
            size=get_file_size(f),
            uploaded_at=datetime.fromtimestamp(f.stat().st_mtime),
            file_type="output"
        )
        for f in files
    ]


@router.delete("/files/{file_type}/{filename}")
async def delete_file_endpoint(
    file_type: str,
    filename: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """Delete a file from this session"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if file_type == "input":
        file_path = SessionManager.get_input_dir(session_id) / filename
    elif file_type == "master-data":
        file_path = SessionManager.get_master_data_dir(session_id) / filename
    elif file_type == "output":
        file_path = SessionManager.get_output_dir(session_id) / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    success = delete_file(file_path)
    SessionManager.update_activity(session_id)

    if success:
        return {"message": f"File {filename} deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete file")


@router.get("/files/download/{file_type}/{filename}")
async def download_file(
    file_type: str,
    filename: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """Download a file from this session"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if file_type == "output":
        file_path = SessionManager.get_output_dir(session_id) / filename
    elif file_type == "validation":
        file_path = SessionManager.get_log_dir(session_id) / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    SessionManager.update_activity(session_id)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# === PROCESSING ENDPOINTS ===

@router.post("/process", response_model=ProcessingResponse)
async def process_billing(
    session_id: str = Header(..., alias="X-Session-ID"),
    request: ProcessingRequest = None,
    db: AsyncSession = Depends(get_db),
):
    """Process utility billing files and generate ERP CSV.
    If session is associated with a project, saves to project."""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    try:
        # Create processor with session directories
        processor = BillingProcessor(session_id)

        # Process with optional specific files
        result = processor.process_billing(
            specific_files=request.input_files if request else None
        )

        SessionManager.update_activity(session_id)

        # Save to project if session is associated with a project
        project_uuid = SessionManager.get_project_uuid(session_id)
        if project_uuid and result.get('success'):
            await save_billing_to_project(db, project_uuid, session_id, result)

        if result['success']:
            # Convert stats dictionary to ProcessingStats object
            stats_obj = ProcessingStats(**result['stats']) if result.get('stats') else None

            # Convert validation_issues list of dicts to ValidationIssue objects
            validation_issues_objs = [
                ValidationIssue(**issue) for issue in result.get('validation_issues', [])
            ] if result.get('validation_issues') else None

            return ProcessingResponse(
                success=True,
                message=result['message'],
                stats=stats_obj,
                output_file=result['output_file'],
                validation_report=result['validation_report'],
                validation_issues=validation_issues_objs
            )
        else:
            return ProcessingResponse(
                success=False,
                message=result['message'],
                error=result.get('error')
            )

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return ProcessingResponse(
            success=False,
            message="Processing failed",
            error=str(e)
        )


# === SYSTEM STATUS ENDPOINTS ===

@router.get("/status", response_model=SystemStatus)
async def get_system_status(session_id: str = Header(..., alias="X-Session-ID")):
    """Get system status and file counts for this session"""
    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    input_dir = SessionManager.get_input_dir(session_id)
    master_data_dir = SessionManager.get_master_data_dir(session_id)
    output_dir = SessionManager.get_output_dir(session_id)

    allowed_extensions = ['.xlsx', '.xls']
    input_files = get_files_in_directory(input_dir, allowed_extensions)
    master_files = get_files_in_directory(master_data_dir, allowed_extensions)
    output_files = get_files_in_directory(output_dir, ['.csv'])

    # Get last processing time
    last_processing = None
    if output_files:
        last_processing = datetime.fromtimestamp(output_files[0].stat().st_mtime)

    SessionManager.update_activity(session_id)

    return SystemStatus(
        status="ready",
        input_files_count=len(input_files),
        master_data_files_count=len(master_files),
        output_files_count=len(output_files),
        last_processing=last_processing
    )


@router.get("/master-data/status", response_model=MasterDataStatus)
async def get_master_data_status(session_id: str = Header(..., alias="X-Session-ID")):
    """Check master data files status for this session"""
    import pandas as pd

    if not SessionManager.validate_session(session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    try:
        master_data_dir = SessionManager.get_master_data_dir(session_id)

        customers_count = None
        units_count = None
        last_updated = None

        # Check Customer Master
        customer_file = master_data_dir / "Customers_Master.xlsx"
        if customer_file.exists():
            df = pd.read_excel(customer_file)
            customers_count = len(df)

        # Check Unit Master
        unit_file = master_data_dir / "UnitForLease_Master.xlsx"
        if unit_file.exists():
            df = pd.read_excel(unit_file)
            units_count = len(df)
            last_updated = datetime.fromtimestamp(unit_file.stat().st_mtime)

        # Check Config Mapping
        config_file = master_data_dir / "Config_Mapping.xlsx"
        subsidiary_config_exists = config_file.exists()
        utility_mapping_exists = config_file.exists()

        SessionManager.update_activity(session_id)

        return MasterDataStatus(
            customers_count=customers_count,
            units_count=units_count,
            subsidiary_config_exists=subsidiary_config_exists,
            utility_mapping_exists=utility_mapping_exists,
            last_updated=last_updated
        )

    except Exception as e:
        logger.error(f"Error checking master data: {e}")
        return MasterDataStatus(
            subsidiary_config_exists=False,
            utility_mapping_exists=False
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for billing module"""
    # Cleanup expired sessions on health check
    SessionManager.cleanup_expired_sessions()

    return {
        "status": "healthy",
        "module": "billing",
        "timestamp": datetime.now().isoformat()
    }
