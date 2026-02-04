"""
GLA Variance Analysis API Router
Provides REST API endpoints for comparing GLA data between periods.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.fpa.gla_variance.gla_variance_use_case import GLAVarianceUseCase
from app.application.project.project_service import ProjectService
from app.core.dependencies import get_db, get_project_service, get_ai_usage_repository
from app.infrastructure.database.models.gla import GLAProjectModel
from app.infrastructure.persistence.repositories.ai_usage_repository import AIUsageRepository
from app.shared.utils.logging_config import get_logger
from app.shared.utils.ai_usage_tracker import log_ai_usage

logger = get_logger(__name__)

router = APIRouter(prefix="/fpa/gla-variance", tags=["FP&A - GLA Variance Analysis"])

# Initialize use case
gla_use_case = GLAVarianceUseCase()


async def save_gla_to_project(
    db: AsyncSession,
    project_uuid: str,
    result: Dict[str, Any],
    filename: str
) -> None:
    """
    Save GLA analysis result to project.

    Args:
        db: Database session
        project_uuid: Project UUID string
        result: Analysis result from GLAVarianceUseCase
        filename: Original filename
    """
    try:
        project_service = ProjectService(db)
        case = await project_service.get_or_create_case(UUID(project_uuid), "gla")

        if case:
            # Get statistics from result
            statistics = result.get('statistics', {})

            # Create GLAProjectModel record
            gla_project = GLAProjectModel(
                case_id=case.id,
                file_name=filename,
                processed_at=datetime.utcnow(),
                project_code=f"GLA_{result.get('timestamp', '')}",
                project_name=f"GLA Variance Analysis - {result.get('previous_period', '')} vs {result.get('current_period', '')}",
                product_type="MIXED",
                region="ALL",
                total_gla_sqm=Decimal(str(statistics.get('total_handover_gla', {}).get('current', 0) or 0)),
                period_label=result.get('current_period'),
            )
            db.add(gla_project)

            # Update case file count
            await project_service.increment_case_file_count(case.id)
            await db.commit()

            logger.info(f"Saved GLA analysis to project {project_uuid}, case {case.id}")
    except Exception as e:
        logger.error(f"Failed to save GLA to project: {e}")
        # Don't fail the request if saving to project fails
        await db.rollback()


@router.get("/health")
async def health_check():
    """Health check endpoint for GLA Variance Analysis module"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "module": "FP&A GLA Variance Analysis",
        "version": "2.0.0"
    }


@router.post("/analyze")
async def analyze_gla_variance(
    file: UploadFile = File(..., description="Excel file with GLA data (standard 4-sheet or pivot table format)"),
    previous_label: Optional[str] = Query(None, description="Label for previous period (e.g., 'Oct 2025')"),
    current_label: Optional[str] = Query(None, description="Label for current period (e.g., 'Nov 2025')"),
    project_uuid: Optional[str] = Form(None, description="Project UUID to save analysis to (optional)"),
    db: AsyncSession = Depends(get_db),
    ai_usage_repo: AIUsageRepository = Depends(get_ai_usage_repository),
):
    """
    Analyze GLA variance between two periods from a single Excel file.
    AI-powered analysis is always enabled for generating explanations and insights.

    This endpoint:
    1. Auto-detects file format (standard 4-sheet or pivot table with monthly columns)
    2. Calculates Handover and Committed GLA per project
    3. Computes variance between periods
    4. Uses AI to generate business explanations
    5. Generates output Excel with variance analysis and PDF report
    6. Optionally saves to project if project_uuid is provided

    **Supported formats:**

    **Standard format** - 4 sheets with exact names:
    - `Handover GLA - Previous`
    - `Handover GLA - Current`
    - `Committed GLA - Previous`
    - `Committed GLA - Current`

    **Pivot table format** - Monthly GLA columns:
    - Dates in row 3, headers in row 4
    - Columns like "Handover GLA" for each month
    - Months auto-detected from filename (e.g., "Dec-Nov") or uses last 2 months

    AI-powered analysis includes:
    - Executive summary of portfolio performance
    - Tenant-based explanations for variances in Excel notes
    - Regional and product type trends
    - PDF report download
    """
    logger.info(f"Analyzing GLA variance: {file.filename} (AI: always enabled)")

    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only Excel files (.xlsx, .xls) are supported."
        )

    try:
        import time as _time
        _gla_start = _time.monotonic()
        result = await gla_use_case.execute(
            file=file,
            previous_label=previous_label,
            current_label=current_label
        )
        _gla_elapsed_ms = (_time.monotonic() - _gla_start) * 1000
        logger.info(f"GLA variance analysis successful: {result['statistics']['total_projects']} projects")

        # Log AI usage
        ai_usage = result.get("ai_usage")
        if ai_usage and ai_usage.get("total_tokens", 0) > 0:
            await log_ai_usage(
                ai_usage_repo,
                provider=ai_usage.get("provider", "openai"),
                model_name=ai_usage.get("model", "gpt-4o"),
                task_type="analysis",
                input_tokens=ai_usage.get("input_tokens", 0),
                output_tokens=ai_usage.get("output_tokens", 0),
                processing_time_ms=_gla_elapsed_ms,
                task_description="GLA variance analysis",
                file_name=file.filename,
            )

        # Save to project if project_uuid provided
        if project_uuid:
            await save_gla_to_project(db, project_uuid, result, file.filename)

        return result
    except ValueError as e:
        # Sheet validation errors - return 400 Bad Request
        logger.error(f"GLA variance validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"GLA variance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated GLA variance output file (Excel or PDF)"""
    logger.info(f"Downloading GLA file: {filename}")

    try:
        file_path = gla_use_case.get_output_file_path(filename)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine media type based on file extension
        if filename.endswith('.pdf'):
            media_type = "application/pdf"
        else:
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def list_output_files():
    """List all available GLA variance output files"""
    try:
        files = gla_use_case.list_output_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a GLA variance output file"""
    logger.info(f"Deleting GLA file: {filename}")

    try:
        gla_use_case.delete_output_file(filename)
        return {"status": "success", "message": f"File {filename} deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
