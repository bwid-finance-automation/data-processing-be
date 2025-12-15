"""
GLA Variance Analysis API Router
Provides REST API endpoints for comparing GLA data between periods.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime

from app.application.fpa.gla_variance.gla_variance_use_case import GLAVarianceUseCase
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/fpa/gla-variance", tags=["FP&A - GLA Variance Analysis"])

# Initialize use case
gla_use_case = GLAVarianceUseCase()


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
    file: UploadFile = File(..., description="Excel file with 4 sheets (2 previous + 2 current periods)"),
    previous_label: Optional[str] = Query(None, description="Label for previous period (e.g., 'Oct 2025')"),
    current_label: Optional[str] = Query(None, description="Label for current period (e.g., 'Nov 2025')"),
    use_ai: bool = Query(False, description="Use AI to generate explanations for variances")
):
    """
    Analyze GLA variance between two periods from a single Excel file.

    This endpoint:
    1. Processes a single Excel file containing 4 sheets with GLA data
    2. Calculates Handover and Committed GLA per project
    3. Computes variance between periods
    4. Generates output Excel with variance analysis
    5. (Optional) Uses AI to generate business explanations and PDF report

    **IMPORTANT: Input file MUST contain exactly these 4 sheets (exact names required):**
    - `Handover GLA - Previous`
    - `Handover GLA - Current`
    - `Committed GLA - Previous`
    - `Committed GLA - Current`

    The sheets should have headers at row 5 with columns:
    - Project Name
    - CCS_Product Type (RBF/RBW)
    - Region
    - GLA for Lease
    - Unit for Lease Status
    - BWID Project.

    Set use_ai=true to get AI-powered analysis with:
    - Executive summary of portfolio performance
    - Tenant-based explanations for variances in Excel notes
    - Regional and product type trends
    - PDF report download
    """
    logger.info(f"Analyzing GLA variance: {file.filename} (AI: {use_ai})")

    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only Excel files (.xlsx, .xls) are supported."
        )

    try:
        result = await gla_use_case.execute(
            file=file,
            previous_label=previous_label,
            current_label=current_label,
            use_ai=use_ai
        )
        logger.info(f"GLA variance analysis successful: {result['statistics']['total_projects']} projects")
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
