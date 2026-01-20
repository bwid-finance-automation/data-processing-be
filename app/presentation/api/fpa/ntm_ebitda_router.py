"""
NTM EBITDA Variance Analysis API Router
Provides REST API endpoints for analyzing NTM EBITDA variance between periods.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.fpa.ntm_ebitda.ntm_ebitda_use_case import NTMEBITDAUseCase
from app.core.dependencies import get_db
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/fpa/ntm-ebitda", tags=["FP&A - NTM EBITDA Variance Analysis"])

# Initialize use case
ntm_use_case = NTMEBITDAUseCase()


@router.get("/health")
async def health_check():
    """Health check endpoint for NTM EBITDA Variance Analysis module"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "module": "FP&A NTM EBITDA Variance Analysis",
        "version": "1.0.0"
    }


@router.post("/analyze")
async def analyze_ntm_variance(
    file: UploadFile = File(..., description="Excel file with leasing model data (.xlsx or .xlsb)"),
    prev_sheet: Optional[str] = Query(None, description="Sheet name for previous period (auto-detected if not provided)"),
    curr_sheet: Optional[str] = Query(None, description="Sheet name for current period (auto-detected if not provided)"),
    previous_label: Optional[str] = Query(None, description="Label for previous period (e.g., 'Sep\\'25')"),
    current_label: Optional[str] = Query(None, description="Label for current period (e.g., 'Nov\\'25')"),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze NTM EBITDA variance between two periods from a leasing model Excel file.

    This endpoint:
    1. Extracts NTM (Next Twelve Months) revenue from the leasing model
    2. Calculates variance between periods by project
    3. Analyzes lease-level changes (new signings, terminations, timing shifts)
    4. Generates professional commentary using AI
    5. Exports to BW standard Excel format with highlighting

    **Input file requirements:**

    The leasing model Excel file should have:
    - Sheets named with period (e.g., "Model_leasing_Sep'25", "Model_leasing_Nov'25")
    - Project codes in column 0
    - Phase in column 2 (filter by "All Phase" for project-level data)
    - Metric type in column 4 (e.g., "Accounting revenue")
    - Tenant name in column 10
    - GLA in column 15
    - Lease dates in columns 24-25
    - Monthly NTM data in columns 40-51

    Optional: "Mapping" sheet with project code -> reporting name mapping

    **Output:**
    - Excel file with variance analysis in BW standard format
    - Projects with >5% variance highlighted in yellow
    - AI-generated commentary for significant variances
    """
    logger.info(f"Analyzing NTM EBITDA variance: {file.filename}")

    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xlsb', '.xls')):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only Excel files (.xlsx, .xlsb, .xls) are supported."
        )

    try:
        result = await ntm_use_case.execute(
            file=file,
            prev_sheet=prev_sheet,
            curr_sheet=curr_sheet,
            previous_label=previous_label,
            current_label=current_label
        )
        logger.info(f"NTM EBITDA analysis successful: {result['statistics']['total_projects']} projects")
        return result

    except ValueError as e:
        logger.error(f"NTM EBITDA validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"NTM EBITDA analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-sheets")
async def detect_sheets(
    file: UploadFile = File(..., description="Excel file to detect sheets from")
):
    """
    Detect available sheet names in an Excel file.

    Returns list of sheet names for the user to select from.
    """
    logger.info(f"Detecting sheets in: {file.filename}")

    if not file.filename.endswith(('.xlsx', '.xlsb', '.xls')):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only Excel files are supported."
        )

    try:
        sheets = ntm_use_case.get_available_sheets(file)
        return {
            "filename": file.filename,
            "sheets": sheets,
            "count": len(sheets)
        }
    except Exception as e:
        logger.error(f"Sheet detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated NTM EBITDA variance output file (Excel)"""
    logger.info(f"Downloading NTM file: {filename}")

    try:
        file_path = ntm_use_case.get_output_file_path(filename)

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
    """List all available NTM EBITDA variance output files"""
    try:
        files = ntm_use_case.list_output_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete an NTM EBITDA variance output file"""
    logger.info(f"Deleting NTM file: {filename}")

    try:
        ntm_use_case.delete_output_file(filename)
        return {"status": "success", "message": f"File {filename} deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
