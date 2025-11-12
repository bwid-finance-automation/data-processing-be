# app/presentation/api/fpa/fpa_router.py
"""
FPA Excel Summary Comparison API Router
Provides REST API endpoints for comparing Excel files
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime

from app.application.fpa.excel_comparison.compare_excel_files import CompareExcelFilesUseCase
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/fpa", tags=["FP&A - Excel Comparison"])

# Initialize use case
compare_use_case = CompareExcelFilesUseCase()

@router.get("/health")
async def health_check():
    """Health check endpoint for FP&A module"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "module": "FP&A Excel Comparison",
        "version": "3.0.0"
    }

@router.post("/compare")
async def compare_files(
    old_file: UploadFile = File(..., description="Previous month Excel file"),
    new_file: UploadFile = File(..., description="Current month Excel file")
):
    """
    Compare two Excel files and return comparison results.

    This endpoint:
    1. Generates an output Excel file with new_rows and update_rows sheets
    2. Applies highlighting to the current file (yellow for new rows, blue for changed cells)
    3. Returns both files for download along with comparison statistics
    """
    logger.info(f"Comparing files: {old_file.filename} vs {new_file.filename}")

    try:
        result = await compare_use_case.execute(old_file, new_file)
        logger.info(f"Comparison successful: {result['statistics']}")
        return result
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated output file"""
    logger.info(f"Downloading file: {filename}")

    try:
        file_path = compare_use_case.get_output_file_path(filename)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_output_files():
    """List all available output files"""
    try:
        files = compare_use_case.list_output_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete an output file"""
    logger.info(f"Deleting file: {filename}")

    try:
        compare_use_case.delete_output_file(filename)
        return {"status": "success", "message": f"File {filename} deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
