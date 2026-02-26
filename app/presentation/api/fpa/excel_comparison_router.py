# app/presentation/api/fpa/fpa_router.py
"""
FPA Excel Summary Comparison API Router
Provides REST API endpoints for comparing Excel files
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime
import uuid as uuid_lib

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.fpa.excel_comparison.compare_excel_files import CompareExcelFilesUseCase
from app.core.dependencies import get_db, get_current_user_optional
from app.infrastructure.database.models.analysis_session import AnalysisSessionModel
from app.infrastructure.database.models.user import UserModel
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/fpa", tags=["FP&A - Excel Comparison"])

# Initialize use case
compare_use_case = CompareExcelFilesUseCase()


async def save_comparison_session(
    db: AsyncSession,
    old_filename: str,
    new_filename: str,
    result: dict,
    user_id: Optional[int] = None,
) -> None:
    """Save excel comparison session to database with user tracking."""
    try:
        statistics = result.get('statistics', {})
        session_id = f"compare_{uuid_lib.uuid4().hex[:8]}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        analysis_session = AnalysisSessionModel(
            session_id=session_id,
            user_id=user_id,
            status="COMPLETED",
            analysis_type="EXCEL_COMPARISON",
            files_count=2,
            progress_percentage=100,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            processing_details={
                "old_file": old_filename,
                "new_file": new_filename,
                "new_rows": statistics.get('new_rows', 0),
                "updated_rows": statistics.get('updated_rows', 0),
                "output_file": result.get('output_file'),
                "highlighted_file": result.get('highlighted_file'),
            },
        )
        db.add(analysis_session)
        await db.commit()
        logger.info(f"Saved excel comparison session {session_id} (user_id: {user_id})")
    except Exception as e:
        logger.error(f"Failed to save excel comparison session: {e}")
        await db.rollback()

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
    new_file: UploadFile = File(..., description="Current month Excel file"),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """
    Compare two Excel files and return comparison results.

    This endpoint:
    1. Generates an output Excel file with new_rows and update_rows sheets
    2. Applies highlighting to the current file (yellow for new rows, blue for changed cells)
    3. Returns both files for download along with comparison statistics
    4. Optionally saves to project if project_uuid is provided
    """
    logger.info(f"Comparing files: {old_file.filename} vs {new_file.filename}")

    try:
        result = await compare_use_case.execute(old_file, new_file)
        logger.info(f"Comparison successful: {result['statistics']}")

        # Save comparison session
        user_id = current_user.id if current_user else None
        await save_comparison_session(db, old_file.filename, new_file.filename, result, user_id=user_id)

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
