"""
FastAPI Backend for Excel Summary Comparison Tool
Provides REST API endpoints for comparing Excel files
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.fpa.excel_processor.summary_comparator import SummaryComparator

# Initialize FastAPI app
app = FastAPI(
    title="Excel Summary Comparison API",
    description="API for comparing Excel summary files with highlighting",
    version="2.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload and output directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Cleanup old files (older than 24 hours)
def cleanup_old_files():
    """Remove files older than 24 hours"""
    import time
    current_time = time.time()
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 86400:  # 24 hours
                    file_path.unlink()

@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup"""
    cleanup_old_files()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/compare")
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
    try:
        # Save uploaded files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_path = UPLOAD_DIR / f"old_{timestamp}_{old_file.filename}"
        new_path = UPLOAD_DIR / f"new_{timestamp}_{new_file.filename}"

        with open(old_path, "wb") as f:
            shutil.copyfileobj(old_file.file, f)
        with open(new_path, "wb") as f:
            shutil.copyfileobj(new_file.file, f)

        # Initialize comparator
        comparator = SummaryComparator()

        # Generate output file with new_rows and update_rows
        output_path = comparator.generate_excel_output_files(
            str(old_path),
            str(new_path),
            str(OUTPUT_DIR / f"comparison_{timestamp}.xlsx")
        )

        # Create highlighted copy (does NOT modify original uploaded files)
        highlighted_path = OUTPUT_DIR / f"highlighted_{timestamp}_{new_file.filename}"
        highlighted_output = comparator.apply_highlighting_to_summary_with_90_day_rule(
            str(old_path),
            str(new_path),
            str(highlighted_path)
        )

        # Get comparison statistics
        comparison_results = comparator.compare_summary_files_by_document_item_key(
            str(old_path),
            str(new_path)
        )

        # Build response
        result = {
            "status": "success",
            "timestamp": timestamp,
            "old_filename": old_file.filename,
            "new_filename": new_file.filename,
            "output_file": os.path.basename(output_path),
            "highlighted_file": os.path.basename(highlighted_path),
            "message": "Comparison and highlighting completed successfully",
            "statistics": {
                "new_rows": len(comparison_results['new_rows_indices']),
                "updated_rows": len(comparison_results['update_rows_indices']),
                "unchanged_rows": len(comparison_results['unchanged_rows_indices'])
            }
        }

        # Cleanup uploaded files (keep outputs)
        old_path.unlink()
        new_path.unlink()

        return result

    except Exception as e:
        # Cleanup files on error
        if old_path.exists():
            old_path.unlink()
        if new_path.exists():
            new_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated output file"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/files")
async def list_output_files():
    """List all available output files"""
    files = []
    for file_path in OUTPUT_DIR.glob("*.xlsx"):
        files.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
        })
    return {"files": files}

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete an output file"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_path.unlink()
    return {"status": "success", "message": f"File {filename} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
