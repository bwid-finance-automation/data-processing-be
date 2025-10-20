"""
Main entry point for the unified Data Processing Backend.
This file combines both:
- Finance Accounting module
- FP&A module
into a single FastAPI application.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from datetime import datetime
from pathlib import Path

# Import routers from finance accounting module
from app.finance_accouting.api import health as finance_health
from app.finance_accouting.api import analysis as finance_analysis
from app.finance_accouting.api import billing as finance_billing
from app.finance_accouting.api import contract_ocr as finance_contract_ocr

# Note: FP&A doesn't use APIRouter pattern, we'll need to handle it separately
from app.fpa.main import (
    health_check as fpa_health_check,
    compare_files as fpa_compare_files,
    download_file as fpa_download_file,
    list_output_files as fpa_list_output_files,
    delete_file as fpa_delete_file,
    cleanup_old_files as fpa_cleanup_old_files,
    UPLOAD_DIR,
    OUTPUT_DIR
)

# Create the root application
app = FastAPI(
    title="Data Processing Backend (Unified)",
    description="Unified API for Finance Accounting and FP&A modules",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS (optional, if your frontend calls this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Finance Accounting routers with /api/finance prefix
app.include_router(finance_health.router, prefix="/api/finance", tags=["Finance - Health"])
app.include_router(finance_analysis.router, prefix="/api/finance")
app.include_router(finance_billing.router, prefix="/api/finance")
app.include_router(finance_contract_ocr.router, prefix="/api/finance")

# Add FP&A endpoints with /api/fpa prefix
@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup"""
    fpa_cleanup_old_files()

@app.get("/api/fpa/health", tags=["FP&A - Health"])
async def fpa_health():
    """Health check endpoint for FP&A module"""
    return await fpa_health_check()

@app.post("/api/fpa/compare", tags=["FP&A - Compare"])
async def fpa_compare(
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
    return await fpa_compare_files(old_file, new_file)

@app.get("/api/fpa/download/{filename}", tags=["FP&A - Files"])
async def fpa_download(filename: str):
    """Download generated output file"""
    return await fpa_download_file(filename)

@app.get("/api/fpa/files", tags=["FP&A - Files"])
async def fpa_list_files():
    """List all available output files"""
    return await fpa_list_output_files()

@app.delete("/api/fpa/files/{filename}", tags=["FP&A - Files"])
async def fpa_delete(filename: str):
    """Delete an output file"""
    return await fpa_delete_file(filename)

# Root health check
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Unified Data Processing Backend running",
        "version": "2.0.0",
        "modules": {
            "finance": {
                "description": "Finance Accounting module",
                "endpoints": [
                    "/api/finance/health",
                    "/api/finance/process",
                    "/api/finance/start-analysis",
                    "/api/finance/billing/*",
                    "/api/finance/contract-ocr/*"
                ]
            },
            "fpa": {
                "description": "FP&A Excel Comparison module",
                "endpoints": [
                    "/api/fpa/health",
                    "/api/fpa/compare",
                    "/api/fpa/files",
                    "/api/fpa/download/{filename}"
                ]
            }
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
