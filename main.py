"""
Main entry point for the unified Data Processing Backend.
This file combines both:
- Finance Accounting module
- FP&A module
into a single FastAPI application.

Architecture: N-Layer Monolith
- Presentation Layer: API routers (presentation/api/)
- Application Layer: Use cases and orchestration (application/)
- Domain Layer: Business logic (domain/)
- Infrastructure Layer: External dependencies (infrastructure/)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from presentation layer (Department + Function Structure)
from app.presentation.api import health_router
from app.presentation.api.finance import variance_analysis_router
from app.presentation.api.finance import utility_billing_router
from app.presentation.api.finance import contract_ocr_router
from app.presentation.api.fpa import excel_comparison_router

# Import FPA use case for startup cleanup
from app.application.fpa.excel_comparison.compare_excel_files import CompareExcelFilesUseCase

# Create the root application
app = FastAPI(
    title="Data Processing Backend (Unified)",
    description="Unified API for Finance Accounting and FP&A modules | N-Layer Architecture",
    version="3.0.0",
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

# Include Finance Department routers with /api/finance prefix
app.include_router(health_router.router, prefix="/api/finance", tags=["Finance - Health"])
app.include_router(variance_analysis_router.router, prefix="/api/finance")
app.include_router(utility_billing_router.router, prefix="/api/finance")
app.include_router(contract_ocr_router.router, prefix="/api/finance")

# Include FP&A Department router with /api prefix (already has /fpa prefix in router)
app.include_router(excel_comparison_router.router, prefix="/api")

# Startup event for cleanup
@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup"""
    fpa_use_case = CompareExcelFilesUseCase()
    fpa_use_case.cleanup_old_files()

# Root health check
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Unified Data Processing Backend running",
        "version": "3.0.0",
        "architecture": "N-Layer Monolith",
        "layers": {
            "presentation": "API routers, middleware, request/response schemas",
            "application": "Use cases, orchestration, application services",
            "domain": "Business logic, domain models, domain services",
            "infrastructure": "External dependencies (Excel, AI, OCR, storage)"
        },
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
