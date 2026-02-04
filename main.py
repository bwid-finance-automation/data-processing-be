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

# IMPORTANT: Load environment variables FIRST, before any other imports
# This ensures DATABASE_URL and other env vars are available when config modules are imported
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Setup logging first (before any other imports that might use logging)
from app.shared.utils.logging_config import setup_logging
setup_logging(level="INFO")

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Import exception handlers
from app.core.exceptions import (
    AnalysisError,
    FileProcessingError,
    analysis_error_handler,
    validation_error_handler,
    http_error_handler,
    general_error_handler
)

# Import routers from presentation layer (Department + Function Structure)
from app.presentation.api import health_router
from app.presentation.api import project_router
from app.presentation.api.finance import variance_analysis_router
from app.presentation.api.finance import utility_billing_router
from app.presentation.api.finance import contract_ocr_router
from app.presentation.api.finance import bank_statement_parser_router
from app.presentation.api.fpa import excel_comparison_router

# Cash Report only available on Windows (uses COM automation)
import sys
if sys.platform == 'win32':
    from app.presentation.api.finance import cash_report_router
    CASH_REPORT_AVAILABLE = True
else:
    CASH_REPORT_AVAILABLE = False
from app.presentation.api.fpa import gla_variance_router
from app.presentation.api.fpa import ntm_ebitda_router
from app.presentation.api import ai_usage_router
from app.presentation.api import auth_router
from app.presentation.api import system_settings_router

# Import FPA use cases for startup cleanup
from app.application.fpa.excel_comparison.compare_excel_files import CompareExcelFilesUseCase
from app.application.fpa.gla_variance.gla_variance_use_case import GLAVarianceUseCase
from app.application.fpa.ntm_ebitda.ntm_ebitda_use_case import NTMEBITDAUseCase

# Create the root application
app = FastAPI(
    title="Data Processing Backend (Unified)",
    description="Unified API for Finance Accounting and FP&A modules | N-Layer Architecture",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS (optional, if your frontend calls this API)
# Note: When allow_credentials=True, cannot use allow_origins=["*"]
# Must specify exact origins for credentials to work
CORS_ORIGINS = [
    # Local development
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
    # Production (Render)
    "https://bwid-automation.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Add GZip compression for responses > 1KB (reduces response size by 70-90%)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Register exception handlers for user-friendly error messages
app.add_exception_handler(AnalysisError, analysis_error_handler)
app.add_exception_handler(FileProcessingError, analysis_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)

# Include Project router (cross-module)
app.include_router(project_router.router, prefix="/api", tags=["Projects"])

# Include Finance Department routers with /api/finance prefix
app.include_router(health_router.router, prefix="/api/finance", tags=["Finance - Health"])
app.include_router(variance_analysis_router.router, prefix="/api/finance")
app.include_router(utility_billing_router.router, prefix="/api/finance")
app.include_router(contract_ocr_router.router, prefix="/api/finance")
app.include_router(bank_statement_parser_router.router, prefix="/api/finance")

# Cash Report only on Windows
if CASH_REPORT_AVAILABLE:
    app.include_router(cash_report_router.router, prefix="/api/finance")

# Include FP&A Department routers with /api prefix (already have /fpa prefix in routers)
app.include_router(excel_comparison_router.router, prefix="/api")
app.include_router(gla_variance_router.router, prefix="/api")
app.include_router(ntm_ebitda_router.router, prefix="/api")

# Include AI Usage tracking router
app.include_router(ai_usage_router.router, prefix="/api", tags=["AI Usage"])

# Include Authentication router
app.include_router(auth_router.router, prefix="/api")

# Include System Settings router
app.include_router(system_settings_router.router, prefix="/api")

# Get logger for main module
import logging
logger = logging.getLogger(__name__)


async def scheduled_cleanup_job():
    """
    Scheduled job to clean up old files daily.
    Runs at 3AM every day to minimize impact on users.
    """
    logger.info("Starting scheduled cleanup job...")

    try:
        from app.infrastructure.database.session import get_db
        from app.application.finance.bank_statement_parser.bank_statement_db_service import BankStatementDbService

        async for db in get_db():
            db_service = BankStatementDbService(db)
            stats = await db_service.cleanup_old_files(retention_days=7)
            logger.info(
                f"Scheduled cleanup completed: "
                f"{stats['files_deleted']} files deleted, "
                f"{stats.get('disk_space_freed', 0) / 1024 / 1024:.2f} MB freed"
            )
            break
    except Exception as e:
        logger.error(f"Scheduled cleanup failed: {e}")

    # Also cleanup FPA files
    try:
        fpa_use_case = CompareExcelFilesUseCase()
        fpa_use_case.cleanup_old_files()

        gla_use_case = GLAVarianceUseCase()
        gla_use_case.cleanup_old_files()

        ntm_use_case = NTMEBITDAUseCase()
        ntm_use_case.cleanup_old_files()
        logger.info("FPA file cleanup completed")
    except Exception as e:
        logger.warning(f"FPA cleanup warning: {e}")


# Startup event for cleanup and database initialization
@app.on_event("startup")
async def startup_event():
    """Run cleanup and initialize database on startup"""
    logger.info("Application starting up...")

    # Initialize database connection
    try:
        from app.infrastructure.database import init_db
        await init_db()
        logger.info("Database initialized successfully")

        # Seed default data (admin user, etc.)
        from app.infrastructure.database.seed import run_all_seeds
        await run_all_seeds()
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
        logger.info("   (Database features will be unavailable)")

    # Initialize Redis cache connection
    try:
        from app.infrastructure.cache.redis_cache import get_cache_service
        cache = await get_cache_service()
        if cache.is_connected:
            logger.info("Redis cache initialized successfully")
        else:
            logger.info("Redis not available, caching disabled")
    except Exception as e:
        logger.warning(f"Redis initialization skipped: {e}")
        logger.info("   (Caching features will be unavailable)")

    # Cleanup old files (legacy use cases)
    fpa_use_case = CompareExcelFilesUseCase()
    fpa_use_case.cleanup_old_files()

    gla_use_case = GLAVarianceUseCase()
    gla_use_case.cleanup_old_files()

    ntm_use_case = NTMEBITDAUseCase()
    ntm_use_case.cleanup_old_files()

    # Cleanup old bank statement uploads (7 days retention)
    try:
        from app.infrastructure.database.session import get_db
        from app.application.finance.bank_statement_parser.bank_statement_db_service import BankStatementDbService

        async for db in get_db():
            db_service = BankStatementDbService(db)
            stats = await db_service.cleanup_old_files(retention_days=7)
            if stats["files_deleted"] > 0:
                logger.info(f"Cleaned up {stats['files_deleted']} old bank statement files")
            break
    except Exception as e:
        logger.warning(f"Bank statement cleanup skipped: {e}")

    # Start the scheduler for daily cleanup at 3AM
    scheduler.add_job(
        scheduled_cleanup_job,
        CronTrigger(hour=3, minute=0),
        id="daily_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduled daily cleanup job at 3:00 AM")

    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down...")

    # Stop the scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    # Close database connections
    try:
        from app.infrastructure.database import close_db
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Error closing database: {e}")

    # Close Redis connection
    try:
        from app.infrastructure.cache.redis_cache import RedisCacheService
        if RedisCacheService._instance:
            await RedisCacheService._instance.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Error closing Redis: {e}")

    logger.info("Shutdown complete")

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
            "infrastructure": "External dependencies (Excel, AI, OCR, storage, database)"
        },
        "database": {
            "type": "PostgreSQL",
            "orm": "SQLAlchemy 2.0 Async",
            "migrations": "Alembic"
        },
        "modules": {
            "finance": {
                "description": "Finance Accounting module",
                "endpoints": [
                    "/api/finance/health",
                    "/api/finance/process",
                    "/api/finance/start-analysis",
                    "/api/finance/billing/*",
                    "/api/finance/contract-ocr/*",
                    "/api/finance/bank-statements/*",
                    "/api/finance/cash-report/*"
                ]
            },
            "fpa": {
                "description": "FP&A Excel Comparison, GLA Variance & NTM EBITDA Analysis",
                "endpoints": [
                    "/api/fpa/health",
                    "/api/fpa/compare",
                    "/api/fpa/files",
                    "/api/fpa/download/{filename}",
                    "/api/fpa/gla-variance/health",
                    "/api/fpa/gla-variance/analyze",
                    "/api/fpa/gla-variance/files",
                    "/api/fpa/gla-variance/download/{filename}",
                    "/api/fpa/ntm-ebitda/health",
                    "/api/fpa/ntm-ebitda/analyze",
                    "/api/fpa/ntm-ebitda/detect-sheets",
                    "/api/fpa/ntm-ebitda/files",
                    "/api/fpa/ntm-ebitda/download/{filename}"
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
