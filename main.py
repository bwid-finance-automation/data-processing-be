"""
Main entry point for the unified Data Processing Backend.
This file combines both:
- Finance Accounting module
- FP&A module
into a single FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import sub-applications
from app.finance_accouting.main import app as finance_app
from app.fpa.main import app as fpa_app

# Create the root application
app = FastAPI(
    title="Data Processing Backend (Unified)",
    description="Unified API for Finance Accounting and FP&A modules",
    version="2.0.0",
)

# Allow CORS (optional, if your frontend calls this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount both apps under different prefixes
app.mount("/api/finance", finance_app)
app.mount("/api/fpa", fpa_app)


# Root health check
@app.get("/")
def root():
    return {"message": "Unified Data Processing Backend running", "modules": ["finance", "fpa"]}

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
