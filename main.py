"""
Main entry point for the Data Processing Backend.
This file imports and exposes the full application from app.finance_accouting.
"""

# Import the full application from the finance_accouting module
from app.finance_accouting.main import app

# The 'app' object is now the complete FastAPI application with all endpoints:
# - /process (POST) - Main data processing endpoint
# - /api/process (POST) - API version of process endpoint
# - /start-analysis (POST) - Start AI-powered analysis
# - /logs/{session_id} (GET) - Stream analysis logs
# - /download/{session_id} (GET) - Download analysis results
# - /health (GET) - Health check endpoint
# - And all other endpoints from the finance_accouting module

# For local development, you can run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
