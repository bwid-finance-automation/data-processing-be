# app/services/analysis_service.py
"""Business logic for financial analysis operations."""

import io
import threading
import queue
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
from fastapi import UploadFile, HTTPException

from ..core.config import get_analysis_config
from ..models.analysis import AnalysisSession, RevenueVarianceAnalysisResponse
from .processing_service import process_all
from ..analysis.revenue_analysis import analyze_revenue_variance_comprehensive
from ..utils.log_capture import LogCapture

class AnalysisService:
    """Service for handling financial analysis operations."""

    def __init__(self):
        self.debug_files_store: Dict[str, Tuple[str, bytes]] = {}
        self.log_streams: Dict[str, queue.Queue] = {}
        self.log_captures: Dict[str, 'LogCapture'] = {}  # Store LogCapture objects for history access
        self.active_sessions: Dict[str, AnalysisSession] = {}
        self.session_progress: Dict[str, Dict[str, Any]] = {}  # Track progress per session

    def create_session(self) -> AnalysisSession:
        """Create a new analysis session."""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        session = AnalysisSession(
            session_id=session_id,
            status="created"
        )
        self.active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)

    def cleanup_old_sessions(self, max_age_minutes: int = 60):
        """Clean up old sessions and their associated data."""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.created_at < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.cleanup_session(session_id)

    def cleanup_session(self, session_id: str):
        """Clean up a specific session."""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # Remove log streams
        if session_id in self.log_streams:
            del self.log_streams[session_id]

        # Remove log captures
        if session_id in self.log_captures:
            del self.log_captures[session_id]

        # Remove progress tracking
        if session_id in self.session_progress:
            del self.session_progress[session_id]

        # Remove debug files
        files_to_remove = [key for key in self.debug_files_store.keys() if key.startswith(session_id)]
        for key in files_to_remove:
            del self.debug_files_store[key]

    async def process_python_analysis(
        self,
        excel_files: List[UploadFile],
        mapping_file: Optional[UploadFile] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Process files using Python-based analysis."""
        try:
            # Read files into memory
            files: List[Tuple[str, bytes]] = [
                (f.filename or "input.xlsx", await f.read())
                for f in excel_files
            ]

            # Handle mapping file if provided
            corr_rules: Optional[pd.DataFrame] = None
            season_rules: Optional[pd.DataFrame] = None
            if mapping_file is not None:
                mapping_bytes = await mapping_file.read()
                corr_rules, season_rules = self._load_mapping_rules(mapping_bytes)

            # Build configuration
            config = get_analysis_config()
            if config_overrides:
                config.update(config_overrides)

            # Disable AI for Python mode
            config["use_llm_analysis"] = False

            # Process files
            xlsx_bytes: bytes = process_all(
                files=files,
                corr_rules=corr_rules,
                season_rules=season_rules,
                CONFIG=config,
            )

            return xlsx_bytes

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def start_ai_analysis(self, excel_files: List[UploadFile]) -> AnalysisSession:
        """Start AI-powered analysis in background thread."""
        try:
            # Create session
            session = self.create_session()
            session.status = "processing"

            # Read files into memory
            files: List[Tuple[str, bytes]] = [
                (f.filename or "input.xlsx", await f.read())
                for f in excel_files
            ]

            # Create log stream BEFORE starting thread so SSE endpoint can access it immediately
            log_capture = LogCapture(session.session_id)
            self.log_streams[session.session_id] = log_capture.queue
            self.log_captures[session.session_id] = log_capture  # Store for history access

            # Start background processing
            thread = threading.Thread(
                target=self._run_ai_analysis,
                args=(session.session_id, files, log_capture)
            )
            thread.daemon = True
            thread.start()

            return session

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start AI analysis: {str(e)}")

    def _run_ai_analysis(self, session_id: str, files: List[Tuple[str, bytes]], log_capture: LogCapture):
        """Run AI analysis in background thread."""
        import sys

        # log_capture is now passed from start_ai_analysis to ensure it exists before SSE connects
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = log_capture

        try:
            def progress_update(percentage, message):
                log_capture.queue.put(f"__PROGRESS__{percentage}__{message}")
                # Also store progress separately for polling to access without disrupting queue
                self.session_progress[session_id] = {
                    "percentage": percentage,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }

            # Send initial log
            log_capture.queue.put(f"🚀 Starting AI-powered variance analysis")
            log_capture.queue.put(f"📁 Processing {len(files)} file(s)")

            # Configure AI analysis
            config = get_analysis_config()
            config["use_llm_analysis"] = True
            config["llm_model"] = "gpt-4o"

            log_capture.queue.put(f"🤖 AI Model: {config['llm_model']}")
            log_capture.queue.put(f"⚙️  Configuration loaded successfully")

            progress_update(10, "Starting AI analysis...")

            # Process with AI
            xlsx_bytes, debug_files = process_all(
                files=files,
                CONFIG=config,
                progress_callback=progress_update
            )

            progress_update(85, "Storing results...")

            # Store debug files
            log_capture.queue.put(f"💾 Storing {len(debug_files)} debug file(s)...")
            for debug_name, debug_bytes in debug_files:
                file_key = f"{session_id}_{debug_name}"
                self.debug_files_store[file_key] = (debug_name, debug_bytes)
                log_capture.queue.put(f"   ✅ Saved: {debug_name} ({len(debug_bytes):,} bytes)")

            # Store main result
            main_file_key = f"{session_id}_main_result"
            result_filename = f"ai_variance_analysis_{session_id}.xlsx"
            self.debug_files_store[main_file_key] = (result_filename, xlsx_bytes)
            log_capture.queue.put(f"📊 Main result: {result_filename} ({len(xlsx_bytes):,} bytes)")

            progress_update(100, "Analysis complete!")
            log_capture.queue.put(f"✅ Analysis completed successfully!")
            log_capture.queue.put(f"📥 Download ready: {result_filename}")
            log_capture.queue.put("__ANALYSIS_COMPLETE__")

            # Update session status
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "completed"

        except Exception as e:
            log_capture.queue.put(f"❌ Error occurred: {str(e)}")
            log_capture.queue.put(f"__ERROR__{str(e)}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "failed"
        finally:
            # Restore original stdout
            sys.stdout = old_stdout

    async def analyze_revenue_variance(
        self,
        excel_file: UploadFile
    ) -> RevenueVarianceAnalysisResponse:
        """Perform comprehensive revenue variance analysis."""
        try:
            # Read file
            file_bytes = await excel_file.read()
            filename = excel_file.filename or "input.xlsx"

            # Get configuration
            config = get_analysis_config()

            # Run analysis
            analysis_result = analyze_revenue_variance_comprehensive(
                file_bytes, filename, config
            )

            # Handle error response
            if "error" in analysis_result:
                raise HTTPException(status_code=400, detail=analysis_result["error"])

            # Convert to Pydantic model
            return RevenueVarianceAnalysisResponse(**analysis_result)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Revenue variance analysis failed: {str(e)}")

    def get_debug_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of debug files for a session."""
        session_files = []
        for key, (name, file_bytes) in self.debug_files_store.items():
            if key.startswith(session_id + "_"):
                session_files.append({
                    "key": key,
                    "name": name,
                    "size": len(file_bytes),
                    "download_url": f"/api/debug/{key}"
                })
        return session_files

    def get_file(self, file_key: str) -> Optional[Tuple[str, bytes]]:
        """Get file by key."""
        return self.debug_files_store.get(file_key)

    def _load_mapping_rules(self, mapping_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load mapping rules from Excel bytes."""
        try:
            xls = pd.ExcelFile(io.BytesIO(mapping_bytes))
            sheets_lower = {name.lower(): name for name in xls.sheet_names}

            # Find correlation rules sheet
            corr_candidates = [
                "correlation", "correlations", "corr", "mapping", "rules", "correlation_rules"
            ]
            corr_name = next((sheets_lower[n] for n in corr_candidates if n in sheets_lower), None)

            # Find seasonality rules sheet
            season_candidates = [
                "seasonality", "season", "season_rules", "seasonality_rules"
            ]
            season_name = next((sheets_lower[n] for n in season_candidates if n in sheets_lower), None)

            corr_df = pd.read_excel(xls, sheet_name=corr_name) if corr_name else pd.DataFrame()
            season_df = pd.read_excel(xls, sheet_name=season_name) if season_name else pd.DataFrame()

            return corr_df, season_df
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

# Create singleton instance
analysis_service = AnalysisService()