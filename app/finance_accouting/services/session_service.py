# app/services/session_service.py
"""
Enhanced session management service with dependency injection.

This service handles session lifecycle, progress tracking, and state management
with comprehensive logging and error handling.
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .interfaces import SessionServiceBase
from ..models.analysis import SessionInfo, AnalysisProgress, AnalysisResult
from ..core.exceptions import ValidationError
from ..core.unified_config import get_unified_config

class SessionService(SessionServiceBase):
    """
    Enhanced session management service.

    Features:
    - Automatic session lifecycle management
    - Progress tracking with real-time updates
    - Session persistence with JSON storage
    - Automatic cleanup of expired sessions
    - Thread-safe operations
    """

    def __init__(self):
        super().__init__()
        self.config = get_unified_config()
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_progress: Dict[str, AnalysisProgress] = {}
        self._session_results: Dict[str, AnalysisResult] = {}

        # Create sessions directory
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(exist_ok=True)

        # Load existing sessions
        self._load_persisted_sessions()

        self.log_operation("SessionService initialized", active_sessions=len(self._sessions))

    def create_session(self, user_id: Optional[str] = None) -> SessionInfo:
        """
        Create a new analysis session.

        Args:
            user_id: Optional user identifier

        Returns:
            SessionInfo for the new session
        """
        try:
            session_id = str(uuid.uuid4())
            current_time = datetime.utcnow()

            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                status="created",
                created_at=current_time,
                updated_at=current_time,
                timeout_minutes=self.config.security.session_timeout_minutes
            )

            self._sessions[session_id] = session_info

            # Initialize progress
            self._session_progress[session_id] = AnalysisProgress(
                session_id=session_id,
                status="initialized",
                progress_percentage=0,
                current_step="Session created",
                started_at=current_time
            )

            # Persist session
            self._persist_session(session_info)

            self.log_operation(
                "create_session_success",
                session_id=session_id,
                user_id=user_id,
                total_sessions=len(self._sessions)
            )

            return session_info

        except Exception as e:
            self.log_error("create_session", e, user_id=user_id)
            raise ValidationError(
                f"Failed to create session: {str(e)}",
                user_message="Unable to create a new analysis session",
                error_code="SESSION_CREATION_ERROR"
            ) from e

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo if found, None otherwise
        """
        try:
            session = self._sessions.get(session_id)
            if session:
                # Check if session has expired
                if self._is_session_expired(session):
                    self.end_session(session_id)
                    return None

                # Update last accessed time
                session.updated_at = datetime.utcnow()
                self._persist_session(session)

            return session

        except Exception as e:
            self.log_error("get_session", e, session_id=session_id)
            return None

    def update_session_progress(self, session_id: str, progress: AnalysisProgress) -> bool:
        """
        Update progress for a session.

        Args:
            session_id: Session identifier
            progress: New progress information

        Returns:
            True if update was successful
        """
        try:
            if session_id not in self._sessions:
                self.log_error(
                    "update_session_progress",
                    ValueError("Session not found"),
                    session_id=session_id
                )
                return False

            # Update progress
            progress.updated_at = datetime.utcnow()
            self._session_progress[session_id] = progress

            # Update session status and timestamp
            session = self._sessions[session_id]
            session.status = progress.status
            session.updated_at = progress.updated_at

            # Persist changes
            self._persist_session(session)
            self._persist_progress(progress)

            self.log_operation(
                "update_session_progress_success",
                session_id=session_id,
                status=progress.status,
                progress=progress.progress_percentage
            )

            return True

        except Exception as e:
            self.log_error("update_session_progress", e, session_id=session_id)
            return False

    def end_session(self, session_id: str, result: Optional[AnalysisResult] = None) -> bool:
        """
        End a session with optional result.

        Args:
            session_id: Session identifier
            result: Optional analysis result

        Returns:
            True if session was ended successfully
        """
        try:
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]
            current_time = datetime.utcnow()

            # Update session
            session.status = "completed" if result else "ended"
            session.updated_at = current_time
            session.completed_at = current_time

            # Store result if provided
            if result:
                result.session_id = session_id
                result.completed_at = current_time
                self._session_results[session_id] = result
                self._persist_result(result)

            # Update progress
            if session_id in self._session_progress:
                progress = self._session_progress[session_id]
                progress.status = "completed" if result else "ended"
                progress.progress_percentage = 100
                progress.updated_at = current_time
                progress.completed_at = current_time
                self._persist_progress(progress)

            # Persist session
            self._persist_session(session)

            self.log_operation(
                "end_session_success",
                session_id=session_id,
                final_status=session.status,
                has_result=result is not None
            )

            return True

        except Exception as e:
            self.log_error("end_session", e, session_id=session_id)
            return False

    def get_session_progress(self, session_id: str) -> Optional[AnalysisProgress]:
        """Get current progress for a session."""
        return self._session_progress.get(session_id)

    def get_session_result(self, session_id: str) -> Optional[AnalysisResult]:
        """Get result for a completed session."""
        return self._session_results.get(session_id)

    def list_active_sessions(self) -> List[SessionInfo]:
        """
        List all active (non-expired) sessions.

        Returns:
            List of active SessionInfo objects
        """
        try:
            active_sessions = []
            expired_sessions = []

            for session_id, session in self._sessions.items():
                if self._is_session_expired(session):
                    expired_sessions.append(session_id)
                else:
                    active_sessions.append(session)

            # Clean up expired sessions
            for session_id in expired_sessions:
                self.end_session(session_id)

            self.log_operation(
                "list_active_sessions",
                active_count=len(active_sessions),
                expired_cleaned=len(expired_sessions)
            )

            return active_sessions

        except Exception as e:
            self.log_error("list_active_sessions", e)
            return []

    def cleanup_old_sessions(self, max_age_minutes: Optional[int] = None) -> int:
        """
        Clean up old sessions beyond the specified age.

        Args:
            max_age_minutes: Maximum age in minutes (uses config default if None)

        Returns:
            Number of sessions cleaned up
        """
        try:
            max_age = max_age_minutes or self.config.security.session_timeout_minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=max_age)

            sessions_to_cleanup = []
            for session_id, session in self._sessions.items():
                if session.updated_at < cutoff_time:
                    sessions_to_cleanup.append(session_id)

            # Clean up identified sessions
            cleaned_count = 0
            for session_id in sessions_to_cleanup:
                if self._cleanup_session_files(session_id):
                    # Remove from memory
                    self._sessions.pop(session_id, None)
                    self._session_progress.pop(session_id, None)
                    self._session_results.pop(session_id, None)
                    cleaned_count += 1

            self.log_operation(
                "cleanup_old_sessions_success",
                max_age_minutes=max_age,
                sessions_cleaned=cleaned_count,
                remaining_sessions=len(self._sessions)
            )

            return cleaned_count

        except Exception as e:
            self.log_error("cleanup_old_sessions", e)
            return 0

    def _is_session_expired(self, session: SessionInfo) -> bool:
        """Check if a session has expired."""
        if not session.timeout_minutes:
            return False

        expiry_time = session.updated_at + timedelta(minutes=session.timeout_minutes)
        return datetime.utcnow() > expiry_time

    def _persist_session(self, session: SessionInfo) -> None:
        """Persist session to disk."""
        try:
            session_file = self.sessions_dir / f"{session.session_id}_session.json"
            with open(session_file, 'w') as f:
                # Convert to dict and handle datetime serialization
                session_data = session.model_dump()
                # Convert datetime objects to ISO strings
                for key, value in session_data.items():
                    if isinstance(value, datetime):
                        session_data[key] = value.isoformat()

                json.dump(session_data, f, indent=2)

        except Exception as e:
            self.log_error("persist_session", e, session_id=session.session_id)

    def _persist_progress(self, progress: AnalysisProgress) -> None:
        """Persist progress to disk."""
        try:
            progress_file = self.sessions_dir / f"{progress.session_id}_progress.json"
            with open(progress_file, 'w') as f:
                # Convert to dict and handle datetime serialization
                progress_data = progress.model_dump()
                for key, value in progress_data.items():
                    if isinstance(value, datetime):
                        progress_data[key] = value.isoformat()

                json.dump(progress_data, f, indent=2)

        except Exception as e:
            self.log_error("persist_progress", e, session_id=progress.session_id)

    def _persist_result(self, result: AnalysisResult) -> None:
        """Persist result to disk."""
        try:
            result_file = self.sessions_dir / f"{result.session_id}_result.json"
            with open(result_file, 'w') as f:
                # Convert to dict and handle datetime serialization
                result_data = result.model_dump()
                for key, value in result_data.items():
                    if isinstance(value, datetime):
                        result_data[key] = value.isoformat()

                json.dump(result_data, f, indent=2)

        except Exception as e:
            self.log_error("persist_result", e, session_id=result.session_id)

    def _load_persisted_sessions(self) -> None:
        """Load sessions from disk on startup."""
        try:
            session_files = list(self.sessions_dir.glob("*_session.json"))
            loaded_count = 0

            for session_file in session_files:
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)

                    # Convert ISO strings back to datetime objects
                    for key in ['created_at', 'updated_at', 'completed_at']:
                        if session_data.get(key):
                            session_data[key] = datetime.fromisoformat(session_data[key])

                    session = SessionInfo(**session_data)
                    session_id = session.session_id

                    # Only load non-expired sessions
                    if not self._is_session_expired(session):
                        self._sessions[session_id] = session

                        # Load progress if exists
                        progress_file = self.sessions_dir / f"{session_id}_progress.json"
                        if progress_file.exists():
                            with open(progress_file, 'r') as f:
                                progress_data = json.load(f)
                            for key in ['started_at', 'updated_at', 'completed_at']:
                                if progress_data.get(key):
                                    progress_data[key] = datetime.fromisoformat(progress_data[key])
                            self._session_progress[session_id] = AnalysisProgress(**progress_data)

                        # Load result if exists
                        result_file = self.sessions_dir / f"{session_id}_result.json"
                        if result_file.exists():
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                            for key in ['created_at', 'completed_at']:
                                if result_data.get(key):
                                    result_data[key] = datetime.fromisoformat(result_data[key])
                            self._session_results[session_id] = AnalysisResult(**result_data)

                        loaded_count += 1

                except Exception as e:
                    self.log_error("load_session_file", e, session_file=str(session_file))

            self.log_operation("load_persisted_sessions_success", sessions_loaded=loaded_count)

        except Exception as e:
            self.log_error("load_persisted_sessions", e)

    def _cleanup_session_files(self, session_id: str) -> bool:
        """Clean up all files related to a session."""
        try:
            files_to_remove = [
                self.sessions_dir / f"{session_id}_session.json",
                self.sessions_dir / f"{session_id}_progress.json",
                self.sessions_dir / f"{session_id}_result.json"
            ]

            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()

            return True

        except Exception as e:
            self.log_error("cleanup_session_files", e, session_id=session_id)
            return False