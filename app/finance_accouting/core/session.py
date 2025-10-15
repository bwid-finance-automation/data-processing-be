"""
Session management for temporary file storage
"""
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from .config import get_settings
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages temporary session-based file storage"""

    # Store session last activity time
    _sessions = {}

    # Session timeout (in minutes)
    SESSION_TIMEOUT = 60  # 1 hour

    @staticmethod
    def create_session() -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        SessionManager._sessions[session_id] = datetime.now()

        # Create session directories
        session_path = SessionManager.get_session_path(session_id)
        (session_path / "input").mkdir(parents=True, exist_ok=True)
        (session_path / "master_data").mkdir(parents=True, exist_ok=True)
        (session_path / "output").mkdir(parents=True, exist_ok=True)
        (session_path / "logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"Created session: {session_id}")
        return session_id

    @staticmethod
    def get_session_path(session_id: str) -> Path:
        """Get the base path for a session"""
        # Use the data directory from project root
        base_dir = Path(__file__).parent.parent.parent.parent
        upload_dir = base_dir / "data" / "uploads"
        return upload_dir / "sessions" / session_id

    @staticmethod
    def get_input_dir(session_id: str) -> Path:
        """Get input directory for session"""
        return SessionManager.get_session_path(session_id) / "input"

    @staticmethod
    def get_master_data_dir(session_id: str) -> Path:
        """Get master data directory for session"""
        return SessionManager.get_session_path(session_id) / "master_data"

    @staticmethod
    def get_output_dir(session_id: str) -> Path:
        """Get output directory for session"""
        return SessionManager.get_session_path(session_id) / "output"

    @staticmethod
    def get_log_dir(session_id: str) -> Path:
        """Get log directory for session"""
        return SessionManager.get_session_path(session_id) / "logs"

    @staticmethod
    def update_activity(session_id: str):
        """Update last activity time for session"""
        if session_id in SessionManager._sessions:
            SessionManager._sessions[session_id] = datetime.now()

    @staticmethod
    def validate_session(session_id: str) -> bool:
        """Check if session exists and is valid"""
        if session_id not in SessionManager._sessions:
            return False

        # Check if session expired
        last_activity = SessionManager._sessions[session_id]
        if datetime.now() - last_activity > timedelta(minutes=SessionManager.SESSION_TIMEOUT):
            logger.info(f"Session expired: {session_id}")
            SessionManager.cleanup_session(session_id)
            return False

        return True

    @staticmethod
    def cleanup_session(session_id: str):
        """Delete all files and folders for a session"""
        try:
            session_path = SessionManager.get_session_path(session_id)
            if session_path.exists():
                shutil.rmtree(session_path)
                logger.info(f"Cleaned up session: {session_id}")

            # Remove from active sessions
            if session_id in SessionManager._sessions:
                del SessionManager._sessions[session_id]
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

    @staticmethod
    def cleanup_expired_sessions():
        """Clean up all expired sessions"""
        expired = []
        now = datetime.now()

        for session_id, last_activity in SessionManager._sessions.items():
            if now - last_activity > timedelta(minutes=SessionManager.SESSION_TIMEOUT):
                expired.append(session_id)

        for session_id in expired:
            SessionManager.cleanup_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    @staticmethod
    def cleanup_all_sessions():
        """Clean up all sessions (for shutdown)"""
        session_ids = list(SessionManager._sessions.keys())
        for session_id in session_ids:
            SessionManager.cleanup_session(session_id)
        logger.info("Cleaned up all sessions")


# Ensure sessions directory exists on import
def _ensure_sessions_dir():
    """Ensure sessions directory exists"""
    try:
        base_dir = Path(__file__).parent.parent.parent.parent
        upload_dir = base_dir / "data" / "uploads"
        sessions_dir = upload_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create sessions directory: {e}")

_ensure_sessions_dir()
