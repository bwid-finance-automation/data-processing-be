"""
Progress event store for cash report upload streaming.
Thread-safe queue-based progress reporting per session.
"""
import json
import queue
from typing import Dict, Optional, Any

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProgressEvent:
    """Typed progress event for SSE streaming."""

    def __init__(
        self,
        event_type: str,
        step: str,
        message: str,
        detail: str = "",
        percentage: int = 0,
        data: Optional[Dict[str, Any]] = None,
        message_key: str = "",
        message_params: Optional[Dict[str, Any]] = None,
        detail_key: str = "",
        detail_params: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type  # step_start | step_update | step_complete | complete | error
        self.step = step              # reading | filtering | classifying | writing | done | error
        self.message = message
        self.detail = detail
        self.percentage = percentage
        self.data = data or {}
        self.message_key = message_key
        self.message_params = message_params or {}
        self.detail_key = detail_key
        self.detail_params = detail_params or {}

    def to_json(self) -> str:
        payload = {
            "type": self.event_type,
            "step": self.step,
            "message": self.message,
            "detail": self.detail,
            "percentage": self.percentage,
            "data": self.data,
        }
        if self.message_key:
            payload["message_key"] = self.message_key
            payload["message_params"] = self.message_params
        if self.detail_key:
            payload["detail_key"] = self.detail_key
            payload["detail_params"] = self.detail_params
        return json.dumps(payload)


class ProgressStore:
    """Module-level store for progress queues, keyed by session_id."""

    def __init__(self):
        self._queues: Dict[str, queue.Queue] = {}

    def create(self, session_id: str) -> queue.Queue:
        """Create a new progress queue for a session."""
        q = queue.Queue()
        self._queues[session_id] = q
        return q

    def get(self, session_id: str) -> Optional[queue.Queue]:
        """Get the progress queue for a session."""
        return self._queues.get(session_id)

    def emit(self, session_id: str, event: ProgressEvent):
        """Emit a progress event to the session's queue."""
        q = self._queues.get(session_id)
        if q:
            q.put(event.to_json())

    def cleanup(self, session_id: str):
        """Remove the progress queue for a session."""
        if session_id in self._queues:
            del self._queues[session_id]


# Module-level singleton
progress_store = ProgressStore()
