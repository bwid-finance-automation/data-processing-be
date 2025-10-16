# app/utils/log_capture.py
"""Log capture utility for streaming analysis progress."""

import sys
import queue
from collections import deque
from typing import Optional

class LogCapture:
    """Custom class to capture all print output and stream it to frontend."""

    def __init__(self, session_id: str, max_history: int = 50):
        self.session_id = session_id
        self.queue = queue.Queue()
        self.history = deque(maxlen=max_history)  # Keep last N logs for polling

    def write(self, message: str):
        """Write message to queue and stdout."""
        if message.strip():  # Only send non-empty messages
            stripped_msg = message.strip()
            self.queue.put(stripped_msg)
            # Store in history buffer (but not progress markers as they're tracked separately)
            if not stripped_msg.startswith("__PROGRESS__") and not stripped_msg.startswith("__"):
                self.history.append(stripped_msg)
        # Also write to original stdout for server logs
        sys.__stdout__.write(message)

    def flush(self):
        """Flush stdout."""
        sys.__stdout__.flush()

    def get_recent_logs(self, count: int = 10):
        """Get the most recent logs from history."""
        return list(self.history)[-count:]