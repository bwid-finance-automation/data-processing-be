# app/utils/log_capture.py
"""Log capture utility for streaming analysis progress."""

import sys
import queue
from typing import Optional

class LogCapture:
    """Custom class to capture all print output and stream it to frontend."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queue = queue.Queue()

    def write(self, message: str):
        """Write message to queue and stdout."""
        if message.strip():  # Only send non-empty messages
            self.queue.put(message.strip())
        # Also write to original stdout for server logs
        sys.__stdout__.write(message)

    def flush(self):
        """Flush stdout."""
        sys.__stdout__.flush()