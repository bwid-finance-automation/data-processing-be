"""
Master Template Manager for Cash Report automation.
Handles copying, configuring, and managing the master Excel template.
"""
import os
import shutil
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any

import openpyxl
from openpyxl.utils import get_column_letter

from app.shared.utils.logging_config import get_logger
from .excel_com_handler import ExcelCOMHandler

logger = get_logger(__name__)

# Flag to use COM automation (preserves all Excel features) vs openpyxl
USE_COM_AUTOMATION = True

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates" / "cash_report"
WORKING_DIR = BASE_DIR / "working" / "cash_report"

# Master template filename - use original file directly to preserve all formulas
MASTER_TEMPLATE_FILENAME = "master_data.xlsx"

# In-memory template cache (shared across all instances)
_template_cache: Optional[bytes] = None


def _load_template_to_cache() -> bytes:
    """Load master template into memory cache (lazy loading)."""
    global _template_cache
    if _template_cache is None:
        template_path = TEMPLATES_DIR / MASTER_TEMPLATE_FILENAME
        if not template_path.exists():
            raise FileNotFoundError(f"Master template not found at {template_path}")

        with open(template_path, 'rb') as f:
            _template_cache = f.read()

        size_mb = len(_template_cache) / (1024 * 1024)
        logger.info(f"Loaded master template into memory cache ({size_mb:.2f} MB)")

    return _template_cache


def clear_template_cache() -> None:
    """Clear the template cache to force reload on next use."""
    global _template_cache
    _template_cache = None
    logger.info("Template cache cleared")


class MasterTemplateManager:
    """
    Manages the master Excel template for cash report generation.

    Responsibilities:
    - Copy master template to working directory for each session
    - Update configuration (dates, FX rate, period name)
    - Clear Movement sheet data rows
    - Reset session to clean state
    """

    def __init__(self):
        self.templates_dir = TEMPLATES_DIR
        self.working_dir = WORKING_DIR
        self.master_template_path = self.templates_dir / MASTER_TEMPLATE_FILENAME

        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get the working directory for a specific session."""
        return self.working_dir / session_id

    def _get_working_file_path(self, session_id: str) -> Path:
        """Get the path to the working Excel file for a session."""
        return self._get_session_dir(session_id) / "working_master.xlsx"

    def create_session(
        self,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal = Decimal("26175"),
        period_name: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new session with a fresh copy of the master template.

        Args:
            opening_date: Report period start date
            ending_date: Report period end date
            fx_rate: VND/USD exchange rate
            period_name: Period name (e.g., "W3-4Jan26")

        Returns:
            Dict with session_id and file path
        """
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Create session directory
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write template from memory cache (fast)
        working_file = self._get_working_file_path(session_id)

        template_bytes = _load_template_to_cache()
        with open(working_file, 'wb') as f:
            f.write(template_bytes)

        logger.info(f"Created session {session_id}, wrote template from memory cache")

        # Break external links to avoid "Update Links" prompt when downloading
        if USE_COM_AUTOMATION:
            handler = ExcelCOMHandler()
            handler.break_external_links(working_file)

        # Update configuration
        self._update_config(
            session_id=session_id,
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=fx_rate,
            period_name=period_name,
        )

        # Clear Movement data rows (keep headers and formula template)
        self._clear_movement_data(session_id)

        return {
            "session_id": session_id,
            "working_file": str(working_file),
            "config": {
                "opening_date": opening_date.isoformat(),
                "ending_date": ending_date.isoformat(),
                "fx_rate": float(fx_rate),
                "period_name": period_name,
            }
        }

    def _update_config(
        self,
        session_id: str,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
    ) -> None:
        """
        Update the Summary sheet configuration.

        Summary sheet structure:
        - B1: Date (ending date)
        - B3: FX rate (VND/USD)
        - B4: Week/Period name
        - B5: Opening date
        - B6: Ending date
        """
        working_file = self._get_working_file_path(session_id)

        if USE_COM_AUTOMATION:
            # Use COM automation to preserve all Excel features
            handler = ExcelCOMHandler()
            handler.update_config(
                file_path=working_file,
                opening_date=opening_date,
                ending_date=ending_date,
                fx_rate=fx_rate,
                period_name=period_name,
            )
            logger.info(f"Updated config for session {session_id} via COM")
        else:
            # Fallback to openpyxl (may lose some Excel features)
            wb = openpyxl.load_workbook(working_file)

            try:
                ws = wb["Summary"]

                # Update configuration cells
                ws["B1"] = ending_date  # Date
                ws["B3"] = float(fx_rate)  # FX rate
                ws["B4"] = period_name  # Period name
                ws["B5"] = opening_date  # Opening date
                ws["B6"] = ending_date  # Ending date

                wb.save(working_file)
                logger.info(f"Updated config for session {session_id}")

            finally:
                wb.close()

    def _clear_movement_data(self, session_id: str) -> None:
        """
        Clear Movement sheet data rows (row 4+), keeping:
        - Row 1: Legend
        - Row 2: Subtotals
        - Row 3: Headers
        - Row 4: Keep as formula template (but clear values)
        """
        working_file = self._get_working_file_path(session_id)

        if USE_COM_AUTOMATION:
            # Use COM automation to preserve all Excel features
            handler = ExcelCOMHandler()
            rows_cleared = handler.clear_movement_data(working_file)
            logger.info(f"Cleared {rows_cleared} Movement rows for session {session_id} via COM")
        else:
            # Fallback to openpyxl (may lose some Excel features)
            wb = openpyxl.load_workbook(working_file)

            try:
                ws = wb["Movement"]

                # Find the last row with data
                max_row = ws.max_row

                # Delete rows from 5 to max_row (keep row 4 as template)
                if max_row > 4:
                    ws.delete_rows(5, max_row - 4)
                    logger.info(f"Cleared Movement rows 5-{max_row} for session {session_id}")

                # Clear data values in row 4 (columns A-G, I) but keep formulas
                data_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I']
                for col in data_columns:
                    cell = ws[f"{col}4"]
                    # Only clear if it's not a formula
                    if cell.value and not str(cell.value).startswith('='):
                        cell.value = None

                wb.save(working_file)
                logger.info(f"Cleared Movement data for session {session_id}")

            finally:
                wb.close()

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """
        Reset a session to clean state (re-copy from master template).
        Preserves the session configuration.
        """
        working_file = self._get_working_file_path(session_id)

        if not working_file.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        # Read current config before reset
        wb = openpyxl.load_workbook(working_file, data_only=True)
        try:
            ws = wb["Summary"]
            opening_date = ws["B5"].value
            ending_date = ws["B6"].value
            fx_rate = ws["B3"].value
            period_name = ws["B4"].value
        finally:
            wb.close()

        # Convert to proper types
        if isinstance(opening_date, datetime):
            opening_date = opening_date.date()
        elif isinstance(opening_date, str):
            try:
                opening_date = date.fromisoformat(opening_date[:10])
            except:
                opening_date = date.today()
        elif opening_date is None:
            opening_date = date.today()

        if isinstance(ending_date, datetime):
            ending_date = ending_date.date()
        elif isinstance(ending_date, str):
            try:
                ending_date = date.fromisoformat(ending_date[:10])
            except:
                ending_date = date.today()
        elif ending_date is None:
            ending_date = date.today()

        if fx_rate is None:
            fx_rate = 26175
        if period_name is None:
            period_name = ""

        # Kill any lingering Excel processes that might be locking the file
        if USE_COM_AUTOMATION:
            from .excel_com_handler import kill_excel_processes
            kill_excel_processes()
            import time
            time.sleep(0.5)  # Wait for file to be released

        # Re-copy master template from memory cache
        template_bytes = _load_template_to_cache()
        with open(working_file, 'wb') as f:
            f.write(template_bytes)

        # Break external links to avoid "Update Links" prompt
        if USE_COM_AUTOMATION:
            handler = ExcelCOMHandler()
            handler.break_external_links(working_file)

        # Re-apply config
        self._update_config(
            session_id=session_id,
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=Decimal(str(fx_rate)),
            period_name=period_name,
        )

        # Clear Movement data
        self._clear_movement_data(session_id)

        logger.info(f"Reset session {session_id}")

        return {
            "session_id": session_id,
            "status": "reset",
            "config": {
                "opening_date": opening_date.isoformat() if opening_date else None,
                "ending_date": ending_date.isoformat() if ending_date else None,
                "fx_rate": float(fx_rate),
                "period_name": period_name,
            }
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its working files."""
        session_dir = self._get_session_dir(session_id)

        if session_dir.exists():
            # Kill any lingering Excel processes that might be locking the file
            if USE_COM_AUTOMATION:
                from .excel_com_handler import kill_excel_processes
                kill_excel_processes()
                import time
                time.sleep(0.5)  # Wait for file to be released

            shutil.rmtree(session_dir)
            logger.info(f"Deleted session {session_id}")
            return True

        return False

    def get_session_info(self, session_id: str, include_movement_count: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.

        Args:
            session_id: The session ID
            include_movement_count: If True, count Movement rows (slower). Set False for listing.
        """
        working_file = self._get_working_file_path(session_id)

        if not working_file.exists():
            return None

        # Use read_only mode for faster loading
        wb = openpyxl.load_workbook(working_file, data_only=True, read_only=True)
        try:
            # Get config from Summary
            summary_ws = wb["Summary"]
            # In read_only mode, we need to iterate to specific cells
            opening_date = None
            ending_date = None
            fx_rate = None
            period_name = None

            for row in summary_ws.iter_rows(min_row=1, max_row=6, min_col=2, max_col=2):
                for cell in row:
                    if cell.row == 3:
                        fx_rate = cell.value
                    elif cell.row == 4:
                        period_name = cell.value
                    elif cell.row == 5:
                        opening_date = cell.value
                    elif cell.row == 6:
                        ending_date = cell.value

            # Count Movement data rows only if requested
            movement_rows = 0
            if include_movement_count:
                movement_ws = wb["Movement"]
                for row in movement_ws.iter_rows(min_row=4, min_col=3, max_col=3):
                    for cell in row:
                        if cell.value:
                            movement_rows += 1

            # Helper to extract date properly (handle timezone offset from Excel)
            def extract_date(value) -> str:
                if value is None:
                    return None
                if isinstance(value, datetime):
                    # Excel stores dates as datetime. If time is 17:00:00 (UTC),
                    # it's actually midnight in Vietnam (UTC+7). Add offset to get correct date.
                    if value.hour == 17 and value.minute == 0 and value.second == 0:
                        from datetime import timedelta
                        corrected = value + timedelta(hours=7)
                        return corrected.date().isoformat()
                    return value.date().isoformat()
                if isinstance(value, date):
                    return value.isoformat()
                return str(value) if value else None

            return {
                "session_id": session_id,
                "working_file": str(working_file),
                "file_size_mb": round(working_file.stat().st_size / (1024 * 1024), 2),
                "movement_rows": movement_rows,
                "config": {
                    "opening_date": extract_date(opening_date),
                    "ending_date": extract_date(ending_date),
                    "fx_rate": float(fx_rate) if fx_rate else None,
                    "period_name": period_name,
                }
            }

        finally:
            wb.close()

    def get_working_file_path(self, session_id: str) -> Optional[Path]:
        """Get the working file path for a session (if exists)."""
        working_file = self._get_working_file_path(session_id)
        return working_file if working_file.exists() else None

    def list_sessions(self) -> list:
        """List all active sessions (fast - skips Movement row counting)."""
        sessions = []

        if not self.working_dir.exists():
            return sessions

        for session_dir in self.working_dir.iterdir():
            if session_dir.is_dir():
                session_id = session_dir.name
                # Skip Movement row counting for faster listing
                info = self.get_session_info(session_id, include_movement_count=False)
                if info:
                    sessions.append(info)

        return sessions
