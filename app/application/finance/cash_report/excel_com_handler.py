"""
Excel COM Automation Handler for Cash Report.
Uses win32com to control Excel directly, preserving all formulas, drawings, and external links.
"""
import os
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any
import pythoncom
import pywintypes

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def to_excel_date(d):
    """Convert Python date/datetime to pywintypes.Time for COM."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return pywintypes.Time(d.timetuple())
    if isinstance(d, date):
        return pywintypes.Time(d.timetuple())
    return d


def kill_excel_processes():
    """Kill any lingering Excel processes that might be locking files."""
    import subprocess
    try:
        subprocess.run(
            ['taskkill', '/F', '/IM', 'EXCEL.EXE'],
            capture_output=True,
            timeout=5
        )
        logger.info("Killed lingering Excel processes")
    except Exception as e:
        # Process might not exist, that's OK
        pass


class ExcelCOMHandler:
    """
    Handles Excel operations using COM automation.
    Preserves all Excel features including formulas, drawings, external links.
    """

    def __init__(self):
        self.excel = None
        self.workbook = None

    def _init_com(self):
        """Initialize COM for current thread."""
        pythoncom.CoInitialize()

    def _uninit_com(self):
        """Uninitialize COM for current thread."""
        try:
            pythoncom.CoUninitialize()
        except:
            pass

    def _get_excel_app(self):
        """Get or create Excel application instance."""
        import win32com.client

        if self.excel is None:
            self.excel = win32com.client.Dispatch("Excel.Application")
            self.excel.Visible = False  # Run in background
            self.excel.DisplayAlerts = False  # Suppress alerts
            self.excel.ScreenUpdating = False  # Disable screen updates for speed
        return self.excel

    def _close_excel(self):
        """Close Excel application and release file locks."""
        import gc
        try:
            if self.workbook:
                try:
                    self.workbook.Close(SaveChanges=False)
                except:
                    pass
                self.workbook = None

            if self.excel:
                try:
                    self.excel.Quit()
                except:
                    pass
                self.excel = None

            # Force garbage collection to release COM objects
            gc.collect()

        except Exception as e:
            logger.warning(f"Error closing Excel: {e}")

    def update_config(
        self,
        file_path: Path,
        opening_date: date,
        ending_date: date,
        fx_rate: Decimal,
        period_name: str,
    ) -> None:
        """
        Update Summary sheet configuration using COM.

        Summary sheet structure:
        - B1: Date (ending date)
        - B3: FX rate (VND/USD)
        - B4: Week/Period name
        - B5: Opening date
        - B6: Ending date
        """
        self._init_com()
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path))

            ws = self.workbook.Sheets("Summary")

            # Update configuration cells (convert dates for COM)
            ws.Range("B1").Value = to_excel_date(ending_date)
            ws.Range("B3").Value = float(fx_rate)
            ws.Range("B4").Value = period_name
            ws.Range("B5").Value = to_excel_date(opening_date)
            ws.Range("B6").Value = to_excel_date(ending_date)

            self.workbook.Save()
            logger.info(f"Updated config via COM: {file_path.name}")

        except Exception as e:
            logger.error(f"COM error updating config: {e}")
            raise
        finally:
            self._close_excel()
            self._uninit_com()

    def clear_movement_data(self, file_path: Path) -> int:
        """
        Clear Movement sheet data rows (row 5+), keeping:
        - Row 1: Legend
        - Row 2: Subtotals
        - Row 3: Headers
        - Row 4: Keep as formula template (but clear input values)

        Returns:
            Number of rows cleared
        """
        self._init_com()
        rows_cleared = 0
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path))

            ws = self.workbook.Sheets("Movement")

            # Find last row with data
            last_row = ws.Cells(ws.Rows.Count, 3).End(-4162).Row  # -4162 = xlUp

            # Delete rows from 5 to last_row
            if last_row > 4:
                rows_cleared = last_row - 4
                ws.Rows(f"5:{last_row}").Delete()
                logger.info(f"Cleared {rows_cleared} rows via COM")

            # Clear input values in row 4 (columns A-G, I) but keep formulas
            input_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I']
            for col in input_cols:
                cell = ws.Range(f"{col}4")
                # Check if it's not a formula
                if cell.Value is not None and not str(cell.Formula).startswith('='):
                    cell.Value = None

            self.workbook.Save()

        except Exception as e:
            logger.error(f"COM error clearing movement: {e}")
            raise
        finally:
            self._close_excel()
            self._uninit_com()

        return rows_cleared

    def append_transactions(
        self,
        file_path: Path,
        transactions: List[Dict[str, Any]],
    ) -> tuple:
        """
        Append transactions to Movement sheet using COM.

        Args:
            file_path: Path to Excel file
            transactions: List of transaction dicts with keys:
                source, bank, account, date, description, debit, credit, nature

        Returns:
            Tuple of (rows_added, total_rows)
        """
        if not transactions:
            return 0, 0

        self._init_com()
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path))

            ws = self.workbook.Sheets("Movement")

            # Find last row with data in column C
            last_row = 3  # Default to header row
            for row in range(4, ws.UsedRange.Rows.Count + 5):
                if ws.Cells(row, 3).Value:  # Column C
                    last_row = row
                else:
                    # Check a few more rows in case of gaps
                    has_more = False
                    for check_row in range(row + 1, min(row + 10, ws.UsedRange.Rows.Count + 5)):
                        if ws.Cells(check_row, 3).Value:
                            has_more = True
                            break
                    if not has_more:
                        break

            # Determine start row
            if last_row < 4:
                start_row = 4
            else:
                start_row = last_row + 1

            # Get formula templates from row 4
            formula_cols = ['H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
            formula_templates = {}
            for col in formula_cols:
                formula = ws.Range(f"{col}4").Formula
                if formula and str(formula).startswith('='):
                    formula_templates[col] = formula

            # Write transactions
            rows_added = 0
            for i, tx in enumerate(transactions):
                row = start_row + i

                # Write input columns
                ws.Cells(row, 1).Value = tx.get('source', '')  # A
                ws.Cells(row, 2).Value = tx.get('bank', '')  # B
                # Account number - format as text to avoid scientific notation
                account_cell = ws.Cells(row, 3)
                account_cell.NumberFormat = "@"  # Text format
                account_cell.Value = str(tx.get('account', ''))  # C

                # Handle date (convert for COM)
                tx_date = tx.get('date')
                if tx_date:
                    ws.Cells(row, 4).Value = to_excel_date(tx_date)

                ws.Cells(row, 5).Value = tx.get('description', '')  # E

                # Handle amounts
                debit = tx.get('debit')
                credit = tx.get('credit')
                ws.Cells(row, 6).Value = float(debit) if debit else None  # F
                ws.Cells(row, 7).Value = float(credit) if credit else None  # G
                ws.Cells(row, 9).Value = tx.get('nature', '')  # I

                # Copy and adjust formulas
                for col, formula in formula_templates.items():
                    adjusted = self._adjust_formula(formula, 4, row)
                    col_num = ord(col) - ord('A') + 1
                    ws.Cells(row, col_num).Formula = adjusted

                rows_added += 1

            self.workbook.Save()

            total_rows = (start_row - 4) + rows_added
            logger.info(f"Appended {rows_added} transactions via COM, total: {total_rows}")

            return rows_added, total_rows

        except Exception as e:
            logger.error(f"COM error appending transactions: {e}")
            raise
        finally:
            self._close_excel()
            self._uninit_com()

    def _adjust_formula(self, formula: str, source_row: int, target_row: int) -> str:
        """
        Adjust formula row references from source_row to target_row.
        Handles both relative and absolute ($) references.
        """
        if not formula or not formula.startswith('='):
            return formula

        import re

        # Pattern to match cell references
        pattern = r'(\$?[A-Z]+)(\$?)(\d+)'

        def replace_ref(match):
            col = match.group(1)
            dollar = match.group(2)
            row_num = int(match.group(3))

            # If row is absolute ($), don't adjust
            if dollar == '$':
                return f"{col}${row_num}"

            # Only adjust if row matches source_row
            if row_num == source_row:
                return f"{col}{target_row}"

            return match.group(0)

        return re.sub(pattern, replace_ref, formula)

    def break_external_links(self, file_path: Path) -> int:
        """
        Break all external links in the workbook.
        This removes the "Update Links" prompt when opening the file.

        Returns:
            Number of links broken
        """
        self._init_com()
        links_broken = 0
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path))

            # Get all external links
            links = self.workbook.LinkSources(1)  # 1 = xlExcelLinks

            if links:
                for link in links:
                    try:
                        self.workbook.BreakLink(link, 1)  # 1 = xlLinkTypeExcelLinks
                        links_broken += 1
                        logger.info(f"Broke external link: {link}")
                    except Exception as e:
                        logger.warning(f"Could not break link {link}: {e}")

            self.workbook.Save()
            logger.info(f"Broke {links_broken} external links in {file_path.name}")

        except Exception as e:
            # Links might not exist, that's OK
            if "NoneType" not in str(e) and "object has no attribute" not in str(e):
                logger.warning(f"Error breaking external links: {e}")
        finally:
            self._close_excel()
            self._uninit_com()

        return links_broken

    def remove_rows_by_source(self, file_path: Path, source_name: str) -> int:
        """
        Remove all rows from Movement sheet that have matching source name (column A).
        Used to implement "overwrite on duplicate file upload".

        Args:
            file_path: Path to Excel file
            source_name: Source name to match in column A

        Returns:
            Number of rows removed
        """
        self._init_com()
        rows_removed = 0
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path))

            ws = self.workbook.Sheets("Movement")

            # Find all rows with matching source (iterate from bottom to top to avoid index shifting)
            last_row = ws.Cells(ws.Rows.Count, 3).End(-4162).Row  # -4162 = xlUp

            # Collect rows to delete (from bottom to top)
            rows_to_delete = []
            for row in range(last_row, 3, -1):  # Start from last row, go up to row 4
                cell_value = ws.Cells(row, 1).Value  # Column A (source)
                if cell_value and str(cell_value).strip() == source_name.strip():
                    rows_to_delete.append(row)

            # Delete rows from bottom to top
            for row in rows_to_delete:
                ws.Rows(row).Delete()
                rows_removed += 1

            if rows_removed > 0:
                self.workbook.Save()
                logger.info(f"Removed {rows_removed} rows with source '{source_name}' via COM")

        except Exception as e:
            logger.error(f"COM error removing rows by source: {e}")
            raise
        finally:
            self._close_excel()
            self._uninit_com()

        return rows_removed

    def get_movement_row_count(self, file_path: Path) -> int:
        """Get the number of data rows in Movement sheet."""
        self._init_com()
        try:
            excel = self._get_excel_app()
            self.workbook = excel.Workbooks.Open(str(file_path), ReadOnly=True)

            ws = self.workbook.Sheets("Movement")

            # Count rows with data in column C starting from row 4
            count = 0
            for row in range(4, ws.UsedRange.Rows.Count + 5):
                if ws.Cells(row, 3).Value:
                    count += 1
                else:
                    # Check a few more rows
                    has_more = False
                    for check_row in range(row + 1, min(row + 10, ws.UsedRange.Rows.Count + 5)):
                        if ws.Cells(check_row, 3).Value:
                            has_more = True
                            break
                    if not has_more:
                        break

            return count

        except Exception as e:
            logger.error(f"COM error getting row count: {e}")
            return 0
        finally:
            self._close_excel()
            self._uninit_com()


# Singleton instance for reuse
_handler: Optional[ExcelCOMHandler] = None


def get_excel_handler() -> ExcelCOMHandler:
    """Get singleton ExcelCOMHandler instance."""
    global _handler
    if _handler is None:
        _handler = ExcelCOMHandler()
    return _handler
