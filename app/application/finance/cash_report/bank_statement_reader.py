"""
Bank Statement Reader for Cash Report automation.
Reads parsed Excel files from the bank statement parser.
"""
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import openpyxl
import pandas as pd

from app.shared.utils.logging_config import get_logger
from .movement_data_writer import MovementTransaction

logger = get_logger(__name__)


class BankStatementReader:
    """
    Reads parsed bank statement Excel files and converts to MovementTransaction objects.

    Expected input: Excel file with "Template details" sheet from bank statement parser.

    Sheet columns (0-indexed):
    0: External ID
    1: External ID BSM Daily
    2: Bank Statement Daily
    3: Name (*)
    4: Bank Code (*)
    5: Bank Account Number (*)
    6: TRANS ID
    7: Trans Date (*)
    8: Description (*)
    9: Currency(*)
    10: DEBIT (*)
    11: CREDIT (*)
    12: Amount
    13: Type
    14: Balance
    15: PARTNER
    16: PARTNER ACCOUNT
    17: PARTNER BANK ID
    """

    # Column indices in Template details sheet
    COL_BANK_CODE = 4  # Bank Code (*)
    COL_ACCOUNT_NUMBER = 5  # Bank Account Number (*)
    COL_TRANS_DATE = 7  # Trans Date (*)
    COL_DESCRIPTION = 8  # Description (*)
    COL_CURRENCY = 9  # Currency(*)
    COL_DEBIT = 10  # DEBIT (*)
    COL_CREDIT = 11  # CREDIT (*)

    # Sheet name
    SHEET_NAME = "Template details"

    def __init__(self):
        pass

    def read_from_file(
        self,
        file_path: Union[str, Path],
        source_name: Optional[str] = None,
    ) -> List[MovementTransaction]:
        """
        Read transactions from a parsed bank statement Excel file.

        Args:
            file_path: Path to the Excel file
            source_name: Source name to use (default: filename)

        Returns:
            List of MovementTransaction objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if source_name is None:
            source_name = file_path.name

        with open(file_path, 'rb') as f:
            content = f.read()

        return self.read_from_bytes(content, source_name)

    def read_from_bytes(
        self,
        content: bytes,
        source_name: str = "uploaded_file",
    ) -> List[MovementTransaction]:
        """
        Read transactions from Excel bytes content.

        Args:
            content: Excel file content as bytes
            source_name: Source name for the transactions

        Returns:
            List of MovementTransaction objects (without Nature - to be classified later)
        """
        try:
            # Try reading with openpyxl first for better date handling
            wb = openpyxl.load_workbook(BytesIO(content), data_only=True)

            # Find the Template details sheet
            sheet_name = self._find_sheet(wb)
            if not sheet_name:
                wb.close()
                raise ValueError(f"Sheet '{self.SHEET_NAME}' not found in the Excel file")

            ws = wb[sheet_name]
            transactions = []

            # Skip header row (row 1)
            for row_num in range(2, ws.max_row + 1):
                try:
                    tx = self._parse_row_openpyxl(ws, row_num, source_name)
                    if tx:
                        transactions.append(tx)
                except Exception as e:
                    logger.warning(f"Error parsing row {row_num}: {e}")
                    continue

            wb.close()

            logger.info(f"Read {len(transactions)} transactions from {source_name}")
            return transactions

        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

    def _find_sheet(self, wb: openpyxl.Workbook) -> Optional[str]:
        """Find the Template details sheet (case-insensitive)."""
        for sheet_name in wb.sheetnames:
            if sheet_name.lower() == self.SHEET_NAME.lower():
                return sheet_name
        return None

    def _parse_row_openpyxl(
        self,
        ws,
        row_num: int,
        source_name: str,
    ) -> Optional[MovementTransaction]:
        """
        Parse a single row from openpyxl worksheet.

        Args:
            ws: openpyxl worksheet
            row_num: Row number (1-indexed)
            source_name: Source name

        Returns:
            MovementTransaction or None if row is empty/invalid
        """
        # Get cell values (column numbers are 1-indexed in openpyxl)
        bank_code = ws.cell(row=row_num, column=self.COL_BANK_CODE + 1).value
        account_number = ws.cell(row=row_num, column=self.COL_ACCOUNT_NUMBER + 1).value
        trans_date = ws.cell(row=row_num, column=self.COL_TRANS_DATE + 1).value
        description = ws.cell(row=row_num, column=self.COL_DESCRIPTION + 1).value
        debit = ws.cell(row=row_num, column=self.COL_DEBIT + 1).value
        credit = ws.cell(row=row_num, column=self.COL_CREDIT + 1).value

        # Skip empty rows
        if not account_number:
            return None

        # Skip if no transaction data
        if not trans_date and not debit and not credit:
            return None

        # Parse date
        parsed_date = self._parse_date(trans_date)
        if not parsed_date:
            return None

        # Parse amounts
        parsed_debit = self._parse_amount(debit)
        parsed_credit = self._parse_amount(credit)

        # Skip if both debit and credit are 0/None
        if not parsed_debit and not parsed_credit:
            return None

        return MovementTransaction(
            source=source_name,
            bank=str(bank_code) if bank_code else "",
            account=str(account_number),
            date=parsed_date,
            description=str(description) if description else "",
            debit=parsed_debit,
            credit=parsed_credit,
            nature="",  # Will be classified later
        )

    def _parse_date(self, value) -> Optional[date]:
        """Parse date from various formats."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

            # Try various date formats
            formats = [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%m-%d-%Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue

            logger.warning(f"Could not parse date: {value}")
            return None

        return None

    def _parse_amount(self, value) -> Optional[Decimal]:
        """Parse amount to Decimal."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            if value == 0:
                return None
            return Decimal(str(value))

        if isinstance(value, Decimal):
            if value == 0:
                return None
            return value

        if isinstance(value, str):
            value = value.strip()
            if not value or value == "0":
                return None

            # Remove commas and spaces
            value = value.replace(",", "").replace(" ", "")

            try:
                result = Decimal(value)
                return result if result != 0 else None
            except:
                return None

        return None

    def filter_by_date_range(
        self,
        transactions: List[MovementTransaction],
        start_date: date,
        end_date: date,
    ) -> Tuple[List[MovementTransaction], int]:
        """
        Filter transactions by date range.

        Args:
            transactions: List of transactions
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Tuple of (filtered_transactions, skipped_count)
        """
        filtered = []
        skipped = 0
        skipped_details = []

        logger.info(f"Filtering transactions with date range: {start_date} to {end_date}")

        for tx in transactions:
            if tx.date is None:
                skipped += 1
                skipped_details.append(f"No date: {tx.description[:50] if tx.description else 'N/A'}")
                continue

            if start_date <= tx.date <= end_date:
                filtered.append(tx)
            else:
                skipped += 1
                skipped_details.append(f"{tx.date.strftime('%d/%m/%Y')} (outside range): {tx.description[:50] if tx.description else 'N/A'}")

        logger.info(f"Filtered {len(filtered)} transactions, skipped {skipped} outside date range")
        if skipped_details:
            logger.info(f"Skipped transactions: {skipped_details[:10]}")  # Log first 10
        return filtered, skipped

    def get_summary(self, transactions: List[MovementTransaction]) -> dict:
        """
        Get summary statistics for a list of transactions.

        Returns:
            Dict with summary statistics
        """
        if not transactions:
            return {
                "total_transactions": 0,
                "total_debit": 0,
                "total_credit": 0,
                "unique_accounts": 0,
                "unique_banks": 0,
                "date_range": None,
            }

        total_debit = sum(tx.debit or Decimal(0) for tx in transactions)
        total_credit = sum(tx.credit or Decimal(0) for tx in transactions)
        unique_accounts = len(set(tx.account for tx in transactions))
        unique_banks = len(set(tx.bank for tx in transactions))
        dates = [tx.date for tx in transactions if tx.date]
        date_range = (min(dates), max(dates)) if dates else None

        return {
            "total_transactions": len(transactions),
            "total_debit": float(total_debit),
            "total_credit": float(total_credit),
            "unique_accounts": unique_accounts,
            "unique_banks": unique_banks,
            "date_range": {
                "start": date_range[0].isoformat() if date_range else None,
                "end": date_range[1].isoformat() if date_range else None,
            }
        }
