"""Base class for all bank statement parsers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance


class BaseBankParser(ABC):
    """Abstract base class for bank-specific parsers."""

    @property
    @abstractmethod
    def bank_name(self) -> str:
        """Return the bank name (e.g., 'ACB', 'OCB')."""
        pass

    @abstractmethod
    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if this parser can handle the given file.

        Args:
            file_bytes: Excel file as binary

        Returns:
            True if this parser recognizes the bank format
        """
        pass

    @abstractmethod
    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse transactions from bank statement.

        Args:
            file_bytes: Excel file as binary
            file_name: Original file name

        Returns:
            List of standardized transactions
        """
        pass

    @abstractmethod
    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse balance information from bank statement.

        Args:
            file_bytes: Excel file as binary
            file_name: Original file name

        Returns:
            Balance information or None if not found
        """
        pass

    # ========== OCR Text Parsing Methods (Optional - Override in subclass) ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if this parser can handle the given OCR text.

        Args:
            text: OCR extracted text from AI Builder

        Returns:
            True if this parser recognizes the text format.
            Default implementation returns False - override in subclass.
        """
        return False

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse transactions from OCR text.

        Args:
            text: OCR extracted text
            file_name: Original file name

        Returns:
            List of standardized transactions.
            Default implementation returns empty list - override in subclass.
        """
        return []

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse balance information from OCR text.

        Args:
            text: OCR extracted text
            file_name: Original file name

        Returns:
            Balance information or None if not found.
            Default implementation returns None - override in subclass.
        """
        return None

    # Helper methods available to all parsers

    @staticmethod
    def to_text(value) -> str:
        """Convert any value to text."""
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def fix_number(value) -> Optional[float]:
        """Convert value to number, handling Vietnamese formatting."""
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        # Remove commas and spaces
        cleaned = txt.replace(",", "").replace(" ", "")
        # Keep only numbers, decimal, and minus
        selected = ''.join(c for c in cleaned if c.isdigit() or c in ['-', '.'])

        if not selected:
            return None

        try:
            return float(selected)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def fix_date(value):
        """Convert value to date."""
        if value is None or pd.isna(value):
            return None

        # If already a datetime/date object
        if hasattr(value, 'date'):
            return value.date()

        # Try parsing text
        txt = str(value).strip()
        if ' ' in txt:
            txt = txt.split(' ')[0]  # Take date part only

        try:
            parsed = pd.to_datetime(txt, errors='coerce')
            if pd.notna(parsed):
                return parsed.date()
        except:
            pass

        return None
