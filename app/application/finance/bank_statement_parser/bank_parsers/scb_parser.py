"""Parser for SCB (Standard Chartered Bank Vietnam) statements."""

import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class SCBParser(BaseBankParser):
    """Parser for Standard Chartered Bank statements - supports PDF (via OCR)."""

    @property
    def bank_name(self) -> str:
        return "SCB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is Standard Chartered bank statement (Excel format).

        SCB markers:
        - "STANDARD CHARTERED" or "Standard Chartered Bank"
        - "STATEMENT OF ACCOUNT" / "SAO KÊ TÀI KHOẢN"
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 20 rows
            top_20 = df.head(20)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_20.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_20[col]])

            txt = " ".join(all_text)

            # Check for Standard Chartered markers
            has_scb = "STANDARD CHARTERED" in txt
            has_statement = "SAO KÊ" in txt or "STATEMENT OF ACCOUNT" in txt

            return has_scb and has_statement

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """Parse SCB transactions from Excel file."""
        # TODO: Implement Excel parsing if needed
        return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """Parse SCB balance from Excel file."""
        # TODO: Implement Excel parsing if needed
        return None

    # ========== OCR Text Parsing Methods (PDF via Gemini) ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if OCR text is from Standard Chartered Bank statement.

        Detection:
        - Contains "Standard Chartered Bank" or "Standard Chartered"
        - Contains statement markers
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for Standard Chartered markers
        has_scb = "standard chartered" in text_lower

        # Check for statement markers
        has_statement = any(marker in text_lower for marker in [
            "sao kê", "statement of account", "statement"
        ])

        return has_scb and has_statement

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse Standard Chartered transactions from OCR text.

        SCB PDF format:
        - Multiple account sections
        - Date format: "DD MMM YY" (e.g., "25 NOV 25")
        - Entry date, Value date on same line
        - Description on same or next line
        - DEBITS and CREDITS columns
        """
        all_transactions = []

        # Split into account sections
        sections = self._split_account_sections(text)

        for section in sections:
            transactions = self._parse_section_transactions(section)
            all_transactions.extend(transactions)

        logger.info(f"SCB: Parsed {len(all_transactions)} transactions total")
        return all_transactions

    def _split_account_sections(self, text: str) -> List[dict]:
        """
        Split OCR text into account sections.

        Each section starts with "STATEMENT OF ACCOUNT" and contains:
        - Currency and account number (e.g., "USD 37443435101" or "VND 90443435101")
        """
        sections = []
        lines = text.split('\n')

        current_section = None
        current_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Check for new section start
            if "STATEMENT OF ACCOUNT" in line_stripped.upper():
                # Save previous section
                if current_section and current_lines:
                    current_section['lines'] = current_lines
                    sections.append(current_section)

                current_section = {'acc_no': None, 'currency': 'VND', 'lines': []}
                current_lines = [line_stripped]
            elif current_section is not None:
                current_lines.append(line_stripped)

                # Extract account number and currency from format "USD 37443435101" or "VND 90443435101"
                if current_section['acc_no'] is None:
                    acc_match = re.match(r'^(USD|VND|EUR)\s+(\d{10,})', line_stripped)
                    if acc_match:
                        current_section['currency'] = acc_match.group(1)
                        current_section['acc_no'] = acc_match.group(2)

        # Save last section
        if current_section and current_lines:
            current_section['lines'] = current_lines
            sections.append(current_section)

        return sections

    def _parse_section_transactions(self, section: dict) -> List[BankTransaction]:
        """
        Parse transactions from a single account section.

        OCR Format (dates on separate lines):
        ```
        25 NOV 25              <- entry date
        25 NOV 25              <- value date
        100 0002 25/11/2025... <- reference line
        VAT CHARGES            <- description
        50,000                 <- amount (debit)
        ```
        """
        transactions = []
        lines = section.get('lines', [])
        acc_no = section.get('acc_no', '')
        currency = section.get('currency', 'VND')

        logger.info(f"SCB: Parsing section account={acc_no}, currency={currency}")

        # Skip patterns for non-transaction lines
        skip_patterns = [
            'statement of account', 'sao kê tài khoản',
            'page', 'branch', 'ledger bal', 'opening balance', 'closing balance',
            'total debits', 'total credits', 'entry', 'value',
            'description', 'debits', 'credits', 'nội dung',
            'ghi nợ', 'ghi có', 'số dư', 'ngày', 'thank you', 'cảm ơn',
            'note:', 'lưu ý', 'standard chartered', 'current account',
            'từ', 'đến', 'liability', 'floor', 'plaza', 'ward', 'district',
            'ho chi minh', 'development', 'investment', 'limited'
        ]

        # Date pattern: "DD MMM YY" (e.g., "25 NOV 25")
        date_pattern = re.compile(r'^(\d{1,2}\s+[A-Z]{3}\s+\d{2})$', re.IGNORECASE)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_lower = line.lower()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Skip header/footer/address lines
            if any(skip in line_lower for skip in skip_patterns):
                i += 1
                continue

            # Skip balance brought forward / carried over
            if 'balance brought forward' in line_lower or 'balance carried over' in line_lower:
                i += 1
                continue

            # Look for standalone date line (entry date)
            date_match = date_pattern.match(line)

            if date_match:
                entry_date_str = date_match.group(1)

                # Check if next line is also a date (value date)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    value_date_match = date_pattern.match(next_line)

                    if value_date_match:
                        # We have entry date + value date, now collect transaction data
                        j = i + 2  # Start after both date lines

                        description_parts = []
                        amounts = []

                        # Collect description and amounts from following lines
                        while j < len(lines):
                            data_line = lines[j].strip()
                            data_lower = data_line.lower()

                            # Stop if we hit another date (next transaction)
                            if date_pattern.match(data_line):
                                break

                            # Stop if we hit balance carried over or footer
                            if any(stop in data_lower for stop in ['balance carried', 'thank you', 'cảm ơn', 'note:']):
                                break

                            # Skip empty lines
                            if not data_line:
                                j += 1
                                continue

                            # Check if line is amount only (number with optional commas)
                            if re.match(r'^[\d,]+(?:\.\d{2})?$', data_line):
                                val = self._parse_ocr_number(data_line)
                                if val is not None:
                                    amounts.append(val)
                            else:
                                # It's part of description/reference
                                description_parts.append(data_line)

                            j += 1

                        # Build description from collected parts
                        description = ' '.join(description_parts)

                        # Filter out transaction amount from description if any amounts embedded
                        # (amounts should be on separate lines in this format)

                        # Determine debit/credit based on description
                        debit_val = None
                        credit_val = None

                        if amounts:
                            # First amount is typically the transaction amount
                            # Standard Chartered DEBITS = money out
                            desc_lower = description.lower()

                            # Check for credit indicators
                            if any(kw in desc_lower for kw in ['credit', 'deposit', 'transfer in', 'receive']):
                                credit_val = amounts[0]
                            else:
                                # Default to debit (charges, fees, payments, etc.)
                                debit_val = amounts[0]

                        # Create transaction if we have valid data
                        if (debit_val or credit_val) and description:
                            tx = BankTransaction(
                                bank_name="SCB",
                                acc_no=acc_no or "",
                                debit=debit_val,
                                credit=credit_val,
                                date=self._parse_ocr_date(entry_date_str),
                                description=description.strip(),
                                currency=currency,
                                transaction_id="",
                                beneficiary_bank="",
                                beneficiary_acc_no="",
                                beneficiary_acc_name=""
                            )
                            transactions.append(tx)

                        i = j
                        continue

            i += 1

        return transactions

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse Standard Chartered balance from OCR text.
        Returns balance for the first account section with non-zero balances.
        """
        all_balances = self.parse_all_balances_from_text(text, file_name)

        # Return first non-zero balance, or first balance
        for bal in all_balances:
            if bal.opening_balance > 0 or bal.closing_balance > 0:
                return bal

        return all_balances[0] if all_balances else None

    def parse_all_balances_from_text(self, text: str, file_name: str) -> List[BankBalance]:
        """
        Parse ALL account balances from Standard Chartered OCR text.

        Format:
        - "OPENING BALANCE" / "Số dư đầu kỳ" followed by amount on next line
        - "CLOSING BALANCE" / "Số dư cuối kỳ" followed by amount on next line
        """
        all_balances = []
        sections = self._split_account_sections(text)

        for section in sections:
            balance = self._parse_section_balance(section)
            if balance:
                all_balances.append(balance)

        logger.info(f"SCB: Parsed {len(all_balances)} account balances")
        return all_balances

    def _parse_section_balance(self, section: dict) -> Optional[BankBalance]:
        """Parse balance from a single account section."""
        lines = section.get('lines', [])
        acc_no = section.get('acc_no', '')
        currency = section.get('currency', 'VND')

        opening = 0.0
        closing = 0.0

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Opening balance patterns
            if 'opening balance' in line_lower or 'số dư đầu kỳ' in line_lower:
                # Check same line for amount
                amounts = re.findall(r'([\d,]+(?:\.\d{2})?)', line)
                if amounts:
                    val = self._parse_ocr_number(amounts[-1])
                    if val is not None:
                        opening = val
                # Check next line
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r'^[\d,]+(?:\.\d{2})?$', next_line):
                        val = self._parse_ocr_number(next_line)
                        if val is not None:
                            opening = val

            # Closing balance patterns
            if 'closing balance' in line_lower or 'số dư cuối kỳ' in line_lower:
                amounts = re.findall(r'([\d,]+(?:\.\d{2})?)', line)
                if amounts:
                    val = self._parse_ocr_number(amounts[-1])
                    if val is not None:
                        closing = val
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r'^[\d,]+(?:\.\d{2})?$', next_line):
                        val = self._parse_ocr_number(next_line)
                        if val is not None:
                            closing = val

        return BankBalance(
            bank_name="SCB",
            acc_no=acc_no or "",
            currency=currency,
            opening_balance=opening,
            closing_balance=closing
        )

    # ========== Helper Methods ==========

    def _parse_ocr_number(self, value: str) -> Optional[float]:
        """Parse number from OCR text."""
        if not value:
            return None

        # Remove commas and spaces
        cleaned = value.replace(",", "").replace(" ", "")

        try:
            num = float(cleaned)
            return num
        except (ValueError, TypeError):
            return None

    def _parse_ocr_date(self, date_str: str):
        """
        Parse date from OCR text.

        Handles formats:
        - "DD MMM YY" (e.g., "25 NOV 25")
        - "DD/MM/YYYY"
        """
        if not date_str:
            return None

        date_str = date_str.strip()

        # Try "DD MMM YY" format (e.g., "25 NOV 25")
        try:
            return datetime.strptime(date_str, "%d %b %y").date()
        except:
            pass

        # Try "DD MMM YYYY" format
        try:
            return datetime.strptime(date_str, "%d %b %Y").date()
        except:
            pass

        # Try DD/MM/YYYY format
        date_str_normalized = date_str.replace("-", "/")
        try:
            return datetime.strptime(date_str_normalized, "%d/%m/%Y").date()
        except:
            pass

        # Try pandas as fallback
        try:
            return pd.to_datetime(date_str, dayfirst=True).date()
        except:
            return None
