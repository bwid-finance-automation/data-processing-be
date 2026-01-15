"""UOB bank statement parser."""

import re
import logging
from typing import List, Optional, Tuple
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser

logger = logging.getLogger(__name__)


class UOBParser(BaseBankParser):
    """Parser for UOB (United Overseas Bank) statements - handles Excel and PDF OCR."""

    @property
    def bank_name(self) -> str:
        return "UOB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is UOB bank statement.

        Detection markers:
        - "Tài khoản Thanh toán/Tiết kiệm" in first rows
        - "Tiền gửi (DEBIT)" and "Tiền rút (CREDIT)" in header
        - Sheet name contains "Account Activities"
        """
        try:
            xls = self.get_excel_file(file_bytes)

            # Check sheet name for UOB pattern
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            has_account_activities = "Account Activities" in sheet_name

            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 15 rows
            top_15 = df.head(15)

            # Convert all cells to text
            all_text = []
            for _, row in top_15.iterrows():
                row_text = " ".join([self.to_text(cell).upper() for cell in row])
                all_text.append(row_text)

            combined = " ".join(all_text)

            # Check for UOB-specific markers
            has_title = "TÀI KHOẢN THANH TOÁN" in combined or "TÀI KHOẢN TIẾT KIỆM" in combined
            has_deposit = "TIỀN GỬI" in combined and "DEBIT" in combined
            has_withdraw = "TIỀN RÚT" in combined and "CREDIT" in combined
            has_ledger_balance = "SỐ DƯ SỔ CÁI" in combined

            # UOB files have specific combination of markers
            is_uob = (has_account_activities or has_title) and (has_deposit or has_withdraw or has_ledger_balance)

            return is_uob

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse UOB transactions.

        UOB column structure (after header row 10):
        - Col 0: Ngày hiệu lực (Effective Date) - DD/MM/YYYY
        - Col 1: Ngày Giao dịch (Transaction Date)
        - Col 2: Dấu thời gian (Timestamp)
        - Col 3: Mô tả (Description) - contains transaction ID in newlines
        - Col 4: Tiền gửi (DEBIT) - Money IN = our Debit
        - Col 5: Tiền rút (CREDIT) - Money OUT = our Credit
        - Col 6: Số dư (Balance)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            acc_no = self._extract_account_number(sheet.head(10))

            # ========== Extract Currency ==========
            currency = self._extract_currency(sheet.head(10))

            # ========== Find Header Row ==========
            header_idx = self._find_header_row(sheet.head(15))

            if header_idx is None:
                return []

            # ========== Parse Transactions ==========
            transactions = []

            for idx in range(header_idx + 1, len(sheet)):
                row = sheet.iloc[idx]

                # Stop at totals row
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Tổng số theo Loại tiền tệ" in row_text:
                    break

                # Skip empty rows
                if pd.isna(row.iloc[0]) or self.to_text(row.iloc[0]).strip() == "":
                    continue

                # Skip note rows
                if "Lưu ý:" in row_text or "Bảo hiểm Tiền gửi" in row_text:
                    break

                # Parse date (Col 0)
                date_val = self._parse_date_uob(row.iloc[0])

                # Parse description and extract transaction ID (Col 3)
                desc_raw = self.to_text(row.iloc[3]) if len(row) > 3 else ""
                tx_id, description = self._parse_description(desc_raw)

                # Parse amounts
                # Col 4: Tiền gửi = Money IN = Debit
                # Col 5: Tiền rút = Money OUT = Credit
                debit_val = self.fix_number(row.iloc[4]) if len(row) > 4 else None
                credit_val = self.fix_number(row.iloc[5]) if len(row) > 5 else None

                # Skip if both are zero or None
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Handle "0" string values
                if debit_val == 0:
                    debit_val = None
                if credit_val == 0:
                    credit_val = None

                tx = BankTransaction(
                    bank_name="UOB",
                    acc_no=acc_no or "",
                    debit=debit_val,
                    credit=credit_val,
                    date=date_val,
                    description=description,
                    currency=currency,
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing UOB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse UOB balance information.

        Balance extraction:
        - Closing balance: Row 3, Col 4 "Số dư Sổ cái:" value
        - Opening balance: Calculate from first transaction's balance minus net change
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            acc_no = self._extract_account_number(sheet.head(10))

            # ========== Extract Currency ==========
            currency = self._extract_currency(sheet.head(10))

            # ========== Extract Closing Balance from Header ==========
            closing_balance = self._extract_closing_balance(sheet.head(10))

            # ========== Calculate Opening Balance ==========
            opening_balance = self._calculate_opening_balance(sheet, closing_balance)

            return BankBalance(
                bank_name="UOB",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening_balance or 0.0,
                closing_balance=closing_balance or 0.0
            )

        except Exception as e:
            print(f"Error parsing UOB balances: {e}")
            return None

    # ========== Helper Methods ==========

    def _extract_account_number(self, top_rows: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from header area.
        Located in Row 2, Col 4 after "Số Tài khoản:"
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            if "Số Tài khoản:" in row_text or "Số tài khoản:" in row_text:
                # Account number is in column 4
                if len(row) > 4:
                    acc_text = self.to_text(row.iloc[4])
                    # Extract digits
                    digits = ''.join(c for c in acc_text if c.isdigit())
                    if 8 <= len(digits) <= 15:
                        return digits

        return None

    def _extract_currency(self, top_rows: pd.DataFrame) -> str:
        """
        Extract currency from header area.
        Located in Row 5 after "Loại tiền tệ Tài khoản:"
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "LOẠI TIỀN TỆ" in row_text:
                if len(row) > 1:
                    curr_val = self.to_text(row.iloc[1]).upper().strip()
                    if curr_val in ["VND", "USD", "EUR", "SGD", "JPY"]:
                        return curr_val

        # Default
        return "VND"

    def _extract_closing_balance(self, top_rows: pd.DataFrame) -> Optional[float]:
        """
        Extract closing balance from "Số dư Sổ cái:" or "Số dư khả dụng:" row.
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            if "Số dư Sổ cái:" in row_text or "Số dư khả dụng:" in row_text:
                # Balance value is in column 4
                if len(row) > 4:
                    return self.fix_number(row.iloc[4])

        return None

    def _calculate_opening_balance(self, sheet: pd.DataFrame, closing_balance: Optional[float]) -> Optional[float]:
        """
        Calculate opening balance by working backwards from closing balance.

        Opening = Closing - Total Credits + Total Debits
        Or: Get balance from first transaction and subtract net change of first tx
        """
        try:
            header_idx = self._find_header_row(sheet.head(15))
            if header_idx is None:
                return None

            # Get first transaction
            first_tx_idx = header_idx + 1
            if first_tx_idx >= len(sheet):
                return closing_balance

            first_row = sheet.iloc[first_tx_idx]

            # Skip if empty
            if pd.isna(first_row.iloc[0]):
                return closing_balance

            # Get first transaction's balance (Col 6)
            first_balance = self.fix_number(first_row.iloc[6]) if len(first_row) > 6 else None

            # Get first transaction's amounts
            first_debit = self.fix_number(first_row.iloc[4]) if len(first_row) > 4 else 0
            first_credit = self.fix_number(first_row.iloc[5]) if len(first_row) > 5 else 0

            if first_balance is not None:
                # Opening = First Balance - First Debit + First Credit
                first_debit = first_debit or 0
                first_credit = first_credit or 0
                opening = first_balance - first_debit + first_credit
                return opening

            return closing_balance

        except Exception:
            return closing_balance

    def _find_header_row(self, top_rows: pd.DataFrame) -> Optional[int]:
        """
        Find header row by looking for specific column names.
        Header contains: "Ngày hiệu lực", "Tiền gửi", "Tiền rút"
        """
        for idx, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            has_date = "NGÀY HIỆU LỰC" in row_text or "NGÀY GIAO DỊCH" in row_text
            has_deposit = "TIỀN GỬI" in row_text
            has_withdraw = "TIỀN RÚT" in row_text

            if has_date and (has_deposit or has_withdraw):
                return int(idx)

        return None

    def _parse_date_uob(self, value) -> Optional:
        """
        Parse UOB date format: DD/MM/YYYY
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Try DD/MM/YYYY format
        try:
            return datetime.strptime(txt, "%d/%m/%Y").date()
        except:
            pass

        # Try pandas parser
        try:
            return pd.to_datetime(txt, dayfirst=True).date()
        except:
            pass

        return None

    def _parse_description(self, desc_raw: str) -> tuple:
        """
        Parse description and extract transaction ID.

        UOB description format (multi-line):
        Line 1: Type (MISC CREDIT/MISC DEBIT)
        Line 2: Transaction ID or "NONE"
        Line 3: Reference
        Line 4+: Details

        Returns: (transaction_id, cleaned_description)
        """
        if not desc_raw:
            return "", ""

        # Split by newlines
        lines = [line.strip() for line in desc_raw.split('\n') if line.strip()]

        if not lines:
            return "", ""

        tx_id = ""
        description_parts = []

        for i, line in enumerate(lines):
            if i == 1:
                # Second line is usually transaction ID
                if line != "NONE" and re.match(r'^[A-Z0-9]+$', line):
                    tx_id = line
                else:
                    description_parts.append(line)
            elif i == 0:
                # First line is type (MISC CREDIT/MISC DEBIT)
                description_parts.append(line)
            else:
                description_parts.append(line)

        description = " | ".join(description_parts)

        return tx_id, description

    # ========== OCR Text Parsing Methods ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if OCR text is from UOB bank statement.

        UOB PDF markers (unique to UOB Vietnam online banking export):
        - "Account Activities" (unique UOB header - very strong indicator)
        - "UOB" or "United Overseas Bank"
        - "Ledger Balance" + "Deposit" + "Withdrawal"
        - "MISC CREDIT" / "MISC DEBIT" transaction types
        - "Statement Date" + "Transaction Date" columns
        """
        if not text:
            return False

        text_lower = text.lower()

        # Primary marker - "Account Activities" is UNIQUE to UOB Vietnam
        has_account_activities = "account activities" in text_lower

        # UOB brand markers (logo might be image, so text might not have "UOB")
        has_uob = "uob" in text_lower or "united overseas bank" in text_lower

        # Secondary markers
        has_ledger_balance = "ledger balance" in text_lower
        has_deposit = "deposit" in text_lower
        has_withdrawal = "withdrawal" in text_lower

        # UOB-specific transaction types
        has_misc_tx = "misc credit" in text_lower or "misc debit" in text_lower

        # UOB-specific column headers
        has_statement_date = "statement date" in text_lower
        has_transaction_date = "transaction date" in text_lower

        # Detection logic:
        # 1. "Account Activities" + "UOB" → definitely UOB
        # 2. "Account Activities" + Ledger Balance + Deposit/Withdrawal → likely UOB
        # 3. "Account Activities" + MISC CREDIT/DEBIT → likely UOB
        # 4. "Account Activities" + Statement Date + Transaction Date + Deposit → likely UOB
        is_uob = (
            (has_account_activities and has_uob) or
            (has_account_activities and has_ledger_balance and has_deposit and has_withdrawal) or
            (has_account_activities and has_misc_tx) or
            (has_account_activities and has_statement_date and has_transaction_date and has_deposit)
        )

        return is_uob

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse UOB transactions from OCR text.

        UOB PDF OCR has TWO formats depending on the PDF source:

        Format A (H5R-style - online banking screenshot):
        - Statement Date: DD/MM/YYYY (standalone line)
        - Transaction Date: DD/MM/YYYY HH:MM:SS AM/PM
        - MISC CREDIT or MISC DEBIT
        - Transaction ID or NONE
        - Description lines
        - Amounts (integers: 5,000,000 or 0)

        Format B (PLC-style - exported PDF):
        - Transaction Date+Time: DD/MM/YYYY HH:MM:SS (no AM/PM)
        - Statement Date: DD/MM/YYYY
        - AM/PM (separate line)
        - MISC CREDIT or MISC DEBIT
        - Transaction ID
        - Description lines
        - Amounts (with decimals: 0.00)

        Column mapping (based on UOB convention):
        - Deposit = Money IN = our Debit
        - Withdrawal = Money OUT = our Credit
        """
        transactions = []

        # Extract account number and currency
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Find where transactions start (after header)
        start_idx = 0
        found_header = None
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            # Format A: Look for "i Advice" or "Advice" column header
            if line_lower in ['advice', 'i advice']:
                start_idx = idx + 1
                found_header = f"Format A: '{line}' at line {idx}"
                break
            # Format B: Look for transaction table header containing multiple column names
            # "Statement Date Transaction Date Description Deposit (VND) Withdrawal(VND) Ledger Balance(VND)"
            if ('statement date' in line_lower and 'transaction date' in line_lower and
                'ledger balance' in line_lower):
                start_idx = idx + 1
                found_header = f"Format B: '{line[:50]}...' at line {idx}"
                break

        logger.info(f"UOB Parser: start_idx={start_idx}, header={found_header}, total_lines={len(lines)}")
        if start_idx > 0 and start_idx < len(lines):
            logger.info(f"UOB Parser: First data line: '{lines[start_idx]}'")

        # Stop patterns - where transactions end
        stop_patterns = [
            'total deposits', 'total withdrawals', 'note', 'copyright',
            'deposit insurance', 'this is a computer', 'date of export'
        ]

        i = start_idx
        while i < len(lines):
            line = lines[i]
            line_lower = line.lower()

            # Stop at footer
            if any(stop in line_lower for stop in stop_patterns):
                break

            # ========== Format A: DD/MM/YYYY (standalone date) ==========
            format_a_match = re.match(r'^(\d{1,2}/\d{1,2}/\d{4})$', line)

            # ========== Format B: DD/MM/YYYY HH:MM:SS (date+time, no AM/PM) ==========
            format_b_match = re.match(r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2})$', line)

            if format_a_match or format_b_match:
                # Initialize transaction data
                statement_date = None
                tx_id = ""
                tx_type = ""
                description_parts = []
                amounts = []

                j = i + 1

                if format_a_match:
                    # Format A: Current line is statement date
                    statement_date = format_a_match.group(1)

                    # Next line should be transaction date+time
                    if j < len(lines):
                        next_line = lines[j].strip()
                        tx_datetime_match = re.match(
                            r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)',
                            next_line,
                            re.IGNORECASE
                        )
                        if tx_datetime_match:
                            j += 1

                elif format_b_match:
                    # Format B: Current line is transaction date+time
                    # Next line should be statement date
                    if j < len(lines):
                        next_line = lines[j].strip()
                        stmt_date_match = re.match(r'^(\d{1,2}/\d{1,2}/\d{4})$', next_line)
                        if stmt_date_match:
                            statement_date = stmt_date_match.group(1)
                            j += 1

                    # Skip AM/PM line if present
                    if j < len(lines) and lines[j].strip().upper() in ['AM', 'PM']:
                        j += 1

                # Parse MISC CREDIT/DEBIT
                if j < len(lines):
                    type_line = lines[j].strip()
                    if type_line in ['MISC CREDIT', 'MISC DEBIT']:
                        tx_type = type_line
                        j += 1

                # Parse Transaction ID (2025XXXXXXXXXX or NONE or MOR...)
                if j < len(lines):
                    id_line = lines[j].strip()
                    if re.match(r'^20\d{14,}$', id_line):
                        tx_id = id_line
                        j += 1
                    elif id_line == 'NONE':
                        j += 1  # Skip NONE
                    elif re.match(r'^M[OI]R\d+C\d+$', id_line):
                        # Format B: Reference line comes before TX ID
                        description_parts.append(id_line)
                        j += 1

                # Parse Reference and Description lines until we hit amounts
                while j < len(lines):
                    desc_line = lines[j].strip()

                    # Check if this is an amount (integer or decimal with commas)
                    if re.match(r'^[\d,]+(?:\.\d+)?$', desc_line):
                        break

                    # Check if next transaction starts (Format A or B)
                    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}:\d{2})?$', desc_line):
                        break

                    # Check for stop patterns
                    if any(stop in desc_line.lower() for stop in stop_patterns):
                        break

                    description_parts.append(desc_line)
                    j += 1

                # Parse 3 amounts: Deposit, Withdrawal, Balance
                while j < len(lines) and len(amounts) < 3:
                    amt_line = lines[j].strip()
                    # Match integers or decimals with commas (e.g., "5,000,000", "0", "0.00")
                    if re.match(r'^[\d,]+(?:\.\d+)?$', amt_line):
                        amounts.append(self._parse_ocr_amount(amt_line))
                        j += 1
                    else:
                        break

                # Extract deposit and withdrawal
                deposit = None
                withdrawal = None
                if len(amounts) >= 2:
                    deposit = amounts[0] if amounts[0] and amounts[0] > 0 else None
                    withdrawal = amounts[1] if amounts[1] and amounts[1] > 0 else None

                # Skip if both are 0 or None
                if (deposit is None or deposit == 0) and (withdrawal is None or withdrawal == 0):
                    i = j if j > i else i + 1
                    continue

                # Build description
                description = " | ".join(description_parts) if description_parts else ""
                if tx_type:
                    description = f"{tx_type} | {description}" if description else tx_type

                # Parse date
                date_val = self._parse_date_uob(statement_date) if statement_date else None

                tx = BankTransaction(
                    bank_name="UOB",
                    acc_no=acc_no or "",
                    debit=deposit,  # Deposit = Money IN = Debit
                    credit=withdrawal,  # Withdrawal = Money OUT = Credit
                    date=date_val,
                    description=description,
                    currency=currency,
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)
                logger.debug(f"UOB Parser: Found tx #{len(transactions)}: date={statement_date}, debit={deposit}, credit={withdrawal}")
                i = j if j > i else i + 1
            else:
                i += 1

        logger.info(f"UOB Parser: Finished parsing, found {len(transactions)} transactions")
        return transactions

    def parse_balance_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """Parse single UOB balance from OCR text."""
        balances = self.parse_all_balances_from_text(text, file_name)
        return balances[0] if balances else None

    def parse_all_balances_from_text(self, text: str, file_name: str) -> List[BankBalance]:
        """
        Parse UOB balance from OCR text.

        UOB PDF balance formats (from Gemini OCR):
        Format A: "Ledger Balance\nVND 4,945,000" (integers)
        Format B: "Ledger Balance\nVND 0.00" (decimals)
        Format A: "Total Deposits (VND)\nVND 154,766,087"
        Format B: "Total Deposits(VND)\n138,729,284,385.00"
        """
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        # Extract Ledger Balance (closing balance)
        # Format: "Ledger Balance\nVND 4,945,000" or "Ledger Balance\nVND 0.00"
        closing_balance = None
        ledger_match = re.search(
            r'Ledger\s*Balance\s*(?:\([^)]*\))?\s*\n?\s*(?:VND)?\s*([\d,]+(?:\.\d+)?)',
            text,
            re.IGNORECASE
        )
        if ledger_match:
            closing_balance = self._parse_ocr_amount(ledger_match.group(1))

        # Extract Total Deposits
        # Format A: "Total Deposits (VND)\nVND 154,766,087"
        # Format B: "Total Deposits(VND)\n138,729,284,385.00"
        total_deposits = None
        deposits_match = re.search(
            r'Total\s*Deposits?\s*\(?VND\)?\s*\n?\s*(?:VND)?\s*([\d,]+(?:\.\d+)?)',
            text,
            re.IGNORECASE
        )
        if deposits_match:
            total_deposits = self._parse_ocr_amount(deposits_match.group(1))

        # Extract Total Withdrawals
        # Format A: "Total Withdrawals (VND)\nVND 149,821,087"
        # Format B: "Total Withdrawals(VND)\n138,729,284,385.00"
        total_withdrawals = None
        withdrawals_match = re.search(
            r'Total\s*Withdrawals?\s*\(?VND\)?\s*\n?\s*(?:VND)?\s*([\d,]+(?:\.\d+)?)',
            text,
            re.IGNORECASE
        )
        if withdrawals_match:
            total_withdrawals = self._parse_ocr_amount(withdrawals_match.group(1))

        # Calculate opening balance
        # Opening = Closing - Deposits + Withdrawals
        opening_balance = None
        if closing_balance is not None and total_deposits is not None and total_withdrawals is not None:
            opening_balance = closing_balance - (total_deposits or 0) + (total_withdrawals or 0)

        if acc_no or closing_balance is not None:
            return [BankBalance(
                bank_name="UOB",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening_balance or 0.0,
                closing_balance=closing_balance or 0.0
            )]

        return []

    def _extract_account_from_ocr(self, text: str) -> Optional[str]:
        """
        Extract account number from UOB OCR text.

        Patterns from Gemini OCR:
        - "CONG TY TNHH DAU TU PHAT TRIEN HOLDCO5R VND 1053067315"
        - "VND 1053067315"
        - "1053051737 CONG TY..."
        """
        # Pattern 1: "VND XXXXXXXXXX" (10-digit account after VND)
        match1 = re.search(r'VND\s+(\d{10})\b', text)
        if match1:
            return match1.group(1)

        # Pattern 2: Account number at start of company line
        match2 = re.search(r'\b(\d{10})\s+CONG TY', text)
        if match2:
            return match2.group(1)

        # Pattern 3: Standalone 10-digit number (UOB account format)
        match3 = re.search(r'\b(105\d{7})\b', text)  # UOB VN accounts start with 105
        if match3:
            return match3.group(1)

        # Pattern 4: Generic 10-digit number
        match4 = re.search(r'\b(\d{10})\b', text)
        if match4:
            return match4.group(1)

        return None

    def _extract_currency_from_ocr(self, text: str) -> str:
        """
        Extract currency from UOB OCR text.

        Default to VND for UOB Vietnam statements.
        """
        text_upper = text.upper()

        # Check for explicit currency markers
        if "USD" in text_upper and "ACCOUNT CURRENCY" in text_upper:
            return "USD"
        if "SGD" in text_upper and "ACCOUNT CURRENCY" in text_upper:
            return "SGD"

        # Default for UOB Vietnam
        return "VND"

    def _parse_ocr_amount(self, value: str) -> Optional[float]:
        """
        Parse amount from OCR text.

        UOB format: "5,000,000" or "0" or "0.00" or "22,679,284,385"
        """
        if not value:
            return None

        try:
            # Remove commas and parse
            clean = value.replace(',', '').strip()
            return float(clean)
        except (ValueError, TypeError):
            return None
