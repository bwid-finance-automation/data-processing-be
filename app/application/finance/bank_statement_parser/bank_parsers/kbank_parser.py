"""KBANK bank statement parser."""

import io
import math
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class KBANKParser(BaseBankParser):
    """Parser for KBANK (Kasikornbank) statements."""

    @property
    def bank_name(self) -> str:
        return "KBANK"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is KBANK bank statement.

        Logic from fxLooksLike_KBANK:
        - Read first 80 rows
        - Look for markers:
          * "kasi", "kasionbank", "kasikorn"
          * "ngân hàng kasikorn"
          * "demand deposit account information"
          * OR ("transaction date" + "debit amount" + "credit amount")
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 80 rows
            top_80 = df.head(80)

            # Flatten all cells to lowercase text
            all_text = []
            for col in top_80.columns:
                all_text.extend([self.to_text(cell).lower() for cell in top_80[col]])

            flat = " ".join(all_text)

            # Check for KBANK markers
            has_kasi = "kasi" in flat
            has_kasionbank = "kasionbank" in flat
            has_kasikorn = "kasikorn" in flat
            has_kasikorn_vn = "ngân hàng kasikorn" in flat
            has_demand = "demand deposit account information" in flat
            has_transaction_pattern = "transaction date" in flat and \
                                     "debit amount" in flat and \
                                     "credit amount" in flat

            is_kbank = has_kasi or has_kasionbank or has_kasikorn or \
                       has_kasikorn_vn or has_demand or has_transaction_pattern

            return is_kbank

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse KBANK transactions.

        Logic:
        - Find header row with: "transaction date" + "debit amount" + "credit amount" + "description"
        - Flexible column detection (bilingual: English/Vietnamese)
        - Standard mapping: Debit = money OUT, Credit = money IN
        - Extract account number from header
        - Currency detection (USD/VND)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_100 = sheet.head(100)
            header_idx = self._find_header_row_kbank(top_100)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            col_date = self._find_column(data, ["transaction date", "ngày giao dịch"])
            col_time = self._find_column(data, ["transaction time", "thời gian giao dịch"])
            col_debit = self._find_column(data, ["debit amount", "số tiền ghi nợ"])    # Money OUT
            col_credit = self._find_column(data, ["credit amount", "số tiền ghi có"])  # Money IN
            col_balance = self._find_column(data, ["balance", "số dư"])
            col_desc = self._find_column(data, ["description", "diễn giải"])

            # Keep only available columns
            keep_cols = [c for c in [col_date, col_time, col_debit, col_credit, col_balance, col_desc] if c]
            if not keep_cols:
                return []

            data = data[keep_cols].copy()

            # ========== Extract Account Number & Currency ==========
            acc_no = self._extract_account_number_kbank(top_100)
            currency = self._extract_currency_kbank(top_100)

            # ========== Parse Transactions ==========
            transactions = []

            # Find STT/No. column for checking summary rows
            col_stt = self._find_column(data, ["stt", "no.", "no"])

            for _, row in data.iterrows():
                # Skip summary/total rows ("Tổng / Total")
                row_text = " ".join([self.to_text(cell).lower() for cell in row])
                if "tổng" in row_text or "total" in row_text:
                    continue

                # Skip rows that look like "Previous Balance" or "Current Balance"
                if "previous balance" in row_text or "số dư ban đầu" in row_text:
                    continue
                if "current balance" in row_text or "số dư hiện tại" in row_text:
                    continue

                # KBANK mapping (swapped for ERP convention):
                # In bank file: "Debit Amount" = money OUT, "Credit Amount" = money IN
                # In ERP output: Debit = money IN, Credit = money OUT
                # So we swap: file's Credit → output Debit, file's Debit → output Credit
                debit_val = self._fix_number_kbank(row.get(col_credit)) if col_credit else None   # Credit Amount → Debit
                credit_val = self._fix_number_kbank(row.get(col_debit)) if col_debit else None    # Debit Amount → Credit

                # Skip rows where both are zero/blank
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                tx = BankTransaction(
                    bank_name="KBANK",
                    acc_no=acc_no or "",
                    debit=debit_val,
                    credit=credit_val,
                    date=self.fix_date(row.get(col_date)) if col_date else None,
                    description=self.to_text(row.get(col_desc, "")) if col_desc else "",
                    currency=currency,
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing KBANK transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse KBANK balance information.

        Logic from fxParse_KBANK_Balances:
        - Opening = from "Previous Balance" / "Số dư ban đầu" row
        - Closing = from "Current Balance" / "Số dư hiện tại" row, or last balance in table
        - Account number from header (2 methods)
        - Currency detection (USD/VND)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_100 = sheet.head(100)
            header_idx = self._find_header_row_kbank(top_100)

            # ========== Extract Account Number & Currency from header ==========
            acc_no = self._extract_account_number_kbank(top_100)
            currency = self._extract_currency_kbank(top_100)

            # ========== Find Opening & Closing Balance from special rows ==========
            opening = 0.0
            closing = 0.0

            # Scan all rows for balance markers
            for idx, row in sheet.iterrows():
                row_text = " ".join([self.to_text(cell).lower() for cell in row])

                # Opening balance: "Số dư ban đầu" / "Previous Balance"
                if "previous balance" in row_text or "số dư ban đầu" in row_text:
                    # Find the balance value in this row (look for number >= 8 digits or formatted number)
                    for cell in row:
                        val = self._fix_number_kbank(cell)
                        if val is not None and val > 0:
                            opening = val
                            break

                # Closing balance: "Số dư hiện tại" / "Current Balance"
                if "current balance" in row_text or "số dư hiện tại" in row_text:
                    for cell in row:
                        val = self._fix_number_kbank(cell)
                        if val is not None and val > 0:
                            closing = val
                            break

            # ========== Fallback: Get closing from last balance in transaction table ==========
            if closing == 0:
                # Promote Headers
                data = sheet.iloc[header_idx:].copy()
                data.columns = data.iloc[0]
                data = data[1:].reset_index(drop=True)

                col_balance = self._find_column(data, ["balance", "số dư"])
                if col_balance:
                    data["_Bal"] = data[col_balance].apply(self._fix_number_kbank)
                    bal_list = data["_Bal"].dropna().tolist()
                    # Filter out potential NaN/inf values
                    bal_list = [b for b in bal_list if isinstance(b, (int, float)) and not math.isnan(b) and not math.isinf(b)]
                    if bal_list:
                        closing = bal_list[-1]

            # ========== Fallback: Calculate opening from first transaction ==========
            if opening == 0 and closing > 0:
                data = sheet.iloc[header_idx:].copy()
                data.columns = data.iloc[0]
                data = data[1:].reset_index(drop=True)

                col_debit_src = self._find_column(data, ["debit amount", "số tiền ghi nợ"])
                col_credit_src = self._find_column(data, ["credit amount", "số tiền ghi có"])
                col_balance = self._find_column(data, ["balance", "số dư"])

                if col_balance:
                    # In bank file: Debit = money OUT, Credit = money IN
                    data["_Out"] = data[col_debit_src].apply(self._fix_number_kbank) if col_debit_src else 0
                    data["_In"] = data[col_credit_src].apply(self._fix_number_kbank) if col_credit_src else 0
                    data["_Bal"] = data[col_balance].apply(self._fix_number_kbank)

                    # Find first row with a valid balance (skip Previous Balance row which has no date)
                    for _, row in data.iterrows():
                        first_bal = row.get("_Bal")
                        first_out = row.get("_Out")  # Debit Amount = money OUT
                        first_in = row.get("_In")    # Credit Amount = money IN

                        # Skip if balance is NaN or if this looks like the "Previous Balance" row
                        if pd.isna(first_bal):
                            continue

                        # Safe NaN handling for debit/credit
                        if first_out is None or (isinstance(first_out, float) and math.isnan(first_out)):
                            first_out = 0
                        if first_in is None or (isinstance(first_in, float) and math.isnan(first_in)):
                            first_in = 0

                        # Calculate opening: balance + outflow - inflow
                        # Bank convention: balance = previous_balance - debit + credit
                        # So: previous_balance = balance + debit - credit = balance + out - in
                        opening = first_bal + first_out - first_in
                        break

            return BankBalance(
                bank_name="KBANK",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening,
                closing_balance=closing
            )

        except Exception as e:
            print(f"Error parsing KBANK balances: {e}")
            return None

    # ========== KBANK-Specific Helper Methods ==========

    def _find_header_row_kbank(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing all required KBANK markers:
        - "transaction date"
        - ("debit amount" OR "số tiền ghi nợ")
        - ("credit amount" OR "số tiền ghi có")
        - ("description" OR "diễn giải")
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell).lower() for cell in row])

            has_date = "transaction date" in row_text or "ngày giao dịch" in row_text
            has_debit = "debit amount" in row_text or "số tiền ghi nợ" in row_text
            has_credit = "credit amount" in row_text or "số tiền ghi có" in row_text
            has_desc = "description" in row_text or "diễn giải" in row_text

            if has_date and has_debit and has_credit and has_desc:
                return int(idx)

        return 10  # Default fallback

    def _find_column(self, df: pd.DataFrame, patterns: list) -> Optional[str]:
        """Find column name matching any of the patterns (case-insensitive)."""
        columns = df.columns.tolist()

        for col in columns:
            col_lower = str(col).lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    return col

        return None

    def _extract_account_number_kbank(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from header using 2 methods:
        1. Find row with "account number" or "số tài khoản", extract digits
        2. Find last digit sequence >= 8 chars in header area
        """
        # Method 1: Look for account number line
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])
            row_lower = row_text.lower()

            if "account number" in row_lower or "số tài khoản" in row_lower:
                digits = ''.join(c for c in row_text if c.isdigit())
                if len(digits) >= 6:
                    return digits

        # Method 2: Find longest digit sequence >= 8 chars
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell) for cell in top_df[col]])

        flat = " ".join(all_text)
        all_digits = ''.join(c if c.isdigit() else ' ' for c in flat)
        parts = [p for p in all_digits.split() if len(p) >= 8]

        if parts:
            return parts[-1]  # Return last match

        return None

    def _extract_currency_kbank(self, top_df: pd.DataFrame) -> str:
        """Extract currency from header area."""
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell).lower() for cell in top_df[col]])

        flat = " ".join(all_text)

        if "usd" in flat:
            return "USD"
        elif "vnd" in flat or "vnđ" in flat:
            return "VND"

        return "VND"  # Default

    def _fix_number_kbank(self, value) -> Optional[float]:
        """
        KBANK-specific number parser.
        Handles: VND prefix, parentheses negatives, comma/dot separators.
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip().upper()
        if not txt:
            return None

        # Remove "VND"
        txt = txt.replace("VND", "")

        # Handle parentheses negatives: (1,234) → -1234
        if txt.startswith("(") and txt.endswith(")"):
            txt = "-" + txt[1:-1]

        # Remove spaces and separators
        # KBANK uses commas as thousand separators, dots might be decimal
        txt = txt.replace(" ", "").replace(",", "")

        # If there's a dot, keep it as decimal separator
        # Otherwise remove dots too
        if "." in txt:
            # Assume dot is decimal separator
            pass
        else:
            txt = txt.replace(".", "")

        try:
            return float(txt)
        except (ValueError, TypeError):
            return None

    # ========== OCR Text Parsing Methods ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if OCR text is from KBANK bank statement.

        Markers:
        - "kasikorn" or "kbank"
        - "demand deposit account" or "tài khoản tiền gửi"
        - "transaction date" + "debit amount" + "credit amount"
        """
        text_lower = text.lower()

        has_kasikorn = "kasikorn" in text_lower or "kbank" in text_lower
        has_demand = "demand deposit" in text_lower or "tài khoản tiền gửi" in text_lower
        has_transaction_pattern = (
            "transaction date" in text_lower and
            "debit" in text_lower and
            "credit" in text_lower
        )

        return has_kasikorn or has_demand or has_transaction_pattern

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse KBANK transactions from OCR text.

        KBANK PDF OCR format (multi-line):
        - Transaction starts with: DD/MM/YYYY HH:MM AM/PM Description
        - May have additional description lines
        - Followed by 3 amount lines: Debit, Credit, Balance
        - "Cộng phát sinh Total" marks end of transactions
        """
        transactions = []

        # Extract account number and currency
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Skip patterns for metadata lines
        skip_patterns = [
            'user:', 'run:', 'system:', 'report:', 'program:', 'file(s):',
            'order by:', 'input:', 'where', 'dep.cid', 'hist.', 'mã báo cáo',
            'ngày hệ thống', 'ngày chạy', 'người in', 'trang:', 'chi nhánh',
            'branch:', 'kính gửi', 'dear customer', 'địa chỉ', 'address',
            'ngân hàng', 'loại tiền tệ', 'currency', 'giao dịch viên',
            'teller', 'kiểm soát', 'supervisor', 'customer sub-type',
            '====', '----', 'đơn vị', 'kasikornbank', 'sao kê chi tiết',
            'transaction statement', 'từ ngày', 'from date', 'đến ngày', 'to date',
            'ngày date', 'giờ time', 'nội dung giao dịch', 'transaction comment',
            'doanh số phát sinh', 'turnover', 'số dư sau giao dịch', 'balance'
        ]

        i = 0
        while i < len(lines):
            line = lines[i]
            line_lower = line.lower()

            # Stop at totals line
            if 'cộng phát sinh' in line_lower or 'total' in line_lower:
                break

            # Skip metadata/header lines
            if any(skip in line_lower for skip in skip_patterns):
                i += 1
                continue

            # Skip "Beginning Balance" line (not a transaction, just opening balance)
            if 'beginning balance' in line_lower:
                i += 1
                continue

            # Look for transaction start: DD/MM/YYYY HH:MM AM/PM Description
            tx_match = re.match(
                r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)\s+(.+)',
                line,
                re.IGNORECASE
            )

            if tx_match:
                date_str = tx_match.group(1)
                time_str = tx_match.group(2)
                description_parts = [tx_match.group(3)]

                # Collect amounts - look at following lines
                amounts = []
                j = i + 1

                while j < len(lines) and len(amounts) < 3:
                    next_line = lines[j].strip()

                    # Check if this line is just an amount (e.g., "33,000.00" or "0.00")
                    amount_match = re.match(r'^[\d,]+\.\d{2}$', next_line)
                    if amount_match:
                        parsed = self._parse_ocr_number(next_line)
                        if parsed is not None:
                            amounts.append(parsed)
                        j += 1
                    # Check if next line is a new transaction (date pattern)
                    elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', next_line):
                        break
                    # Check if it's a stop pattern
                    elif any(skip in next_line.lower() for skip in ['cộng phát sinh', 'total', 'beginning balance']):
                        break
                    # Otherwise it might be additional description
                    elif next_line and not any(skip in next_line.lower() for skip in skip_patterns):
                        # Check if line contains amounts mixed with text
                        line_amounts = re.findall(r'[\d,]+\.\d{2}', next_line)
                        if line_amounts:
                            # Extract amounts from this line
                            for amt_str in line_amounts:
                                parsed = self._parse_ocr_number(amt_str)
                                if parsed is not None:
                                    amounts.append(parsed)
                            # Get text before first amount as description
                            first_amt_pos = next_line.find(line_amounts[0])
                            if first_amt_pos > 0:
                                desc_text = next_line[:first_amt_pos].strip()
                                if desc_text:
                                    description_parts.append(desc_text)
                        else:
                            # Pure description line
                            description_parts.append(next_line)
                        j += 1
                    else:
                        j += 1

                # Build final description
                description = ' '.join(description_parts).strip()

                # Parse amounts: [Debit, Credit, Balance] from bank file
                # Swap for ERP convention: file's Credit → output Debit, file's Debit → output Credit
                debit_val = None
                credit_val = None

                if len(amounts) >= 3:
                    # Bank file format: [debit (out), credit (in), balance]
                    # ERP output: Debit = money IN, Credit = money OUT
                    if amounts[1] > 0:  # Bank's Credit (money IN) → ERP Debit
                        debit_val = amounts[1]
                    if amounts[0] > 0:  # Bank's Debit (money OUT) → ERP Credit
                        credit_val = amounts[0]
                elif len(amounts) == 2:
                    # Could be [amount, balance] - check description for hints
                    desc_lower = description.lower()
                    if 'withdrawal' in desc_lower or 'debit' in desc_lower:
                        # Money OUT → ERP Credit
                        credit_val = amounts[0] if amounts[0] > 0 else None
                    else:
                        # Money IN → ERP Debit
                        debit_val = amounts[0] if amounts[0] > 0 else None

                # Create transaction if we have valid amounts
                if debit_val or credit_val:
                    tx = BankTransaction(
                        bank_name="KBANK",
                        acc_no=acc_no or "",
                        debit=debit_val,
                        credit=credit_val,
                        date=self._parse_ocr_date(date_str),
                        description=description,
                        currency=currency,
                        transaction_id="",
                        beneficiary_bank="",
                        beneficiary_acc_no="",
                        beneficiary_acc_name=""
                    )
                    transactions.append(tx)

                # Move to next unprocessed line
                i = j
            else:
                i += 1

        return transactions

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse KBANK balance from OCR text.

        KBANK PDF OCR format (multi-line):
        - "Beginning Balance" followed by opening balance on next line
        - "Cộng phát sinh Total" followed by 3 amounts: debit total, credit total, closing balance
        """
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        opening = 0.0
        closing = 0.0

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Opening balance - "Beginning Balance" line
            # OCR format: amount may be on same line or next line
            if "beginning balance" in line_lower:
                # First try same line
                amounts = re.findall(r'[\d,]+\.\d{2}', line)
                if amounts:
                    val = self._parse_ocr_number(amounts[-1])
                    if val:
                        opening = val
                # If no amount on same line, check next line
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r'^[\d,]+\.\d{2}$', next_line):
                        val = self._parse_ocr_number(next_line)
                        if val:
                            opening = val

            # Closing balance from "Cộng phát sinh Total" section
            # OCR format: 3 amounts on following lines (debit total, credit total, closing balance)
            if 'cộng phát sinh' in line_lower or ('total' in line_lower and i > 10):
                # First try same line
                amounts = re.findall(r'[\d,]+\.\d{2}', line)
                if amounts and len(amounts) >= 3:
                    val = self._parse_ocr_number(amounts[-1])
                    if val:
                        closing = val
                else:
                    # Look for amounts on following lines
                    total_amounts = []
                    j = i + 1
                    while j < len(lines) and len(total_amounts) < 3:
                        next_line = lines[j].strip()
                        if re.match(r'^[\d,]+\.\d{2}$', next_line):
                            val = self._parse_ocr_number(next_line)
                            if val is not None:
                                total_amounts.append(val)
                        elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}', next_line):
                            # Hit a new transaction, stop
                            break
                        j += 1

                    # Last amount is closing balance
                    if len(total_amounts) >= 3:
                        closing = total_amounts[-1]
                    elif total_amounts:
                        closing = total_amounts[-1]

        # If no closing found, use opening as fallback
        if closing == 0 and opening > 0:
            closing = opening

        if opening == 0 and closing == 0:
            return None

        return BankBalance(
            bank_name="KBANK",
            acc_no=acc_no or "",
            currency=currency,
            opening_balance=opening,
            closing_balance=closing
        )

    def _extract_account_from_ocr(self, text: str) -> Optional[str]:
        """Extract account number from OCR text."""
        # Look for "account" keyword followed by digits
        patterns = [
            r'account\s*(?:no|number|#)?[:\s]*(\d{8,})',
            r'số\s*(?:tài khoản|tk)[:\s]*(\d{8,})',
            r'a/c[:\s]*(\d{8,})',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)

        # Fallback: find any 10+ digit sequence
        all_digits = re.findall(r'\d{10,}', text)
        if all_digits:
            return all_digits[0]

        return None

    def _extract_currency_from_ocr(self, text: str) -> str:
        """Extract currency from OCR text."""
        text_lower = text.lower()
        if "usd" in text_lower:
            return "USD"
        elif "thb" in text_lower or "baht" in text_lower:
            return "THB"
        return "VND"

    def _parse_ocr_number(self, value: str) -> Optional[float]:
        """Parse number from OCR text."""
        if not value:
            return None

        # Remove commas and spaces
        cleaned = value.replace(",", "").replace(" ", "")

        try:
            num = float(cleaned)
            return num  # Return 0 too, important for amount arrays
        except (ValueError, TypeError):
            return None

    def _parse_ocr_date(self, date_str: str):
        """Parse date from OCR text."""
        if not date_str:
            return None

        # Normalize separators
        date_str = date_str.replace("-", "/")

        try:
            parsed = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
            if pd.notna(parsed):
                return parsed.date()
        except Exception:
            pass

        return None

    def _extract_balance_value(self, line: str, all_lines: List[str], line_idx: int) -> float:
        """Extract balance value from line or subsequent lines."""
        # Try to find number on same line
        numbers = re.findall(r'[\d,]+\.?\d*', line)
        for num_str in reversed(numbers):
            val = self._parse_ocr_number(num_str)
            if val and val > 0:
                return val

        # Check next few lines
        for j in range(line_idx + 1, min(line_idx + 3, len(all_lines))):
            next_line = all_lines[j]
            numbers = re.findall(r'[\d,]+\.?\d*', next_line)
            for num_str in numbers:
                val = self._parse_ocr_number(num_str)
                if val and val > 0:
                    return val

        return 0.0
