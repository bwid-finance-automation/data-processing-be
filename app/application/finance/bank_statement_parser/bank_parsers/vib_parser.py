"""VIB bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class VIBParser(BaseBankParser):
    """Parser for VIB (Vietnam International Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "VIB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is VIB bank statement.

        Logic from fxLooksLike_VIB:
        - Check sheet name for "CURRENTACCOUNTSTATEMENT"
        - OR check content for VIB-specific markers:
          * "sao kê theo ngày giao dịch"
          * "số dư đầu kỳ"
          * "số chứng từ"
          * "ghi nợ" and "ghi có"
        - Exclude VCB and ACB markers
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))

            # Check sheet name first (unique to VIB)
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            is_vib_sheet = "CURRENTACCOUNTSTATEMENT" in sheet_name.upper()

            if is_vib_sheet:
                return True

            # Check content
            df = pd.read_excel(xls, sheet_name=0, header=None)
            top_25 = df.head(25)

            # Convert all cells to lowercase text
            all_text = []
            for col in top_25.columns:
                all_text.extend([self.to_text(cell).lower() for cell in top_25[col]])

            combined = "|".join(all_text)

            # VIB-specific markers
            has_sao_ke_theo_ngay = "sao kê theo ngày giao dịch" in combined
            has_so_du_dau_ky = "số dư đầu kỳ" in combined
            has_so_chung_tu = "số chứng từ" in combined
            has_ghi_no = "ghi nợ" in combined
            has_ghi_co = "ghi có" in combined

            # Exclude VCB markers
            not_vcb = "vietcombank" not in combined and "statement of account" not in combined

            # Exclude ACB markers
            not_acb = "bảng sao kê giao dịch" not in combined and \
                      "rút ra" not in combined and \
                      "gửi vào" not in combined

            # VIB if: has VIB markers AND not VCB/ACB
            is_vib = has_sao_ke_theo_ngay and has_so_du_dau_ky and \
                     has_so_chung_tu and not_vcb and not_acb

            return is_vib

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse VIB transactions.

        Logic from fxParse_VIB_Transactions:
        - Find header row with "Ngày giao dịch"
        - Extract columns: Ngày giao dịch, Số chứng từ, Mô tả, Ghi nợ, Ghi có, Loại tiền, Số dư
        - Map: Debit = "Ghi có", Credit = "Ghi nợ" (REVERSED from ACB!)
        - Extract account number from top 12 rows
        - Drop rows where both Debit and Credit are zero
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))

            # Try Statement sheet, otherwise first sheet
            try:
                sheet = pd.read_excel(xls, sheet_name="Statement", header=None)
            except:
                sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_50 = sheet.head(50)
            start_at = self._find_header_row_vib(top_50)

            # ========== Promote Headers ==========
            data = sheet.iloc[start_at:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Select VIB Columns ==========
            vib_columns = ["Ngày giao dịch", "Số chứng từ", "Mô tả", "Ghi nợ", "Ghi có", "Loại tiền", "Số dư"]

            # Filter to existing columns only
            available_cols = [col for col in vib_columns if col in data.columns]
            if not available_cols:
                return []

            data = data[available_cols].copy()

            # ========== Extract Account Number and Currency ==========
            top_12 = sheet.head(12)
            acc_no = self._extract_account_number_vib(top_12)
            currency_hdr = self._extract_currency_vib(top_12)

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                # VIB REVERSED MAPPING: Debit = "Ghi có", Credit = "Ghi nợ"
                debit_val = self._fix_number_vib(row.get("Ghi có")) if "Ghi có" in row else None
                credit_val = self._fix_number_vib(row.get("Ghi nợ")) if "Ghi nợ" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Extract currency from row, fallback to header
                currency = row.get("Loại tiền", "")
                if not currency or pd.isna(currency) or str(currency).strip() == "":
                    currency = currency_hdr or "VND"
                else:
                    currency = str(currency).strip()

                tx = BankTransaction(
                    bank_name="VIB",
                    acc_no=acc_no or "",
                    debit=debit_val,
                    credit=credit_val,
                    date=self.fix_date(row.get("Ngày giao dịch")) if "Ngày giao dịch" in row else None,
                    description=self.to_text(row.get("Mô tả", "")) if "Mô tả" in row else "",
                    currency=currency,
                    transaction_id=self.to_text(row.get("Số chứng từ", "")) if "Số chứng từ" in row else "",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing VIB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse VIB balance information.

        Logic from fxParse_VIB_Balances:
        - Search first 120 rows for balance labels
        - Use smart number extraction (handles embedded numbers)
        - Extract: Số dư đầu kỳ, Số dư cuối kỳ
        - Extract account number and currency
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))

            try:
                sheet = pd.read_excel(xls, sheet_name="Statement", header=None)
            except:
                sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 120 rows
            top_120 = sheet.head(120)
            rows = top_120.values.tolist()

            # ========== Extract Account Number ==========
            acc_no_text = self._find_right_text_contains(rows, ["Số tài khoản", "so tai khoan"])
            acc_no = ''.join(c for c in (acc_no_text or "") if c.isdigit())

            # ========== Extract Currency ==========
            curr_text = self._find_right_text_contains(rows, ["Loại tiền", "loai tien"])
            currency = curr_text.strip() if curr_text and curr_text.strip() else "VND"

            # ========== Extract Balances ==========
            opening = self._find_right_or_same_num_contains(rows, ["Số dư đầu kỳ", "so du dau ky"])
            closing = self._find_right_or_same_num_contains(rows, ["Số dư cuối kỳ", "so du cuoi ky"])

            return BankBalance(
                bank_name="VIB",
                acc_no=acc_no,
                currency=currency,
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing VIB balances: {e}")
            return None

    # ========== VIB-Specific Helper Methods ==========

    def _find_header_row_vib(self, top_df: pd.DataFrame) -> int:
        """Find row containing 'Ngày giao dịch'."""
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell) for cell in row])
            if "Ngày giao dịch" in row_text:
                return int(idx)
        return 14  # Default fallback

    def _extract_account_number_vib(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract account number from top rows (last 5+ digit sequence)."""
        # Flatten all cells to text
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell) for cell in top_df[col]])

        flat = " ".join(all_text)

        # Extract digit sequences
        digits = ''.join(c if c.isdigit() else ' ' for c in flat)
        parts = [p for p in digits.split() if len(p) >= 5]

        if parts:
            return parts[-1]  # Return last match
        return None

    def _extract_currency_vib(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract currency from top rows."""
        # Flatten all cells to lowercase text
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell).lower() for cell in top_df[col]])

        flat = " ".join(all_text)

        if "vnd" in flat or "vnđ" in flat:
            return "VND"
        elif "usd" in flat:
            return "USD"
        return None

    def _fix_number_vib(self, value) -> Optional[float]:
        """
        VIB-specific number parser.
        Handles: VND prefix, parentheses negatives, thousand separators.
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

        # Remove spaces and thousand separators (dots and commas)
        txt = txt.replace(" ", "").replace(".", "").replace(",", "")

        try:
            return float(txt)
        except (ValueError, TypeError):
            return None

    def _find_row_idx_contains(self, rows: list, labels: list) -> int:
        """Find first row containing any of the labels."""
        for i, row in enumerate(rows):
            row_text_lower = " ".join([self.to_text(cell).lower() for cell in row])
            for label in labels:
                if label.lower() in row_text_lower:
                    return i
        return -1

    def _find_pos_in_row_contains(self, row: list, labels: list) -> int:
        """Find first cell position in row containing any of the labels."""
        for i, cell in enumerate(row):
            cell_text_lower = self.to_text(cell).lower()
            for label in labels:
                if label.lower() in cell_text_lower:
                    return i
        return -1

    def _find_right_or_same_num_contains(self, rows: list, labels: list) -> Optional[float]:
        """
        Find numeric value to the right of (or in same cell as) a label.
        Uses smart number extraction.
        """
        ridx = self._find_row_idx_contains(rows, labels)
        if ridx == -1:
            return None

        row = rows[ridx]
        pos = self._find_pos_in_row_contains(row, labels)
        if pos == -1:
            return None

        # Try cells to the right first
        right_cells = row[pos+1:]
        for cell in right_cells:
            val = self._fix_number_vib(cell)
            if val is not None:
                return val

        # Try same cell (smart extraction)
        val_same = self._num_smart(row[pos])
        return val_same

    def _num_smart(self, value) -> Optional[float]:
        """
        Smart number extraction - extracts digits from text.
        Example: "Số dư: 1234567" → 1234567
        """
        # First try normal parsing
        normal = self._fix_number_vib(value)
        if normal is not None:
            return normal

        # Extract digits and minus signs
        txt = self.to_text(value)
        digits = ''.join(c if c.isdigit() or c == '-' else ' ' for c in txt)
        parts = [p for p in digits.split() if p]

        if not parts:
            return None

        # Take last number sequence
        best = parts[-1]
        try:
            return float(best)
        except (ValueError, TypeError):
            return None

    def _find_right_text_contains(self, rows: list, labels: list) -> Optional[str]:
        """Find text value to the right of a label."""
        ridx = self._find_row_idx_contains(rows, labels)
        if ridx == -1:
            return None

        row = rows[ridx]
        pos = self._find_pos_in_row_contains(row, labels)
        if pos == -1:
            return None

        # Get cells to the right
        right_cells = row[pos+1:]
        for cell in right_cells:
            text = self.to_text(cell).strip()
            if text:
                return text

        return None

    # ========== OCR Text Parsing Methods ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if this parser can handle the given OCR text.

        VIB Detection Logic:
        - Text contains "VIB" (case-insensitive)
        - Text contains "SO TK" or "ACCOUNT NO"
        - Text contains "SO DU" or "BALANCE"
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for VIB marker
        has_vib = "vib" in text_lower

        # Check for account number marker
        has_account = "so tk" in text_lower or "account no" in text_lower or "số tk" in text_lower

        # Check for balance marker
        has_balance = "so du" in text_lower or "balance" in text_lower or "số dư" in text_lower

        return has_vib and has_account and has_balance

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse VIB transactions from OCR text.

        AI Builder OCR splits transaction data across multiple lines:
        Line 1: TransactionID + Date1
        Line 2: Date2
        Line 3: TranCode
        Line 4: Debit
        Line 5: Credit
        Line 6: Balance
        Line 7+: Description (optional)

        IMPORTANT: Apply REVERSED mapping same as Excel parser
        - Debit output = Credit column in source (Phat sinh co / Deposit)
        - Credit output = Debit column in source (Phat sinh no / Withdrawal)
        """
        try:
            transactions = []

            # Extract account number
            acc_no = self._extract_account_from_text(text)

            # Extract currency from the account line specifically
            currency = self._extract_currency_from_text(text)

            lines = text.split('\n')
            lines = [line.strip() for line in lines]

            i = 0
            while i < len(lines):
                line = lines[i]

                # Look for transaction start: TransactionID (8-12 digits) followed by date
                # Pattern: "7047991960 29/11/2025"
                tx_start_match = re.match(r'^(\d{8,12})\s+(\d{1,2}/\d{1,2}/\d{4})$', line)

                if tx_start_match:
                    tx_id = tx_start_match.group(1)
                    date_str = tx_start_match.group(2)

                    # Collect next lines for transaction data
                    # Expected: Date2, TranCode, Debit, Credit, Balance
                    tx_lines = [line]
                    j = i + 1

                    # Collect up to 10 more lines until we hit another transaction or section
                    while j < len(lines) and j < i + 10:
                        next_line = lines[j]
                        # Stop if we hit another transaction ID
                        if re.match(r'^\d{8,12}\s+\d{1,2}/\d{1,2}/\d{4}$', next_line):
                            break
                        # Stop if we hit a section header (So du cuoi, etc.)
                        if any(marker in next_line.lower() for marker in ['so du cuoi', 'ending balance', 'closing balance']):
                            break
                        tx_lines.append(next_line)
                        j += 1

                    # Parse the collected transaction data
                    tx_data = self._parse_transaction_block(tx_lines)

                    if tx_data:
                        debit_raw = tx_data.get('debit', 0)
                        credit_raw = tx_data.get('credit', 0)

                        # REVERSED MAPPING: Debit = Credit column (Deposit), Credit = Debit column (Withdrawal)
                        debit_val = credit_raw
                        credit_val = debit_raw

                        # Skip if both are zero
                        if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                            i = j
                            continue

                        # Parse date
                        tx_date = self._parse_date_from_text(date_str)

                        tx = BankTransaction(
                            bank_name="VIB",
                            acc_no=acc_no or "",
                            debit=debit_val,
                            credit=credit_val,
                            date=tx_date,
                            description=tx_data.get('description', ''),
                            currency=currency,
                            transaction_id=tx_id,
                            beneficiary_bank="",
                            beneficiary_acc_no="",
                            beneficiary_acc_name=""
                        )
                        transactions.append(tx)

                    i = j
                else:
                    i += 1

            return transactions

        except Exception as e:
            print(f"Error parsing VIB transactions from text: {e}")
            return []

    def _parse_transaction_block(self, lines: List[str]) -> Optional[dict]:
        """
        Parse a block of lines representing a single transaction.

        Expected line sequence after TransactionID line:
        - Date2 (DD/MM/YYYY)
        - TranCode (2-6 uppercase letters like CRIN, SC60, VATX)
        - Debit number (Phat sinh no / Withdrawal)
        - Credit number (Phat sinh co / Deposit)
        - Balance number
        - Description text (optional, may span multiple lines)
        """
        if len(lines) < 5:
            return None

        result = {
            'debit': 0,
            'credit': 0,
            'description': ''
        }

        # Skip first line (already parsed as tx_id + date)
        # Look for numbers and transaction code in remaining lines
        numbers_found = []
        description_lines = []

        for line in lines[1:]:  # Skip first line
            line = line.strip()
            if not line:
                continue

            # Check if it's a transaction code (2-6 uppercase letters) - skip
            if re.match(r'^[A-Z]{2,6}$', line):
                continue

            # Check if it's a date (skip)
            if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', line):
                continue

            # Check if it's a number (with comma separators)
            if re.match(r'^[\d,\.]+$', line):
                num_val = self._parse_number_from_text(line)
                if num_val is not None:
                    numbers_found.append(num_val)
                continue

            # Otherwise it's description text
            # Skip common OCR artifacts
            if line.lower() not in ['tran date effect date', 'withdrawal', 'deposit', 'balance', 'remarks']:
                description_lines.append(line)

        # Assign numbers: expect [Debit, Credit, Balance] in order
        if len(numbers_found) >= 3:
            result['debit'] = numbers_found[0]  # Phat sinh no / Withdrawal
            result['credit'] = numbers_found[1]  # Phat sinh co / Deposit
            # numbers_found[2] is Balance, not used
        elif len(numbers_found) == 2:
            result['debit'] = numbers_found[0]
            result['credit'] = numbers_found[1]

        # Join description
        result['description'] = ' '.join(description_lines)

        return result

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse VIB balance information from OCR text.

        Balance Markers:
        - Opening: Line containing "so du dau" or "opening balance"
        - Closing: Line containing "so du cuoi" or "closing balance"
        """
        try:
            # Extract account number
            acc_no = self._extract_account_from_text(text)

            # Extract currency
            currency = self._extract_currency_from_text(text)

            # Extract opening balance
            opening = self._extract_balance_from_text(
                text,
                ["so du dau", "số dư đầu", "opening balance"]
            )

            # Extract closing balance
            closing = self._extract_balance_from_text(
                text,
                ["so du cuoi", "số dư cuối", "closing balance"]
            )

            return BankBalance(
                bank_name="VIB",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing VIB balances from text: {e}")
            return None

    # ========== OCR Text Helper Methods ==========

    def _extract_account_from_text(self, text: str) -> Optional[str]:
        """Extract account number from OCR text."""
        # Look for pattern: "So TK" or "Account No" followed by digits
        # Example: "So TK/Loai TK/Loai tien: 053376900 651/VND"
        patterns = [
            r'(?:so tk|số tk|account no)[^0-9]*(\d{9,13})',  # 9-13 digit account
            r'(\d{9,13})\s*\d*/\s*(?:vnd|usd)',  # Account before currency
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)

        # Fallback: find first 9-13 digit sequence
        match = re.search(r'\b(\d{9,13})\b', text)
        if match:
            return match.group(1)

        return None

    def _extract_currency_from_text(self, text: str) -> str:
        """
        Extract currency from OCR text.

        Priority: Extract from account line (e.g., "053376900 651/VND")
        Format: "So TK/Loai TK/Loai tien: 053376900 651/VND"
        """
        # First, try to extract currency from the account line specifically
        # Pattern: account_number followed by /VND or /USD
        account_line_pattern = r'\d{9,13}\s+\d*/\s*(vnd|usd)'
        match = re.search(account_line_pattern, text.lower())
        if match:
            currency = match.group(1).upper()
            return currency

        # Alternative pattern: "Loai tien:" followed by currency
        loai_tien_pattern = r'loai\s*tien[:\s]+([a-zA-Z]{3})'
        match = re.search(loai_tien_pattern, text.lower())
        if match:
            currency = match.group(1).upper()
            if currency in ["VND", "USD"]:
                return currency

        # Default to VND (most common for VIB)
        return "VND"

    def _parse_number_from_text(self, value: str) -> Optional[float]:
        """Parse number from OCR text, handling comma separators."""
        if not value:
            return None

        # Remove commas and dots (thousand separators)
        cleaned = value.replace(",", "").replace(".", "")

        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _parse_date_from_text(self, date_str: str):
        """Parse date from DD/MM/YYYY format."""
        if not date_str:
            return None

        try:
            # Parse DD/MM/YYYY
            parts = date_str.split("/")
            if len(parts) == 3:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                from datetime import date
                return date(year, month, day)
        except (ValueError, IndexError):
            pass

        return None

    def _extract_balance_from_text(self, text: str, labels: list) -> Optional[float]:
        """
        Extract balance value following a label.

        AI Builder OCR often places the balance value on subsequent lines:
        Line N: "So du dau ky/Beginning Balance"
        Line N+1: "8,000,000"

        Strategy:
        1. Find line containing the label
        2. Check for number on same line first
        3. If not found, check subsequent lines for a number
        """
        lines = text.split('\n')
        lines = [line.strip() for line in lines]

        for label in labels:
            label_lower = label.lower()
            for i, line in enumerate(lines):
                if label_lower in line.lower():
                    # First, try to find number on same line
                    numbers = re.findall(r'[\d,\.]+', line)
                    for num_str in reversed(numbers):
                        val = self._parse_number_from_text(num_str)
                        if val is not None and val > 0:
                            return val

                    # If not found on same line, check subsequent lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue

                        # Check if line is a pure number (with comma separators)
                        if re.match(r'^[\d,\.]+$', next_line):
                            val = self._parse_number_from_text(next_line)
                            if val is not None and val > 0:
                                return val

                        # Check if line contains a number at end
                        numbers = re.findall(r'[\d,\.]+', next_line)
                        if numbers:
                            for num_str in reversed(numbers):
                                val = self._parse_number_from_text(num_str)
                                if val is not None and val > 0:
                                    return val

                        # Stop if we hit another balance label or section
                        if any(lbl.lower() in next_line.lower() for lbl in labels):
                            break

        return None

    def _extract_description_for_tx(self, full_text: str, tx_line: str, tx_id: str) -> str:
        """Extract description for a transaction - usually on next line or after pattern."""
        lines = full_text.split('\n')

        for i, line in enumerate(lines):
            if tx_id in line and line.strip() == tx_line.strip():
                # Check next line for description
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # If next line doesn't start with a transaction ID, it's description
                    if not re.match(r'^\d{8,12}\s', next_line):
                        return next_line

        return ""
