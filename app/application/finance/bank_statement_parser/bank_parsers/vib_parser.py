"""VIB bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


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
        - Map: Debit = "Ghi nợ" (tiền RA), Credit = "Ghi có" (tiền VÀO)
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
                # CORRECT MAPPING: Debit = "Ghi nợ" (tiền RA), Credit = "Ghi có" (tiền VÀO)
                debit_val = self._fix_number_vib(row.get("Ghi nợ")) if "Ghi nợ" in row else None
                credit_val = self._fix_number_vib(row.get("Ghi có")) if "Ghi có" in row else None

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

    # ========== OCR Text Parsing Methods (Gemini OCR format) ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if this parser can handle the given OCR text.

        VIB Detection Logic:
        - Text contains "VIB" (case-insensitive)
        - Text contains "SO TK" or "ACCOUNT NO" or "Sổ chi tiết tài khoản"
        - Text contains "SO DU" or "BALANCE" or "Số dư"
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for VIB marker
        has_vib = "vib" in text_lower

        # Check for account/statement markers
        has_account = any(marker in text_lower for marker in [
            "so tk", "số tk", "account no", "sổ chi tiết tài khoản", "statement of account"
        ])

        # Check for balance marker
        has_balance = any(marker in text_lower for marker in [
            "so du", "số dư", "balance", "opening balance", "ending balance"
        ])

        return has_vib and has_account and has_balance

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse VIB transactions from Gemini OCR text.

        Gemini OCR format - space-separated (no | separators):
        7047991960 29/11/2025 29/11/2025 CRIN 0 1,217 3,065,877

        Columns: Seq.No, Tran Date, Effect Date, Tran, Withdrawal, Deposit, Balance

        CORRECT MAPPING:
        - Debit = Withdrawal column (Phát sinh nợ) = tiền RA
        - Credit = Deposit column (Phát sinh có) = tiền VÀO
        """
        try:
            transactions = []
            lines = text.split('\n')

            logger.info(f"VIB OCR text length: {len(text)} chars, {len(lines)} lines")

            # Log lines that look like transactions (10-digit start)
            tx_lines = [l for l in lines if re.match(r'^\s*\d{10}', l.strip())]
            logger.info(f"VIB: Found {len(tx_lines)} potential transaction lines")

            # Log actual transaction lines for debugging format mismatch
            for i, tl in enumerate(tx_lines[:5]):  # Log first 5
                logger.info(f"VIB TX line {i+1}: [{tl.strip()}]")

            # Extract account number and currency for each page/section
            # VIB PDFs can have multiple accounts (VND and USD)
            current_acc_no = None
            current_currency = "VND"

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Update account info when we see account line
                # Format: "SỐ TK/Loại TK/Loại tiền: 053376900 651/VND"
                if "số tk" in line.lower() or "so tk" in line.lower() or "a/c no" in line.lower():
                    acc_match = re.search(r'(\d{9,12})\s+\d*/\s*(VND|USD)', line, re.IGNORECASE)
                    if acc_match:
                        current_acc_no = acc_match.group(1)
                        current_currency = acc_match.group(2).upper()
                    continue

                # Skip header lines
                if any(skip in line.lower() for skip in [
                    'seq. no', 'số ct', 'ngày gd', 'tran date', 'withdrawal', 'deposit',
                    'phát sinh nợ', 'phát sinh có', 'nội dung', 'remarks', 'reference'
                ]):
                    continue

                # Skip balance lines
                if any(skip in line.lower() for skip in [
                    'số dư đầu', 'số dư cuối', 'opening balance', 'ending balance',
                    'available balance', 'transaction summary', 'doanh số giao dịch',
                    'tổng số giao dịch', 'number of transactions'
                ]):
                    continue

                # Parse transaction line - starts with 10-digit transaction ID
                # Support both formats:
                # 1. With | separator: 7047991960  | 29/11/2025 | 29/11/2025 | CRIN | | 0 | 1,217 | 3,065,877 |
                # 2. Space-separated: 7047991960 29/11/2025 29/11/2025 CRIN 0 1,217 3,065,877

                if re.match(r'^\d{10}', line):
                    tx = None
                    if '|' in line:
                        # Format with | separators
                        logger.debug(f"VIB: Trying pipe-separated parse for: [{line}]")
                        tx = self._parse_pipe_separated_transaction(line, current_acc_no, current_currency)
                    else:
                        # Space-separated format
                        # Transaction code can be alphanumeric: CRIN, SC60, VATX, etc.
                        tx_match = re.match(r'^(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s+([A-Z][A-Z0-9]{1,5})\s+(.+)$', line)
                        if tx_match:
                            tx = self._parse_space_separated_transaction(tx_match, current_acc_no, current_currency)
                        else:
                            # Try multiline format: amounts may be on next line(s)
                            # Format: "7047991960 29/11/2025 29/11/2025 CRIN" (no amounts on same line)
                            tx_match_partial = re.match(r'^(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s+([A-Z][A-Z0-9]{1,5})$', line)
                            if tx_match_partial:
                                # Look for amounts in subsequent lines
                                tx = self._parse_multiline_transaction(tx_match_partial, lines, lines.index(line), current_acc_no, current_currency)
                                if tx:
                                    logger.info(f"VIB: Parsed multiline transaction: {tx.transaction_id}")
                            else:
                                logger.info(f"VIB: Regex failed for line: [{line[:100]}]")

                    if tx:
                        transactions.append(tx)
                    else:
                        logger.debug(f"VIB: No transaction parsed from: [{line[:80]}]")

            logger.info(f"VIB: Successfully parsed {len(transactions)} transactions from OCR text")
            return transactions

        except Exception as e:
            logger.error(f"Error parsing VIB transactions from text: {e}")
            return []

    def _parse_pipe_separated_transaction(self, line: str, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Parse a transaction line with | separators.

        Format: 7047991960  | 29/11/2025 | 29/11/2025 | CRIN | | 0 | 1,217 | 3,065,877 |
        Columns: Seq.No | Tran Date | Effect Date | Tran | Ref | Withdrawal | Deposit | Balance | Remarks
        """
        try:
            # Split by | and clean each part
            parts = [p.strip() for p in line.split('|')]

            if len(parts) < 8:
                return None

            # Extract fields
            tx_id = parts[0].strip()
            date_str = parts[1].strip()  # Tran Date
            # parts[2] = Effect Date (skip)
            # parts[3] = Tran code (skip)
            # parts[4] = Reference (skip)
            withdrawal_str = parts[5].strip()  # Phát sinh nợ / Withdrawal
            deposit_str = parts[6].strip()     # Phát sinh có / Deposit
            # parts[7] = Balance (skip)
            description = parts[8].strip() if len(parts) > 8 else ""

            # Validate transaction ID (10 digits)
            if not re.match(r'^\d{10}$', tx_id):
                return None

            # Parse amounts
            withdrawal = self._parse_number_from_text(withdrawal_str) or 0
            deposit = self._parse_number_from_text(deposit_str) or 0

            # Skip if both are zero
            if withdrawal == 0 and deposit == 0:
                return None

            # CORRECT MAPPING:
            # Debit = Withdrawal (Phát sinh nợ) = tiền RA
            # Credit = Deposit (Phát sinh có) = tiền VÀO
            debit_val = withdrawal
            credit_val = deposit

            # Parse date
            tx_date = self._parse_date_from_text(date_str)

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=debit_val if debit_val > 0 else None,
                credit=credit_val if credit_val > 0 else None,
                date=tx_date,
                description=description,
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            print(f"Error parsing VIB pipe-separated transaction: {e}")
            return None

    def _parse_multiline_transaction(self, match, lines: List[str], current_idx: int, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Parse transaction when amounts are on subsequent lines.

        Gemini OCR may split transaction data across lines:
        Line N: "7047991960 29/11/2025 29/11/2025 CRIN"
        Line N+1: "0"  (withdrawal)
        Line N+2: "1,217"  (deposit)
        Line N+3: "3,065,877"  (balance)

        Or amounts may be on a single line after the transaction header.
        """
        try:
            tx_id = match.group(1)
            date_str = match.group(2)
            # group(3) = Effect Date (skip)
            # group(4) = Tran code (skip)

            # Collect numbers from subsequent lines (until next transaction or header)
            amounts = []
            for i in range(current_idx + 1, min(current_idx + 10, len(lines))):
                next_line = lines[i].strip()

                # Stop if we hit another transaction line (starts with 10 digits)
                if re.match(r'^\d{10}', next_line):
                    break

                # Stop if we hit a header/label line
                if any(skip in next_line.lower() for skip in [
                    'seq. no', 'số ct', 'ngày gd', 'tran date', 'withdrawal', 'deposit',
                    'phát sinh', 'nội dung', 'remarks', 'reference', 'số dư đầu', 'số dư cuối',
                    'opening balance', 'ending balance', 'available balance', 'transaction summary'
                ]):
                    break

                # Extract numbers from this line
                line_numbers = re.findall(r'[\d,]+(?:\.\d+)?', next_line)
                for num_str in line_numbers:
                    val = self._parse_number_from_text(num_str)
                    if val is not None:
                        amounts.append(val)

                # If we found 3+ numbers, we have enough for withdrawal/deposit/balance
                if len(amounts) >= 3:
                    break

            logger.info(f"VIB multiline: tx_id={tx_id}, found amounts={amounts[:5]}")

            if len(amounts) < 2:
                # Need at least withdrawal and deposit
                return None

            # First amount = Withdrawal (Phát sinh nợ)
            # Second amount = Deposit (Phát sinh có)
            # Third amount = Balance (skip if present)
            withdrawal = amounts[0] if len(amounts) > 0 else 0
            deposit = amounts[1] if len(amounts) > 1 else 0

            # Skip if both are zero
            if withdrawal == 0 and deposit == 0:
                return None

            # CORRECT MAPPING:
            # Debit = Withdrawal (Phát sinh nợ) = tiền RA
            # Credit = Deposit (Phát sinh có) = tiền VÀO
            debit_val = withdrawal
            credit_val = deposit

            # Parse date
            tx_date = self._parse_date_from_text(date_str)

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=debit_val if debit_val > 0 else None,
                credit=credit_val if credit_val > 0 else None,
                date=tx_date,
                description="",
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            logger.error(f"Error parsing VIB multiline transaction: {e}")
            return None

    def _parse_space_separated_transaction(self, match, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Parse a space-separated transaction line.

        Format: 7047991960 29/11/2025 29/11/2025 CRIN 0 1,217 3,065,877
        Groups: (1)TxID (2)TranDate (3)EffectDate (4)TranCode (5)remaining: Withdrawal Deposit Balance
        """
        try:
            tx_id = match.group(1)
            date_str = match.group(2)
            # group(3) = Effect Date (skip)
            # group(4) = Tran code (skip)
            remaining = match.group(5).strip()

            # Parse remaining part: "0 1,217 3,065,877" or "50,000 0 3,015,877"
            # Find all numbers (with comma separators)
            numbers = re.findall(r'[\d,]+', remaining)

            if len(numbers) < 3:
                return None

            # First number = Withdrawal (Phát sinh nợ)
            # Second number = Deposit (Phát sinh có)
            # Third number = Balance (skip)
            withdrawal = self._parse_number_from_text(numbers[0]) or 0
            deposit = self._parse_number_from_text(numbers[1]) or 0

            # Skip if both are zero
            if withdrawal == 0 and deposit == 0:
                return None

            # CORRECT MAPPING:
            # Debit = Withdrawal (Phát sinh nợ) = tiền RA
            # Credit = Deposit (Phát sinh có) = tiền VÀO
            debit_val = withdrawal
            credit_val = deposit

            # Parse date
            tx_date = self._parse_date_from_text(date_str)

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=debit_val if debit_val > 0 else None,
                credit=credit_val if credit_val > 0 else None,
                date=tx_date,
                description="",
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            print(f"Error parsing VIB transaction line: {e}")
            return None

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse VIB balance information from Gemini OCR text.

        Gemini OCR format:
        Số dư đầu ngày : 16 November 2025
        Opening Balance as of
        3,064,660

        Số dư cuối ngày : 30 November 2025
        Ending Balance as
        3,010,877

        Returns the FIRST account's balance (usually VND account).
        """
        try:
            lines = text.split('\n')
            lines = [line.strip() for line in lines]

            # Find first account info
            acc_no = None
            currency = "VND"

            for line in lines:
                if "số tk" in line.lower() or "so tk" in line.lower() or "a/c no" in line.lower():
                    acc_match = re.search(r'(\d{9,12})\s+\d*/\s*(VND|USD)', line, re.IGNORECASE)
                    if acc_match:
                        acc_no = acc_match.group(1)
                        currency = acc_match.group(2).upper()
                        break

            # Extract opening balance
            opening = self._extract_balance_gemini(lines, ["số dư đầu", "opening balance"])

            # Extract closing balance
            closing = self._extract_balance_gemini(lines, ["số dư cuối", "ending balance"])

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

    def _extract_balance_gemini(self, lines: List[str], labels: List[str]) -> Optional[float]:
        """
        Extract balance value from Gemini OCR format.

        Gemini format places balance on a separate line after the label:
        Line N: "Số dư đầu ngày : 16 November 2025"
        Line N+1: "Opening Balance as of"
        Line N+2: "3,064,660"

        OR on same line:
        "Ending Balance as 3,010,877"
        """
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line contains any of the labels
            if any(label.lower() in line_lower for label in labels):
                # First check: if balance is on same line (e.g., "Ending Balance as 3,010,877")
                # Look for large number (6+ digits) with comma separator on this line
                same_line_numbers = re.findall(r'[\d,]+', line)
                for num_str in reversed(same_line_numbers):
                    # Must have comma (thousand separator) and be large enough
                    if ',' in num_str and len(num_str.replace(',', '')) >= 5:
                        val = self._parse_number_from_text(num_str)
                        if val is not None and val >= 0:
                            return val

                # Second check: Look for number on subsequent lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    check_line = lines[j].strip()

                    # Skip empty lines
                    if not check_line:
                        continue

                    # Skip lines that are clearly not balance values
                    if any(skip in check_line.lower() for skip in [
                        'opening', 'ending', 'available', 'balance', 'november', 'december',
                        'january', 'february', 'march', 'april', 'may', 'june', 'july',
                        'august', 'september', 'october', 'statement', 'transaction'
                    ]):
                        # But still check if there's a large number on this line
                        numbers = re.findall(r'[\d,]+', check_line)
                        for num_str in reversed(numbers):
                            if ',' in num_str and len(num_str.replace(',', '')) >= 5:
                                val = self._parse_number_from_text(num_str)
                                if val is not None and val >= 0:
                                    return val
                        continue

                    # Check if line is a pure number (with comma separators)
                    # Pattern: digits with commas like "3,064,660" or "3,010,877"
                    if re.match(r'^[\d,]+$', check_line):
                        val = self._parse_number_from_text(check_line)
                        if val is not None and val >= 0:
                            return val

                    # Check if line is a decimal number like "0.00"
                    if re.match(r'^[\d\.]+$', check_line):
                        val = float(check_line)
                        if val >= 0:
                            return val

        return None

    # ========== OCR Text Helper Methods ==========

    def _parse_number_from_text(self, value: str) -> Optional[float]:
        """Parse number from OCR text, handling comma separators."""
        if not value:
            return None

        # Remove commas (thousand separators)
        cleaned = value.replace(",", "")

        # Handle case where dot is thousand separator (European format)
        # If there's a dot and no decimal places after, treat as thousand separator
        if '.' in cleaned:
            parts = cleaned.split('.')
            if len(parts[-1]) == 3:  # Likely thousand separator
                cleaned = cleaned.replace(".", "")

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
