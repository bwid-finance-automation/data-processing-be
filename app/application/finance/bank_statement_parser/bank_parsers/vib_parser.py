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

        Gemini OCR format - multiple variations:
        1. Space-separated: 7047991960 29/11/2025 29/11/2025 CRIN 0 1,217 3,065,877
        2. Pipe-separated: 7047991960 | 29/11/2025 | 29/11/2025 | CRIN | | 0 | 1,217 | 3,065,877 |
        3. Multiline: transaction header on one line, amounts on subsequent lines

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

            # Log first few for debugging
            for i, tl in enumerate(tx_lines[:5]):
                logger.info(f"VIB TX line {i+1}: [{tl.strip()}]")

            # Track current account info as we parse
            current_acc_no = None
            current_currency = "VND"
            processed_tx_ids = set()  # Avoid duplicates

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Update account info when we see account line
                # Multiple patterns to catch different OCR formats
                if any(marker in line.lower() for marker in ["số tk", "so tk", "a/c no", "account no"]):
                    # Pattern 1: "053376900 651/VND"
                    acc_match = re.search(r'(\d{9,12})\s+\d*/\s*(VND|USD)', line, re.IGNORECASE)
                    if acc_match:
                        current_acc_no = acc_match.group(1)
                        current_currency = acc_match.group(2).upper()
                        logger.info(f"VIB: Found account {current_acc_no} ({current_currency})")
                    else:
                        # Pattern 2: Just account number
                        acc_match2 = re.search(r'(\d{9,12})', line)
                        if acc_match2:
                            current_acc_no = acc_match2.group(1)
                            # Check for currency in same line
                            if 'USD' in line.upper():
                                current_currency = "USD"
                            elif 'VND' in line.upper() or 'VNĐ' in line:
                                current_currency = "VND"
                            logger.info(f"VIB: Found account {current_acc_no} ({current_currency}) via pattern 2")
                    continue

                # Skip header lines
                if any(skip in line.lower() for skip in [
                    'seq. no', 'số ct', 'ngày gd', 'tran date', 'withdrawal', 'deposit',
                    'phát sinh nợ', 'phát sinh có', 'nội dung', 'remarks', 'reference',
                    'cheque no', 'loại gd'
                ]):
                    continue

                # Skip balance/summary lines
                if any(skip in line.lower() for skip in [
                    'số dư đầu', 'số dư cuối', 'opening balance', 'ending balance',
                    'available balance', 'transaction summary', 'doanh số giao dịch',
                    'tổng số giao dịch', 'number of transactions', 'số dư khả dụng'
                ]):
                    continue

                # Parse transaction line - starts with 10-digit transaction ID
                if re.match(r'^\d{10}', line):
                    tx = None

                    # Extract transaction ID first to check for duplicates
                    tx_id_match = re.match(r'^(\d{10})', line)
                    tx_id = tx_id_match.group(1) if tx_id_match else None

                    if tx_id and tx_id in processed_tx_ids:
                        continue  # Skip duplicate

                    if '|' in line:
                        # Format with | separators
                        tx = self._parse_pipe_separated_transaction(line, current_acc_no, current_currency)
                    else:
                        # Try full line match first (all data on one line)
                        # Pattern: TxID Date Date Code Amounts...
                        tx_match = re.match(
                            r'^(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s+([A-Z][A-Z0-9]{0,5})\s+(.+)$',
                            line
                        )
                        if tx_match:
                            tx = self._parse_space_separated_transaction(tx_match, current_acc_no, current_currency)
                        else:
                            # Try multiline format (header only, amounts on next lines)
                            tx_match_partial = re.match(
                                r'^(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s*([A-Z][A-Z0-9]{0,5})?$',
                                line
                            )
                            if tx_match_partial:
                                tx = self._parse_multiline_transaction(tx_match_partial, lines, line_idx, current_acc_no, current_currency)
                            else:
                                # Try even simpler format: just TxID and date
                                tx_match_simple = re.match(r'^(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})', line)
                                if tx_match_simple:
                                    tx = self._parse_flexible_transaction(line, lines, line_idx, current_acc_no, current_currency)

                    if tx:
                        if tx.transaction_id not in processed_tx_ids:
                            transactions.append(tx)
                            processed_tx_ids.add(tx.transaction_id)
                            logger.debug(f"VIB: Parsed tx {tx.transaction_id} for account {tx.acc_no}")

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

                # Skip lines that look like date labels (e.g., "16 November 2025", "30 November 2025")
                # These contain month names and would extract wrong numbers
                if any(month in next_line.lower() for month in [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december',
                    'tháng', 'thang'
                ]):
                    logger.debug(f"VIB multiline: skipping date line: {next_line[:50]}")
                    continue

                # Extract numbers from this line
                line_numbers = re.findall(r'[\d,]+(?:\.\d+)?', next_line)
                for num_str in line_numbers:
                    # Skip numbers that look like years (2020-2030) without commas
                    if re.match(r'^20[2-3]\d$', num_str):
                        logger.debug(f"VIB multiline: skipping year-like number: {num_str}")
                        continue
                    val = self._parse_number_from_text(num_str)
                    if val is not None:
                        amounts.append(val)

                # If we found 3+ numbers, we have enough for withdrawal/deposit/balance
                if len(amounts) >= 3:
                    break

            logger.info(f"VIB multiline: tx_id={tx_id}, found amounts={amounts[:5]}")

            # Need at least 1 amount (some transactions only have withdrawal OR deposit)
            if len(amounts) < 1:
                return None

            # First amount = Withdrawal (Phát sinh nợ)
            # Second amount = Deposit (Phát sinh có)
            # Third amount = Balance (skip if present)
            withdrawal = amounts[0] if len(amounts) > 0 else 0
            deposit = amounts[1] if len(amounts) > 1 else 0

            # If only 1 amount found, it's likely the non-zero value
            # Check if the single amount should be withdrawal or deposit based on context
            if len(amounts) == 1 and amounts[0] > 0:
                # Default: assume it's a deposit (credit) unless we can determine otherwise
                withdrawal = 0
                deposit = amounts[0]

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

    def _parse_flexible_transaction(self, line: str, lines: List[str], current_idx: int, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Flexible transaction parser for non-standard formats.

        Handles various OCR output formats where transaction data
        may be split or formatted differently.
        """
        try:
            # Extract transaction ID (first 10 digits)
            tx_id_match = re.match(r'^(\d{10})', line)
            if not tx_id_match:
                return None
            tx_id = tx_id_match.group(1)

            # Try to find date in the line
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            date_str = date_match.group(1) if date_match else None

            # Extract all numbers from this line and subsequent lines
            amounts = []

            # First, get numbers from current line (after tx_id and dates)
            remaining = re.sub(r'^\d{10}', '', line)  # Remove tx_id
            remaining = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', remaining)  # Remove dates
            remaining = re.sub(r'[A-Z]{2,5}', '', remaining)  # Remove transaction codes

            line_numbers = re.findall(r'[\d,]+(?:\.\d+)?', remaining)
            for num_str in line_numbers:
                if re.match(r'^20[2-3]\d$', num_str):  # Skip years
                    continue
                val = self._parse_number_from_text(num_str)
                if val is not None:
                    amounts.append(val)

            # If not enough amounts, look at subsequent lines
            if len(amounts) < 2:
                for i in range(current_idx + 1, min(current_idx + 8, len(lines))):
                    next_line = lines[i].strip()

                    # Stop conditions
                    if re.match(r'^\d{10}', next_line):  # Next transaction
                        break
                    if any(skip in next_line.lower() for skip in [
                        'seq. no', 'số ct', 'ngày gd', 'tran date', 'số dư đầu', 'số dư cuối',
                        'opening balance', 'ending balance', 'available balance'
                    ]):
                        break

                    # Skip date lines
                    if any(month in next_line.lower() for month in [
                        'january', 'february', 'march', 'april', 'may', 'june',
                        'july', 'august', 'september', 'october', 'november', 'december',
                        'tháng', 'thang'
                    ]):
                        continue

                    # Get numbers from this line
                    line_nums = re.findall(r'[\d,]+(?:\.\d+)?', next_line)
                    for num_str in line_nums:
                        if re.match(r'^20[2-3]\d$', num_str):
                            continue
                        val = self._parse_number_from_text(num_str)
                        if val is not None:
                            amounts.append(val)

                    if len(amounts) >= 3:
                        break

            if len(amounts) < 2:
                logger.debug(f"VIB flexible: tx_id={tx_id}, not enough amounts: {amounts}")
                return None

            # First amount = Withdrawal, Second = Deposit
            withdrawal = amounts[0] if len(amounts) > 0 else 0
            deposit = amounts[1] if len(amounts) > 1 else 0

            if withdrawal == 0 and deposit == 0:
                return None

            # Parse date
            tx_date = self._parse_date_from_text(date_str) if date_str else None

            logger.debug(f"VIB flexible: tx_id={tx_id}, withdrawal={withdrawal}, deposit={deposit}")

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=withdrawal if withdrawal > 0 else None,
                credit=deposit if deposit > 0 else None,
                date=tx_date,
                description="",
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            logger.error(f"Error in flexible transaction parser: {e}")
            return None

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse VIB balance information from Gemini OCR text.

        Returns the FIRST account's balance for backward compatibility.
        For multiple accounts, use parse_all_balances_from_text instead.
        """
        all_balances = self.parse_all_balances_from_text(text, file_name)
        return all_balances[0] if all_balances else None

    def parse_all_balances_from_text(self, text: str, file_name: str) -> List[BankBalance]:
        """
        Parse ALL VIB balance information from Gemini OCR text.

        VIB PDFs can contain multiple accounts (VND and USD).
        Each account section has:
        - Account number line: "SỐ TK/Loại TK/Loại tiền: 053376900 651/VND"
        - Opening balance: "Số dư đầu ngày" followed by amount
        - Closing balance: "Số dư cuối ngày" followed by amount

        Returns list of BankBalance objects, one per account (no duplicates).
        """
        try:
            balances = []
            processed_accounts = set()  # Track (acc_no, currency) to avoid duplicates

            # Split text into account sections
            # Each account section starts with account number pattern
            account_sections = self._split_into_account_sections(text)

            logger.info(f"VIB: Found {len(account_sections)} account sections")

            for section_idx, section in enumerate(account_sections):
                acc_no, currency = self._extract_account_info_from_section(section)

                if not acc_no:
                    logger.warning(f"VIB: Could not extract account number from section {section_idx}")
                    continue

                # Skip duplicate accounts (same acc_no + currency)
                account_key = (acc_no, currency)
                if account_key in processed_accounts:
                    logger.info(f"VIB: Skipping duplicate account {acc_no} ({currency})")
                    continue

                logger.info(f"VIB: Processing account {acc_no} ({currency})")

                # Extract opening and closing balance for this specific section
                opening = self._extract_section_balance(section, ["số dư đầu", "opening balance"])
                closing = self._extract_section_balance(section, ["số dư cuối", "ending balance"])

                # If balance extraction failed, try fallback with full section text
                if opening is None or opening == 0:
                    opening = self._extract_balance_fallback_section(section, ["số dư đầu", "opening balance"], currency)
                if closing is None or closing == 0:
                    closing = self._extract_balance_fallback_section(section, ["số dư cuối", "ending balance"], currency)

                logger.info(f"VIB: Account {acc_no} - Opening: {opening}, Closing: {closing}")

                balances.append(BankBalance(
                    bank_name="VIB",
                    acc_no=acc_no,
                    currency=currency,
                    opening_balance=opening or 0.0,
                    closing_balance=closing or 0.0
                ))
                processed_accounts.add(account_key)

            # If no sections found, try legacy single-account parsing
            if not balances:
                logger.info("VIB: No account sections found, trying legacy parsing")
                legacy_balance = self._parse_single_balance_legacy(text)
                if legacy_balance:
                    balances.append(legacy_balance)

            logger.info(f"VIB: Total balances parsed: {len(balances)}")
            return balances

        except Exception as e:
            logger.error(f"Error parsing VIB balances from text: {e}")
            return []

    def _extract_balance_fallback_section(self, section: str, labels: List[str], currency: str) -> Optional[float]:
        """
        Fallback balance extraction for a section when primary method fails.
        """
        section_lower = section.lower()

        for label in labels:
            label_lower = label.lower()
            pos = section_lower.find(label_lower)

            if pos == -1:
                continue

            # Extract window after the label
            window = section[pos:pos + 400]

            # For USD - look for decimal numbers
            if currency == "USD":
                usd_matches = re.findall(r'(\d+\.\d{2})', window)
                for num_str in usd_matches:
                    val = float(num_str)
                    if val >= 0:
                        logger.info(f"VIB fallback (USD): found {val} for '{label}'")
                        return val

            # For VND - look for comma-separated numbers with proper format
            vnd_matches = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', window)
            for num_str in vnd_matches:
                # Skip transaction IDs (exactly 10 digits)
                digits = num_str.replace(',', '')
                if len(digits) == 10:
                    continue
                val = self._parse_number_from_text(num_str)
                if val is not None and val >= 1000:
                    logger.info(f"VIB fallback (VND): found {val} for '{label}'")
                    return val

        return None

    def _split_into_account_sections(self, text: str) -> List[str]:
        """
        Split OCR text into sections, one per account.

        Account sections are identified by:
        - "SỐ TK" or "Số TK" or "A/C No" patterns with account numbers
        - Or "Statement of Account For" patterns
        """
        sections = []

        # Pattern to match account header lines
        # Matches: "SỐ TK/Loại TK/Loại tiền: 053376900 651/VND" or similar
        account_pattern = re.compile(
            r'(số\s*tk|so\s*tk|a/c\s*no|account\s*no|statement\s*of\s*account)',
            re.IGNORECASE
        )

        lines = text.split('\n')
        current_section_start = 0
        section_starts = []

        for i, line in enumerate(lines):
            # Check if this line contains account number pattern
            if account_pattern.search(line):
                # Verify it has an actual account number (9-12 digits)
                if re.search(r'\d{9,12}', line):
                    section_starts.append(i)

        # If no account headers found, return entire text as one section
        if not section_starts:
            return [text]

        # Split text into sections
        for i, start_idx in enumerate(section_starts):
            if i + 1 < len(section_starts):
                end_idx = section_starts[i + 1]
            else:
                end_idx = len(lines)

            section_text = '\n'.join(lines[start_idx:end_idx])
            sections.append(section_text)

        return sections

    def _extract_account_info_from_section(self, section: str) -> tuple:
        """
        Extract account number and currency from a section.

        Returns (acc_no, currency) tuple.
        """
        acc_no = None
        currency = "VND"  # Default

        # Pattern 1: "053376900 651/VND" or "045681644 651/USD"
        match = re.search(r'(\d{9,12})\s+\d*/\s*(VND|USD)', section, re.IGNORECASE)
        if match:
            acc_no = match.group(1)
            currency = match.group(2).upper()
            return acc_no, currency

        # Pattern 2: Account number followed by currency on same or nearby line
        acc_match = re.search(r'(\d{9,12})', section)
        if acc_match:
            acc_no = acc_match.group(1)

            # Look for currency nearby
            if 'USD' in section.upper():
                currency = "USD"
            elif 'VND' in section.upper() or 'VNĐ' in section:
                currency = "VND"

        return acc_no, currency

    def _extract_section_balance(self, section: str, labels: List[str]) -> Optional[float]:
        """
        Extract balance from a specific section.

        This is more accurate than searching the entire text because
        it only looks within the account's section.

        IMPORTANT: Must avoid picking up:
        - Transaction IDs (10-digit numbers like 7048070432)
        - Credit/Debit amounts from transaction lines
        - Date components
        """
        section_lower = section.lower()
        lines = section.split('\n')

        for label in labels:
            label_lower = label.lower()

            # Find line index containing the label
            label_line_idx = -1
            for i, line in enumerate(lines):
                if label_lower in line.lower():
                    label_line_idx = i
                    break

            if label_line_idx == -1:
                continue

            # Search in lines after the label (not in transaction table area)
            # Balance should be within 5 lines of the label
            for i in range(label_line_idx, min(label_line_idx + 6, len(lines))):
                line = lines[i].strip()

                # Skip lines that look like transaction lines (start with 10-digit ID)
                if re.match(r'^\d{10}', line):
                    continue

                # Skip header lines
                if any(skip in line.lower() for skip in [
                    'seq.', 'withdrawal', 'deposit', 'phát sinh', 'tran date', 'effect date'
                ]):
                    continue

                # For USD accounts - look for decimal numbers like "56.00", "100.00"
                if 'usd' in section_lower:
                    usd_match = re.search(r'(\d+\.\d{2})\b', line)
                    if usd_match:
                        val = float(usd_match.group(1))
                        logger.info(f"VIB section balance (USD): found {val} for '{label}'")
                        return val

                # For VND - look for comma-separated numbers
                # Pattern: number with at least one comma, representing >= 1,000
                vnd_matches = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', line)
                for num_str in vnd_matches:
                    # Skip if it looks like a transaction ID (10 digits no comma)
                    digits_only = num_str.replace(',', '')
                    if len(digits_only) == 10:
                        continue

                    val = self._parse_number_from_text(num_str)
                    if val is not None and val >= 1000:
                        logger.info(f"VIB section balance (VND): found {val} for '{label}'")
                        return val

                # Check if line is just a number (balance on its own line)
                clean_line = line.replace(',', '').replace('.', '').strip()
                if clean_line.isdigit() and len(clean_line) >= 3:
                    # Might be a balance value
                    if '.' in line and re.match(r'^\d+\.\d{2}$', line.strip()):
                        # USD format
                        val = float(line.strip())
                        logger.info(f"VIB section balance (USD line): found {val} for '{label}'")
                        return val
                    elif ',' in line:
                        # VND format
                        val = self._parse_number_from_text(line)
                        if val and val >= 100:
                            logger.info(f"VIB section balance (VND line): found {val} for '{label}'")
                            return val

        return None

    def _parse_single_balance_legacy(self, text: str) -> Optional[BankBalance]:
        """
        Legacy single-balance parsing for backward compatibility.
        """
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
        if opening is None or opening == 0:
            opening = self._extract_balance_fallback(text, ["số dư đầu", "opening balance"])

        # Extract closing balance
        closing = self._extract_balance_gemini(lines, ["số dư cuối", "ending balance"])
        if closing is None or closing == 0:
            closing = self._extract_balance_fallback(text, ["số dư cuối", "ending balance"])

        if acc_no:
            return BankBalance(
                bank_name="VIB",
                acc_no=acc_no,
                currency=currency,
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )
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

    def _extract_balance_fallback(self, text: str, labels: List[str]) -> Optional[float]:
        """
        Fallback balance extraction - search for large numbers near balance labels.

        OCR text can be messy with balance values far from labels.
        This method finds the label position and searches nearby for large numbers.
        """
        text_lower = text.lower()

        for label in labels:
            label_lower = label.lower()
            pos = text_lower.find(label_lower)

            if pos == -1:
                continue

            # Extract a window of text after the label (up to 500 chars)
            window_start = pos
            window_end = min(pos + 500, len(text))
            window = text[window_start:window_end]

            # Find all large numbers (with comma separator, 5+ digits)
            numbers = re.findall(r'[\d,]+', window)

            for num_str in numbers:
                # Must have comma (thousand separator) and be large enough
                if ',' in num_str and len(num_str.replace(',', '')) >= 5:
                    val = self._parse_number_from_text(num_str)
                    if val is not None and val > 0:
                        logger.info(f"VIB balance fallback found: {val} near '{label}'")
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
