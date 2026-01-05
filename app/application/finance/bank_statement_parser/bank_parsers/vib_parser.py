"""VIB bank statement parser."""

import io
import re
from datetime import datetime
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
            xls = self.get_excel_file(file_bytes)

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
            xls = self.get_excel_file(file_bytes)

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
            xls = self.get_excel_file(file_bytes)

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

        VIB OCR has TWO formats:
        1. Simple format (BWID): TX ID on one line, amounts on next 3 lines
           7047991960 29/11/2025 29/11/2025 CRIN
           0
           1,217
           3,065,877

        2. Complex format (QV2): All TX IDs listed first, then amounts section after header
           7006560673 17/11/2025 17/11/2025 FTCR
           60125025761  (ref number)
           ...
           7038881652 27/11/2025 27/11/2025 FTDR
           ...
           [Header: Withdrawal/Deposit/Balance]
           0              (tx1 withdrawal)
           1,000,000,000  (tx1 deposit)
           2,694,600      (tx2 withdrawal)
           0              (tx2 deposit)
           ...

        CORRECT MAPPING:
        - Debit = Withdrawal column (Phát sinh nợ) = tiền RA
        - Credit = Deposit column (Phát sinh có) = tiền VÀO
        """
        try:
            # Split by account sections first
            account_sections = self._split_into_account_sections(text)
            all_transactions = []
            processed_tx_ids = set()

            for section in account_sections:
                acc_no, currency = self._extract_account_info_from_section(section)
                logger.info(f"VIB: Processing section for account {acc_no} ({currency})")

                # Count total TX IDs in section (allow optional non-digit prefix for OCR errors)
                total_tx_count = len(re.findall(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', section, re.MULTILINE))

                # Get opening balance for this section (to detect if simple format picked it up incorrectly)
                section_opening = self._extract_section_balance(section, ["số dư đầu", "opening balance"])
                logger.debug(f"VIB: Section opening balance: {section_opening}")

                # Try BOTH simple and complex format, then intelligently combine results
                # Simple format: amounts appear right after each TX line (most common, more reliable)
                simple_txs = self._parse_simple_format_transactions(section, acc_no, currency)
                simple_tx_map = {tx.transaction_id: tx for tx in simple_txs}
                logger.info(f"VIB: Simple format found {len(simple_txs)} TXs: {list(simple_tx_map.keys())}")

                # Complex format: amounts appear after header section for a group of TXs
                complex_txs = self._parse_complex_format_transactions(section, acc_no, currency)
                complex_tx_map = {tx.transaction_id: tx for tx in complex_txs}
                logger.info(f"VIB: Complex format found {len(complex_txs)} TXs: {list(complex_tx_map.keys())}")

                # Combine: Compare simple vs complex and pick the better one for each TX
                txs = []
                all_tx_ids = set(simple_tx_map.keys()) | set(complex_tx_map.keys())

                for tx_id in all_tx_ids:
                    simple_tx = simple_tx_map.get(tx_id)
                    complex_tx = complex_tx_map.get(tx_id)

                    if simple_tx and complex_tx:
                        # Both formats found this TX - pick the better one
                        # If simple format's debit or credit matches opening/closing balance, it's likely wrong
                        section_closing = self._extract_section_balance(section, ["số dư cuối", "ending balance"])

                        simple_has_balance = (
                            (section_opening and section_opening > 0 and
                             (simple_tx.debit == section_opening or simple_tx.credit == section_opening)) or
                            (section_closing and section_closing > 0 and
                             (simple_tx.debit == section_closing or simple_tx.credit == section_closing))
                        )

                        # Check if simple format has suspiciously large values compared to complex
                        # Typical TX: one side is 0/None, the other is the amount
                        # If simple has huge values (> 100x complex) and different pattern, likely wrong
                        complex_max = max(complex_tx.debit or 0, complex_tx.credit or 0)
                        simple_max = max(simple_tx.debit or 0, simple_tx.credit or 0)

                        # Complex has typical pattern: one side is zero/None
                        complex_has_zero = (complex_tx.debit is None or complex_tx.debit == 0 or
                                            complex_tx.credit is None or complex_tx.credit == 0)
                        # Simple has both sides with values (no zero/None)
                        simple_both_valued = ((simple_tx.debit is not None and simple_tx.debit > 0) and
                                              (simple_tx.credit is not None and simple_tx.credit > 0))

                        simple_suspicious = (
                            complex_max > 0 and simple_max > complex_max * 100 and complex_has_zero
                        )

                        if simple_has_balance or simple_suspicious:
                            # Simple format picked up balance or has suspicious values - use complex format
                            txs.append(complex_tx)
                            logger.info(f"VIB: TX {tx_id} - using complex format (simple has balance or suspicious values)")
                        else:
                            # Simple format looks correct
                            txs.append(simple_tx)
                            logger.debug(f"VIB: TX {tx_id} - using simple format")
                    elif simple_tx:
                        txs.append(simple_tx)
                    elif complex_tx:
                        txs.append(complex_tx)
                        logger.info(f"VIB: TX {tx_id} - using complex format (only found there)")

                # Extract ALL descriptions from balance+description lines in the section
                # These lines appear as: "1,305,591,350 QV2_VC3_Chuyen tien BCC transfer"
                all_descriptions = self._extract_all_descriptions(section)
                logger.info(f"VIB: Found {len(all_descriptions)} descriptions total")

                # Sort txs by date and TX ID to match with descriptions in order
                txs_sorted = sorted(txs, key=lambda t: (t.date or datetime.min, t.transaction_id or ""))

                # Assign descriptions to transactions (in order)
                for i, tx in enumerate(txs_sorted):
                    if i < len(all_descriptions) and not tx.description:
                        tx.description = all_descriptions[i]
                        logger.info(f"VIB: Assigned desc to TX {tx.transaction_id}: {tx.description[:30]}...")

                for tx in txs:
                    if tx.transaction_id not in processed_tx_ids:
                        all_transactions.append(tx)
                        processed_tx_ids.add(tx.transaction_id)

            # Sort transactions by date, then by transaction ID
            all_transactions.sort(key=lambda tx: (tx.date or datetime.min, tx.transaction_id or ""))

            logger.info(f"VIB: Successfully parsed {len(all_transactions)} transactions from OCR text")
            return all_transactions

        except Exception as e:
            logger.error(f"Error parsing VIB transactions from text: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _parse_complex_format_transactions(self, section: str, acc_no: str, currency: str) -> List[BankTransaction]:
        """
        Parse transactions in complex format where amounts are listed after header section.

        This format appears in QV2 where:
        1. TX IDs are listed first (with ref numbers interspersed)
        2. Then header section (Withdrawal/Deposit/Balance)
        3. Then amounts for those TXs in sequence

        IMPORTANT: In mixed format, there can be multiple TX groups:
        - Group 1: TXs 1-3, then header, then amounts for TXs 1-3
        - Group 2: TXs 4-5 (handled by simple format)

        This method finds the FIRST contiguous group of TXs before the header/amounts section.

        Returns list of transactions.
        """
        transactions = []
        lines = section.split('\n')

        # Step 1: Find the header/amounts section first
        # Look for "Withdrawal" or "Phát sinh nợ" followed by amounts
        header_line_idx = -1
        amounts_start_idx = -1

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if line_lower in ['withdrawal', 'phát sinh nợ']:
                # Look for the line after the header block
                # Skip lines that are headers (Deposit, Balance, Remarks, etc.)
                for j in range(i + 1, len(lines)):
                    check_line = lines[j].strip()
                    check_lower = check_line.lower()
                    if check_lower in ['deposit', 'phát sinh có', 'balance', 'số dư', 'remarks', 'nội dung']:
                        continue
                    # Must be a pure amount line, not a TX line
                    if check_line and re.match(r'^[\d,\.]+$', check_line):
                        header_line_idx = i
                        amounts_start_idx = j
                        break
                    # Or amount with description at end
                    if check_line and re.match(r'^[\d,]+\s+\w', check_line):
                        header_line_idx = i
                        amounts_start_idx = j
                        break
                if amounts_start_idx != -1:
                    break

        if amounts_start_idx == -1:
            logger.debug(f"VIB complex: No valid amounts section found")
            return []

        logger.info(f"VIB complex: Header at line {header_line_idx}, amounts start at line {amounts_start_idx}")

        # Step 2: Collect TX IDs that appear BEFORE the header line
        # This handles the mixed format where some TXs appear after the amounts section
        tx_info_list = []  # [(tx_id, date_str, code, line_idx), ...]

        for i, line in enumerate(lines):
            # Stop before the header section
            if i >= header_line_idx:
                break

            line = line.strip()
            # Match TX line: 10-digit ID followed by date
            # Allow optional non-digit prefix (OCR error like "n7006560673")
            tx_match = re.match(r'^[^0-9]?(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}/\d{1,2}/\d{4}\s+([A-Z]{2,6})', line)
            if tx_match:
                tx_id = tx_match.group(1)
                date_str = tx_match.group(2)
                code = tx_match.group(3)
                tx_info_list.append((tx_id, date_str, code, i))

        if not tx_info_list:
            logger.debug(f"VIB complex: No TX IDs found before header")
            return []

        logger.info(f"VIB complex: Found {len(tx_info_list)} TX IDs before header")

        # Step 3: Extract amounts AND descriptions
        # Stop when we hit another TX line (which would be in simple format section)
        amounts = []
        descriptions = []  # List of descriptions for each TX
        current_desc_parts = []  # Accumulate multi-line descriptions

        for i in range(amounts_start_idx, len(lines)):
            line = lines[i].strip()

            # Stop if we hit another TX line (this marks start of simple format section)
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                logger.info(f"VIB complex: Stopping at TX line {i}: {line[:30]}")
                # Save any pending description
                if current_desc_parts:
                    descriptions.append(' '.join(current_desc_parts))
                    current_desc_parts = []
                break

            # Stop at summary/ending sections
            if any(stop in line.lower() for stop in [
                'số dư cuối', 'ending balance', 'số dư khả dụng', 'available balance',
                'doanh số giao dịch', 'transaction summary', 'tổng số giao dịch'
            ]):
                if current_desc_parts:
                    descriptions.append(' '.join(current_desc_parts))
                break

            # Skip empty lines
            if not line:
                continue

            # Skip header lines
            if any(skip in line.lower() for skip in [
                'withdrawal', 'deposit', 'balance', 'remarks',
                'phát sinh nợ', 'phát sinh có', 'số dư', 'nội dung'
            ]):
                continue

            # Extract balance+description lines (e.g., "1,305,591,350 QV2_VC3_Chuyen")
            balance_desc_match = re.match(r'^([\d,]+)\s+([A-Za-z_].*)$', line)
            if balance_desc_match:
                # Save previous description if exists
                if current_desc_parts:
                    descriptions.append(' '.join(current_desc_parts))
                # Start new description
                current_desc_parts = [balance_desc_match.group(2).strip()]
                logger.debug(f"VIB complex: Found balance+desc line: {line[:50]}")
                continue

            # Check if this is a continuation of description (text without leading number)
            if current_desc_parts and not re.match(r'^[\d,]+$', line) and not re.match(r'^\d{11,}$', line):
                # Skip small numbers that are part of invoice numbers
                if not re.match(r'^\d{1,4}$', line):
                    current_desc_parts.append(line)
                continue

            if currency == "USD":
                # USD: look for decimal numbers
                usd_match = re.match(r'^(\d+\.\d{2})$', line)  # Must be exact match
                if usd_match:
                    amounts.append(float(usd_match.group(1)))
                    continue
                # Also try plain integer for USD (no trailing text)
                plain_match = re.match(r'^(\d+)$', line)
                if plain_match and len(plain_match.group(1)) <= 6:
                    amounts.append(float(plain_match.group(1)))
                    continue
            else:
                # Skip small numbers without comma (likely invoice numbers, part of description)
                # Valid amounts are either 0, or have commas (>= 1,000)
                if re.match(r'^[1-9]\d{0,3}$', line):  # Numbers 1-9999 without comma
                    logger.debug(f"VIB complex: Skipping small number (likely desc): {line}")
                    continue

                # VND: comma-separated or plain 0 (no trailing text except whitespace)
                vnd_match = re.match(r'^([\d,]+)$', line)
                if vnd_match:
                    num_str = vnd_match.group(1)
                    clean_num = num_str.replace(',', '')

                    # Skip if it's a reference number (11+ digits, no comma)
                    if len(clean_num) >= 11:
                        continue

                    # For VND, valid amounts are:
                    # - "0" (zero amount)
                    # - Numbers with comma (>= 1,000)
                    # - Skip small numbers without comma (might be invoice/part numbers)
                    if clean_num != '0' and ',' not in num_str and int(clean_num) < 10000:
                        logger.debug(f"VIB complex: Skipping small VND without comma: {num_str}")
                        continue

                    val = self._parse_number_from_text(num_str)
                    if val is not None:
                        amounts.append(val)
                        continue

        logger.info(f"VIB complex: Found {len(amounts)} amount values: {amounts[:20]}")

        # Step 3b: Extract descriptions from balance+description lines
        # These lines appear AFTER the simple format TXs in the format: "1,305,591,350 QV2_VC3_Chuyen..."
        # Search the ENTIRE section for balance+description patterns
        descriptions = []
        current_desc_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match balance+description: "1,305,591,350 QV2_VC3_Chuyen"
            balance_desc_match = re.match(r'^([\d,]{7,})\s+([A-Za-z_].*)$', line)
            if balance_desc_match:
                # Save previous description if exists
                if current_desc_parts:
                    descriptions.append(' '.join(current_desc_parts))
                # Start new description
                current_desc_parts = [balance_desc_match.group(2).strip()]
                continue

            # Check if this is a continuation of description (text-only line after balance+desc)
            if current_desc_parts:
                # Skip if it's an amount, reference number, or TX line
                if re.match(r'^[\d,]+$', line):
                    continue
                if re.match(r'^\d{10,}$', line):
                    continue
                if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                    continue
                # Skip header/label lines
                if any(skip in line.lower() for skip in [
                    'withdrawal', 'deposit', 'balance', 'remarks', 'phat sinh', 'so du', 'noi dung',
                    'statement', 'page', 'trang', 'in ngay', 'tu ngay'
                ]):
                    continue
                # This is likely description continuation
                current_desc_parts.append(line)

        # Save last description
        if current_desc_parts:
            descriptions.append(' '.join(current_desc_parts))

        logger.info(f"VIB complex: Found {len(descriptions)} descriptions")

        # Step 4: Match amounts to transactions
        # Each TX should have 2 amounts: withdrawal, deposit
        tx_count = len(tx_info_list)
        expected_amounts = tx_count * 2  # withdrawal + deposit per TX

        # Determine amounts per TX based on what we collected
        if len(amounts) == tx_count * 3:
            amounts_per_tx = 3
            logger.info(f"VIB complex: Detected format with balance (3 per TX)")
        elif len(amounts) == tx_count * 2:
            amounts_per_tx = 2
            logger.info(f"VIB complex: Detected format without balance (2 per TX)")
        elif len(amounts) > tx_count * 2:
            amounts = amounts[:expected_amounts]
            amounts_per_tx = 2
            logger.info(f"VIB complex: Truncated amounts to {expected_amounts} (extra balances removed)")
        else:
            amounts_per_tx = len(amounts) // tx_count if tx_count > 0 else 0
            logger.info(f"VIB complex: {len(amounts)} amounts / {tx_count} TXs = {amounts_per_tx} per TX")

        amount_idx = 0
        for tx_idx, (tx_id, date_str, code, _) in enumerate(tx_info_list):
            if amount_idx + 1 >= len(amounts):
                logger.debug(f"VIB complex: Not enough amounts for TX {tx_id}")
                break

            withdrawal = amounts[amount_idx]
            deposit = amounts[amount_idx + 1]

            # If we have 3 amounts per TX, skip the balance
            if amounts_per_tx >= 3:
                amount_idx += 3
            else:
                amount_idx += 2

            tx_date = self._parse_date_from_text(date_str)

            # Get description for this transaction (if available)
            description = descriptions[tx_idx] if tx_idx < len(descriptions) else ""

            # CUSTOMER PERSPECTIVE (Bank Statement):
            # - Phát sinh nợ (Withdrawal) = tiền RA = DEBIT (giảm số dư)
            # - Phát sinh có (Deposit) = tiền VÀO = CREDIT (tăng số dư)
            tx = BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=withdrawal if withdrawal > 0 else None,   # Withdrawal → Debit (tiền ra)
                credit=deposit if deposit > 0 else None,        # Deposit → Credit (tiền vào)
                date=tx_date,
                description=description,
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )
            transactions.append(tx)
            logger.info(f"VIB complex: TX {tx_id} - debit(withdrawal)={withdrawal}, credit(deposit)={deposit}, desc={description[:30] if description else ''}")

        return transactions

    def _parse_simple_format_transactions(self, section: str, acc_no: str, currency: str) -> List[BankTransaction]:
        """
        Parse transactions in simple format where amounts follow each TX ID.

        This format appears in BWID/QV2 where:
        7047991960 29/11/2025 29/11/2025 CRIN
        0           <- withdrawal
        1,217       <- deposit
        3,065,877   <- balance (or balance + description)

        Also handles cases where amounts appear BEFORE the TX line (VIB.pdf edge case):
        0
        1,158,175
        777,510,925
        7047991948 29/11/2025 29/11/2025 CRIN

        Returns list of transactions.
        """
        transactions = []
        lines = section.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for TX line: 10-digit ID followed by date
            # Allow optional non-digit prefix (OCR error like "n7006560673")
            tx_match = re.match(r'^[^0-9]?(\d{10})\s+(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}/\d{1,2}/\d{4}\s+([A-Z]{2,6})', line)
            if not tx_match:
                i += 1
                continue

            tx_id = tx_match.group(1)
            date_str = tx_match.group(2)
            code = tx_match.group(3)

            # Try to find amounts AFTER the TX line (normal case)
            amounts_after = self._extract_amounts_after_line(lines, i, currency)

            # Also try looking BEFORE the TX line (for edge cases like VIB.pdf TX 7047991948)
            amounts_before = self._extract_amounts_before_line(lines, i, currency)

            # Decide which amounts to use:
            # 1. If amounts_after looks suspicious (first amount is very large, e.g., balance value),
            #    but amounts_before starts with 0 (typical withdrawal=0 pattern), prefer amounts_before
            # 2. Otherwise prefer amounts_after if it has enough values
            use_before = False
            if len(amounts_before) >= 2 and len(amounts_after) >= 2:
                # Both have enough amounts - check which looks more like TX amounts
                # TX amounts typically have 0 as first value (withdrawal=0 for deposits)
                # Balance values are typically large
                after_suspicious = amounts_after[0] > 1000000 and amounts_after[1] > 1000000
                before_has_zero = amounts_before[0] == 0
                if after_suspicious and before_has_zero:
                    use_before = True
                    logger.debug(f"VIB simple: TX {tx_id} - amounts_after looks suspicious, using amounts_before")

            # Use amounts from whichever direction is better
            if use_before or (len(amounts_before) >= 2 and len(amounts_after) < 2):
                amounts = amounts_before
                description = ""  # Description typically after TX, not before
                logger.debug(f"VIB simple: TX {tx_id} - using amounts_before: {amounts}")
            elif len(amounts_after) >= 2:
                amounts = amounts_after
                description = self._extract_description_after_tx(lines, i)
            else:
                logger.debug(f"VIB simple: TX {tx_id} - not enough amounts (after={len(amounts_after)}, before={len(amounts_before)})")
                i += 1
                continue

            withdrawal = amounts[0]  # Phát sinh nợ (tiền ra)
            deposit = amounts[1]     # Phát sinh có (tiền vào)
            tx_date = self._parse_date_from_text(date_str)

            # CUSTOMER PERSPECTIVE (Bank Statement):
            # - Phát sinh nợ (Withdrawal) = tiền RA = DEBIT (giảm số dư)
            # - Phát sinh có (Deposit) = tiền VÀO = CREDIT (tăng số dư)
            tx = BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=withdrawal if withdrawal > 0 else None,   # Withdrawal → Debit (tiền ra)
                credit=deposit if deposit > 0 else None,        # Deposit → Credit (tiền vào)
                date=tx_date,
                description=description,
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )
            transactions.append(tx)
            logger.info(f"VIB simple: TX {tx_id} - withdrawal={withdrawal}, deposit={deposit} -> DEBIT={tx.debit}, CREDIT={tx.credit}")

            i += 1

        return transactions

    def _extract_amounts_after_line(self, lines: List[str], tx_line_idx: int, currency: str) -> List[float]:
        """Extract amounts from lines AFTER the TX line.

        Note: In some VIB OCR formats:
        1. Amounts appear AFTER summary labels like 'Số dư cuối ngày'
        2. Opening balance may appear BETWEEN TX line and actual amounts
        3. Multiple TX lines may be grouped, then amounts come as a group (BWID format)

        For case 3, we need to skip past all consecutive TX lines first, then look for amounts.
        """
        amounts = []
        prev_was_date = False  # Track if previous line was a date

        # BWID format detection: TX lines may be grouped together, then amounts come as a group
        # We need to:
        # 1. Find all TXs in our group (looking both backward and forward)
        # 2. Calculate our position in the group
        # 3. Skip to the correct amount position

        # First, find the start of the TX group (look backward)
        group_start_idx = tx_line_idx
        for j in range(tx_line_idx - 1, max(-1, tx_line_idx - 20), -1):
            line = lines[j].strip()
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                group_start_idx = j
            else:
                # Stop if we hit a non-TX line (could be header, amount, etc.)
                break

        # Then find the end of the TX group (look forward from start)
        group_end_idx = group_start_idx
        for j in range(group_start_idx, min(group_start_idx + 20, len(lines))):
            line = lines[j].strip()
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                group_end_idx = j
            else:
                break

        # Calculate group size and our position
        total_txs_in_group = group_end_idx - group_start_idx + 1
        tx_position_in_group = tx_line_idx - group_start_idx  # 0-indexed position

        # Start looking for amounts after the entire TX group
        start_idx = group_end_idx + 1

        if total_txs_in_group > 1:
            logger.debug(f"VIB BWID format: TX at position {tx_position_in_group} in group of {total_txs_in_group}, amounts start at line {start_idx}")

        # For BWID format, we need to skip amounts for previous TXs in the group
        # Each TX has 3 amounts (withdrawal, deposit, balance)
        amounts_to_skip = tx_position_in_group * 3
        all_amounts = []  # Collect all amounts, then slice for our TX

        for j in range(start_idx, min(start_idx + 50, len(lines))):
            next_line = lines[j].strip()

            # Stop if we hit another TX line (not part of our group)
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', next_line):
                break

            # Stop at next account section (VIB marker)
            if next_line == 'VIB' or re.match(r'^\*\s*VIB|VIB\s*\*', next_line):
                break

            # Skip (don't stop at) summary labels - amounts may come after them
            if any(skip in next_line.lower() for skip in [
                'số dư cuối', 'ending balance', 'số dư khả dụng', 'available balance',
                'doanh số', 'transaction summary', 'tổng số', 'number of transactions',
                'số dư đầu', 'opening balance'
            ]):
                prev_was_date = False
                continue

            # Track if this is a date line (month name pattern)
            if any(month in next_line.lower() for month in [
                'november', 'december', 'january', 'february', 'march', 'april',
                'may', 'june', 'july', 'august', 'september', 'october'
            ]):
                prev_was_date = True
                continue

            # Skip headers, empty lines
            if not next_line:
                continue
            if any(skip in next_line.lower() for skip in [
                'withdrawal', 'deposit', 'balance', 'phát sinh nợ', 'phát sinh có',
                'remarks', 'nội dung', 'seq.', 'số ct', 'số dư', 'số tk', 'a/c no', 'type/ccy'
            ]):
                prev_was_date = False
                continue

            # Skip reference numbers (11+ digits)
            if re.match(r'^\d{11,}$', next_line):
                prev_was_date = False
                continue

            # Skip small standalone numbers (1-99 except "0")
            if re.match(r'^\d{1,2}$', next_line) and next_line not in ['0']:
                prev_was_date = False
                continue

            # Extract amount - try single first, then multiple
            val = self._extract_single_amount(next_line, currency)
            if val is not None:
                # Detect opening balance pattern (only for first TX in group):
                # - Previous line was a date ("16 November 2025")
                # - This is a large number (> 1000)
                # - We haven't collected any amounts yet
                if prev_was_date and val > 1000 and len(all_amounts) == 0 and tx_position_in_group == 0:
                    logger.debug(f"VIB: Skipping potential opening balance {val} (after date line)")
                    prev_was_date = False
                    continue

                all_amounts.append(val)
                prev_was_date = False

                # For BWID format, collect enough amounts for our TX
                # For normal format (single TX), stop after 3 amounts
                if total_txs_in_group <= 1:
                    if len(all_amounts) >= 3:
                        break
                elif len(all_amounts) >= amounts_to_skip + 3:
                    break
            else:
                # Try extracting multiple amounts from single line (e.g., "0 33,000,000,000")
                multi_vals = self._extract_multiple_amounts(next_line, currency)
                if multi_vals:
                    all_amounts.extend(multi_vals)
                    logger.debug(f"VIB: Extracted multiple amounts from line: {multi_vals}")
                    # Check if we have enough amounts
                    if total_txs_in_group <= 1:
                        if len(all_amounts) >= 3:
                            break
                    elif len(all_amounts) >= amounts_to_skip + 3:
                        break
                prev_was_date = False

        # Extract our TX's amounts from the collected amounts
        if amounts_to_skip > 0:
            amounts = all_amounts[amounts_to_skip:amounts_to_skip + 3]
            logger.debug(f"VIB BWID: Extracted amounts {amounts} (skipped first {amounts_to_skip})")
        else:
            amounts = all_amounts[:3]

        return amounts

    def _extract_amounts_before_line(self, lines: List[str], tx_line_idx: int, currency: str) -> List[float]:
        """Extract amounts from lines BEFORE the TX line (fallback for edge cases)."""
        amounts = []

        # Look back up to 10 lines, but stop at previous TX or section boundaries
        for j in range(tx_line_idx - 1, max(-1, tx_line_idx - 10), -1):
            prev_line = lines[j].strip()

            # Stop if we hit another TX line
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', prev_line):
                break

            # Stop at section boundaries
            if any(stop in prev_line.lower() for stop in [
                'số dư cuối', 'ending balance', 'doanh số', 'transaction summary',
                'remarks', 'nội dung'
            ]):
                break

            # Skip empty lines and headers
            if not prev_line:
                continue
            if any(skip in prev_line.lower() for skip in [
                'withdrawal', 'deposit', 'balance', 'phát sinh',
                'november', 'december', 'january', 'february'
            ]):
                continue

            # Extract amount (pure number line only, no description text)
            if currency == "USD":
                usd_match = re.match(r'^(\d+\.\d{2})\s*$', prev_line)
                if usd_match:
                    amounts.insert(0, float(usd_match.group(1)))
            else:
                vnd_match = re.match(r'^([\d,]+)\s*$', prev_line)
                if vnd_match:
                    num_str = vnd_match.group(1)
                    clean = num_str.replace(',', '')
                    # Skip reference numbers and small numbers
                    if len(clean) >= 11 or (len(clean) <= 2 and num_str != '0'):
                        continue
                    val = self._parse_number_from_text(num_str)
                    if val is not None:
                        amounts.insert(0, val)

            if len(amounts) >= 3:
                break

        return amounts

    def _extract_single_amount(self, line: str, currency: str) -> Optional[float]:
        """Extract a single amount from a line."""
        if currency == "USD":
            usd_match = re.match(r'^(\d+\.\d{2})\s*$', line)
            if usd_match:
                return float(usd_match.group(1))
        else:
            # Match pure number line OR balance+description line
            # For balance+description, we still want to extract the balance
            vnd_match = re.match(r'^([\d,]+)(?:\s*$|\s+[A-Za-z_])', line)
            if vnd_match:
                num_str = vnd_match.group(1)
                clean = num_str.replace(',', '')
                # Skip reference numbers
                if len(clean) >= 11:
                    return None
                return self._parse_number_from_text(num_str)
        return None

    def _extract_multiple_amounts(self, line: str, currency: str) -> List[float]:
        """Extract multiple amounts from a single line (e.g., '0 33,000,000,000')."""
        amounts = []

        # Skip lines that contain date patterns (dd/mm/yyyy) - years can be misinterpreted as amounts
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}', line):
            return amounts

        if currency == "USD":
            # USD: find all decimal numbers
            matches = re.findall(r'(\d+\.\d{2})', line)
            for m in matches:
                amounts.append(float(m))
        else:
            # VND: find all comma-separated numbers or plain numbers
            # Pattern: standalone 0 OR comma-separated numbers
            matches = re.findall(r'\b(0|[\d,]{4,})\b', line)
            for m in matches:
                clean = m.replace(',', '')
                # Skip reference numbers (11+ digits WITHOUT commas)
                # Numbers with commas are monetary amounts, not reference numbers
                if len(clean) >= 11 and ',' not in m:
                    continue
                # Skip years (2020-2030) - these appear in date strings
                if re.match(r'^20[2-3]\d$', clean):
                    continue
                val = self._parse_number_from_text(m)
                if val is not None:
                    amounts.append(val)
        return amounts

    def _extract_description_after_tx(self, lines: List[str], tx_line_idx: int) -> str:
        """Extract description from balance+description line after TX."""
        for j in range(tx_line_idx + 1, min(tx_line_idx + 15, len(lines))):
            line = lines[j].strip()

            # Stop at next TX
            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                break

            # Look for balance+description pattern
            balance_desc = re.match(r'^[\d,]{5,}\s+([A-Za-z_].*)$', line)
            if balance_desc:
                desc_parts = [balance_desc.group(1).strip()]

                # Collect continuation lines
                for k in range(j + 1, min(j + 15, len(lines))):
                    cont_line = lines[k].strip()
                    if not cont_line:
                        continue

                    # Stop at next TX line
                    if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', cont_line):
                        break

                    # Stop at next balance+description line (new TX's description)
                    if re.match(r'^[\d,]{5,}\s+[A-Za-z_]', cont_line):
                        break

                    # Skip small numbers (1-2 digits) - likely part of reference numbers
                    if re.match(r'^\d{1,2}$', cont_line):
                        continue

                    # Stop at large pure numbers (amounts/balances)
                    if re.match(r'^[\d,]{3,}$', cont_line):
                        break

                    # Skip reference numbers (11+ digits)
                    if re.match(r'^\d{11,}$', cont_line):
                        continue

                    # Stop at headers
                    if any(skip in cont_line.lower() for skip in [
                        'withdrawal', 'deposit', 'balance', 'số dư', 'phát sinh',
                        'ending balance', 'available balance', 'doanh số'
                    ]):
                        break

                    # This is description continuation text
                    desc_parts.append(cont_line)

                return ' '.join(desc_parts)

        return ""

    def _extract_all_descriptions(self, section: str) -> List[str]:
        """
        Extract all descriptions from balance+description lines.

        These lines appear as: "1,305,591,350 QV2_VC3_Chuyen tien BCC transfer"
        where the number is the running balance and text is the description.

        Returns list of descriptions in order of appearance.
        """
        descriptions = []
        current_desc_parts = []
        lines = section.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Match balance+description: "1,305,591,350 QV2_VC3_Chuyen"
            # Balance must be at least 7 digits to avoid matching small numbers
            balance_desc_match = re.match(r'^([\d,]{7,})\s+([A-Za-z_].*)$', line)
            if balance_desc_match:
                # Save previous description if exists
                if current_desc_parts:
                    descriptions.append(' '.join(current_desc_parts))
                # Start new description
                current_desc_parts = [balance_desc_match.group(2).strip()]
                continue

            # Check if this is a continuation of description (text-only line after balance+desc)
            if current_desc_parts:
                # Skip TX lines - they start a new context
                if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', line):
                    continue

                # Skip small numbers (1-2 digits) - likely part of reference numbers like "21", "66"
                if re.match(r'^\d{1,2}$', line):
                    continue

                # Skip large pure numbers (amounts/balances) - but don't stop collecting
                if re.match(r'^[\d,]{3,}$', line):
                    continue

                # Skip reference numbers (11+ digits)
                if re.match(r'^\d{11,}$', line):
                    continue

                # Skip header/label lines
                if any(skip in line.lower() for skip in [
                    'withdrawal', 'deposit', 'balance', 'remarks', 'phat sinh', 'phát sinh',
                    'so du', 'số dư', 'noi dung', 'nội dung', 'statement', 'page', 'trang',
                    'in ngay', 'tu ngay', 'so tk', 'a/c no', 'ending', 'available', 'doanh số'
                ]):
                    continue

                # This is likely description continuation text
                current_desc_parts.append(line)

        # Save last description
        if current_desc_parts:
            descriptions.append(' '.join(current_desc_parts))

        return descriptions

    def _parse_vib_transaction_line(self, line: str, lines: List[str], line_idx: int, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Parse VIB transaction from a single line.

        VIB format:
        7047991960 29/11/2025  29/11/2025  CRIN                    0        1,217   3,065,877
        7048182367 29/11/2025  29/11/2025  SC60                   50,000        0   3,015,877
        7048180567 29/11/2025  29/11/2025  SC60                   2.00      0.00     54.00  (USD)

        Columns: TxID | TranDate | EffectDate | TranCode | Withdrawal | Deposit | Balance | [Remarks]

        Numbers may have comma separators (VND) or decimal points (USD).
        """
        try:
            # Extract transaction ID (first 10 digits)
            tx_id_match = re.match(r'^(\d{10})', line)
            if not tx_id_match:
                return None
            tx_id = tx_id_match.group(1)

            # Extract dates (DD/MM/YYYY format)
            dates = re.findall(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            date_str = dates[0] if dates else None

            # Remove tx_id and dates from line to get remaining content
            remaining = line
            remaining = re.sub(r'^\d{10}\s*', '', remaining)  # Remove tx_id
            remaining = re.sub(r'\d{1,2}/\d{1,2}/\d{4}\s*', '', remaining)  # Remove dates

            # Extract transaction code (2-4 uppercase letters/digits like CRIN, SC60, VATX, FTCR, FTDR)
            code_match = re.match(r'^([A-Z][A-Z0-9]{1,5})\s*', remaining)
            if code_match:
                remaining = remaining[len(code_match.group(0)):]

            # Now remaining should contain: Withdrawal Deposit Balance [Remarks]
            # Extract all numbers from remaining text
            # For VND: numbers with comma separators like "50,000" or plain "0"
            # For USD: numbers with decimal points like "2.00" or plain "0"

            if currency == "USD":
                # USD format: look for decimal numbers like "2.00", "0.00", "54.00"
                # Or plain integers like "0", "2"
                numbers = re.findall(r'(\d+\.\d{2}|\d+)', remaining)
            else:
                # VND format: look for comma-separated numbers and plain integers
                # Pattern matches: "50,000" or "1,217" or "3,065,877" or "0"
                numbers = re.findall(r'(\d{1,3}(?:,\d{3})+|\d+)', remaining)

            logger.debug(f"VIB tx {tx_id}: currency={currency}, remaining='{remaining[:60]}', numbers={numbers[:5]}")

            if len(numbers) < 2:
                # Not enough numbers - try multiline parsing
                return self._parse_multiline_transaction_v2(tx_id, date_str, lines, line_idx, acc_no, currency)

            # Parse withdrawal and deposit
            # First number = Withdrawal (Phát sinh nợ)
            # Second number = Deposit (Phát sinh có)
            # Third number = Balance (skip)

            withdrawal = self._parse_number_from_text(numbers[0]) or 0
            deposit = self._parse_number_from_text(numbers[1]) or 0

            # Skip if both are zero
            if withdrawal == 0 and deposit == 0:
                return None

            # Parse date
            tx_date = self._parse_date_from_text(date_str)

            # Extract description/remarks (text after the numbers)
            description = ""
            # Find where numbers end and look for text after
            last_num_pos = 0
            for num in numbers[:3]:  # Check first 3 numbers (withdrawal, deposit, balance)
                pos = remaining.find(num)
                if pos != -1:
                    last_num_pos = max(last_num_pos, pos + len(num))
            if last_num_pos < len(remaining):
                description = remaining[last_num_pos:].strip()

            logger.info(f"VIB: Parsed tx {tx_id} - withdrawal={withdrawal}, deposit={deposit}, currency={currency}")

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=withdrawal if withdrawal > 0 else None,
                credit=deposit if deposit > 0 else None,
                date=tx_date,
                description=description,
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            logger.error(f"Error parsing VIB transaction line '{line[:50]}': {e}")
            return None

    def _parse_multiline_transaction_v2(self, tx_id: str, date_str: str, lines: List[str], current_idx: int, acc_no: str, currency: str) -> Optional[BankTransaction]:
        """
        Parse transaction when amounts are on subsequent lines (multiline OCR output).

        Gemini OCR format example:
        7006560673 17/11/2025 17/11/2025 FTCR
        60125025761
        21
        ...
        0                          <- Withdrawal
        1,000,000,000              <- Deposit
        1,305,591,350 QV2_VC3...   <- Balance + Description

        We need to find the Withdrawal and Deposit values which appear before Balance.
        """
        try:
            amounts = []
            description = ""

            # Look at subsequent lines for amounts
            for i in range(current_idx + 1, min(current_idx + 15, len(lines))):
                next_line = lines[i].strip()

                # Stop if we hit another transaction line (10-digit number at start, allow optional prefix)
                if re.match(r'^[^0-9]?\d{10}\s', next_line):
                    break

                # Stop if we hit header/summary lines
                if any(skip in next_line.lower() for skip in [
                    'seq.', 'số ct', 'withdrawal', 'deposit', 'số dư đầu', 'số dư cuối',
                    'opening balance', 'ending balance', 'available balance',
                    'transaction summary', 'doanh số', 'tổng số giao dịch', 'number of transactions'
                ]):
                    break

                # Skip date/month lines
                if any(month in next_line.lower() for month in [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december'
                ]):
                    continue

                # Skip reference number lines (long digit sequences that are not amounts)
                # Reference numbers like "60125025761", "88225693987" are 11 digits
                if re.match(r'^\d{11,}$', next_line):
                    continue

                # Skip small standalone numbers that are likely part of reference (like "21", "66", "17")
                if re.match(r'^\d{1,2}$', next_line):
                    continue

                # Extract numbers based on currency
                if currency == "USD":
                    # USD format: decimal numbers like "2.00", "0.00"
                    line_numbers = re.findall(r'(\d+\.\d{2})', next_line)
                    if not line_numbers:
                        # Also try plain integers
                        if re.match(r'^(\d+)$', next_line):
                            line_numbers = [next_line]
                else:
                    # VND format: comma-separated or plain numbers
                    # Check if line is primarily a number (with optional description after)
                    vnd_match = re.match(r'^([\d,]+)\s*(.*)?$', next_line)
                    if vnd_match:
                        num_part = vnd_match.group(1)
                        desc_part = vnd_match.group(2) or ""

                        # Skip years
                        if re.match(r'^20[2-3]\d$', num_part.replace(',', '')):
                            continue

                        val = self._parse_number_from_text(num_part)
                        if val is not None:
                            amounts.append(val)
                            # If there's description text after the number, save it
                            if desc_part and len(amounts) == 3:
                                description = desc_part.strip()

                        if len(amounts) >= 3:
                            break
                        continue

                    line_numbers = re.findall(r'(\d{1,3}(?:,\d{3})+|\d+)', next_line)

                for num_str in line_numbers:
                    # Skip years
                    if re.match(r'^20[2-3]\d$', num_str.replace(',', '')):
                        continue
                    # Skip reference numbers (11+ digits without comma)
                    if len(num_str.replace(',', '')) >= 11:
                        continue
                    val = self._parse_number_from_text(num_str)
                    if val is not None:
                        amounts.append(val)

                if len(amounts) >= 3:
                    break

            if len(amounts) < 2:
                logger.debug(f"VIB multiline: tx {tx_id} - not enough amounts found: {amounts}")
                return None

            # First two amounts are Withdrawal and Deposit
            withdrawal = amounts[0]
            deposit = amounts[1]

            if withdrawal == 0 and deposit == 0:
                return None

            tx_date = self._parse_date_from_text(date_str)

            logger.info(f"VIB multiline: Parsed tx {tx_id} - withdrawal={withdrawal}, deposit={deposit}")

            return BankTransaction(
                bank_name="VIB",
                acc_no=acc_no or "",
                debit=withdrawal if withdrawal > 0 else None,
                credit=deposit if deposit > 0 else None,
                date=tx_date,
                description=description,
                currency=currency,
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

        except Exception as e:
            logger.error(f"Error in multiline v2 parsing: {e}")
            return None

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

                # Extract opening and closing balance
                # IMPORTANT: In some OCR formats, opening balance appears BEFORE the
                # account number line. So we search in both section AND full text.

                # For opening balance - try section first, then full text
                opening = self._extract_section_balance(section, ["số dư đầu", "opening balance"])
                if opening is None or opening == 0:
                    opening = self._find_balance_near_account(text, acc_no, currency, ["số dư đầu", "opening balance"], is_opening=True)
                if opening is None or opening == 0:
                    opening = self._extract_balance_fallback_section(section, ["số dư đầu", "opening balance"], currency)

                # For closing balance - try section first (more reliable for VIB format)
                # _find_balance_near_account may pick up transaction balances instead of closing
                closing = self._extract_section_balance(section, ["số dư cuối", "ending balance"])
                if closing is None or closing == 0:
                    closing = self._find_balance_near_account(text, acc_no, currency, ["số dư cuối", "ending balance"], is_opening=False)
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
            # For closing balance, we want the LAST valid candidate (actual balance comes after totals)
            is_closing = any(marker in label_lower for marker in ['cuoi', 'cuối', 'ending'])

            vnd_matches = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', window)
            vnd_candidates = []
            for num_str in vnd_matches:
                # Skip transaction IDs (exactly 10 digits)
                digits = num_str.replace(',', '')
                if len(digits) == 10:
                    continue
                val = self._parse_number_from_text(num_str)
                if val is not None and val >= 1000:
                    vnd_candidates.append(val)

            if vnd_candidates:
                if is_closing:
                    val = vnd_candidates[-1]  # Last candidate for closing balance
                else:
                    val = vnd_candidates[0]   # First candidate for opening balance
                logger.info(f"VIB fallback (VND): found {val} for '{label}' from candidates {vnd_candidates}")
                return val

        return None

    def _find_balance_near_account(self, text: str, acc_no: str, currency: str, labels: List[str], is_opening: bool) -> Optional[float]:
        """
        Find balance in full text by looking near the account number.

        In some OCR formats, opening balance appears BEFORE the account number line
        OR the balance number appears BEFORE the label (e.g., "187,462,935\nSố dư đầu ngày").

        This method searches in a window around the account number position.
        """
        lines = text.split('\n')

        # Find line index where account number appears
        acc_line_idx = -1
        for i, line in enumerate(lines):
            if acc_no in line and (currency.lower() in line.lower()):
                acc_line_idx = i
                break

        if acc_line_idx == -1:
            return None

        # For both opening and closing, search BEFORE and AFTER account line
        # In some OCR formats, labels appear before account line but balance after
        if is_opening:
            search_start = max(0, acc_line_idx - 30)
            search_end = min(len(lines), acc_line_idx + 35)
        else:
            search_start = max(0, acc_line_idx - 30)
            search_end = min(len(lines), acc_line_idx + 70)

        # Look for label and find balance near it
        for label in labels:
            label_lower = label.lower()

            # Find the label in search window
            for i in range(search_start, search_end):
                if label_lower in lines[i].lower():
                    # Found label - look for balance both BEFORE and AFTER this line
                    # (OCR may put balance before or after the label)

                    balance_candidates = []

                    # Look in lines BEFORE the label (up to 5 lines)
                    for j in range(max(search_start, i - 5), i):
                        line = lines[j].strip()
                        bal = self._extract_balance_from_line(line, currency)
                        if bal is not None and bal > 0:
                            balance_candidates.append((abs(i - j), bal))

                    # Look in lines AFTER the label (up to 20 lines)
                    # Balance can be quite far from label in some OCR formats
                    for j in range(i + 1, min(search_end, i + 20)):
                        line = lines[j].strip()
                        bal = self._extract_balance_from_line(line, currency)
                        if bal is not None and bal > 0:
                            balance_candidates.append((abs(i - j), bal))

                    # Return the best balance
                    if balance_candidates:
                        if is_opening:
                            # For opening balance, return closest non-zero
                            balance_candidates.sort(key=lambda x: x[0])
                            val = balance_candidates[0][1]
                        else:
                            # For closing balance, there may be multiple numbers nearby:
                            # - Transaction amounts (smaller)
                            # - Balance value (usually largest)
                            # Return the LARGEST value as it's most likely the actual balance
                            val = max(c[1] for c in balance_candidates)
                        logger.info(f"VIB near-account: found {val} for '{label}' acc={acc_no}")
                        return val

        return None

    def _extract_balance_from_line(self, line: str, currency: str) -> Optional[float]:
        """Extract balance value from a single line."""
        line = line.strip()
        if not line:
            return None

        # USD format: "56.00", "100.00"
        if currency == "USD":
            usd_match = re.match(r'^(\d+\.\d{2})$', line)
            if usd_match:
                return float(usd_match.group(1))
            return None

        # VND format: "187,462,935" (comma-separated)
        # Must be standalone number line
        vnd_match = re.match(r'^([\d,]+)$', line)
        if vnd_match:
            num_str = vnd_match.group(1)
            digits = num_str.replace(',', '')

            # Skip transaction IDs (exactly 10 digits)
            if len(digits) == 10:
                return None

            # Skip reference numbers (11+ digits without comma)
            if len(digits) >= 11:
                return None

            # Skip small numbers (likely not balances)
            if len(digits) < 4 and ',' not in num_str:
                return None

            val = self._parse_number_from_text(num_str)
            return val

        return None

    def _split_into_account_sections(self, text: str) -> List[str]:
        """
        Split OCR text into sections, one per account.

        Account sections are identified by:
        - "SỐ TK" or "Số TK" or "A/C No" patterns with account numbers
        - Or "Statement of Account For" patterns

        VIB PDF format typically has:
        - Account header: "Số TK/Loại TK/Loại tiền: 053376900 651/VND"
        - Each page may repeat the header, so we need to detect unique accounts

        IMPORTANT: In some OCR formats (like QV2), TX IDs appear BEFORE the
        account number line. We need to capture the entire section.
        """
        sections = []

        # Pattern to match account header lines
        # Matches: "SỐ TK/Loại TK/Loại tiền: 053376900 651/VND" or similar
        account_pattern = re.compile(
            r'(số\s*tk|so\s*tk|a/c\s*no|account\s*no|loại\s*tiền)',
            re.IGNORECASE
        )

        lines = text.split('\n')
        section_starts = []
        seen_accounts = set()  # Track unique (acc_no, currency) pairs

        for i, line in enumerate(lines):
            # Check if this line contains account number pattern
            if account_pattern.search(line):
                # Extract account number and currency from line
                # Pattern: "053376900 651/VND" or "004368306 651/VND"
                acc_match = re.search(r'(\d{6,12})\s+\d*/?\s*(VND|USD)', line, re.IGNORECASE)
                if acc_match:
                    acc_no = acc_match.group(1)
                    currency = acc_match.group(2).upper()
                    account_key = (acc_no, currency)

                    # Only add if this is a new account (not duplicate from multi-page)
                    if account_key not in seen_accounts:
                        seen_accounts.add(account_key)

                        # Look backward for TX lines that might belong to this section
                        # TX lines can appear before account number in some OCR formats
                        section_start = i
                        # Use -1 as stop value to include line 0
                        for j in range(i - 1, max(-1, i - 101), -1):
                            check_line = lines[j].strip()
                            # If we find a TX line, extend section start backward (allow optional prefix)
                            if re.match(r'^[^0-9]?\d{10}\s+\d{1,2}/\d{1,2}/\d{4}', check_line):
                                section_start = j
                            # Stop at previous account header or VIB marker
                            elif 'số tk' in check_line.lower() or check_line.strip() == 'VIB':
                                break

                        section_starts.append(section_start)
                        logger.info(f"VIB: Found new account section at line {section_start} (header at {i}): {acc_no}/{currency}")

        # If no account headers found, return entire text as one section
        if not section_starts:
            logger.info("VIB: No account sections found, treating entire text as one section")
            return [text]

        # Sort section starts and remove duplicates
        section_starts = sorted(set(section_starts))
        logger.info(f"VIB: Found {len(section_starts)} unique account sections")

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

        VIB PDF format has balance at end of the line containing the label:
        "Số dư đầu ngày :        16 November 2025                                3,064,660"

        Or balance may be on a separate line after the label line.

        IMPORTANT: Must avoid picking up:
        - Transaction IDs (10-digit numbers like 7048070432)
        - Credit/Debit amounts from transaction lines
        - Date components (day numbers like 16, 30)
        """
        section_lower = section.lower()
        lines = section.split('\n')
        is_usd = 'usd' in section_lower

        for label in labels:
            label_lower = label.lower()

            # Find lines containing the label
            for i, line in enumerate(lines):
                if label_lower not in line.lower():
                    continue

                # Found the label line - VIB format puts balance at end of this line
                # Example: "Số dư đầu ngày :        16 November 2025                                3,064,660"

                # For USD accounts - look for decimal numbers like "56.00", "100.00", "0.00"
                if is_usd:
                    # Find all decimal numbers in the line (USD format)
                    usd_matches = re.findall(r'(\d+\.\d{2})\b', line)
                    if usd_matches:
                        # Take the last decimal number (balance is at the end)
                        val = float(usd_matches[-1])
                        logger.info(f"VIB section balance (USD): found {val} for '{label}' on line: {line[:80]}")
                        return val

                    # Also check next lines for USD value - need to search further for some OCR formats
                    # where closing balance appears after summary headers
                    usd_candidates = []
                    for j in range(i + 1, min(i + 25, len(lines))):
                        next_line = lines[j].strip()
                        # Skip empty lines and labels
                        if not next_line or any(skip in next_line.lower() for skip in ['opening', 'seq']):
                            continue
                        # Stop at next VIB section
                        if next_line == 'VIB' or next_line.startswith('SO TK'):
                            break
                        usd_match = re.search(r'^(\d+\.\d{2})$', next_line)
                        if usd_match:
                            val = float(usd_match.group(1))
                            usd_candidates.append(val)

                    # For closing balance, if we have multiple candidates, take the LAST one
                    # (closest to the end of the section)
                    if usd_candidates:
                        # Check for closing balance labels (both with and without Vietnamese diacritics)
                        is_closing = any(marker in label.lower() for marker in ['cuoi', 'cuối', 'ending'])
                        if is_closing:
                            val = usd_candidates[-1]  # Last value for closing
                        else:
                            val = usd_candidates[0]   # First value for opening
                        logger.info(f"VIB section balance (USD next line): found {val} for '{label}' from candidates {usd_candidates}")
                        return val

                # For VND - look for comma-separated numbers at end of line
                # Pattern: number with at least one comma, representing >= 1,000
                # Or plain numbers like "3064660"

                # First try comma-separated (most common VIB format)
                vnd_matches = re.findall(r'(\d{1,3}(?:,\d{3})+)', line)
                if vnd_matches:
                    # Take the last number (balance is at the end of line)
                    num_str = vnd_matches[-1]
                    digits_only = num_str.replace(',', '')
                    # Skip if it looks like a transaction ID (exactly 10 digits)
                    if len(digits_only) != 10:
                        val = self._parse_number_from_text(num_str)
                        if val is not None and val >= 0:
                            logger.info(f"VIB section balance (VND comma): found {val} for '{label}' on line: {line[:80]}")
                            return val

                # Try plain numbers without comma (smaller values or different format)
                # Look for numbers at end of line that are not dates/years
                plain_matches = re.findall(r'\b(\d{4,})\b', line)
                for num_str in reversed(plain_matches):
                    # Skip years (2020-2030) and dates
                    if re.match(r'^20[2-3]\d$', num_str):
                        continue
                    # Skip transaction IDs (exactly 10 digits)
                    if len(num_str) == 10:
                        continue
                    val = float(num_str)
                    if val >= 0:
                        logger.info(f"VIB section balance (VND plain): found {val} for '{label}' on line: {line[:80]}")
                        return val

                # Check next few lines for standalone balance value
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()

                    # Skip empty lines
                    if not next_line:
                        continue

                    # Skip header/label lines but still check for balance number
                    if any(skip in next_line.lower() for skip in [
                        'opening', 'ending', 'seq.', 'withdrawal', 'deposit'
                    ]):
                        continue

                    # Skip month names as standalone (they are part of date, not balance)
                    if any(month in next_line.lower() for month in [
                        'november', 'december', 'january', 'february', 'march', 'april',
                        'may', 'june', 'july', 'august', 'september', 'october'
                    ]):
                        # But check if there's also a number on this line (balance after date)
                        vnd_in_line = re.findall(r'(\d{1,3}(?:,\d{3})+)', next_line)
                        if vnd_in_line:
                            num_str = vnd_in_line[-1]
                            digits_only = num_str.replace(',', '')
                            if len(digits_only) != 10:  # Not a transaction ID
                                val = self._parse_number_from_text(num_str)
                                if val is not None and val >= 0:
                                    logger.info(f"VIB section balance (VND with month): found {val} for '{label}'")
                                    return val
                        continue

                    # Check for comma-separated number
                    if re.match(r'^[\d,]+$', next_line) and ',' in next_line:
                        val = self._parse_number_from_text(next_line)
                        if val is not None and val >= 0:
                            logger.info(f"VIB section balance (VND next line): found {val} for '{label}'")
                            return val

                    # Check for plain number (e.g., small balance like 74472)
                    if re.match(r'^\d+$', next_line):
                        val = float(next_line)
                        # Skip years and small day numbers
                        if val > 100 and not re.match(r'^20[2-3]\d$', next_line):
                            logger.info(f"VIB section balance (plain next line): found {val} for '{label}'")
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
