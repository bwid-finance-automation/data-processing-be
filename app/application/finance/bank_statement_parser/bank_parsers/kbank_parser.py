"""KBANK bank statement parser."""

import io
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
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
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

        Logic from fxParse_KBANK_Transactions:
        - Find header row with: "transaction date" + "debit amount" + "credit amount" + "description"
        - Flexible column detection (bilingual: English/Vietnamese)
        - Map: Debit = "credit amount" (số tiền ghi có), Credit = "debit amount" (số tiền ghi nợ) - REVERSED!
        - Extract account number from header (2 methods)
        - Currency detection (USD/VND)
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
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
            col_credit_src = self._find_column(data, ["debit amount", "số tiền ghi nợ"])  # Source: debit
            col_debit_src = self._find_column(data, ["credit amount", "số tiền ghi có"])   # Source: credit
            col_balance = self._find_column(data, ["balance", "số dư"])
            col_desc = self._find_column(data, ["description", "diễn giải"])

            # Keep only available columns
            keep_cols = [c for c in [col_date, col_time, col_debit_src, col_credit_src, col_balance, col_desc] if c]
            if not keep_cols:
                return []

            data = data[keep_cols].copy()

            # ========== Extract Account Number & Currency ==========
            acc_no = self._extract_account_number_kbank(top_100)
            currency = self._extract_currency_kbank(top_100)

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                # KBANK REVERSED MAPPING:
                # Debit output = "credit amount" column (money in)
                # Credit output = "debit amount" column (money out)
                debit_val = self._fix_number_kbank(row.get(col_debit_src)) if col_debit_src else None
                credit_val = self._fix_number_kbank(row.get(col_credit_src)) if col_credit_src else None

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
        - Closing = last non-null "Balance"
        - Opening = first row's balance - first row's debit + first row's credit
        - Account number from header (2 methods)
        - Currency detection (USD/VND)
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
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
            col_debit_src = self._find_column(data, ["debit amount", "số tiền ghi nợ"])    # Money out
            col_credit_src = self._find_column(data, ["credit amount", "số tiền ghi có"])  # Money in
            col_balance = self._find_column(data, ["balance", "số dư"])

            keep_cols = [c for c in [col_date, col_debit_src, col_credit_src, col_balance] if c]
            if not keep_cols:
                return None

            data = data[keep_cols].copy()

            # ========== Parse Numeric Columns ==========
            data["_Date"] = data[col_date].apply(self.fix_date) if col_date else None
            data["_Out"] = data[col_debit_src].apply(self._fix_number_kbank) if col_debit_src else 0
            data["_In"] = data[col_credit_src].apply(self._fix_number_kbank) if col_credit_src else 0
            data["_Bal"] = data[col_balance].apply(self._fix_number_kbank) if col_balance else None

            # ========== Closing Balance (last non-null) ==========
            bal_list = data["_Bal"].dropna().tolist()
            closing = bal_list[-1] if bal_list else 0.0

            # ========== Opening Balance (from first row) ==========
            opening = 0.0
            if len(data) > 0:
                first_row = data.iloc[0]
                first_bal = first_row.get("_Bal")
                first_out = first_row.get("_Out", 0) or 0
                first_in = first_row.get("_In", 0) or 0

                if pd.notna(first_bal):
                    opening = first_bal - first_out + first_in

            # ========== Extract Account Number & Currency ==========
            acc_no = self._extract_account_number_kbank(top_100)
            currency = self._extract_currency_kbank(top_100)

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
