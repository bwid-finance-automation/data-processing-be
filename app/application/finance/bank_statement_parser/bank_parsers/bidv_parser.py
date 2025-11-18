"""BIDV bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class BIDVParser(BaseBankParser):
    """Parser for BIDV (Bank for Investment and Development of Vietnam) statements."""

    @property
    def bank_name(self) -> str:
        return "BIDV"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is BIDV bank statement.

        Logic from fxLooksLike_BIDV:
        - Read first 80 rows
        - Look for any of these markers:
          * "NGÂN HÀNG TMCP ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM"
          * "BÁO CÁO CHI TIẾT SỐ DƯ TÀI KHOẢN"
          * "BIDV"
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 80 rows
            top_80 = df.head(80)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_80.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_80[col]])

            txt = " ".join(all_text)

            # Check for BIDV markers
            is_bidv = "NGÂN HÀNG TMCP ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM" in txt or \
                     "BÁO CÁO CHI TIẾT SỐ DƯ TÀI KHOẢN" in txt or \
                     "BIDV" in txt

            return is_bidv

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse BIDV transactions.

        Logic from fxParse_BIDV_Transactions:
        - Find header row with "NGÀY HIỆU LỰC" + "GHI NỢ" + "GHI CÓ"
        - Vietnamese/English column detection
        - Map: Debit = "GHI CÓ" (CREDIT), Credit = "GHI NỢ" (DEBIT) - REVERSED!
        - Extract account number from "SỐ TÀI KHOẢN" line (NOT "cũ"/"OLD")
        - DD/MM/YYYY date format support
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number from Header ==========
            top_40 = sheet.head(40)
            acc_no = self._extract_account_number_bidv(top_40)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_bidv(sheet.head(80))

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            date_col = self._find_any_column(data, ["NGÀY HIỆU LỰC", "DATE"])
            credit_col = self._find_any_column(data, ["GHI NỢ", "DEBIT"])  # Money out = Credit
            debit_col = self._find_any_column(data, ["GHI CÓ", "CREDIT"])  # Money in = Debit
            desc_col = self._find_any_column(data, ["MÔ TẢ", "DESCRIPTION"])

            # ========== Rename Columns ==========
            rename_map = {}
            if date_col: rename_map[date_col] = "Date"
            if debit_col: rename_map[debit_col] = "Debit"
            if credit_col: rename_map[credit_col] = "Credit"
            if desc_col: rename_map[desc_col] = "Description"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["Date", "Debit", "Credit", "Description"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Data Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self._fix_date_bidv)
            if "Debit" in data.columns:
                data["Debit"] = data["Debit"].apply(self._fix_number_bidv)
            if "Credit" in data.columns:
                data["Credit"] = data["Credit"].apply(self._fix_number_bidv)

            # ========== Filter Real Transactions (Date not null) ==========
            if "Date" in data.columns:
                data = data[data["Date"].notna()].copy()

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                debit_val = row.get("Debit") if "Debit" in row else None
                credit_val = row.get("Credit") if "Credit" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or (isinstance(debit_val, float) and pd.isna(debit_val)) or debit_val == 0) and \
                   (credit_val is None or (isinstance(credit_val, float) and pd.isna(credit_val)) or credit_val == 0):
                    continue

                tx = BankTransaction(
                    bank_name="BIDV",
                    acc_no=acc_no or "",
                    debit=debit_val if pd.notna(debit_val) else None,
                    credit=credit_val if pd.notna(credit_val) else None,
                    date=row.get("Date") if "Date" in row and pd.notna(row.get("Date")) else None,
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency="VND",
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing BIDV transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse BIDV balance information.

        Logic from fxParse_BIDV_Balances:
        - Opening from "DƯ ĐẦU" row (last number in that row)
        - Closing from "DƯ CUỐI" or "CUỐI NGÀY" row (last number)
        - Fallback: Grid-based calculation
        - Robust currency detection (VND, USD, EUR, etc.)
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_80 = sheet.head(80)
            acc_no = self._extract_account_number_bidv(top_80)
            currency = self._extract_currency_bidv(top_80)

            # ========== Extract Opening Balance from Header Text ==========
            opening = self._extract_opening_balance_bidv(top_80)

            # ========== Extract Closing Balance from Header Text ==========
            closing = self._extract_closing_balance_bidv(top_80)

            # ========== Grid-based Fallback if needed ==========
            if opening is None or closing is None:
                header_idx = self._find_header_row_bidv(top_80)
                data = sheet.iloc[header_idx:].copy()
                data.columns = data.iloc[0]
                data = data[1:].reset_index(drop=True)

                date_col = self._find_any_column(data, ["NGÀY HIỆU LỰC", "DATE"])
                credit_col = self._find_any_column(data, ["GHI NỢ", "DEBIT"])
                debit_col = self._find_any_column(data, ["GHI CÓ", "CREDIT"])
                balance_col = self._find_any_column(data, ["SỐ DƯ", "BALANCE"])

                keep_cols = []
                if date_col: keep_cols.append(date_col)
                if debit_col: keep_cols.append(debit_col)
                if credit_col: keep_cols.append(credit_col)
                if balance_col: keep_cols.append(balance_col)

                if keep_cols:
                    data = data[keep_cols].copy()

                    # Parse data types
                    if date_col:
                        data[date_col] = data[date_col].apply(self._fix_date_bidv)
                    if debit_col:
                        data[debit_col] = data[debit_col].apply(self._fix_number_bidv)
                    if credit_col:
                        data[credit_col] = data[credit_col].apply(self._fix_number_bidv)
                    if balance_col:
                        data[balance_col] = data[balance_col].apply(self._fix_number_bidv)

                    # Filter to real transaction rows
                    if date_col:
                        data_mov = data[data[date_col].notna()].copy()
                    else:
                        data_mov = data.copy()

                    # Calculate balances
                    sum_debit = data_mov[debit_col].sum() if debit_col and debit_col in data_mov.columns else 0
                    sum_credit = data_mov[credit_col].sum() if credit_col and credit_col in data_mov.columns else 0
                    net_move = sum_debit - sum_credit

                    # Closing = last non-null balance
                    if closing is None and balance_col and balance_col in data_mov.columns:
                        bal_list = data_mov[balance_col].dropna().tolist()
                        closing = bal_list[-1] if bal_list else None

                    # Opening = Closing - NetMove
                    if opening is None and closing is not None:
                        opening = closing - net_move

            return BankBalance(
                bank_name="BIDV",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing BIDV balances: {e}")
            return None

    # ========== BIDV-Specific Helper Methods ==========

    def _find_header_row_bidv(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing:
        "NGÀY HIỆU LỰC" + "GHI NỢ" + "GHI CÓ"
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])

            has_date = "NGÀY HIỆU LỰC" in row_text
            has_debit = "GHI NỢ" in row_text
            has_credit = "GHI CÓ" in row_text

            if has_date and has_debit and has_credit:
                return int(idx)

        return 11  # Default fallback

    def _find_any_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find first column matching any of the candidate names (case-insensitive)."""
        columns = df.columns.tolist()
        cols_upper = [str(col).upper() for col in columns]
        cands_upper = [c.upper() for c in candidates]

        for cand in cands_upper:
            for i, col_up in enumerate(cols_upper):
                if cand in col_up:
                    return columns[i]

        return None

    def _extract_account_number_bidv(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from "SỐ TÀI KHOẢN" line.
        IMPORTANT: Exclude lines with "CŨ" (old) or "OLD".
        Looks for 10-13 digit sequences.
        """
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            # Must contain "SỐ TÀI KHOẢN" but NOT "CŨ" or "OLD"
            if "SỐ TÀI KHOẢN" in row_text and "CŨ" not in row_text and "OLD" not in row_text:
                # Split by punctuation and spaces
                tokens = re.split(r'[ ,.;:|(){}\\[\\]<>\\-_/]+', row_text)
                tokens = [t for t in tokens if t]

                # Find first token with 10-13 digits
                for token in tokens:
                    digits = ''.join(c for c in token if c.isdigit())
                    if 10 <= len(digits) <= 13:
                        return digits

        return None

    def _extract_currency_bidv(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract currency from header area.
        Robust detection for VND, USD, EUR, JPY, CNY, HKD, SGD, THB, KRW, GBP, AUD, MYR, IDR, PHP, LAK.
        """
        known_currencies = ["VND", "VNĐ", "USD", "EUR", "JPY", "CNY", "HKD", "SGD", "THB", "KRW", "GBP", "AUD", "MYR", "IDR", "PHP", "LAK"]

        # Look for currency in "LOẠI TIỀN" or "ĐƠN VỊ TIỀN TỆ" or "TIỀN TỆ" rows first
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "LOẠI TIỀN" in row_text or "ĐƠN VỊ TIỀN TỆ" in row_text or "TIỀN TỆ" in row_text:
                for currency_code in known_currencies:
                    if currency_code in row_text:
                        return "VND" if currency_code == "VNĐ" else currency_code

        # Fallback: Search entire header area
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell).upper() for cell in top_df[col]])

        txt = " ".join(all_text)
        for currency_code in known_currencies:
            if currency_code in txt:
                return "VND" if currency_code == "VNĐ" else currency_code

        # Check for "ĐỒNG" keyword (Vietnamese for VND)
        if "ĐỒNG" in txt or "DONG" in txt:
            return "VND"

        return None

    def _extract_opening_balance_bidv(self, top_df: pd.DataFrame) -> Optional[float]:
        """
        Extract opening balance from "DƯ ĐẦU" row.
        Returns last number found in that row.
        """
        for idx, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "DƯ ĐẦU" in row_text and "CUỐI" not in row_text:
                # Extract all numbers from this row
                nums = []
                for cell in row:
                    num_val = self._fix_number_bidv(cell)
                    if num_val is not None:
                        nums.append(num_val)

                if nums:
                    return nums[-1]  # Return last number

        return None

    def _extract_closing_balance_bidv(self, top_df: pd.DataFrame) -> Optional[float]:
        """
        Extract closing balance from "DƯ CUỐI" or "CUỐI NGÀY" row.
        Returns last number found in that row.
        """
        for idx, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "DƯ CUỐI" in row_text or "CUỐI NGÀY" in row_text:
                # Extract all numbers from this row
                nums = []
                for cell in row:
                    num_val = self._fix_number_bidv(cell)
                    if num_val is not None:
                        nums.append(num_val)

                if nums:
                    return nums[-1]  # Return last number

        return None

    def _fix_number_bidv(self, value) -> Optional[float]:
        """
        BIDV-specific number parser.
        Removes: VND, commas, spaces
        Supports negative numbers with "-"
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Remove VND, commas, spaces
        txt = txt.upper().replace("VND", "").replace(",", "").replace(" ", "")

        # Extract digits, decimal point, and negative sign
        digits = ''.join(c for c in txt if c.isdigit() or c in ['-', '.'])
        if not digits or digits == '-' or digits == '.':
            return None

        try:
            return float(digits)
        except (ValueError, TypeError):
            return None

    def _fix_date_bidv(self, value) -> Optional[datetime]:
        """
        BIDV-specific date parser.
        Handles:
        - Excel date numbers
        - DD/MM/YYYY format (Vietnamese culture)
        - Datetime strings with space separator
        """
        if value is None or pd.isna(value):
            return None

        # Try as Excel date number first
        try:
            return pd.to_datetime(value).date()
        except:
            pass

        # Try parsing text
        txt = str(value).strip()
        if not txt:
            return None

        # Split if datetime string (e.g., "31/12/2023 10:30:45")
        if " " in txt:
            txt = txt.split(" ")[0]

        # Try DD/MM/YYYY format
        try:
            return datetime.strptime(txt, "%d/%m/%Y").date()
        except:
            pass

        # Fallback to pandas
        try:
            return pd.to_datetime(txt, dayfirst=True).date()
        except:
            return None
