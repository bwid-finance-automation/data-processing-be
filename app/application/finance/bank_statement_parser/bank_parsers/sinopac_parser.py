"""SINOPAC bank statement parser."""

import io
from typing import List, Optional
import pandas as pd
import math

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class SINOPACParser(BaseBankParser):
    """Parser for SINOPAC (SinoPac Bank / 永豐銀行) statements."""

    @property
    def bank_name(self) -> str:
        return "SINOPAC"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is SINOPAC bank statement.

        Logic from fxLooksLike_SINOPAC:
        - Read first 40 rows
        - Look for Vietnamese headers:
          * ("CÂN ĐỐI" AND "NGÀY GIÁ TRỊ")
          * OR ("SỐ TÀI KHOẢN" AND "NHẬN XÉT")
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 40 rows
            top_40 = df.head(40)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_40.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_40[col]])

            row_text = "|".join(all_text)

            # Check for SINOPAC markers
            has_can_doi = "CÂN ĐỐI" in row_text
            has_ngay_gia_tri = "NGÀY GIÁ TRỊ" in row_text
            has_so_tai_khoan = "SỐ TÀI KHOẢN" in row_text
            has_nhan_xet = "NHẬN XÉT" in row_text

            is_sinopac = (has_can_doi and has_ngay_gia_tri) or \
                         (has_so_tai_khoan and has_nhan_xet)

            return is_sinopac

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse SINOPAC transactions.
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_80 = sheet.head(80)
            header_idx = self._find_header_row_sinopac(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Vietnamese Columns ==========
            col_acc = self._find_column_sinopac(data, "SỐ TÀI KHOẢN")
            col_date = self._find_column_sinopac(data, "NGÀY GIÁ TRỊ")
            col_currency = self._find_column_sinopac(data, "TIỀN TỆ")
            col_debit = self._find_column_sinopac(data, "TIỀN GỬI")     # Deposit = Debit
            col_credit = self._find_column_sinopac(data, "RÚT TIỀN")    # Withdraw = Credit
            col_balance = self._find_column_sinopac(data, "CÂN ĐỐI")    # Balance
            col_desc = self._find_column_sinopac(data, "NHẬN XÉT")      # Description

            # ========== Rename Columns ==========
            rename_map = {}
            if col_acc: rename_map[col_acc] = "AccNoRaw"
            if col_date: rename_map[col_date] = "Date"
            if col_currency: rename_map[col_currency] = "Currency"
            if col_debit: rename_map[col_debit] = "Debit"
            if col_credit: rename_map[col_credit] = "Credit"
            if col_balance: rename_map[col_balance] = "Balance"
            if col_desc: rename_map[col_desc] = "Description"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["AccNoRaw", "Date", "Currency", "Debit", "Credit", "Balance", "Description"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                # Parse numeric values
                debit_val = self._fix_number_sinopac(row.get("Debit")) if "Debit" in row else None
                credit_val = self._fix_number_sinopac(row.get("Credit")) if "Credit" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Extract account number (digits only)
                acc_no = ""
                if "AccNoRaw" in row:
                    acc_text = self.to_text(row.get("AccNoRaw"))
                    acc_no = ''.join(c for c in acc_text if c.isdigit())

                # Get currency
                currency = self.to_text(row.get("Currency", "VND")).strip()
                if not currency:
                    currency = "VND"

                tx = BankTransaction(
                    bank_name="SINOPAC",
                    acc_no=acc_no,
                    debit=debit_val,
                    credit=credit_val,
                    date=self.fix_date(row.get("Date")) if "Date" in row else None,
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency=currency,
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing SINOPAC transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse SINOPAC balance information.
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_80 = sheet.head(80)
            header_idx = self._find_header_row_sinopac(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Vietnamese Columns ==========
            col_acc = self._find_column_sinopac(data, "SỐ TÀI KHOẢN")
            col_date = self._find_column_sinopac(data, "NGÀY GIÁ TRỊ")
            col_currency = self._find_column_sinopac(data, "TIỀN TỆ")
            col_balance = self._find_column_sinopac(data, "CÂN ĐỐI")
            col_deposit = self._find_column_sinopac(data, "TIỀN GỬI")
            col_withdraw = self._find_column_sinopac(data, "RÚT TIỀN")

            # ========== Rename Columns ==========
            rename_map = {}
            if col_acc: rename_map[col_acc] = "AccNoRaw"
            if col_date: rename_map[col_date] = "Date"
            if col_currency: rename_map[col_currency] = "Currency"
            if col_balance: rename_map[col_balance] = "Balance"
            if col_deposit: rename_map[col_deposit] = "Deposit"
            if col_withdraw: rename_map[col_withdraw] = "Withdraw"

            if not rename_map:
                return None

            data = data.rename(columns=rename_map)

            # Keep columns
            keep_cols = ["AccNoRaw", "Date", "Currency", "Balance", "Deposit", "Withdraw"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return None

            data = data[available].copy()

            # ========== Parse Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self.fix_date)
            if "Balance" in data.columns:
                data["Balance"] = data["Balance"].apply(self._fix_number_sinopac)
            if "Deposit" in data.columns:
                data["Deposit"] = data["Deposit"].apply(self._fix_number_sinopac)
            if "Withdraw" in data.columns:
                data["Withdraw"] = data["Withdraw"].apply(self._fix_number_sinopac)

            # Extract account number (digits only)
            data["Acc No"] = data["AccNoRaw"].apply(lambda x: ''.join(c for c in self.to_text(x) if c.isdigit()))

            # ========== Get First Account (simplified - no grouping) ==========
            if len(data) == 0:
                return None

            # Sort by date
            data_sorted = data.sort_values(by="Date", na_position='last')

            # Get account number and currency
            acc_no = data_sorted["Acc No"].iloc[0] if "Acc No" in data_sorted.columns else ""

            currency = "VND"
            if "Currency" in data_sorted.columns:
                curr_vals = data_sorted["Currency"].dropna()
                if len(curr_vals) > 0:
                    currency = str(curr_vals.iloc[0]).strip() or "VND"

            # ========== Calculate Balances ==========
            # Closing = last non-null balance
            bal_list = data_sorted["Balance"].dropna().tolist()
            closing = bal_list[-1] if bal_list else 0.0

            # Opening = first row's balance - first row's deposit + first row's withdraw
            opening = 0.0
            if len(data_sorted) > 0:
                first_row = data_sorted.iloc[0]
                
                # [FIX] Handle NaN properly to prevent 500 error
                first_bal = first_row.get("Balance")
                if pd.isna(first_bal): first_bal = 0.0
                
                first_dep = first_row.get("Deposit")
                if pd.isna(first_dep): first_dep = 0.0
                
                first_wdr = first_row.get("Withdraw")
                if pd.isna(first_wdr): first_wdr = 0.0

                # Only calculate if we have a valid balance to start with
                if pd.notna(first_row.get("Balance")):
                    delta = first_dep - first_wdr
                    opening = first_bal - delta

            return BankBalance(
                bank_name="SINOPAC",
                acc_no=acc_no,
                currency=currency,
                opening_balance=opening,
                closing_balance=closing
            )

        except Exception as e:
            print(f"Error parsing SINOPAC balances: {e}")
            return None

    # ========== SINOPAC-Specific Helper Methods ==========

    def _find_header_row_sinopac(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing:
        "NGÀY GIÁ TRỊ" + ("CÂN ĐỐI" OR "TIỀN GỬI" OR "RÚT TIỀN")
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])

            has_date = "NGÀY GIÁ TRỊ" in row_text
            has_balance = "CÂN ĐỐI" in row_text
            has_deposit = "TIỀN GỬI" in row_text
            has_withdraw = "RÚT TIỀN" in row_text

            if has_date and (has_balance or has_deposit or has_withdraw):
                return int(idx)

        return 8  # Default fallback

    def _find_column_sinopac(self, df: pd.DataFrame, needle: str) -> Optional[str]:
        """Find column name containing the Vietnamese keyword."""
        columns = df.columns.tolist()

        for col in columns:
            col_upper = str(col).upper()
            if needle.upper() in col_upper:
                return col

        return None

    def _fix_number_sinopac(self, value) -> Optional[float]:
        """
        SINOPAC-specific number parser.
        Handles:
        - VND prefix
        - Spaces and commas
        - Trailing ".00" (removes only if zeros)
        - Bracket negatives: (1,234.00) -> -1234.0
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip().upper()
        if not txt:
            return None

        # Remove "VND" and non-breaking spaces
        txt = txt.replace("VND", "").replace(" ", "").replace("\xa0", "")

        # Handle bracket negatives
        is_negative = txt.startswith("(") and txt.endswith(")")
        if is_negative:
            txt = txt[1:-1]  # Remove parentheses

        # Remove commas
        txt = txt.replace(",", "")

        # Remove trailing ".00" only if all zeros after decimal
        if "." in txt:
            parts = txt.split(".")
            if len(parts) == 2:
                frac = parts[1].strip("0")
                if not frac:  # All zeros
                    txt = parts[0]

        # [FIX] Keep digits AND decimal point
        # Old code stripped '.', turning 10.50 into 1050
        chars = ''.join(c for c in txt if c.isdigit() or c == '.')
        
        if not chars:
            return None

        try:
            num = float(chars)
            return -num if is_negative else num
        except (ValueError, TypeError):
            return None