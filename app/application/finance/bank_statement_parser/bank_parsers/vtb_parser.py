"""VTB bank statement parser (VietinBank)."""

import io
import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class VTBParser(BaseBankParser):
    """Parser for VTB (VietinBank) eFAST statements."""

    @property
    def bank_name(self) -> str:
        return "VTB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is VTB bank statement.

        Logic from fxLooksLike_VTB:
        - Read first 40 rows
        - Look for markers:
          * "VIETINBANK"
          * "NGÂN HÀNG TMCP CÔNG THƯƠNG"
          * "LICH SỬ GIAO DỊCH - TRANSACTION HISTORY"
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 40 rows
            top_40 = df.head(40)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_40.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_40[col]])

            txt = " ".join(all_text)

            # Check for VTB markers
            is_vtb = "VIETINBANK" in txt or \
                     "NGÂN HÀNG TMCP CÔNG THƯƠNG" in txt or \
                     "LICH SỬ GIAO DỊCH - TRANSACTION HISTORY" in txt

            return is_vtb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse VTB transactions.

        Logic from fxParse_VTB_Transactions:
        - Find header row with "Ngày hạch toán" OR "Transaction date" OR "Mô tả giao dịch"
        - Bilingual column detection
        - Map: Debit = "Có / Credit", Credit = "Nợ / Debit" (REVERSED!)
        - Extract account number from "Account No" or "Số tài khoản"
        - Currency detection from header
        - Date fill-down support
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_120 = sheet.head(120)
            acc_no = self._extract_account_number_vtb(top_120)
            currency = self._extract_currency_vtb(top_120)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_vtb(sheet)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            date_col = self._find_column_contains(data, ["Ngày phát sinh", "Transaction date", "Ngày hạch toán", "Accounting date"])
            debit_col = self._find_column_contains(data, ["Có / Credit", "Có  Credit", "Credit"])
            credit_col = self._find_column_contains(data, ["Nợ/ Debit", "Nợ / Debit", "Debit"])
            desc_col = self._find_column_contains(data, ["Mô tả giao dịch", "Transaction description"])
            tx_id_col = self._find_column_contains(data, ["Số giao dịch", "Transaction number"])

            # ========== Rename Columns ==========
            rename_map = {}
            if date_col: rename_map[date_col] = "Date"
            if debit_col: rename_map[debit_col] = "Debit"
            if credit_col: rename_map[credit_col] = "Credit"
            if desc_col: rename_map[desc_col] = "Description"
            if tx_id_col: rename_map[tx_id_col] = "Transaction ID"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["Date", "Debit", "Credit", "Description", "Transaction ID"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Data Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self._fix_date_vtb)
            if "Debit" in data.columns:
                data["Debit"] = data["Debit"].apply(self.fix_number)
            if "Credit" in data.columns:
                data["Credit"] = data["Credit"].apply(self.fix_number)

            # ========== Fill Down Date (some rows have blank dates) ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].fillna(method='ffill')

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
                    bank_name="VTB",
                    acc_no=acc_no or "",
                    debit=debit_val if pd.notna(debit_val) else None,
                    credit=credit_val if pd.notna(credit_val) else None,
                    date=row.get("Date") if "Date" in row and pd.notna(row.get("Date")) else None,
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency=currency or "VND",
                    transaction_id=self.to_text(row.get("Transaction ID", "")) if "Transaction ID" in row else "",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing VTB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse VTB balance information.

        Logic from fxParse_VTB_Balances:
        - Opening from "SỐ DƯ ĐẦU KỲ" or "OPENING BALANCE" row
        - Closing from "SỐ DƯ CUỐI KỲ" or "CLOSING BALANCE" row
        - Fallback: Grid-based calculation
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_120 = sheet.head(120)
            acc_no = self._extract_account_number_vtb(top_120)
            currency = self._extract_currency_vtb(top_120)

            # ========== Extract Opening Balance from Header Text ==========
            opening = self._extract_balance_vtb(top_120, ["SỐ DƯ ĐẦU KỲ", "OPENING BALANCE"])

            # ========== Extract Closing Balance from Header Text ==========
            closing = self._extract_balance_vtb(top_120, ["SỐ DƯ CUỐI KỲ", "CLOSING BALANCE"])

            # ========== Grid-based Fallback if needed ==========
            if opening is None or closing is None:
                header_idx = self._find_header_row_vtb(sheet.head(120))
                data = sheet.iloc[header_idx:].copy()
                data.columns = data.iloc[0]
                data = data[1:].reset_index(drop=True)

                debit_col = self._find_column_contains(data, ["CÓ / CREDIT", "CÓ  CREDIT", "CREDIT"])
                credit_col = self._find_column_contains(data, ["NỢ/ DEBIT", "NỢ / DEBIT", "DEBIT"])
                balance_col = self._find_column_contains(data, ["SỐ DƯ TK", "ACCOUNT BALANCE"])

                keep_cols = []
                if debit_col: keep_cols.append(debit_col)
                if credit_col: keep_cols.append(credit_col)
                if balance_col: keep_cols.append(balance_col)

                if keep_cols:
                    data = data[keep_cols].copy()

                    # Parse data types
                    if debit_col:
                        data[debit_col] = data[debit_col].apply(self.fix_number)
                    if credit_col:
                        data[credit_col] = data[credit_col].apply(self.fix_number)
                    if balance_col:
                        data[balance_col] = data[balance_col].apply(self.fix_number)

                    # Calculate balances
                    sum_in = data[debit_col].sum() if debit_col else 0
                    sum_out = data[credit_col].sum() if credit_col else 0
                    net_move = sum_in - sum_out

                    # Closing = last non-null balance
                    if closing is None and balance_col:
                        bal_list = data[balance_col].dropna().tolist()
                        closing = bal_list[-1] if bal_list else None

                    # Opening = Closing - NetMove
                    if opening is None and closing is not None:
                        opening = closing - net_move

            return BankBalance(
                bank_name="VTB",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing VTB balances: {e}")
            return None

    # ========== VTB-Specific Helper Methods ==========

    def _find_header_row_vtb(self, sheet: pd.DataFrame) -> int:
        """
        Find header row containing:
        "Ngày hạch toán" OR "Accounting date" OR "Mô tả giao dịch" OR "Transaction description"
        """
        row_texts = []
        for idx, row in sheet.head(120).iterrows():
            row_text = "|".join([self.to_text(cell) for cell in row])
            row_texts.append(row_text)

        for idx, txt in enumerate(row_texts):
            if "Ngày hạch toán" in txt or "Accounting date" in txt or \
               "Mô tả giao dịch" in txt or "Transaction description" in txt:
                return idx

        return 20  # Default fallback

    def _find_column_contains(self, df: pd.DataFrame, patterns: list) -> Optional[str]:
        """Find column name containing any of the patterns."""
        columns = df.columns.tolist()

        for col in columns:
            col_text = str(col)
            for pattern in patterns:
                if pattern in col_text:
                    return col

        return None

    def _extract_account_number_vtb(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from "Account No" or "Số tài khoản" line.
        Looks for 6-20 digit sequences.
        """
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            if "Account No" in row_text or "Số tài khoản" in row_text:
                # Split by punctuation and spaces
                tokens = re.split(r'[ ,.;:|(){}\\[\\]<>\\-_/]+', row_text)
                tokens = [t for t in tokens if t]

                # Find first token with 6-20 digits
                for token in tokens:
                    digits = ''.join(c for c in token if c.isdigit())
                    if 6 <= len(digits) <= 20:
                        return digits

        return None

    def _extract_currency_vtb(self, top_df: pd.DataFrame) -> str:
        """Extract currency from header area."""
        known_currencies = ["VND", "USD", "EUR", "JPY", "CNY", "AUD", "SGD", "GBP"]

        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "CURRENCY" in row_text or "LOẠI TIỀN" in row_text:
                for currency in known_currencies:
                    if currency in row_text:
                        return currency

        return "VND"  # Default

    def _extract_balance_vtb(self, top_df: pd.DataFrame, markers: list) -> Optional[float]:
        """Extract balance from row containing any of the markers."""
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            for marker in markers:
                if marker in row_text:
                    # Try to extract number from this row
                    num_val = self.fix_number(row_text)
                    if num_val is not None:
                        return num_val

        return None

    def _fix_date_vtb(self, value) -> Optional[datetime]:
        """
        VTB-specific date parser.
        Handles DD/MM/YYYY format and datetime strings.
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
