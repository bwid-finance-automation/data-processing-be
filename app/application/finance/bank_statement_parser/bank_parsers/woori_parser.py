"""Woori bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class WooriParser(BaseBankParser):
    """Parser for Woori Bank statements."""

    @property
    def bank_name(self) -> str:
        return "WOORI"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is Woori bank statement.

        Look for Woori-specific markers:
        - "Account holder :" + "Account number :" + "Inquiry Period :" (unique Woori header pattern)
        - OR "Woori" in content
        - OR "Amount withdrawn" + "Amount deposited" (unique column headers)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 30 rows
            top_30 = df.head(30)

            # Flatten all cells to text (case-sensitive first for exact patterns)
            all_text = []
            for col in top_30.columns:
                all_text.extend([self.to_text(cell) for cell in top_30[col]])

            txt = " ".join(all_text)
            txt_upper = txt.upper()

            # Method 1: Check for Woori-specific header pattern with colons
            # "Account holder :" and "Account number :" and "Inquiry Period :" are unique to Woori
            has_account_holder_colon = "Account holder :" in txt or "ACCOUNT HOLDER :" in txt_upper
            has_account_number_colon = "Account number :" in txt or "ACCOUNT NUMBER :" in txt_upper
            has_inquiry_period = "Inquiry Period :" in txt or "INQUIRY PERIOD :" in txt_upper
            has_woori_header_pattern = has_account_holder_colon and has_account_number_colon and has_inquiry_period

            # Method 2: Check for "Woori" bank name
            has_woori = "WOORI" in txt_upper

            # Method 3: Check for unique Woori column headers
            # "Amount withdrawn" and "Amount deposited" are English column headers unique to Woori
            has_amount_withdrawn = "AMOUNT WITHDRAWN" in txt_upper
            has_amount_deposited = "AMOUNT DEPOSITED" in txt_upper
            has_woori_columns = has_amount_withdrawn and has_amount_deposited

            # Method 4: Check for Opening/Closing balance with colon format
            has_opening_balance_colon = "Opening balance :" in txt or "OPENING BALANCE :" in txt_upper
            has_closing_balance_colon = "Closing balance :" in txt or "CLOSING BALANCE :" in txt_upper
            has_balance_pattern = has_opening_balance_colon and has_closing_balance_colon

            is_woori = has_woori_header_pattern or has_woori or has_woori_columns or has_balance_pattern

            return is_woori

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse Woori transactions.

        Woori Excel format:
        - Row 4: "ACCOUNT STATEMENT"
        - Row 6: "Account holder : ..."
        - Row 7: "Account number : ... [CURRENCY]"
        - Row 8: "Inquiry Period : ..."
        - Row 9: "Opening balance : ..."
        - Row 10: "Closing balance : ..."
        - Row 13: Header row: "Transaction time and date | Currency | Amount withdrawn | Amount deposited | Account balance | Status | Remarks | Summary"
        - Row 14+: Transaction data
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_30 = sheet.head(30)
            acc_no, currency = self._extract_account_info_woori(top_30)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_woori(top_30)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns ==========
            date_col = self._find_column_woori(data, ["Transaction time and date", "Transaction date"])
            currency_col = self._find_column_woori(data, ["Currency"])
            credit_col = self._find_column_woori(data, ["Amount withdrawn"])  # Money out = Credit
            debit_col = self._find_column_woori(data, ["Amount deposited"])   # Money in = Debit
            balance_col = self._find_column_woori(data, ["Account balance"])
            status_col = self._find_column_woori(data, ["Status"])
            remarks_col = self._find_column_woori(data, ["Remarks"])
            summary_col = self._find_column_woori(data, ["Summary"])

            # ========== Rename Columns ==========
            rename_map = {}
            if date_col: rename_map[date_col] = "Date"
            if currency_col: rename_map[currency_col] = "CurrencyRow"
            if debit_col: rename_map[debit_col] = "Debit"
            if credit_col: rename_map[credit_col] = "Credit"
            if balance_col: rename_map[balance_col] = "Balance"
            if status_col: rename_map[status_col] = "Status"
            if remarks_col: rename_map[remarks_col] = "Remarks"
            if summary_col: rename_map[summary_col] = "Summary"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["Date", "CurrencyRow", "Debit", "Credit", "Balance", "Status", "Remarks", "Summary"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Data Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self._fix_date_woori)
            if "Debit" in data.columns:
                data["Debit"] = data["Debit"].apply(self._fix_number_woori)
            if "Credit" in data.columns:
                data["Credit"] = data["Credit"].apply(self._fix_number_woori)
            if "Balance" in data.columns:
                data["Balance"] = data["Balance"].apply(self._fix_number_woori)

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                debit_val = row.get("Debit") if "Debit" in row else None
                credit_val = row.get("Credit") if "Credit" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or (isinstance(debit_val, float) and pd.isna(debit_val)) or debit_val == 0) and \
                   (credit_val is None or (isinstance(credit_val, float) and pd.isna(credit_val)) or credit_val == 0):
                    continue

                # Get row currency or use header currency
                row_currency = self.to_text(row.get("CurrencyRow", "")).strip()
                if not row_currency:
                    row_currency = currency or "VND"

                # Build description from Remarks + Summary
                desc_parts = []
                if "Remarks" in row and pd.notna(row.get("Remarks")):
                    desc_parts.append(self.to_text(row.get("Remarks")))
                if "Summary" in row and pd.notna(row.get("Summary")):
                    desc_parts.append(self.to_text(row.get("Summary")))
                description = " - ".join(desc_parts) if desc_parts else ""

                tx = BankTransaction(
                    bank_name="WOORI",
                    acc_no=acc_no or "",
                    debit=debit_val if pd.notna(debit_val) else None,
                    credit=credit_val if pd.notna(credit_val) else None,
                    date=row.get("Date") if "Date" in row and pd.notna(row.get("Date")) else None,
                    description=description,
                    currency=row_currency,
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing Woori transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse Woori balance information.

        Extract from header:
        - Row 9: "Opening balance : XX,XXX,XXX VND"
        - Row 10: "Closing balance : XX,XXX,XXX VND"
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_30 = sheet.head(30)
            acc_no, currency = self._extract_account_info_woori(top_30)

            # ========== Extract Opening Balance ==========
            opening = self._extract_balance_woori(top_30, ["Opening balance"])

            # ========== Extract Closing Balance ==========
            closing = self._extract_balance_woori(top_30, ["Closing balance"])

            # ========== Fallback: Use Last Balance from Grid ==========
            if closing is None:
                header_idx = self._find_header_row_woori(top_30)
                data = sheet.iloc[header_idx:].copy()
                data.columns = data.iloc[0]
                data = data[1:].reset_index(drop=True)

                balance_col = self._find_column_woori(data, ["Account balance"])
                if balance_col:
                    data["_Bal"] = data[balance_col].apply(self._fix_number_woori)
                    bal_list = data["_Bal"].dropna().tolist()
                    if bal_list:
                        closing = bal_list[-1]

            return BankBalance(
                bank_name="WOORI",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing Woori balances: {e}")
            return None

    # ========== Woori-Specific Helper Methods ==========

    def _find_header_row_woori(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing:
        "Transaction time and date" OR "Amount withdrawn" OR "Amount deposited"
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell) for cell in row])

            has_date = "Transaction time and date" in row_text or "Transaction date" in row_text
            has_withdrawn = "Amount withdrawn" in row_text
            has_deposited = "Amount deposited" in row_text

            if has_date or (has_withdrawn and has_deposited):
                return int(idx)

        return 13  # Default fallback based on sample

    def _find_column_woori(self, df: pd.DataFrame, patterns: list) -> Optional[str]:
        """Find column name containing any of the patterns."""
        columns = df.columns.tolist()

        for col in columns:
            col_text = str(col)
            for pattern in patterns:
                if pattern in col_text:
                    return col

        return None

    def _extract_account_info_woori(self, top_df: pd.DataFrame) -> tuple:
        """
        Extract account number and currency from header.
        Format: "Account number : 123456789012 [VND]"
        """
        acc_no = None
        currency = "VND"

        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            # Look for account number line
            if "Account number" in row_text:
                # Extract digits for account number
                match = re.search(r':\s*(\d+)', row_text)
                if match:
                    acc_no = match.group(1)

                # Extract currency in brackets [VND] or [USD]
                curr_match = re.search(r'\[([A-Z]{3})\]', row_text)
                if curr_match:
                    currency = curr_match.group(1)

                break

        return acc_no, currency

    def _extract_balance_woori(self, top_df: pd.DataFrame, markers: list) -> Optional[float]:
        """Extract balance from row containing any of the markers."""
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            for marker in markers:
                if marker in row_text:
                    # Extract number after ":"
                    if ":" in row_text:
                        after_colon = row_text.split(":", 1)[1]
                        num_val = self._fix_number_woori(after_colon)
                        if num_val is not None:
                            return num_val

        return None

    def _fix_number_woori(self, value) -> Optional[float]:
        """
        Woori-specific number parser.
        Handles:
        - VND/USD suffix
        - Commas as thousand separators
        - Spaces
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Remove currency codes and spaces
        txt = txt.replace("VND", "").replace("USD", "").replace("EUR", "")
        txt = txt.replace(",", "").replace(" ", "")

        # Extract digits and decimal
        digits = ''.join(c for c in txt if c.isdigit() or c in ['-', '.'])
        if not digits:
            return None

        try:
            return float(digits)
        except (ValueError, TypeError):
            return None

    def _fix_date_woori(self, value) -> Optional[datetime]:
        """
        Woori-specific date parser.
        Handles format: "DD.MM.YYYY HH:MM:SS" or "DD.MM.YYYY"
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

        # Split if datetime string (e.g., "16.11.2025 16:50:55")
        if " " in txt:
            txt = txt.split(" ")[0]

        # Try DD.MM.YYYY format
        try:
            return datetime.strptime(txt, "%d.%m.%Y").date()
        except:
            pass

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
