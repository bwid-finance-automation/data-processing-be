"""CTBC bank statement parser."""

import io
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class CTBCParser(BaseBankParser):
    """Parser for CTBC (Chinatrust Commercial Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "CTBC"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is CTBC bank statement.

        Logic from fxLooksLike_CTBC:
        - Read first 30 rows
        - Look for markers:
          * "ctbc"
          * "account activity"
          * "debit account number"
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 30 rows
            top_30 = df.head(30)

            # Flatten all cells to lowercase text
            all_text = []
            for col in top_30.columns:
                all_text.extend([self.to_text(cell).lower() for cell in top_30[col]])

            flat = " ".join(all_text)

            # Check for CTBC markers
            is_ctbc = "ctbc" in flat or \
                      "account activity" in flat or \
                      "debit account number" in flat

            return is_ctbc

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse CTBC transactions.

        Logic from fxParse_CTBC_Transactions:
        - Find header row with "Date" + ("Debit amount" OR "Credit/Remittance Amount")
        - Extract columns: Date, Memo, Debit amount, Credit/Remittance Amount, Currency, Debit/credit account
        - Map: Debit = "Credit/Remittance Amount", Credit = "Debit amount" (REVERSED!)
        - Account number from "Debit/credit account" column (digits only)
        - Drop rows where both amounts are zero or Acc No is null
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_60 = sheet.head(60)
            start_at = self._find_header_row_ctbc(top_60)

            # ========== Promote Headers ==========
            data = sheet.iloc[start_at:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Select CTBC Columns ==========
            ctbc_columns = [
                "Date",
                "Memo",
                "Debit amount",
                "Credit/Remittance Amount",
                "Currency",
                "Debit/credit account",
                "Account Balance"
            ]

            # Filter to existing columns only
            available_cols = [col for col in ctbc_columns if col in data.columns]
            if not available_cols:
                return []

            data = data[available_cols].copy()

            # ========== Extract Currency from Header ==========
            currency_hdr = self._extract_currency_ctbc(top_60)

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                # CTBC REVERSED MAPPING:
                # Debit output = "Credit/Remittance Amount"
                # Credit output = "Debit amount"
                debit_val = self.fix_number(row.get("Credit/Remittance Amount")) if "Credit/Remittance Amount" in row else None
                credit_val = self.fix_number(row.get("Debit amount")) if "Debit amount" in row else None

                # Extract account number (digits only from "Debit/credit account")
                acc_no = None
                if "Debit/credit account" in row:
                    acc_text = self.to_text(row.get("Debit/credit account"))
                    acc_digits = ''.join(c for c in acc_text if c.isdigit())
                    acc_no = acc_digits if acc_digits else None

                # Skip rows with no account number or both amounts zero
                if not acc_no:
                    continue
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Extract currency from row, fallback to header
                currency = self.to_text(row.get("Currency", "")).strip()
                if not currency:
                    currency = currency_hdr or "VND"

                tx = BankTransaction(
                    bank_name="CTBC",
                    acc_no=acc_no,
                    debit=debit_val,
                    credit=credit_val,
                    date=self.fix_date(row.get("Date")) if "Date" in row else None,
                    description=self.to_text(row.get("Memo", "")) if "Memo" in row else "",
                    currency=currency,
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing CTBC transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse CTBC balance information.

        Logic from fxParse_CTBC_Balances:
        - Closing = last non-null "Account Balance"
        - Opening = first row's balance - first row's credit + first row's debit
        - Account number from "Debit/credit account" (first valid)
        - Currency from row or header fallback
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Header Row ==========
            top_60 = sheet.head(60)
            start_at = self._find_header_row_ctbc(top_60)

            # ========== Promote Headers ==========
            data = sheet.iloc[start_at:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Select CTBC Columns ==========
            ctbc_columns = [
                "Date",
                "Memo",
                "Debit amount",
                "Credit/Remittance Amount",
                "Account Balance",
                "Debit/credit account",
                "Currency"
            ]

            available_cols = [col for col in ctbc_columns if col in data.columns]
            if not available_cols:
                return None

            data = data[available_cols].copy()

            # ========== Parse Balance Columns ==========
            data["_In"] = data.get("Credit/Remittance Amount", pd.Series()).apply(self.fix_number)
            data["_Out"] = data.get("Debit amount", pd.Series()).apply(self.fix_number)
            data["_Bal"] = data.get("Account Balance", pd.Series()).apply(self.fix_number)

            # ========== Closing Balance (last non-null) ==========
            bal_list = data["_Bal"].dropna().tolist()
            closing = bal_list[-1] if bal_list else 0.0

            # ========== Opening Balance (from first row) ==========
            opening = 0.0
            if len(data) > 0:
                first_row = data.iloc[0]
                first_bal = first_row.get("_Bal")
                first_in = first_row.get("_In", 0) or 0
                first_out = first_row.get("_Out", 0) or 0

                if pd.notna(first_bal):
                    opening = first_bal - first_in + first_out

            # ========== Account Number (from "Debit/credit account") ==========
            acc_no = None
            if "Debit/credit account" in data.columns:
                for acc_text in data["Debit/credit account"]:
                    acc_digits = ''.join(c for c in self.to_text(acc_text) if c.isdigit())
                    if acc_digits:
                        acc_no = acc_digits
                        break

            # ========== Currency ==========
            currency = "VND"  # Default
            if "Currency" in data.columns:
                curr_vals = data["Currency"].dropna().astype(str).str.strip()
                curr_vals = curr_vals[curr_vals != ""]
                if len(curr_vals) > 0:
                    currency = curr_vals.iloc[0]

            # Fallback to header check
            if currency == "VND":
                currency_hdr = self._extract_currency_ctbc(top_60)
                if currency_hdr:
                    currency = currency_hdr

            return BankBalance(
                bank_name="CTBC",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening,
                closing_balance=closing
            )

        except Exception as e:
            print(f"Error parsing CTBC balances: {e}")
            return None

    # ========== CTBC-Specific Helper Methods ==========

    def _find_header_row_ctbc(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing "Date" + ("Debit amount" OR "Credit/Remittance Amount").
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell) for cell in row])
            has_date = "Date" in row_text
            has_debit = "Debit amount" in row_text
            has_credit = "Credit/Remittance Amount" in row_text

            if has_date and (has_debit or has_credit):
                return int(idx)

        return 17  # Default fallback

    def _extract_currency_ctbc(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract currency from header area."""
        # Flatten all cells
        all_text = []
        for col in top_df.columns:
            all_text.extend([self.to_text(cell).lower() for cell in top_df[col]])

        flat = " ".join(all_text)

        if "vnd" in flat or "vnđ" in flat:
            return "VND"
        elif "usd" in flat:
            return "USD"

        return None
