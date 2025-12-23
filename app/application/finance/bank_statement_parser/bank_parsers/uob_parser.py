"""UOB bank statement parser."""

import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class UOBParser(BaseBankParser):
    """Parser for UOB (United Overseas Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "UOB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is UOB bank statement.

        Detection markers:
        - "Tài khoản Thanh toán/Tiết kiệm" in first rows
        - "Tiền gửi (DEBIT)" and "Tiền rút (CREDIT)" in header
        - Sheet name contains "Account Activities"
        """
        try:
            xls = self.get_excel_file(file_bytes)

            # Check sheet name for UOB pattern
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            has_account_activities = "Account Activities" in sheet_name

            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 15 rows
            top_15 = df.head(15)

            # Convert all cells to text
            all_text = []
            for _, row in top_15.iterrows():
                row_text = " ".join([self.to_text(cell).upper() for cell in row])
                all_text.append(row_text)

            combined = " ".join(all_text)

            # Check for UOB-specific markers
            has_title = "TÀI KHOẢN THANH TOÁN" in combined or "TÀI KHOẢN TIẾT KIỆM" in combined
            has_deposit = "TIỀN GỬI" in combined and "DEBIT" in combined
            has_withdraw = "TIỀN RÚT" in combined and "CREDIT" in combined
            has_ledger_balance = "SỐ DƯ SỔ CÁI" in combined

            # UOB files have specific combination of markers
            is_uob = (has_account_activities or has_title) and (has_deposit or has_withdraw or has_ledger_balance)

            return is_uob

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse UOB transactions.

        UOB column structure (after header row 10):
        - Col 0: Ngày hiệu lực (Effective Date) - DD/MM/YYYY
        - Col 1: Ngày Giao dịch (Transaction Date)
        - Col 2: Dấu thời gian (Timestamp)
        - Col 3: Mô tả (Description) - contains transaction ID in newlines
        - Col 4: Tiền gửi (DEBIT) - Money IN = our Debit
        - Col 5: Tiền rút (CREDIT) - Money OUT = our Credit
        - Col 6: Số dư (Balance)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            acc_no = self._extract_account_number(sheet.head(10))

            # ========== Extract Currency ==========
            currency = self._extract_currency(sheet.head(10))

            # ========== Find Header Row ==========
            header_idx = self._find_header_row(sheet.head(15))

            if header_idx is None:
                return []

            # ========== Parse Transactions ==========
            transactions = []

            for idx in range(header_idx + 1, len(sheet)):
                row = sheet.iloc[idx]

                # Stop at totals row
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Tổng số theo Loại tiền tệ" in row_text:
                    break

                # Skip empty rows
                if pd.isna(row.iloc[0]) or self.to_text(row.iloc[0]).strip() == "":
                    continue

                # Skip note rows
                if "Lưu ý:" in row_text or "Bảo hiểm Tiền gửi" in row_text:
                    break

                # Parse date (Col 0)
                date_val = self._parse_date_uob(row.iloc[0])

                # Parse description and extract transaction ID (Col 3)
                desc_raw = self.to_text(row.iloc[3]) if len(row) > 3 else ""
                tx_id, description = self._parse_description(desc_raw)

                # Parse amounts
                # Col 4: Tiền gửi = Money IN = Debit
                # Col 5: Tiền rút = Money OUT = Credit
                debit_val = self.fix_number(row.iloc[4]) if len(row) > 4 else None
                credit_val = self.fix_number(row.iloc[5]) if len(row) > 5 else None

                # Skip if both are zero or None
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Handle "0" string values
                if debit_val == 0:
                    debit_val = None
                if credit_val == 0:
                    credit_val = None

                tx = BankTransaction(
                    bank_name="UOB",
                    acc_no=acc_no or "",
                    debit=debit_val,
                    credit=credit_val,
                    date=date_val,
                    description=description,
                    currency=currency,
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing UOB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse UOB balance information.

        Balance extraction:
        - Closing balance: Row 3, Col 4 "Số dư Sổ cái:" value
        - Opening balance: Calculate from first transaction's balance minus net change
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            acc_no = self._extract_account_number(sheet.head(10))

            # ========== Extract Currency ==========
            currency = self._extract_currency(sheet.head(10))

            # ========== Extract Closing Balance from Header ==========
            closing_balance = self._extract_closing_balance(sheet.head(10))

            # ========== Calculate Opening Balance ==========
            opening_balance = self._calculate_opening_balance(sheet, closing_balance)

            return BankBalance(
                bank_name="UOB",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening_balance or 0.0,
                closing_balance=closing_balance or 0.0
            )

        except Exception as e:
            print(f"Error parsing UOB balances: {e}")
            return None

    # ========== Helper Methods ==========

    def _extract_account_number(self, top_rows: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from header area.
        Located in Row 2, Col 4 after "Số Tài khoản:"
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            if "Số Tài khoản:" in row_text or "Số tài khoản:" in row_text:
                # Account number is in column 4
                if len(row) > 4:
                    acc_text = self.to_text(row.iloc[4])
                    # Extract digits
                    digits = ''.join(c for c in acc_text if c.isdigit())
                    if 8 <= len(digits) <= 15:
                        return digits

        return None

    def _extract_currency(self, top_rows: pd.DataFrame) -> str:
        """
        Extract currency from header area.
        Located in Row 5 after "Loại tiền tệ Tài khoản:"
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "LOẠI TIỀN TỆ" in row_text:
                if len(row) > 1:
                    curr_val = self.to_text(row.iloc[1]).upper().strip()
                    if curr_val in ["VND", "USD", "EUR", "SGD", "JPY"]:
                        return curr_val

        # Default
        return "VND"

    def _extract_closing_balance(self, top_rows: pd.DataFrame) -> Optional[float]:
        """
        Extract closing balance from "Số dư Sổ cái:" or "Số dư khả dụng:" row.
        """
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            if "Số dư Sổ cái:" in row_text or "Số dư khả dụng:" in row_text:
                # Balance value is in column 4
                if len(row) > 4:
                    return self.fix_number(row.iloc[4])

        return None

    def _calculate_opening_balance(self, sheet: pd.DataFrame, closing_balance: Optional[float]) -> Optional[float]:
        """
        Calculate opening balance by working backwards from closing balance.

        Opening = Closing - Total Credits + Total Debits
        Or: Get balance from first transaction and subtract net change of first tx
        """
        try:
            header_idx = self._find_header_row(sheet.head(15))
            if header_idx is None:
                return None

            # Get first transaction
            first_tx_idx = header_idx + 1
            if first_tx_idx >= len(sheet):
                return closing_balance

            first_row = sheet.iloc[first_tx_idx]

            # Skip if empty
            if pd.isna(first_row.iloc[0]):
                return closing_balance

            # Get first transaction's balance (Col 6)
            first_balance = self.fix_number(first_row.iloc[6]) if len(first_row) > 6 else None

            # Get first transaction's amounts
            first_debit = self.fix_number(first_row.iloc[4]) if len(first_row) > 4 else 0
            first_credit = self.fix_number(first_row.iloc[5]) if len(first_row) > 5 else 0

            if first_balance is not None:
                # Opening = First Balance - First Debit + First Credit
                first_debit = first_debit or 0
                first_credit = first_credit or 0
                opening = first_balance - first_debit + first_credit
                return opening

            return closing_balance

        except Exception:
            return closing_balance

    def _find_header_row(self, top_rows: pd.DataFrame) -> Optional[int]:
        """
        Find header row by looking for specific column names.
        Header contains: "Ngày hiệu lực", "Tiền gửi", "Tiền rút"
        """
        for idx, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            has_date = "NGÀY HIỆU LỰC" in row_text or "NGÀY GIAO DỊCH" in row_text
            has_deposit = "TIỀN GỬI" in row_text
            has_withdraw = "TIỀN RÚT" in row_text

            if has_date and (has_deposit or has_withdraw):
                return int(idx)

        return None

    def _parse_date_uob(self, value) -> Optional:
        """
        Parse UOB date format: DD/MM/YYYY
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Try DD/MM/YYYY format
        try:
            return datetime.strptime(txt, "%d/%m/%Y").date()
        except:
            pass

        # Try pandas parser
        try:
            return pd.to_datetime(txt, dayfirst=True).date()
        except:
            pass

        return None

    def _parse_description(self, desc_raw: str) -> tuple:
        """
        Parse description and extract transaction ID.

        UOB description format (multi-line):
        Line 1: Type (MISC CREDIT/MISC DEBIT)
        Line 2: Transaction ID or "NONE"
        Line 3: Reference
        Line 4+: Details

        Returns: (transaction_id, cleaned_description)
        """
        if not desc_raw:
            return "", ""

        # Split by newlines
        lines = [line.strip() for line in desc_raw.split('\n') if line.strip()]

        if not lines:
            return "", ""

        tx_id = ""
        description_parts = []

        for i, line in enumerate(lines):
            if i == 1:
                # Second line is usually transaction ID
                if line != "NONE" and re.match(r'^[A-Z0-9]+$', line):
                    tx_id = line
                else:
                    description_parts.append(line)
            elif i == 0:
                # First line is type (MISC CREDIT/MISC DEBIT)
                description_parts.append(line)
            else:
                description_parts.append(line)

        description = " | ".join(description_parts)

        return tx_id, description
