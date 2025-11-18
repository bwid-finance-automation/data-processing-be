"""ACB bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class ACBParser(BaseBankParser):
    """Parser for ACB (Asia Commercial Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "ACB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is ACB bank statement.

        Logic from fxLooksLike_ACB:
        - Read first 15 rows
        - Look for Vietnamese markers:
          * "BẢNG SAO KÊ GIAO DỊCH"
          * "SỐ DƯ ĐẦU"
          * "TIỀN RÚT RA" or "RÚT RA"
          * "TIỀN GỬI VÀO" or "GỬI VÀO"
        """
        try:
            # Read Excel workbook
            xls = pd.ExcelFile(io.BytesIO(file_bytes))

            # Try to read "Statement" sheet, otherwise first sheet
            try:
                df = pd.read_excel(xls, sheet_name="Statement", header=None)
            except:
                df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 15 rows
            top = df.head(15)

            # Convert all cells to uppercase text
            lines_upper = []
            for _, row in top.iterrows():
                row_text = " ".join([self.to_text(cell).upper() for cell in row])
                lines_upper.append(row_text)

            combined = " ".join(lines_upper)

            # Check for ACB-specific markers
            has_sao_ke = "BẢNG SAO KÊ GIAO DỊCH" in combined
            has_so_du_dau = "SỐ DƯ ĐẦU" in combined
            has_rut_ra = "TIỀN RÚT RA" in combined or "RÚT RA" in combined
            has_gui_vao = "TIỀN GỬI VÀO" in combined or "GỬI VÀO" in combined

            # ACB files have "BẢNG SAO KÊ" + balance structure
            is_acb = has_sao_ke and (has_so_du_dau or (has_rut_ra and has_gui_vao))

            return is_acb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse ACB transactions.

        Logic from fxParse_ACB_Transactions:
        1. Extract account number and currency from top 8 rows
        2. Find header row by scoring keywords
        3. Promote headers and map columns
        4. Clean and standardize data
        5. Return 11-column schema
        """
        try:
            # Read Excel
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            try:
                sheet = pd.read_excel(xls, sheet_name="Statement", header=None)
            except:
                sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            top_8 = sheet.head(8)
            acc_no = self._extract_account_number(top_8)

            # ========== Extract Currency ==========
            currency = self._extract_currency(top_8)

            # ========== Find Header Row ==========
            first_15 = sheet.head(15)
            header_idx = self._find_header_row(first_15)

            # ========== Parse Data Table ==========
            # Skip to header row and promote headers
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]  # First row as header
            data = data[1:].reset_index(drop=True)  # Remove header row from data

            # Clean column names
            data.columns = [self.to_text(col).strip() for col in data.columns]

            # ========== Map Columns ==========
            cols = list(data.columns)

            # Column mapping (0-indexed after header)
            date_col = cols[1] if len(cols) > 1 else None
            tx_id_col = cols[2] if len(cols) > 2 else None
            desc_col = cols[3] if len(cols) > 3 else None
            credit_col = cols[4] if len(cols) > 4 else None
            debit_col = cols[5] if len(cols) > 5 else None
            ben_name_col = cols[7] if len(cols) > 7 else None
            ben_acc_col = cols[8] if len(cols) > 8 else None

            # ========== Build Transaction List ==========
            transactions = []

            for _, row in data.iterrows():
                # Skip rows with no debit and no credit
                debit_val = self.fix_number(row.get(debit_col)) if debit_col else None
                credit_val = self.fix_number(row.get(credit_col)) if credit_col else None

                if debit_val is None and credit_val is None:
                    continue

                tx = BankTransaction(
                    bank_name="ACB",
                    acc_no=acc_no,
                    debit=debit_val,
                    credit=credit_val,
                    date=self.fix_date(row.get(date_col)) if date_col else None,
                    description=self.to_text(row.get(desc_col)) if desc_col else "",
                    currency=currency,
                    transaction_id=self.to_text(row.get(tx_id_col)) if tx_id_col else "",
                    beneficiary_bank="",
                    beneficiary_acc_no=self.to_text(row.get(ben_acc_col)) if ben_acc_col else "",
                    beneficiary_acc_name=self.to_text(row.get(ben_name_col)) if ben_name_col else ""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing ACB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse ACB balance information.

        Logic from fxParse_ACB_Balances:
        - Extract account number and currency
        - Use 3 methods to find Opening/Closing balance:
          1. Search by label ("số dư đầu", "số dư cuối")
          2. Fixed position (Row 6, Columns 3 & 6)
          3. Search for balance header row
        - Return first non-null result
        """
        try:
            # Read Excel
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            try:
                sheet = pd.read_excel(xls, sheet_name="Statement", header=None)
            except:
                sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number ==========
            top_8 = sheet.head(8)
            acc_no = self._extract_account_number(top_8)

            # ========== Extract Currency ==========
            currency = self._extract_currency(top_8)

            # ========== Method 1: Search by Label ==========
            top_12 = sheet.head(12)
            opening_m1 = self._find_value_under_label(top_12, "số dư đầu")
            closing_m1 = self._find_value_under_label(top_12, "số dư cuối")

            # ========== Method 2: Fixed Position (Row 6, Col 3 & 6) ==========
            opening_m2 = None
            closing_m2 = None
            if len(sheet) > 6:
                row_6 = sheet.iloc[6]
                if len(row_6) > 3:
                    opening_m2 = self.fix_number(row_6.iloc[3])
                if len(row_6) > 6:
                    closing_m2 = self.fix_number(row_6.iloc[6])

            # ========== Method 3: Search for Balance Header Row ==========
            opening_m3, closing_m3 = self._find_balance_by_header(sheet.head(12))

            # ========== Choose Best Result ==========
            opening_balance = opening_m1 or opening_m2 or opening_m3 or 0.0
            closing_balance = closing_m1 or closing_m2 or closing_m3 or 0.0

            return BankBalance(
                bank_name="ACB",
                acc_no=acc_no,
                currency=currency,
                opening_balance=opening_balance,
                closing_balance=closing_balance
            )

        except Exception as e:
            print(f"Error parsing ACB balances: {e}")
            return None

    # ========== Helper Methods ==========

    def _extract_account_number(self, top_rows: pd.DataFrame) -> str:
        """Extract account number from top rows."""
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])
            if "Số tài khoản" in row_text or "tài khoản" in row_text:
                # Extract digits only
                digits = ''.join(c if c.isdigit() else ' ' for c in row_text)
                chunks = [chunk for chunk in digits.split() if chunk]
                # Find chunks with >= 5 digits
                long_chunks = [chunk for chunk in chunks if len(chunk) >= 5]
                if long_chunks:
                    return long_chunks[-1]  # Take last one
        return ""

    def _extract_currency(self, top_rows: pd.DataFrame) -> str:
        """Extract currency from top rows."""
        for _, row in top_rows.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])
            if "USD" in row_text:
                return "USD"
            elif "VND" in row_text:
                return "VND"
        return "VND"  # Default

    def _find_header_row(self, first_15: pd.DataFrame) -> int:
        """
        Find header row by scoring keywords.

        Score based on presence of:
        - NGÀY GIAO DỊCH
        - RÚT RA / TIỀN RÚT
        - GỬI VÀO / TIỀN GỬI
        - NỘI DUNG
        """
        scores = []

        for idx, row in first_15.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            score = 0
            if "NGÀY GIAO DỊCH" in row_text:
                score += 1
            if "RÚT RA" in row_text or "TIỀN RÚT" in row_text:
                score += 1
            if "GỬI VÀO" in row_text or "TIỀN GỬI" in row_text:
                score += 1
            if "NỘI DUNG" in row_text:
                score += 1

            scores.append(score)

        # Find row with highest score
        max_score = max(scores) if scores else 0
        if max_score >= 3:
            return scores.index(max_score)

        return 7  # Default fallback

    def _find_value_under_label(self, top_rows: pd.DataFrame, label: str) -> Optional[float]:
        """Find numeric value in cell below a label."""
        rows_list = top_rows.values.tolist()

        for i, row in enumerate(rows_list):
            for j, cell in enumerate(row):
                cell_text = self.to_text(cell).lower()
                if label in cell_text:
                    # Check next row, same column
                    if i + 1 < len(rows_list) and j < len(rows_list[i + 1]):
                        return self.fix_number(rows_list[i + 1][j])
        return None

    def _find_balance_by_header(self, top_12: pd.DataFrame) -> tuple:
        """
        Find balance by searching for header row with both balance labels.
        Returns (opening_balance, closing_balance).
        """
        rows_list = top_12.values.tolist()

        # Find row with both "SỐ DƯ ĐẦU" and "SỐ DƯ CUỐI"
        for i, row in enumerate(rows_list):
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "SỐ DƯ ĐẦU" in row_text and "SỐ DƯ CUỐI" in row_text:
                # Found header row, get values from next row
                if i + 1 >= len(rows_list):
                    return None, None

                header_row = row
                value_row = rows_list[i + 1]

                # Find column indices
                opening_col = None
                closing_col = None

                for j, cell in enumerate(header_row):
                    cell_text = self.to_text(cell).upper()
                    if "SỐ DƯ ĐẦU" in cell_text:
                        opening_col = j
                    if "SỐ DƯ CUỐI" in cell_text:
                        closing_col = j

                opening = self.fix_number(value_row[opening_col]) if opening_col is not None and opening_col < len(value_row) else None
                closing = self.fix_number(value_row[closing_col]) if closing_col is not None and closing_col < len(value_row) else None

                return opening, closing

        return None, None
