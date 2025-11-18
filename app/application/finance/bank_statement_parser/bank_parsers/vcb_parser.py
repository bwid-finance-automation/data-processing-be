"""VCB bank statement parser (Vietcombank)."""

import io
import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class VCBParser(BaseBankParser):
    """Parser for VCB (Vietcombank) statements - handles both Template 1 and Template 2."""

    @property
    def bank_name(self) -> str:
        return "VCB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is VCB bank statement.

        Logic from fxLooksLike_VCB:
        - Template 1: Has "VCB CASHUP" or "STATEMENT OF ACCOUNT"
        - Read first 20 rows for detection
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 20 rows
            top_20 = df.head(20)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_20.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_20[col]])

            txt = " ".join(all_text)

            # Check for VCB markers (Template 1)
            is_vcb = "VCB CASHUP" in txt or "STATEMENT OF ACCOUNT" in txt

            return is_vcb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse VCB transactions - handles BOTH templates.

        Logic from fxParse_VCB_Transactions:
        - Template detection: Simple format has "Ngày giao dịch" as column 0, Multi has "STT" as column 0
        - Map: Debit = "Credit/Remittance" (CÓ), Credit = "Debit" (NỢ) - SWAPPED!
        - Template 1 (Multi-account): Multiple account sections
        - Template 2 (Simple): Single account
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Detect Template ==========
            is_simple_template = self._is_simple_template(sheet)

            # ========== Parse Based on Template ==========
            if is_simple_template:
                return self._parse_simple_template(sheet)
            else:
                return self._parse_multi_template(sheet)

        except Exception as e:
            print(f"Error parsing VCB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse VCB balance information.

        Logic from fxParse_VCB_Balances:
        - Extracts opening/closing balances for all accounts
        - Opening from "Số dư đầu kỳ/ Opening balance :" row
        - Closing from "Số dư cuối kỳ/ Closing balance :" row
        - Fallback: Last balance in transaction table
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Find Account Sections ==========
            account_sections = self._find_account_sections(sheet)

            if not account_sections:
                return None

            # For simplicity, extract balance from first account
            # (In practice, VCB may have multiple accounts per file)
            section = account_sections[0]
            section_rows = sheet.iloc[section['start']:section['end']+1]

            # ========== Extract Account Number ==========
            acc_no = self._extract_account_number_vcb(section_rows.head(8))

            # ========== Extract Currency ==========
            currency = self._extract_currency_vcb(section_rows.head(15))

            # ========== Extract Opening Balance ==========
            opening = self._extract_balance_by_label(section_rows.head(20), ["Số dư đầu kỳ", "Opening balance"])

            # ========== Extract Closing Balance ==========
            closing = self._extract_balance_by_label(section_rows, ["Số dư cuối kỳ", "Closing balance"])

            # ========== Fallback: Use Last Balance from Grid ==========
            if closing is None:
                closing = self._extract_last_balance_from_grid(section_rows)

            return BankBalance(
                bank_name="VCB",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing VCB balances: {e}")
            return None

    # ========== VCB-Specific Helper Methods ==========

    def _is_simple_template(self, sheet: pd.DataFrame) -> bool:
        """
        Detect if using Simple Template (single account) or Multi Template (multi-account).
        Simple: Has "Ngày giao dịch|Số tham chiếu|Số tiền" pattern in first 20 rows.
        """
        top_20 = sheet.head(20)
        for _, row in top_20.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])
            if "NGÀY GIAO DỊCH" in row_text and "SỐ THAM CHIẾU" in row_text and "SỐ TIỀN" in row_text:
                return True
        return False

    def _parse_simple_template(self, sheet: pd.DataFrame) -> List[BankTransaction]:
        """Parse VCB Simple Template (Vietcombank_Account_Statement format)."""
        # ========== Extract Account Number and Currency ==========
        top_40 = sheet.head(40)
        acc_no = self._extract_account_number_from_header(top_40, "SỐ TÀI KHOẢN")
        currency = self._extract_currency_from_header(top_40)

        # ========== Find Header Row ==========
        header_idx = None
        for idx, row in top_40.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])
            if "NGÀY GIAO DỊCH" in row_text and "SỐ TIỀN GHI" in row_text:
                header_idx = int(idx)
                break

        if header_idx is None:
            return []

        # ========== Promote Headers ==========
        data = sheet.iloc[header_idx:].copy()
        data.columns = data.iloc[0]
        data = data[1:].reset_index(drop=True)

        # Get columns (positional for Simple Template)
        cols = list(data.columns)
        if len(cols) < 5:
            return []

        date_col = cols[0] if len(cols) > 0 else None
        tx_id_col = cols[1] if len(cols) > 1 else None
        debit_col = cols[2] if len(cols) > 2 else None  # Actually "Credit" in source (money in)
        credit_col = cols[3] if len(cols) > 3 else None  # Actually "Debit" in source (money out)
        desc_col = cols[4] if len(cols) > 4 else None

        # ========== Select and Rename Columns ==========
        keep = [c for c in [date_col, tx_id_col, debit_col, credit_col, desc_col] if c]
        if not keep:
            return []

        data = data[keep].copy()

        # Parse types
        data[date_col] = data[date_col].apply(self._fix_date_vcb) if date_col else None
        data[debit_col] = data[debit_col].apply(self.fix_number) if debit_col else None
        data[credit_col] = data[credit_col].apply(self.fix_number) if credit_col else None

        # ========== Build Transactions ==========
        transactions = []
        for _, row in data.iterrows():
            # Skip rows with "Tổng" (totals)
            tx_id = self.to_text(row.get(tx_id_col, "")) if tx_id_col else ""
            if "TỔNG" in tx_id.upper():
                continue

            debit_val = row.get(debit_col) if debit_col else None
            credit_val = row.get(credit_col) if credit_col else None

            if (debit_val is None or pd.isna(debit_val) or debit_val == 0) and \
               (credit_val is None or pd.isna(credit_val) or credit_val == 0):
                continue

            tx = BankTransaction(
                bank_name="VCB",
                acc_no=acc_no or "",
                debit=credit_val if pd.notna(credit_val) else None,  # SWAPPED
                credit=debit_val if pd.notna(debit_val) else None,   # SWAPPED
                date=row.get(date_col) if date_col and pd.notna(row.get(date_col)) else None,
                description=self.to_text(row.get(desc_col, "")) if desc_col else "",
                currency=currency or "VND",
                transaction_id=tx_id,
                beneficiary_bank="",
                beneficiary_acc_no="",
                beneficiary_acc_name=""
            )

            transactions.append(tx)

        return transactions

    def _parse_multi_template(self, sheet: pd.DataFrame) -> List[BankTransaction]:
        """Parse VCB Multi-Account Template (AccountStatement format)."""
        # Find all account section start points
        account_sections = self._find_account_sections(sheet)

        if not account_sections:
            return []

        all_transactions = []

        for section in account_sections:
            section_rows = sheet.iloc[section['start']:section['end']+1]

            # ========== Extract Account Number and Currency for this section ==========
            acc_no = self._extract_account_number_vcb(section_rows.head(8))
            currency = self._extract_currency_vcb(section_rows.head(15))

            # ========== Find Transaction Header in Section ==========
            header_idx = None
            for idx, row in section_rows.iterrows():
                row_text = "|".join([self.to_text(cell).upper() for cell in row])
                if "NGÀY" in row_text and "NỢ" in row_text and "CÓ" in row_text:
                    header_idx = int(idx)
                    break

            if header_idx is None:
                continue

            # ========== Promote Headers ==========
            tx_data = sheet.iloc[header_idx:section['end']+1].copy()
            tx_data.columns = tx_data.iloc[0]
            tx_data = tx_data[1:].reset_index(drop=True)

            # Get columns (positional)
            cols = list(tx_data.columns)
            if len(cols) < 6:
                continue

            tx_id_col = cols[0] if len(cols) > 0 else None
            date_col = cols[1] if len(cols) > 1 else None
            debit_col = cols[2] if len(cols) > 2 else None  # Actually "Debit" (money out)
            credit_col = cols[3] if len(cols) > 3 else None  # Actually "Credit" (money in)
            desc_col = cols[5] if len(cols) > 5 else None

            # Parse types
            tx_data[date_col] = tx_data[date_col].apply(self._fix_date_vcb) if date_col else None
            tx_data[debit_col] = tx_data[debit_col].apply(self.fix_number) if debit_col else None
            tx_data[credit_col] = tx_data[credit_col].apply(self.fix_number) if credit_col else None

            # ========== Build Transactions ==========
            for _, row in tx_data.iterrows():
                # Skip rows with "Tổng số" or "Total"
                tx_id = self.to_text(row.get(tx_id_col, "")) if tx_id_col else ""
                if "TỔNG" in tx_id.upper() or "TOTAL" in tx_id.upper():
                    continue

                debit_val = row.get(debit_col) if debit_col else None
                credit_val = row.get(credit_col) if credit_col else None

                if (debit_val is None or pd.isna(debit_val) or debit_val == 0) and \
                   (credit_val is None or pd.isna(credit_val) or credit_val == 0):
                    continue

                tx = BankTransaction(
                    bank_name="VCB",
                    acc_no=acc_no or "",
                    debit=credit_val if pd.notna(credit_val) else None,  # SWAPPED
                    credit=debit_val if pd.notna(debit_val) else None,   # SWAPPED
                    date=row.get(date_col) if date_col and pd.notna(row.get(date_col)) else None,
                    description=self.to_text(row.get(desc_col, "")) if desc_col else "",
                    currency=currency or "VND",
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                all_transactions.append(tx)

        return all_transactions

    def _find_account_sections(self, sheet: pd.DataFrame) -> List[dict]:
        """Find all account section start/end indices."""
        all_rows = []
        for idx, row in sheet.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])
            all_rows.append((int(idx), row_text.upper()))

        # Find rows with "SỐ TÀI KHOẢN" and "ACCOUNT NUMBER"
        account_starts = []
        for idx, txt in all_rows:
            if "SỐ TÀI KHOẢN" in txt and "ACCOUNT NUMBER" in txt:
                account_starts.append(idx)

        if not account_starts:
            return [{'start': 0, 'end': len(all_rows) - 1}]

        # Create ranges
        sections = []
        for i, start_idx in enumerate(account_starts):
            end_idx = account_starts[i + 1] - 1 if i < len(account_starts) - 1 else len(all_rows) - 1
            sections.append({'start': start_idx, 'end': end_idx})

        return sections

    def _extract_account_number_vcb(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract account number from row 0 column 2 (typical VCB format)."""
        if len(top_df) > 0:
            first_row = top_df.iloc[0]
            if len(first_row) > 2:
                acc_text = self.to_text(first_row.iloc[2])
                digits = ''.join(c for c in acc_text if c.isdigit())
                if 10 <= len(digits) <= 13:
                    return digits

        return None

    def _extract_account_number_from_header(self, top_df: pd.DataFrame, marker: str) -> Optional[str]:
        """Extract account number from header using marker."""
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])
            if marker in row_text:
                digits = ''.join(c for c in row_text if c.isdigit())
                return digits if digits else None
        return None

    def _extract_currency_vcb(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract currency from header area."""
        known_currencies = ["VND", "USD", "EUR", "JPY", "CNY", "AUD", "SGD", "GBP"]

        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "LOẠI TIỀN" in row_text or "CURRENCY" in row_text:
                for currency in known_currencies:
                    if currency in row_text:
                        return currency

        return None

    def _extract_currency_from_header(self, top_df: pd.DataFrame) -> str:
        """Extract currency from simple template header (column 2 of currency row)."""
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])
            if "LOẠI TIỀN" in row_text and "CURRENCY" in row_text:
                if len(row) > 2:
                    currency_val = self.to_text(row.iloc[2]).strip().upper()
                    if currency_val:
                        return currency_val
        return "VND"

    def _extract_balance_by_label(self, section: pd.DataFrame, markers: list) -> Optional[float]:
        """Extract balance from row containing any of the markers."""
        for _, row in section.iterrows():
            row_text = " ".join([self.to_text(cell) for cell in row])

            for marker in markers:
                if marker in row_text:
                    # Try to extract number after ":"
                    if ":" in row_text:
                        after_colon = row_text.split(":", 1)[1]
                        num_val = self.fix_number(after_colon)
                        if num_val is not None:
                            return num_val

        return None

    def _extract_last_balance_from_grid(self, section: pd.DataFrame) -> Optional[float]:
        """Extract last balance from transaction grid (column 4)."""
        # Find header row with "NỢ" and "CÓ" and "SỐ DƯ"
        header_idx = None
        for idx, row in section.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])
            if "NỢ" in row_text and "CÓ" in row_text and "SỐ DƯ" in row_text:
                header_idx = int(idx)
                break

        if header_idx is None:
            return None

        # Find last balance before "Tổng số/Total"
        for idx in range(len(section) - 1, header_idx, -1):
            row = section.iloc[idx - section.index[0]]  # Adjust index
            row_text = "|".join([self.to_text(cell).upper() for cell in row])

            # Skip totals row
            if "TỔNG SỐ" in row_text or "TOTAL" in row_text:
                continue

            # Get balance from column 4
            if len(row) > 4:
                bal_val = self.fix_number(row.iloc[4])
                if bal_val is not None:
                    return bal_val

        return None

    def _fix_date_vcb(self, value) -> Optional[datetime]:
        """
        VCB-specific date parser.
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

        # Split if datetime string
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
