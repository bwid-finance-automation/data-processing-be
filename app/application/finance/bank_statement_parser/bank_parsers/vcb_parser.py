"""VCB bank statement parser (Vietcombank)."""

import io
import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class VCBParser(BaseBankParser):
    """Parser for VCB (Vietcombank) statements - handles multiple templates."""

    @property
    def bank_name(self) -> str:
        return "VCB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is VCB bank statement.

        Supported templates:
        - Template 1 (CashUp): Has "VCB CASHUP" or "SAO KÊ TÀI KHOẢN STATEMENT OF ACCOUNT"
        - Template 2 (Simple): Has "NGÀY GIAO DỊCH" + "SỐ THAM CHIẾU" + "SỐ TIỀN"
        - Template 3 (HTML): Has "SAO KÊ TÀI KHOẢN" and HTML tags
        - Template 4 (Multi): Has "STATEMENT OF ACCOUNT" with account sections
        """
        try:
            # Check if HTML file
            if self._is_html_file(file_bytes):
                content = file_bytes.decode('utf-8', errors='ignore')
                return "SAO KÊ TÀI KHOẢN" in content and "VIETCOMBANK" in content.upper()

            # Excel file
            xls = self.get_excel_file(file_bytes)

            # Check sheet name for VCB CashUp format
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            if "VCBACCOUNTDETAIL" in sheet_name.upper():
                return True

            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 20 rows
            top_20 = df.head(20)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_20.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_20[col]])

            txt = " ".join(all_text)

            # Check for VCB markers
            # Template 1: VCB CashUp format with bilingual header
            has_cashup = "VCB CASHUP" in txt
            has_sao_ke_statement = "SAO KÊ TÀI KHOẢN STATEMENT OF ACCOUNT" in txt
            has_statement_of_account = "STATEMENT OF ACCOUNT" in txt

            is_vcb = has_cashup or has_sao_ke_statement or has_statement_of_account

            return is_vcb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse VCB transactions - handles multiple templates and HTML.

        Templates:
        - CashUp: Sheet name contains "VCBACCOUNTDETAIL", has merged date/doc column
        - Simple: Has "Ngày giao dịch|Số tham chiếu|Số tiền" pattern
        - Multi: Multiple account sections with "SỐ TÀI KHOẢN/ACCOUNT NUMBER"
        - HTML: HTML table format

        Column mapping (SWAPPED):
        - VCB "Số tiền ghi nợ/Debit" = money OUT = our Credit
        - VCB "Số tiền ghi có/Credit" = money IN = our Debit
        """
        try:
            # Check if HTML file
            if self._is_html_file(file_bytes):
                return self._parse_html_template(file_bytes)

            # Excel file
            xls = self.get_excel_file(file_bytes)

            # Check for CashUp template (VCBACCOUNTDETAIL sheet name)
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            if "VCBACCOUNTDETAIL" in sheet_name.upper():
                return self._parse_cashup_template(xls)

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
            # Check if HTML file
            if self._is_html_file(file_bytes):
                return self._parse_html_balances(file_bytes)

            # Excel file
            xls = self.get_excel_file(file_bytes)

            # Check for CashUp template (VCBACCOUNTDETAIL sheet name)
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            if "VCBACCOUNTDETAIL" in sheet_name.upper():
                return self._parse_cashup_balances(xls)

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

    def parse_all_balances_from_bytes(self, file_bytes: bytes, file_name: str) -> List[BankBalance]:
        """
        Parse ALL account balances from VCB file (multi-account support).

        Returns list of BankBalance objects, one per account found.
        """
        try:
            # Check if HTML file
            if self._is_html_file(file_bytes):
                bal = self._parse_html_balances(file_bytes)
                return [bal] if bal else []

            # Excel file
            xls = self.get_excel_file(file_bytes)

            # Check for CashUp template (VCBACCOUNTDETAIL sheet name)
            sheet_name = xls.sheet_names[0] if xls.sheet_names else ""
            if "VCBACCOUNTDETAIL" in sheet_name.upper():
                return self._parse_all_cashup_balances(xls)

            # Default: single balance
            bal = self.parse_balances(file_bytes, file_name)
            return [bal] if bal else []

        except Exception as e:
            print(f"Error parsing VCB all balances: {e}")
            return []

    def _parse_cashup_balances(self, xls: pd.ExcelFile) -> Optional[BankBalance]:
        """Parse balance from first account in CashUp template."""
        balances = self._parse_all_cashup_balances(xls)
        return balances[0] if balances else None

    def _parse_all_cashup_balances(self, xls: pd.ExcelFile) -> List[BankBalance]:
        """
        Parse ALL account balances from CashUp template.

        Balance format:
        - Row with "Số dư đầu kỳ/ Opening balance" has value in column 2 (e.g., "3,359,662,231 VND")
        - Row with "Số dư cuối kỳ/ Closing balance" has value in column 2
        """
        sheet = pd.read_excel(xls, sheet_name=0, header=None)
        all_balances = []

        # Find all account sections
        # Note: The text may have newline between Vietnamese and English parts
        section_starts = []
        for idx, row in sheet.iterrows():
            row_text = " ".join([self.to_text(cell).replace('\n', ' ') for cell in row])
            if "SAO KÊ TÀI KHOẢN" in row_text and "STATEMENT OF ACCOUNT" in row_text:
                section_starts.append(int(idx))

        if not section_starts:
            section_starts = [0]

        # Create section ranges
        sections = []
        for i, start_idx in enumerate(section_starts):
            end_idx = section_starts[i + 1] - 1 if i < len(section_starts) - 1 else len(sheet) - 1
            sections.append({'start': start_idx, 'end': end_idx})

        # Parse each account section
        for section in sections:
            section_rows = sheet.iloc[section['start']:section['end'] + 1]

            # Extract account number
            acc_no = None
            for _, row in section_rows.head(15).iterrows():
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Số tài khoản" in row_text or "Account number" in row_text:
                    if len(row) > 2:
                        acc_text = self.to_text(row.iloc[2])
                        digits = ''.join(c for c in acc_text if c.isdigit())
                        if 8 <= len(digits) <= 15:
                            acc_no = digits
                            break

            # Extract currency
            currency = "VND"
            for _, row in section_rows.head(15).iterrows():
                row_text = " ".join([self.to_text(cell).upper() for cell in row])
                if "LOẠI TIỀN" in row_text or "CURRENCY" in row_text:
                    if len(row) > 2:
                        curr_val = self.to_text(row.iloc[2]).upper().strip()
                        if curr_val in ["VND", "USD", "EUR", "JPY", "CNY"]:
                            currency = curr_val
                            break

            # Extract opening balance (row with "Số dư đầu kỳ/ Opening balance")
            opening = None
            for _, row in section_rows.head(20).iterrows():
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Số dư đầu kỳ" in row_text or "Opening balance" in row_text:
                    if len(row) > 2:
                        # Value is in column 2, may have currency suffix
                        val_text = self.to_text(row.iloc[2])
                        opening = self.fix_number(val_text)
                        break

            # Extract closing balance (row with "Số dư cuối kỳ/ Closing balance")
            closing = None
            for _, row in section_rows.iterrows():
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Số dư cuối kỳ" in row_text or "Closing balance" in row_text:
                    if len(row) > 2:
                        val_text = self.to_text(row.iloc[2])
                        closing = self.fix_number(val_text)
                        break

            if acc_no:
                all_balances.append(BankBalance(
                    bank_name="VCB",
                    acc_no=acc_no,
                    currency=currency,
                    opening_balance=opening or 0.0,
                    closing_balance=closing or 0.0
                ))

        return all_balances

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

    def _parse_cashup_template(self, xls: pd.ExcelFile) -> List[BankTransaction]:
        """
        Parse VCB CashUp Template (VCBACCOUNTDETAILBASEXLS format).

        This template has:
        - Multiple account sections, each starting with "SAO KÊ TÀI KHOẢN STATEMENT OF ACCOUNT"
        - Merged date/doc column: "DD/MM/YYYY                       DOCNO"
        - 6 columns: STT, Date+Doc, Debit, Credit, Balance, Description
        - Bilingual labels throughout
        """
        sheet = pd.read_excel(xls, sheet_name=0, header=None)
        all_transactions = []

        # Find all account sections by looking for "SAO KÊ TÀI KHOẢN" and "STATEMENT OF ACCOUNT"
        # Note: The text may have newline between Vietnamese and English parts
        section_starts = []
        for idx, row in sheet.iterrows():
            row_text = " ".join([self.to_text(cell).replace('\n', ' ') for cell in row])
            if "SAO KÊ TÀI KHOẢN" in row_text and "STATEMENT OF ACCOUNT" in row_text:
                section_starts.append(int(idx))

        if not section_starts:
            # Fallback: treat entire sheet as one section
            section_starts = [0]

        # Create section ranges
        sections = []
        for i, start_idx in enumerate(section_starts):
            end_idx = section_starts[i + 1] - 1 if i < len(section_starts) - 1 else len(sheet) - 1
            sections.append({'start': start_idx, 'end': end_idx})

        # Parse each account section
        for section in sections:
            section_rows = sheet.iloc[section['start']:section['end'] + 1]

            # ========== Extract Account Number ==========
            acc_no = None
            for _, row in section_rows.head(15).iterrows():
                row_text = " ".join([self.to_text(cell) for cell in row])
                if "Số tài khoản" in row_text or "Account number" in row_text:
                    # Account number is in column 2
                    if len(row) > 2:
                        acc_text = self.to_text(row.iloc[2])
                        digits = ''.join(c for c in acc_text if c.isdigit())
                        if 8 <= len(digits) <= 15:
                            acc_no = digits
                            break

            # ========== Extract Currency ==========
            currency = "VND"
            for _, row in section_rows.head(15).iterrows():
                row_text = " ".join([self.to_text(cell).upper() for cell in row])
                if "LOẠI TIỀN" in row_text or "CURRENCY" in row_text:
                    if len(row) > 2:
                        curr_val = self.to_text(row.iloc[2]).upper().strip()
                        if curr_val in ["VND", "USD", "EUR", "JPY", "CNY"]:
                            currency = curr_val
                            break

            # ========== Find Transaction Header Row ==========
            header_idx = None
            for idx, row in section_rows.iterrows():
                row_text = "|".join([self.to_text(cell).upper() for cell in row])
                # Look for header with STT and Debit/Credit columns
                if ("STT" in row_text or "NO" in row_text) and \
                   ("GHI NỢ" in row_text or "DEBIT" in row_text) and \
                   ("GHI CÓ" in row_text or "CREDIT" in row_text):
                    header_idx = int(idx)
                    break

            if header_idx is None:
                continue

            # ========== Parse Transactions ==========
            # Start from row after header
            for idx in range(header_idx + 1, section['end'] + 1):
                if idx >= len(sheet):
                    break

                row = sheet.iloc[idx]
                row_text = " ".join([self.to_text(cell) for cell in row])

                # Skip empty rows
                if not row_text.strip():
                    continue

                # Stop at totals row
                if "Tổng số" in row_text or "Total" in row_text:
                    break

                # Skip non-data rows (must have STT number in col 0)
                stt_val = self.to_text(row.iloc[0]).strip() if len(row) > 0 else ""
                if not stt_val or not stt_val.isdigit():
                    continue

                # Column structure:
                # Col 0: STT (sequence number)
                # Col 1: "DD/MM/YYYY                       DOCNO" (merged date + doc)
                # Col 2: Số tiền ghi nợ/Debit (money OUT = our Credit)
                # Col 3: Số tiền ghi có/Credit (money IN = our Debit)
                # Col 4: Số dư/Balance
                # Col 5: Nội dung/Description

                # Parse date from merged column
                date_doc_val = self.to_text(row.iloc[1]) if len(row) > 1 else ""
                date_val = None
                tx_id = ""
                if date_doc_val:
                    # Extract date (first 10 chars: DD/MM/YYYY)
                    date_match = re.match(r'^(\d{2}/\d{2}/\d{4})', date_doc_val.strip())
                    if date_match:
                        date_val = self._fix_date_vcb(date_match.group(1))
                    # Extract doc number (after spaces)
                    doc_match = re.search(r'\s{2,}(\S+)$', date_doc_val)
                    if doc_match:
                        tx_id = doc_match.group(1)

                # Get amounts (SWAPPED)
                debit_val = self.fix_number(row.iloc[2]) if len(row) > 2 else None  # Money out = our Credit
                credit_val = self.fix_number(row.iloc[3]) if len(row) > 3 else None  # Money in = our Debit

                # Skip if both are zero/None
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    continue

                # Get description
                desc = self.to_text(row.iloc[5]) if len(row) > 5 else ""

                tx = BankTransaction(
                    bank_name="VCB",
                    acc_no=acc_no or "",
                    debit=credit_val if credit_val else None,  # SWAPPED: Ghi có = money in = our Debit
                    credit=debit_val if debit_val else None,   # SWAPPED: Ghi nợ = money out = our Credit
                    date=date_val,
                    description=desc,
                    currency=currency,
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                all_transactions.append(tx)

        return all_transactions

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
        Handles DD/MM/YYYY format, YYYY-MM-DD format, and datetime strings.
        """
        if value is None or pd.isna(value):
            return None

        # Try parsing text first
        txt = str(value).strip()
        if not txt:
            return None

        # Split if datetime string
        if " " in txt:
            txt = txt.split(" ")[0]

        # Try DD/MM/YYYY format (most common VCB format)
        try:
            return datetime.strptime(txt, "%d/%m/%Y").date()
        except:
            pass

        # Try YYYY-MM-DD format (HTML format)
        try:
            return datetime.strptime(txt, "%Y-%m-%d").date()
        except:
            pass

        # Try as Excel date number (fallback)
        try:
            return pd.to_datetime(value, dayfirst=True).date()
        except:
            pass

        # Fallback to pandas
        try:
            return pd.to_datetime(txt, dayfirst=True).date()
        except:
            return None

    def _is_html_file(self, file_bytes: bytes) -> bool:
        """Check if file is HTML format."""
        try:
            # Check first 100 bytes for HTML markers
            header = file_bytes[:100].decode('utf-8', errors='ignore').lower()
            return '<html' in header or '<!doctype html' in header
        except:
            return False

    def _parse_html_template(self, file_bytes: bytes) -> List[BankTransaction]:
        """Parse VCB HTML template transactions."""
        try:
            # Read HTML tables
            from io import StringIO
            content = file_bytes.decode('utf-8', errors='ignore')
            tables = pd.read_html(StringIO(content))

            if not tables:
                return []

            # The main table contains all data
            df = tables[0]

            # Extract account number (row 3, column 1)
            acc_no = None
            if len(df) > 3 and len(df.columns) > 1:
                acc_text = str(df.iloc[3, 1])
                acc_no = ''.join(c for c in acc_text if c.isdigit())

            # Extract currency (row 6, column 1)
            currency = "VND"
            if len(df) > 6 and len(df.columns) > 1:
                currency_text = str(df.iloc[6, 1]).upper()
                if any(curr in currency_text for curr in ["VND", "USD", "EUR"]):
                    currency = currency_text.strip()

            # Find transaction header row (contains "Ngày giao dịch")
            header_idx = None
            for idx, row in df.iterrows():
                row_text = " ".join([str(cell) for cell in row if pd.notna(cell)]).upper()
                if "NGÀY GIAO DỊCH" in row_text:
                    header_idx = int(idx)
                    break

            if header_idx is None:
                return []

            # Extract transactions
            transactions = []
            for idx in range(header_idx + 1, len(df)):
                row = df.iloc[idx]

                # Stop at total row
                row_text = " ".join([str(cell) for cell in row if pd.notna(cell)]).upper()
                if "TỔNG SỐ" in row_text or "TOTAL" in row_text:
                    break

                # Get transaction data
                if len(row) < 5:
                    continue

                date_val = row.iloc[0] if pd.notna(row.iloc[0]) else None
                tx_id = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
                debit_val = row.iloc[2] if pd.notna(row.iloc[2]) else None  # "Số tiền ghi nợ" = money out = credit in our system
                credit_val = row.iloc[3] if pd.notna(row.iloc[3]) else None  # "Số tiền ghi có" = money in = debit in our system
                desc = str(row.iloc[4]) if pd.notna(row.iloc[4]) else ""

                # Skip if both amounts are zero/blank
                debit_num = self.fix_number(debit_val)
                credit_num = self.fix_number(credit_val)

                if (debit_num is None or debit_num == 0) and (credit_num is None or credit_num == 0):
                    continue

                tx = BankTransaction(
                    bank_name="VCB",
                    acc_no=acc_no or "",
                    debit=credit_num if credit_num else None,  # SWAPPED: Ghi có = money in = debit
                    credit=debit_num if debit_num else None,  # SWAPPED: Ghi nợ = money out = credit
                    date=self._fix_date_vcb(date_val),
                    description=desc,
                    currency=currency,
                    transaction_id=tx_id,
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing VCB HTML transactions: {e}")
            return []

    def _parse_html_balances(self, file_bytes: bytes) -> Optional[BankBalance]:
        """Parse VCB HTML template balances."""
        try:
            # Read HTML tables
            from io import StringIO
            content = file_bytes.decode('utf-8', errors='ignore')
            tables = pd.read_html(StringIO(content))

            if not tables:
                return None

            df = tables[0]

            # Extract account number (row 3, column 1)
            acc_no = None
            if len(df) > 3 and len(df.columns) > 1:
                acc_text = str(df.iloc[3, 1])
                acc_no = ''.join(c for c in acc_text if c.isdigit())

            # Extract currency (row 6, column 1)
            currency = "VND"
            if len(df) > 6 and len(df.columns) > 1:
                currency_text = str(df.iloc[6, 1]).upper()
                if any(curr in currency_text for curr in ["VND", "USD", "EUR"]):
                    currency = currency_text.strip()

            # Find balance row (contains "Số dư đầu kỳ" and "Số dư cuối kỳ")
            opening = None
            closing = None

            for idx, row in df.iterrows():
                row_text = " ".join([str(cell) for cell in row if pd.notna(cell)]).upper()
                if "SỐ DƯ ĐẦU KỲ" in row_text:
                    # Opening balance is in column 1, closing in column 3
                    if len(row) > 1:
                        opening = self.fix_number(row.iloc[1])
                    if len(row) > 3:
                        closing = self.fix_number(row.iloc[3])
                    break

            return BankBalance(
                bank_name="VCB",
                acc_no=acc_no or "",
                currency=currency,
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing VCB HTML balances: {e}")
            return None
