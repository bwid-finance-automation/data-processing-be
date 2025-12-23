"""OCB bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class OCBParser(BaseBankParser):
    """Parser for OCB (Orient Commercial Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "OCB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is OCB bank statement.

        Improved detection:
        - Check for "SAO KÊ TÀI KHOẢN/ ACCOUNT STATEMENT" (bilingual header)
        - OR check for OCB-specific structure: PS GIẢM/PS TĂNG + Starting Balance/Ending Balance
        - Structure-based detector (no bank-name keyword needed)
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 30 rows
            top_30 = df.head(30)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_30.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_30[col]])

            txt = " ".join(all_text)

            # Method 1: Check for OCB bilingual header pattern
            # "SAO KÊ TÀI KHOẢN/ ACCOUNT STATEMENT" is unique to OCB
            has_ocb_header = "SAO KÊ TÀI KHOẢN/ ACCOUNT STATEMENT" in txt

            # Method 2: Check for OCB-specific column markers
            has_ps_giam = "PS GIẢM" in txt or "PS NỢ" in txt
            has_ps_tang = "PS TĂNG" in txt or "PS CÓ" in txt
            has_starting_balance = "STARTING BALANCE" in txt or "SỐ DƯ ĐẦU KỲ/ STARTING BALANCE" in txt
            has_ending_balance = "ENDING BALANCE" in txt or "SỐ DƯ CUỐI KỲ/ ENDING BALANCE" in txt
            has_ocb_structure = (has_ps_giam and has_ps_tang) or (has_starting_balance and has_ending_balance)

            # Method 3: Check for specific OCB markers not in other banks
            has_debit_total = "DEBIT TOTAL" in txt or "TỔNG PS NỢ" in txt
            has_credit_total = "CREDIT TOTAL" in txt or "TỔNG PS CÓ" in txt

            is_ocb = has_ocb_header or (has_ocb_structure and (has_debit_total or has_credit_total))

            return is_ocb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse OCB transactions.

        Logic from fxParse_OCB_Transactions:
        - Find header row with flexible bilingual detection
        - Vietnamese column headers
        - ERP Convention: Debit = tiền VÀO (PS TĂNG), Credit = tiền RA (PS GIẢM)
        - Extract account number from "SỐ TÀI KHOẢN" line with sophisticated tokenization
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_80 = sheet.head(80)
            acc_no = self._extract_account_number_ocb(top_80)
            currency = self._extract_currency_ocb(top_80)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_ocb(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            # ERP Convention (SWAPPED):
            # - "PS GIẢM (NỢ)" = tiền RA = Credit in ERP
            # - "PS TĂNG (CÓ)" = tiền VÀO = Debit in ERP
            date_col = self._find_any_column(data, ["NGÀY THỰC HIỆN", "TRANSACTION DATE", "NGÀY", "DATE"])
            credit_col = self._find_any_column(data, ["PS GIẢM", "PS NỢ", "DEBIT"])  # Money out = Credit (ERP)
            debit_col = self._find_any_column(data, ["PS TĂNG", "PS CÓ", "CREDIT"])  # Money in = Debit (ERP)
            balance_col = self._find_any_column(data, ["SỐ DƯ", "BALANCE"])
            desc_col = self._find_any_column(data, ["NỘI DUNG", "DESCRIPTION", "DIỄN GIẢI", "CONTENT"])
            transaction_id_col = self._find_any_column(data, ["SỐ GD", "TRANSACTION NUMBER"])

            # ========== Rename Columns ==========
            rename_map = {}
            if date_col: rename_map[date_col] = "Date"
            if debit_col: rename_map[debit_col] = "Debit"
            if credit_col: rename_map[credit_col] = "Credit"
            if balance_col: rename_map[balance_col] = "Balance"
            if desc_col: rename_map[desc_col] = "Description"
            if transaction_id_col: rename_map[transaction_id_col] = "TransactionID"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["Date", "Debit", "Credit", "Balance", "Description", "TransactionID"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Data Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self.fix_date)
            if "Debit" in data.columns:
                data["Debit"] = data["Debit"].apply(self._fix_number_ocb)
            if "Credit" in data.columns:
                data["Credit"] = data["Credit"].apply(self._fix_number_ocb)
            if "Balance" in data.columns:
                data["Balance"] = data["Balance"].apply(self._fix_number_ocb)

            # ========== Parse Transactions ==========
            transactions = []

            for _, row in data.iterrows():
                # Skip rows without a valid date (filters out summary rows)
                date_val = row.get("Date") if "Date" in row else None
                if date_val is None or pd.isna(date_val):
                    continue

                debit_val = row.get("Debit") if "Debit" in row else None
                credit_val = row.get("Credit") if "Credit" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or (isinstance(debit_val, float) and pd.isna(debit_val)) or debit_val == 0) and \
                   (credit_val is None or (isinstance(credit_val, float) and pd.isna(credit_val)) or credit_val == 0):
                    continue

                tx = BankTransaction(
                    bank_name="OCB",
                    acc_no=acc_no or "",
                    debit=debit_val if pd.notna(debit_val) else None,
                    credit=credit_val if pd.notna(credit_val) else None,
                    date=date_val,  # Already validated above
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency=currency or "VND",
                    transaction_id=self.to_text(row.get("TransactionID", "")) if "TransactionID" in row else "",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing OCB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse OCB balance information.

        Logic from fxParse_OCB_Balances:
        - Opening from "SỐ DƯ ĐẦU:" text extraction
        - Closing = last balance + net movement calculation
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_80 = sheet.head(80)
            acc_no = self._extract_account_number_ocb(top_80)
            currency = self._extract_currency_ocb(top_80)

            # ========== Extract Opening Balance from Header Text ==========
            opening = self._extract_opening_balance_ocb(top_80)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_ocb(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns ==========
            # ERP Convention (SWAPPED):
            # - "PS GIẢM (NỢ)" = tiền RA = Credit in ERP
            # - "PS TĂNG (CÓ)" = tiền VÀO = Debit in ERP
            date_col = self._find_any_column(data, ["NGÀY", "DATE"])
            credit_col = self._find_any_column(data, ["PS GIẢM", "PS NỢ", "DEBIT"])  # Money out = Credit (ERP)
            debit_col = self._find_any_column(data, ["PS TĂNG", "PS CÓ", "CREDIT"])  # Money in = Debit (ERP)
            balance_col = self._find_any_column(data, ["SỐ DƯ", "BALANCE"])

            # Keep columns
            keep_cols = []
            if date_col: keep_cols.append(date_col)
            if debit_col: keep_cols.append(debit_col)
            if credit_col: keep_cols.append(credit_col)
            if balance_col: keep_cols.append(balance_col)

            if not keep_cols:
                return None

            data = data[keep_cols].copy()

            # ========== Parse Data Types ==========
            if date_col:
                data[date_col] = data[date_col].apply(self.fix_date)
            if debit_col:
                data[debit_col] = data[debit_col].apply(self._fix_number_ocb)
            if credit_col:
                data[credit_col] = data[credit_col].apply(self._fix_number_ocb)
            if balance_col:
                data[balance_col] = data[balance_col].apply(self._fix_number_ocb)

            # ========== Filter to Real Transaction Rows (Date not null) ==========
            if date_col:
                data_mov = data[data[date_col].notna()].copy()
            else:
                data_mov = data.copy()

            # ========== Calculate Balances ==========
            # Sum movements
            sum_debit = data_mov[debit_col].sum() if debit_col and debit_col in data_mov.columns else 0
            sum_credit = data_mov[credit_col].sum() if credit_col and credit_col in data_mov.columns else 0
            net_move = sum_debit - sum_credit

            # Closing = Opening + NetMove
            closing = opening + net_move if opening is not None else None

            # Alternatively, use last balance if available
            if balance_col and balance_col in data_mov.columns:
                bal_list = data_mov[balance_col].dropna().tolist()
                if bal_list:
                    closing = bal_list[-1]

            return BankBalance(
                bank_name="OCB",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing OCB balances: {e}")
            return None

    # ========== OCB-Specific Helper Methods ==========

    def _find_header_row_ocb(self, top_df: pd.DataFrame) -> int:
        """
        Find header row with flexible detection.
        Must contain: NGÀY + NỘI DUNG + (PS GIẢM OR PS NỢ) + (PS TĂNG OR PS CÓ)
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])

            has_date = "NGÀY" in row_text
            has_desc = "NỘI DUNG" in row_text or "DIỄN GIẢI" in row_text
            has_credit = "PS GIẢM" in row_text or "PS NỢ" in row_text
            has_debit = "PS TĂNG" in row_text or "PS CÓ" in row_text

            if has_date and has_desc and has_credit and has_debit:
                return int(idx)

        return 10  # Default fallback

    def _find_any_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find first column matching any of the candidate names (case-insensitive)."""
        columns = df.columns.tolist()
        # Normalize: replace newlines with spaces, then uppercase
        cols_upper = [str(col).replace('\n', ' ').upper() for col in columns]
        cands_upper = [c.upper() for c in candidates]

        for cand in cands_upper:
            for i, col_up in enumerate(cols_upper):
                if cand in col_up:
                    return columns[i]

        return None

    def _extract_account_number_ocb(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from header area.

        Flexible detection:
        1. Look for rows with account keywords (SỐ TÀI KHOẢN, ACCOUNT NO, TK, STK, etc.)
        2. Look for cells containing only 6-20 digit numbers (likely account numbers)
        3. Vietnamese bank accounts typically have 6-20 digits (OCB can have 16 digits).
        """
        # Keywords to search for account number context
        account_keywords = [
            "SỐ TÀI KHOẢN", "ACCOUNT NO", "ACCOUNT NUMBER", "ACC NO", "A/C NO",
            "TÀI KHOẢN", "TK:", "STK:", "SỐ TK", "SO TK", "ACCOUNT:",
            "TK SỐ", "SỐ HIỆU TK", "MÃ TK"
        ]

        # First pass: Look for rows with account keywords
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if any(kw in row_text for kw in account_keywords):
                # Split by punctuation and spaces
                tokens = re.split(r'[ ,.;:|(){}\\[\\]<>\\-_/]+', row_text)
                tokens = [t for t in tokens if t]

                # Find the LONGEST digit sequence (OCB can have 16 digit account numbers)
                # This ensures we pick 0010100013992005 over shorter numbers like 2330640
                best_digits = ""
                for token in tokens:
                    digits = ''.join(c for c in token if c.isdigit())
                    if 6 <= len(digits) <= 20:
                        if len(digits) > len(best_digits):
                            best_digits = digits
                if best_digits:
                    return best_digits

        # Second pass: Look for standalone cells with 6-20 digit numbers
        # These are likely account numbers in header area
        for _, row in top_df.iterrows():
            for cell in row:
                cell_text = self.to_text(cell).strip()
                # Check if cell is purely numeric (or with minor separators)
                cleaned = cell_text.replace(" ", "").replace("-", "").replace(".", "")
                if cleaned.isdigit() and 6 <= len(cleaned) <= 20:
                    # Exclude dates (format like 20241105) and amounts (too long or with decimals)
                    if not re.match(r'^\d{8}$', cleaned):  # Exclude YYYYMMDD dates
                        return cleaned

        return None

    def _extract_currency_ocb(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract currency from header area.

        Flexible detection:
        1. Look for currency keywords (VND, USD, EUR, etc.)
        2. Look for currency context words (LOẠI TIỀN, CURRENCY, etc.)
        """
        currency_keywords = {
            "VND": ["VND", "VNĐ", "ĐỒNG", "DONG"],
            "USD": ["USD", "US DOLLAR", "DOLLAR"],
            "EUR": ["EUR", "EURO"],
            "JPY": ["JPY", "YEN"],
            "GBP": ["GBP", "POUND"]
        }

        # Convert to rows and search
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            # Check for each currency
            for currency, keywords in currency_keywords.items():
                if any(kw in row_text for kw in keywords):
                    return currency

        # Default to VND for Vietnamese banks
        return "VND"

    def _extract_opening_balance_ocb(self, top_df: pd.DataFrame) -> Optional[float]:
        """
        Extract opening balance from header area.

        Flexible detection:
        1. Look for opening balance keywords (SỐ DƯ ĐẦU, OPENING BALANCE, etc.)
        2. Extract number from the same row or adjacent cell
        """
        opening_keywords = [
            "SỐ DƯ ĐẦU", "DƯ ĐẦU", "OPENING BALANCE", "OPENING BAL",
            "SỐ DƯ ĐẦU KỲ", "SỐ DƯ MỞ", "BEGIN BALANCE", "BALANCE B/F",
            "SỐ DƯ NGÀY", "BALANCE BROUGHT"
        ]

        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if any(kw in row_text for kw in opening_keywords):
                # Try to extract number from this row
                for cell in row:
                    num_val = self._fix_number_ocb(cell)
                    if num_val is not None and num_val != 0:
                        return num_val

        return None

    def _fix_number_ocb(self, value) -> Optional[float]:
        """
        OCB-specific number parser.
        Removes: VND, commas, spaces
        Removes trailing .00
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Remove VND, commas, spaces
        txt = txt.replace("VND", "").replace("VNĐ", "").replace(",", "").replace(" ", "")

        # Remove trailing .00
        if txt.endswith(".00"):
            txt = txt[:-3]

        # Extract digits and decimal
        digits = ''.join(c for c in txt if c.isdigit() or c in ['-', '.'])
        if not digits:
            return None

        try:
            return float(digits)
        except (ValueError, TypeError):
            return None
