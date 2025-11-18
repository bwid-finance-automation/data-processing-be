"""MBB bank statement parser."""

import io
import re
from typing import List, Optional
import pandas as pd

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser


class MBBParser(BaseBankParser):
    """Parser for MBB (MB Bank - Military Commercial Joint Stock Bank) statements."""

    @property
    def bank_name(self) -> str:
        return "MBB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is MBB bank statement.

        Logic from fxLooksLike_MBB:
        - Read first 80 rows
        - Look for any of these markers:
          * "NGÂN HÀNG TMCP QUÂN ĐỘI"
          * "MBBANK"
          * "SAO KÊ CHI TIẾT TÀI KHOẢN"
          * "ACCOUNT STATEMENT"
          * ("TÀI KHOẢN/ACCOUNT NO" AND "SỐ DƯ CUỐI KỲ")
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

            txt = "|".join(all_text)

            # Check for MBB markers
            is_mbb = "NGÂN HÀNG TMCP QUÂN ĐỘI" in txt or \
                     "MBBANK" in txt or \
                     "SAO KÊ CHI TIẾT TÀI KHOẢN" in txt or \
                     "ACCOUNT STATEMENT" in txt or \
                     ("TÀI KHOẢN/ACCOUNT NO" in txt and "SỐ DƯ CUỐI KỲ" in txt)

            return is_mbb

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse MBB transactions.

        Logic from fxParse_MBB_Transactions:
        - Find header row with flexible bilingual detection
        - Primary date header: "NGÀY HẠCH TOÁN" or "POSTING DATE"
        - Fallback date header: "NGÀY GIAO DỊCH" or "TRANSACTION DATE"
        - Vietnamese/English column detection
        - Map: Debit = "PHÁT SINH CÓ" (CREDIT), Credit = "PHÁT SINH NỢ" (DEBIT)
        - Extract account number from "ACCOUNT NO:" line
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_80 = sheet.head(80)
            acc_no = self._extract_account_number_mbb(top_80)
            currency = self._extract_currency_mbb(top_80)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_mbb(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            # Primary date column
            date_col = self._find_any_column(data, ["NGÀY HẠCH TOÁN", "POSTING DATE"])
            if not date_col:
                # Fallback date column
                date_col = self._find_any_column(data, ["NGÀY GIAO DỊCH", "TRANSACTION DATE"])

            debit_col = self._find_any_column(data, ["PHÁT SINH CÓ", "CREDIT"])
            credit_col = self._find_any_column(data, ["PHÁT SINH NỢ", "DEBIT"])
            balance_col = self._find_any_column(data, ["SỐ DƯ", "BALANCE"])
            desc_col = self._find_any_column(data, ["NỘI DUNG", "DESCRIPTION", "DIỄN GIẢI"])

            # ========== Rename Columns ==========
            rename_map = {}
            if date_col: rename_map[date_col] = "Date"
            if debit_col: rename_map[debit_col] = "Debit"
            if credit_col: rename_map[credit_col] = "Credit"
            if balance_col: rename_map[balance_col] = "Balance"
            if desc_col: rename_map[desc_col] = "Description"

            if not rename_map:
                return []

            data = data.rename(columns=rename_map)

            # Keep only renamed columns
            keep_cols = ["Date", "Debit", "Credit", "Balance", "Description"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                return []

            data = data[available].copy()

            # ========== Parse Data Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self.fix_date)
            if "Debit" in data.columns:
                data["Debit"] = data["Debit"].apply(self._fix_number_mbb)
            if "Credit" in data.columns:
                data["Credit"] = data["Credit"].apply(self._fix_number_mbb)
            if "Balance" in data.columns:
                data["Balance"] = data["Balance"].apply(self._fix_number_mbb)

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
                    bank_name="MBB",
                    acc_no=acc_no or "",
                    debit=debit_val if pd.notna(debit_val) else None,
                    credit=credit_val if pd.notna(credit_val) else None,
                    date=row.get("Date") if "Date" in row and pd.notna(row.get("Date")) else None,
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency=currency or "VND",
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            return transactions

        except Exception as e:
            print(f"Error parsing MBB transactions: {e}")
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse MBB balance information.

        Logic from fxParse_MBB_Balances:
        - Closing = last non-null "Số dư" (excluding totals row - Date must not be null)
        - Opening = Closing - (Sum(Credit) - Sum(Debit))
        - Only count rows with non-null Date (to exclude totals row)
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet = pd.read_excel(xls, sheet_name=0, header=None)

            # ========== Extract Account Number and Currency from Header ==========
            top_80 = sheet.head(80)
            acc_no = self._extract_account_number_mbb(top_80)
            currency = self._extract_currency_mbb(top_80)

            # ========== Find Header Row ==========
            header_idx = self._find_header_row_mbb(top_80)

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Bilingual) ==========
            date_col = self._find_any_column(data, ["NGÀY HẠCH TOÁN", "POSTING DATE"])
            if not date_col:
                date_col = self._find_any_column(data, ["NGÀY GIAO DỊCH", "TRANSACTION DATE"])

            debit_col = self._find_any_column(data, ["PHÁT SINH NỢ", "DEBIT"])
            credit_col = self._find_any_column(data, ["PHÁT SINH CÓ", "CREDIT"])
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
                data[debit_col] = data[debit_col].apply(self._fix_number_mbb)
            if credit_col:
                data[credit_col] = data[credit_col].apply(self._fix_number_mbb)
            if balance_col:
                data[balance_col] = data[balance_col].apply(self._fix_number_mbb)

            # ========== Filter to Real Transaction Rows (Date not null) ==========
            if date_col:
                data_mov = data[data[date_col].notna()].copy()
            else:
                data_mov = data.copy()

            # ========== Calculate Balances ==========
            # Sum movements
            sum_debit = data_mov[debit_col].sum() if debit_col and debit_col in data_mov.columns else 0
            sum_credit = data_mov[credit_col].sum() if credit_col and credit_col in data_mov.columns else 0
            net_move = sum_credit - sum_debit

            # Closing = last non-null balance
            closing = None
            if balance_col and balance_col in data_mov.columns:
                bal_list = data_mov[balance_col].dropna().tolist()
                closing = bal_list[-1] if bal_list else None

            # Opening = Closing - NetMove
            opening = closing - net_move if closing is not None else None

            return BankBalance(
                bank_name="MBB",
                acc_no=acc_no or "",
                currency=currency or "VND",
                opening_balance=opening or 0.0,
                closing_balance=closing or 0.0
            )

        except Exception as e:
            print(f"Error parsing MBB balances: {e}")
            return None

    # ========== MBB-Specific Helper Methods ==========

    def _find_header_row_mbb(self, top_df: pd.DataFrame) -> int:
        """
        Find header row with flexible bilingual detection.
        Must contain:
        - (Date primary OR Date fallback) AND
        - (Balance OR (Debit AND Credit)) AND
        - Description
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self.to_text(cell).upper() for cell in row])

            has_date_primary = "NGÀY HẠCH TOÁN" in row_text or "POSTING DATE" in row_text
            has_date_fallback = "NGÀY GIAO DỊCH" in row_text or "TRANSACTION DATE" in row_text
            has_date = has_date_primary or has_date_fallback

            has_balance = "SỐ DƯ" in row_text or "BALANCE" in row_text
            has_debit = "PHÁT SINH NỢ" in row_text or "DEBIT" in row_text
            has_credit = "PHÁT SINH CÓ" in row_text or "CREDIT" in row_text
            has_amounts = has_balance or (has_debit and has_credit)

            has_desc = "NỘI DUNG" in row_text or "DESCRIPTION" in row_text or "DIỄN GIẢI" in row_text

            if has_date and has_amounts and has_desc:
                return int(idx)

        return 10  # Default fallback

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

    def _extract_account_number_mbb(self, top_df: pd.DataFrame) -> Optional[str]:
        """
        Extract account number from "ACCOUNT NO:" line.
        Looks for 8-20 digit token after the colon.
        """
        # Convert to rows
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "ACCOUNT NO" in row_text:
                # Get text after colon
                if ":" in row_text:
                    after_colon = row_text.split(":", 1)[1]
                else:
                    after_colon = row_text

                # Split by punctuation and spaces
                tokens = re.split(r'[ ,.;:|(){}\[\]<>\-_/]+', after_colon)
                tokens = [t for t in tokens if t]

                # Find first token with 8-20 digits
                for token in tokens:
                    digits = ''.join(c for c in token if c.isdigit())
                    if 8 <= len(digits) <= 20:
                        return digits

        return None

    def _extract_currency_mbb(self, top_df: pd.DataFrame) -> Optional[str]:
        """Extract currency from header area."""
        # Convert to rows
        for _, row in top_df.iterrows():
            row_text = " ".join([self.to_text(cell).upper() for cell in row])

            if "VND" in row_text:
                return "VND"
            elif "USD" in row_text:
                return "USD"

        return None

    def _fix_number_mbb(self, value) -> Optional[float]:
        """
        MBB-specific number parser.
        Removes: VND, commas, spaces
        Removes trailing .00
        """
        if value is None or pd.isna(value):
            return None

        txt = str(value).strip()
        if not txt:
            return None

        # Remove VND, commas, spaces
        txt = txt.replace("VND", "").replace(",", "").replace(" ", "")

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
