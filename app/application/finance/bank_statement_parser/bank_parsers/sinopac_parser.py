"""SINOPAC bank statement parser."""

import io
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union
import pandas as pd
import math
import unicodedata

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from app.shared.utils.logging_config import get_logger
from .base_parser import BaseBankParser

logger = get_logger(__name__)


class SINOPACParser(BaseBankParser):
    """Parser for SINOPAC (SinoPac Bank / 永豐銀行) statements."""

    @property
    def bank_name(self) -> str:
        return "SINOPAC"

    def _is_erp_template(self, file_bytes: bytes) -> bool:
        """Check if file is ERP template format with SINOPAC data."""
        try:
            xls = self.get_excel_file(file_bytes)
            sheet_names_lower = [s.strip().lower() for s in xls.sheet_names]
            logger.info(f"[SNP] _is_erp_template: sheets={xls.sheet_names}")
            if "template balance" not in sheet_names_lower and "template details" not in sheet_names_lower:
                logger.info("[SNP] _is_erp_template: no ERP template sheets found")
                return False

            # Find actual sheet name (preserving case)
            detail_sheet = None
            balance_sheet = None
            for s in xls.sheet_names:
                if s.strip().lower() == "template details":
                    detail_sheet = s
                elif s.strip().lower() == "template balance":
                    balance_sheet = s

            logger.info(f"[SNP] _is_erp_template: detail_sheet={detail_sheet}, balance_sheet={balance_sheet}")

            # Check bank code in either sheet
            check_sheet = detail_sheet or balance_sheet
            if not check_sheet:
                return False

            df = pd.read_excel(xls, sheet_name=check_sheet, header=0, nrows=10)
            # Look for bank code column
            for col in df.columns:
                col_upper = str(col).upper()
                if "BANK CODE" in col_upper or "BANK_CODE" in col_upper:
                    bank_codes = df[col].dropna().astype(str).str.upper()
                    matched = bank_codes.str.contains("SINOPAC|SNP").any()
                    logger.info(f"[SNP] _is_erp_template: bank_codes={bank_codes.tolist()}, matched={matched}")
                    if matched:
                        return True
            return False
        except Exception as e:
            logger.warning(f"[SNP] _is_erp_template error: {e}")
            return False

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is SINOPAC bank statement.

        Supports three formats:
        1. Raw Vietnamese format
        2. Raw English format ("Transaction details report")
        3. ERP template format: "Template balance" + "Template details" sheets
           with Bank Code = SINOPAC
        """
        logger.info("[SNP] can_parse: checking file...")

        # Check ERP template format first
        if self._is_erp_template(file_bytes):
            logger.info("[SNP] can_parse: detected ERP template format")
            return True

        # Check raw statement format
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 40 rows
            top_40 = df.head(40)

            # Flatten all cells to normalized text (accent/punctuation insensitive)
            all_text = []
            for col in top_40.columns:
                all_text.extend([self._normalize_sinopac_text(cell) for cell in top_40[col]])

            row_text = "|".join(all_text)
            sheet_names = {self._normalize_sinopac_text(name) for name in xls.sheet_names}

            # Vietnamese markers
            has_can_doi = "CAN DOI" in row_text
            has_ngay_gia_tri = "NGAY GIA TRI" in row_text
            has_so_tai_khoan = "SO TAI KHOAN" in row_text
            has_nhan_xet = "NHAN XET" in row_text

            # English markers
            has_txn_report_sheet = "TRANSACTION DETAILS REPORT" in sheet_names
            has_account_no = "ACCOUNT NO" in row_text
            has_value_date = "VALUE DATE" in row_text
            has_remarks = "REMARKS" in row_text
            has_withdrawal = "WITHDRAWAL" in row_text
            has_deposit = "DEPOSIT" in row_text
            has_client_header = "CLIENT NO ACCOUNT NAME" in row_text
            has_date_of_data = "DATE OF DATA" in row_text

            logger.info(
                f"[SNP] can_parse Vietnamese markers: "
                f"CAN_DOI={has_can_doi}, NGAY_GIA_TRI={has_ngay_gia_tri}, "
                f"SO_TAI_KHOAN={has_so_tai_khoan}, NHAN_XET={has_nhan_xet}"
            )
            logger.info(
                f"[SNP] can_parse English markers: "
                f"TXN_REPORT_SHEET={has_txn_report_sheet}, ACCOUNT_NO={has_account_no}, "
                f"VALUE_DATE={has_value_date}, WITHDRAWAL={has_withdrawal}, "
                f"DEPOSIT={has_deposit}, REMARKS={has_remarks}, "
                f"CLIENT_HEADER={has_client_header}, DATE_OF_DATA={has_date_of_data}"
            )

            is_vietnamese_raw = (
                (has_can_doi and has_ngay_gia_tri)
                or (has_so_tai_khoan and has_nhan_xet)
            )
            is_english_raw = (
                has_txn_report_sheet
                and has_account_no
                and has_value_date
                and (has_withdrawal or has_deposit)
                and (has_remarks or has_client_header or has_date_of_data)
            )
            is_sinopac = is_vietnamese_raw or is_english_raw

            logger.info(f"[SNP] can_parse: result={is_sinopac}")
            return is_sinopac

        except Exception as e:
            logger.warning(f"[SNP] can_parse error: {e}")
            return False

    def parse_statement(
        self, file_bytes: bytes, file_name: str
    ) -> Tuple[List[BankTransaction], List[BankBalance]]:
        """
        Override to handle ERP template format (multiple balances per file).
        Falls back to base implementation for raw statement format.
        """
        # Set up caching (same as base class)
        _cached: dict = {}
        _cls = type(self)

        def _cached_get_excel_file(fb: bytes) -> "pd.ExcelFile":
            fid = id(fb)
            if fid not in _cached:
                _cached[fid] = _cls.get_excel_file(fb)
            return _cached[fid]

        self.get_excel_file = _cached_get_excel_file  # type: ignore[assignment]
        try:
            logger.info(f"[SNP] parse_statement: file={file_name}, size={len(file_bytes)} bytes")
            if self._is_erp_template(file_bytes):
                logger.info("[SNP] parse_statement: using ERP template path")
                transactions = self._parse_template_transactions(file_bytes, file_name)
                balances = self._parse_template_balances(file_bytes, file_name)
                logger.info(f"[SNP] parse_statement ERP result: {len(transactions)} transactions, {len(balances)} balances")
                return transactions, balances

            # Fall back to raw statement parsing (Vietnamese + English)
            logger.info("[SNP] parse_statement: using raw statement path")
            transactions = self.parse_transactions(file_bytes, file_name)
            balance = self.parse_balances(file_bytes, file_name)
            logger.info(f"[SNP] parse_statement raw result: {len(transactions)} transactions, balance={'found' if balance else 'none'}")
            return transactions, [balance] if balance else []
        finally:
            if "get_excel_file" in self.__dict__:
                del self.get_excel_file
            _cached.clear()

    # ========== ERP Template Format Methods ==========

    def _parse_template_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """Parse transactions from ERP template 'Template details' sheet."""
        try:
            xls = self.get_excel_file(file_bytes)

            # Find "Template details" sheet
            detail_sheet = None
            for s in xls.sheet_names:
                if s.strip().lower() == "template details":
                    detail_sheet = s
                    break
            if not detail_sheet:
                logger.warning("[SNP] _parse_template_transactions: 'Template details' sheet not found")
                return []

            df = pd.read_excel(xls, sheet_name=detail_sheet, header=0)
            logger.info(f"[SNP] _parse_template_transactions: sheet='{detail_sheet}', total_rows={len(df)}, columns={list(df.columns)}")

            # Column mapping: template column name → internal field
            col_map = {
                "Bank Account Number (*)": "acc_no",
                "Trans Date (*)": "date",
                "Description (*)": "description",
                "Currency(*)": "currency",
                "DEBIT (*)": "debit",
                "CREDIT (*)": "credit",
                "TRANS ID": "transaction_id",
                "PARTNER": "beneficiary_acc_name",
                "PARTNER ACCOUNT": "beneficiary_acc_no",
                "PARTNER BANK ID": "beneficiary_bank",
            }

            transactions = []
            skipped = 0
            for _, row in df.iterrows():
                debit_val = self._fix_number_sinopac(row.get("DEBIT (*)"))
                credit_val = self._fix_number_sinopac(row.get("CREDIT (*)"))

                # Skip rows where both are zero/blank
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    skipped += 1
                    continue

                acc_no = self.to_text(row.get("Bank Account Number (*)", ""))
                # Keep only digits for account number
                acc_no = ''.join(c for c in acc_no if c.isdigit())

                currency = self.to_text(row.get("Currency(*)", "VND")).strip()
                if not currency:
                    currency = "VND"

                tx = BankTransaction(
                    bank_name="SINOPAC",
                    acc_no=acc_no,
                    debit=debit_val,
                    credit=credit_val,
                    date=self._fix_date_sinopac(row.get("Trans Date (*)")),
                    description=self.to_text(row.get("Description (*)", "")),
                    currency=currency,
                    transaction_id=self.to_text(row.get("TRANS ID", "")),
                    beneficiary_bank=self.to_text(row.get("PARTNER BANK ID", "")),
                    beneficiary_acc_no=self.to_text(row.get("PARTNER ACCOUNT", "")),
                    beneficiary_acc_name=self.to_text(row.get("PARTNER", "")),
                )
                transactions.append(tx)

            logger.info(f"[SNP] _parse_template_transactions: parsed={len(transactions)}, skipped_empty={skipped}")
            if transactions:
                accounts = set(tx.acc_no for tx in transactions)
                currencies = set(tx.currency for tx in transactions)
                dates = [tx.date for tx in transactions if tx.date]
                logger.info(
                    f"[SNP] _parse_template_transactions summary: "
                    f"accounts={accounts}, currencies={currencies}, "
                    f"date_range={min(dates) if dates else 'N/A'} ~ {max(dates) if dates else 'N/A'}"
                )
            return transactions

        except Exception as e:
            logger.error(f"[SNP] _parse_template_transactions error: {e}", exc_info=True)
            return []

    def _parse_template_balances(self, file_bytes: bytes, file_name: str) -> List[BankBalance]:
        """Parse balances from ERP template 'Template balance' sheet."""
        try:
            xls = self.get_excel_file(file_bytes)

            # Find "Template balance" sheet
            balance_sheet = None
            for s in xls.sheet_names:
                if s.strip().lower() == "template balance":
                    balance_sheet = s
                    break
            if not balance_sheet:
                logger.warning("[SNP] _parse_template_balances: 'Template balance' sheet not found")
                return []

            df = pd.read_excel(xls, sheet_name=balance_sheet, header=0)
            logger.info(f"[SNP] _parse_template_balances: sheet='{balance_sheet}', total_rows={len(df)}, columns={list(df.columns)}")

            balances = []
            skipped_no_acc = 0
            for _, row in df.iterrows():
                acc_no = self.to_text(row.get("Bank Account Number (*)", ""))
                acc_no = ''.join(c for c in acc_no if c.isdigit())

                if not acc_no:
                    skipped_no_acc += 1
                    continue

                currency = self.to_text(row.get("Currency (*)", "VND")).strip()
                if not currency:
                    currency = "VND"

                opening = self._fix_number_sinopac(row.get("Openning Balance (*)"))
                closing = self._fix_number_sinopac(row.get("Closing Balance (*)"))

                # Parse statement date
                stmt_date = self._fix_date_sinopac(row.get("Date (*)"))

                logger.info(
                    f"[SNP] _parse_template_balances row: acc={acc_no}, currency={currency}, "
                    f"opening={opening}, closing={closing}, date={stmt_date}"
                )

                balances.append(BankBalance(
                    bank_name="SINOPAC",
                    acc_no=acc_no,
                    currency=currency,
                    opening_balance=opening if opening is not None else 0.0,
                    closing_balance=closing if closing is not None else 0.0,
                    statement_date=stmt_date,
                ))

            logger.info(f"[SNP] _parse_template_balances: parsed={len(balances)}, skipped_no_acc={skipped_no_acc}")
            return balances

        except Exception as e:
            logger.error(f"[SNP] _parse_template_balances error: {e}", exc_info=True)
            return []

    # ========== Vietnamese Raw Format Methods ==========

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """
        Parse SINOPAC transactions (Vietnamese/English raw format).
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)
            logger.info(f"[SNP] parse_transactions: sheet0 shape={sheet.shape}")

            # ========== Find Header Row ==========
            top_80 = sheet.head(80)
            header_idx = self._find_header_row_sinopac(top_80)
            logger.info(f"[SNP] parse_transactions: header_idx={header_idx}")

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)
            logger.info(f"[SNP] parse_transactions: data rows after header={len(data)}, columns={list(data.columns)}")

            # ========== Find Columns (Vietnamese + English) ==========
            col_acc = self._find_column_sinopac(data, ["SO TAI KHOAN", "ACCOUNT NO"])
            col_date = self._find_column_sinopac(data, ["NGAY GIA TRI", "VALUE DATE", "TRANSACTION DATE"])
            col_currency = self._find_column_sinopac(data, ["TIEN TE", "CURRENCY"])
            col_debit = self._find_column_sinopac(data, ["TIEN GUI", "DEPOSIT"])      # Deposit = Debit
            col_credit = self._find_column_sinopac(data, ["RUT TIEN", "WITHDRAWAL"])  # Withdraw = Credit
            col_balance = self._find_column_sinopac(data, ["CAN DOI", "BALANCE"])      # Balance
            # Remarks has richer narrative than Description in raw exports.
            col_desc = self._find_column_sinopac(data, ["NHAN XET", "REMARKS", "SU MIEU TA", "DESCRIPTION"])

            logger.info(
                f"[SNP] parse_transactions columns found: "
                f"acc={col_acc}, date={col_date}, currency={col_currency}, "
                f"debit={col_debit}, credit={col_credit}, balance={col_balance}, desc={col_desc}"
            )

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
                logger.warning("[SNP] parse_transactions: no columns matched, returning empty")
                return []

            data = data.rename(columns=rename_map)
            # Keep only the last duplicate label (e.g. prefer "Remarks" mapped to
            # Description over raw short-code "Description" column in English export).
            data = data.loc[:, ~data.columns.duplicated(keep="last")]

            # Keep only renamed columns
            keep_cols = ["AccNoRaw", "Date", "Currency", "Debit", "Credit", "Balance", "Description"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                logger.warning("[SNP] parse_transactions: no usable columns after rename")
                return []

            data = data[available].copy()

            # ========== Parse Transactions ==========
            transactions = []
            skipped = 0

            for _, row in data.iterrows():
                # Parse numeric values
                debit_val = self._fix_number_sinopac(row.get("Debit")) if "Debit" in row else None
                credit_val = self._fix_number_sinopac(row.get("Credit")) if "Credit" in row else None

                # Skip rows where both are zero/blank
                if (debit_val is None or debit_val == 0) and (credit_val is None or credit_val == 0):
                    skipped += 1
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
                    date=self._fix_date_sinopac(row.get("Date")) if "Date" in row else None,
                    description=self.to_text(row.get("Description", "")) if "Description" in row else "",
                    currency=currency,
                    transaction_id="",
                    beneficiary_bank="",
                    beneficiary_acc_no="",
                    beneficiary_acc_name=""
                )

                transactions.append(tx)

            logger.info(f"[SNP] parse_transactions: parsed={len(transactions)}, skipped_empty={skipped}")
            if transactions:
                accounts = set(tx.acc_no for tx in transactions)
                currencies = set(tx.currency for tx in transactions)
                dates = [tx.date for tx in transactions if tx.date]
                total_debit = sum(tx.debit or 0 for tx in transactions)
                total_credit = sum(tx.credit or 0 for tx in transactions)
                logger.info(
                    f"[SNP] parse_transactions summary: "
                    f"accounts={accounts}, currencies={currencies}, "
                    f"date_range={min(dates) if dates else 'N/A'} ~ {max(dates) if dates else 'N/A'}, "
                    f"total_debit={total_debit:,.0f}, total_credit={total_credit:,.0f}"
                )
            return transactions

        except Exception as e:
            logger.error(f"[SNP] parse_transactions error: {e}", exc_info=True)
            return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """
        Parse SINOPAC balance information.
        """
        try:
            xls = self.get_excel_file(file_bytes)
            sheet = pd.read_excel(xls, sheet_name=0, header=None)
            logger.info(f"[SNP] parse_balances: sheet0 shape={sheet.shape}")

            # ========== Find Header Row ==========
            top_80 = sheet.head(80)
            header_idx = self._find_header_row_sinopac(top_80)
            logger.info(f"[SNP] parse_balances: header_idx={header_idx}")

            # ========== Promote Headers ==========
            data = sheet.iloc[header_idx:].copy()
            data.columns = data.iloc[0]
            data = data[1:].reset_index(drop=True)

            # ========== Find Columns (Vietnamese + English) ==========
            col_acc = self._find_column_sinopac(data, ["SO TAI KHOAN", "ACCOUNT NO"])
            col_date = self._find_column_sinopac(data, ["NGAY GIA TRI", "VALUE DATE", "TRANSACTION DATE"])
            col_currency = self._find_column_sinopac(data, ["TIEN TE", "CURRENCY"])
            col_balance = self._find_column_sinopac(data, ["CAN DOI", "BALANCE"])
            col_deposit = self._find_column_sinopac(data, ["TIEN GUI", "DEPOSIT"])
            col_withdraw = self._find_column_sinopac(data, ["RUT TIEN", "WITHDRAWAL"])

            logger.info(
                f"[SNP] parse_balances columns found: "
                f"acc={col_acc}, date={col_date}, currency={col_currency}, "
                f"balance={col_balance}, deposit={col_deposit}, withdraw={col_withdraw}"
            )

            # ========== Rename Columns ==========
            rename_map = {}
            if col_acc: rename_map[col_acc] = "AccNoRaw"
            if col_date: rename_map[col_date] = "Date"
            if col_currency: rename_map[col_currency] = "Currency"
            if col_balance: rename_map[col_balance] = "Balance"
            if col_deposit: rename_map[col_deposit] = "Deposit"
            if col_withdraw: rename_map[col_withdraw] = "Withdraw"

            if not rename_map:
                logger.warning("[SNP] parse_balances: no columns matched")
                return None

            data = data.rename(columns=rename_map)
            # Keep deterministic column labels when source has synonyms.
            data = data.loc[:, ~data.columns.duplicated(keep="last")]

            # Keep columns
            keep_cols = ["AccNoRaw", "Date", "Currency", "Balance", "Deposit", "Withdraw"]
            available = [c for c in keep_cols if c in data.columns]
            if not available:
                logger.warning("[SNP] parse_balances: no usable columns after rename")
                return None

            data = data[available].copy()

            if "AccNoRaw" not in data.columns:
                logger.warning("[SNP] parse_balances: account column missing after normalize/rename")
                return None
            if "Balance" not in data.columns:
                logger.warning("[SNP] parse_balances: balance column missing after normalize/rename")
                return None

            # ========== Parse Types ==========
            if "Date" in data.columns:
                data["Date"] = data["Date"].apply(self._fix_date_sinopac)
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
                logger.warning("[SNP] parse_balances: no data rows after parsing")
                return None

            logger.info(f"[SNP] parse_balances: {len(data)} data rows to process")

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

            logger.info(
                f"[SNP] parse_balances result: acc={acc_no}, currency={currency}, "
                f"opening={opening:,.0f}, closing={closing:,.0f}"
            )

            return BankBalance(
                bank_name="SINOPAC",
                acc_no=acc_no,
                currency=currency,
                opening_balance=opening,
                closing_balance=closing
            )

        except Exception as e:
            logger.error(f"[SNP] parse_balances error: {e}", exc_info=True)
            return None

    # ========== SINOPAC-Specific Helper Methods ==========

    def _fix_date_sinopac(self, value):
        """
        SINOPAC-specific date parser.
        SINOPAC is a Taiwanese bank that uses MM/DD/YYYY format (US standard),
        not DD/MM/YYYY (Vietnamese standard).
        """
        if value is None or pd.isna(value):
            return None

        # If already a datetime/date object (from openpyxl Excel date serial)
        if hasattr(value, 'date') and callable(value.date):
            return value.date()

        txt = str(value).strip()
        if not txt:
            return None

        if " " in txt:
            txt = txt.split(" ")[0]

        # Try MM/DD/YYYY format first (SINOPAC standard)
        try:
            return datetime.strptime(txt, "%m/%d/%Y").date()
        except (ValueError, TypeError):
            pass

        # Try YYYY-MM-DD (ISO format)
        try:
            return datetime.strptime(txt, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

        # Try YYYY/MM/DD (common in SINOPAC English export)
        try:
            return datetime.strptime(txt, "%Y/%m/%d").date()
        except (ValueError, TypeError):
            pass

        # Fallback to pandas with dayfirst=False (US format)
        try:
            return pd.to_datetime(txt, dayfirst=False).date()
        except:
            return None

    def _normalize_sinopac_text(self, value) -> str:
        """
        Normalize text for robust header matching.
        - Remove accents/diacritics
        - Replace punctuation with spaces
        - Collapse duplicated whitespace
        - Uppercase
        """
        txt = self.to_text(value)
        if not txt:
            return ""

        txt = unicodedata.normalize("NFKD", txt)
        txt = "".join(c for c in txt if not unicodedata.combining(c))
        # Convert Vietnamese D-stroke so comparisons can use plain ASCII.
        txt = txt.replace("Đ", "D").replace("đ", "d")
        txt = txt.replace("\xa0", " ").upper()
        txt = "".join(c if c.isalnum() else " " for c in txt)
        return " ".join(txt.split())

    def _find_header_row_sinopac(self, top_df: pd.DataFrame) -> int:
        """
        Find header row containing:
        - Vietnamese: NGAY GIA TRI + (CAN DOI or TIEN GUI or RUT TIEN)
        - English: ACCOUNT NO + VALUE DATE + (BALANCE or DEPOSIT or WITHDRAWAL)
        """
        for idx, row in top_df.iterrows():
            row_text = "|".join([self._normalize_sinopac_text(cell) for cell in row])

            has_vi_date = "NGAY GIA TRI" in row_text
            has_vi_balance = "CAN DOI" in row_text
            has_vi_deposit = "TIEN GUI" in row_text
            has_vi_withdraw = "RUT TIEN" in row_text

            has_en_acc = "ACCOUNT NO" in row_text
            has_en_value_date = "VALUE DATE" in row_text
            has_en_balance = "BALANCE" in row_text
            has_en_deposit = "DEPOSIT" in row_text
            has_en_withdraw = "WITHDRAWAL" in row_text

            if (
                has_vi_date and (has_vi_balance or has_vi_deposit or has_vi_withdraw)
            ) or (
                has_en_acc and has_en_value_date and (has_en_balance or has_en_deposit or has_en_withdraw)
            ):
                logger.info(f"[SNP] _find_header_row: found at row {idx}")
                return int(idx)

        logger.warning("[SNP] _find_header_row: not found, using default row 8")
        return 8  # Default fallback

    def _find_column_sinopac(self, df: pd.DataFrame, needles: Union[str, Sequence[str]]) -> Optional[str]:
        """Find first column containing one of the normalized keywords."""
        ordered_needles = [needles] if isinstance(needles, str) else list(needles)
        columns = df.columns.tolist()
        normalized_columns = [
            (col, self._normalize_sinopac_text(col))
            for col in columns
        ]

        # Needle order matters (e.g. prefer VALUE DATE over TRANSACTION DATE).
        for needle in ordered_needles:
            normalized_needle = self._normalize_sinopac_text(needle)
            if not normalized_needle:
                continue

            for col, normalized_col in normalized_columns:
                if normalized_needle in normalized_col:
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
