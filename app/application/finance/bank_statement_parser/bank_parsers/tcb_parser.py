"""Parser for TCB (Techcombank / Ngân hàng TMCP Kỹ Thương Việt Nam) statements."""

import re
from typing import List, Optional
import pandas as pd
from datetime import datetime

from app.domain.finance.bank_statement_parser.models import BankTransaction, BankBalance
from .base_parser import BaseBankParser
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class TCBParser(BaseBankParser):
    """Parser for Techcombank statements - supports PDF (via OCR)."""

    @property
    def bank_name(self) -> str:
        return "TCB"

    def can_parse(self, file_bytes: bytes) -> bool:
        """
        Detect if file is Techcombank statement (Excel format).

        TCB markers:
        - "TECHCOMBANK" or "KỸ THƯƠNG VIỆT NAM"
        - "SỐ PHỤ KIÊM PHIẾU BÁO NỢ/CÓ" or "BANK STATEMENT"
        """
        try:
            xls = self.get_excel_file(file_bytes)
            df = pd.read_excel(xls, sheet_name=0, header=None)

            # Get first 20 rows
            top_20 = df.head(20)

            # Flatten all cells to uppercase text
            all_text = []
            for col in top_20.columns:
                all_text.extend([self.to_text(cell).upper() for cell in top_20[col]])

            txt = " ".join(all_text)

            # Check for Techcombank markers
            has_tcb = "TECHCOMBANK" in txt or "KỸ THƯƠNG" in txt
            has_statement = "PHIẾU BÁO" in txt or "BANK STATEMENT" in txt

            return has_tcb and has_statement

        except Exception:
            return False

    def parse_transactions(self, file_bytes: bytes, file_name: str) -> List[BankTransaction]:
        """Parse TCB transactions from Excel file."""
        # TODO: Implement Excel parsing if needed
        return []

    def parse_balances(self, file_bytes: bytes, file_name: str) -> Optional[BankBalance]:
        """Parse TCB balance from Excel file."""
        # TODO: Implement Excel parsing if needed
        return None

    # ========== OCR Text Parsing Methods (PDF via Gemini) ==========

    def can_parse_text(self, text: str) -> bool:
        """
        Detect if OCR text is from Techcombank statement.

        Detection:
        - Contains "TECHCOMBANK" or "Kỹ Thương Việt Nam"
        - Contains statement markers
        """
        if not text:
            return False

        text_upper = text.upper()

        # Check for Techcombank markers
        has_tcb = "TECHCOMBANK" in text_upper or "KỸ THƯƠNG" in text_upper

        # Check for statement markers
        has_statement = any(marker in text_upper for marker in [
            "PHIẾU BÁO NỢ", "BANK STATEMENT", "SỐ PHỤ KIÊM"
        ])

        return has_tcb and has_statement

    def parse_transactions_from_text(self, text: str, file_name: str) -> List[BankTransaction]:
        """
        Parse Techcombank transactions from OCR text.

        TCB PDF OCR format (each column on separate lines):
        - Date line: DD/MM/YYYY
        - Description/counterparty lines (multiple)
        - Transaction ID (like FT25331141130003)
        - Amount lines (debit and/or credit amounts)
        - Balance line
        """
        transactions = []

        # Extract account info
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        logger.info(f"TCB: Parsing transactions, account={acc_no}, currency={currency}")

        lines = text.split('\n')

        # Date pattern: DD/MM/YYYY
        date_pattern = re.compile(r'^(\d{2}/\d{2}/\d{4})$')

        # Amount pattern: numbers with commas and .00 (e.g., 50,600,000,000.00)
        amount_pattern = re.compile(r'^[\d,]+\.\d{2}$')

        # Transaction ID pattern (e.g., FT25331141130003)
        tx_id_pattern = re.compile(r'^[A-Z]{2}\d{10,}')

        # Skip patterns for non-transaction content
        skip_patterns = [
            'ngày giao dịch', 'transaction date',
            'số dư đầu', 'open balance', 'số dư cuối', 'ending balance',
            'cộng doanh số', 'total volume',
            'ngày giờ in', 'printed', 'phiếu này', 'this paper',
            'ngân hàng tmcp', 'chi nhánh', 'branch', 'mst', 'tax code',
            'tên khách hàng', 'customer name', 'địa chỉ', 'address',
            'loại tiền', 'currency', 'số tài khoản', 'account no',
            'loại tài khoản', 'type of account', 'tên tài khoản', 'account name',
            'chúng tôi xin', 'we would like', 'techcombank',
            'đối tác', 'remitter', 'nh đối tác', 'remitter bank',
            'diễn giải', 'details', 'số bút toán', 'transaction no',
            'nợ', 'có', 'phí', 'lãi', 'thuế', 'tax', 'fee', 'interest',
            'debit', 'credit', 'balance'
        ]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            line_lower = line.lower()

            # Skip header/footer lines
            if any(skip in line_lower for skip in skip_patterns):
                i += 1
                continue

            # Look for date line to start a transaction
            date_match = date_pattern.match(line)
            if date_match:
                tx_date_str = date_match.group(1)

                # Collect transaction data from following lines
                j = i + 1
                description_parts = []
                amounts = []
                tx_id = ""

                hit_summary = False
                while j < len(lines):
                    next_line = lines[j].strip()
                    next_lower = next_line.lower()

                    # Stop if we hit another date (next transaction)
                    if date_pattern.match(next_line):
                        break

                    # Check if we hit summary section (Total Volume / Ending balance)
                    if any(stop in next_lower for stop in ['số dư cuối', 'ending balance', 'cộng doanh số', 'total volume']):
                        hit_summary = True
                        # If we have description but no amounts yet, this might be interest tx
                        # Try to find amount in summary section
                        if description_parts and not amounts:
                            desc_check = ' '.join(description_parts).lower()
                            if 'tra lai' in desc_check or 'tien lai' in desc_check or 'interest' in desc_check:
                                # Look for first small amount in summary (interest is usually small)
                                for k in range(j + 1, min(j + 8, len(lines))):
                                    sum_line = lines[k].strip()
                                    if amount_pattern.match(sum_line):
                                        val = self._parse_ocr_number(sum_line)
                                        # Interest is typically smaller than main transaction amounts
                                        if val is not None and val < 1000000:  # Less than 1M
                                            amounts.append(val)
                                            break
                        break

                    # Skip empty lines
                    if not next_line:
                        j += 1
                        continue

                    # Check if it's a transaction ID
                    if tx_id_pattern.match(next_line):
                        tx_id = next_line
                        j += 1
                        continue

                    # Check if it's an amount line
                    if amount_pattern.match(next_line):
                        val = self._parse_ocr_number(next_line)
                        if val is not None:
                            amounts.append(val)
                        j += 1
                        continue

                    # Skip header-like lines
                    if any(skip in next_lower for skip in skip_patterns):
                        j += 1
                        continue

                    # Otherwise it's part of description
                    description_parts.append(next_line)
                    j += 1

                # Build description
                description = ' '.join(description_parts)

                # Determine debit/credit from amounts
                # TCB format: usually has 2-3 amounts per transaction:
                # - First amount could be debit OR credit
                # - Last amount is usually balance
                # Need to analyze context
                debit_val = None
                credit_val = None

                if len(amounts) >= 2:
                    # If description mentions deposit/credit keywords -> credit
                    # Otherwise -> debit
                    desc_lower = description.lower()

                    # Check for credit indicators (money IN to account)
                    # "tat toan tien gui" = closing deposit = money comes back
                    credit_keywords = ['tra lai', 'tien lai', 'interest', 'deposit', 'nhan', 'receive',
                                       'tat toan tien gui', 'tien gui dao han']
                    is_credit = any(kw in desc_lower for kw in credit_keywords)

                    # Check for debit indicators (money OUT from account)
                    debit_keywords = ['transfer', 'chuyen', 'thanh toan', 'payment', 'phi', 'fee',
                                      'from tcb to', 'internal transfer']
                    is_debit = any(kw in desc_lower for kw in debit_keywords)

                    # First non-balance amount is the transaction amount
                    # Last amount is usually balance
                    tx_amount = amounts[0]

                    if is_credit and not is_debit:
                        credit_val = tx_amount
                    else:
                        # Default to debit (transfers out, payments, etc.)
                        debit_val = tx_amount

                elif len(amounts) == 1:
                    # Single amount - determine by context
                    desc_lower = description.lower()
                    credit_keywords = ['tra lai', 'tien lai', 'interest', 'deposit', 'tat toan tien gui']
                    debit_keywords = ['transfer', 'chuyen', 'from tcb to', 'internal transfer']

                    is_credit = any(kw in desc_lower for kw in credit_keywords)
                    is_debit = any(kw in desc_lower for kw in debit_keywords)

                    if is_credit and not is_debit:
                        credit_val = amounts[0]
                    else:
                        debit_val = amounts[0]

                # Create transaction if we have valid data
                if (debit_val or credit_val) and description:
                    tx = BankTransaction(
                        bank_name="TCB",
                        acc_no=acc_no or "",
                        debit=debit_val,
                        credit=credit_val,
                        date=self._parse_ocr_date(tx_date_str),
                        description=description.strip(),
                        currency=currency,
                        transaction_id=tx_id,
                        beneficiary_bank="",
                        beneficiary_acc_no="",
                        beneficiary_acc_name=""
                    )
                    transactions.append(tx)

                i = j
                continue

            i += 1

        logger.info(f"TCB: Parsed {len(transactions)} transactions")
        return transactions

    def parse_balances_from_text(self, text: str, file_name: str) -> Optional[BankBalance]:
        """
        Parse Techcombank balance from OCR text.

        Look for:
        - "Số dư đầu kỳ/ Open balance" followed by amount
        - "Số dư cuối kì / Ending balance" followed by amount
        """
        acc_no = self._extract_account_from_ocr(text)
        currency = self._extract_currency_from_ocr(text)

        logger.info(f"TCB: Parsing balances, account={acc_no}, currency={currency}")

        lines = text.split('\n')

        opening = 0.0
        closing = 0.0

        for idx, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()

            # Opening balance: "Số dư đầu kỳ/ Open balance" followed by amount
            if 'số dư đầu' in line_lower or 'open balance' in line_lower:
                # Try to find amount in same line
                amounts = re.findall(r'[\d,]+\.\d{2}', line)
                if amounts:
                    val = self._parse_ocr_number(amounts[0])
                    if val is not None:
                        opening = val
                else:
                    # Look at next line for amount
                    if idx + 1 < len(lines):
                        next_line = lines[idx + 1].strip()
                        amounts = re.findall(r'[\d,]+\.\d{2}', next_line)
                        if amounts:
                            val = self._parse_ocr_number(amounts[0])
                            if val is not None:
                                opening = val

            # Closing balance: "Số dư cuối kì / Ending balance" followed by amount
            # TCB format: After "Ending balance" line, there are several amounts on following lines
            # The LAST unique amount in this section is the closing balance
            if 'số dư cuối' in line_lower or 'ending balance' in line_lower:
                amounts = re.findall(r'[\d,]+\.\d{2}', line)
                if amounts:
                    val = self._parse_ocr_number(amounts[-1])
                    if val is not None:
                        closing = val
                else:
                    # Collect ALL amounts from following lines until we hit footer
                    all_amounts = []
                    for k in range(idx + 1, min(idx + 10, len(lines))):
                        check_line = lines[k].strip()
                        check_lower = check_line.lower()

                        # Stop at footer
                        if 'ngày giờ in' in check_lower or 'printed' in check_lower:
                            break

                        amounts = re.findall(r'[\d,]+\.\d{2}', check_line)
                        for amt in amounts:
                            val = self._parse_ocr_number(amt)
                            if val is not None:
                                all_amounts.append(val)

                    # The closing balance is the LAST amount (or second-to-last if duplicated)
                    if all_amounts:
                        # TCB often repeats the closing balance twice at the end
                        closing = all_amounts[-1]

        if opening == 0 and closing == 0:
            logger.warning("TCB: No balances found")
            return None

        logger.info(f"TCB: Found opening={opening}, closing={closing}")

        return BankBalance(
            bank_name="TCB",
            acc_no=acc_no or "",
            currency=currency,
            opening_balance=opening,
            closing_balance=closing
        )

    # ========== Helper Methods ==========

    def _extract_account_from_ocr(self, text: str) -> Optional[str]:
        """Extract account number from OCR text."""
        # Pattern: "Số tài khoản / Account No: 19037134677017"
        patterns = [
            r'(?:số tài khoản|account\s*no)[:\s]*(\d{10,})',
            r'(?:stk|a/c)[:\s]*(\d{10,})',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)

        return None

    def _extract_currency_from_ocr(self, text: str) -> str:
        """Extract currency from OCR text."""
        # Pattern: "Loại tiền / Currency: VND"
        match = re.search(r'(?:loại tiền|currency)[:\s]*(VND|USD|EUR)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Default
        if "usd" in text.lower():
            return "USD"
        return "VND"

    def _parse_ocr_number(self, value: str) -> Optional[float]:
        """Parse number from OCR text."""
        if not value:
            return None

        # Remove commas and spaces
        cleaned = value.replace(",", "").replace(" ", "")

        try:
            num = float(cleaned)
            return num if num > 0 else None
        except (ValueError, TypeError):
            return None

    def _parse_ocr_date(self, date_str: str):
        """Parse date from OCR text."""
        if not date_str:
            return None

        date_str = date_str.strip()

        # Try DD/MM/YYYY format
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").date()
        except:
            pass

        # Try pandas as fallback
        try:
            return pd.to_datetime(date_str, dayfirst=True).date()
        except:
            return None
