"""Use case for parsing bank statements (batch processing)."""

from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import csv
from io import BytesIO, StringIO
from collections import defaultdict

from app.domain.finance.bank_statement_parser.models import BankStatement, BankTransaction, BankBalance
from .bank_parsers.parser_factory import ParserFactory
from .gemini_ocr_service import GeminiOCRService


class ParseBankStatementsUseCase:
    """
    Use case for batch parsing bank statements.

    Handles:
    - Auto-detection of banks
    - Parsing transactions and balances
    - Aggregating results from multiple files
    """

    def __init__(self):
        self.parser_factory = ParserFactory

    def execute(
        self,
        files: List[Tuple[str, bytes]]
    ) -> Dict[str, Any]:
        """
        Parse multiple bank statement files.

        Args:
            files: List of (filename, file_bytes) tuples

        Returns:
            Dictionary with:
            - statements: List of parsed statements
            - all_transactions: Combined list of transactions
            - all_balances: Combined list of balances
            - summary: Processing summary
        """
        statements: List[BankStatement] = []
        all_transactions: List[BankTransaction] = []
        all_balances: List[BankBalance] = []

        successful = 0
        failed = 0
        failed_files = []

        for file_name, file_bytes in files:
            try:
                # Auto-detect bank
                parser = self.parser_factory.get_parser(file_bytes)

                if parser is None:
                    failed += 1
                    failed_files.append({
                        "file_name": file_name,
                        "error": "Bank not recognized"
                    })
                    continue

                # Parse transactions
                transactions = parser.parse_transactions(file_bytes, file_name)

                # Parse balances
                balance = parser.parse_balances(file_bytes, file_name)

                # Create statement
                statement = BankStatement(
                    bank_name=parser.bank_name,
                    file_name=file_name,
                    balance=balance,
                    transactions=transactions
                )

                statements.append(statement)
                all_transactions.extend(transactions)
                if balance:
                    all_balances.append(balance)

                successful += 1

            except Exception as e:
                failed += 1
                failed_files.append({
                    "file_name": file_name,
                    "error": str(e)
                })

        return {
            "statements": statements,
            "all_transactions": all_transactions,
            "all_balances": all_balances,
            "summary": {
                "total_files": len(files),
                "successful": successful,
                "failed": failed,
                "failed_files": failed_files,
                "total_transactions": len(all_transactions),
                "total_balances": len(all_balances)
            }
        }

    def execute_from_text(
        self,
        text_inputs: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Parse multiple bank statements from OCR text.

        Args:
            text_inputs: List of (file_name, ocr_text, bank_code) tuples
                        bank_code can be None for auto-detection

        Returns:
            Dictionary with same structure as execute():
            - statements: List of parsed statements
            - all_transactions: Combined list of transactions
            - all_balances: Combined list of balances
            - summary: Processing summary
        """
        statements: List[BankStatement] = []
        all_transactions: List[BankTransaction] = []
        all_balances: List[BankBalance] = []

        successful = 0
        failed = 0
        failed_files = []

        for file_name, ocr_text, bank_code in text_inputs:
            try:
                parser = None

                # If bank_code provided, get parser by name
                if bank_code:
                    parser = self.parser_factory.get_parser_by_name(bank_code)
                    if parser is None:
                        failed += 1
                        failed_files.append({
                            "file_name": file_name,
                            "error": f"Unknown bank code: {bank_code}"
                        })
                        continue

                    # Validate parser supports text parsing
                    if not parser.can_parse_text(ocr_text):
                        failed += 1
                        failed_files.append({
                            "file_name": file_name,
                            "error": f"Parser {parser.bank_name} does not support OCR text parsing"
                        })
                        continue
                else:
                    # Auto-detect bank from text
                    parser = self.parser_factory.get_parser_for_text(ocr_text)
                    if parser is None:
                        failed += 1
                        failed_files.append({
                            "file_name": file_name,
                            "error": "Bank not recognized from OCR text"
                        })
                        continue

                # Parse transactions from text
                transactions = parser.parse_transactions_from_text(ocr_text, file_name)

                # Parse balances from text
                balance = parser.parse_balances_from_text(ocr_text, file_name)

                # Create statement
                statement = BankStatement(
                    bank_name=parser.bank_name,
                    file_name=file_name,
                    balance=balance,
                    transactions=transactions
                )

                statements.append(statement)
                all_transactions.extend(transactions)
                if balance:
                    all_balances.append(balance)

                successful += 1

            except Exception as e:
                failed += 1
                failed_files.append({
                    "file_name": file_name,
                    "error": str(e)
                })

        return {
            "statements": statements,
            "all_transactions": all_transactions,
            "all_balances": all_balances,
            "summary": {
                "total_files": len(text_inputs),
                "successful": successful,
                "failed": failed,
                "failed_files": failed_files,
                "total_transactions": len(all_transactions),
                "total_balances": len(all_balances)
            }
        }

    def execute_from_pdf(
        self,
        pdf_inputs: List[Tuple[str, bytes, Optional[str]]]
    ) -> Dict[str, Any]:
        """
        Parse multiple bank statements from PDF files using Gemini OCR.

        Args:
            pdf_inputs: List of (file_name, pdf_bytes, bank_code) tuples
                        bank_code can be None for auto-detection

        Returns:
            Dictionary with same structure as execute():
            - statements: List of parsed statements
            - all_transactions: Combined list of transactions
            - all_balances: Combined list of balances
            - summary: Processing summary (includes ocr_results for debugging)
        """
        # Initialize Gemini OCR service
        gemini_service = GeminiOCRService()

        # Step 1: Extract text from all PDFs using Gemini
        pdf_files = [(file_name, pdf_bytes) for file_name, pdf_bytes, _ in pdf_inputs]
        ocr_results = gemini_service.extract_text_from_pdf_batch(pdf_files)

        # Step 2: Build text_inputs for execute_from_text
        text_inputs: List[Tuple[str, str, str]] = []
        ocr_failed = []

        for i, (file_name, ocr_text, ocr_error) in enumerate(ocr_results):
            bank_code = pdf_inputs[i][2]  # Get bank_code from original input

            if ocr_error:
                ocr_failed.append({
                    "file_name": file_name,
                    "error": f"OCR failed: {ocr_error}"
                })
            elif not ocr_text.strip():
                ocr_failed.append({
                    "file_name": file_name,
                    "error": "OCR returned empty text"
                })
            else:
                text_inputs.append((file_name, ocr_text, bank_code or ""))

        # Step 3: Parse the extracted text using existing execute_from_text
        if text_inputs:
            result = self.execute_from_text(text_inputs)
        else:
            result = {
                "statements": [],
                "all_transactions": [],
                "all_balances": [],
                "summary": {
                    "total_files": 0,
                    "successful": 0,
                    "failed": 0,
                    "failed_files": [],
                    "total_transactions": 0,
                    "total_balances": 0
                }
            }

        # Step 4: Merge OCR failures into the result
        result["summary"]["total_files"] = len(pdf_inputs)
        result["summary"]["failed"] += len(ocr_failed)
        result["summary"]["failed_files"].extend(ocr_failed)

        return result

    def export_to_excel(
        self,
        all_transactions: List[BankTransaction],
        all_balances: List[BankBalance]
    ) -> bytes:
        """
        Export parsed data to Excel format matching your schema.

        Creates Excel with sheets:
        1. Transactions - All transactions in standardized format
        2. Balances - All balance information
        3. Summary - Summary by bank and account

        Args:
            all_transactions: List of all transactions
            all_balances: List of all balances

        Returns:
            Excel file as bytes
        """
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # ========== Handle Empty Data ==========
            if not all_transactions and not all_balances:
                # Create an info sheet when there's no data
                df_info = pd.DataFrame([{
                    "Message": "No transactions or balances found",
                    "Possible Reasons": "Bank not recognized from OCR text, or parser doesn't support this bank format"
                }])
                df_info.to_excel(writer, sheet_name="Info", index=False)
                output.seek(0)
                return output.read()

            # ========== Sheet 1: Transactions ==========
            if all_transactions:
                tx_data = []
                for tx in all_transactions:
                    tx_data.append({
                        "Bank Name": tx.bank_name,
                        "Acc No": tx.acc_no,
                        "Debit": tx.debit,
                        "Credit": tx.credit,
                        "Date": tx.date,
                        "Description": tx.description,
                        "Currency": tx.currency,
                        "Transaction ID": tx.transaction_id,
                        "Beneficiary Acc No": tx.beneficiary_acc_no,
                        "Beneficiary Acc Name": tx.beneficiary_acc_name,
                        "Beneficiary Bank": tx.beneficiary_bank
                    })

                # Create DataFrame with explicit column order
                df_transactions = pd.DataFrame(tx_data, columns=[
                    "Bank Name",
                    "Acc No",
                    "Debit",
                    "Credit",
                    "Date",
                    "Description",
                    "Currency",
                    "Transaction ID",
                    "Beneficiary Acc No",
                    "Beneficiary Acc Name",
                    "Beneficiary Bank"
                ])
                df_transactions.to_excel(writer, sheet_name="Transactions", index=False)

                # Convert Transactions sheet to Table for formula references
                ws_transactions = writer.sheets["Transactions"]
                from openpyxl.worksheet.table import Table, TableStyleInfo
                tab = Table(displayName="Transactions_All_banks", ref=f"A1:K{len(df_transactions) + 1}")
                style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                                     showLastColumn=False, showRowStripes=True, showColumnStripes=False)
                tab.tableStyleInfo = style
                ws_transactions.add_table(tab)

            # ========== Sheet 2: Balances ==========
            if all_balances:
                bal_data = []
                for bal in all_balances:
                    bal_data.append({
                        "Bank Name": bal.bank_name,
                        "Acc No": bal.acc_no,
                        "Currency": bal.currency,
                        "Opening Balance": bal.opening_balance,
                        "Closing Balance": bal.closing_balance,
                        "Debit": "",  # Will be filled with formula
                        "Credit": "",  # Will be filled with formula
                        "Checking": "",  # Will be filled with formula
                        "Remark": ""  # Will be filled with formula
                    })

                # Create DataFrame with explicit column order
                df_balances = pd.DataFrame(bal_data, columns=[
                    "Bank Name",
                    "Acc No",
                    "Currency",
                    "Opening Balance",
                    "Closing Balance",
                    "Debit",
                    "Credit",
                    "Checking",
                    "Remark"
                ])
                df_balances.to_excel(writer, sheet_name="Balances", index=False)

                # Add formulas to Balances sheet
                ws_balances = writer.sheets["Balances"]
                for row_idx in range(2, len(df_balances) + 2):  # Start from row 2 (after header)
                    # Debit formula: =SUMIFS(Transactions_All_banks[Debit];Transactions_All_banks[Acc No];[@[Acc No]];Transactions_All_banks[Bank Name];[@[Bank Name]])
                    ws_balances[f'F{row_idx}'] = f'=SUMIFS(Transactions_All_banks[Debit],Transactions_All_banks[Acc No],B{row_idx},Transactions_All_banks[Bank Name],A{row_idx})'

                    # Credit formula: =SUMIFS(Transactions_All_banks[Credit];Transactions_All_banks[Acc No];[@[Acc No]];Transactions_All_banks[Bank Name];[@[Bank Name]])
                    ws_balances[f'G{row_idx}'] = f'=SUMIFS(Transactions_All_banks[Credit],Transactions_All_banks[Acc No],B{row_idx},Transactions_All_banks[Bank Name],A{row_idx})'

                    # Checking formula: =[@[Opening Balance]]+[@Debit]-[@Credit]-[@[Closing Balance]]
                    ws_balances[f'H{row_idx}'] = f'=D{row_idx}+F{row_idx}-G{row_idx}-E{row_idx}'

                    # Remark formula: =IF([@Checking]=0;"Reconciled";"Mismatched")
                    ws_balances[f'I{row_idx}'] = f'=IF(H{row_idx}=0,"Reconciled","Mismatched")'

            # ========== Sheet 3: Summary ==========
            summary_data = []

            # Group by bank and account
            from collections import defaultdict
            by_account = defaultdict(lambda: {
                "transactions": 0,
                "total_debit": 0.0,
                "total_credit": 0.0,
                "opening": 0.0,
                "closing": 0.0
            })

            for tx in all_transactions:
                key = f"{tx.bank_name}_{tx.acc_no}"
                by_account[key]["transactions"] += 1
                by_account[key]["total_debit"] += tx.debit or 0.0
                by_account[key]["total_credit"] += tx.credit or 0.0

            for bal in all_balances:
                key = f"{bal.bank_name}_{bal.acc_no}"
                by_account[key]["opening"] = bal.opening_balance
                by_account[key]["closing"] = bal.closing_balance

            for key, stats in by_account.items():
                bank_name, acc_no = key.split("_", 1)
                summary_data.append({
                    "Bank Name": bank_name,
                    "Acc No": acc_no,
                    "Transaction Count": stats["transactions"],
                    "Total Debit": stats["total_debit"],
                    "Total Credit": stats["total_credit"],
                    "Opening Balance": stats["opening"],
                    "Closing Balance": stats["closing"],
                    "Calculated Closing": stats["opening"] - stats["total_debit"] + stats["total_credit"]
                })

            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

        output.seek(0)
        return output.read()

    def export_to_netsuite_balance_csv(
        self,
        all_transactions: List[BankTransaction],
        all_balances: List[BankBalance]
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Export Balance CSV for NetSuite import.

        Creates CSV with 10 columns:
        External ID, Name (*), Bank Account Number (*), Bank code (*),
        Openning Balance (*), Closing Balance (*), Total Debit (*),
        Total Credit (*), Currency (*), Date (*)

        Args:
            all_transactions: List of all transactions
            all_balances: List of all balances

        Returns:
            Tuple of (CSV file as bytes, dict mapping bank_acc to external_id)
        """
        output = StringIO()
        writer = csv.writer(output)

        # Write header row (10 columns)
        headers = [
            "External ID",
            "Name (*)",
            "Bank Account Number (*)",
            "Bank code (*)",
            "Openning Balance (*)",
            "Closing Balance (*)",
            "Total Debit (*)",
            "Total Credit (*)",
            "Currency (*)",
            "Date (*)"
        ]
        writer.writerow(headers)

        # Get current date for External ID format
        now = datetime.now()
        date_suffix = now.strftime("%m%d%y")  # MMDDYY format

        # Aggregate transactions by bank_name + acc_no
        tx_aggregates = defaultdict(lambda: {"total_debit": 0, "total_credit": 0, "max_date": None})
        for tx in all_transactions:
            key = f"{tx.bank_name}_{tx.acc_no}"
            tx_aggregates[key]["total_debit"] += int(tx.debit or 0)
            tx_aggregates[key]["total_credit"] += int(tx.credit or 0)
            if tx.date:
                if tx_aggregates[key]["max_date"] is None or tx.date > tx_aggregates[key]["max_date"]:
                    tx_aggregates[key]["max_date"] = tx.date

        # Map to store balance external IDs for linking with details
        balance_external_ids = {}

        seq = 0
        for bal in all_balances:
            seq += 1
            key = f"{bal.bank_name}_{bal.acc_no}"

            # External ID format: External ID{MMDDYY}_{SEQ:04d}
            external_id = f"External ID{date_suffix}_{seq:04d}"
            balance_external_ids[key] = external_id

            # Get aggregated totals
            agg = tx_aggregates.get(key, {"total_debit": 0, "total_credit": 0, "max_date": None})

            # Get date from transactions or use current date
            if agg["max_date"]:
                tx_date = agg["max_date"]
                date_str_yyyymmdd = tx_date.strftime("%Y%m%d")
                # Format: M/D/YYYY (no leading zeros)
                formatted_date = f"{tx_date.month}/{tx_date.day}/{tx_date.year}"
            else:
                date_str_yyyymmdd = now.strftime("%Y%m%d")
                formatted_date = f"{now.month}/{now.day}/{now.year}"

            # Currency default
            currency = bal.currency if bal.currency else "VND"

            # Name format: BS/{BankCode}/{Currency}-{AccNo}/{YYYYMMDD}
            name = f"BS/{bal.bank_name}/{currency}-{bal.acc_no}/{date_str_yyyymmdd}"

            row = [
                external_id,
                name,
                bal.acc_no,
                bal.bank_name,
                int(bal.opening_balance or 0),
                int(bal.closing_balance or 0),
                agg["total_debit"],
                agg["total_credit"],
                currency,
                formatted_date
            ]
            writer.writerow(row)

        # Get CSV content and encode with UTF-8 BOM
        csv_content = output.getvalue()
        bom = b'\xef\xbb\xbf'
        return bom + csv_content.encode('utf-8'), balance_external_ids

    def export_to_netsuite_details_csv(
        self,
        all_transactions: List[BankTransaction],
        balance_external_ids: Dict[str, str]
    ) -> bytes:
        """
        Export Details CSV for NetSuite import.

        Creates CSV with 18 columns:
        External ID, External ID BSM Daily, Bank Statement Daily, Name (*),
        Bank Code (*), Bank Account Number (*), TRANS ID, Trans Date (*),
        Description (*), Currency(*), DEBIT (*), CREDIT (*), Amount, Type,
        Balance, PARTNER, PARTNER ACCOUNT, PARTNER BANK ID

        Args:
            all_transactions: List of all transactions
            balance_external_ids: Dict mapping bank_acc key to Balance External ID

        Returns:
            CSV file as bytes (UTF-8 with BOM)
        """
        output = StringIO()
        writer = csv.writer(output)

        # Write header row (18 columns)
        headers = [
            "External ID",
            "External ID BSM Daily",
            "Bank Statement Daily",
            "Name (*)",
            "Bank Code (*)",
            "Bank Account Number (*)",
            "TRANS ID",
            "Trans Date (*)",
            "Description (*)",
            "Currency(*)",
            "DEBIT (*)",
            "CREDIT (*)",
            "Amount",
            "Type",
            "Balance",
            "PARTNER",
            "PARTNER ACCOUNT",
            "PARTNER BANK ID"
        ]
        writer.writerow(headers)

        # Get current date for External ID format
        now = datetime.now()
        date_suffix = now.strftime("%m%d%y")  # MMDDYY format

        seq = 0
        for tx in all_transactions:
            seq += 1

            # External ID format: External Idline_{MMDDYY}_{SEQ:04d}
            external_id = f"External Idline_{date_suffix}_{seq:04d}"

            # Link to parent Balance
            key = f"{tx.bank_name}_{tx.acc_no}"
            external_id_bsm_daily = balance_external_ids.get(key, "")

            # Currency default
            currency = tx.currency if tx.currency else "VND"

            # Date handling
            if tx.date:
                date_str_yyyymmdd = tx.date.strftime("%Y%m%d")
                # Format: M/D/YYYY (no leading zeros)
                formatted_date = f"{tx.date.month}/{tx.date.day}/{tx.date.year}"
            else:
                date_str_yyyymmdd = now.strftime("%Y%m%d")
                formatted_date = f"{now.month}/{now.day}/{now.year}"

            # Bank Statement Daily format: BS/{Bank}/{Currency}-{AccNo}{TxID}/{YYYYMMDD}
            trans_id = tx.transaction_id or ""
            bank_statement_daily = f"BS/{tx.bank_name}/{currency}-{tx.acc_no}{trans_id}/{date_str_yyyymmdd}"

            # Name format: same as Bank Statement Daily with trailing /
            name = f"{bank_statement_daily}/"

            # Debit/Credit as integers, 0 if None
            debit_val = int(tx.debit or 0)
            credit_val = int(tx.credit or 0)

            # Amount and Type
            if debit_val > 0:
                amount = debit_val
                tx_type = "D"
            else:
                amount = credit_val
                tx_type = "C"

            row = [
                external_id,
                external_id_bsm_daily,
                bank_statement_daily,
                name,
                tx.bank_name,
                tx.acc_no,
                trans_id,
                formatted_date,
                tx.description or "",
                currency,
                debit_val,
                credit_val,
                amount,
                tx_type,
                "",  # Balance - leave empty
                tx.beneficiary_acc_name or "",
                tx.beneficiary_acc_no or "",
                tx.beneficiary_bank or ""
            ]
            writer.writerow(row)

        # Get CSV content and encode with UTF-8 BOM
        csv_content = output.getvalue()
        bom = b'\xef\xbb\xbf'
        return bom + csv_content.encode('utf-8')
