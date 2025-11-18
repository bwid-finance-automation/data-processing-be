"""Use case for parsing bank statements (batch processing)."""

from typing import List, Tuple, Dict, Any
import pandas as pd
from io import BytesIO

from app.domain.finance.bank_statement_parser.models import BankStatement, BankTransaction, BankBalance
from .bank_parsers.parser_factory import ParserFactory


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
