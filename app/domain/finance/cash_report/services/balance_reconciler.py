"""
Balance Reconciler Service - reconciles calculated balances with bank statements.
"""
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..models.cash_movement import CashMovement
from ..models.cash_balance import CashBalance


@dataclass
class ReconciliationResult:
    """Result of reconciliation for a single account"""
    account_number: str
    bank_name: str
    currency: str
    opening_balance: Decimal
    total_debit: Decimal
    total_credit: Decimal
    calculated_closing: Decimal
    bank_statement_balance: Optional[Decimal]
    difference: Decimal
    is_reconciled: bool
    issues: List[str]


class BalanceReconciler:
    """
    Service to reconcile cash balances:
    1. Calculate closing balance from opening + movements
    2. Compare with bank statement balance
    3. Identify discrepancies
    """

    def __init__(self, tolerance_vnd: Decimal = Decimal(1), tolerance_usd: Decimal = Decimal("0.01")):
        """
        Args:
            tolerance_vnd: Acceptable difference for VND accounts
            tolerance_usd: Acceptable difference for USD accounts
        """
        self.tolerance_vnd = tolerance_vnd
        self.tolerance_usd = tolerance_usd

    def reconcile_all(
        self,
        cash_balances: List[CashBalance],
        movements: List[CashMovement],
        bank_statement_balances: Optional[Dict[str, Decimal]] = None
    ) -> List[ReconciliationResult]:
        """
        Reconcile all cash balances.

        Args:
            cash_balances: List of CashBalance from prior period (opening balances)
            movements: List of all movements in current period
            bank_statement_balances: Dict of account_number -> actual bank balance

        Returns:
            List of ReconciliationResult
        """
        results = []
        bank_balances = bank_statement_balances or {}

        # Group movements by account
        movements_by_account = self._group_movements_by_account(movements)

        for balance in cash_balances:
            acc_no = balance.account_number
            account_movements = movements_by_account.get(acc_no, [])

            result = self._reconcile_account(
                balance=balance,
                movements=account_movements,
                bank_balance=bank_balances.get(acc_no)
            )
            results.append(result)

        return results

    def reconcile_account(
        self,
        balance: CashBalance,
        movements: List[CashMovement],
        bank_balance: Optional[Decimal] = None
    ) -> ReconciliationResult:
        """
        Reconcile a single account.
        """
        return self._reconcile_account(balance, movements, bank_balance)

    def calculate_closing_balance(
        self,
        opening_balance: Decimal,
        movements: List[CashMovement],
        currency: str = "VND"
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate closing balance from opening and movements.

        Returns:
            Tuple of (total_debit, total_credit, closing_balance)
        """
        total_debit = Decimal(0)
        total_credit = Decimal(0)

        for m in movements:
            if m.currency != currency:
                continue
            total_debit += m.debit_amount or Decimal(0)
            total_credit += m.credit_amount or Decimal(0)

        closing = opening_balance + total_debit - total_credit
        return total_debit, total_credit, closing

    def find_missing_transactions(
        self,
        calculated_closing: Decimal,
        bank_balance: Decimal,
        movements: List[CashMovement]
    ) -> Dict[str, any]:
        """
        Analyze difference and suggest possible missing transactions.

        Returns:
            Dict with analysis and suggestions
        """
        diff = bank_balance - calculated_closing

        analysis = {
            'difference': diff,
            'difference_type': 'over' if diff > 0 else 'under',
            'suggestions': []
        }

        if diff == 0:
            return analysis

        # Check for common issues
        abs_diff = abs(diff)

        # Check if diff matches any movement (possible duplicate or missing)
        for m in movements:
            amount = m.amount
            if abs(amount - abs_diff) < self.tolerance_vnd:
                if diff > 0:
                    analysis['suggestions'].append({
                        'type': 'possible_missing_receipt',
                        'amount': amount,
                        'similar_transaction': m.description[:50]
                    })
                else:
                    analysis['suggestions'].append({
                        'type': 'possible_duplicate_or_missing_payment',
                        'amount': amount,
                        'similar_transaction': m.description[:50]
                    })

        # Check for bank charges (usually small amounts)
        if abs_diff < Decimal(1000000):  # Less than 1M VND
            analysis['suggestions'].append({
                'type': 'possible_bank_charge',
                'amount': abs_diff
            })

        # Check for interest (usually positive difference)
        if diff > 0 and abs_diff < Decimal(10000000):  # Less than 10M VND
            analysis['suggestions'].append({
                'type': 'possible_interest_not_recorded',
                'amount': diff
            })

        return analysis

    def validate_opening_balance(
        self,
        current_opening: Decimal,
        prior_closing: Decimal
    ) -> Tuple[bool, Decimal]:
        """
        Validate that current opening equals prior closing.

        Returns:
            Tuple of (is_valid, difference)
        """
        diff = abs(current_opening - prior_closing)
        is_valid = diff <= self.tolerance_vnd
        return is_valid, diff

    def check_internal_transfer_balance(
        self,
        movements: List[CashMovement]
    ) -> Tuple[bool, Decimal, Decimal]:
        """
        Check that internal transfers in/out net to zero.

        Returns:
            Tuple of (is_balanced, total_in, total_out)
        """
        total_in = Decimal(0)
        total_out = Decimal(0)

        for m in movements:
            if m.nature and 'internal transfer' in m.nature.value.lower():
                if m.debit_amount:
                    total_in += m.debit_amount
                if m.credit_amount:
                    total_out += m.credit_amount

        is_balanced = abs(total_in - total_out) <= self.tolerance_vnd
        return is_balanced, total_in, total_out

    def check_internal_contribution_balance(
        self,
        movements: List[CashMovement]
    ) -> Tuple[bool, Decimal, Decimal]:
        """
        Check that internal contributions in/out net to zero.

        Returns:
            Tuple of (is_balanced, total_in, total_out)
        """
        total_in = Decimal(0)
        total_out = Decimal(0)

        for m in movements:
            if m.nature and 'internal contribution' in m.nature.value.lower():
                if m.debit_amount:
                    total_in += m.debit_amount
                if m.credit_amount:
                    total_out += m.credit_amount

        is_balanced = abs(total_in - total_out) <= self.tolerance_vnd
        return is_balanced, total_in, total_out

    def _reconcile_account(
        self,
        balance: CashBalance,
        movements: List[CashMovement],
        bank_balance: Optional[Decimal]
    ) -> ReconciliationResult:
        """Internal method to reconcile a single account"""
        issues = []
        currency = balance.currency

        # Get opening balance based on currency
        if currency == "VND":
            opening = balance.opening_balance_vnd
        else:
            opening = balance.opening_balance_usd

        # Calculate from movements
        total_debit, total_credit, calculated_closing = self.calculate_closing_balance(
            opening, movements, currency
        )

        # Determine tolerance
        tolerance = self.tolerance_vnd if currency == "VND" else self.tolerance_usd

        # Calculate difference
        if bank_balance is not None:
            diff = abs(calculated_closing - bank_balance)
            is_reconciled = diff <= tolerance

            if not is_reconciled:
                if calculated_closing > bank_balance:
                    issues.append(f"Calculated balance exceeds bank by {diff}")
                else:
                    issues.append(f"Calculated balance below bank by {diff}")
        else:
            diff = Decimal(0)
            is_reconciled = False
            issues.append("No bank statement balance provided for reconciliation")

        return ReconciliationResult(
            account_number=balance.account_number,
            bank_name=balance.bank_name,
            currency=currency,
            opening_balance=opening,
            total_debit=total_debit,
            total_credit=total_credit,
            calculated_closing=calculated_closing,
            bank_statement_balance=bank_balance,
            difference=diff,
            is_reconciled=is_reconciled,
            issues=issues
        )

    def _group_movements_by_account(
        self,
        movements: List[CashMovement]
    ) -> Dict[str, List[CashMovement]]:
        """Group movements by account number"""
        grouped = {}
        for m in movements:
            acc_no = m.account_number
            if acc_no not in grouped:
                grouped[acc_no] = []
            grouped[acc_no].append(m)
        return grouped

    def generate_reconciliation_report(
        self,
        results: List[ReconciliationResult]
    ) -> Dict[str, any]:
        """
        Generate a summary report of reconciliation results.
        """
        total = len(results)
        reconciled = sum(1 for r in results if r.is_reconciled)
        not_reconciled = total - reconciled

        total_diff = sum(r.difference for r in results if not r.is_reconciled)

        issues_by_account = {
            r.account_number: r.issues
            for r in results if r.issues
        }

        return {
            'total_accounts': total,
            'reconciled_count': reconciled,
            'not_reconciled_count': not_reconciled,
            'reconciliation_rate': reconciled / total if total > 0 else 0,
            'total_difference': total_diff,
            'accounts_with_issues': issues_by_account,
            'not_reconciled_accounts': [
                {
                    'account': r.account_number,
                    'bank': r.bank_name,
                    'calculated': r.calculated_closing,
                    'bank_balance': r.bank_statement_balance,
                    'difference': r.difference
                }
                for r in results if not r.is_reconciled
            ]
        }
