"""
Report Aggregator Service - aggregates data into summary reports.
"""
from datetime import date
from decimal import Decimal
from typing import List, Dict, Optional
from collections import defaultdict

from ..models.cash_movement import CashMovement, TransactionNature
from ..models.cash_balance import CashBalance
from ..models.saving_account import SavingAccount
from ..models.cash_report import (
    CashReport, CashReportSummary, EntitySummary,
    BankSummary, CashMovementSummary
)


class ReportAggregator:
    """
    Service to aggregate cash data into summary reports.
    """

    def __init__(self, fx_rate: Decimal = Decimal(26175)):
        """
        Args:
            fx_rate: VND/USD exchange rate
        """
        self.fx_rate = fx_rate

    def aggregate(
        self,
        movements: List[CashMovement],
        cash_balances: List[CashBalance],
        saving_accounts: List[SavingAccount],
        prior_cash_balances: Optional[List[CashBalance]] = None,
        report_period: str = "",
        opening_date: Optional[date] = None,
        ending_date: Optional[date] = None,
    ) -> CashReport:
        """
        Aggregate all data into a complete CashReport.

        Args:
            movements: All cash movements
            cash_balances: Current cash balances
            saving_accounts: All saving accounts
            prior_cash_balances: Prior period cash balances
            report_period: Period string (e.g., "W1-2Oct25")
            opening_date: Report opening date
            ending_date: Report ending date

        Returns:
            Complete CashReport
        """
        # Set defaults
        if ending_date is None:
            ending_date = date.today()
        if opening_date is None:
            opening_date = ending_date

        # Create summary
        summary = self._create_summary(
            movements=movements,
            cash_balances=cash_balances,
            prior_cash_balances=prior_cash_balances,
            report_period=report_period,
            opening_date=opening_date,
            ending_date=ending_date,
        )

        # Create report
        report = CashReport(
            summary=summary,
            movements=movements,
            cash_balances=cash_balances,
            saving_accounts=saving_accounts,
            prior_cash_balances=prior_cash_balances or [],
        )

        # Validate
        report.validate()

        return report

    def _create_summary(
        self,
        movements: List[CashMovement],
        cash_balances: List[CashBalance],
        prior_cash_balances: Optional[List[CashBalance]],
        report_period: str,
        opening_date: date,
        ending_date: date,
    ) -> CashReportSummary:
        """Create the summary section of the report"""
        # Entity summaries
        bwid_jsc = self._aggregate_entity("BWID JSC", movements, cash_balances)
        vc3 = self._aggregate_entity("VC3", movements, cash_balances)
        subsidiaries = self._aggregate_entity("Subsidiaries", movements, cash_balances)

        # Bank summaries
        bank_summaries = self._aggregate_banks(cash_balances)

        # Movement summary
        movement_summary = self._aggregate_movements(movements)

        # Total calculations
        total_vnd = bwid_jsc.cash_in_vnd + vc3.cash_in_vnd + subsidiaries.cash_in_vnd
        total_usd = bwid_jsc.cash_in_usd + vc3.cash_in_usd + subsidiaries.cash_in_usd
        total_usd_equivalent = (
            bwid_jsc.total_cash_usd_equivalent +
            vc3.total_cash_usd_equivalent +
            subsidiaries.total_cash_usd_equivalent
        )

        # Prior period comparison
        prior_total = Decimal(0)
        if prior_cash_balances:
            for bal in prior_cash_balances:
                if bal.currency == "VND":
                    prior_total += bal.closing_balance_vnd / self.fx_rate
                else:
                    prior_total += bal.closing_balance_usd

        cash_change = total_usd_equivalent - prior_total
        cash_change_pct = (cash_change / prior_total * 100) if prior_total > 0 else Decimal(0)

        # Internal transfer/contribution balance check
        internal_transfer_net = movement_summary.internal_transfer_in - movement_summary.internal_transfer_out
        internal_contribution_net = movement_summary.internal_contribution_in - movement_summary.internal_contribution_out

        return CashReportSummary(
            report_date=ending_date,
            period=report_period,
            opening_date=opening_date,
            ending_date=ending_date,
            fx_rate=self.fx_rate,
            total_cash_usd_equivalent=total_usd_equivalent,
            total_cash_vnd=total_vnd,
            total_cash_usd=total_usd,
            prior_period_cash=prior_total,
            cash_change_amount=cash_change,
            cash_change_percentage=cash_change_pct,
            bwid_jsc=bwid_jsc,
            vc3=vc3,
            subsidiaries=subsidiaries,
            bank_summaries=bank_summaries,
            movement_summary=movement_summary,
            internal_transfer_net=internal_transfer_net,
            internal_contribution_net=internal_contribution_net,
            is_balanced=(
                abs(internal_transfer_net) < Decimal(1) and
                abs(internal_contribution_net) < Decimal(1)
            ),
        )

    def _aggregate_entity(
        self,
        grouping: str,
        movements: List[CashMovement],
        cash_balances: List[CashBalance]
    ) -> EntitySummary:
        """Aggregate data for a single entity grouping"""
        entity_movements = [m for m in movements if m.grouping == grouping]
        entity_balances = [b for b in cash_balances if b.grouping == grouping]

        # Calculate cash totals
        cash_vnd = sum(
            b.closing_balance_vnd for b in entity_balances
            if b.currency == "VND"
        )
        cash_usd = sum(
            b.closing_balance_usd for b in entity_balances
            if b.currency == "USD"
        )

        cash_vnd_equivalent = cash_vnd / self.fx_rate / Decimal(1000000)  # Convert to USDmn
        total_usd_equivalent = cash_vnd_equivalent + (cash_usd / Decimal(1000000))

        # Movement totals
        total_receipt = sum(
            m.debit_amount or Decimal(0) for m in entity_movements
        )
        total_payment = sum(
            m.credit_amount or Decimal(0) for m in entity_movements
        )

        # Opening/Closing
        opening = sum(
            b.opening_balance_vnd for b in entity_balances if b.currency == "VND"
        )
        closing = sum(
            b.closing_balance_vnd for b in entity_balances if b.currency == "VND"
        )

        return EntitySummary(
            grouping=grouping,
            cash_in_vnd=cash_vnd,
            cash_in_vnd_equivalent=cash_vnd_equivalent,
            cash_in_usd=cash_usd,
            total_cash_usd_equivalent=total_usd_equivalent,
            total_receipt=total_receipt,
            total_payment=total_payment,
            net_movement=total_receipt - total_payment,
            opening_balance=opening,
            closing_balance=closing,
        )

    def _aggregate_banks(
        self,
        cash_balances: List[CashBalance]
    ) -> List[BankSummary]:
        """Aggregate cash by bank"""
        # Group by bank branch
        by_bank: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {'BWID JSC': Decimal(0), 'VC3': Decimal(0), 'Subsidiaries': Decimal(0)}
        )

        for bal in cash_balances:
            key = bal.bank_branch
            grouping = bal.grouping

            # Normalize grouping to valid values
            if grouping not in ['BWID JSC', 'VC3', 'Subsidiaries']:
                grouping = 'Subsidiaries'  # Default to Subsidiaries for unknown

            # Convert to USD equivalent
            if bal.currency == "VND":
                amount = bal.closing_balance_vnd / self.fx_rate / Decimal(1000000)
            else:
                amount = bal.closing_balance_usd / Decimal(1000000)

            by_bank[key][grouping] += amount

        # Calculate totals and create summaries
        summaries = []
        grand_total = Decimal(0)

        for bank_branch, amounts in by_bank.items():
            total = sum(amounts.values())
            grand_total += total

            summaries.append(BankSummary(
                bank_name=bank_branch.split(' - ')[0] if ' - ' in bank_branch else bank_branch,
                bank_branch=bank_branch,
                bwid_jsc_amount=amounts['BWID JSC'],
                vc3_amount=amounts['VC3'],
                subsidiaries_amount=amounts['Subsidiaries'],
                total_amount=total,
                percentage=Decimal(0),  # Will update below
            ))

        # Update percentages
        for summary in summaries:
            if grand_total > 0:
                summary.percentage = summary.total_amount / grand_total * 100

        # Sort by total amount descending
        summaries.sort(key=lambda x: x.total_amount, reverse=True)

        return summaries

    def _aggregate_movements(
        self,
        movements: List[CashMovement]
    ) -> CashMovementSummary:
        """Aggregate movements by category"""
        summary = CashMovementSummary()

        category_mapping = {
            # Receipts
            TransactionNature.RECEIPT_FROM_TENANTS: 'receipt_from_tenants',
            TransactionNature.REFUND_LAND_DEAL_DEPOSIT: 'refund_land_deal_deposit',
            TransactionNature.REFINANCING: 'refinancing',
            TransactionNature.LOAN_DRAWDOWN: 'loan_drawdown',
            TransactionNature.VAT_REFUND: 'vat_refund',
            TransactionNature.CORPORATE_LOAN_DRAWDOWN: 'corporate_loan_drawdown',
            TransactionNature.LOAN_RECEIPTS: 'loan_receipts',
            TransactionNature.CONTRIBUTION: 'contribution',
            TransactionNature.OTHER_RECEIPTS: 'other_receipts',
            TransactionNature.INTERNAL_TRANSFER_IN: 'internal_transfer_in',
            TransactionNature.INTERNAL_CONTRIBUTION: 'internal_contribution_in',
            TransactionNature.DIVIDEND_RECEIPT_INSIDE_GROUP: 'dividend_receipt_inside_group',
            TransactionNature.CASH_RECEIVED_FROM_ACQUISITION: 'cash_received_from_acquisition',

            # Payments
            TransactionNature.LAND_ACQUISITION: 'land_acquisition',
            TransactionNature.DEAL_PAYMENT: 'deal_payment',
            TransactionNature.CONSTRUCTION_EXPENSE: 'construction_expense',
            TransactionNature.OPERATING_EXPENSE: 'operating_expense',
            TransactionNature.LOAN_REPAYMENT: 'loan_repayment',
            TransactionNature.LOAN_INTEREST: 'loan_interest',
            TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT: 'internal_transfer_out',
            TransactionNature.INTERNAL_CONTRIBUTION_OUT: 'internal_contribution_out',
            TransactionNature.DIVIDEND_PAID_INSIDE_GROUP: 'dividend_paid_inside_group',
            TransactionNature.PAYMENT_FOR_ACQUISITION: 'payment_for_acquisition',
        }

        for m in movements:
            if m.nature is None:
                continue

            amount = m.amount
            attr_name = category_mapping.get(m.nature)

            if attr_name and hasattr(summary, attr_name):
                current = getattr(summary, attr_name)
                setattr(summary, attr_name, current + amount)

            # Update totals
            if m.debit_amount and m.debit_amount > 0:
                summary.total_cash_in += m.debit_amount
            if m.credit_amount and m.credit_amount > 0:
                summary.total_cash_out += m.credit_amount

        # Calculate ordinary receipt/payment (non-internal, non-special)
        ordinary_receipt_categories = [
            'receipt_from_tenants', 'refund_land_deal_deposit', 'other_receipts',
            'contribution', 'loan_receipts', 'vat_refund'
        ]
        summary.ordinary_receipt = sum(
            getattr(summary, cat) for cat in ordinary_receipt_categories
        )

        ordinary_payment_categories = [
            'land_acquisition', 'deal_payment', 'construction_expense',
            'operating_expense', 'loan_repayment', 'loan_interest'
        ]
        summary.ordinary_payment = sum(
            getattr(summary, cat) for cat in ordinary_payment_categories
        )

        return summary

    def convert_to_usd(self, amount_vnd: Decimal) -> Decimal:
        """Convert VND amount to USD"""
        return amount_vnd / self.fx_rate

    def convert_to_usd_million(self, amount_vnd: Decimal) -> Decimal:
        """Convert VND amount to USD million"""
        return amount_vnd / self.fx_rate / Decimal(1000000)
