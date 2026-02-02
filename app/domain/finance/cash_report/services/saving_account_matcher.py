"""
Saving Account Matcher Service - handles saving account transactions and counter entries.
"""
import re
from datetime import date
from decimal import Decimal
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..models.cash_movement import CashMovement, TransactionNature, MovementType
from ..models.saving_account import SavingAccount, SavingAccountStatus


@dataclass
class SavingAccountTransaction:
    """Represents a saving account transaction that needs a counter entry"""
    original_movement: CashMovement
    saving_account_number: str
    transaction_type: str  # 'open', 'close', 'partial_withdraw', 'interest'
    amount: Decimal


class SavingAccountMatcher:
    """
    Service to handle saving account related transactions:
    1. Detect saving account transactions from movements
    2. Create counter entries for saving accounts
    3. Update saving account balances
    4. Handle auto-roll detection
    """

    def __init__(self):
        self._init_patterns()

    def _init_patterns(self):
        """Initialize patterns for detecting saving account transactions"""
        self.patterns = {
            'saving_account_number': [
                r'(?:tài khoản|tai khoan|account|tk|a/c)\s*[:\s]*(\d{10,20})',
                r'(\d{12,20})\s*(?:tiền gửi|tien gui|saving)',
                r'hđtk\s*[:\s]*(\d{10,20})',
                r'hdtk\s*[:\s]*(\d{10,20})',
            ],
            'open_saving': [
                r'mở\s*(?:mới|tk|tài khoản)',
                r'mo\s*(?:moi|tk|tai khoan)',
                r'open\s*(?:new|account|saving)',
                r'gửi\s*tiền',
                r'gui\s*tien',
            ],
            'close_saving': [
                r'tất\s*toán',
                r'tat\s*toan',
                r'close\s*(?:account|saving)',
                r'rút\s*(?:tiền|gốc)',
                r'rut\s*(?:tien|goc)',
                r'withdraw',
            ],
            'interest_payment': [
                r'trả\s*lãi',
                r'tra\s*lai',
                r'interest\s*(?:payment|paid)',
                r'lãi\s*(?:tiền gửi|suất)',
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in self.patterns.items()
        }

    def detect_saving_transactions(
        self,
        movements: List[CashMovement]
    ) -> List[SavingAccountTransaction]:
        """
        Detect saving account related transactions from movements.

        Args:
            movements: List of all movements

        Returns:
            List of detected saving account transactions
        """
        saving_transactions = []

        for movement in movements:
            if not self._is_saving_related(movement):
                continue

            # Extract saving account number
            saving_acc_no = self._extract_saving_account_number(movement.description)
            if not saving_acc_no:
                # Use movement account number if can't extract
                saving_acc_no = f"SAVING_{movement.account_number}"

            # Determine transaction type
            tx_type = self._determine_transaction_type(movement)

            saving_transactions.append(SavingAccountTransaction(
                original_movement=movement,
                saving_account_number=saving_acc_no,
                transaction_type=tx_type,
                amount=movement.amount
            ))

        return saving_transactions

    def create_counter_entries(
        self,
        saving_transactions: List[SavingAccountTransaction],
        saving_accounts: List[SavingAccount],
        period: str
    ) -> List[CashMovement]:
        """
        Create counter entries for saving account transactions.

        For each saving transaction:
        - If cash out to open saving -> create cash in for saving account
        - If cash in from close saving -> create cash out for saving account

        Args:
            saving_transactions: Detected saving transactions
            saving_accounts: Existing saving accounts
            period: Current report period

        Returns:
            List of counter entry movements
        """
        counter_entries = []
        saving_acc_map = {sa.account_number: sa for sa in saving_accounts}

        for tx in saving_transactions:
            original = tx.original_movement

            # Create counter entry with opposite direction
            counter = CashMovement(
                source="Counter Entry",
                bank_code=original.bank_code,
                account_number=tx.saving_account_number,
                transaction_date=original.transaction_date,
                description=f"Counter entry: {original.description[:100]}",
                currency=original.currency,
                account_type="Saving Account",
                period=period,
                entity=original.entity,
                grouping=original.grouping,
                related_saving_account=original.account_number,
                is_counter_entry=True,
            )

            # Reverse the direction
            if original.movement_type == MovementType.CREDIT:
                # Original is cash out -> counter is cash in to saving
                counter.debit_amount = original.credit_amount
                counter.nature = TransactionNature.INTERNAL_TRANSFER_IN
                counter.key_payment = "Internal transfer in"
            else:
                # Original is cash in -> counter is cash out from saving
                counter.credit_amount = original.debit_amount
                counter.nature = TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT
                counter.key_payment = "Internal transfer out"

            counter_entries.append(counter)

            # Update saving account if exists
            if tx.saving_account_number in saving_acc_map:
                self._update_saving_account(
                    saving_acc_map[tx.saving_account_number],
                    tx
                )

        return counter_entries

    def match_internal_transfers(
        self,
        movements: List[CashMovement]
    ) -> List[Tuple[CashMovement, CashMovement]]:
        """
        Match internal transfer pairs (in and out should match).

        Returns:
            List of matched pairs (transfer_out, transfer_in)
        """
        # Filter internal transfers
        transfers_out = [
            m for m in movements
            if m.nature in [TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT,
                          TransactionNature.INTERNAL_TRANSFER]
            and m.credit_amount
        ]

        transfers_in = [
            m for m in movements
            if m.nature == TransactionNature.INTERNAL_TRANSFER_IN
            and m.debit_amount
        ]

        matched_pairs = []
        used_ins = set()

        for out_tx in transfers_out:
            for i, in_tx in enumerate(transfers_in):
                if i in used_ins:
                    continue

                # Match by amount and date
                if (out_tx.credit_amount == in_tx.debit_amount and
                    out_tx.transaction_date == in_tx.transaction_date):
                    matched_pairs.append((out_tx, in_tx))
                    used_ins.add(i)
                    break

        return matched_pairs

    def find_unmatched_transfers(
        self,
        movements: List[CashMovement]
    ) -> Dict[str, List[CashMovement]]:
        """
        Find internal transfers that don't have matching counter entries.

        Returns:
            Dict with 'unmatched_out' and 'unmatched_in' lists
        """
        matched_pairs = self.match_internal_transfers(movements)

        # Use id() to track matched objects since CashMovement is not hashable
        matched_out_ids = {id(pair[0]) for pair in matched_pairs}
        matched_in_ids = {id(pair[1]) for pair in matched_pairs}

        all_outs = [
            m for m in movements
            if m.nature in [TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT,
                          TransactionNature.INTERNAL_TRANSFER]
            and m.credit_amount
        ]

        all_ins = [
            m for m in movements
            if m.nature == TransactionNature.INTERNAL_TRANSFER_IN
            and m.debit_amount
        ]

        return {
            'unmatched_out': [m for m in all_outs if id(m) not in matched_out_ids],
            'unmatched_in': [m for m in all_ins if id(m) not in matched_in_ids],
        }

    def detect_auto_rolled_accounts(
        self,
        saving_accounts: List[SavingAccount],
        movements: List[CashMovement],
        report_end_date: date
    ) -> List[SavingAccount]:
        """
        Detect saving accounts that should have matured but still have balance.
        These are likely auto-rolled.

        Returns:
            List of auto-rolled saving accounts
        """
        auto_rolled = []

        for sa in saving_accounts:
            if sa.maturity_date > report_end_date:
                continue  # Not matured yet

            if sa.principal_amount <= 0:
                continue  # Already closed

            # Check if there's a close transaction
            has_close_tx = any(
                m.related_saving_account == sa.account_number and
                m.credit_amount and m.credit_amount > 0
                for m in movements
            )

            if not has_close_tx:
                sa.status = SavingAccountStatus.AUTO_ROLLED
                auto_rolled.append(sa)

        return auto_rolled

    def _is_saving_related(self, movement: CashMovement) -> bool:
        """Check if movement is related to saving accounts"""
        desc = movement.description.lower()
        keywords = [
            'tiền gửi', 'tien gui', 'saving', 'term deposit',
            'hđtk', 'hdtk', 'tất toán', 'tat toan',
            'mở tk', 'mo tk', 'gửi tiền', 'gui tien'
        ]
        return any(kw in desc for kw in keywords)

    def _extract_saving_account_number(self, description: str) -> Optional[str]:
        """Extract saving account number from description"""
        for pattern in self.compiled_patterns['saving_account_number']:
            match = pattern.search(description)
            if match:
                return match.group(1)
        return None

    def _determine_transaction_type(self, movement: CashMovement) -> str:
        """Determine the type of saving account transaction"""
        desc = movement.description.lower()

        if any(p.search(desc) for p in self.compiled_patterns['close_saving']):
            return 'close'
        if any(p.search(desc) for p in self.compiled_patterns['open_saving']):
            return 'open'
        if any(p.search(desc) for p in self.compiled_patterns['interest_payment']):
            return 'interest'

        # Default based on direction
        if movement.credit_amount and movement.credit_amount > 0:
            return 'open'  # Cash out usually means opening/adding to saving
        return 'close'  # Cash in usually means closing/withdrawing

    def _update_saving_account(
        self,
        saving_account: SavingAccount,
        transaction: SavingAccountTransaction
    ) -> None:
        """Update saving account based on transaction"""
        if transaction.transaction_type == 'close':
            saving_account.principal_amount -= transaction.amount
            if saving_account.principal_amount <= 0:
                saving_account.status = SavingAccountStatus.CLOSED
        elif transaction.transaction_type == 'open':
            saving_account.principal_amount += transaction.amount
        elif transaction.transaction_type == 'interest':
            saving_account.accrued_interest += transaction.amount

    def create_new_saving_accounts(
        self,
        saving_transactions: List[SavingAccountTransaction],
        existing_accounts: List[SavingAccount],
    ) -> List[SavingAccount]:
        """
        Create new SavingAccount objects from 'open' transactions that don't match existing accounts.

        Args:
            saving_transactions: Detected saving transactions
            existing_accounts: Existing saving accounts from template

        Returns:
            List of newly created SavingAccount (needs_review=True for manual completion)
        """
        existing_acc_numbers = {sa.account_number for sa in existing_accounts}
        new_accounts = []

        for tx in saving_transactions:
            if tx.transaction_type != 'open':
                continue

            # Skip if account already exists
            if tx.saving_account_number in existing_acc_numbers:
                continue

            # Skip placeholder account numbers
            if tx.saving_account_number.startswith('SAVING_'):
                continue

            original = tx.original_movement

            # Create placeholder SavingAccount with available data
            new_account = SavingAccount(
                account_number=tx.saving_account_number,
                entity_code=original.entity or "",
                entity_name=original.entity or "",
                bank_branch=original.bank_code or "",
                bank_name=original.bank_code or "",
                currency=original.currency or "VND",
                grouping=original.grouping or "Subsidiaries",
                opening_date=original.transaction_date or date.today(),
                maturity_date=original.transaction_date or date.today(),  # Placeholder - needs manual update
                term_months=0,  # Unknown - needs manual update
                interest_rate=Decimal(0),  # Unknown - needs manual update
                principal_amount=tx.amount,
                linked_current_account=original.account_number,
                needs_review=True,
                review_notes="Mở mới HĐTG - cần bổ sung: term, interest rate, maturity date",
            )

            new_accounts.append(new_account)
            existing_acc_numbers.add(tx.saving_account_number)

        return new_accounts
