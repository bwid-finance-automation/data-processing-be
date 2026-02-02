"""
Movement Classifier Service - classifies transactions based on description keywords.
"""
import re
from typing import List, Optional, Dict, Tuple
from decimal import Decimal

from ..models.cash_movement import CashMovement, TransactionNature
from ..models.key_payment import KeyPaymentClassifier, PaymentType, KeyPaymentMapping


class MovementClassifier:
    """
    Service to classify cash movements based on their descriptions.
    Uses keyword matching and pattern recognition.
    """

    def __init__(self, custom_mappings: Optional[List[KeyPaymentMapping]] = None):
        self.classifier = KeyPaymentClassifier(custom_mappings)
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for common transaction types"""
        self.patterns = {
            # Internal transfers
            'internal_transfer': [
                r'transfer\s+(?:to|from)',
                r'chuyển\s+(?:khoản|tiền)',
                r'chuyen\s+(?:khoan|tien)',
                r'internal\s+transfer',
                r'nội\s*bộ',
                r'noi\s*bo',
            ],
            # Saving account related
            'saving_account': [
                r'tiền\s*gửi',
                r'tien\s*gui',
                r'saving',
                r'hđtk',
                r'hdtk',
                r'term\s*deposit',
                r'tất\s*toán',
                r'tat\s*toan',
                r'mở\s*(?:mới|tk)',
                r'mo\s*(?:moi|tk)',
            ],
            # Rental income
            'rental': [
                r'rental',
                r'rent\s+(?:payment|income)',
                r'thuê',
                r'thue',
                r'cho\s+thuê',
            ],
            # Loan related
            'loan': [
                r'loan\s+(?:payment|drawdown|repayment)',
                r'vay',
                r'trả\s*nợ',
                r'tra\s*no',
                r'giải\s*ngân',
                r'giai\s*ngan',
            ],
            # Tax
            'tax': [
                r'tax\s+payment',
                r'thuế',
                r'thue',
                r'nộp\s+thuế',
            ],
            # Salary
            'salary': [
                r'salary',
                r'payroll',
                r'lương',
                r'luong',
            ],
            # Interest
            'interest': [
                r'interest',
                r'lãi',
                r'lai',
                r'tiền\s+lãi',
            ],
            # Bank charge
            'bank_charge': [
                r'bank\s+(?:charge|fee)',
                r'phí\s+(?:ngân\s+hàng|dịch\s+vụ)',
                r'phi\s+(?:ngan\s+hang|dich\s+vu)',
                r'service\s+fee',
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in self.patterns.items()
        }

    def classify_movement(self, movement: CashMovement) -> CashMovement:
        """
        Classify a single movement and update its nature and key_payment fields.

        Args:
            movement: CashMovement to classify

        Returns:
            Updated CashMovement with nature and key_payment set
        """
        is_debit = movement.debit_amount is not None and movement.debit_amount > 0
        description = movement.description

        # First check for special patterns
        if self._matches_pattern(description, 'internal_transfer'):
            if is_debit:
                movement.nature = TransactionNature.INTERNAL_TRANSFER_IN
                movement.key_payment = "Internal transfer in"
            else:
                movement.nature = TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT
                movement.key_payment = "Internal transfer out"
            return movement

        if self._matches_pattern(description, 'saving_account'):
            movement.nature = TransactionNature.INTERNAL_TRANSFER_IN if is_debit else TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT
            movement.key_payment = "Transfer account"
            return movement

        # Use classifier for other transactions
        key_payment, category, payment_type = self.classifier.classify(description, is_debit)
        movement.key_payment = key_payment
        movement.nature = self._map_category_to_nature(category, is_debit)

        return movement

    def classify_movements(self, movements: List[CashMovement]) -> List[CashMovement]:
        """
        Classify a list of movements.

        Args:
            movements: List of CashMovement to classify

        Returns:
            List of classified CashMovement
        """
        return [self.classify_movement(m) for m in movements]

    def _matches_pattern(self, text: str, pattern_key: str) -> bool:
        """Check if text matches any pattern in the given category"""
        if pattern_key not in self.compiled_patterns:
            return False
        return any(p.search(text) for p in self.compiled_patterns[pattern_key])

    def _map_category_to_nature(self, category: str, is_debit: bool) -> TransactionNature:
        """Map category string to TransactionNature enum"""
        category_lower = category.lower()

        # Mapping for receipts (cash in)
        receipt_mapping = {
            'receipt from tenants': TransactionNature.RECEIPT_FROM_TENANTS,
            'refund land/deal deposit payment': TransactionNature.REFUND_LAND_DEAL_DEPOSIT,
            'cash from wp': TransactionNature.CASH_FROM_WP,
            'internal contribution': TransactionNature.INTERNAL_CONTRIBUTION,
            'internal transfer': TransactionNature.INTERNAL_TRANSFER_IN,
            'other receipts': TransactionNature.OTHER_RECEIPTS,
            'contribution': TransactionNature.CONTRIBUTION,
            'loan receipts': TransactionNature.LOAN_RECEIPTS,
            'corporate loan drawdown': TransactionNature.CORPORATE_LOAN_DRAWDOWN,
            'vat refund': TransactionNature.VAT_REFUND,
            'cash received from acquisition': TransactionNature.CASH_RECEIVED_FROM_ACQUISITION,
            'loan drawdown': TransactionNature.LOAN_DRAWDOWN,
            'dividend receipt (inside group)': TransactionNature.DIVIDEND_RECEIPT_INSIDE_GROUP,
            'management fee from subsidiaries': TransactionNature.MANAGEMENT_FEE_FROM_SUBSIDIARIES,
            'refinancing': TransactionNature.REFINANCING,
            'loan repayment': TransactionNature.LOAN_REPAYMENT_RECEIPT,
        }

        # Mapping for payments (cash out)
        payment_mapping = {
            'operating expense': TransactionNature.OPERATING_EXPENSE,
            'construction expense': TransactionNature.CONSTRUCTION_EXPENSE,
            'land acquisition': TransactionNature.LAND_ACQUISITION,
            'deal payment': TransactionNature.DEAL_PAYMENT,
            'internal transfer': TransactionNature.INTERNAL_TRANSFER_OUT_EXPLICIT,
            'loan repayment': TransactionNature.LOAN_REPAYMENT,
            'loan interest': TransactionNature.LOAN_INTEREST,
            'payment for acquisition': TransactionNature.PAYMENT_FOR_ACQUISITION,
            'dividend paid (inside group)': TransactionNature.DIVIDEND_PAID_INSIDE_GROUP,
            'internal contribution': TransactionNature.INTERNAL_CONTRIBUTION_OUT,
        }

        mapping = receipt_mapping if is_debit else payment_mapping

        for key, nature in mapping.items():
            if key in category_lower:
                return nature

        return TransactionNature.UNKNOWN

    def get_classification_stats(self, movements: List[CashMovement]) -> Dict[str, int]:
        """
        Get statistics about classified movements.

        Returns:
            Dict with nature -> count
        """
        stats = {}
        for m in movements:
            nature = m.nature.value if m.nature else "Unclassified"
            stats[nature] = stats.get(nature, 0) + 1
        return stats

    def get_unclassified_movements(self, movements: List[CashMovement]) -> List[CashMovement]:
        """Get list of movements that couldn't be classified"""
        return [m for m in movements if m.nature is None or m.nature == TransactionNature.UNKNOWN]

    def suggest_classification(self, description: str) -> List[Tuple[str, float]]:
        """
        Suggest possible classifications for a description with confidence scores.

        Returns:
            List of (nature, confidence) tuples sorted by confidence
        """
        suggestions = []

        for pattern_key, patterns in self.compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(description))
            if matches > 0:
                confidence = min(matches / len(patterns), 1.0)
                suggestions.append((pattern_key, confidence))

        return sorted(suggestions, key=lambda x: x[1], reverse=True)
