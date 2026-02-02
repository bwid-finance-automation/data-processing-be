"""
Key Payment Mapping - maps transaction descriptions to categories.
Contains built-in system mappings (Key payment -> Category) that don't require user upload.
"""
from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class PaymentType(str, Enum):
    """Type of transaction - Payment (cash out) or Receipt (cash in)"""
    PAYMENT = "Payment"
    RECEIPT = "Receipt"


# =============================================================================
# SYSTEM BUILT-IN MAPPING: Key payment -> Category
# This is the master mapping table stored in the system.
# Users do NOT need to upload Def sheet - the system uses these mappings.
# =============================================================================

# Payment (Cash Out) - Key payment name -> Category
PAYMENT_KEY_TO_CATEGORY: Dict[str, str] = {
    "G&A expense": "Operating expense",
    "Construction expense": "Construction expense",
    "Repair and maintenance expense": "Operating expense",
    "Professional service fee": "Operating expense",
    "Land deposit payment": "Land acquisition",
    "Land lease price payment": "Land acquisition",
    "Deal payment": "Deal payment",
    "Fee related land payment": "Construction expense",
    "S&M expense": "Operating expense",
    "Bank charge": "Operating expense",
    "Salary expense": "Operating expense",
    "Transfer account": "Internal transfer out",  # Payment = transfer OUT
    "Return booking fee": "Operating expense",
    "Return rental deposit": "Operating expense",
    "Capital contribution": "Internal Contribution",
    "Shareholder loan transfer": "Internal transfer out",  # Payment = transfer OUT
    "Loan payment": "Loan repayment",
    "Loan interest": "Loan interest",
    "Tax payment": "Operating expense",
    "Payment for acquisition": "Payment for acquisition",
    "Other payments": "Operating expense",
    "Other payment": "Operating expense",  # Alternative name
    "Dividend paid (inside group)": "Dividend paid (inside group)",
}

# Receipt (Cash In) - Key payment name -> Category
RECEIPT_KEY_TO_CATEGORY: Dict[str, str] = {
    "Refund land deposit payment": "Refund land/deal deposit payment",
    "Refund deal deposit payment": "Refund land/deal deposit payment",
    "Shareholder loan received from WP": "Cash from WP",
    "Shareholder loan transfer": "Internal transfer in",  # Receipt = transfer IN
    "Capital contribution": "Internal Contribution",
    "Rental income": "Receipt from tenants",
    "Other charge to tenant": "Receipt from tenants",
    "Booking fee from tenant": "Receipt from tenants",
    "Bank interest": "Other receipts",
    "Rental deposits": "Receipt from tenants",
    "Transfer account": "Internal transfer in",  # Receipt = transfer IN
    "Capital receipts": "Contribution",
    "Capital receipts from WP": "Contribution",
    "Capital receipts from Becamex": "Contribution",
    "Loan receipts": "Loan receipts",
    "Corporate Loan drawdown": "Corporate Loan drawdown",
    "VAT refund": "VAT refund",
    "Refinancing": "Refinancing",
    "Cash received from acquisition": "Cash received from acquisition",
    "Cash balance on MBC account": "N/a",
    "Other receipts": "Other receipts",
    "Loan drawdown": "Loan drawdown",
    "Dividend receipt (inside group)": "Dividend receipt (inside group)",
    "Management fee from subsidiaries": "Management fee from subsidiaries",
}


def get_category_from_key_payment(key_payment: str, is_receipt: bool) -> Optional[str]:
    """
    Get category from key payment name using built-in system mapping.
    This does NOT require user to upload Def sheet.

    Args:
        key_payment: The key payment name (e.g., "Rental income", "G&A expense")
        is_receipt: True if Receipt (cash in), False if Payment (cash out)

    Returns:
        Category string or None if not found
    """
    if is_receipt:
        return RECEIPT_KEY_TO_CATEGORY.get(key_payment)
    else:
        return PAYMENT_KEY_TO_CATEGORY.get(key_payment)


class KeyPaymentMapping(BaseModel):
    """
    Mapping from key payment to category and type.
    Used to classify transactions based on description keywords.
    """
    key_payment: str = Field(description="Key payment name (e.g., 'Rental income')")
    category: str = Field(description="Category (e.g., 'Receipt from tenants')")
    payment_type: PaymentType = Field(description="Payment or Receipt")
    keywords: List[str] = Field(default_factory=list, description="Keywords to match in description")

    def matches_description(self, description: str) -> bool:
        """Check if description matches any keyword"""
        desc_lower = description.lower()
        return any(kw.lower() in desc_lower for kw in self.keywords)


# Default key payment mappings based on the sample file
DEFAULT_KEY_PAYMENT_MAPPINGS: List[KeyPaymentMapping] = [
    # RECEIPTS (Cash In)
    KeyPaymentMapping(
        key_payment="Rental income",
        category="Receipt from tenants",
        payment_type=PaymentType.RECEIPT,
        keywords=[
            "rental", "rent", "thuê", "thue", "cho thuê", "cho thue",
            "payment for", "paid for", "INV.", "invoice",  # Common invoice patterns
            "thanh toán", "thanh toan", "tt ", "TT ",  # Vietnamese payment terms
            "fee from", "tiền thuê", "tien thue"
        ]
    ),
    KeyPaymentMapping(
        key_payment="Booking fee from tenant",
        category="Receipt from tenants",
        payment_type=PaymentType.RECEIPT,
        keywords=["booking fee", "đặt cọc", "dat coc", "deposit"]
    ),
    KeyPaymentMapping(
        key_payment="Other charge to tenant",
        category="Receipt from tenants",
        payment_type=PaymentType.RECEIPT,
        keywords=["service charge", "phí dịch vụ", "phi dich vu", "management fee", "electricity", "điện", "dien"]
    ),
    KeyPaymentMapping(
        key_payment="Transfer account",
        category="Internal transfer in",  # Receipt = transfer IN
        payment_type=PaymentType.RECEIPT,
        keywords=["ck sang tk", "CK SANG TK", "timemo", "TIMEMO", "transfer in", "nội bộ vào", "noi bo vao"]
    ),
    KeyPaymentMapping(
        key_payment="Bank interest",
        category="Other receipts",
        payment_type=PaymentType.RECEIPT,
        keywords=["interest income", "lãi tiền gửi", "lai tien gui"]
    ),
    KeyPaymentMapping(
        key_payment="Capital contribution",
        category="Internal Contribution",
        payment_type=PaymentType.RECEIPT,
        keywords=["capital", "góp vốn", "gop von", "contribution"]
    ),
    KeyPaymentMapping(
        key_payment="Shareholder loan received from WP",
        category="Cash from WP",
        payment_type=PaymentType.RECEIPT,
        keywords=["shareholder loan", "vay cổ đông", "vay co dong", "loan from wp"]
    ),
    KeyPaymentMapping(
        key_payment="Corporate Loan drawdown",
        category="Corporate Loan drawdown",
        payment_type=PaymentType.RECEIPT,
        keywords=["loan drawdown", "giải ngân", "giai ngan", "drawdown"]
    ),
    KeyPaymentMapping(
        key_payment="VAT refund",
        category="VAT refund",
        payment_type=PaymentType.RECEIPT,
        keywords=["vat refund", "hoàn thuế", "hoan thue", "refund vat"]
    ),
    KeyPaymentMapping(
        key_payment="Refund land deposit payment",
        category="Refund land/deal deposit payment",
        payment_type=PaymentType.RECEIPT,
        keywords=["refund deposit", "hoàn cọc", "hoan coc", "trả cọc", "tra coc"]
    ),
    KeyPaymentMapping(
        key_payment="Other receipts",
        category="Receipt from tenants",  # Default Receipt to Receipt from tenants
        payment_type=PaymentType.RECEIPT,
        keywords=[]  # Catch-all for unmatched receipts - default to Receipt from tenants
    ),

    # PAYMENTS (Cash Out)
    KeyPaymentMapping(
        key_payment="G&A expense",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["g&a", "admin", "văn phòng phẩm", "van phong pham"]
    ),
    KeyPaymentMapping(
        key_payment="Construction expense",
        category="Construction expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["construction", "xây dựng", "xay dung", "contractor", "nhà thầu", "nha thau"]
    ),
    KeyPaymentMapping(
        key_payment="Repair and maintenance expense",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["repair", "maintenance", "sửa chữa", "sua chua", "bảo trì", "bao tri"]
    ),
    KeyPaymentMapping(
        key_payment="Professional service fee",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["professional", "consulting", "tư vấn", "tu van", "legal", "audit", "lawyer"]
    ),
    KeyPaymentMapping(
        key_payment="Land deposit payment",
        category="Land acquisition",
        payment_type=PaymentType.PAYMENT,
        keywords=["land deposit", "đặt cọc đất", "dat coc dat"]
    ),
    KeyPaymentMapping(
        key_payment="Land lease price payment",
        category="Land acquisition",
        payment_type=PaymentType.PAYMENT,
        keywords=["land lease", "tiền thuê đất", "tien thue dat", "land rent"]
    ),
    KeyPaymentMapping(
        key_payment="Bank charge",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["bank charge", "phí ngân hàng", "phi ngan hang", "bank fee", "service fee"]
    ),
    KeyPaymentMapping(
        key_payment="Salary expense",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["salary", "lương", "luong", "wage", "payroll"]
    ),
    KeyPaymentMapping(
        key_payment="Tax payment",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["tax", "thuế", "thue", "nộp thuế", "nop thue"]
    ),
    KeyPaymentMapping(
        key_payment="Loan payment",
        category="Loan repayment",
        payment_type=PaymentType.PAYMENT,
        keywords=["loan payment", "trả nợ", "tra no", "repayment"]
    ),
    KeyPaymentMapping(
        key_payment="Loan interest",
        category="Loan interest",
        payment_type=PaymentType.PAYMENT,
        keywords=["loan interest", "lãi vay", "lai vay", "interest payment"]
    ),
    KeyPaymentMapping(
        key_payment="Capital contribution",
        category="Internal contribution",
        payment_type=PaymentType.PAYMENT,
        keywords=["capital contribution", "góp vốn", "gop von"]
    ),
    KeyPaymentMapping(
        key_payment="Transfer account",
        category="Internal transfer out",  # Payment = transfer OUT
        payment_type=PaymentType.PAYMENT,
        keywords=["ck sang tk", "CK SANG TK", "timemo", "TIMEMO", "transfer out", "nội bộ ra", "noi bo ra", "chuyển nội bộ", "chuyen noi bo"]
    ),
    KeyPaymentMapping(
        key_payment="Shareholder loan transfer",
        category="Internal transfer out",  # Payment = transfer OUT
        payment_type=PaymentType.PAYMENT,
        keywords=["shareholder loan", "cho vay cổ đông", "cho vay co dong"]
    ),
    KeyPaymentMapping(
        key_payment="S&M expense",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["marketing", "sales", "quảng cáo", "quang cao", "s&m"]
    ),
    KeyPaymentMapping(
        key_payment="Dividend paid (inside group)",
        category="Dividend paid (inside group)",
        payment_type=PaymentType.PAYMENT,
        keywords=["dividend", "cổ tức", "co tuc", "chia lợi nhuận", "chia loi nhuan"]
    ),
    KeyPaymentMapping(
        key_payment="Other payment",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=[]  # Catch-all for unmatched payments
    ),
]


class KeyPaymentClassifier:
    """
    Classifier to determine transaction nature based on description.
    Also provides direct lookup from Key payment -> Category using built-in system mapping.
    """

    def __init__(self, mappings: Optional[List[KeyPaymentMapping]] = None):
        self.mappings = mappings or DEFAULT_KEY_PAYMENT_MAPPINGS

    def get_category_by_key_payment(self, key_payment: str, is_receipt: bool) -> Optional[str]:
        """
        Get category directly from key payment name using BUILT-IN SYSTEM MAPPING.
        This does NOT require user to upload Def sheet.

        Args:
            key_payment: The key payment name (e.g., "Rental income", "G&A expense")
            is_receipt: True if Receipt (cash in), False if Payment (cash out)

        Returns:
            Category string or None if not found
        """
        return get_category_from_key_payment(key_payment, is_receipt)

    def classify(self, description: str, is_debit: bool) -> tuple[str, str, PaymentType]:
        """
        Classify a transaction based on its description.

        Args:
            description: Transaction description/memo
            is_debit: True if debit (cash in), False if credit (cash out)

        Returns:
            Tuple of (key_payment, category, payment_type)
        """
        expected_type = PaymentType.RECEIPT if is_debit else PaymentType.PAYMENT

        # Find matching mapping
        for mapping in self.mappings:
            if mapping.payment_type != expected_type:
                continue
            if mapping.keywords and mapping.matches_description(description):
                return mapping.key_payment, mapping.category, mapping.payment_type

        # Default fallback
        if is_debit:
            return "Other receipts", "Other receipts", PaymentType.RECEIPT
        else:
            return "Other payment", "Operating expense", PaymentType.PAYMENT

    def add_mapping(self, mapping: KeyPaymentMapping) -> None:
        """Add a new mapping"""
        self.mappings.insert(0, mapping)  # Insert at beginning for priority

    def load_from_excel(self, key_payment_data: List[dict]) -> None:
        """
        Load mappings from Excel data (key payment sheet).

        Args:
            key_payment_data: List of dicts with 'Key payment - receipt', 'Category', 'Type'
        """
        for row in key_payment_data:
            key_payment = row.get('Key payment - receipt', '')
            category = row.get('Category', '')
            type_str = row.get('Type', '')

            if not key_payment or not category:
                continue

            payment_type = PaymentType.RECEIPT if type_str == 'Receipt' else PaymentType.PAYMENT

            # Create mapping with key_payment as keyword
            mapping = KeyPaymentMapping(
                key_payment=key_payment,
                category=category,
                payment_type=payment_type,
                keywords=[key_payment.lower()]
            )
            self.mappings.append(mapping)
