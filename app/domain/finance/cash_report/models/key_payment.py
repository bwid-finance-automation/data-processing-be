"""
Key Payment Mapping - maps transaction descriptions to categories.
Contains built-in system mappings (Key payment -> Category) that don't require user upload.

Keyword matching priority: most specific first, generic defaults last.
Entity code detection for intercompany (BW Industrial group) transfers.
"""
import re
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
    "Capital contribution": "Internal transfer out",
    "Shareholder loan transfer": "Internal transfer out",  # Payment = transfer OUT
    "Loan payment": "Loan repayment",
    "Loan interest": "Loan interest",
    "Tax payment": "Operating expense",
    "Payment for acquisition": "Payment for acquisition",
    "Other payments": "Operating expense",
    "Other payment": "Operating expense",  # Alternative name
    "Dividend paid (inside group)": "Dividend paid (inside group)",
    "Internal transfer": "Internal transfer out",
}

# Receipt (Cash In) - Key payment name -> Category
RECEIPT_KEY_TO_CATEGORY: Dict[str, str] = {
    "Refund land deposit payment": "Receipt from tenants",
    "Refund deal deposit payment": "Receipt from tenants",
    "Shareholder loan transfer": "Internal transfer in",  # Receipt = transfer IN
    "Capital contribution": "Internal transfer in",
    "Rental income": "Receipt from tenants",
    "Other charge to tenant": "Receipt from tenants",
    "Booking fee from tenant": "Receipt from tenants",
    "Bank interest": "Other receipts",
    "Rental deposits": "Receipt from tenants",
    "Transfer account": "Internal transfer in",  # Receipt = transfer IN
    "Loan receipts": "Loan receipts",
    "Corporate Loan drawdown": "Corporate Loan drawdown",
    "VAT refund": "VAT refund",
    "Refinancing": "Refinancing",
    "Other receipts": "Other receipts",
    "Loan drawdown": "Loan drawdown",
    "Dividend receipt (inside group)": "Dividend receipt (inside group)",
    "Management fee from subsidiaries": "Internal transfer in",
    "Internal transfer": "Internal transfer in",
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


# ── BW Industrial group entity codes ──
# Used to detect intercompany transfers via "XXX_YYY_" patterns in descriptions.
_ENTITY_CODES = {
    "BB3", "BB4", "BB5", "BB6", "BBA", "BBN", "BCI", "BDG", "BDH", "BHP",
    "BLB", "BNA", "BNT", "BSG", "BTD", "BTH", "BTP", "BTU", "BWD", "BWP",
    "BYP", "DC2", "DVC", "GTC", "H5H", "H5S", "HD2", "HD3", "IRE", "JHP",
    "LCC", "MHN", "NSHL", "NT2", "PAE", "PDE", "PDHD", "PNA", "QV2",
    "S5B", "S5D", "SC2", "ST2", "STL", "TCS", "TPT", "VC3", "WL2A", "WL2B",
    "XA1", "BWI", "BWID", "HOLDCO", "SPV", "SPV4B", "SPV5B", "SPV5C", "SPV5D",
    "BW ", "SHTQ",
}

# Pre-compile regex for entity code pair detection: "XXX_YYY_" or "XXX-YYY-"
_ENTITY_PAIR_PATTERN = re.compile(
    r'(?:^|[\s./])('
    + '|'.join(re.escape(code) for code in sorted(_ENTITY_CODES, key=len, reverse=True))
    + r')[_\-]('
    + '|'.join(re.escape(code) for code in sorted(_ENTITY_CODES, key=len, reverse=True))
    + r')[_\-]',
    re.IGNORECASE
)

# Pattern for single entity code at start followed by underscore (e.g., "VC3_Cash transfer to PAE")
_ENTITY_START_PATTERN = re.compile(
    r'^(?:So GD[^:]*:\s*\S+\s+)?('
    + '|'.join(re.escape(code) for code in sorted(_ENTITY_CODES, key=len, reverse=True))
    + r')[_]',
    re.IGNORECASE
)


def _is_intercompany_transfer(description: str) -> bool:
    """
    Detect if a transaction is an intercompany transfer within BW Industrial group.
    Checks for entity code pairs (XXX_YYY_) or single entity code at start
    with internal transfer keywords.

    Returns True if the description matches intercompany transfer patterns.
    """
    desc_upper = description.upper()

    # Skip REM (external remittances) - these are payments from/to external parties
    if desc_upper.startswith("REM "):
        return False

    # Check for entity code pair: "XXX_YYY_..." (strongest signal)
    if _ENTITY_PAIR_PATTERN.search(description):
        return True

    # Check for single entity code at start with transfer-related content
    if _ENTITY_START_PATTERN.search(description):
        # Entity code at start - check for transfer/SHL/BCC/management fee keywords
        transfer_signals = [
            "SHL", "BCC", "TRANSFER", "CHUYEN TIEN", "MANAGEMENT FEE",
            "CHI PHI DICH VU", "INTERCO", "VAY NOI BO", "REPAYMENT",
            "GOP VON", "CAPITAL CONTRIBUTION", "CHARTER CAPITAL",
            "PAYMENT FOR DEVELOPMENT", "PAYMENT FM", "SUBLEASING",
        ]
        if any(kw in desc_upper for kw in transfer_signals):
            return True

    # Check for VCBCSH prefix - only if it also contains entity code pairs
    # VCBCSH can be vendor payments (Operating expense) or intercompany transfers
    if desc_upper.startswith("VCBCSH"):
        # Look for entity code patterns within the VCBCSH description
        found_codes = []
        for code in _ENTITY_CODES:
            code_upper = code.upper().strip()
            if not code_upper:
                continue
            # Look for entity codes as standalone tokens (not part of longer words)
            # Check various separators: space, dot, underscore, hyphen
            patterns = [
                f".{code_upper} ",  # after dot: ".BLB "
                f" {code_upper} ",  # between spaces
                f" {code_upper}_",  # space then underscore
                f".{code_upper}_",  # dot then underscore
            ]
            if any(p in desc_upper for p in patterns):
                found_codes.append(code_upper)
                if len(found_codes) >= 2:
                    return True
        return False

    # Check for "CONG TY CO PHAN PHAT TRIEN CONG NGHIEP BW" (BW Industrial full name)
    # Only if combined with entity transfer signals
    if "PHAT TRIEN CONG NGHIEP BW" in desc_upper:
        transfer_signals = ["TRANSFER", "CHUYEN", "SHL", "BCC"]
        if any(kw in desc_upper for kw in transfer_signals):
            return True

    return False


# ── DEFAULT KEY PAYMENT MAPPINGS ──
# Ordered from MOST SPECIFIC to MOST GENERIC within each payment type.
# First match wins, so specific rules MUST come before generic ones.

DEFAULT_KEY_PAYMENT_MAPPINGS: List[KeyPaymentMapping] = [
    # ================================================================
    # RECEIPTS (Cash In) - ordered by specificity
    # ================================================================

    # ── 1. INTERNAL TRANSFER IN (highest priority for receipts) ──
    # Savings withdrawals / settlements
    KeyPaymentMapping(
        key_payment="Internal transfer",
        category="Internal transfer in",
        payment_type=PaymentType.RECEIPT,
        keywords=[
            # Deposit principal return (NOT interest - "Tra lai" goes to AI)
            "Tra goc TK tien gui",
            # BCC / SHL
            "BCC transfer", "BCC TRANSFER",
            "chuyen tien hop tac kinh doanh",
            "CHUYEN TIEN HOP TAC KINH DOANH",
            "SHL", "shareholder loan", "SHAREHOLDER LOAN",
            "vay noi bo", "VAY NOI BO",
            "Interco loan", "INTERCO LOAN",
            # Internal transfer keywords
            "INTERNAL TRANSFER", "internal transfer",
            "Internal transfer for Payroll",
            # UOB format
            "MISC CREDIT",
            # Entity-specific patterns
            "TRANSFER MONEY UNDER BCC",
            "CHUYEN TIEN THEO HOP DONG BCC",
            # Capital contribution between entities
            "capital contribution", "gop von", "Charter capital",
            "gop von dieu le",
        ]
    ),

    # ── 2. OTHER RECEIPTS (bank interest only) ──
    KeyPaymentMapping(
        key_payment="Bank interest",
        category="Other receipts",
        payment_type=PaymentType.RECEIPT,
        keywords=[
            # Bank interest credit keywords
            "CRIN",
            "DDA Interest",
            "Deposit Interest Credit",
            "ghi co lai tien gui",
            "ghi co lai",
            "lai nhap von",
            "tra lai tai khoan DDA",
            "thanh toan lai thang",
            "Thanh toan lai tai khoan tien gui",
            "hoan ta LCT",
            "TFR",
        ]
    ),

    # ── 3. REFINANCING ──
    KeyPaymentMapping(
        key_payment="Refinancing",
        category="Refinancing",
        payment_type=PaymentType.RECEIPT,
        keywords=["giai ngan bu dap", "refinancing", "KLHT"]
    ),

    # ── 4. LOAN RECEIPTS ──
    KeyPaymentMapping(
        key_payment="Loan receipts",
        category="Loan receipts",
        payment_type=PaymentType.RECEIPT,
        keywords=["giai ngan khoan vay"]
    ),

    # ── 5. VAT REFUND ──
    KeyPaymentMapping(
        key_payment="VAT refund",
        category="VAT refund",
        payment_type=PaymentType.RECEIPT,
        keywords=["vat refund", "hoan thue"]
    ),

    # ── 6. DIVIDEND RECEIPT ──
    KeyPaymentMapping(
        key_payment="Dividend receipt (inside group)",
        category="Dividend receipt (inside group)",
        payment_type=PaymentType.RECEIPT,
        keywords=["dividend", "co tuc", "chia loi nhuan", "profit distribution"]
    ),

    # ── 7. RECEIPT FROM TENANTS (specific keywords) ──
    KeyPaymentMapping(
        key_payment="Rental income",
        category="Receipt from tenants",
        payment_type=PaymentType.RECEIPT,
        keywords=[
            # Rental keywords
            "thue xuong", "thue kho", "tien thue",
            "rental fee", "rent fee", "rent for factory",
            "RENTING WAREHOUSE", "RENTAL FACTORY",
            # Utility charges from tenants
            "dien nuoc", "tien dien", "tien nuoc",
            "phi tien ich",
            # Specific tenant-related
            "TT tien thue",
        ]
    ),

    # ── 8. CATCH-ALL RECEIPT → send to AI (no keywords) ──
    KeyPaymentMapping(
        key_payment="Other receipts",
        category="Receipt from tenants",
        payment_type=PaymentType.RECEIPT,
        keywords=[]  # No keywords = never matches = falls through to default
    ),

    # ================================================================
    # PAYMENTS (Cash Out) - ordered by specificity
    # ================================================================

    # ── 1. INTERNAL TRANSFER OUT (highest priority for payments) ──
    KeyPaymentMapping(
        key_payment="Internal transfer",
        category="Internal transfer out",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            # Savings/deposit opening
            "GUI HDTG", "MO HDTG", "HDTG ",
            "HOP DONG TIEN GUI",
            "CK SANG TK TIME", "TIMEMO",
            "Tien gui co ky han",
            "TKKH",
            # Transfer to CD account
            "Completed transfer to BIDV CD",
            # BCC / SHL
            "BCC transfer", "BCC TRANSFER",
            "chuyen tien hop tac kinh doanh",
            "CHUYEN TIEN HOP TAC KINH DOANH",
            "SHL", "shareholder loan", "SHAREHOLDER LOAN",
            "vay noi bo", "VAY NOI BO",
            "Interco loan", "INTERCO LOAN",
            # Internal transfer keywords
            "INTERNAL TRANSFER", "internal transfer",
            "Internal transfer for Payroll",
            "CHUYEN KHOAN" + "INTERNAL",  # Combined pattern handled below
            # UOB format
            "MISC DEBIT",
            # Entity-specific patterns
            "TRANSFER MONEY UNDER BCC",
            "CHUYEN TIEN THEO HOP DONG BCC",
            # Rollover
            "ROLLOVER",
            # Capital contribution between entities
            "capital contribution", "gop von", "Charter capital",
            "gop von dieu le",
        ]
    ),

    # ── 2. LOAN REPAYMENT (bank loans only) ──
    KeyPaymentMapping(
        key_payment="Loan payment",
        category="Loan repayment",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            "Tra no TK vay",
            "THU NO GOC VAY",
            "THU NO GOC",
            "THU NO TAI KHOAN VAY",
            "THU NO TKV",
            "LD-PR",
            "REPAY LOAN",
            "TRA NO VAY",
        ]
    ),

    # ── 3. LOAN INTEREST (bank loan interest) ──
    KeyPaymentMapping(
        key_payment="Loan interest",
        category="Loan interest",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            "Thu no LAI",
            "Thu no lai (LD-IN)",
            "LD-IN",
            "lai phat sinh",
        ]
    ),

    # ── 4. CONSTRUCTION EXPENSE (specific contractors - only very specific ones) ──
    # NOTE: Many construction company names also do repair/maintenance work (→ Operating expense).
    # Only include contractors that are VERY strongly associated with new construction (CIP).
    # Ambiguous ones (KANSAI VINA, CENTRAL, HBC, etc.) should go to AI for context analysis.
    KeyPaymentMapping(
        key_payment="Construction expense",
        category="Construction expense",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            "ICIC", "VEV", "GLC", "NEWTECONS", "VITECCONS",
            "MEGASPACE", "Earth Works",
            "INTERIM PAYMENT",
        ]
    ),

    # ── 5. DEAL PAYMENT ──
    KeyPaymentMapping(
        key_payment="Deal payment",
        category="Deal payment",
        payment_type=PaymentType.PAYMENT,
        keywords=["Final payment SPA", "SPA"]
    ),

    # ── 6. DIVIDEND PAID ──
    KeyPaymentMapping(
        key_payment="Dividend paid (inside group)",
        category="Dividend paid (inside group)",
        payment_type=PaymentType.PAYMENT,
        keywords=["dividend", "co tuc", "chia loi nhuan", "profit distribution"]
    ),

    # ── 7. TAX PAYMENTS (specific) ──
    KeyPaymentMapping(
        key_payment="Tax payment",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=["NOP NSNN", "NOPTHUE", "/NOP NSNN"]
    ),

    # ── 8. BANK FEES (specific) ──
    KeyPaymentMapping(
        key_payment="Bank charge",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            "SC60", "VATX", "EMF", "IMF",
            "Recovery Charge", "Phi phat hanh",
            "Phi duy tri tai khoan", "Phi QUAN LY tai khoan",
            "THU Phi QLTK TO CHUC",
            "Thu phi duy tri dich vu",
        ]
    ),

    # ── 9. OPERATING EXPENSE (other specific patterns) ──
    KeyPaymentMapping(
        key_payment="Operating expense",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=[
            "REFUND BOOKING FEE",
            "Refund Security deposit",
            "AUDIT",
            "RECRUITMENT",
            "REPAIR",
            "WWTP", "PMC", "EES",
            "IMPC", "BECAMEX", "BA DUONG", "KZN",
        ]
    ),

    # ── 10. CATCH-ALL PAYMENT → send to AI (no keywords) ──
    KeyPaymentMapping(
        key_payment="Other payment",
        category="Operating expense",
        payment_type=PaymentType.PAYMENT,
        keywords=[]  # No keywords = never matches = falls through to default
    ),
]


class KeyPaymentClassifier:
    """
    Classifier to determine transaction nature based on description.
    Uses keyword matching with entity code detection for intercompany transfers.
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

        Uses a two-phase approach:
        1. Check for intercompany entity code patterns (highest priority)
        2. Keyword matching against the mapping list (specific → generic)

        Args:
            description: Transaction description/memo
            is_debit: True if debit (cash in), False if credit (cash out)

        Returns:
            Tuple of (key_payment, category, payment_type)
        """
        expected_type = PaymentType.RECEIPT if is_debit else PaymentType.PAYMENT

        # Phase 1: Entity code detection for intercompany transfers
        if _is_intercompany_transfer(description):
            if is_debit:
                return "Internal transfer", "Internal transfer in", PaymentType.RECEIPT
            else:
                return "Internal transfer", "Internal transfer out", PaymentType.PAYMENT

        # Phase 2: Keyword matching (first match wins)
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
