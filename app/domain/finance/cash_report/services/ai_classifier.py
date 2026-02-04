"""
AI-based Transaction Classifier using Google Gemini API.
Classifies bank transactions into Nature categories with higher accuracy than keyword matching.

Improvements for consistency:
- temperature=0 for deterministic output
- Few-shot examples in prompt for better accuracy
- Output validation against allowed categories
- Caching for identical descriptions
"""
import os
import hashlib
from typing import List, Tuple, Optional, Dict
import google.generativeai as genai

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Nature categories for Classification
PAYMENT_CATEGORIES = [
    "Operating expense",
    "Construction expense",
    "Land acquisition",
    "Deal payment",
    "Internal transfer out",
    "Internal Contribution out",  # explicit out
    "Internal contribution out",  # lowercase variant
    "Loan repayment",
    "Loan interest",
    "Payment for acquisition",
    "Dividend paid (inside group)",
]

RECEIPT_CATEGORIES = [
    "Receipt from tenants",
    "Other receipts",
    "Internal transfer in",
    "Internal Contribution in",  # explicit in
    "Internal contribution in",  # lowercase variant
    "Contribution",
    "Cash from WP",
    "Refund land/deal deposit payment",
    "Loan receipts",
    "Corporate Loan drawdown",
    "Loan drawdown",
    "VAT refund",
    "Refinancing",
    "Cash received from acquisition",
    "Dividend receipt (inside group)",
    "Management fee from subsidiaries",
]


# All valid categories for validation
ALL_PAYMENT_CATEGORIES = set(PAYMENT_CATEGORIES)
ALL_RECEIPT_CATEGORIES = set(RECEIPT_CATEGORIES)
ALL_CATEGORIES = ALL_PAYMENT_CATEGORIES | ALL_RECEIPT_CATEGORIES


class AITransactionClassifier:
    """
    AI-based classifier for bank transactions.
    Uses Google Gemini API to understand transaction descriptions and classify them.

    Features:
    - temperature=0 for deterministic, reproducible results
    - Few-shot examples for better accuracy
    - Validation against allowed categories
    - In-memory caching to avoid redundant API calls
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self._cache: Dict[str, str] = {}  # Cache: hash(description+is_receipt) -> category

        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use generation_config with temperature=0 for deterministic output
            self.generation_config = genai.GenerationConfig(
                temperature=0,  # Deterministic output - same input always gives same output
                top_p=1,
                top_k=1,
            )
            self.client = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config=self.generation_config
            )
        else:
            logger.warning("No GEMINI_API_KEY found, AI classifier will not work")

    def classify_batch(
        self,
        transactions: List[Tuple[str, bool]],  # List of (description, is_receipt)
        batch_size: int = 50
    ) -> List[str]:
        """
        Classify a batch of transactions using AI.

        Args:
            transactions: List of (description, is_receipt) tuples
            batch_size: Number of transactions per API call

        Returns:
            List of Nature category strings
        """
        if not self.client:
            logger.error("AI classifier not initialized - missing API key")
            return ["" for _ in transactions]

        results = []

        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            batch_results = self._classify_batch_internal(batch)
            results.extend(batch_results)
            logger.info(f"AI classified batch {i//batch_size + 1}: {len(batch)} transactions")

        return results

    def _get_cache_key(self, description: str, is_receipt: bool) -> str:
        """Generate cache key for a transaction."""
        key_str = f"{description.lower().strip()}|{is_receipt}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _classify_batch_internal(self, batch: List[Tuple[str, bool]]) -> List[str]:
        """Classify a single batch of transactions with caching and validation."""
        results = [""] * len(batch)
        uncached_indices = []
        uncached_batch = []

        # Check cache first
        for idx, (desc, is_receipt) in enumerate(batch):
            cache_key = self._get_cache_key(desc, is_receipt)
            if cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_batch.append((desc, is_receipt))

        if not uncached_batch:
            logger.info(f"All {len(batch)} transactions found in cache")
            return results

        logger.info(f"Cache hit: {len(batch) - len(uncached_batch)}, need to classify: {len(uncached_batch)}")

        # Build the prompt with few-shot examples
        transactions_text = ""
        for idx, (desc, is_receipt) in enumerate(uncached_batch):
            tx_type = "RECEIPT (cash in)" if is_receipt else "PAYMENT (cash out)"
            transactions_text += f"{idx + 1}. [{tx_type}] {desc}\n"

        prompt = f"""You are a financial transaction classifier for Vietnamese bank statements.
Classify each transaction into exactly ONE Nature category.

CRITICAL: The transaction type [RECEIPT] or [PAYMENT] is ALREADY DETERMINED. You must select a category that matches this type.
- [RECEIPT] = cash IN = money received → use RECEIPT categories only
- [PAYMENT] = cash OUT = money paid → use PAYMENT categories only

=== PAYMENT (cash out) categories ===
{chr(10).join(f'- {c}' for c in PAYMENT_CATEGORIES)}

=== RECEIPT (cash in) categories ===
{chr(10).join(f'- {c}' for c in RECEIPT_CATEGORIES)}

=== KEYWORD-BASED CLASSIFICATION RULES ===

**INTERNAL TRANSFER** (HIGHEST PRIORITY - check first!):
Keywords: HDTG, GUI TIEN TK, tien gui, internal transfer, transfer money, SHL, shareholder loan, SHL repayment, SAVING, tiet kiem, Repayment loan to SPV/holdco
IMPORTANT: Intercompany loan repayments (to SPV, holdco, subsidiaries) = Internal transfer, NOT Loan repayment. "Loan repayment" is only for BANK loans.
[PAYMENT] "GUI HDTG KY HAN 51D, ST: 3,000,000,000. SO HDTG: 285/2025/17687" → Internal transfer out
[PAYMENT] "GUI TIEN GUI KY HAN 1 THANG THEO HDTG SO 902/2025/18909" → Internal transfer out
[PAYMENT] "TRICH TK MO HDTG 1 THANG RGLH LCK" → Internal transfer out
[PAYMENT] "GUI TIEN VAO HOP DONG TIEN GUI SO 285 2025 16811" → Internal transfer out
[PAYMENT] "INTERNAL TRANSFER FROM VTB TO UOB" → Internal transfer out
[PAYMENT] "VC3_DC2_TRANSFER MONEY FROM VC3 TO DC2" → Internal transfer out
[PAYMENT] "GTC_VC3_Repayment sharehoder loan" → Internal transfer out
[PAYMENT] "BDH_Repayment principal loan to SPV5C tra goc vay cho SPV5C" → Internal transfer out (intercompany)
[PAYMENT] "SAVING DEPOSIT 12 MONTHS" → Internal transfer out
[RECEIPT] "TAT TOAN HDTG 640.2024.85809" → Internal transfer in
[RECEIPT] "RUT TIEN THEO PHU LUC 640/2023/29169/PL3" → Internal transfer in
[RECEIPT] "TAT TOAN TAI KHOAN TIET KIEM SO 216000382800" → Internal transfer in
[RECEIPT] "VC3_DC2_TRANSFER MONEY FROM VC3 TO DC2" → Internal transfer in
[RECEIPT] "INTERNAL TRANSFER FR SNP TO VTB.BINH DUONG" → Internal transfer in
[RECEIPT] "MISC CREDIT...Repayment sharehoder loan" → Internal transfer in
[RECEIPT] "SPV4C REPAYMENT LOAN PRINCIPAL TO..." → Internal transfer in

**CONSTRUCTION EXPENSE**:
Keywords: IPC, APG, design service, geological survey, permitting service, Main Contractor, CENTRAL, HOABINH, HBC, NEWTECONS, ICIC, ARTELIA, NAGECCO, GLC, UNICONS, VEV, RICONS, ICC, KANSAI VINA, IBST, APAVE, RLB, VINAINCON, ARCADIS, An Phu Gia, construction, COTECCONS
[PAYMENT] "DAL - ICIC - CS Contract - PR.11" → Construction expense
[PAYMENT] "DVC_VEV- Permitting contract to VEV - PR.02" → Construction expense
[PAYMENT] "BB5_GLC_Earth Works - PR.03 IPC.02" → Construction expense
[PAYMENT] "WL2B_VEV_Design Consultancy Service" → Construction expense
[PAYMENT] "BYP_COTECCONS_Interim Payment Certificate No.8" → Construction expense
[PAYMENT] "Construction Supervision Service" → Construction expense

**OPERATING EXPENSE**:
Keywords: PHI QUAN LY TAI KHOAN, PHI BSMS, THP, NOPTHUE, Thu phi duy tri dich vu, Phi duy tri tai khoan, THU Phi QLTK TO CHUC, NSNN, IMF, EMF, nuoc, dien, water, IMPC, BECAMEX, BA DUONG, KZN, NL, AB.MKH, meeting, electricity, Refund fit out deposit, Refund fitout deposit, Refund booking fee, FM service, Wastewater, scrap
IMPORTANT: Repair/typhoon damage works by contractors (even construction companies like Kansai Vina) = Operating expense, NOT Construction expense. Construction expense is only for NEW construction projects (CIP).
[PAYMENT] "PHI QUAN LY TAI KHOAN 150xxx128 T06 2025" → Operating expense
[PAYMENT] "PHI BSMS T05.2025. MA KH10725855" → Operating expense
[PAYMENT] "KTR_IDICO_Electricity fee in 062025_Term 1" → Operating expense
[PAYMENT] "TTHD Tien nuoc Ma KH-8801010100" → Operating expense
[PAYMENT] "BWD_Dan On_Refund fit out deposit" → Operating expense
[PAYMENT] "BWD_LIANGMU_Refund booking fee" → Operating expense
[PAYMENT] "NO PHI QLY TAI KHOAN" → Operating expense
[PAYMENT] "BDH_CREW24_FM Service MAY 2025_INV 86" → Operating expense
[PAYMENT] "BTP_Kansai Vina_Advance 30 contract value Works after Typhoon Yagi" → Operating expense (repair, not construction)
[PAYMENT] "BHA_DEEPC Blue_Payment of Water consumption and Wastewater treatment" → Operating expense
[PAYMENT] "BTP_BTPIZ_EMF fee in June 2025" → Operating expense
[PAYMENT] "BDH_MTTT_Refund the different amount after selling scrap" → Operating expense

**LOAN REPAYMENT**:
Keywords: GOC, principal, THU NO GOC, TAT TOAN KV, Loan Repayment, Prepayment principal
[PAYMENT] "Thu no GOC KU659541" → Loan repayment
[PAYMENT] "THU NO BO SUNG NAM 2024 DOT 2" → Loan repayment
[PAYMENT] "BHD THANH TOAN MOT PHAN GOC VAY NGAN HANG" → Loan repayment
[PAYMENT] "TRA 1 PHAN GOC TK VAY 806006543309" → Loan repayment
[PAYMENT] "TAT TOAN KV 401500272759" → Loan repayment
[PAYMENT] "THU GOC TK 803005764624" → Loan repayment
[PAYMENT] "Loan Repayment at Counter - Prepayment principal" → Loan repayment

**LOAN INTEREST**:
Keywords: Thu no (without GOC), thanh toan lai TD, thanh toan lai, tra no TK vay, thu lai
[PAYMENT] "Thu no KU404000876719" → Loan interest (no GOC = interest only)
[PAYMENT] "Tra no TK vay 402220660075,so tien 215326582 VND" → Loan interest
[PAYMENT] "Thu no GOC LAI T6.2025 KV 401500276964" → Loan interest (combined principal+interest)

**DIVIDEND PAID (inside group)**:
Keywords: dividend, co tuc, Profit Distribution, Phan phoi loi nhuan, TAM UNG PHAN LOI NHUAN
[PAYMENT] "BHP_2nd installment of Profit Distribution for FY2024" → Dividend paid (inside group)
[PAYMENT] "Profit distribution for FY2024 3rd payment" → Dividend paid (inside group)
[PAYMENT] "TAM UNG PHAN LOI NHUAN PHAN PHOI NAM 2025" → Dividend paid (inside group)
[PAYMENT] "Distribute dividends for FY2025" → Dividend paid (inside group)

**DIVIDEND RECEIPT (inside group)**:
[RECEIPT] "BHP_2nd installment of Profit Distribution for FY2024" → Dividend receipt (inside group)
[RECEIPT] "Profit distribution for FY2024 3rd payment" → Dividend receipt (inside group)
[RECEIPT] "Distribute dividends for FY2025_1st payment" → Dividend receipt (inside group)

**DEAL PAYMENT**:
Keywords: SPA, escrow account, payment for SPA, upfront LURC, brokerage fee
[PAYMENT] "2nd Release escrow account to payment for SPA" → Deal payment
[PAYMENT] "2nd Release money from escrow account to seller" → Deal payment
[PAYMENT] "4th payment after issuance of upfront LURC" → Deal payment
[PAYMENT] "Final 50% brokerage fee payment after completion of upfront LURC" → Deal payment

**LAND ACQUISITION**:
Keywords: land payment, land price, land lease, Land rental fee, Land Lease fee
[PAYMENT] "BLB_AMATA_Land lease fee in 2025" → Land acquisition
[PAYMENT] "Payment of land lease fee after having land handover" → Land acquisition
[PAYMENT] "BHI_KCN SM_Land rental fee in 2025" → Land acquisition
[PAYMENT] "BDG_KCN.DG_Land Lease fee in 2025" → Land acquisition

**PAYMENT FOR ACQUISITION**:
Keywords: acquisition, M&A
[PAYMENT] "Payment for acquisition of shares in ABC Company" → Payment for acquisition
[PAYMENT] "Transfer for M&A deal completion" → Payment for acquisition

**INTERNAL CONTRIBUTION IN/OUT**:
Keywords: capital contribution, gop von
[RECEIPT] "Capital contribution from holdco N4C to Nastec JSC" → Internal contribution in
[PAYMENT] "Capital contribution from holdco N4C to Nastec JSC" → Internal contribution out
[RECEIPT] "H5Q_Capital Contribution to LMX" → Internal contribution in
[PAYMENT] "H5Q_Capital Contribution to LMX" → Internal contribution out

**REFINANCING**:
Keywords: giai ngan bu dap, refinancing, HOAN TRA PHAN VON GOP, THANH TOAN BU DAP, hoan von
[RECEIPT] "GIAI NGAN HOAN VON DU AN BCI / REFINANCING BCI PROJECT" → Refinancing
[RECEIPT] "HOAN TRA PHAN VON GOP HOP TAC KINH DOANH" → Refinancing
[RECEIPT] "THANH TOAN BU DAP CHI PHI NHAN CHUYEN NHUONG" → Refinancing
[RECEIPT] "PNA REFINANCING LAND RENTAL PAYMENT" → Refinancing
[RECEIPT] "Giai ngan hoan von/Refinancing for BMP" → Refinancing

**LOAN DRAWDOWN**:
Keywords: DISBURSE, GIAI NGAN (without bu dap/hoan von), syndicated loan fund, Facility A, Facility B, Loan Execute
[RECEIPT] "Transfer from KBank Account - DISBURSE FOR PAY INTEREST" → Loan drawdown
[RECEIPT] "Kasikornbank HCMC transfers syndicated loan fund (Facility A)" → Loan drawdown
[RECEIPT] "GIAI NGAN CHO KHOAN TIN DUNG A" → Loan drawdown
[RECEIPT] "DISBURSEMENT FOR FACILITY B" → Loan drawdown
[RECEIPT] "Loan Execute" → Loan drawdown

**VAT REFUND**:
Keywords: VAT refund, hoan thue, HOAN THUE GTGT
[RECEIPT] "HOAN THUE GTGT QUY 3/2024" → VAT refund
[RECEIPT] "18294/QD-CCTKV16-KDT...hoan thue" → VAT refund

**OTHER RECEIPTS** (bank interest, misc):
Keywords: Thanh toan lai tai khoan tien gui, Tra lai TK tien gui, lai nhap von, ghi co lai tien gui, Deposit Interest Credit, hoan ta LCT, TFR, CA - TARGET, interest payment, tra lai tai khoan DDA
[RECEIPT] "Thanh toan lai tai khoan tien gui 819009112551" → Other receipts
[RECEIPT] "Tra lai TK tien gui 211000396185" → Other receipts
[RECEIPT] "lai nhap von" → Other receipts
[RECEIPT] "ghi co lai tien gui" → Other receipts
[RECEIPT] "Deposit Interest Credit" → Other receipts
[RECEIPT] "tra lai tai khoan DDA 110002855448" → Other receipts

**RECEIPT FROM TENANTS** (DEFAULT for business payments):
Keywords: TT tien, rental fee, rent fee, tien thue, thue xuong, thue kho, dien nuoc, tien dien, tien nuoc, phi tien ich, phi quan ly, management fee, REM B/O, THANH TOAN, TT (payment)
[RECEIPT] "REM 9901CI250606000041934 B/O COHERENT VIETNAM" → Receipt from tenants
[RECEIPT] "CONG TY TNHH...chuyen tien phi dich vu" → Receipt from tenants
[RECEIPT] "Cimiya TT tien nuoc sach theo HD so 0000060" → Receipt from tenants
[RECEIPT] "WARTON VIET NAM TT TIEN PHI QUAN LY" → Receipt from tenants
[RECEIPT] "thanh toan tien nuoc sach va xu ly nc thai" → Receipt from tenants
[RECEIPT] "ITL mien Bac thanh toan INV 65" → Receipt from tenants
[RECEIPT] "CT Gia Viet thanh toan hoa don so 71" → Receipt from tenants
DEFAULT RULE: Most RECEIPT transactions from company names (CTY, TNHH, CORP, LTD, CO., INC) → Receipt from tenants

=== PRIORITY ORDER ===
1. INTERNAL TRANSFER - check for HDTG, internal transfer, SHL, tien gui, transfer money first
2. DIVIDEND - check for profit distribution, dividend, co tuc
3. LOAN REPAYMENT - check for GOC, principal
4. LOAN INTEREST - check for Thu no, Tra no TK vay (without GOC)
5. CONSTRUCTION - check for IPC, construction contractors (GLC, VEV, ICIC, etc.)
6. LAND ACQUISITION - check for land lease, land rental
7. DEAL PAYMENT - check for SPA, escrow, upfront LURC
8. REFINANCING - check for giai ngan bu dap, refinancing, hoan von
9. VAT REFUND - check for VAT refund, hoan thue
10. OPERATING EXPENSE - bank fees, utilities, refund deposits
11. OTHER RECEIPTS - only for bank interest (lai tien gui)
12. RECEIPT FROM TENANTS - DEFAULT for business receipts

=== TRANSACTIONS TO CLASSIFY ===
{transactions_text}

=== OUTPUT ===
Return ONLY numbered list with category names. No explanations.
1. Category name
2. Category name
..."""

        try:
            response = self.client.generate_content(prompt)
            response_text = response.text
            ai_results = self._parse_classification_response(response_text, len(uncached_batch))

            # Validate and cache results
            for i, (idx, (desc, is_receipt)) in enumerate(zip(uncached_indices, uncached_batch)):
                category = ai_results[i] if i < len(ai_results) else ""
                # Validate category
                category = self._validate_category(category, is_receipt)
                results[idx] = category

                # Cache the result
                cache_key = self._get_cache_key(desc, is_receipt)
                self._cache[cache_key] = category

            return results

        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return results  # Return partial results (cached ones + empty for failed)

    def _validate_category(self, category: str, is_receipt: bool) -> str:
        """
        Validate and normalize category to ensure it's in the allowed list.
        Returns empty string if invalid (will trigger fallback).
        """
        if not category:
            return ""

        category = category.strip()

        # Normalize non-breaking spaces (found in reference files like "Dividend paid (inside\xa0group)")
        category = category.replace('\xa0', ' ')

        # Check exact match first
        if is_receipt:
            if category in ALL_RECEIPT_CATEGORIES:
                return category
        else:
            if category in ALL_PAYMENT_CATEGORIES:
                return category

        # Try case-insensitive match
        category_lower = category.lower()
        target_categories = ALL_RECEIPT_CATEGORIES if is_receipt else ALL_PAYMENT_CATEGORIES

        for valid_cat in target_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat

        # Try partial match (in case AI adds extra text)
        for valid_cat in target_categories:
            if valid_cat.lower() in category_lower or category_lower in valid_cat.lower():
                logger.debug(f"Partial match: '{category}' -> '{valid_cat}'")
                return valid_cat

        # If category is valid but wrong type, return empty for fallback
        if category in ALL_CATEGORIES:
            logger.warning(f"Category '{category}' valid but wrong type (is_receipt={is_receipt})")
            return ""

        logger.warning(f"Invalid category '{category}', will use fallback")
        return ""

    def _parse_classification_response(self, response: str, expected_count: int) -> List[str]:
        """Parse the AI response into category list."""
        results = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse "1. Category name" or "1: Category name"
            if '.' in line:
                parts = line.split('.', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    category = parts[1].strip()
                    results.append(category)
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    category = parts[1].strip()
                    results.append(category)

        # Pad with empty strings if needed
        while len(results) < expected_count:
            results.append("")

        return results[:expected_count]

    def classify_single(self, description: str, is_receipt: bool) -> str:
        """Classify a single transaction."""
        results = self.classify_batch([(description, is_receipt)])
        return results[0] if results else ""
