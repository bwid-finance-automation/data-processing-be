"""
AI-based Transaction Classifier using Google Gemini API.
Classifies bank transactions into Nature categories.

Features:
- Dynamic keyword rules loaded from CSV file (movement_nature_filter/Key Words.csv)
- temperature=0 for deterministic output
- Comprehensive few-shot examples from verified data
- Output validation against allowed categories
- Direction-aware fallback defaults
- In-memory caching to avoid redundant API calls
"""
import csv
import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import google.generativeai as genai

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Nature categories ──
# Primary categories derived from verified correct classifications.
# Direction (RECEIPT/PAYMENT) is the primary discriminator.

PAYMENT_CATEGORIES = [
    "Operating expense",          # DEFAULT for payments - bank fees, taxes, salaries, services, utilities paid
    "Internal transfer out",      # Intercompany transfers, BCC, SHL, HDTG deposits, savings
    "Loan repayment",            # BANK loan principal repayments only (Tra no TK vay, THU NO GOC)
    "Loan interest",             # BANK loan interest (Thu no LAI, LD-IN)
    "Construction expense",      # New construction projects, CIP contractors (IPC, ICIC, VEV, GLC)
    "Deal payment",              # Share Purchase Agreement (SPA) payments
    "Land acquisition",          # Land payments, land lease
    "Dividend paid (inside group)",  # Dividend/profit distribution payments
    "Payment for acquisition",   # Acquisition payments
]

RECEIPT_CATEGORIES = [
    "Receipt from tenants",      # DEFAULT for receipts - rental income, utility charges, tenant payments
    "Other receipts",            # Bank interest ONLY (lai tien gui, DDA Interest, CRIN, Deposit Interest Credit)
    "Internal transfer in",      # Intercompany transfers received, BCC, SHL, HDTG withdrawals, savings settlement
    "Refinancing",               # Refinancing disbursements (giai ngan bu dap)
    "Loan receipts",             # Loan disbursements (giai ngan khoan vay)
    "VAT refund",                # VAT refund (hoan thue)
    "Dividend receipt (inside group)",  # Dividend received
    "Corporate Loan drawdown",   # Loan drawdown receipts
    "Loan drawdown",             # Loan drawdown alternative name
]

# All valid categories for validation
ALL_PAYMENT_CATEGORIES = set(PAYMENT_CATEGORIES)
ALL_RECEIPT_CATEGORIES = set(RECEIPT_CATEGORIES)
ALL_CATEGORIES = ALL_PAYMENT_CATEGORIES | ALL_RECEIPT_CATEGORIES

# Default rules file path (relative to project root)
# ai_classifier.py is at: app/domain/finance/cash_report/services/ai_classifier.py
# parents: [0]=services, [1]=cash_report, [2]=finance, [3]=domain, [4]=app, [5]=project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RULES_FILE = _PROJECT_ROOT / "movement_nature_filter" / "Key Words.csv"


def load_keyword_rules_from_csv(file_path: Path) -> Dict[str, List[str]]:
    """
    Load keyword → category mapping from CSV file.

    CSV format (semicolon-separated):
        Category Cash consol;Key word;SUB_Category;...

    Returns:
        Dict of category → list of keywords
    """
    rules: Dict[str, List[str]] = defaultdict(list)

    if not file_path.exists():
        logger.warning(f"Rules CSV not found: {file_path}")
        return dict(rules)

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)  # skip header

            for row in reader:
                if len(row) < 2:
                    continue
                category = row[0].strip()
                keyword = row[1].strip()
                if category and keyword and keyword.lower() != 'nan':
                    rules[category].append(keyword)

        total_keywords = sum(len(kws) for kws in rules.values())
        logger.info(f"Loaded {total_keywords} keywords across {len(rules)} categories from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load rules from CSV: {e}")

    return dict(rules)


def build_keyword_rules_prompt(rules: Dict[str, List[str]]) -> str:
    """
    Build the keyword rules section of the prompt from loaded rules.

    Args:
        rules: Dict of category → list of keywords

    Returns:
        Formatted string for the prompt
    """
    if not rules:
        return "No keyword rules loaded.\n"

    lines = []
    for category, keywords in rules.items():
        kw_str = ", ".join(keywords)
        lines.append(f"**{category}**: {kw_str}")

    return "\n".join(lines)


class AITransactionClassifier:
    """
    AI-based classifier for bank transactions.
    Uses Google Gemini API to understand transaction descriptions and classify them.

    Features:
    - Dynamic keyword rules loaded from CSV (movement_nature_filter/Key Words.csv)
    - temperature=0 for deterministic, reproducible results
    - Comprehensive override rules and few-shot examples
    - Validation against allowed categories with direction-aware fallback
    - In-memory caching to avoid redundant API calls
    """

    def __init__(self, api_key: Optional[str] = None, rules_file: Optional[Path] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self._cache: Dict[str, str] = {}  # Cache: hash(description+is_receipt) -> category
        self._total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._cache_stats = {"cache_hit": 0, "api_call": 0}  # Track cache hits vs API calls

        # Load keyword rules from CSV
        rules_path = rules_file or DEFAULT_RULES_FILE
        self._keyword_rules = load_keyword_rules_from_csv(rules_path)
        self._keyword_rules_prompt = build_keyword_rules_prompt(self._keyword_rules)
        self._rules_file = rules_path

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.generation_config = genai.GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                response_mime_type="application/json",
            )
            self.client = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config=self.generation_config
            )
        else:
            logger.warning("No GEMINI_API_KEY found, AI classifier will not work")

    def reload_rules(self, rules_file: Optional[Path] = None) -> int:
        """
        Reload keyword rules from CSV. Call this after updating the CSV file.

        Returns:
            Total number of keywords loaded
        """
        path = rules_file or self._rules_file
        self._keyword_rules = load_keyword_rules_from_csv(path)
        self._keyword_rules_prompt = build_keyword_rules_prompt(self._keyword_rules)
        total = sum(len(kws) for kws in self._keyword_rules.values())
        logger.info(f"Reloaded {total} keywords from {path}")
        return total

    def get_and_reset_usage(self) -> Dict:
        """Return accumulated token usage and reset counters."""
        usage = dict(self._total_usage)
        usage["model"] = "gemini-2.0-flash"
        usage["provider"] = "gemini"
        usage["cache_hit"] = self._cache_stats["cache_hit"]
        usage["api_call"] = self._cache_stats["api_call"]
        self._total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._cache_stats = {"cache_hit": 0, "api_call": 0}
        return usage

    def classify_batch(
        self,
        transactions: List[Tuple[str, bool]],  # List of (description, is_receipt)
        batch_size: int = 50
    ) -> List[Tuple[str, bool]]:
        """
        Classify a batch of transactions using AI.

        Args:
            transactions: List of (description, is_receipt) tuples
            batch_size: Number of transactions per API call

        Returns:
            List of (category, was_cached) tuples
        """
        if not self.client:
            logger.error("AI classifier not initialized - missing API key")
            return [("", False) for _ in transactions]

        results = []

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

    def _build_prompt(self, transactions_text: str, expected_count: int) -> str:
        """Build the full classification prompt with dynamic keyword rules."""
        return f"""You are a financial transaction classifier for Vietnamese industrial park companies (BW Industrial group).
Classify each transaction into exactly ONE Nature category.

CRITICAL RULE #1: The [RECEIPT] or [PAYMENT] tag is ALREADY DETERMINED by cash flow direction. You MUST respect it:
- [RECEIPT] = cash IN = debit = money received → ONLY use RECEIPT categories
- [PAYMENT] = cash OUT = credit = money paid out → ONLY use PAYMENT categories
NEVER assign a RECEIPT category to a [PAYMENT] transaction or vice versa.

=== PAYMENT (cash out) categories ===
- Operating expense (DEFAULT for unmatched payments)
- Internal transfer out
- Loan repayment
- Loan interest
- Construction expense
- Deal payment
- Land acquisition
- Dividend paid (inside group)
- Payment for acquisition

=== RECEIPT (cash in) categories ===
- Receipt from tenants (DEFAULT for unmatched receipts)
- Other receipts
- Internal transfer in
- Refinancing
- Loan receipts
- VAT refund
- Dividend receipt (inside group)
- Corporate Loan drawdown
- Loan drawdown

=== KEYWORD-BASED CLASSIFICATION RULES ===
{self._keyword_rules_prompt}

NOTE: "Internal transfer" in keyword rules means:
- If [PAYMENT] → "Internal transfer out"
- If [RECEIPT] → "Internal transfer in"

=== CRITICAL CLASSIFICATION RULES ===

**RULE 1 — INTERNAL TRANSFER (highest priority after loan repayment)**
Any transaction between entities within the BW Industrial group (subsidiaries, SPVs, holdcos) = Internal transfer.
Detection patterns:
- Entity code pairs in description: "XXX_YYY_" format (e.g., "VC3_BB3_", "BCI_GTC_", "NT2_VC3_")
  Common entity codes: BB3, BB4, BB5, BB6, BBA, BBN, BCI, BDG, BDH, BHP, BLB, BNA, BNT, BSG, BTD, BTH, BTP, BTU, BWD, BWP, BYP, DC2, DVC, GTC, H5H, H5S, HD2, HD3, IRE, JHP, LCC, MHN, NSHL, NT2, PAE, PDE, PDHD, PNA, QV2, S5B, S5D, SC2, ST2, STL, TCS, TPT, VC3, WL2A, WL2B, XA1
- BCC keywords: "BCC transfer", "chuyen tien hop tac kinh doanh", "BCC"
- SHL keywords: "SHL", "shareholder loan", "SHL repayment", "SHL interest", "vay noi bo"
- Savings/Deposit operations: "HDTG", "GUI HDTG", "MO HDTG", "TAT TOAN", "CK SANG TK TIME", "TIMEMO", "TKKH", "tien gui co ky han", "HOP DONG TIEN GUI", "Completed transfer to BIDV CD"
- Internal transfer keywords: "INTERNAL TRANSFER", "CHUYEN KHOAN", "Rut tien gui", "ROLLOVER", "VCBCSH"
- "Tra goc TK tien gui" (return of deposit principal) → Internal transfer in
- "MISC CREDIT" → Internal transfer in; "MISC DEBIT" → Internal transfer out
- Management fees BETWEEN group entities (e.g., "BDG_VC3_MANAGEMENT FEE", "BTU-Payment management fee to VC3") = Internal transfer, NOT Operating expense or Receipt from tenants
- "CA - TARGET" (SINOPAC settlement) → Internal transfer in
- "CLOSING TDA" (close term deposit) → Internal transfer in
- Capital contribution / gop von between entities → Internal transfer

**RULE 2 — LOAN REPAYMENT (BANK loans only)**
ONLY for repayment of principal to BANKS. NOT intercompany.
Keywords: "Tra no TK vay" (+ account number), "THU NO GOC VAY", "THU NO TAI KHOAN VAY", "THU NO TKV", "LD-PR", "REPAY LOAN"
IMPORTANT: "SHL repayment" or "Repayment shareholder loan" = Internal transfer out (NOT Loan repayment)

**RULE 3 — LOAN INTEREST (BANK loan interest only)**
Keywords: "Thu no LAI", "Thu no lai (LD-IN)", "lai phat sinh"
NOT "thanh toan lai" which is bank deposit interest = Other receipts
NOT SHL interest = Internal transfer

**RULE 4 — OTHER RECEIPTS (bank interest/deposit returns ONLY)**
This is STRICTLY for bank-generated interest credits. NOT for business receipts.
Keywords: "lai tien gui", "DDA Interest", "CRIN", "Deposit Interest Credit", "ghi co lai", "lai nhap von", "tra lai tai khoan DDA", "Tra lai TK tien gui", "thanh toan lai thang", "Thanh toan lai tai khoan tien gui", "TFR", "hoan ta LCT"
If unsure whether a RECEIPT is "Other receipts" or "Receipt from tenants", choose "Receipt from tenants" (it's the default).

**RULE 5 — RECEIPT FROM TENANTS (default for all business receipts)**
Any receipt from a company/tenant paying rent, utilities, services, booking fees, or general business payments.
Keywords: "thue xuong", "thue kho", "rental", "rent", "tien thue", "tien dien", "tien nuoc", "dien nuoc", "phi tien ich", "RENTING WAREHOUSE"
DEFAULT: If a RECEIPT transaction doesn't match any other category, classify as "Receipt from tenants".

**RULE 6 — OPERATING EXPENSE (default for all business payments)**
Bank fees (SC60, VATX, EMF, Recovery Charge), taxes (NOP NSNN, NOPTHUE), salaries, service fees paid to vendors, refund of booking fees/deposits, utilities paid by the company, professional services.
DEFAULT: If a PAYMENT transaction doesn't match any other category, classify as "Operating expense".

**RULE 7 — CONSTRUCTION EXPENSE**
ONLY for new construction projects (CIP). Specific contractors: IPC, ICIC, VEV, GLC, NEWTECONS, VITECCONS, MEGASPACE, ARTELIA, Earth Works.
Repair/maintenance work (even by construction companies) = Operating expense.

**RULE 8 — DEAL PAYMENT**
Share Purchase Agreement payments: "SPA", "Final payment SPA"

**RULE 9 — REFINANCING**
"giai ngan bu dap", "refinancing", "KLHT"

=== FEW-SHOT EXAMPLES (verified correct) ===

[PAYMENT] "NT2_VC3_MANAGEMENT FEE/ TT CHI PHI DICH VU" → Internal transfer out (entity-to-entity management fee)
[RECEIPT] "NT2_VC3_MANAGEMENT FEE/ TT CHI PHI DICH VU" → Internal transfer in (same description, opposite direction)
[PAYMENT] "VC3_BB3_BCC transfer_Chuyen tien hop tac kinh doanh" → Internal transfer out
[RECEIPT] "VC3_BB3_BCC transfer_Chuyen tien hop tac kinh doanh" → Internal transfer in
[PAYMENT] "CK SANG TK TIMEMO TKKH 1 THANG THEO HD SO..." → Internal transfer out (savings deposit)
[RECEIPT] "TAT TOAN HDTG SO 285.2025.38679" → Internal transfer in (savings withdrawal)
[PAYMENT] "GUI HDTG SO 285.2026.46820, KY HAN: 1M" → Internal transfer out (savings deposit)
[RECEIPT] "Tra goc TK tien gui 217000486074" → Internal transfer in (deposit principal return)
[RECEIPT] "Tra lai TK tien gui 212000488544" → Internal transfer in (deposit interest - same entity's own deposit)
[RECEIPT] "CA - TARGET" → Internal transfer in (SINOPAC settlement)
[RECEIPT] "CLOSING TDA - CLOSING TDA" → Internal transfer in
[RECEIPT] "MISC CREDIT | MIR601281251C01 | BTPVC3Tra lai BCC" → Internal transfer in
[PAYMENT] "MISC DEBIT | ..." → Internal transfer out
[PAYMENT] "VCBCSH. C719220126052748.BLB VC3 thanh toan chi phi dich vu management fee" → Internal transfer out
[PAYMENT] "SHAREHOLDER LOAN FROM VC3 TO SPV4B" → Internal transfer out (intercompany, NOT loan repayment)
[RECEIPT] "SHAREHOLDER LOAN FROM VC3 TO SPV4B" → Internal transfer in
[PAYMENT] "BCI_GTC_Payment for SHL interest/Thanh toan lai vay noi bo" → Internal transfer out (SHL = intercompany)
[RECEIPT] "BCI_GTC_Payment for SHL interest/Thanh toan lai vay noi bo" → Internal transfer in
[PAYMENT] "So GD: 285A2611CPQB0FM7 VC3_SHL to BLA - VC3 cho BLA vay" → Internal transfer out
[PAYMENT] "Tra no TK vay 800004949853,so tien 4546359927 VND" → Loan repayment (bank loan)
[PAYMENT] "THU NO GOC VAY DEN HAN" → Loan repayment
[PAYMENT] "Thu no goc (LD-PR) - So khe uoc MMD20253243541/01" → Loan repayment
[PAYMENT] "Thu no LAI U2628493" → Loan interest
[PAYMENT] "Thu no lai (LD-IN) - So khe uoc MMD20253243541/01" → Loan interest
[RECEIPT] "Tra lai TK tien gui 217000486074" → Internal transfer in (NOT Other receipts - this is own deposit interest)
[RECEIPT] "DDA Interest Paid..." → Other receipts (bank interest)
[RECEIPT] "CRIN" → Other receipts (bank interest credit)
[RECEIPT] "ghi co lai tien gui" → Other receipts
[RECEIPT] "Deposit Interest Credit" → Other receipts
[PAYMENT] "262010206A-0018348/285N261193P8Z7BV/NOP NSNN" → Operating expense (tax payment)
[PAYMENT] "SC60..." → Operating expense (bank fee)
[PAYMENT] "REFUND BOOKING FEE..." → Operating expense
[PAYMENT] "BB5 - ICIC - CS Contract - PR.03" → Construction expense
[PAYMENT] "PDE - VITECCONS INTERIM PAYMENT 04 INV 21" → Construction expense
[PAYMENT] "So GD: 285A2611ECKSJ96G S5B_LTH_Final payment SPA of Tan Hiep" → Deal payment
[RECEIPT] "So GD goc: 10000112 ORION THANH TOAN CONG NO..." → Loan receipts
[RECEIPT] "REM 9901CI250606... B/O COHERENT VIETNAM" → Receipt from tenants
[RECEIPT] "CTY TNHH ... TT TIEN THUE XUONG" → Receipt from tenants
[PAYMENT] "REM 9901SP260126... H5S_TCS_Charter capital contribution to TCS/Gop von dieu le" → Internal transfer out

=== TRANSACTIONS TO CLASSIFY ===
{transactions_text}

=== OUTPUT FORMAT (JSON) ===
Return a JSON object with a single key "classifications" containing an array of category strings.
The array MUST have exactly {expected_count} elements, one for each transaction above, in the same order.
REMEMBER: [PAYMENT] → use only PAYMENT categories. [RECEIPT] → use only RECEIPT categories.
When unsure: [PAYMENT] → "Operating expense". [RECEIPT] → "Receipt from tenants".

Example for 3 transactions:
{{"classifications": ["Operating expense", "Receipt from tenants", "Internal transfer out"]}}
"""

    def _classify_batch_internal(self, batch: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
        """Classify a single batch of transactions with caching and validation.

        Returns:
            List of (category, was_cached) tuples
        """
        results: List[Tuple[str, bool]] = [("", False)] * len(batch)
        uncached_indices = []
        uncached_batch = []

        # Check cache first
        cached_count = 0
        for idx, (desc, is_receipt) in enumerate(batch):
            cache_key = self._get_cache_key(desc, is_receipt)
            if cache_key in self._cache:
                results[idx] = (self._cache[cache_key], True)
                cached_count += 1
            else:
                uncached_indices.append(idx)
                uncached_batch.append((desc, is_receipt))

        self._cache_stats["cache_hit"] += cached_count
        self._cache_stats["api_call"] += len(uncached_batch)

        if not uncached_batch:
            logger.info(f"All {len(batch)} transactions found in cache")
            return results

        logger.info(f"Cache hit: {cached_count}, need API call: {len(uncached_batch)}")

        # Build transactions text
        transactions_text = ""
        for idx, (desc, is_receipt) in enumerate(uncached_batch):
            tx_type = "RECEIPT (cash in)" if is_receipt else "PAYMENT (cash out)"
            transactions_text += f"{idx + 1}. [{tx_type}] {desc}\n"

        prompt = self._build_prompt(transactions_text, len(uncached_batch))

        try:
            response = self.client.generate_content(prompt)

            # Track token usage from Gemini response
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                inp = getattr(um, 'prompt_token_count', 0) or 0
                out = getattr(um, 'candidates_token_count', 0) or 0
                self._total_usage["input_tokens"] += inp
                self._total_usage["output_tokens"] += out
                self._total_usage["total_tokens"] = self._total_usage["input_tokens"] + self._total_usage["output_tokens"]
                logger.debug(f"Gemini token usage batch: in={inp}, out={out}, cumulative={self._total_usage}")
            else:
                logger.warning(f"Gemini response missing usage_metadata: hasattr={hasattr(response, 'usage_metadata')}, value={getattr(response, 'usage_metadata', 'N/A')}")

            response_text = response.text
            ai_results = self._parse_json_response(response_text, len(uncached_batch))

            # Validate and cache results
            for i, (idx, (desc, is_receipt)) in enumerate(zip(uncached_indices, uncached_batch)):
                category = ai_results[i] if i < len(ai_results) else ""
                category = self._validate_category(category, is_receipt)
                results[idx] = (category, False)

                cache_key = self._get_cache_key(desc, is_receipt)
                self._cache[cache_key] = category

            return results

        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return results

    def _validate_category(self, category: str, is_receipt: bool) -> str:
        """
        Validate and normalize category to ensure it's in the allowed list.
        Returns direction-appropriate default if invalid.
        """
        if not category:
            # Default fallback based on direction
            return "Receipt from tenants" if is_receipt else "Operating expense"

        category = category.strip()
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

        # Handle "Internal transfer" without direction → add direction
        if "internal transfer" in category_lower:
            return "Internal transfer in" if is_receipt else "Internal transfer out"

        # Handle "Internal contribution" without direction → map to Internal transfer
        if "internal contribution" in category_lower or "contribution" in category_lower:
            return "Internal transfer in" if is_receipt else "Internal transfer out"

        # Handle "Saving" category → map to Internal transfer
        if category_lower in ("saving", "savings"):
            return "Internal transfer in" if is_receipt else "Internal transfer out"

        # Try partial match (in case AI adds extra text)
        for valid_cat in target_categories:
            if valid_cat.lower() in category_lower or category_lower in valid_cat.lower():
                logger.debug(f"Partial match: '{category}' -> '{valid_cat}'")
                return valid_cat

        # If category is valid but wrong direction, log and use default
        if category in ALL_CATEGORIES:
            logger.warning(f"Category '{category}' valid but wrong direction (is_receipt={is_receipt}), using default")
            return "Receipt from tenants" if is_receipt else "Operating expense"

        logger.warning(f"Invalid category '{category}', using direction-based default")
        return "Receipt from tenants" if is_receipt else "Operating expense"

    def _parse_json_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse the JSON AI response into category list."""
        try:
            data = json.loads(response_text)
            classifications = data.get("classifications", [])

            if not isinstance(classifications, list):
                logger.warning(f"JSON 'classifications' is not a list: {type(classifications)}")
                return [""] * expected_count

            # Pad or trim to expected count
            results = [str(c).strip() for c in classifications]
            while len(results) < expected_count:
                results.append("")
            return results[:expected_count]

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed, falling back to text parser: {e}")
            return self._parse_text_response_fallback(response_text, expected_count)

    def _parse_text_response_fallback(self, response: str, expected_count: int) -> List[str]:
        """Fallback: parse numbered text list if JSON mode fails."""
        results = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '.' in line:
                parts = line.split('.', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    results.append(parts[1].strip())
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    results.append(parts[1].strip())

        while len(results) < expected_count:
            results.append("")

        return results[:expected_count]

    def preload_cache(self, entries: Dict[str, str]) -> None:
        """Bulk-load cache entries (e.g. from Redis) into in-memory cache."""
        self._cache.update(entries)
        if entries:
            logger.info(f"Pre-loaded {len(entries)} entries into in-memory cache")

    def get_new_cache_entries(self, before_keys: set) -> Dict[str, str]:
        """Get cache entries that were added after a known snapshot."""
        return {k: v for k, v in self._cache.items() if k not in before_keys}

    def get_cache_keys_for_batch(self, transactions: List[Tuple[str, bool]]) -> Dict[str, Tuple[str, bool]]:
        """Generate cache keys for a batch of transactions."""
        result = {}
        for desc, is_receipt in transactions:
            cache_key = self._get_cache_key(desc, is_receipt)
            result[cache_key] = (desc, is_receipt)
        return result

    def classify_single(self, description: str, is_receipt: bool) -> str:
        """Classify a single transaction."""
        results = self.classify_batch([(description, is_receipt)])
        return results[0][0] if results else ""
