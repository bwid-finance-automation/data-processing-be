"""
TF-IDF similarity-based Nature classifier for cash report transactions.

Uses the labeled examples in Transactions.csv as a reference corpus.
For each new transaction, finds the most similar reference description
via cosine similarity of TF-IDF vectors (character n-grams).

Direction-safe: maintains separate indexes for receipt vs payment
to prevent cross-direction leakage.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy-import sklearn to keep module-level import lightweight.
# sklearn is imported only when fit/predict are actually called.
_sklearn_available: Optional[bool] = None


def _ensure_sklearn():
    """Import sklearn on first use; raise clear error if missing."""
    global _sklearn_available
    if _sklearn_available is True:
        return
    try:
        import sklearn  # noqa: F401
        _sklearn_available = True
    except ImportError:
        _sklearn_available = False
        raise ImportError(
            "scikit-learn is required for TF-IDF classification. "
            "Install it with: pip install scikit-learn"
        )


# ── Text normalisation (mirrors CashReportService helpers) ──

_RE_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_DIGITS = re.compile(r"\d+")


def _normalize_for_tfidf(text: str) -> str:
    """
    Normalise bank description for TF-IDF vectorisation.

    Strips punctuation, collapses whitespace, lowercases.
    Keeps digits because they carry meaning (account numbers, amounts).
    """
    normalized = (text or "").lower().replace("_", " ").strip()
    normalized = _RE_NON_WORD.sub(" ", normalized)
    return _RE_MULTI_SPACE.sub(" ", normalized).strip()


def _normalize_template(text: str) -> str:
    """
    Number-stripped normalisation for template-level similarity.

    Same as _normalize_for_tfidf but additionally removes ALL digits
    so that recurring descriptions with different dates/amounts collapse.
    """
    normalized = _normalize_for_tfidf(text)
    normalized = _RE_DIGITS.sub("", normalized)
    return _RE_MULTI_SPACE.sub(" ", normalized).strip()


class TfidfNatureClassifier:
    """
    Classify transaction Nature using TF-IDF cosine similarity against a
    corpus of labeled reference descriptions.

    Two vectorisers are maintained—one for receipts and one for payments—
    so that direction context is preserved.

    Workflow
    --------
    1. ``load_from_csv(path)``  — parse Transactions.csv into (desc, nature, is_receipt)
    2. ``fit()``                — build TF-IDF matrices (called automatically by load_from_csv)
    3. ``predict(desc, is_receipt)`` → Optional[str]  — classify a single description
    """

    # ── Nature validation (same sets used in ai_classifier.py) ──
    _RECEIPT_NATURES = frozenset({
        "Receipt from tenants", "Other receipts", "Internal transfer in",
        "Refinancing", "Loan receipts", "VAT refund",
        "Dividend receipt (inside group)", "Corporate Loan drawdown",
        "Loan drawdown",
    })
    _PAYMENT_NATURES = frozenset({
        "Operating expense", "Internal transfer out", "Loan repayment",
        "Loan interest", "Construction expense", "Deal payment",
        "Land acquisition", "Dividend paid (inside group)",
        "Payment for acquisition",
    })

    # Aliases to normalise legacy/variant names in reference CSV.
    _ALIASES: Dict[str, str] = {
        "Operating Expense": "Operating expense",
        "Dividend paid (inside group)": "Dividend paid (inside group)",
        "Internal transfer": "Internal transfer in",  # receipt default
    }

    def __init__(self, min_similarity: float = 0.55):
        """
        Args:
            min_similarity: Minimum cosine similarity to accept a match.
                            Higher → more precise but fewer matches.
        """
        self.min_similarity = min_similarity

        # Populated by fit()
        self._receipt_vectorizer = None
        self._payment_vectorizer = None
        self._receipt_matrix = None
        self._payment_matrix = None
        self._receipt_natures: List[str] = []
        self._payment_natures: List[str] = []
        self._is_fitted = False

        # Stats
        self._corpus_size = 0

    # ── Public API ──────────────────────────────────────────────

    def load_from_csv(self, csv_path: Path) -> int:
        """
        Load labeled examples from Transactions.csv and fit vectorisers.

        CSV format (semicolon-delimited):
            Bank description; Nature

        Returns:
            Number of usable rows loaded.
        """
        if not csv_path.exists():
            logger.warning("TF-IDF reference file not found: %s", csv_path)
            return 0

        descriptions: List[str] = []
        natures: List[str] = []
        is_receipts: List[bool] = []
        skipped = 0

        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader, None)  # skip header

                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    desc_raw = row[0].strip()
                    # File format: desc;debit;credit;nature (nature is LAST column)
                    nature_raw = row[-1].strip()

                    nature = self._canonical_nature(nature_raw)
                    if not nature or not desc_raw:
                        skipped += 1
                        continue

                    is_receipt = nature in self._RECEIPT_NATURES
                    descriptions.append(desc_raw)
                    natures.append(nature)
                    is_receipts.append(is_receipt)

        except Exception as e:
            logger.error("Failed to load TF-IDF reference CSV: %s", e)
            return 0

        if not descriptions:
            logger.warning("No usable rows in TF-IDF reference CSV")
            return 0

        count = self.fit(descriptions, natures, is_receipts)
        logger.info(
            "TF-IDF classifier loaded: %d rows (%d skipped), "
            "receipt corpus=%d, payment corpus=%d",
            count, skipped,
            len(self._receipt_natures),
            len(self._payment_natures),
        )
        return count

    def fit(
        self,
        descriptions: List[str],
        natures: List[str],
        is_receipts: List[bool],
    ) -> int:
        """
        Build TF-IDF matrices from labeled data.

        Args:
            descriptions: Raw bank descriptions.
            natures:      Canonical Nature label for each description.
            is_receipts:  Direction flag for each description.

        Returns:
            Total number of examples fitted.
        """
        _ensure_sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer

        receipt_descs: List[str] = []
        receipt_nats: List[str] = []
        payment_descs: List[str] = []
        payment_nats: List[str] = []

        for desc, nature, is_receipt in zip(descriptions, natures, is_receipts):
            norm = _normalize_template(desc)
            if not norm:
                continue
            if is_receipt:
                receipt_descs.append(norm)
                receipt_nats.append(nature)
            else:
                payment_descs.append(norm)
                payment_nats.append(nature)

        # Deduplicate: keep first occurrence of each (norm_desc, nature) pair.
        receipt_descs, receipt_nats = self._deduplicate(receipt_descs, receipt_nats)
        payment_descs, payment_nats = self._deduplicate(payment_descs, payment_nats)

        # Build vectorisers with char n-grams for robust partial matching.
        vectorizer_params = dict(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30_000,
            sublinear_tf=True,
        )

        if receipt_descs:
            self._receipt_vectorizer = TfidfVectorizer(**vectorizer_params)
            self._receipt_matrix = self._receipt_vectorizer.fit_transform(receipt_descs)
            self._receipt_natures = receipt_nats
        else:
            self._receipt_vectorizer = None
            self._receipt_matrix = None
            self._receipt_natures = []

        if payment_descs:
            self._payment_vectorizer = TfidfVectorizer(**vectorizer_params)
            self._payment_matrix = self._payment_vectorizer.fit_transform(payment_descs)
            self._payment_natures = payment_nats
        else:
            self._payment_vectorizer = None
            self._payment_matrix = None
            self._payment_natures = []

        self._corpus_size = len(receipt_descs) + len(payment_descs)
        self._is_fitted = self._corpus_size > 0
        return self._corpus_size

    def predict(self, description: str, is_receipt: bool) -> Optional[str]:
        """
        Classify a single transaction description.

        Args:
            description: Raw bank description.
            is_receipt:  True if cash-in (debit), False if cash-out (credit).

        Returns:
            Nature category string if similarity >= threshold, else None.
        """
        if not self._is_fitted:
            return None

        vectorizer = self._receipt_vectorizer if is_receipt else self._payment_vectorizer
        matrix = self._receipt_matrix if is_receipt else self._payment_matrix
        natures = self._receipt_natures if is_receipt else self._payment_natures

        if vectorizer is None or matrix is None or not natures:
            return None

        _ensure_sklearn()
        from sklearn.metrics.pairwise import cosine_similarity

        norm = _normalize_template(description)
        if not norm:
            return None

        try:
            query_vec = vectorizer.transform([norm])
            similarities = cosine_similarity(query_vec, matrix).flatten()
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]

            if best_score >= self.min_similarity:
                return natures[best_idx]
        except Exception as e:
            logger.debug("TF-IDF predict error: %s", e)

        return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    # ── Internals ────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(
        descriptions: List[str], natures: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Remove exact duplicate (description, nature) pairs, keeping first."""
        seen = set()
        deduped_descs: List[str] = []
        deduped_nats: List[str] = []
        for desc, nat in zip(descriptions, natures):
            key = (desc, nat)
            if key not in seen:
                seen.add(key)
                deduped_descs.append(desc)
                deduped_nats.append(nat)
        return deduped_descs, deduped_nats

    @classmethod
    def _canonical_nature(cls, raw: str) -> Optional[str]:
        """Normalise Nature value to canonical category."""
        nature = (raw or "").replace("\u00A0", " ").strip()
        if not nature:
            return None
        nature = cls._ALIASES.get(nature, nature)
        all_valid = cls._RECEIPT_NATURES | cls._PAYMENT_NATURES
        if nature in all_valid:
            return nature
        # Case-insensitive fallback
        nature_lower = nature.lower()
        for valid in all_valid:
            if valid.lower() == nature_lower:
                return valid
        return None
