"""
Tenant Name Matcher - Fuzzy matching utility for cross-file correlation.

This module provides fuzzy matching capabilities to link tenants across
RevenueBreakdown and UnitForLeaseList files, handling variations in naming.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# Common suffixes/prefixes to normalize
COMPANY_SUFFIXES = [
    "LLC", "LTD", "CO., LTD", "CO.,LTD", "CO LTD", "COMPANY LIMITED",
    "JSC", "JOINT STOCK COMPANY", "INC", "CORP", "CORPORATION",
    "TNHH", "CỔ PHẦN", "CP", "CÔNG TY", "CTY"
]

LEGAL_PREFIXES = [
    "CÔNG TY", "CTY", "CÔNG TY CỔ PHẦN", "CÔNG TY TNHH",
    "IE/", "S00", "C00"
]


@dataclass
class TenantMatch:
    """Result of a tenant matching attempt."""
    revenue_entity: str
    unit_tenant: str
    confidence: float  # 0.0 to 1.0
    match_type: str  # "exact", "code", "fuzzy", "partial"
    matched_on: str  # What was matched (e.g., "entity_code", "name_normalized")


class TenantMatcher:
    """Fuzzy matcher for tenant names across revenue and unit data."""

    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize matcher.

        Args:
            min_confidence: Minimum confidence threshold for matches (0.0-1.0)
        """
        self.min_confidence = min_confidence
        self._cache: Dict[str, str] = {}  # Normalized name cache

    def normalize_name(self, name: str) -> str:
        """
        Normalize a company/tenant name for comparison.

        Args:
            name: Original name

        Returns:
            Normalized name (uppercase, no special chars, no suffixes)
        """
        if not name:
            return ""

        # Check cache
        if name in self._cache:
            return self._cache[name]

        normalized = name.upper()

        # Remove entity codes at start (e.g., "C00000164 CTTV" -> "CTTV")
        normalized = re.sub(r'^[SC]\d+\s*', '', normalized)

        # Remove common prefixes
        for prefix in sorted(LEGAL_PREFIXES, key=len, reverse=True):
            if normalized.startswith(prefix.upper()):
                normalized = normalized[len(prefix):].strip()

        # Remove common suffixes
        for suffix in sorted(COMPANY_SUFFIXES, key=len, reverse=True):
            if normalized.endswith(suffix.upper()):
                normalized = normalized[:-len(suffix)].strip()

        # Remove Vietnamese diacritics for simpler matching
        normalized = self._remove_diacritics(normalized)

        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Cache result
        self._cache[name] = normalized
        return normalized

    def _remove_diacritics(self, text: str) -> str:
        """Remove Vietnamese diacritics from text."""
        diacritic_map = {
            'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
            'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
            'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
            'Đ': 'D',
            'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
            'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
            'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
            'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
            'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
            'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
            'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
            'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
            'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        }
        result = text.upper()
        for viet, ascii_char in diacritic_map.items():
            result = result.replace(viet, ascii_char)
        return result

    def extract_entity_code(self, text: str) -> Optional[str]:
        """
        Extract entity code from text.

        Args:
            text: Text that may contain entity code

        Returns:
            Entity code if found (e.g., "S000056", "C00000164")
        """
        # Match patterns like S000056, C00000164
        match = re.search(r'([SC]\d{5,8})', text)
        if match:
            return match.group(1)
        return None

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity ratio between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        if not str1 or not str2:
            return 0.0

        # Normalize both strings
        norm1 = self.normalize_name(str1)
        norm2 = self.normalize_name(str2)

        if not norm1 or not norm2:
            return 0.0

        # Exact match after normalization
        if norm1 == norm2:
            return 1.0

        # Check if one contains the other (partial match)
        if norm1 in norm2 or norm2 in norm1:
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return shorter / longer * 0.9  # Slight penalty for partial match

        # Fuzzy match using SequenceMatcher
        return SequenceMatcher(None, norm1, norm2).ratio()

    def match_tenant(
        self,
        revenue_entity_name: str,
        revenue_entity_code: str,
        unit_tenants: List[Tuple[str, str]]  # List of (tenant_name, tenant_code)
    ) -> Optional[TenantMatch]:
        """
        Find best matching tenant from unit list for a revenue entity.

        Args:
            revenue_entity_name: Entity name from RevenueBreakdown
            revenue_entity_code: Entity code from RevenueBreakdown
            unit_tenants: List of (tenant_name, tenant_code) from UnitForLeaseList

        Returns:
            Best TenantMatch if confidence >= min_confidence, else None
        """
        if not unit_tenants:
            return None

        best_match: Optional[TenantMatch] = None
        best_confidence = 0.0

        # Extract entity code from revenue entity
        rev_code = self.extract_entity_code(revenue_entity_code) or self.extract_entity_code(revenue_entity_name)

        for tenant_name, tenant_code in unit_tenants:
            # Try code matching first (highest confidence)
            if rev_code and tenant_code:
                unit_code = self.extract_entity_code(tenant_code) or self.extract_entity_code(tenant_name)
                if rev_code == unit_code:
                    return TenantMatch(
                        revenue_entity=revenue_entity_name,
                        unit_tenant=tenant_name,
                        confidence=1.0,
                        match_type="code",
                        matched_on="entity_code"
                    )

            # Try name matching
            similarity = self.calculate_similarity(revenue_entity_name, tenant_name)

            if similarity > best_confidence:
                best_confidence = similarity
                match_type = "exact" if similarity == 1.0 else "fuzzy" if similarity >= 0.8 else "partial"
                best_match = TenantMatch(
                    revenue_entity=revenue_entity_name,
                    unit_tenant=tenant_name,
                    confidence=similarity,
                    match_type=match_type,
                    matched_on="name_normalized"
                )

        # Return best match if above threshold
        if best_match and best_match.confidence >= self.min_confidence:
            return best_match

        return None

    def match_all_tenants(
        self,
        revenue_entities: List[Tuple[str, str]],  # List of (entity_name, entity_code)
        unit_tenants: List[Tuple[str, str]]  # List of (tenant_name, tenant_code)
    ) -> Dict[str, TenantMatch]:
        """
        Match all revenue entities to unit tenants.

        Args:
            revenue_entities: List of (entity_name, entity_code) from RevenueBreakdown
            unit_tenants: List of (tenant_name, tenant_code) from UnitForLeaseList

        Returns:
            Dict mapping revenue_entity_name -> TenantMatch
        """
        matches: Dict[str, TenantMatch] = {}
        matched_unit_tenants = set()  # Track which unit tenants have been matched

        # Sort by potential exact matches first (by code)
        for entity_name, entity_code in revenue_entities:
            match = self.match_tenant(entity_name, entity_code, unit_tenants)

            if match:
                # Avoid duplicate matching if already matched
                if match.unit_tenant not in matched_unit_tenants or match.confidence > 0.95:
                    matches[entity_name] = match
                    if match.confidence > 0.8:
                        matched_unit_tenants.add(match.unit_tenant)

        logger.info(f"Matched {len(matches)}/{len(revenue_entities)} revenue entities to unit tenants")
        return matches


def create_tenant_matcher(min_confidence: float = 0.6) -> TenantMatcher:
    """
    Create a TenantMatcher instance.

    Args:
        min_confidence: Minimum confidence threshold (default 0.6)

    Returns:
        TenantMatcher instance
    """
    return TenantMatcher(min_confidence=min_confidence)
