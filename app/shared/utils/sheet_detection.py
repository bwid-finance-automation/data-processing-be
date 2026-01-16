# app/finance_accouting/utils/sheet_detection.py
"""Fuzzy sheet name detection utilities - shared by Python and AI modes."""

from typing import Optional, List
from difflib import SequenceMatcher
import io
import pandas as pd


def find_sheet_fuzzy(sheet_names: List[str], is_balance_sheet: bool = True) -> Optional[str]:
    """
    Find sheet name using fuzzy matching with 3-tier algorithm (same as AI mode).

    Handles variations like:
    - "BS Breakdown", "BS breakdown", "BSbreakdown", "bs breakdown"
    - "PL Breakdown", "PL breakdown", "PLbreakdown", "pl breakdown"
    - "Balance Sheet", "BẢNG CÂN ĐỐI KẾ TOÁN"
    - "Profit Loss", "Income Statement", "BÁO CÁO KẾT QUẢ KINH DOANH"

    Args:
        sheet_names: List of available sheet names in the Excel file
        is_balance_sheet: True to find BS sheet, False to find PL sheet

    Returns:
        Best matching sheet name, or None if no good match found
    """
    if is_balance_sheet:
        # BS sheet patterns (priority order)
        patterns = [
            "bs breakdown", "bs_breakdown", "bsbreakdown",
            "balance sheet breakdown", "balance_sheet",
            "balance sheet", "bs", "bảng cân đối kế toán",
            "sofp", "statement of financial position"
        ]
    else:
        # PL sheet patterns (priority order)
        patterns = [
            "pl breakdown", "pl_breakdown", "plbreakdown",
            "profit loss breakdown", "profit_loss",
            "profit loss", "income statement", "p&l", "p/l", "pl",
            "báo cáo kết quả kinh doanh"
        ]

    # Normalize sheet names for comparison
    sheet_names_lower = {name.lower().replace(' ', '').replace('_', ''): name for name in sheet_names}

    # TIER 1: Try exact matches first (ignoring case, spaces, underscores)
    for pattern in patterns:
        pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
        for normalized, original in sheet_names_lower.items():
            if pattern_normalized == normalized:
                return original

    # TIER 2: Try contains match
    for pattern in patterns:
        pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
        for normalized, original in sheet_names_lower.items():
            if pattern_normalized in normalized or normalized in pattern_normalized:
                return original

    # TIER 3: Try fuzzy matching with threshold
    best_match = None
    best_score = 0.0
    threshold = 0.6  # 60% similarity required (same as AI mode)

    for pattern in patterns:
        pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
        for normalized, original in sheet_names_lower.items():
            score = SequenceMatcher(None, pattern_normalized, normalized).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = original

    return best_match


def detect_sheets_from_bytes(xl_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    """
    Detect BS and PL sheet names from Excel file bytes using fuzzy matching.

    Args:
        xl_bytes: Excel file as bytes

    Returns:
        Tuple of (bs_sheet_name, pl_sheet_name). Either can be None if not found.
    """
    try:
        # Get all sheet names without loading full data
        xls = pd.ExcelFile(io.BytesIO(xl_bytes))
        sheet_names = xls.sheet_names

        # Find BS and PL sheets using fuzzy matching
        bs_sheet = find_sheet_fuzzy(sheet_names, is_balance_sheet=True)
        pl_sheet = find_sheet_fuzzy(sheet_names, is_balance_sheet=False)

        return bs_sheet, pl_sheet

    except Exception as e:
        print(f"Error detecting sheets: {str(e)}")
        return None, None
