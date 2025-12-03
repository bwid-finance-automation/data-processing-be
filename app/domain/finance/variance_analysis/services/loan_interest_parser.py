#!/usr/bin/env python3
"""
Loan Interest Rate Parser

Parses ERP Loan Interest Rate Save Search Excel file to extract
interest rate history by entity for enhanced Rule A2 analysis.

File Structure (ERP_Loan interest rate_Save search.xlsx):
- Sheet: Sheet0
- Columns:
  - Bank Loan account ID: Loan ID (only on first row of each loan)
  - Entites: Entity name (match key)
  - Phase: Phase code
  - Interet Effect from: Date when rate became effective
  - Interest Rate: Rate as decimal (0.077 = 7.7%)
"""

import io
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from calendar import monthrange
import pandas as pd


# Days in each month (will calculate dynamically for leap years)
MONTH_ABBR_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


def get_days_in_month(month_name: str, year: int) -> int:
    """Get actual days in a month, accounting for leap years."""
    month_num = MONTH_ABBR_TO_NUM.get(month_name[:3].title(), 1)
    return monthrange(year, month_num)[1]


def parse_loan_interest_file(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse ERP Loan Interest Rate Save Search Excel file.

    Args:
        file_bytes: Raw bytes of the Excel file

    Returns:
        DataFrame with columns: Entity, Effective_Date, Interest_Rate
    """
    file_obj = io.BytesIO(file_bytes)

    # Read with header at row 4 (0-indexed)
    df = pd.read_excel(file_obj, sheet_name='Sheet0', header=4)

    # Rename columns for easier access
    column_mapping = {
        'Entites': 'Entity',
        'Interet Effect from': 'Effective_Date',
        'Interest Rate': 'Interest_Rate',
        'Phase': 'Phase',
        'Bank Loan account ID': 'Loan_ID'
    }

    df = df.rename(columns=column_mapping)

    # Forward fill Entity and Phase from main rows to continuation rows
    # (continuation rows have NaN in Loan_ID column)
    df['Entity'] = df['Entity'].ffill()
    df['Phase'] = df['Phase'].ffill()

    # Filter out header row if it slipped in
    df = df[df['Entity'] != 'Entites']

    # Convert Effective_Date to datetime
    df['Effective_Date'] = pd.to_datetime(df['Effective_Date'], errors='coerce')

    # Convert Interest_Rate to float
    df['Interest_Rate'] = pd.to_numeric(df['Interest_Rate'], errors='coerce')

    # Filter out rows with invalid data
    df = df[df['Effective_Date'].notna() & df['Interest_Rate'].notna()]

    # Filter out placeholder values
    df = df[~df['Interest_Rate'].astype(str).str.contains('No Interest', case=False, na=False)]

    # Sort by Entity and Effective_Date
    df = df.sort_values(['Entity', 'Effective_Date']).reset_index(drop=True)

    return df[['Entity', 'Phase', 'Effective_Date', 'Interest_Rate']]


def build_entity_rate_lookup(df: pd.DataFrame) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Build a lookup dictionary of interest rates by entity.

    Args:
        df: DataFrame from parse_loan_interest_file

    Returns:
        Dict mapping entity name to list of (effective_date, rate) tuples sorted by date
    """
    lookup = {}

    for entity in df['Entity'].unique():
        entity_df = df[df['Entity'] == entity].sort_values('Effective_Date')
        rates = [
            (row['Effective_Date'].to_pydatetime(), float(row['Interest_Rate']))
            for _, row in entity_df.iterrows()
        ]
        lookup[entity] = rates

    return lookup


def get_applicable_rate(
    entity_rates: List[Tuple[datetime, float]],
    target_date: datetime
) -> Optional[float]:
    """
    Find the applicable interest rate for a given date.

    The applicable rate is the most recent rate where Effective_Date <= target_date.

    Args:
        entity_rates: List of (effective_date, rate) tuples sorted by date
        target_date: The date to find the rate for

    Returns:
        The applicable interest rate, or None if no rate found
    """
    if not entity_rates:
        return None

    applicable_rate = None
    for effective_date, rate in entity_rates:
        if effective_date <= target_date:
            applicable_rate = rate
        else:
            break

    return applicable_rate


def get_rate_for_period(
    entity_rates: List[Tuple[datetime, float]],
    period_start: datetime,
    period_end: datetime
) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
    """
    Get the interest rate(s) applicable during a period.

    If rate changed during the period, returns both rates.

    Args:
        entity_rates: List of (effective_date, rate) tuples sorted by date
        period_start: Start of the period
        period_end: End of the period

    Returns:
        Tuple of (current_rate, previous_rate, rate_change_date)
        - current_rate: Rate at end of period
        - previous_rate: Rate at start of period (if different from current)
        - rate_change_date: Date when rate changed (if it changed)
    """
    if not entity_rates:
        return None, None, None

    rate_at_start = get_applicable_rate(entity_rates, period_start)
    rate_at_end = get_applicable_rate(entity_rates, period_end)

    # Check if rate changed during the period
    rate_change_date = None
    if rate_at_start != rate_at_end:
        # Find when the rate changed
        for effective_date, rate in entity_rates:
            if period_start < effective_date <= period_end:
                rate_change_date = effective_date
                break

    previous_rate = rate_at_start if rate_at_start != rate_at_end else None

    return rate_at_end, previous_rate, rate_change_date


def calculate_expected_interest(
    principal: float,
    rate: float,
    month_name: str,
    year: int
) -> float:
    """
    Calculate expected monthly interest based on actual days in month.

    Formula: Principal × Rate × (Days in Month / 365)

    Args:
        principal: Loan principal amount
        rate: Annual interest rate as decimal (e.g., 0.077 for 7.7%)
        month_name: Month name (e.g., 'Sep')
        year: Year (for leap year calculation)

    Returns:
        Expected interest amount
    """
    days_in_month = get_days_in_month(month_name, year)
    return principal * rate * (days_in_month / 365)


def extract_entity_from_row1(file_bytes: bytes, sheet_name: str) -> Optional[str]:
    """
    Extract entity name from Row 1 of BS/PL Breakdown file.

    Row 1 format: "Parent Company : BWID : VC1 : VC2 : ENTITY_NAME"
    Entity is always the LAST part after ":"

    Args:
        file_bytes: Raw bytes of the Excel file
        sheet_name: Name of the sheet to read

    Returns:
        Entity name, or None if not found
    """
    try:
        file_obj = io.BytesIO(file_bytes)
        df = pd.read_excel(file_obj, sheet_name=sheet_name, header=None, nrows=5)

        # Row 1 is index 1 (Row 0 is company name)
        if len(df) < 2:
            return None

        row1_text = str(df.iloc[1, 0])

        # Split by ":" and get the last part
        if ':' in row1_text:
            parts = row1_text.split(':')
            entity = parts[-1].strip()
            return entity if entity else None

        return None

    except Exception as e:
        print(f"Error extracting entity from Row 1: {e}")
        return None


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for matching.

    - Strip whitespace
    - Convert to uppercase
    - Remove common suffixes/prefixes
    """
    if not name:
        return ""

    normalized = name.strip().upper()

    # Remove common suffixes that might differ
    suffixes_to_remove = [
        ' CO., LTD', ' CO.,LTD', ' LTD.,CO', ' LTD,.CO',
        ' JSC', ' LLC', ' CO., LTD.', ' LTD'
    ]

    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix.upper()):
            normalized = normalized[:-len(suffix)]

    return normalized.strip()


def find_matching_entity(
    entity_from_file: str,
    available_entities: List[str]
) -> Optional[str]:
    """
    Find the best matching entity from the ERP loan file.

    Args:
        entity_from_file: Entity name extracted from BS/PL file
        available_entities: List of entity names from ERP loan file

    Returns:
        Best matching entity name, or None if no match found
    """
    if not entity_from_file:
        return None

    normalized_target = normalize_entity_name(entity_from_file)

    # First try exact match (after normalization)
    for entity in available_entities:
        if normalize_entity_name(entity) == normalized_target:
            return entity

    # Try partial match (target contained in entity or vice versa)
    for entity in available_entities:
        normalized_entity = normalize_entity_name(entity)
        if normalized_target in normalized_entity or normalized_entity in normalized_target:
            return entity

    # Try matching key parts (e.g., "BWID NAM HAI" matches "BWID NAM HAI CO., LTD")
    target_parts = set(normalized_target.split())
    for entity in available_entities:
        entity_parts = set(normalize_entity_name(entity).split())
        # If most words match
        common = target_parts & entity_parts
        if len(common) >= min(2, len(target_parts)):
            return entity

    return None


def generate_a2_explanation(
    loan_change: float,
    interest_change: float,
    current_rate: Optional[float],
    previous_rate: Optional[float],
    rate_change_date: Optional[datetime],
    expected_interest: Optional[float],
    actual_interest: float
) -> str:
    """
    Generate predefined explanation for Rule A2 based on the scenario.

    Args:
        loan_change: Change in loan principal
        interest_change: Change in interest expense
        current_rate: Current applicable interest rate
        previous_rate: Previous interest rate (if changed)
        rate_change_date: Date when rate changed (if applicable)
        expected_interest: Calculated expected interest
        actual_interest: Actual interest from P&L

    Returns:
        Predefined short explanation string
    """
    # Format rate as percentage
    def fmt_rate(r):
        return f"{r*100:.2f}%" if r else "N/A"

    # Scenario 1: Loan increased, interest didn't increase, rate same
    if loan_change > 0 and interest_change <= 0 and previous_rate is None:
        return f"Missing interest accrual - loan increased but interest not recorded. Rate: {fmt_rate(current_rate)}"

    # Scenario 2: Loan unchanged, interest changed, rate changed
    if abs(loan_change) < 1_000_000 and previous_rate is not None:
        date_str = rate_change_date.strftime('%Y-%m-%d') if rate_change_date else "unknown"
        return f"Interest change due to rate adjustment from {fmt_rate(previous_rate)} to {fmt_rate(current_rate)} effective {date_str}"

    # Scenario 3: Loan increased, interest decreased, rate dropped
    if loan_change > 0 and interest_change < 0 and previous_rate is not None:
        return f"Rate reduction offset loan increase - rate changed from {fmt_rate(previous_rate)} to {fmt_rate(current_rate)}"

    # Scenario 4: Loan decreased, interest unchanged
    if loan_change < 0 and abs(interest_change) < abs(loan_change * 0.01):
        return "Interest not adjusted after loan repayment"

    # Scenario 5: Variance between expected and actual
    if expected_interest is not None:
        variance = actual_interest - expected_interest
        if abs(variance) > 1_000_000:  # 1M VND threshold
            return f"Variance between expected interest ({expected_interest:,.0f}) and actual ({actual_interest:,.0f}) - review calculation"

    # Default
    return f"Review loan/interest relationship. Rate: {fmt_rate(current_rate)}"
