# app/utils/data_recovery.py
"""Data recovery and error handling utilities for malformed financial data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import re

from app.core.exceptions import DataQualityError, FileProcessingError
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)

class DataRecoveryEngine:
    """Engine for recovering from malformed financial data."""

    def __init__(self):
        self.recovery_stats = {
            "total_issues": 0,
            "recovered_issues": 0,
            "unrecoverable_issues": 0,
            "data_quality_warnings": []
        }

    def recover_financial_data(
        self,
        df: pd.DataFrame,
        sheet_name: str = "Unknown",
        expected_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Attempt to recover from various data quality issues in financial datasets.

        Returns:
            Tuple of (recovered_dataframe, recovery_report)
        """
        logger.info(f"Starting data recovery for sheet: {sheet_name}")
        self.recovery_stats = {
            "total_issues": 0,
            "recovered_issues": 0,
            "unrecoverable_issues": 0,
            "data_quality_warnings": []
        }

        if df is None or df.empty:
            raise DataQualityError(
                "DataFrame is empty or None",
                sheet_name=sheet_name
            )

        recovered_df = df.copy()

        try:
            # 1. Fix column names
            recovered_df = self._fix_column_names(recovered_df, sheet_name)

            # 2. Handle missing headers
            recovered_df = self._recover_missing_headers(recovered_df, expected_columns)

            # 3. Clean numeric data
            recovered_df = self._clean_numeric_data(recovered_df, sheet_name)

            # 4. Fix date/period columns
            recovered_df = self._fix_date_columns(recovered_df, sheet_name)

            # 5. Handle account codes
            recovered_df = self._fix_account_codes(recovered_df, sheet_name)

            # 6. Remove completely empty rows/columns
            recovered_df = self._remove_empty_data(recovered_df, sheet_name)

            # 7. Validate data consistency
            self._validate_data_consistency(recovered_df, sheet_name)

            logger.info(f"Data recovery completed for {sheet_name}. "
                       f"Issues: {self.recovery_stats['recovered_issues']}/{self.recovery_stats['total_issues']}")

            return recovered_df, self._generate_recovery_report(sheet_name)

        except Exception as e:
            logger.error(f"Data recovery failed for {sheet_name}: {str(e)}", exc_info=True)
            raise DataQualityError(
                f"Failed to recover data in sheet {sheet_name}",
                sheet_name=sheet_name,
                details=str(e)
            )

    def _fix_column_names(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Fix common column name issues."""
        original_columns = df.columns.tolist()
        fixed_columns = []

        for col in original_columns:
            fixed_col = str(col).strip()

            # Handle unnamed columns
            if pd.isna(col) or str(col).lower().startswith('unnamed'):
                self.recovery_stats["total_issues"] += 1
                # Try to infer column name from first few rows
                if len(df) > 0:
                    for i in range(min(5, len(df))):
                        potential_name = df.iloc[i][col]
                        if pd.notna(potential_name) and str(potential_name).strip():
                            fixed_col = f"Inferred_{str(potential_name).strip()}"
                            self.recovery_stats["recovered_issues"] += 1
                            break
                    else:
                        fixed_col = f"Column_{len(fixed_columns)}"
                        self.recovery_stats["data_quality_warnings"].append(
                            f"Unnamed column renamed to {fixed_col}"
                        )

            # Clean column names
            fixed_col = re.sub(r'[^\w\s.-]', '_', fixed_col)
            fixed_col = re.sub(r'\s+', '_', fixed_col)

            fixed_columns.append(fixed_col)

        if fixed_columns != original_columns:
            df.columns = fixed_columns
            logger.info(f"Fixed column names in {sheet_name}")

        return df

    def _recover_missing_headers(self, df: pd.DataFrame, expected_columns: List[str] = None) -> pd.DataFrame:
        """Attempt to recover missing headers by scanning first few rows."""
        if expected_columns is None:
            return df

        # Check if we have reasonable headers
        current_headers = df.columns.tolist()
        header_quality_score = self._assess_header_quality(current_headers)

        if header_quality_score < 0.5:  # Poor header quality
            logger.info("Poor header quality detected, attempting recovery")
            self.recovery_stats["total_issues"] += 1

            # Look for better headers in first 5 rows
            for row_idx in range(min(5, len(df))):
                potential_headers = df.iloc[row_idx].tolist()
                if self._assess_header_quality(potential_headers) > header_quality_score:
                    # Use this row as headers
                    new_df = df.iloc[row_idx + 1:].copy()
                    new_df.columns = [str(h).strip() if pd.notna(h) else f"Col_{i}"
                                     for i, h in enumerate(potential_headers)]
                    new_df = new_df.reset_index(drop=True)

                    self.recovery_stats["recovered_issues"] += 1
                    self.recovery_stats["data_quality_warnings"].append(
                        f"Used row {row_idx} as column headers"
                    )
                    return new_df

        return df

    def _assess_header_quality(self, headers: List[Any]) -> float:
        """Assess the quality of potential headers (0-1 score)."""
        if not headers:
            return 0.0

        score = 0.0
        total_headers = len(headers)

        for header in headers:
            if pd.isna(header):
                continue

            header_str = str(header).strip()

            # Good indicators
            if any(keyword in header_str.lower() for keyword in
                   ['account', 'period', 'amount', 'value', 'code', 'date', 'entity']):
                score += 0.3

            # Text-like (not pure numbers)
            if not header_str.replace('.', '').replace('-', '').isdigit():
                score += 0.2

            # Reasonable length
            if 3 <= len(header_str) <= 50:
                score += 0.1

            # Contains letters
            if any(c.isalpha() for c in header_str):
                score += 0.2

        return score / total_headers if total_headers > 0 else 0.0

    def _clean_numeric_data(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Clean and recover numeric data."""
        numeric_columns = []

        for col in df.columns:
            # Skip obvious non-numeric columns
            if any(keyword in str(col).lower() for keyword in
                   ['account', 'code', 'name', 'description', 'entity', 'customer']):
                continue

            # Try to identify numeric columns
            sample_values = df[col].dropna().head(10)
            if len(sample_values) == 0:
                continue

            # Check if values look numeric
            numeric_count = 0
            for val in sample_values:
                if self._is_potentially_numeric(val):
                    numeric_count += 1

            if numeric_count / len(sample_values) > 0.5:  # More than 50% look numeric
                numeric_columns.append(col)

        # Clean numeric columns
        for col in numeric_columns:
            original_count = df[col].notna().sum()
            df[col] = df[col].apply(self._clean_numeric_value)
            cleaned_count = df[col].notna().sum()

            if cleaned_count < original_count:
                lost_values = original_count - cleaned_count
                self.recovery_stats["total_issues"] += lost_values
                self.recovery_stats["data_quality_warnings"].append(
                    f"Lost {lost_values} non-numeric values in column '{col}'"
                )

        return df

    def _is_potentially_numeric(self, value: Any) -> bool:
        """Check if a value could be converted to numeric."""
        if pd.isna(value):
            return False

        str_val = str(value).strip()

        # Remove common formatting
        cleaned = str_val.replace(',', '').replace(' ', '').replace('(', '-').replace(')', '')

        try:
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False

    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """Clean and convert a single value to numeric."""
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value) if not pd.isna(value) else None

        str_val = str(value).strip()

        # Handle empty strings
        if not str_val:
            return None

        # Handle various formats
        str_val = str_val.replace(',', '')  # Remove thousands separator
        str_val = str_val.replace(' ', '')  # Remove spaces

        # Handle parentheses (negative numbers)
        if str_val.startswith('(') and str_val.endswith(')'):
            str_val = '-' + str_val[1:-1]

        # Handle currency symbols
        str_val = re.sub(r'[₫$€£¥]', '', str_val)

        try:
            return float(str_val)
        except (ValueError, TypeError):
            return None

    def _fix_date_columns(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Fix date and period columns."""
        date_columns = []

        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['date', 'period', 'month', 'year', 'time']):
                date_columns.append(col)

        for col in date_columns:
            # Try to standardize date/period formats
            df[col] = df[col].apply(self._standardize_period)

        return df

    def _standardize_period(self, value: Any) -> str:
        """Standardize period/date values."""
        if pd.isna(value):
            return None

        str_val = str(value).strip()

        # Common patterns
        patterns = [
            (r'(\d{4})-(\d{1,2})', r'\1-\2'),  # 2024-1 -> 2024-1
            (r'(\d{1,2})/(\d{4})', r'\2-\1'),  # 1/2024 -> 2024-1
            (r'T(\d{1,2})\s*(\d{4})', r'\2-\1'),  # T1 2024 -> 2024-1
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, str_val)
            if match:
                return re.sub(pattern, replacement, str_val)

        return str_val

    def _fix_account_codes(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Fix account code formatting."""
        account_columns = []

        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['account', 'code', 'acc']):
                account_columns.append(col)

        for col in account_columns:
            df[col] = df[col].apply(self._standardize_account_code)

        return df

    def _standardize_account_code(self, value: Any) -> str:
        """Standardize account code format."""
        if pd.isna(value):
            return None

        str_val = str(value).strip()

        # Remove spaces and special characters except dots
        str_val = re.sub(r'[^\w.]', '', str_val)

        return str_val

    def _remove_empty_data(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Remove completely empty rows and columns."""
        original_shape = df.shape

        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        # Remove empty rows
        df = df.dropna(axis=0, how='all')

        new_shape = df.shape
        if new_shape != original_shape:
            self.recovery_stats["data_quality_warnings"].append(
                f"Removed empty data: {original_shape[0] - new_shape[0]} rows, "
                f"{original_shape[1] - new_shape[1]} columns"
            )

        return df

    def _validate_data_consistency(self, df: pd.DataFrame, sheet_name: str) -> None:
        """Validate data consistency and add warnings."""
        # Check for suspicious patterns
        if len(df) < 5:
            self.recovery_stats["data_quality_warnings"].append(
                f"Very few rows of data ({len(df)}) in {sheet_name}"
            )

        if len(df.columns) < 3:
            self.recovery_stats["data_quality_warnings"].append(
                f"Very few columns ({len(df.columns)}) in {sheet_name}"
            )

        # Check for completely missing data in key areas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.recovery_stats["data_quality_warnings"].append(
                f"No numeric columns detected in {sheet_name}"
            )

    def _generate_recovery_report(self, sheet_name: str) -> Dict[str, Any]:
        """Generate a comprehensive recovery report."""
        return {
            "sheet_name": sheet_name,
            "recovery_success_rate": (
                self.recovery_stats["recovered_issues"] / max(1, self.recovery_stats["total_issues"])
            ),
            "total_issues_found": self.recovery_stats["total_issues"],
            "issues_recovered": self.recovery_stats["recovered_issues"],
            "unrecoverable_issues": self.recovery_stats["unrecoverable_issues"],
            "data_quality_warnings": self.recovery_stats["data_quality_warnings"],
            "recovery_timestamp": datetime.now().isoformat()
        }

# Utility functions
def safe_read_excel_with_recovery(
    file_path_or_bytes: Union[str, bytes],
    sheet_name: str,
    expected_columns: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely read Excel data with automatic recovery from common issues.

    Returns:
        Tuple of (dataframe, recovery_report)
    """
    recovery_engine = DataRecoveryEngine()

    try:
        # Try to read the Excel file
        if isinstance(file_path_or_bytes, bytes):
            import io
            df = pd.read_excel(io.BytesIO(file_path_or_bytes), sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path_or_bytes, sheet_name=sheet_name)

        # Apply recovery
        recovered_df, recovery_report = recovery_engine.recover_financial_data(
            df, sheet_name, expected_columns
        )

        return recovered_df, recovery_report

    except Exception as e:
        logger.error(f"Failed to read Excel sheet {sheet_name}: {str(e)}")
        raise FileProcessingError(
            f"Cannot read sheet '{sheet_name}' from Excel file",
            details=str(e)
        )

def validate_financial_data_structure(
    df: pd.DataFrame,
    sheet_type: str,
    required_patterns: Dict[str, List[str]] = None
) -> List[str]:
    """
    Validate that financial data has expected structure.

    Args:
        df: DataFrame to validate
        sheet_type: Type of sheet ('BS' or 'PL')
        required_patterns: Expected column patterns

    Returns:
        List of validation warnings
    """
    warnings = []

    if required_patterns is None:
        required_patterns = {
            'BS': ['account', 'period', 'amount'],
            'PL': ['account', 'period', 'amount']
        }

    expected_patterns = required_patterns.get(sheet_type, [])

    # Check for required column patterns
    found_patterns = []
    for pattern in expected_patterns:
        for col in df.columns:
            if pattern.lower() in str(col).lower():
                found_patterns.append(pattern)
                break

    missing_patterns = set(expected_patterns) - set(found_patterns)
    if missing_patterns:
        warnings.append(f"Missing expected column patterns: {', '.join(missing_patterns)}")

    # Check data volume
    if len(df) < 10:
        warnings.append(f"Very few rows of data ({len(df)}) for {sheet_type} sheet")

    # Check for reasonable number of periods
    period_columns = [col for col in df.columns if 'period' in str(col).lower() or
                     any(str(col).startswith(year) for year in ['2020', '2021', '2022', '2023', '2024', '2025'])]

    if len(period_columns) < 2:
        warnings.append("Insufficient period columns for meaningful analysis")

    return warnings