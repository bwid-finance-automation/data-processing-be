"""Utility for reading and processing unit breakdown Excel files."""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class UnitBreakdown(BaseModel):
    """Model for individual unit breakdown."""
    customer_code: Optional[str] = None
    customer_name: Optional[str] = None
    tax_rate: Optional[float] = None
    unit: str
    gfa: float  # Gross Floor Area for this specific unit


class UnitBreakdownResult(BaseModel):
    """Result of processing unit breakdown file."""
    success: bool
    units: List[UnitBreakdown] = []
    total_gfa: Optional[float] = None
    error: Optional[str] = None
    source_file: Optional[str] = None


class UnitBreakdownReader:
    """Service for reading unit breakdown Excel files."""

    def __init__(self):
        """Initialize the unit breakdown reader."""
        logger.info("UnitBreakdownReader initialized")

    def read_unit_breakdown(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[int, str] = 0
    ) -> UnitBreakdownResult:
        """
        Read unit breakdown from Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name or index (default: 0 for first sheet)

        Returns:
            UnitBreakdownResult with list of units and total GFA
        """
        file_path = Path(file_path)

        logger.info(f"Reading unit breakdown from: {file_path.name}")

        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return UnitBreakdownResult(
                success=False,
                error=error_msg,
                source_file=str(file_path.name)
            )

        try:
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            logger.info(f"Loaded Excel with {len(df)} rows and columns: {list(df.columns)}")

            # Normalize column names (case-insensitive matching)
            df.columns = df.columns.str.strip()
            column_map = {col.lower(): col for col in df.columns}

            # Required columns
            required_cols = ['unit', 'gfa']
            missing_cols = [col for col in required_cols if col not in column_map]

            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
                logger.error(error_msg)
                return UnitBreakdownResult(
                    success=False,
                    error=error_msg,
                    source_file=str(file_path.name)
                )

            # Get actual column names
            unit_col = column_map['unit']
            gfa_col = column_map['gfa']
            customer_code_col = column_map.get('customer code')
            customer_name_col = column_map.get('customer name')
            tax_rate_col = column_map.get('tax rate')

            # Parse units
            units = []
            for idx, row in df.iterrows():
                try:
                    # Skip rows with empty unit or GFA
                    if pd.isna(row[unit_col]) or pd.isna(row[gfa_col]):
                        logger.warning(f"Skipping row {idx + 1}: missing unit or GFA")
                        continue

                    unit_data = {
                        'unit': str(row[unit_col]).strip(),
                        'gfa': float(row[gfa_col])
                    }

                    # Add optional fields if present
                    if customer_code_col and not pd.isna(row.get(customer_code_col)):
                        unit_data['customer_code'] = str(row[customer_code_col]).strip()

                    if customer_name_col and not pd.isna(row.get(customer_name_col)):
                        unit_data['customer_name'] = str(row[customer_name_col]).strip()

                    if tax_rate_col and not pd.isna(row.get(tax_rate_col)):
                        unit_data['tax_rate'] = float(row[tax_rate_col])

                    unit_breakdown = UnitBreakdown(**unit_data)
                    units.append(unit_breakdown)

                    logger.debug(f"Parsed unit: {unit_breakdown.unit} with GFA: {unit_breakdown.gfa}")

                except Exception as e:
                    logger.warning(f"Error parsing row {idx + 1}: {e}")
                    continue

            if not units:
                error_msg = "No valid units found in Excel file"
                logger.error(error_msg)
                return UnitBreakdownResult(
                    success=False,
                    error=error_msg,
                    source_file=str(file_path.name)
                )

            # Calculate total GFA
            total_gfa = sum(unit.gfa for unit in units)

            logger.info(f"Successfully parsed {len(units)} units with total GFA: {total_gfa}")

            # Print summary
            print("\n" + "="*80)
            print("📊 UNIT BREAKDOWN SUMMARY:")
            print("="*80)
            print(f"Total units: {len(units)}")
            print(f"Total GFA: {total_gfa:,.2f} sqm")
            print("\nUnits:")
            for i, unit in enumerate(units, 1):
                print(f"  {i}. {unit.unit}: {unit.gfa:,.2f} sqm")
            print("="*80 + "\n")

            return UnitBreakdownResult(
                success=True,
                units=units,
                total_gfa=total_gfa,
                source_file=str(file_path.name)
            )

        except Exception as e:
            error_msg = f"Error reading unit breakdown file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return UnitBreakdownResult(
                success=False,
                error=error_msg,
                source_file=str(file_path.name)
            )

    def validate_gfa_match(
        self,
        breakdown_result: UnitBreakdownResult,
        contract_gla: float,
        tolerance: float = 0.01  # 1% tolerance
    ) -> Dict[str, any]:
        """
        Validate that the sum of unit GFAs matches the contract GLA.

        Args:
            breakdown_result: Result from read_unit_breakdown
            contract_gla: GLA from contract (total)
            tolerance: Percentage tolerance for mismatch (default: 0.01 = 1%)

        Returns:
            Dictionary with validation result
        """
        if not breakdown_result.success or not breakdown_result.total_gfa:
            return {
                'valid': False,
                'error': 'Invalid breakdown result or missing total GFA'
            }

        total_gfa = breakdown_result.total_gfa
        difference = abs(total_gfa - contract_gla)
        difference_pct = (difference / contract_gla) * 100 if contract_gla > 0 else 100

        is_valid = difference_pct <= (tolerance * 100)

        result = {
            'valid': is_valid,
            'contract_gla': contract_gla,
            'total_unit_gfa': total_gfa,
            'difference': difference,
            'difference_pct': difference_pct,
            'tolerance_pct': tolerance * 100
        }

        if is_valid:
            logger.info(f"✓ GFA validation passed: Contract GLA ({contract_gla}) matches "
                       f"unit breakdown total ({total_gfa}) within {tolerance*100}% tolerance")
            print("\n" + "="*80)
            print("✓ GFA VALIDATION PASSED")
            print("="*80)
            print(f"Contract GLA: {contract_gla:,.2f} sqm")
            print(f"Unit breakdown total: {total_gfa:,.2f} sqm")
            print(f"Difference: {difference:,.2f} sqm ({difference_pct:.2f}%)")
            print("="*80 + "\n")
        else:
            logger.warning(f"✗ GFA validation failed: Contract GLA ({contract_gla}) does not match "
                          f"unit breakdown total ({total_gfa}). Difference: {difference} ({difference_pct:.2f}%)")
            print("\n" + "="*80)
            print("✗ GFA VALIDATION FAILED")
            print("="*80)
            print(f"Contract GLA: {contract_gla:,.2f} sqm")
            print(f"Unit breakdown total: {total_gfa:,.2f} sqm")
            print(f"Difference: {difference:,.2f} sqm ({difference_pct:.2f}%)")
            print(f"Tolerance: {tolerance*100}%")
            print("="*80 + "\n")

        return result
