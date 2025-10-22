"""Excel export service for contract extraction results."""
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime

from ..models.contract_schemas import ContractExtractionResult, ContractInfo, RatePeriod
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ContractExcelExporter:
    """Service for exporting contract extraction results to Excel format."""

    def __init__(self):
        """Initialize the Excel exporter."""
        self.column_order = [
            # Exact order as requested by user
            'Customer Name',
            'Contract No',
            'Contract Date',
            'Payment term',
            'Tax rate',
            'Unit',
            'Booking fee',
            'Deposit',
            'Handover Date',
            'Rent from',
            'Rent to',
            'No. Month of rent',
            'FOC from',
            'FOC to',
            'No month of FOC',
            'GFA',
            'Unit price/month',
            'Monthly Rental fee',
            'Service charge per m²/month',
            'Total service charge per month'
        ]

    def _calculate_months(self, start_date_str: str, end_date_str: str) -> Optional[int]:
        """
        Calculate number of months between two dates in MM-DD-YYYY format.
        Logic: Count month-to-month billing periods.
        - 09-15-2025 to 11-14-2025 = 2 months (Sept 15 to Oct 15, Oct 15 to Nov 15)
        - 10-01-2026 to 10-31-2026 = 1 month (partial month counts as 1)
        - 09-15-2025 to 09-14-2026 = 12 months (exactly 12 month periods)
        """
        try:
            start = datetime.strptime(start_date_str, '%m-%d-%Y')
            end = datetime.strptime(end_date_str, '%m-%d-%Y')

            # Calculate difference in months
            month_diff = (end.year - start.year) * 12 + (end.month - start.month)

            # If end day is >= start day, it's a complete month, otherwise it's partial
            # But we still count partial months
            if end.day >= start.day:
                # Complete month (e.g., Sept 15 to Oct 15 or later)
                months = month_diff + 1
            else:
                # Partial month (e.g., Sept 15 to Oct 14)
                months = month_diff

            # Ensure at least 1 month if there's any period
            return max(1, months)
        except (ValueError, AttributeError):
            return None

    def _contract_to_rows(self, result: ContractExtractionResult) -> List[dict]:
        """
        Convert a single contract extraction result into multiple rows (one per rate period).

        Args:
            result: ContractExtractionResult object

        Returns:
            List of dictionaries, one per rate period (or one row if no rate periods)
        """
        if not result.success or not result.data:
            logger.warning(f"Skipping failed extraction: {result.source_file}")
            return []

        contract = result.data
        rows = []

        # Base contract data (shared across all rows)
        base_data = {
            'Customer Name': contract.customer_name or contract.tenant or '',
            'Contract No': contract.contract_number or '',
            'Contract Date': contract.contract_date or '',
            'Payment term': contract.payment_terms_details or '',
            'Tax rate': '',  # Left blank as requested
            'Unit': '',  # Left blank as requested
            'Booking fee': '',  # Left blank as requested
            'Deposit': contract.deposit_amount or '',
            'Handover Date': contract.handover_date or '',
        }

        # If there are rate periods, create one row per period
        if contract.rate_periods and len(contract.rate_periods) > 0:
            for period in contract.rate_periods:
                row = base_data.copy()

                # Use AI-calculated months if available, otherwise fallback to calculation
                months = period.num_months if hasattr(period, 'num_months') and period.num_months else None
                if months is None and period.start_date and period.end_date:
                    months = str(self._calculate_months(period.start_date, period.end_date))

                # Use AI-calculated FOC months if available, otherwise fallback to calculation
                foc_months = period.foc_num_months if hasattr(period, 'foc_num_months') and period.foc_num_months else None
                if foc_months is None and period.foc_from and period.foc_to:
                    foc_months = str(self._calculate_months(period.foc_from, period.foc_to))

                # Service charge - get raw rate and calculate total
                service_charge_rate = period.service_charge_rate_per_sqm if hasattr(period, 'service_charge_rate_per_sqm') else ''
                total_service_charge = ''
                if service_charge_rate:
                    try:
                        rate = float(service_charge_rate)
                        gfa = float(contract.gfa) if contract.gfa else 0
                        if gfa > 0:
                            total_service_charge = str(rate * gfa)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not calculate total service charge: rate={service_charge_rate}, gfa={contract.gfa}")

                # Log for debugging
                logger.debug(f"Period {period.start_date} to {period.end_date}: months={months}, foc_months={foc_months}, service_charge_rate={service_charge_rate}, total_service_charge={total_service_charge}")

                row.update({
                    'Rent from': period.start_date or '',
                    'Rent to': period.end_date or '',
                    'No. Month of rent': months or '',
                    'FOC from': period.foc_from or '',
                    'FOC to': period.foc_to or '',
                    'No month of FOC': foc_months or '',
                    'GFA': contract.gfa or '',
                    'Unit price/month': period.monthly_rate_per_sqm or '',
                    'Monthly Rental fee': period.total_monthly_rate or '',
                    'Service charge per m²/month': service_charge_rate or '',
                    'Total service charge per month': total_service_charge or ''
                })
                rows.append(row)
        else:
            # No rate periods - create single row with base data
            row = base_data.copy()
            row.update({
                'Rent from': '',
                'Rent to': '',
                'No. Month of rent': '',
                'FOC from': '',
                'FOC to': '',
                'No month of FOC': '',
                'GFA': contract.gfa or '',
                'Unit price/month': '',
                'Monthly Rental fee': '',
                'Service charge per m²/month': '',
                'Total service charge per month': ''
            })
            rows.append(row)

        return rows

    def export_to_excel(
        self,
        results: List[ContractExtractionResult],
        output_path: Union[str, Path],
        include_failed: bool = False
    ) -> Path:
        """
        Export contract extraction results to Excel file.

        Format: One row per rate period (normalized format)
        - Each contract may have multiple rows if it has multiple rate periods
        - Contract-level data is duplicated across rows for the same contract

        Args:
            results: List of ContractExtractionResult objects
            output_path: Path where Excel file should be saved
            include_failed: If True, include failed extractions with error info

        Returns:
            Path to the created Excel file
        """
        output_path = Path(output_path)
        logger.info(f"Exporting {len(results)} contract(s) to Excel: {output_path}")

        all_rows = []

        for result in results:
            if result.success:
                rows = self._contract_to_rows(result)
                all_rows.extend(rows)
            elif include_failed:
                # Add a row for failed extraction
                all_rows.append({
                    'source_file': result.source_file,
                    'processing_time': result.processing_time,
                    'error': result.error,
                    'success': False
                })

        if not all_rows:
            logger.warning("No data to export")
            # Create empty DataFrame with column headers
            df = pd.DataFrame(columns=self.column_order)
        else:
            # Create DataFrame
            df = pd.DataFrame(all_rows)

            # Reorder columns according to column_order (keep any extra columns at the end)
            existing_ordered_cols = [col for col in self.column_order if col in df.columns]
            extra_cols = [col for col in df.columns if col not in self.column_order]
            df = df[existing_ordered_cols + extra_cols]

        # Write to Excel with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Contract Extractions', index=False)

            # Get the worksheet for formatting
            worksheet = writer.sheets['Contract Extractions']

            # Auto-adjust column widths
            for idx, col in enumerate(df.columns, start=1):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                # Cap at 50 characters
                max_length = min(max_length, 50)
                worksheet.column_dimensions[self._get_column_letter(idx)].width = max_length

            # Freeze header row
            worksheet.freeze_panes = 'A2'

        logger.info(f"Successfully exported {len(all_rows)} row(s) to {output_path}")
        return output_path

    def _get_column_letter(self, col_num: int) -> str:
        """
        Convert column number to Excel column letter (1 -> A, 2 -> B, etc.)

        Args:
            col_num: Column number (1-indexed)

        Returns:
            Excel column letter(s)
        """
        letter = ''
        while col_num > 0:
            col_num -= 1
            letter = chr(col_num % 26 + 65) + letter
            col_num //= 26
        return letter

    def export_summary_statistics(
        self,
        results: List[ContractExtractionResult],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Export summary statistics about the extraction batch.

        Args:
            results: List of ContractExtractionResult objects
            output_path: Path where Excel file should be saved

        Returns:
            Path to the created Excel file
        """
        output_path = Path(output_path)
        logger.info(f"Exporting extraction statistics to: {output_path}")

        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        avg_time = sum(r.processing_time or 0 for r in results) / total if total > 0 else 0

        # Count rate periods
        total_periods = 0
        for r in results:
            if r.success and r.data and r.data.rate_periods:
                total_periods += len(r.data.rate_periods)

        # Field extraction rates
        field_stats = {}
        if successful > 0:
            for field in ['customer_name', 'contract_number', 'contract_date', 'gfa',
                         'deposit_amount', 'handover_date', 'service_charge_rate']:
                extracted_count = sum(
                    1 for r in results
                    if r.success and r.data and getattr(r.data, field, None) is not None
                )
                field_stats[field] = f"{extracted_count}/{successful} ({extracted_count/successful*100:.1f}%)"

        # Create statistics DataFrame
        stats_data = {
            'Metric': [
                'Total Contracts Processed',
                'Successful Extractions',
                'Failed Extractions',
                'Success Rate',
                'Average Processing Time (seconds)',
                'Total Rate Periods Extracted',
                '',
                'Field Extraction Rates:',
                '  Customer Name',
                '  Contract Number',
                '  Contract Date',
                '  GFA',
                '  Deposit Amount',
                '  Handover Date',
                '  Service Charge Rate'
            ],
            'Value': [
                total,
                successful,
                failed,
                f"{successful/total*100:.1f}%" if total > 0 else "0%",
                f"{avg_time:.2f}",
                total_periods,
                '',
                '',
                field_stats.get('customer_name', 'N/A'),
                field_stats.get('contract_number', 'N/A'),
                field_stats.get('contract_date', 'N/A'),
                field_stats.get('gfa', 'N/A'),
                field_stats.get('deposit_amount', 'N/A'),
                field_stats.get('handover_date', 'N/A'),
                field_stats.get('service_charge_rate', 'N/A')
            ]
        }

        df_stats = pd.DataFrame(stats_data)

        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            worksheet = writer.sheets['Statistics']
            worksheet.column_dimensions['A'].width = 40
            worksheet.column_dimensions['B'].width = 30

        logger.info(f"Statistics exported to {output_path}")
        return output_path
