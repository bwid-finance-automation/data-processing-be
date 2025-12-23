"""
GLA Processor Service
Handles parsing and processing of GLA Excel data from Unit For Lease lists.
"""
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from app.shared.utils.logging_config import get_logger
from ..models.gla_models import GLARecord, ProjectGLASummary, LeaseStatus, ProductType, TenantGLA

logger = get_logger(__name__)


# Project name mapping from code to readable name
PROJECT_NAME_MAPPING = {
    'PTU1': 'VSIP 2A',
    'PBLB': 'Amata',
    'PLV1': 'Deep C 1',
    'PSG1': 'Tan Phu Trung 1',
    'PTPT': 'Tan Phu Trung 4',
    'PSTL': 'Song Than 1',
    'PMP3': 'My Phuoc 3',
    'PDAL': 'Dong An 1',
    'PBN1': 'VSIP Bac Ninh',
    'PSCC': 'Supply Chain City',
    'PSL1': 'Tan Phu Trung 3',
    'PHD1': 'VSIP Hai Duong 1',
    'PHP1': 'VSIP Hai Phong',
    'PBB1': 'Bau Bang',
    'PBBA': 'Bau Bang',
    'PTH2': 'My Phuoc 4 A4',
    'PTH1': 'My Phuoc 4 A3',
    'PHD2': 'VSIP Hai Duong 2',
    'PNV1': 'Nam Dinh Vu',
    'PNV2': 'Nam Dinh Vu 2',
    'PLV2': 'Deep C 2',
    'PTDH': 'Tan Dong Hiep B',
    'PBNA': 'VSIP Nghe An',
    'PNTR': 'Nhon Trach 2 (Lot A)',
    'PNT2': 'Nhon Trach 2 (Lot B)',
    'PNT3': 'Nhon Trach 2 (Lot C)',
    'PHD3': 'VSIP Hai Duong 3',
    'PPN1': 'Phu Nghia',
    'PNKT': 'Nhon Trach 1',
    'PLV3': 'Deep C Quang Ninh',
    'PBDG': 'Dau Giay',
    'PTT3': 'Thuan Thanh 3B',
    'PLMX': 'Electron_LMX',
    'PETD': 'Electron_TDH B',
    'PST2': 'Electron_ST',
    'PNVH': 'Nam Dinh Vu-Nam Hai',
    'PBB6': 'Bau Bang 6',
    'PNVJ': 'Nam Dinh Vu JV',
    'PXAH': 'Xuyen A - Lot HK',
    'PDAR': 'Dong An 1 redevelopment',
    'PVL2': 'Vinh Loc 2_LOT A',
    'PXAL': 'Xuyen A - Lot LK',
    'PBSM': 'Song May',
    'PYPJ': 'Yen Phong JV',
    'PBN3': 'VSIP Bac Ninh 2',
    'PBTD': 'Thuan Dao',
    'PNSH': 'Nam Son Hap Linh JV_Lot A2',
    'PDC2': 'Electron_Deep C2',
    'PPAT': 'Electron Project_Phu An Thanh',
    'PLLA': 'Xuyen A - Lot HK',  # Logistics LOGOS Long An
}


class GLAProcessor:
    """
    Processes GLA Excel files and extracts Unit For Lease data.
    """

    # Expected column names (row 4 is header in input files)
    HEADER_ROW = 4
    REQUIRED_COLUMNS = [
        'Project Name',
        'CCS_Product Type',
        'Region',
        'GLA for Lease',
        'Unit for Lease Status',
        'BWID Project.'
    ]

    # Required sheet names (exact match)
    REQUIRED_SHEETS = {
        'Handover GLA - Previous': ('handover', 'previous'),
        'Handover GLA - Current': ('handover', 'current'),
        'Committed GLA - Previous': ('committed', 'previous'),
        'Committed GLA - Current': ('committed', 'current'),
    }

    def __init__(self):
        self.project_mapping = PROJECT_NAME_MAPPING.copy()

    def detect_sheet_type(self, sheet_name: str) -> Tuple[str, str]:
        """
        Detect if sheet is Handover or Committed and which period (previous/current).

        Returns:
            Tuple of (data_type, period) e.g., ('handover', 'previous') or ('committed', 'current')
        """
        sheet_lower = sheet_name.lower()

        # Determine data type
        if 'handover' in sheet_lower:
            data_type = 'handover'
        elif 'committed' in sheet_lower:
            data_type = 'committed'
        else:
            data_type = 'unknown'

        # Determine period based on sheet naming convention
        # T10 = previous, T11 = current (or similar patterns)
        if 't10' in sheet_lower or 'previous' in sheet_lower or 'old' in sheet_lower:
            period = 'previous'
        elif 't11' in sheet_lower or 'current' in sheet_lower or 'new' in sheet_lower:
            period = 'current'
        else:
            # Default based on position - will be overridden by file order
            period = 'unknown'

        return data_type, period

    def read_gla_sheet(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Read a GLA sheet from Excel file with proper header handling.
        """
        logger.info(f"Reading sheet '{sheet_name}' from {file_path}")

        try:
            # Read with header at row 4 (0-indexed)
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=self.HEADER_ROW)

            # Log available columns for debugging
            logger.debug(f"Available columns: {list(df.columns)}")

            return df
        except Exception as e:
            logger.error(f"Error reading sheet {sheet_name}: {str(e)}")
            raise

    def filter_by_status(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Filter records by status based on data type.

        - Handover: Only 'Handed Over' status
        - Committed: 'Open' + 'Handed Over' status
        """
        status_col = 'Unit for Lease Status'

        if status_col not in df.columns:
            logger.warning(f"Status column '{status_col}' not found in DataFrame")
            return df

        if data_type == 'handover':
            # Handover: Only records that have been handed over
            mask = df[status_col] == LeaseStatus.HANDED_OVER.value
        elif data_type == 'committed':
            # Committed: Open + Handed Over
            mask = df[status_col].isin([LeaseStatus.OPEN.value, LeaseStatus.HANDED_OVER.value])
        else:
            # No filtering for unknown type
            return df

        filtered = df[mask].copy()
        logger.info(f"Filtered {len(df)} records to {len(filtered)} for {data_type}")
        return filtered

    def extract_readable_project_name(self, bwid_project: str, project_code: str) -> str:
        """
        Extract readable project name from BWID Project field or project code.

        BWID Project format: 'PBBA: Bau Bang'
        """
        if pd.notna(bwid_project) and ':' in str(bwid_project):
            return str(bwid_project).split(':', 1)[1].strip()

        # Fall back to mapping
        if project_code in self.project_mapping:
            return self.project_mapping[project_code]

        return project_code

    def aggregate_by_project_type(
        self,
        df: pd.DataFrame,
        data_type: str
    ) -> Dict[Tuple[str, str], ProjectGLASummary]:
        """
        Aggregate GLA by project name and product type, including tenant details.

        Returns:
            Dict mapping (project_name, product_type) to ProjectGLASummary
        """
        # Filter by status first
        filtered_df = self.filter_by_status(df, data_type)

        # Check required columns exist
        required = ['Project Name', 'CCS_Product Type', 'Region', 'GLA for Lease', 'BWID Project.']
        missing = [col for col in required if col not in filtered_df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            # Try to work with available columns
            if 'Project Name' not in filtered_df.columns:
                return {}

        # Check for tenant column (may have different names)
        tenant_col = None
        for col_name in ['Tenant Name', 'Tenant', 'Customer Name', 'Customer', 'Account Name']:
            if col_name in filtered_df.columns:
                tenant_col = col_name
                break

        # Check for status column
        status_col = 'Unit for Lease Status' if 'Unit for Lease Status' in filtered_df.columns else None

        # Check for new attribute columns (Handover sheet)
        handover_gla_col = 'Handover GLA' if 'Handover GLA' in filtered_df.columns else None
        monthly_gross_rent_col = 'Monthly Gross rent' if 'Monthly Gross rent' in filtered_df.columns else None
        monthly_rate_col = 'Monthly rate' if 'Monthly rate' in filtered_df.columns else None

        # Check for new attribute columns (Committed sheet) - note typos in Excel
        committed_gla_col = 'Commited GLA' if 'Commited GLA' in filtered_df.columns else None
        months_to_expire_col = 'Months to expire' if 'Months to expire' in filtered_df.columns else None
        months_to_expire_x_gla_col = 'Month to exprire x commited GLA' if 'Month to exprire x commited GLA' in filtered_df.columns else None

        summaries: Dict[Tuple[str, str], ProjectGLASummary] = {}
        # Track tenant GLA per project-type
        tenant_data: Dict[Tuple[str, str], Dict[str, TenantGLA]] = {}

        for _, row in filtered_df.iterrows():
            try:
                project_code = str(row.get('Project Name', '')).strip()
                bwid_project = row.get('BWID Project.', '')
                product_type = str(row.get('CCS_Product Type', '')).strip()
                region = str(row.get('Region', '')).strip()
                gla = float(row.get('GLA for Lease', 0) or 0)

                # Get tenant name
                tenant_name = "Unknown"
                if tenant_col and pd.notna(row.get(tenant_col)):
                    tenant_name = str(row.get(tenant_col)).strip()
                    if not tenant_name or tenant_name == '- None -':
                        tenant_name = "Vacant"

                # Get status
                status = "Unknown"
                if status_col and pd.notna(row.get(status_col)):
                    status = str(row.get(status_col)).strip()

                # Get new attributes - Handover sheet
                handover_gla = 0.0
                if handover_gla_col and pd.notna(row.get(handover_gla_col)):
                    handover_gla = float(row.get(handover_gla_col) or 0)

                monthly_gross_rent = 0.0
                if monthly_gross_rent_col and pd.notna(row.get(monthly_gross_rent_col)):
                    monthly_gross_rent = float(row.get(monthly_gross_rent_col) or 0)

                monthly_rate = 0.0
                if monthly_rate_col and pd.notna(row.get(monthly_rate_col)):
                    monthly_rate = float(row.get(monthly_rate_col) or 0)

                # Get new attributes - Committed sheet
                committed_gla = 0.0
                if committed_gla_col and pd.notna(row.get(committed_gla_col)):
                    committed_gla = float(row.get(committed_gla_col) or 0)

                months_to_expire = 0.0
                if months_to_expire_col and pd.notna(row.get(months_to_expire_col)):
                    months_to_expire = float(row.get(months_to_expire_col) or 0)

                months_to_expire_x_gla = 0.0
                if months_to_expire_x_gla_col and pd.notna(row.get(months_to_expire_x_gla_col)):
                    months_to_expire_x_gla = float(row.get(months_to_expire_x_gla_col) or 0)

                # Skip invalid records
                if not project_code or not product_type or pd.isna(gla):
                    continue

                # Only process RBF and RBW
                if product_type not in [ProductType.RBF.value, ProductType.RBW.value]:
                    continue

                # Get readable project name
                readable_name = self.extract_readable_project_name(bwid_project, project_code)

                key = (readable_name, product_type)

                if key not in summaries:
                    summaries[key] = ProjectGLASummary(
                        project_name=readable_name,
                        product_type=product_type,
                        region=region,
                        gla_sqm=0.0,
                        tenants=[]
                    )
                    tenant_data[key] = {}

                summaries[key].gla_sqm += gla

                # Aggregate new attributes - Handover sheet (sum for rent, weighted avg for rate)
                summaries[key].handover_gla += handover_gla
                summaries[key].monthly_gross_rent += monthly_gross_rent
                # For monthly_rate, we'll calculate weighted average at the end

                # Aggregate new attributes - Committed sheet
                summaries[key].committed_gla += committed_gla
                summaries[key].months_to_expire_x_committed_gla += months_to_expire_x_gla
                # For months_to_expire, we'll calculate weighted average at the end

                # Track tenant GLA (aggregate by tenant within project)
                if tenant_name not in tenant_data[key]:
                    tenant_data[key][tenant_name] = TenantGLA(
                        tenant_name=tenant_name,
                        gla_sqm=0.0,
                        status=status
                    )
                tenant_data[key][tenant_name].gla_sqm += gla
                tenant_data[key][tenant_name].handover_gla += handover_gla
                tenant_data[key][tenant_name].monthly_gross_rent += monthly_gross_rent
                tenant_data[key][tenant_name].committed_gla += committed_gla
                tenant_data[key][tenant_name].months_to_expire_x_committed_gla += months_to_expire_x_gla

            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue

        # Convert tenant dicts to lists and calculate weighted averages
        for key, tenant_dict in tenant_data.items():
            summaries[key].tenants = list(tenant_dict.values())

            # Calculate weighted average monthly rate (weighted by handover GLA)
            if summaries[key].handover_gla > 0:
                summaries[key].monthly_rate = summaries[key].monthly_gross_rent / summaries[key].handover_gla

            # Calculate weighted average months to expire (weighted by committed GLA)
            if summaries[key].committed_gla > 0:
                summaries[key].months_to_expire = summaries[key].months_to_expire_x_committed_gla / summaries[key].committed_gla

        logger.info(f"Aggregated {len(summaries)} project-type combinations for {data_type}")
        return summaries

    def process_file(self, file_path: str) -> Dict[str, Dict[str, Dict[Tuple[str, str], ProjectGLASummary]]]:
        """
        Process a single Excel file containing multiple GLA sheets.

        Returns:
            Nested dict: {data_type: {period: {(project, type): summary}}}
        """
        logger.info(f"Processing file: {file_path}")

        result = {
            'handover': {},
            'committed': {}
        }

        try:
            # Get all sheet names
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            logger.info(f"Found sheets: {sheet_names}")

            for sheet_name in sheet_names:
                # Skip output sheets
                if sheet_name.startswith('>>') or sheet_name.lower() == 'output':
                    continue

                data_type, period = self.detect_sheet_type(sheet_name)

                if data_type == 'unknown':
                    logger.debug(f"Skipping unknown sheet type: {sheet_name}")
                    continue

                df = self.read_gla_sheet(file_path, sheet_name)
                summaries = self.aggregate_by_project_type(df, data_type)

                result[data_type][sheet_name] = summaries

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

        return result

    def validate_sheet_names(self, sheet_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that the Excel file contains exactly the required sheet names.

        Returns:
            Tuple of (is_valid, missing_sheets)
        """
        missing = []
        for required_sheet in self.REQUIRED_SHEETS.keys():
            if required_sheet not in sheet_names:
                missing.append(required_sheet)

        return len(missing) == 0, missing

    def process_single_file_with_periods(
        self,
        file_path: str
    ) -> Dict[str, Dict[str, Dict[Tuple[str, str], ProjectGLASummary]]]:
        """
        Process a single Excel file containing exactly 4 sheets with specific names:
        - 'Handover GLA - Previous'
        - 'Handover GLA - Current'
        - 'Committed GLA - Previous'
        - 'Committed GLA - Current'

        Returns:
            Dict with structure: {
                'handover': {'previous': {...}, 'current': {...}},
                'committed': {'previous': {...}, 'current': {...}}
            }

        Raises:
            ValueError: If required sheets are missing
        """
        logger.info(f"Processing single file with 4 sheets: {file_path}")

        result = {
            'handover': {'previous': {}, 'current': {}},
            'committed': {'previous': {}, 'current': {}}
        }

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            logger.info(f"Found sheets: {sheet_names}")

            # Validate sheet names
            is_valid, missing_sheets = self.validate_sheet_names(sheet_names)

            if not is_valid:
                error_msg = (
                    f"Invalid sheet names. Missing required sheets: {missing_sheets}\n\n"
                    f"Please ensure your Excel file contains exactly these 4 sheets:\n"
                    f"  • Handover GLA - Previous\n"
                    f"  • Handover GLA - Current\n"
                    f"  • Committed GLA - Previous\n"
                    f"  • Committed GLA - Current\n\n"
                    f"Found sheets: {sheet_names}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Process each required sheet
            for sheet_name, (data_type, period) in self.REQUIRED_SHEETS.items():
                logger.info(f"Processing sheet '{sheet_name}' as {data_type}/{period}")
                df = self.read_gla_sheet(file_path, sheet_name)
                summaries = self.aggregate_by_project_type(df, data_type)
                result[data_type][period] = summaries
                logger.info(f"Processed {len(summaries)} projects for {data_type}/{period}")

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

        return result

    def process_two_files(
        self,
        previous_file: str,
        current_file: str
    ) -> Dict[str, Dict[str, Dict[Tuple[str, str], ProjectGLASummary]]]:
        """
        Process two separate files (previous and current period).

        DEPRECATED: Use process_single_file_with_periods instead.

        Returns:
            Dict with structure: {
                'handover': {'previous': {...}, 'current': {...}},
                'committed': {'previous': {...}, 'current': {...}}
            }
        """
        logger.info(f"Processing previous file: {previous_file}")
        logger.info(f"Processing current file: {current_file}")

        result = {
            'handover': {'previous': {}, 'current': {}},
            'committed': {'previous': {}, 'current': {}}
        }

        # Process previous file
        prev_data = self._process_single_period_file(previous_file)
        for data_type in ['handover', 'committed']:
            if data_type in prev_data:
                result[data_type]['previous'] = prev_data[data_type]

        # Process current file
        curr_data = self._process_single_period_file(current_file)
        for data_type in ['handover', 'committed']:
            if data_type in curr_data:
                result[data_type]['current'] = curr_data[data_type]

        return result

    def _process_single_period_file(
        self,
        file_path: str
    ) -> Dict[str, Dict[Tuple[str, str], ProjectGLASummary]]:
        """
        Process a file that contains data for a single period.

        Detects sheets and aggregates by data type.
        """
        result = {
            'handover': {},
            'committed': {}
        }

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            logger.info(f"Processing file with sheets: {sheet_names}")

            for sheet_name in sheet_names:
                # Skip output sheets
                if sheet_name.startswith('>>') or sheet_name.lower() == 'output':
                    continue

                data_type, _ = self.detect_sheet_type(sheet_name)

                if data_type == 'unknown':
                    # Try to infer from sheet content or use heuristics
                    sheet_lower = sheet_name.lower()
                    if 'hand' in sheet_lower:
                        data_type = 'handover'
                    elif 'commit' in sheet_lower:
                        data_type = 'committed'
                    else:
                        logger.debug(f"Skipping sheet: {sheet_name}")
                        continue

                df = self.read_gla_sheet(file_path, sheet_name)
                summaries = self.aggregate_by_project_type(df, data_type)

                # Merge summaries for this data type
                for key, summary in summaries.items():
                    if key not in result[data_type]:
                        result[data_type][key] = summary
                    else:
                        # Add GLA values if same key appears in multiple sheets
                        result[data_type][key].gla_sqm += summary.gla_sqm

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

        return result

    def process_pivot_table_file(
        self,
        file_path: str,
        previous_month: Tuple[int, int] = None,
        current_month: Tuple[int, int] = None,
        ai_analyzer=None
    ) -> Dict[str, Dict[str, Dict[Tuple[str, str], ProjectGLASummary]]]:
        """
        Process a pivot table format Excel file where monthly GLA values are in separate columns.

        This method uses AI detection to identify the file structure and extract the correct
        monthly GLA columns for comparison.

        Args:
            file_path: Path to the Excel file
            previous_month: Tuple of (year, month) for previous period, e.g., (2024, 11)
            current_month: Tuple of (year, month) for current period, e.g., (2024, 12)
            ai_analyzer: Optional GLAAIAnalyzer instance for structure detection

        Returns:
            Dict with structure: {
                'handover': {'previous': {...}, 'current': {...}},
                'committed': {'previous': {...}, 'current': {...}}
            }
        """
        from .gla_ai_analyzer import GLAAIAnalyzer

        logger.info(f"Processing pivot table file: {file_path}")

        # Initialize AI analyzer if not provided
        if ai_analyzer is None:
            ai_analyzer = GLAAIAnalyzer()

        # Detect file structure
        structure = ai_analyzer.detect_file_structure(file_path)
        logger.info(f"Detected format: {structure.get('format')}")

        if structure.get('format') != 'pivot_table':
            logger.info("File is not pivot table format, using standard processing")
            return self.process_single_file_with_periods(file_path)

        # Detect comparison months if not provided
        if previous_month is None or current_month is None:
            months = ai_analyzer.detect_comparison_months(file_path)
            previous_month = previous_month or months.get('previous_month')
            current_month = current_month or months.get('current_month')
            logger.info(f"Detected months - Previous: {previous_month}, Current: {current_month}")

        monthly_columns = structure.get('monthly_columns', {})
        date_row = structure.get('date_row', 3)
        header_row = structure.get('header_row', 4)
        data_start_row = structure.get('data_start_row', 5)

        # Validate that we have the required months
        if previous_month not in monthly_columns:
            logger.warning(f"Previous month {previous_month} not found in data. Available: {list(monthly_columns.keys())}")
        if current_month not in monthly_columns:
            logger.warning(f"Current month {current_month} not found in data. Available: {list(monthly_columns.keys())}")

        pivot_result = {
            'handover': {'previous': {}, 'current': {}},
            'committed': {'previous': {}, 'current': {}}
        }

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            for sheet_name in sheet_names:
                # Skip output sheets
                if sheet_name.startswith('>>') or sheet_name.lower() == 'output':
                    continue

                data_type, _ = self.detect_sheet_type(sheet_name)
                if data_type == 'unknown':
                    continue

                logger.info(f"Processing pivot sheet '{sheet_name}' for {data_type}")

                # Get monthly columns for this sheet (they might differ slightly)
                sheet_structure = ai_analyzer.detect_file_structure(file_path, sheet_name)
                sheet_monthly_cols = sheet_structure.get('monthly_columns', monthly_columns)

                # Process previous month
                if previous_month in sheet_monthly_cols:
                    prev_summaries = self._aggregate_pivot_data(
                        file_path, sheet_name, data_type,
                        sheet_monthly_cols[previous_month],
                        header_row, data_start_row
                    )
                    pivot_result[data_type]['previous'] = prev_summaries
                    logger.info(f"  Previous ({previous_month}): {len(prev_summaries)} projects")

                # Process current month
                if current_month in sheet_monthly_cols:
                    curr_summaries = self._aggregate_pivot_data(
                        file_path, sheet_name, data_type,
                        sheet_monthly_cols[current_month],
                        header_row, data_start_row
                    )
                    pivot_result[data_type]['current'] = curr_summaries
                    logger.info(f"  Current ({current_month}): {len(curr_summaries)} projects")

        except Exception as e:
            logger.error(f"Error processing pivot table file: {e}")
            import traceback
            traceback.print_exc()
            raise

        return pivot_result

    def _aggregate_pivot_data(
        self,
        file_path: str,
        sheet_name: str,
        data_type: str,
        gla_column_idx: int,
        header_row: int,
        data_start_row: int
    ) -> Dict[Tuple[str, str], ProjectGLASummary]:
        """
        Aggregate GLA data from a specific column in a pivot table format file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to process
            data_type: 'handover' or 'committed'
            gla_column_idx: Column index for the GLA values
            header_row: Row index containing column headers
            data_start_row: Row index where data starts

        Returns:
            Dict mapping (project_name, product_type) to ProjectGLASummary
        """
        logger.debug(f"Aggregating pivot data from column {gla_column_idx}")

        # Read the full sheet
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Get headers for column name lookup
        headers = df_raw.iloc[header_row].tolist()

        # Find key column indices
        project_col = None
        product_type_col = None
        region_col = None
        tenant_col = None
        status_col = None
        bwid_project_col = None

        for col_idx, header in enumerate(headers):
            if pd.isna(header):
                continue
            header_str = str(header).strip()

            if header_str == 'Project Name':
                project_col = col_idx
            elif header_str == 'CCS_Product Type':
                product_type_col = col_idx
            elif header_str == 'Region':
                region_col = col_idx
            elif header_str in ['Tenant', 'Tenant Name', 'Customer Name']:
                tenant_col = col_idx
            elif header_str == 'Unit for Lease Status':
                status_col = col_idx
            elif header_str == 'BWID Project.':
                bwid_project_col = col_idx

        if project_col is None or product_type_col is None:
            logger.warning(f"Could not find required columns. Project: {project_col}, Type: {product_type_col}")
            return {}

        # Get data rows
        data = df_raw.iloc[data_start_row:].copy()

        # Aggregate by project and product type
        summaries: Dict[Tuple[str, str], ProjectGLASummary] = {}
        tenant_data: Dict[Tuple[str, str], Dict[str, TenantGLA]] = {}

        for idx in range(len(data)):
            row = data.iloc[idx]

            try:
                project_code = str(row.iloc[project_col]).strip() if pd.notna(row.iloc[project_col]) else ''
                product_type = str(row.iloc[product_type_col]).strip() if pd.notna(row.iloc[product_type_col]) else ''
                region = str(row.iloc[region_col]).strip() if region_col and pd.notna(row.iloc[region_col]) else ''
                gla = pd.to_numeric(row.iloc[gla_column_idx], errors='coerce')

                if pd.isna(gla):
                    gla = 0.0

                # Get tenant name
                tenant_name = "Unknown"
                if tenant_col and pd.notna(row.iloc[tenant_col]):
                    tenant_name = str(row.iloc[tenant_col]).strip()
                    if not tenant_name or tenant_name == '- None -':
                        tenant_name = "Vacant"

                # Get status for filtering
                status = ""
                if status_col and pd.notna(row.iloc[status_col]):
                    status = str(row.iloc[status_col]).strip()

                # Apply status filter based on data type
                if data_type == 'handover':
                    if status != LeaseStatus.HANDED_OVER.value:
                        continue
                elif data_type == 'committed':
                    if status not in [LeaseStatus.OPEN.value, LeaseStatus.HANDED_OVER.value]:
                        continue

                # Skip invalid records
                if not project_code or not product_type:
                    continue

                # Only process RBF and RBW
                if product_type not in [ProductType.RBF.value, ProductType.RBW.value]:
                    continue

                # Get readable project name
                bwid_project = row.iloc[bwid_project_col] if bwid_project_col and pd.notna(row.iloc[bwid_project_col]) else ''
                readable_name = self.extract_readable_project_name(bwid_project, project_code)

                key = (readable_name, product_type)

                if key not in summaries:
                    summaries[key] = ProjectGLASummary(
                        project_name=readable_name,
                        product_type=product_type,
                        region=region,
                        gla_sqm=0.0,
                        tenants=[]
                    )
                    tenant_data[key] = {}

                summaries[key].gla_sqm += gla

                # Track tenant GLA
                if tenant_name not in tenant_data[key]:
                    tenant_data[key][tenant_name] = TenantGLA(
                        tenant_name=tenant_name,
                        gla_sqm=0.0,
                        status=status
                    )
                tenant_data[key][tenant_name].gla_sqm += gla

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        # Convert tenant dicts to lists
        for key, tenant_dict in tenant_data.items():
            summaries[key].tenants = list(tenant_dict.values())

        logger.debug(f"Aggregated {len(summaries)} project-type combinations")
        return summaries
