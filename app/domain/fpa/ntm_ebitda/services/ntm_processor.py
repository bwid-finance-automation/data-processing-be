"""
NTM Processor Service
Handles Excel file parsing for NTM (Next Twelve Months) data extraction from leasing models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date

from app.shared.utils.logging_config import get_logger
from ..models.ntm_ebitda_models import (
    LeaseRecord,
    ProjectNTMSummary,
    AnalysisConfig,
)

logger = get_logger(__name__)


class NTMProcessor:
    """
    Processor for extracting NTM data from leasing model Excel files.

    Supports both .xlsx and .xlsb formats.

    Expected Excel structure:
    - Column 0: Project code (e.g., "BWD-Bau Bang")
    - Column 2: Phase (e.g., "Phase 1", "All Phase")
    - Column 4: Metric type (e.g., "Accounting revenue")
    - Column 10: Tenant name
    - Column 15: GLA (sqm)
    - Column 24: Lease start date
    - Column 25: Lease end date
    - Column 26: Term (months)
    - Columns 40-51: Monthly NTM data (12 columns)
    """

    # Column indices for data extraction (0-indexed)
    COL_PROJECT_CODE = 0
    COL_PHASE = 2
    COL_METRIC_TYPE = 4
    COL_TENANT = 10
    COL_GLA = 15
    COL_LEASE_START = 24
    COL_LEASE_END = 25
    COL_TERM = 26
    COL_NTM_START = 40  # First NTM column
    COL_NTM_END = 51    # Last NTM column (inclusive)

    # Metric type keywords for detection
    REVENUE_KEYWORDS = ["accounting revenue", "revenue", "rental income", "rent"]
    OPEX_KEYWORDS = ["opex", "operating expense", "operating cost"]
    SGA_KEYWORDS = ["sg&a", "sga", "general admin", "administrative"]

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the NTM processor.

        Args:
            config: Optional analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.project_mapping: Dict[str, str] = {}  # Model code -> Reporting name

    def process_file(
        self,
        file_path: str,
        prev_sheet: str,
        curr_sheet: str
    ) -> Dict[str, Dict[str, ProjectNTMSummary]]:
        """
        Process Excel file and extract NTM data for both periods.

        Args:
            file_path: Path to the Excel file
            prev_sheet: Sheet name for previous period
            curr_sheet: Sheet name for current period

        Returns:
            Dict with 'previous' and 'current' keys, each containing
            project name -> ProjectNTMSummary mapping
        """
        logger.info(f"Processing NTM file: {file_path}")
        logger.info(f"Previous sheet: {prev_sheet}, Current sheet: {curr_sheet}")

        # Load project mapping first
        self._load_project_mapping(file_path)

        # Process both sheets
        previous_data = self._process_sheet(file_path, prev_sheet)
        current_data = self._process_sheet(file_path, curr_sheet)

        return {
            "previous": previous_data,
            "current": current_data,
        }

    def _load_project_mapping(self, file_path: str):
        """
        Load project name mapping from 'Mapping' sheet.

        Args:
            file_path: Path to the Excel file
        """
        try:
            # Try to read mapping sheet
            mapping_df = pd.read_excel(
                file_path,
                sheet_name="Mapping",
                header=None,
                engine=self._get_engine(file_path)
            )

            # Column 1: Model code, Column 2: Reporting name
            for _, row in mapping_df.iterrows():
                if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    model_code = str(row.iloc[0]).strip()
                    reporting_name = str(row.iloc[1]).strip()
                    self.project_mapping[model_code] = reporting_name

            logger.info(f"Loaded {len(self.project_mapping)} project mappings")

        except Exception as e:
            logger.warning(f"Could not load project mapping: {e}")
            # Continue without mapping - will use project codes as-is

    def _get_engine(self, file_path: str) -> str:
        """Get the appropriate pandas engine for the file type."""
        if file_path.endswith('.xlsb'):
            return 'pyxlsb'
        return 'openpyxl'

    def _process_sheet(
        self,
        file_path: str,
        sheet_name: str
    ) -> Dict[str, ProjectNTMSummary]:
        """
        Process a single sheet and extract NTM data.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to process

        Returns:
            Dict mapping project name -> ProjectNTMSummary
        """
        logger.info(f"Processing sheet: {sheet_name}")

        try:
            # Read the sheet
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=None,
                engine=self._get_engine(file_path)
            )

            logger.info(f"Sheet {sheet_name}: {len(df)} rows, {len(df.columns)} columns")

            # Find data start row (skip headers)
            data_start = self._find_data_start_row(df)
            logger.info(f"Data starts at row {data_start}")

            # Extract lease records
            leases = self._extract_lease_records(df, data_start)
            logger.info(f"Extracted {len(leases)} lease records")

            # Aggregate by project
            project_summaries = self._aggregate_by_project(leases)
            logger.info(f"Aggregated to {len(project_summaries)} projects")

            return project_summaries

        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name}: {e}")
            raise ValueError(f"Failed to process sheet '{sheet_name}': {e}")

    def _find_data_start_row(self, df: pd.DataFrame) -> int:
        """
        Find the row where actual data starts (after headers).

        Args:
            df: DataFrame to analyze

        Returns:
            Row index where data starts
        """
        # Look for first row with a project code pattern
        for idx in range(min(20, len(df))):
            cell = df.iloc[idx, self.COL_PROJECT_CODE]
            if pd.notna(cell):
                cell_str = str(cell).strip()
                # Project codes typically start with "BWD-" or similar
                if cell_str.startswith("BWD") or cell_str.startswith("Project"):
                    continue  # Header row
                if len(cell_str) > 3 and not cell_str.lower().startswith("project"):
                    return idx

        return 5  # Default to row 5 if detection fails

    def _extract_lease_records(
        self,
        df: pd.DataFrame,
        start_row: int
    ) -> List[LeaseRecord]:
        """
        Extract lease records from DataFrame.

        Args:
            df: DataFrame with lease data
            start_row: Row index to start extraction

        Returns:
            List of LeaseRecord objects
        """
        records = []

        for idx in range(start_row, len(df)):
            row = df.iloc[idx]

            # Skip empty rows
            project_code = row.iloc[self.COL_PROJECT_CODE] if len(row) > self.COL_PROJECT_CODE else None
            if pd.isna(project_code) or str(project_code).strip() == "":
                continue

            project_code = str(project_code).strip()

            # Get phase - only process "All Phase" rows for project-level aggregation
            phase = row.iloc[self.COL_PHASE] if len(row) > self.COL_PHASE else ""
            phase = str(phase).strip() if pd.notna(phase) else ""

            # Get metric type
            metric_type = row.iloc[self.COL_METRIC_TYPE] if len(row) > self.COL_METRIC_TYPE else ""
            metric_type = str(metric_type).strip() if pd.notna(metric_type) else ""

            # Get tenant name
            tenant = row.iloc[self.COL_TENANT] if len(row) > self.COL_TENANT else None
            tenant = str(tenant).strip() if pd.notna(tenant) else None

            # Get GLA
            gla = 0.0
            if len(row) > self.COL_GLA and pd.notna(row.iloc[self.COL_GLA]):
                try:
                    gla = float(row.iloc[self.COL_GLA])
                except (ValueError, TypeError):
                    gla = 0.0

            # Get lease dates
            lease_start = self._parse_date(row.iloc[self.COL_LEASE_START] if len(row) > self.COL_LEASE_START else None)
            lease_end = self._parse_date(row.iloc[self.COL_LEASE_END] if len(row) > self.COL_LEASE_END else None)

            # Get term
            term_months = 0
            if len(row) > self.COL_TERM and pd.notna(row.iloc[self.COL_TERM]):
                try:
                    term_months = int(float(row.iloc[self.COL_TERM]))
                except (ValueError, TypeError):
                    term_months = 0

            # Get monthly NTM values
            monthly_ntm = []
            for col_idx in range(self.COL_NTM_START, min(self.COL_NTM_END + 1, len(row))):
                val = row.iloc[col_idx] if col_idx < len(row) else 0
                if pd.notna(val):
                    try:
                        monthly_ntm.append(float(val))
                    except (ValueError, TypeError):
                        monthly_ntm.append(0.0)
                else:
                    monthly_ntm.append(0.0)

            # Pad to 12 months if needed
            while len(monthly_ntm) < 12:
                monthly_ntm.append(0.0)

            # Map project code to reporting name
            project_name = self.project_mapping.get(project_code, project_code)

            # Create record
            record = LeaseRecord(
                project_code=project_code,
                project_name=project_name,
                phase=phase,
                metric_type=metric_type,
                tenant_name=tenant,
                gla_sqm=gla,
                lease_start_date=lease_start,
                lease_end_date=lease_end,
                term_months=term_months,
                monthly_ntm=monthly_ntm[:12],  # Ensure exactly 12 months
            )
            record.calculate_total_ntm()

            records.append(record)

        return records

    def _parse_date(self, value: Any) -> Optional[date]:
        """
        Parse a date value from Excel.

        Args:
            value: Cell value that might be a date

        Returns:
            date object or None
        """
        if pd.isna(value):
            return None

        try:
            if isinstance(value, (datetime, date)):
                return value.date() if isinstance(value, datetime) else value

            # Try parsing Excel serial date
            if isinstance(value, (int, float)):
                # Excel serial date (days since 1899-12-30)
                excel_epoch = datetime(1899, 12, 30)
                return (excel_epoch + pd.Timedelta(days=int(value))).date()

            # Try parsing string date
            if isinstance(value, str):
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue

            return None
        except Exception:
            return None

    def _aggregate_by_project(
        self,
        records: List[LeaseRecord]
    ) -> Dict[str, ProjectNTMSummary]:
        """
        Aggregate lease records by project.

        Args:
            records: List of lease records

        Returns:
            Dict mapping project name -> ProjectNTMSummary
        """
        projects: Dict[str, ProjectNTMSummary] = {}

        for record in records:
            project_name = record.project_name

            # Initialize project if needed
            if project_name not in projects:
                projects[project_name] = ProjectNTMSummary(
                    project_name=project_name,
                    project_code=record.project_code,
                    monthly_revenue=[0.0] * 12,
                    monthly_opex=[0.0] * 12,
                    monthly_sga=[0.0] * 12,
                    monthly_ebitda=[0.0] * 12,
                )

            project = projects[project_name]

            # Determine metric type and aggregate
            metric_lower = record.metric_type.lower()

            if any(kw in metric_lower for kw in self.REVENUE_KEYWORDS):
                # Revenue record
                if record.phase.lower() == "all phase":
                    project.revenue_ntm += record.total_ntm
                    for i, val in enumerate(record.monthly_ntm):
                        project.monthly_revenue[i] += val
                project.leases.append(record)

            elif any(kw in metric_lower for kw in self.OPEX_KEYWORDS):
                # OPEX record
                if record.phase.lower() == "all phase":
                    project.opex_ntm += record.total_ntm
                    for i, val in enumerate(record.monthly_ntm):
                        project.monthly_opex[i] += val

            elif any(kw in metric_lower for kw in self.SGA_KEYWORDS):
                # SG&A record
                if record.phase.lower() == "all phase":
                    project.sga_ntm += record.total_ntm
                    for i, val in enumerate(record.monthly_ntm):
                        project.monthly_sga[i] += val

        # Calculate EBITDA for each project
        for project in projects.values():
            project.calculate_ebitda()
            # Calculate monthly EBITDA
            for i in range(12):
                project.monthly_ebitda[i] = (
                    project.monthly_revenue[i] -
                    project.monthly_opex[i] -
                    project.monthly_sga[i]
                )

        return projects

    def detect_sheets(self, file_path: str) -> List[str]:
        """
        Detect available sheet names in an Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of sheet names
        """
        try:
            xl = pd.ExcelFile(file_path, engine=self._get_engine(file_path))
            return xl.sheet_names
        except Exception as e:
            logger.error(f"Error detecting sheets: {e}")
            raise

    def detect_period_sheets(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Auto-detect previous and current period sheets.

        Args:
            file_path: Path to the Excel file

        Returns:
            Tuple of (previous_sheet, current_sheet) or (None, None) if not detected
        """
        sheets = self.detect_sheets(file_path)

        # Look for Model_leasing patterns
        leasing_sheets = [s for s in sheets if "Model_leasing" in s or "leasing" in s.lower()]

        if len(leasing_sheets) >= 2:
            # Sort by month name if present
            month_order = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }

            def get_month_order(sheet_name: str) -> int:
                for month, order in month_order.items():
                    if month in sheet_name:
                        return order
                return 0

            sorted_sheets = sorted(leasing_sheets, key=get_month_order)
            if len(sorted_sheets) >= 2:
                return sorted_sheets[-2], sorted_sheets[-1]

        # Look for Previous/Current naming
        prev_sheet = next((s for s in sheets if "previous" in s.lower() or "prev" in s.lower()), None)
        curr_sheet = next((s for s in sheets if "current" in s.lower() or "curr" in s.lower()), None)

        if prev_sheet and curr_sheet:
            return prev_sheet, curr_sheet

        # Default: return first two sheets if available
        if len(sheets) >= 2:
            return sheets[0], sheets[1]

        return None, None

    def extract_period_label(self, sheet_name: str) -> str:
        """
        Extract period label from sheet name.

        Args:
            sheet_name: Sheet name like "Model_leasing_Sep'25"

        Returns:
            Period label like "Sep'25"
        """
        import re

        # Try to extract month'year pattern
        match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\s]?(\d{2,4})", sheet_name, re.IGNORECASE)
        if match:
            month = match.group(1)
            year = match.group(2)
            if len(year) == 4:
                year = year[-2:]  # Use last 2 digits
            return f"{month}'{year}"

        # Try T## pattern (Vietnamese month notation)
        match = re.search(r"T(\d{1,2})", sheet_name)
        if match:
            return f"T{match.group(1)}"

        # Return original name if no pattern found
        return sheet_name
